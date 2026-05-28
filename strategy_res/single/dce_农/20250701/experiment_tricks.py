"""
experiment_tricks.py
对比不同 trick 对 test_corr / 方向正确率 的提升效果
支持：8品种 × 多方案 批量实验，单核顺序执行，输出 CSV 结果表

运行: python experiment_tricks.py
输出: experiment_results_YYYYMMDD.csv
"""

import sys
import json
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold

sys.path.append('/home/future_commodity')
import function_future.train_model as tm
import function_future.DataLoader as DL
import pipeline_v1 as pl

LOCAL_MODEL_BASE = pl.LOCAL_MODEL_BASE


# ==================== 自定义损失函数 ====================

def _direction_aware_obj(pred, dtrain, penalty=3.0):
    """
    方向敏感损失：方向错了重罚，|pred| 越大罚越狠。
    """
    true = dtrain.get_label()
    
    # 基础 MAE 梯度
    grad_base = np.sign(pred - true)
    hess_base = np.ones_like(pred) * 0.01
    
    # 方向错误的惩罚
    wrong = (pred * true < 0).astype(float)
    grad_penalty = wrong * penalty * np.sign(pred)
    hess_penalty = wrong * penalty * 0.01
    
    grad = grad_base + grad_penalty
    hess = hess_base + hess_penalty
    
    return grad, hess


def _direction_aware_eval(pred, dtrain):
    """评估指标：方向正确率（pred 和 true 同号的比例），过滤 NaN"""
    true = dtrain.get_label()
    mask = ~np.isnan(pred) & ~np.isnan(true)
    if np.sum(mask) == 0:
        return [('dir_acc', 0.0, True)]
    correct = (pred[mask] * true[mask] > 0).astype(float)
    return [('dir_acc', np.mean(correct), True)]


# ==================== 核心训练函数 ====================

def _train_kfold(
    analyzer: tm.TimeSeriesAnalyzer,
    folder_name: str,
    sample_weight: np.ndarray = None,
    custom_params: dict = None,
    custom_obj=None,
    custom_eval=None,
    sample_weight_type: str = None,
    n_splits: int = 5
) -> pd.DataFrame:
    """
    训练并返回 metrics_df [fold, train_rmse, val_rmse, best_iteration, test_corr, high_conf_dir_acc]
    
    sample_weight_type: 'quantile' 表示用分位数压缩的 |rtn| 权重
    """
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.005,
        'num_leaves': 32,
        'max_depth': 5,
        'min_data_in_leaf': 500,
        'lambda_l1': 1,
        'lambda_l2': 1,
        'feature_fraction': 0.7,
        'bagging_freq': 10,
        'extra_trees': True,
        'max_bin': 32,
        'verbose': -1,
        'seed': 142,
        "num_threads": 20,
        'deterministic': True
    }
    if custom_params:
        params.update(custom_params)
    if custom_obj is not None:
        params['objective'] = custom_obj

    X = analyzer.train_data[analyzer.factor_col + analyzer.category_col].values
    y = analyzer.train_data[analyzer.target_col].values

    categorical_col_idx = [
        i for i, col in enumerate(analyzer.factor_col + analyzer.category_col)
        if col in analyzer.category_col
    ] if analyzer.category_col else []

    model_dir = LOCAL_MODEL_BASE / folder_name
    model_dir.mkdir(parents=True, exist_ok=True)

    kf = KFold(n_splits=n_splits, shuffle=False)
    metrics_rows = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 和原始代码一致的边界处理
        X_val[:60*4] = np.nan
        y_val[:60*4] = np.nan
        X_val[-60*4:] = np.nan
        y_val[-60*4:] = np.nan

        y_val = y_val / np.nanstd(y_val)
        y_train = y_train / np.nanstd(y_train)

        # sample weight 计算
        w_train = None
        if sample_weight_type == 'quantile':
            abs_y = np.abs(y_train)
            q10 = np.quantile(abs_y, 0.1)
            q90 = np.quantile(abs_y, 0.9)
            w_train = np.clip(abs_y, q10, q90)
            w_train = np.sqrt(w_train)
            w_train = w_train / np.mean(w_train)
        elif sample_weight is not None:
            w_train = sample_weight[train_idx]
            w_train = w_train / np.nanmean(w_train)

        train_set = lgb.Dataset(
            X_train, y_train, weight=w_train,
            categorical_feature=categorical_col_idx,
            feature_name=analyzer.factor_col + analyzer.category_col,
            free_raw_data=True
        )
        valid_set = lgb.Dataset(
            X_val, y_val, categorical_feature=categorical_col_idx,
            feature_name=analyzer.factor_col + analyzer.category_col,
            reference=train_set
        )

        evals_result = {}
        model = lgb.train(
            params, train_set, valid_sets=[valid_set],
            num_boost_round=10000,
            feval=custom_eval,
            callbacks=[
                lgb.early_stopping(stopping_rounds=500, min_delta=5e-7, verbose=False),
                lgb.record_evaluation(evals_result)
            ]
        )

        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        # test corr（测试标签始终用原始的，保证所有方案口径一致）
        pred_test = model.predict(analyzer.test_data[analyzer.factor_col + analyzer.category_col].values)
        true_test = analyzer.test_data[analyzer.target_col].values
        mask = ~np.isnan(pred_test) & ~np.isnan(true_test)
        test_corr = np.corrcoef(pred_test[mask], true_test[mask])[0, 1] if np.sum(mask) > 1 else np.nan

        # 高 |pred| 样本的方向正确率（模拟实际交易时的开仓样本，th=0.9）
        high_conf_mask = mask & (np.abs(pred_test) > 0.9)
        if np.sum(high_conf_mask) > 1:
            high_conf_dir_acc = np.mean((pred_test[high_conf_mask] * true_test[high_conf_mask]) > 0)
        else:
            high_conf_dir_acc = np.nan

        metrics_rows.append({
            'fold': f'fold_{fold}',
            'train_rmse': np.sqrt(np.nanmean((y_train - train_pred)**2)),
            'val_rmse': np.sqrt(np.nanmean((y_val - val_pred)**2)),
            'best_iteration': model.best_iteration,
            'test_corr': test_corr,
            'high_conf_dir_acc': high_conf_dir_acc,
        })

        # 保存模型和 meta
        model.save_model(str(model_dir / f'kfold_fold{fold}_0.lgb'))
        with open(model_dir / f'kfold_fold{fold}_0_meta.json', 'w') as f:
            json.dump({
                'best_iteration': int(model.best_iteration),
                'test_corr': float(test_corr),
                'high_conf_dir_acc': float(high_conf_dir_acc),
                'fac_num': int(len(analyzer.factor_col))
            }, f)

    return pd.DataFrame(metrics_rows)


# ==================== 辅助函数 ====================

def _make_analyzer(symbol, factor_col, train_end_date, config_loader, train_label, label_smooth=None):
    """
    统一创建 analyzer 并加载数据。
    label_smooth: 如 [0.6, 0.3, 0.1]，则对训练集标签做三期加权平滑（推理端后处理前移到训练端）
    """
    analyzer = tm.TimeSeriesAnalyzer(symbol, factor_col, train_end_date, config_loader)
    analyzer.load_and_prepare_data(
        log_rtn=True, set_category_col=['hour'],
        label_col=f'rtn_{train_label}', cut=True
    )
    
    if label_smooth is not None:
        target = analyzer.target_col
        df = analyzer.train_data.sort_values('datetime').copy()
        smoothed = (
            label_smooth[0] * df[target] +
            label_smooth[1] * df[target].shift(1) +
            label_smooth[2] * df[target].shift(2)
        )
        df[target] = smoothed
        df = df.dropna(subset=[target])
        analyzer.train_data = df
    
    return analyzer


# ==================== 各种 Trick（8个方案）====================

def baseline(symbol, factor_col, train_end_date, config_loader, train_label, folder_name):
    """方案1: 基线 MSE"""
    analyzer = _make_analyzer(symbol, factor_col, train_end_date, config_loader, train_label)
    return _train_kfold(analyzer, folder_name + '_baseline')


def baseline_smooth(symbol, factor_col, train_end_date, config_loader, train_label, folder_name):
    """方案2: MSE + 标签平滑（0.6/0.3/0.1）"""
    analyzer = _make_analyzer(symbol, factor_col, train_end_date, config_loader, train_label, label_smooth=[0.6, 0.3, 0.1])
    return _train_kfold(analyzer, folder_name + '_baseline_smooth')


def trick5_huber(symbol, factor_col, train_end_date, config_loader, train_label, folder_name):
    """方案3: Huber"""
    analyzer = _make_analyzer(symbol, factor_col, train_end_date, config_loader, train_label)
    return _train_kfold(
        analyzer, folder_name + '_trick5_huber',
        custom_params={'objective': 'huber', 'metric': 'l1', 'alpha': 0.9}
    )


def trick5_huber_smooth(symbol, factor_col, train_end_date, config_loader, train_label, folder_name):
    """方案4: Huber + 标签平滑"""
    analyzer = _make_analyzer(symbol, factor_col, train_end_date, config_loader, train_label, label_smooth=[0.6, 0.3, 0.1])
    return _train_kfold(
        analyzer, folder_name + '_trick5_huber_smooth',
        custom_params={'objective': 'huber', 'metric': 'l1', 'alpha': 0.9}
    )


def trick6_direction(symbol, factor_col, train_end_date, config_loader, train_label, folder_name):
    """方案5: 方向敏感损失"""
    analyzer = _make_analyzer(symbol, factor_col, train_end_date, config_loader, train_label)
    obj_fn = partial(_direction_aware_obj, penalty=3.0)
    return _train_kfold(
        analyzer, folder_name + '_trick6_direction',
        custom_obj=obj_fn,
        custom_eval=_direction_aware_eval,
        custom_params={'metric': 'l1'}
    )


def trick6_direction_smooth(symbol, factor_col, train_end_date, config_loader, train_label, folder_name):
    """方案6: 方向敏感损失 + 标签平滑"""
    analyzer = _make_analyzer(symbol, factor_col, train_end_date, config_loader, train_label, label_smooth=[0.6, 0.3, 0.1])
    obj_fn = partial(_direction_aware_obj, penalty=3.0)
    return _train_kfold(
        analyzer, folder_name + '_trick6_direction_smooth',
        custom_obj=obj_fn,
        custom_eval=_direction_aware_eval,
        custom_params={'metric': 'l1'}
    )


def trick8_quantile_mse(symbol, factor_col, train_end_date, config_loader, train_label, folder_name):
    """方案7: 分位数压缩样本权重 + MSE"""
    analyzer = _make_analyzer(symbol, factor_col, train_end_date, config_loader, train_label)
    return _train_kfold(
        analyzer, folder_name + '_trick8_quantile_mse',
        sample_weight_type='quantile'
    )


def trick8_quantile_huber(symbol, factor_col, train_end_date, config_loader, train_label, folder_name):
    """方案8: 分位数压缩样本权重 + Huber"""
    analyzer = _make_analyzer(symbol, factor_col, train_end_date, config_loader, train_label)
    return _train_kfold(
        analyzer, folder_name + '_trick8_quantile_huber',
        sample_weight_type='quantile',
        custom_params={'objective': 'huber', 'metric': 'l1', 'alpha': 0.9}
    )


# ==================== 批量运行入口 ====================

def run_all_experiments(
    symbols=None,
    train_end_date='2025-07-01',
    train_label=5,
    output_path=None
):
    """
    批量运行：8品种 × 8方案，单核顺序执行，输出 CSV 结果表。
    
    参数:
        symbols: 品种列表，默认 ['A','B','C','CS','M','Y','P','LH']
        train_end_date: 训练截止日期
        train_label: 预测周期
        output_path: 输出 CSV 路径，默认 experiment_results_YYYYMMDD.csv
    
    返回:
        combined: 所有 fold 级原始结果
        summary: 按 symbol + experiment 汇总的平均指标
    """
    if symbols is None:
        symbols = ['A', 'B', 'C', 'CS', 'M', 'Y', 'P', 'LH']
    
    if output_path is None:
        output_path = f"experiment_results_{train_end_date.replace('-', '')}.csv"
    
    experiments = [
        ('baseline', baseline),
        ('baseline_smooth', baseline_smooth),
        ('trick5_huber', trick5_huber),
        ('trick5_huber_smooth', trick5_huber_smooth),
        ('trick6_direction', trick6_direction),
        ('trick6_direction_smooth', trick6_direction_smooth),
        ('trick8_quantile_mse', trick8_quantile_mse),
        ('trick8_quantile_huber', trick8_quantile_huber),
    ]
    
    all_results = []
    total = len(symbols) * len(experiments)
    count = 0
    
    for symbol in symbols:
        print(f"\n{'='*70}")
        print(f"[品种] {symbol} ({symbols.index(symbol)+1}/{len(symbols)})")
        print(f"{'='*70}")
        
        try:
            # 1. 配置
            config_loader = DL.InstrumentConfig()
            config_loader.get_instrument_config(symbol)
            
            # 2. 加载因子数据
            print(f"[{symbol}] 加载因子数据...")
            fac_df = pl.load_factor_data(symbol, train_end_date, train_label, config_loader)
            
            # 3. 预训练（文件已存在则自动跳过）
            print(f"[{symbol}] 预训练...")
            pl.run_pretrain(symbol, fac_df, train_end_date, train_label)
            
            # 4. 因子筛选
            print(f"[{symbol}] 因子筛选...")
            factor_col = [c for c in fac_df.columns if c not in ['datetime', 'instrument', 'pred_ret', 'hour']]
            factor_filter, summary, _, _ = pl.run_factor_filter(
                symbol, fac_df, train_end_date, train_label, factor_col,
                {
                    "info_select": {"nan_rate": 0.8, "mode_rate": 0.9},
                    "importance_select_by_group": {"cut_num_1": 300, "cut_num_2": 200, "same_name_cut": 5},
                    "sp_select": {"th": 0},
                    "day_cut": {"num_limit": 5},
                }
            )
            selected_factors = factor_filter.factor_to_choose
            if not selected_factors:
                print(f"  警告: {symbol} 没有因子通过筛选，跳过")
                continue
            print(f"  筛选后因子数: {len(selected_factors)}")
            
            # 5. 跑所有方案
            folder_base = f"{symbol}_pred{train_label}_{train_end_date}"
            
            for exp_name, exp_func in experiments:
                count += 1
                print(f"\n[{count}/{total}] {symbol} - {exp_name}")
                try:
                    metrics_df = exp_func(
                        symbol, selected_factors, train_end_date,
                        config_loader, train_label, folder_base
                    )
                    metrics_df['experiment'] = exp_name
                    metrics_df['symbol'] = symbol
                    all_results.append(metrics_df)
                    print(f"  ✓ 完成: test_corr={metrics_df['test_corr'].mean():.4f}, dir_acc={metrics_df['high_conf_dir_acc'].mean():.4f}")
                except Exception as e:
                    print(f"  ✗ 失败: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # 每跑完一个品种就保存中间结果（防崩溃丢失）
            if all_results:
                combined_so_far = pd.concat(all_results, ignore_index=True)
                combined_so_far.to_csv(output_path, index=False)
                print(f"  [已保存中间结果: {output_path}]")
        
        except Exception as e:
            print(f"[{symbol}] 品种级错误: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 最终汇总
    if not all_results:
        print("没有成功运行的实验")
        return None, None
    
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(output_path, index=False)
    
    # 汇总表：按 symbol + experiment 分组
    summary = combined.groupby(['symbol', 'experiment']).agg({
        'test_corr': 'mean',
        'high_conf_dir_acc': 'mean',
        'val_rmse': 'mean',
        'best_iteration': 'mean'
    }).reset_index().sort_values(['symbol', 'test_corr'], ascending=[True, False])
    
    # pivot 表：品种 × 方案的 test_corr
    pivot_corr = combined.pivot_table(
        index='symbol', columns='experiment', values='test_corr', aggfunc='mean'
    )
    pivot_dir = combined.pivot_table(
        index='symbol', columns='experiment', values='high_conf_dir_acc', aggfunc='mean'
    )
    
    print("\n" + "="*100)
    print("各品种最佳方案（按 avg_test_corr 排序）")
    print("="*100)
    for sym in symbols:
        sym_df = summary[summary['symbol'] == sym]
        if not sym_df.empty:
            best = sym_df.iloc[0]
            print(f"{sym}: {best['experiment']:25s}  test_corr={best['test_corr']:.4f}  dir_acc={best['high_conf_dir_acc']:.4f}")
    
    print("\n" + "="*100)
    print("test_corr Pivot 表（品种 × 方案）")
    print("="*100)
    print(pivot_corr.to_string())
    
    print("\n" + "="*100)
    print("high_conf_dir_acc Pivot 表（品种 × 方案）")
    print("="*100)
    print(pivot_dir.to_string())
    
    print(f"\n[最终结果已保存] {output_path}")
    print(f"总行数: {len(combined)} ({len(symbols)}品种 × {len(experiments)}方案 × 5 folds)")
    
    return combined, summary


# ==================== 单品种运行（兼容旧接口）====================

def run_experiments(symbol='A', train_end_date='2025-07-01', train_label=5):
    """单品种运行（保留旧接口）"""
    return run_all_experiments(
        symbols=[symbol],
        train_end_date=train_end_date,
        train_label=train_label,
        output_path=f"experiment_results_{symbol}_{train_end_date.replace('-', '')}.csv"
    )


if __name__ == '__main__':
    combined, summary = run_all_experiments()
