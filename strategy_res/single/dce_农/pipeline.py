"""
单品种训练+回测 Pipeline V1
把框架性代码从 notebook 中抽出来，cell 里只保留配置和一行调用。
"""
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import yaml

sys.path.append('/home/future_commodity')

import function_future.pre_train as pt
import function_future.train_model as tm
import function_future.FactorFilter as FF
import function_future.backtest_v3 as bv
import function_future.DataLoader as DL
import function_future.trading_visualization as TV
from function_future.margin_calculator import calculate_margin

# ------------------------------------------------------------------
# 0. 路径工具
# ------------------------------------------------------------------
def get_model_base_dir(train_end_date: str) -> Path:
    """按训练截止日期生成模型根目录，如 20250701/model/"""
    date_str = train_end_date.replace('-', '')
    base = Path(f'/home/strategy_res/single/dce_农/{date_str}/model')
    base.mkdir(parents=True, exist_ok=True)
    return base


def clear_pretrain_files(symbol: str, train_end_date: str, train_label: int):
    """删除该品种在该训练截止日期下的预训练文件，强制重新生成。"""
    save_name = f"{train_end_date}_{symbol}_{train_label}"
    base = Path(f'/mnt/Data/writable/liaoyuyang/factor_eval_commodity/{train_end_date}')

    files_to_remove = [
        base / f'importance/{save_name}_feature_importance_reg.csv',
        base / f'correlation/{save_name}_feature_corr.csv',
        base / f'group/{save_name}_feature_group.csv',
        base / f'{symbol}_single_factor_eval_{train_label}.csv',
        base / f'stability/{save_name}_feature_stability.csv',
    ]
    removed = 0
    for f in files_to_remove:
        if f.exists():
            f.unlink()
            removed += 1
    print(f"[clear_pretrain] {symbol} {train_end_date}: 删除 {removed} 个预训练文件")


# 向后兼容：保留旧常量，但新代码应使用 get_model_base_dir()
LOCAL_MODEL_BASE = Path('/home/strategy_res/single/dce_农/20250701/model')
LOCAL_MODEL_BASE.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# 1. 数据加载
# ------------------------------------------------------------------
def load_factor_data(symbol: str, train_end_date: str, train_label: int, config_loader=None) -> pd.DataFrame:
    """
    加载因子数据并做时间切割，同时拼接 pred_ret（收益率标签）和 hour。
    返回的 DataFrame 可直接传给 Pretrainer 和 FactorFilter。
    """
    if config_loader is None:
        config_loader = DL.InstrumentConfig()
    fac_df = pd.read_feather(
        f'/mnt/Data/writable/liaoyuyang/factor/{symbol}/all_fac/all_factor.feather'
    ).set_index('datetime').loc[:train_end_date]
    fac_df = config_loader.df_cut_time(
        fac_df,
        config_loader.get_instrument_config(symbol)['trading_hours'],
        10
    )

    # 拼接收益率标签
    rtn_df = pd.read_csv(
        f'/mnt/Data/writable/liaoyuyang/data/1min/active/main_{symbol}.csv',
        index_col=0, parse_dates=['ts']
    ).set_index('ts').reindex(index=fac_df.index)
    fac_df['pred_ret'] = rtn_df[f'rtn_{train_label}']
    fac_df = fac_df.replace([np.inf, -np.inf], np.nan)
    fac_df['hour'] = fac_df.index.hour

    return fac_df


# ------------------------------------------------------------------
# 2. 资金计算
# ------------------------------------------------------------------
def calc_initial_capital(symbol: str, lots: int = 10, leverage: float = 0.4) -> int:
    """
    按 40% 保证金比例计算所需初始资金。
    与 dce农_独立版本.ipynb 中的逻辑保持一致。
    """
    margin = calculate_margin(symbol, lots)
    capital = int(margin / leverage)
    print(f"[{symbol}] 初始资金: {capital:,}  "
          f"(10手保证金 {margin:,.0f} / {leverage:.0%} 杠杆)")
    return capital


# ------------------------------------------------------------------
# 3. 预训练（生成 importance / corr / group / eval）
# ------------------------------------------------------------------
def run_pretrain(symbol: str, fac_df: pd.DataFrame, train_end_date: str, train_label: int):
    """
    运行完整预训练流程；结果自动落到 /factor_eval_commodity/ 下。
    如果输出文件均已存在，直接跳过，不再初始化 Pretrainer。
    """
    save_name = f"{train_end_date}_{symbol}_{train_label}"
    base = Path(f'/mnt/Data/writable/liaoyuyang/factor_eval_commodity/{train_end_date}')

    files_to_check = [
        base / f'importance/{save_name}_feature_importance_reg.csv',
        base / f'correlation/{save_name}_feature_corr.csv',
        base / f'group/{save_name}_feature_group.csv',
        base / f'{symbol}_single_factor_eval_{train_label}.csv',
    ]

    if all(f.exists() for f in files_to_check):
        print(f"[pretrain] 文件均已存在，跳过: {save_name}")
        return None

    pretrainer = pt.Pretrainer(symbol, fac_df, train_end_date, train_label=train_label)
    pretrainer.run_full_pretraining(type_lgb='reg')
    return pretrainer


# ------------------------------------------------------------------
# 4. 因子筛选
# ------------------------------------------------------------------


def _load_dce_symbols() -> List[str]:
    """从 strat_lab 配置中读取 DCE 农产品品种列表。"""
    config_path = Path('/home/strat_lab/config/instruments.yaml')
    if config_path.exists():
        with open(config_path) as f:
            instruments = yaml.safe_load(f)
        return sorted(instruments.keys())
    # fallback
    return ['A', 'B', 'C', 'CS', 'M', 'Y', 'P', 'LH']


DCE_SYMBOLS = _load_dce_symbols()


def _build_category_df(factor_filter: FF.FactorFilter, symbols: List[str] = None) -> pd.DataFrame:
    """根据品种列表自动生成分类统计表。"""
    if symbols is None:
        symbols = DCE_SYMBOLS

    chosen = factor_filter.factor_to_choose
    total = len(chosen)

    # 跨品种前缀正则: ^(A_|B_|...)
    cross_prefix = '|'.join(symbols)
    cross_pattern = re.compile(rf'^({cross_prefix})_')

    fac_list = [x for x in chosen if x.startswith('FAC')]
    cross_list = [x for x in chosen if cross_pattern.match(x)]
    min_list = [x for x in chosen if not (x.startswith('FAC') or cross_pattern.match(x))]

    cat_df = pd.DataFrame({
        '类别': ['FAC', '跨品种', '分钟因子'],
        '数量': [len(fac_list), len(cross_list), len(min_list)],
        '占比': [
            f"{len(fac_list)/total:.1%}",
            f"{len(cross_list)/total:.1%}",
            f"{len(min_list)/total:.1%}",
        ],
    })
    return cat_df


def run_factor_filter(
    symbol: str,
    fac_df: pd.DataFrame,
    train_end_date: str,
    train_label: int,
    factor_to_choose: List[str],
    pipeline_config: Dict[str, Any],
    symbols: List[str] = None
) -> Tuple[FF.FactorFilter, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    一键运行完整筛选链。
    返回: (factor_filter 实例, summary_df, stability_df, category_df)
    """
    base = f'/mnt/Data/writable/liaoyuyang/factor_eval_commodity/{train_end_date}'
    factor_info = pd.read_csv(
        f'{base}/{symbol}_single_factor_eval_{train_label}.csv', index_col=0
    )
    importance_df = pd.read_csv(
        f'{base}/importance/{train_end_date}_{symbol}_{train_label}_feature_importance_reg.csv'
    )
    corr_df = pd.read_csv(
        f'{base}/correlation/{train_end_date}_{symbol}_{train_label}_feature_corr.csv', index_col=0
    )
    group_df = pd.read_csv(
        f'{base}/group/{train_end_date}_{symbol}_{train_label}_feature_group.csv'
    )

    factor_filter = FF.FactorFilter(importance_df, corr_df, group_df, factor_info, factor_to_choose)
    summary, stability_df = factor_filter.run_full_pipeline(fac_df, pipeline_config)
    cat_df = _build_category_df(factor_filter, symbols)
    return factor_filter, summary, stability_df, cat_df


def save_stability_df(
    stability_df: pd.DataFrame,
    train_end_date: str,
    symbol: str,
    train_label: int
) -> Path:
    """把月度稳定性评估表写到 /factor_eval_commodity/{date}/stability/ 下。"""
    out_dir = Path(f'/mnt/Data/writable/liaoyuyang/factor_eval_commodity/{train_end_date}/stability')
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'{train_end_date}_{symbol}_{train_label}_feature_stability.csv'
    stability_df.to_csv(path)
    print(f"月度稳定性表已保存到 {path}")
    return path


# ------------------------------------------------------------------
# 5. 模型训练
# ------------------------------------------------------------------
def train_model(
    symbol: str,
    factor_col: List[str],
    train_end_date: str,
    config_loader,
    train_label: int,
    folder_name: str,
    label_transform: callable = None,
    lgb_params: dict = None,
    model_base_dir: Path = None,
):
    """
    训练 LightGBM K-Fold 模型。

    Parameters
    ----------
    label_transform : callable, optional
        对 label 做自定义变换，例如 lambda x: np.sign(x) * np.abs(x)**3
    model_base_dir : Path, optional
        模型保存目录。默认按 train_end_date 自动推导。

    Returns
    -------
    trainer : LGBMTrainer 实例
    metrics_df : pd.DataFrame
        每折训练的关键指标（train_rmse, val_rmse, best_iteration, test_corr）
    """
    if model_base_dir is None:
        model_base_dir = get_model_base_dir(train_end_date)

    analyzer = tm.TimeSeriesAnalyzer(
        symbol=symbol,
        factor_col=factor_col,
        train_end_date=train_end_date,
        config_loader=config_loader
    )
    analyzer.load_and_prepare_data(
        log_rtn=True,
        set_category_col=['hour'],
        label_col=f'rtn_{train_label}',
        cut=True
    )

    # 自定义 label 变换（仅作用于训练集）
    if label_transform is not None:
        analyzer.train_data = analyzer.train_data.copy()
        analyzer.train_data[analyzer.target_col] = label_transform(
            analyzer.train_data[analyzer.target_col]
        )
        print(f"[train_model] label 已做自定义变换: {label_transform.__name__ if hasattr(label_transform, '__name__') else 'lambda'}")

    trainer = tm.LGBMTrainer(analyzer)
    # 重定向模型保存路径
    trainer.model_dir = model_base_dir
    # 如果用户传入了完整 lgb_params，覆盖默认参数
    if lgb_params is not None:
        trainer.set_params(lgb_params)
    fold_results = trainer.train_kfold_v0(
        custom_params={'verbose': -1},
        model_folder_name=folder_name,
        plot_train=True,
        n_splits=5
    )

    # 收集每折关键指标为 DataFrame
    metrics_rows = []
    for fold_name, metrics in fold_results.items():
        row = {
            'fold': fold_name,
            'train_rmse': metrics.get('train_rmse'),
            'val_rmse': metrics.get('val_rmse'),
            'best_iteration': metrics.get('best_iteration'),
        }
        # 读取 meta.json 中的 test_corr
        meta_file = model_base_dir / folder_name / f"{fold_name.replace('fold_', 'kfold_fold')}_0_meta.json"
        if meta_file.exists():
            import json
            with open(meta_file) as f:
                meta = json.load(f)
            row['test_corr'] = meta.get('test_corr')
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)
    return trainer, metrics_df


# ------------------------------------------------------------------
# 6. 回测
# ------------------------------------------------------------------
def run_backtest(
    symbol: str,
    train_end_date: str,
    folder_name: str,
    bt_params: Dict[str, Any],
    initial_capital: int,
    window_end: str = '2027-01-01',
    model_base_dir: Path = None,
):
    """
    回测封装。
    bt_params 示例:
        {'th1': 0.9, 'th2': 0.5, 'holding_bars': 10, 'day': 1725, 'fee': 0}
    model_base_dir : Path, optional
        模型加载目录。默认按 train_end_date 自动推导。
    """
    if model_base_dir is None:
        model_base_dir = get_model_base_dir(train_end_date)

    config = bv.load_config(r"/mnt/Data/writable/liaoyuyang/backtest/backtest_config.json")
    # 重定向模型加载路径
    config['MODEL_DIR'] = model_base_dir / str(folder_name)

    bt = bv.ModelBacktester(train_end_date, config)
    bt.load_config(symbol)
    bt.window_end = window_end
    bt.money = initial_capital
    bt.fee = bt_params.get('fee', 0)

    bt.load_factor(symbol, end_date=bt.window_end)
    bt.load_mktdata(symbol, end_date=bt.window_end)
    bt.load_models()
    bt.generate_predictions()
    pred = bt.combine_models('best_iteration_log_weighted', avg=True)

    merged_data = bt.backtest(
        th1=bt_params['th1'],
        th2=bt_params['th2'],
        save=False,
        open_drop=True,
        holding_bars=bt_params['holding_bars'],
        day=bt_params['day'],
        model_name='best_iteration_log_weighted',
        v=2
    )
    return bt, merged_data
