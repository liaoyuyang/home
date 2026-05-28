"""
多品种训练模块
用于多品种联合训练，输出特征重要性表
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from typing import List, Tuple, Optional
from sklearn.model_selection import KFold
from tqdm.auto import tqdm


def nanstd(arr, axis=None, ddof=0):
    """
    计算忽略NaN的标准差
    """
    arr = np.asarray(arr)
    mask = ~np.isnan(arr)
    
    if axis is None:
        valid_sum = np.nansum((arr - np.nanmean(arr)) ** 2)
        valid_count = np.sum(mask)
    else:
        mean = np.nanmean(arr, axis=axis, keepdims=True)
        squared_diff = (arr - mean) ** 2
        valid_sum = np.nansum(squared_diff, axis=axis)
        valid_count = np.sum(mask, axis=axis)
    
    if valid_count - ddof <= 0:
        return np.nan
    variance = valid_sum / (valid_count - ddof)
    return np.sqrt(variance)


def train_multi_symbol(
    data: pd.DataFrame,
    features: Optional[List[str]] = None,
    target_col: str = 'rtn_5',
    n_splits: int = 5,
    plot_train: bool = False,
    category_col: List[str] = None,
    lgb_params: Optional[dict] = None
) -> Tuple[List[lgb.Booster], pd.DataFrame]:
    """
    多品种联合训练模型并计算特征重要性
    
    Parameters:
    -----------
    data : pd.DataFrame
        合并后的数据，包含特征和目标变量
    rtn_df : pd.DataFrame
        目标变量数据，索引为 (datetime, symbol, instrument)
    features : List[str], optional
        特征列表，None则使用fac_df的所有列
    target_col : str
        目标变量列名
    n_splits : int
        KFold折数
    plot_train : bool
        是否绘制训练过程
    category_col : List[str]
        类别特征列名列表
    lgb_params : dict, optional
        LightGBM参数，None则使用默认参数
        
    Returns:
    --------
    Tuple[List[lgb.Booster], pd.DataFrame]
        (模型列表, 特征重要性DataFrame)
    """
    if features is None:
        features = [x for x in data.columns if x != target_col and x != 'symbol']
    
    if category_col is None:
        category_col = ['hour']
    
    # 默认参数
    default_params = {
        'objective': 'regression',
        'metric': 'l2',
        'boosting_type': 'gbdt',
        'learning_rate': 0.002,
        'num_leaves': 63,
        'max_depth': 5,
        'min_data_in_leaf': 1000,
        'lambda_l1': 1,
        'lambda_l2': 1,
        'feature_fraction': 0.6,
        'bagging_freq': 10,
        'extra_trees': True,
        'max_bin': 64,
        'verbose': -1,
        'feature_pre_filter': False,
        'seed': 42,
        "num_threads": 20
    }
    
    if lgb_params is not None:
        default_params.update(lgb_params)
    
    params = default_params
    
    # 合并特征和目标
    # data = data[features].copy()
    # data[target_col] = data[target_col]
    
    # 删除包含NaN的行
    data = data.dropna(subset=[target_col])
    
    X = data[features]
    y = data[target_col]
    
    # 类别特征索引
    categorical_col_idx = [
        i for i, col in enumerate(features)
        if col in category_col
    ]
    
    kf = KFold(n_splits=n_splits, shuffle=False)
    models = []
    df_importance = []
    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold+1}/{n_splits}")
        print(f"Fold {fold+1} 验证集日期范围: "
              f"{X.index[valid_idx].get_level_values(0).min()} 至 " f"{X.index[valid_idx].get_level_values(0).max()}")
        print(f"Fold {fold+1} 训练集样本数: {len(train_idx)}, 验证集样本数: {len(valid_idx)}")
        
        # 划分数据
        X_train = np.ascontiguousarray(X.iloc[train_idx].values, dtype=np.float32)
        y_train = y.iloc[train_idx].values.astype(np.float32)
        X_valid = np.ascontiguousarray(X.iloc[valid_idx].values, dtype=np.float32)
        y_valid = y.iloc[valid_idx].values.astype(np.float32)
        
        # 边界置NaN（避免未来信息泄露）
        X_valid[:60*4] = np.nan
        y_valid[:60*4] = np.nan
        X_valid[-60*4:] = np.nan
        y_valid[-60*4:] = np.nan
        
        # 标准化
        y_valid = y_valid / nanstd(y_valid)
        y_train = y_train / nanstd(y_train)
        
        train_set = lgb.Dataset(
            X_train, y_train,
            categorical_feature=categorical_col_idx,
            feature_name=features,
            free_raw_data=True
        )
        valid_set = lgb.Dataset(
            X_valid, y_valid,
            categorical_feature=categorical_col_idx,
            feature_name=features,
            reference=train_set
        )
        
        evals_result = {}
        
        model = lgb.train(
            params,
            train_set,
            num_boost_round=10000,
            valid_sets=[valid_set],
            valid_names=['valid_0'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=500, min_delta=1e-7),
                lgb.log_evaluation(100),
                lgb.record_evaluation(evals_result),
            ]
        )
        
        if plot_train:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            lgb.plot_metric(
                evals_result,
                metric=params['metric'],
                dataset_names=['valid_0'],
                title=f'Fold {fold + 1} Training Progress'
            )
            plt.show()
        
        # 预测并计算IC
        valid_pred = model.predict(X_valid)
        valid_mask = ~np.isnan(y_valid) & ~np.isnan(valid_pred)
        
        if np.sum(valid_mask) > 1:
            valid_ic = np.corrcoef(valid_pred[valid_mask], y_valid[valid_mask])[0, 1]
        else:
            valid_ic = np.nan
        print("valid ic:", valid_ic)
        
        models.append(model)
        
        print(f"模型最佳迭代次数 = {model.best_iteration}")
        if model.best_iteration == 1:
            print("模型不收敛，跳过该模型")
            continue
        
        # 特征重要性
        importance_split = model.feature_importance(importance_type='split')
        importance_gain = model.feature_importance(importance_type='gain')
        
        fold_importance = pd.DataFrame({
            'Feature': features,
            'Importance_split': importance_split,
            'Importance_gain': importance_gain,
        }).set_index('Feature')
        
        df_importance.append(fold_importance)
    
    if len(df_importance) > 0:
        importance_all = pd.concat(df_importance).groupby('Feature').sum().sort_values(
            by='Importance_split', ascending=False
        )
    else:
        importance_all = pd.DataFrame(columns=['Importance_split', 'Importance_gain'])
    
    return models, importance_all


def save_importance(importance_df: pd.DataFrame, filepath: str):
    """
    保存特征重要性结果
    
    Parameters:
    -----------
    importance_df : pd.DataFrame
        特征重要性DataFrame
    filepath : str
        保存路径
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(path, index=True)
    print(f"特征重要性已保存到 {path}")
