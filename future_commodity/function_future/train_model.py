import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from pathlib import Path
import json
from scipy.stats import skew, kurtosis,entropy 
import os
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import mutual_info_score
import numpy as np
import lightgbm as lgb
from typing import Dict, Tuple, Any
os.environ['LIGHTGBM_VERBOSITY'] = '0'

def nanmean(arr, axis=None):
    """
    计算忽略NaN的均值
    :param arr: 输入数组（支持多维）
    :param axis: 沿指定轴计算（None表示整个数组）
    :return: 均值结果
    """
    arr = np.asarray(arr)
    mask = ~np.isnan(arr)  # 非NaN的掩码
    
    if axis is None:
        valid_sum = np.sum(arr[mask])
        valid_count = np.sum(mask)
    else:
        valid_sum = np.nansum(arr, axis=axis)
        valid_count = np.sum(mask, axis=axis)
    
    return valid_sum / valid_count

def nanstd(arr, axis=None, ddof=0):
    """
    计算忽略NaN的标准差
    :param arr: 输入数组（支持多维）
    :param axis: 沿指定轴计算（None表示整个数组）
    :param ddof: 自由度调整（默认0，与nanstd一致）
    :return: 标准差结果
    """
    arr = np.asarray(arr)
    mean = nanmean(arr, axis=axis)  # 使用nanmean计算均值
    
    if axis is None:
        squared_diff = (arr - mean) ** 2
        valid_sum = np.sum(squared_diff[~np.isnan(arr)])
        valid_count = np.sum(~np.isnan(arr))
    else:
        # 保持维度以便广播
        if axis == 0:
            keepdims = (arr.ndim == 1)
        else:
            keepdims = True
        squared_diff = (arr - np.expand_dims(mean, axis=axis)) ** 2
        valid_sum = np.nansum(squared_diff, axis=axis, keepdims=keepdims)
        valid_count = np.sum(~np.isnan(arr), axis=axis, keepdims=keepdims)
    
    variance = valid_sum / (valid_count - ddof)
    return np.sqrt(variance)

class TimeSeriesAnalyzer:
    def __init__(self, symbol, factor_col, train_end_date, config_loader):
    
        self.factor_col = factor_col
        self.symbol = symbol
        self.config_loader = config_loader

        # 数据配置
        self.train_end_date = train_end_date
        self.full_data = None
        self.train_data = None
        self.test_data = None

        # 路径配置
        # self.window_end = '2025-03-01'
        self.data_path = Path(f'/mnt/Data/writable/liaoyuyang/factor/{self.symbol}/all_fac/all_factor.feather')
        self.rtn_path = Path(f"/mnt/Data/writable/liaoyuyang/data/1min/active/main_{symbol}.csv")
        self.model_save_path = Path("/mnt/Data/writable/liaoyuyang/model/lightgbm/model_save")

        # 列名配置（根据实际数据调整）
        self.ts_col = 'datetime'
        self.instrument_col = 'instrument'
        self.target_col = 'pred_ret'
        self.category_col = []

    def load_and_prepare_data(self, start_date=None, set_category_col=None, log_rtn=False, label_col=None, cut=True):
        """加载并预处理数据"""

        self.label_col = label_col
        print(f"正在从 {self.data_path} 加载数据...")
        self.full_data = pd.read_feather(self.data_path)
        if start_date:
            self.full_data = self.full_data[self.full_data['datetime'] >= start_date]
            print(f"数据已截取至 {start_date} 之后，当前形状: {self.full_data.shape}")

        if set_category_col == ['hour']:
            self.full_data['hour'] = self.full_data.datetime.dt.hour
            self.full_data['minute'] = self.full_data.datetime.dt.minute
            self.full_data['hour'] = self.full_data['hour'].astype('category')
            self.category_col = ['hour']

        if not pd.api.types.is_datetime64_any_dtype(self.full_data[self.ts_col]):
            self.full_data[self.ts_col] = pd.to_datetime(self.full_data[self.ts_col])
        if not isinstance(self.train_end_date, (pd.Timestamp)):
            try:
                self.train_end_date = pd.to_datetime(self.train_end_date)
            except Exception as e:
                print(f"无法转换 train_end_date: {self.train_end_date}, 错误: {e}")

        rtn_df = pd.read_csv(self.rtn_path).rename(columns={'ts': 'datetime'})
        if not pd.api.types.is_datetime64_any_dtype(rtn_df[self.label_col]):
            rtn_df[self.ts_col] = pd.to_datetime(rtn_df[self.ts_col])

        if log_rtn:
            rtn_df[self.label_col] = np.log1p(rtn_df[self.label_col])

        self.full_data = self.full_data.merge(
            rtn_df[['datetime', self.label_col]], 
            left_on='datetime',      # 使用full_data的ts列作为键
            right_on='datetime',  # 使用rtn_df的索引（ts）作为键
            how='left'         # 左连接保留所有full_data的行
        )

        if cut:
            self.full_data = self.full_data.set_index('datetime')
            self.full_data = self.config_loader.df_cut_time(self.full_data, self.config_loader.get_instrument_config(self.symbol)['trading_hours'], 10)
            self.full_data = self.full_data.reset_index()
            
        self.full_data.rename(columns={self.label_col: self.target_col},inplace=True)

        self.train_data = self.full_data[self.full_data.datetime <= self.train_end_date]
        self.test_data = self.full_data[self.full_data.datetime > self.train_end_date]

        self.full_data = pd.concat([self.train_data, self.test_data])

        print(f"训练数据加载完成，形状: {self.train_data.shape}")
        print(f"训练集时间范围: {self.train_data[self.ts_col].min()} 至 {self.train_data[self.ts_col].max()}")

        assert self.train_data.datetime.max() <= self.train_end_date, "训练集包含超出训练截止日期的数据！"
        assert self.test_data.datetime.min() > self.train_end_date, "测试集包含早于训练截止日期的数据！"
        assert not set(self.train_data.datetime).intersection(set(self.test_data.datetime)), "训练集和测试集的时间戳存在重叠！"

class LGBMTrainer:
    """LightGBM训练器，支持多种时间序列划分策略"""
    
    def __init__(self, analyzer):
        """
        初始化训练器
        :param analyzer: TimeSeriesAnalyzer实例，提供数据和划分方法
        """
        # 列名配置（根据实际数据调整）
        
        self.instrument_col = 'instrument'

        self.analyzer = analyzer
        self.ts_col = analyzer.ts_col
        self.label_col = analyzer.label_col
        self.target_col = analyzer.target_col
        self.category_col = analyzer.category_col
        self.train_end_date = analyzer.train_end_date
        self.factor_col = analyzer.factor_col

        self.models = {}
        self.results = {}
        self.model_dir = Path("/mnt/Data/writable/liaoyuyang/model/lightgbm/KFoldModel/models")
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 默认参数（可根据需要修改）
        self.default_params = {
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

        self.lgb_round_params = {
            "round": 10000,
            "min_delta": 5e-7,
            "early_stopping_rounds": 500,
            "verbose": False
        }

    def set_params(self, params: Dict):
        """更新模型参数"""
        self.default_params.update(params)
    
    def train_kfold_v0(self, n_splits: int = 4, model_folder_name: str='temp', custom_params: Dict = None, group: str='0', plot_train: bool=False) -> Dict:
        """
        传统K折交叉验证训练
        :param n_splits: 折数
        :param custom_params: 自定义参数
        :return: 训练结果字典
        """
        params = self.default_params.copy()
        if custom_params:
            params.update(custom_params)
            
        X = self.analyzer.train_data[self.factor_col + self.category_col].values
        y = self.analyzer.train_data[self.analyzer.target_col].values
        
        if self.category_col:
            categorical_col_idx = [
                    i for i, col in enumerate(self.factor_col + self.category_col) 
                    if col in self.category_col
                ]
        else:
            categorical_col_idx = []

        model_dir = self.model_dir / model_folder_name
        model_dir.mkdir(parents=True, exist_ok=True)

        kf = KFold(n_splits=n_splits, shuffle=False)
        fold_results = {}
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            X_val[:60*4] = np.nan
            y_val[:60*4] = np.nan
            X_val[-60*4:] = np.nan
            y_val[-60*4:] = np.nan

            y_val = y_val / nanstd(y_val)
            y_train = y_train / nanstd(y_train)
            
            train_set = lgb.Dataset(X_train, y_train, categorical_feature=categorical_col_idx,
                                    feature_name = self.factor_col + self.category_col,free_raw_data=True)
            valid_set = lgb.Dataset(X_val, y_val, categorical_feature=categorical_col_idx,         
                                    feature_name = self.factor_col + self.category_col,reference=train_set)
            
            # 训练模型
            evals_result = {}
            model = lgb.train(
                params,
                train_set,
                valid_sets=[valid_set],
                num_boost_round=self.lgb_round_params['round'],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=self.lgb_round_params['early_stopping_rounds'],min_delta=self.lgb_round_params['min_delta'], verbose=self.lgb_round_params['verbose']),
                    lgb.log_evaluation(period=200),
                    lgb.record_evaluation(evals_result)
                ]
            )
            
            if plot_train:        
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 6))
                lgb.plot_metric(
                    evals_result,
                    metric=self.default_params['metric'],
                    dataset_names=['valid_0'],  
                    title=f'Fold {fold} Training Progress'
                )
                plt.show()  # 确保显示图形

            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            metrics = {
                'train_rmse': np.sqrt(nanmean((y_train - train_pred)**2)),
                'val_rmse': np.sqrt(nanmean((y_val - val_pred)**2)),
                'best_iteration': model.best_iteration
            }
            fold_results[f"fold_{fold}"] = metrics
            self.models[f"kfold_fold{fold}"] = model
            print(metrics)

            pred = model.predict(self.analyzer.test_data[self.factor_col + self.category_col].values)
            true = self.analyzer.test_data[self.target_col].values

            mask = ~np.isnan(pred) & ~np.isnan(true)
            corr_coef = np.corrcoef(pred[mask], true[mask])[0, 1] if np.sum(mask) > 1 else np.nan
            print(f'test_corr: {corr_coef:.4f}')

            model_file = self.model_dir / model_folder_name / f"kfold_fold{fold}_{group}.lgb"
            model.save_model(str(model_file))

            meta_file = self.model_dir / model_folder_name / f"kfold_fold{fold}_{group}_meta.json"
            with open(meta_file, 'w') as f:
                json.dump({
                    'best_iteration': int(model.best_iteration),
                    'feature_importance': model.feature_importance().tolist(),
                    'params': model.params,
                    'test_corr': float(corr_coef),
                    'fac_num': int(len(self.factor_col)),
                    'avg_abs_true': float(np.nanmean(np.abs(y_val))),
                    'avg_abs_pred': float(np.nanmean(np.abs(val_pred)))
                }, f, indent=2)

        self.results['kfold'] = fold_results

        return fold_results

import json

def save_results(data, filename):
    """改进版存储：同时保存原始数据和可读文本"""
    with open(filename, 'w') as f:
        # 可读文本部分
        f.write("# Human-readable summary\n")
        for i, (score, params, fold_scores, avg_rounds, fac_params) in enumerate(data, 1):
            f.write(f"\n=== Model {i} (Score: {score:.6f}) ===\n")
            f.write("Main Parameters:\n" + json.dumps(params, indent=4) + "\n")
            f.write(f"Fold Scores: {json.dumps(fold_scores)}\n")
            f.write(f"Avg Rounds: {avg_rounds}\n")
            f.write("Factor Params:\n" + json.dumps(fac_params, indent=4) + "\n")
        
        # 完整数据结构部分（便于程序读取）
        f.write("\n# Machine-readable data\n")
        json.dump(
            [{
                "score": item[0],
                "params": item[1],
                "fold_scores": item[2],
                "avg_rounds": item[3],
                "fac_params": item[4]
            } for item in data],
            f,
            indent=2
        )

def load_results(filename):
    """读取改进版存储文件"""
    with open(filename, 'r') as f:
        content = f.read()
    
    # 提取JSON部分（最后一行开始）
    json_str = content.split("# Machine-readable data\n")[-1]
    return json.loads(json_str)