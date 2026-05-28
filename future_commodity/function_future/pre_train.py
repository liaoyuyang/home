from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm  # 进度条（可选）
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import time 
from pathlib import Path
import os
from typing import List, Tuple, Dict
from sklearn.model_selection import KFold 
from datetime import datetime
import pyarrow.feather as feather
import function_future.single_fac_eval as sfe

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

class Pretrainer:
    """预训练处理器，负责特征选择和初始模型训练"""
    def __init__(self, 
                 variety: str,
                 data: pd.DataFrame,
                 train_end_date: str, 
                 ts_col: str = 'datetime', 
                 instrument_col: str = 'instrument', target_col: str = 'pred_ret', train_label: int = 5):
        """
        初始化预训练器
        :param data: 原始数据DataFrame
        :param ts_col: 时间戳列名
        :param instrument_col: 标的物列名
        :param target_col: 目标变量列名
        """
        self.variety = variety
        self.raw_data = data
        self.train_end_date = train_end_date
        self.ts_col = ts_col
        self.instrument_col = instrument_col
        self.target_col = target_col
        self.features = self._get_feature_columns()
        self.importance_dir = Path(f"/mnt/Data/writable/liaoyuyang/factor_eval_commodity/{train_end_date}/importance")
        self.corrdir = Path(f"/mnt/Data/writable/liaoyuyang/factor_eval_commodity/{train_end_date}/correlation")
        self.groupdir = Path(f"/mnt/Data/writable/liaoyuyang/factor_eval_commodity/{train_end_date}/group")
        self.factor_eval_dir = Path(f"/mnt/Data/writable/liaoyuyang/factor_eval_commodity/{train_end_date}")

        self.importance_df = pd.DataFrame()
        self.corr_df = pd.DataFrame()
        self.train_label = train_label
        self.category_col = ['hour']

        os.makedirs(self.importance_dir, exist_ok=True)
        os.makedirs(self.corrdir, exist_ok=True)
        os.makedirs(self.groupdir, exist_ok=True)
        os.makedirs(self.factor_eval_dir, exist_ok=True)
        # 默认预训练参数
        self.default_params = {
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
    
    def _get_feature_columns(self) -> List[str]:
        """获取特征列名（排除时间、标的物和目标列）"""
        return [col for col in self.raw_data.columns 
                if col not in [self.ts_col, self.instrument_col, self.target_col]]
    
    def set_params(self, params = dict()):
        """更新预训练参数"""
        self.default_params.update(params)
    
    def _evaluate_model(self, y_val ,val_pred):
        """評估模型""" 
        position_ = val_pred
        rtn = position_ * y_val
        return -np.mean(rtn) / np.std(rtn)
    
    def _calculate_permutation_importance(
        self, 
        model: lgb.Booster, 
        X_val: pd.DataFrame, 
        y_val: pd.Series, 
        metric: callable = mean_squared_error,
        n_repeats: int = 5,
        random_state: int = 42
    ) -> Dict[str, float]:
        """线程安全的特征重要性计算（修复只读数组问题）"""
        np.random.seed(random_state)
        y_clean = y_val.fillna(0).values
        importance = {}

        # 获取模型实际使用的特征
        used_features = set(model.feature_name()) if hasattr(model, 'feature_name') else set(X_val.columns)
        
        # 转换为可写的numpy数组（关键修复）
        X_arr = np.array(X_val.values, copy=True)  # 确保创建可写副本
        X_arr.setflags(write=True)  # 显式设置为可写
        
        baseline_pred = model.predict(X_arr, num_threads=1)
        baseline_score = metric(y_clean, baseline_pred)
        use_parallel = len(used_features) > 50
        
        if use_parallel:
            from joblib import Parallel, delayed
            import psutil
            import os
            
            # 设置CPU亲和性（跳过前两个核心）
            p = psutil.Process()
            all_cores = set(range(os.cpu_count()))
            available_cores = sorted(all_cores - {0, 1})
            p.cpu_affinity(available_cores)
            
            # 并行计算每个特征
            def process_feature(col_idx, feature):
                if feature not in used_features:
                    return (feature, 0.0)
                    
                # 创建当前列的本地可写副本
                local_X = np.array(X_arr, copy=True)  # 每个进程独立副本
                original_col = local_X[:, col_idx].copy()
                delta_scores = []
                
                for _ in range(n_repeats):
                    np.random.shuffle(local_X[:, col_idx])
                    pred = model.predict(local_X, num_threads=1)
                    delta_scores.append(baseline_score - metric(y_clean, pred))
                    local_X[:, col_idx] = original_col
                    
                return (feature, np.mean(delta_scores))
            
            n_jobs = 40
            results = Parallel(n_jobs=n_jobs)(
                delayed(process_feature)(col_idx, feature)
                for col_idx, feature in tqdm(enumerate(X_val.columns), total=len(X_val.columns),desc="Processing Features")
            )
            importance = dict(results)
            
            # 恢复原始CPU绑定
            p.cpu_affinity(all_cores)
        else:
            # 单进程模式（保持原始逻辑）
            for col_idx, feature in enumerate(X_val.columns):
                if feature not in used_features:
                    importance[feature] = 0.0
                    continue
                    
                original_col = X_arr[:, col_idx].copy()
                delta_scores = []
                for _ in range(n_repeats):
                    np.random.shuffle(X_arr[:, col_idx])
                    pred = model.predict(X_arr, num_threads=1)
                    delta_scores.append(baseline_score - metric(y_clean, pred))
                    X_arr[:, col_idx] = original_col
                importance[feature] = np.mean(delta_scores)
        
        return importance
   
    def get_fac_category_kfold(self, features=None, n_splits=5, n_jobs=32):
        
        if features is None:
            features = self.features

        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_splits, shuffle=False)

        X = self.raw_data[features].fillna(self.raw_data[features].mean())
        y = self.raw_data[self.target_col].fillna(0)
        abs_y = y.abs()

        df_lst = []
        for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
            X_train = X.iloc[train_idx].values
            y_train = y[train_idx].values
            abs_y_train = abs_y[train_idx].values

            def column_corr(col_idx, X_train, y_train, abs_y_train):
                x = X_train[:, col_idx]
                return (
                    np.corrcoef(x, y_train)[0, 1],  
                    np.corrcoef(x, abs_y_train)[0, 1]  
                )
            
            results = Parallel(n_jobs=n_jobs)(
                delayed(column_corr)(i, X_train, y_train, abs_y_train) for i in range(len(features))
            )
            
            # 4. 组装结果DataFrame
            corr_original = [r[0] for r in results]
            corr_abs = [r[1] for r in results]
            
            df_res =  pd.DataFrame({
                'feature': features,
                'corr_original': corr_original,
                'corr_abs': corr_abs
            }).sort_values('corr_original', ascending=False)
            df_res['fold'] = fold
            df_res['group1'] = (df_res['corr_original'] > df_res['corr_abs']).astype(int)
            df_res['group2'] = (df_res['corr_original'] < df_res['corr_abs']).astype(int)
            df_lst.append(df_res)
        res = pd.concat(df_lst)
        self.group_df = res.groupby('feature')[['group1', 'group2']].sum()
        return self.group_df
    
    # def calc_corr(self, features: List[str] = None):
    #     if features is None:
    #         features = self.features
    #     fac = self.raw_data[features].fillna(self.raw_data[features].mean())
    #     feature_corr = pd.DataFrame(
    #         np.abs(np.corrcoef(fac.values, rowvar=False)), 
    #         columns=fac.columns,
    #         index=fac.columns
    #     )
    #     np.fill_diagonal(feature_corr.values, 0)  # 忽略对角线
    #     self.corr_df = feature_corr

    def calc_corr(self, features: List[str] = None):
            if features is None:
                features = self.features
            
            fac = self.raw_data[features].fillna(self.raw_data[features].mean())
            
            # 新增：检查并处理填充后仍有 NaN 的列（全 NaN 列）
            cols_with_all_nan = fac.columns[fac.isna().all()].tolist()
            if cols_with_all_nan:
                print(f"Warning: {len(cols_with_all_nan)} columns are all NaN, filling with 0")
                fac[cols_with_all_nan] = 0
                
            # 分离常数列和有效列
            const_cols = [col for col in features if fac[col].var() < 1e-10]
            valid_cols = [col for col in features if col not in const_cols]
            
            # 计算有效列的相关性
            if valid_cols:
                fac_valid = fac[valid_cols]
                corr_valid = np.abs(np.corrcoef(fac_valid.values, rowvar=False))
                
                # 创建完整矩阵
                n = len(features)
                corr_full = np.zeros((n, n))
                
                # 建立索引映射
                idx_map = {col: i for i, col in enumerate(features)}
                valid_indices = [idx_map[col] for col in valid_cols]
                
                # 填充有效部分
                for i_pos, i_full in enumerate(valid_indices):
                    for j_pos, j_full in enumerate(valid_indices):
                        corr_full[i_full, j_full] = corr_valid[i_pos, j_pos]
                
                # 常数列对角线设为1
                for col in const_cols:
                    idx = idx_map[col]
                    corr_full[idx, idx] = 1.0
            else:
                # 全是常数列
                n = len(features)
                corr_full = np.eye(n)
            
            # 创建DataFrame
            feature_corr = pd.DataFrame(corr_full, columns=features, index=features)
            np.fill_diagonal(feature_corr.values, 0)
            self.corr_df = feature_corr
            
            return self.corr_df

    def train_initial_model(self, features: List[str] = None, 
                          n_splits: int =5, plot_train = True) -> Tuple[lgb.Booster, pd.DataFrame]:
        """
        训练初始模型并计算特征重要性
        :param features: 使用的特征列表（None则使用全部特征）
        :param n_estimators: 树的数量
        :return: (训练好的模型, 特征重要性DataFrame)
        """
        if features is None:
            features = self.features

        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_splits, shuffle=False)
        models = []
        df_importance = []

        X = self.raw_data[features]
        y = self.raw_data[self.target_col]

        categorical_col_idx = [
                i for i, col in enumerate(X.columns.tolist()) 
                if col in self.category_col
            ]

        params = self.default_params

        for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
            print(f"\nFold {fold+1}/{n_splits}")
            print(f"Fold {fold+1} 验证集日期范围: "
                f"{self.raw_data.index[valid_idx].min()} 至 {self.raw_data.index[valid_idx].max()}")
            print(f"Fold {fold+1} 训练集样本数: {len(train_idx)}, 验证集样本数: {len(valid_idx)}")
            # 划分数据（保持原始数据处理逻辑不变）
            X_train = np.ascontiguousarray(X.iloc[train_idx].values, dtype=np.float32)
            y_train = y.iloc[train_idx].values.astype(np.float32)
            X_valid = np.ascontiguousarray(X.iloc[valid_idx].values, dtype=np.float32)  
            y_valid = y.iloc[valid_idx].values.astype(np.float32)

            X_valid[:60*4] = np.nan
            y_valid[:60*4] = np.nan
            X_valid[-60*4:] = np.nan
            y_valid[-60*4:] = np.nan    

            y_valid = y_valid / nanstd(y_valid)
            y_train = y_train / nanstd(y_train)

            train_set = lgb.Dataset(X_train, y_train, categorical_feature=categorical_col_idx,
                                    feature_name = X.columns.tolist(),free_raw_data=True)
            valid_set = lgb.Dataset(X_valid, y_valid, categorical_feature=categorical_col_idx,         
                                    feature_name = X.columns.tolist(),reference=train_set)

            evals_result = {}

            model = lgb.train(
                params,
                train_set,
                num_boost_round=10000,  # 设置足够大的初始轮数让早停生效
                valid_sets=[valid_set],
                valid_names=['valid_0'],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=500,min_delta=1e-7),  # IF

                    lgb.log_evaluation(100),
                    lgb.record_evaluation(evals_result),  # 记录评估结果
                ]
            )
            if plot_train:        
                import matplotlib.pyplot as plt
                # 绘制训练过程指标图（关键修改点：独立显示每个fold的图）
                plt.figure(figsize=(10, 6))
                lgb.plot_metric(
                    evals_result,
                    metric=self.default_params['metric'],
                    dataset_names=['valid_0'],  # 对应valid_names中的名称
                    title=f'Fold {fold + 1} Training Progress'
                )
                plt.show()  # 确保显示图形

            # 预测并计算IC（带NaN过滤）
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
                print("模型不收敛 , 跳过该模型")
                continue
        
            # === 特征重要性对齐 ===
            # 1. 获取模型原始重要性（按模型内部顺序）
            importance_split = model.feature_importance(importance_type='split')
            importance_gain = model.feature_importance(importance_type='gain')
            
            # 2. 构建有序DataFrame
            fold_importance = pd.DataFrame({
                'Feature': X.columns.tolist(),
                'Importance_split': importance_split,
                'Importance_gain': importance_gain,
            }).set_index('Feature')

            # res = self._calculate_permutation_importance(model, pd.DataFrame(X_valid, columns=self.features), pd.Series(y_valid))
            # s = pd.Series(res, name='importance_permutation')
            # fold_importance = fold_importance.join(s).reset_index()

            df_importance.append(fold_importance)

        importance_all = pd.concat(df_importance).groupby('Feature').sum().sort_values(by='Importance_split', ascending=False)
        self.importance_df = importance_all
        return models, importance_all

    def train_initial_model_vol(self, plot_train=True, features: List[str] = None, 
                          n_splits: int =5) -> Tuple[lgb.Booster, pd.DataFrame]:
        """
        训练初始模型并计算特征重要性
        :param features: 使用的特征列表（None则使用全部特征）
        :param n_estimators: 树的数量
        :return: (训练好的模型, 特征重要性DataFrame)
        """
        if features is None:
            features = self.features

        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_splits, shuffle=False)
        models = []
        df_importance = []

        params = self.default_params.copy()
        params['learning_rate'] = 0.01

        X = self.raw_data[features]
        y = self.raw_data[self.target_col]

        for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
            print(f"\nFold {fold+1}/{n_splits}")
            
            X_train = np.ascontiguousarray(X.iloc[train_idx].values, dtype=np.float32)
            y_train = y.iloc[train_idx].values.astype(np.float32)
            X_valid = np.ascontiguousarray(X.iloc[valid_idx].values, dtype=np.float32)
            y_valid = y.iloc[valid_idx].values.astype(np.float32)

            X_valid[:60*4] = np.nan
            y_valid[:60*4] = np.nan
            X_valid[-60*4:] = np.nan
            y_valid[-60*4:] = np.nan    

            train_set = lgb.Dataset(X_train, y_train,
                                    feature_name = X.columns.tolist(),free_raw_data=True)
            valid_set = lgb.Dataset(X_valid, y_valid,        
                                    feature_name = X.columns.tolist(),reference=train_set)  

            evals_result = {}
            model = lgb.train(
                params,
                train_set,
                num_boost_round=10000,  # 设置足够大的初始轮数让早停生效
                valid_sets=[valid_set],
                valid_names=['valid_0'],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=500, min_delta=1e-8),  # 保持早停
                    lgb.log_evaluation(100),
                    lgb.record_evaluation(evals_result),  # 记录评估结果
                ]
            )

            # 预测并计算IC（带NaN过滤）
            valid_pred = model.predict(X_valid)
            valid_mask = ~np.isnan(y_valid) & ~np.isnan(valid_pred)
            
            if np.sum(valid_mask) > 1:
                valid_ic = np.corrcoef(valid_pred[valid_mask], y_valid[valid_mask])[0, 1]
            else:
                valid_ic = np.nan    
            print("valid ic:", valid_ic)

            if plot_train:        
                import matplotlib.pyplot as plt
                # 绘制训练过程指标图（关键修改点：独立显示每个fold的图）
                plt.figure(figsize=(10, 6))
                lgb.plot_metric(
                    evals_result,
                    metric=self.default_params['metric'],
                    dataset_names=['valid_0'],  # 对应valid_names中的名称
                    title=f'Fold {fold+1} Training Progress'
                )
                plt.show()  # 确保显示图形

            models.append(model)

            importance_split = model.feature_importance(importance_type='split')
            importance_gain = model.feature_importance(importance_type='gain')

            feature_names = model.feature_name()

            # 创建 DataFrame
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance_split': importance_split,
                'Importance_gain': importance_gain

            })

            # 按重要性排序
            feature_importance_df = feature_importance_df.sort_values(by='Importance_split', ascending=False)
            feature_importance_df = feature_importance_df.reset_index(drop=True)
            df_importance.append(feature_importance_df)

        importance_all = pd.concat(df_importance).groupby('Feature').sum().sort_values(by='Importance_split', ascending=False)
        self.importance_df = importance_all
        return models, importance_all

    def save_importance(self, filename: str):
        """保存特征重要性结果"""
        path = self.importance_dir / filename
        self.importance_df.to_csv(path, index=True)
        print(f"特征重要性已保存到 {path}")

    def save_corr(self, filename: str):
        """保存特征相关系数结果"""
        path = self.corrdir / filename
        self.corr_df.to_csv(path, index=True)
        print(f"特征相关系数表已保存到 {path}")

    def save_group(self, filename: str):
        """保存因子组别结果"""
        path = self.groupdir / filename
        self.group_df.to_csv(path, index=True)
        print(f"因子组别表已保存到 {path}")

    def run_full_pretraining(self, save_name=None, type_lgb='reg') -> Tuple[List[str], pd.DataFrame]:
        """
        执行完整预训练流程
        :param save_name: 结果保存名称
        :return: (选中特征列表, 特征重要性DataFrame)
        """
        if not save_name:
            save_name = self.train_end_date + '_' + self.variety + '_' + str(self.train_label)

        # 检查并剔除全 NaN 列
        all_nan_cols = self.raw_data.columns[self.raw_data.isna().all()].tolist()
        if all_nan_cols:
            print(f"Warning: 发现 {len(all_nan_cols)} 列全为 NaN，将被剔除: {all_nan_cols}")
            self.raw_data = self.raw_data.drop(columns=all_nan_cols)
            # 更新特征列表
            self.features = self._get_feature_columns()

        if not os.path.exists(self.importance_dir / f"{save_name}_feature_importance_reg.csv"):
            self.train_initial_model()
            self.save_importance(f"{save_name}_feature_importance_reg.csv")
        else: print(save_name)

        if not os.path.exists(self.corrdir / f"{save_name}_feature_corr.csv"):
            self.calc_corr()
            self.save_corr(f"{save_name}_feature_corr.csv") 

        if not os.path.exists(self.groupdir / f"{save_name}_feature_group.csv"):
            self.get_fac_category_kfold()
            self.save_group(f"{save_name}_feature_group.csv") 

        if not os.path.exists(self.factor_eval_dir / f"{self.variety}_single_factor_eval_{self.train_label}.csv"):
            print('最后两列', f'{self.raw_data.columns[[-2,-1]]}')
            # print(1)
            ic_results = sfe.parallel_calc_ic_optimized(self.raw_data.iloc[:, :-2], self.raw_data.iloc[:, -2], n_jobs=1, symbol=self.variety)
            # print(2)
            sharpe_results = sfe.parallel_calc_sharpe_optimized(self.raw_data.iloc[:, :-2], self.raw_data.iloc[:, -2], n_jobs=1, symbol=self.variety)
            # print(3)
            result_df = sfe.analyze_factors_wide(self.raw_data.iloc[:, :-2])
            # print(4)
            result_df.join(pd.concat([ic_results.rename('ic'), sharpe_results.rename('sharpe')], axis=1)).to_csv(f'/mnt/Data/writable/liaoyuyang/factor_eval_commodity/{self.train_end_date}/{self.variety}_single_factor_eval_{self.train_label}.csv')
            # print(5)
        return None
