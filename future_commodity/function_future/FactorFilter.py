import pandas as pd
import numpy as np
from typing import List
from scipy import stats


class FactorFilter:
    def __init__(
        self,
        importance_df: pd.DataFrame,
        corr_df: pd.DataFrame,
        group_df: pd.DataFrame,
        factor_info: pd.DataFrame,
        factor_to_choose: List[str]
    ):
        self.importance_df = importance_df
        self.corr_df = corr_df
        self.group_df = group_df

        self.factor_info = factor_info.reset_index(names='factor')
        self.factor_to_choose = factor_to_choose
        self.importance_df = self.importance_df.merge(self.group_df, left_on='Feature', right_on='feature', how='left')
        self.importance_df['group'] = ((self.importance_df.group1 - self.importance_df.group2)<0).astype(int)
        # print(self.importance_df)

    def factor_info_select(self, nan_rate=0.5, mode_rate=0.8, verbose=True):
        if verbose:
            print("特征分布性过滤：")
        fac = self.factor_info[self.factor_info.nan_rate < nan_rate]
        fac = fac[fac.mode_rate < mode_rate]

        raw_len = len(self.factor_to_choose)
        self.factor_to_choose = [x for x in self.factor_to_choose if x in fac.factor.to_list()]

        new_len = len(self.factor_to_choose)
        if verbose:
            print(f'筛选之前因子数量：{raw_len}， 筛选之后因子数量：{new_len}')
        return None
    
    def importance_select_by_group(self, cut_num_1, cut_num_2, same_name_cut, verbose=True):
        if verbose:
            print("\n特征重要性过滤：", cut_num_1, cut_num_2, "\t待筛因子个数", len(self.factor_to_choose))
            print('无向因子有效个数：', len(self.importance_df[(self.importance_df.group==0)&(self.importance_df.Importance_split>0)]),
                  '有向因子有效个数：', len(self.importance_df[(self.importance_df.group==1)&(self.importance_df.Importance_split>0)]))
        # 1有向因子
        importance_df1 = self.importance_df[self.importance_df.group==0].reset_index(drop=True)
        importance_df1 = importance_df1[importance_df1['Importance_split']>0]

        fac1 = importance_df1.head(cut_num_1)

        filtered_factors = []
        seen_suffixes = {}

        factor_to_choose1 = [x for x in fac1.Feature.to_list() if x in self.factor_to_choose]
        for factor in factor_to_choose1:
            prefix = factor.split("_")[0]
            if prefix == 'STK':
                if prefix not in seen_suffixes:
                    seen_suffixes[prefix] = 1
                    filtered_factors.append(factor)
                elif seen_suffixes[prefix] < 20:
                    seen_suffixes[prefix] += 1  
                    filtered_factors.append(factor)      
                continue
            
            suffix = "_".join(factor.split("_")[:-1])  # 获取前缀部分
            if suffix not in seen_suffixes:
                seen_suffixes[suffix] = 1  
                filtered_factors.append(factor)
            elif seen_suffixes[suffix] < same_name_cut:
                seen_suffixes[suffix] += 1  
                filtered_factors.append(factor)
                
        factor_to_choose1 = filtered_factors

        _importance = self.importance_df[self.importance_df.Feature.isin(factor_to_choose1)]
        if verbose:
            print(f"  选择有向特征 {len(factor_to_choose1)} 个，重要性范围: "
            f"{_importance['Importance_split'].max():.2f} - "
            f"{_importance['Importance_split'].min():.2f}")
    
        # 2无向因子
        importance_df2 = self.importance_df[self.importance_df.group==1].reset_index(drop=True)
        importance_df2 = importance_df2[importance_df2['Importance_split']>0]

        fac2 = importance_df2.head(cut_num_2)

        filtered_factors = []
        seen_suffixes = {}

        factor_to_choose2 = [x for x in fac2.Feature.to_list() if x in self.factor_to_choose]
        for factor in factor_to_choose2:

            prefix = factor.split("_")[0]
            if prefix == 'STK':
                if prefix not in seen_suffixes:
                    seen_suffixes[prefix] = 1
                    filtered_factors.append(factor)
                elif seen_suffixes[prefix] < 10:
                    seen_suffixes[prefix] += 1  
                    filtered_factors.append(factor)     
                continue

            suffix = "_".join(factor.split("_")[:-1])  # 获取前缀部分
            if suffix not in seen_suffixes:
                seen_suffixes[suffix] = 1  
                filtered_factors.append(factor)
            elif seen_suffixes[suffix] < same_name_cut:
                seen_suffixes[suffix] += 1  
                filtered_factors.append(factor)

        factor_to_choose2 = filtered_factors
        _importance = self.importance_df[self.importance_df.Feature.isin(factor_to_choose2)]
        if verbose:
            print(f"  选择无向特征 {len(factor_to_choose2)} 个，重要性范围: "
            f"{_importance['Importance_split'].max():.2f} - "
            f"{_importance['Importance_split'].min():.2f}")

        self.factor_to_choose = factor_to_choose1 + factor_to_choose2
        # print(self.factor_to_choose)
        return None

    def permutation_select(self, cut_num, same_name_cut, verbose=True):

        fac = self.importance_df.sort_values('importance_permutation').head(cut_num)
        if verbose:
            print("\n特征重要性过滤：", cut_num, "  待筛因子个数", len(self.factor_to_choose), 
                  "重要性范围: "
                    f"{fac['importance_permutation'].max() * 10000:.2f}%% - "
                    f"{fac['importance_permutation'].min() * 10000:.2f}%%")
        
        filtered_factors = []
        seen_prefixes = {}

        self.factor_to_choose = [x for x in self.factor_to_choose if x in fac.Feature.to_list()]
        for factor in self.factor_to_choose:
            prefix = "_".join(factor.split("_")[:-1])  # 获取前缀部分
            if prefix not in seen_prefixes:
                seen_prefixes[prefix] = 1  
                filtered_factors.append(factor)
            elif seen_prefixes[prefix] < same_name_cut:
                seen_prefixes[prefix] += 1  
                filtered_factors.append(factor)
        self.factor_to_choose = filtered_factors
        _importance = self.importance_df[self.importance_df.Feature.isin(self.factor_to_choose)]
        if verbose:
            print(f"  选择特征 {len(filtered_factors)} 个，重要性范围: "
            f"{_importance['Importance_split'].max():.2f} - "
            f"{_importance['Importance_split'].min():.2f}")
        return None

    def ic_select(self, th, verbose=True):
        if verbose:
            print("\n单因子ic过滤：", th, f"ic范围{round(self.factor_info.ic.min(), 4)} ~ {round(self.factor_info.ic.max(), 4)}",  "  待筛因子个数", len(self.factor_to_choose))
        factor_info = self.factor_info.copy()
        factor_info = factor_info[factor_info.ic.abs() >= th].reset_index(drop=True)
        self.factor_to_choose = [x for x in self.factor_to_choose if x in factor_info.factor.to_list()]

    def sp_select(self, th, verbose=True):
        if verbose:
            print("\n单因子sharpe过滤：", th, f"sp范围{round(self.factor_info.sharpe.min(), 4)} ~ {round(self.factor_info.sharpe.max(), 4)}",  "  待筛因子个数", len(self.factor_to_choose))
        factor_info = self.factor_info.copy()
        factor_info = factor_info[factor_info.sharpe.abs() >= th].reset_index(drop=True)
        self.factor_to_choose = [x for x in self.factor_to_choose if x in factor_info.factor.to_list()]

    def day_cut(self, num_limit: int=2, verbose=True):
        day_fac_lst = [x for x in self.factor_to_choose if 'day_' in x]
        if verbose:
            print("\n已挑选日频开盘因子数量：", len(day_fac_lst), day_fac_lst)
        if len(day_fac_lst) < num_limit:
            return True
        else:
            self.factor_to_choose = [x for x in self.factor_to_choose if x not in day_fac_lst[num_limit:]]
        day_fac_lst = [x for x in self.factor_to_choose if 'day_' in x]
        if verbose:
            print("\n保留日频开盘因子数量：", len(day_fac_lst), day_fac_lst)
        
    def corr_select(self, feature_num_limit: int=300,
                    corr_limit: float=0.9, verbose=True):
        if verbose:
            print("\n特征相关性性过滤：", feature_num_limit, "    待筛因子个数", len(self.factor_to_choose))
        corr = self.corr_df.copy().fillna(0)
        cols = self.factor_to_choose
        corr = corr.loc[cols, cols]
        
        final_choose = []

        remaining_features = self.factor_to_choose.copy()
        while remaining_features:
            if len(final_choose) == feature_num_limit:
                break
            best_feature = remaining_features[0]
            corr_with_selected = corr.loc[best_feature, final_choose]
            if (corr_with_selected < corr_limit).all(): # type: ignore
                final_choose.append(best_feature)
            remaining_features.remove(best_feature)  
        if verbose:
            print(f"  选择特征 {len(final_choose)} 个, 特征最大相关性{corr.loc[final_choose, final_choose].values.max()}")
        self.factor_to_choose = final_choose
        return None

    def cross_symbol_select(self, symbol, max_cross_ratio=0.15, verbose=True):
        """
        跨品种因子比例限制。
        如果跨品种因子占比超过 max_cross_ratio，按重要性保留 top N，去掉多余的。
        """
        import re
        # 识别跨品种因子：如 A_B_xxx, A_M_xxx
        cross_pattern = re.compile(rf'^{re.escape(symbol)}_[A-Z]')
        cross_factors = [f for f in self.factor_to_choose if cross_pattern.match(f)]
        total = len(self.factor_to_choose)
        cross_count = len(cross_factors)
        
        if total == 0:
            return
        
        current_ratio = cross_count / total
        if verbose:
            print(f"\n跨品种因子限制：当前 {cross_count}/{total} ({current_ratio:.1%}), 上限 {max_cross_ratio:.1%}")
        
        if current_ratio <= max_cross_ratio:
            return
        
        # 超过比例，按重要性排序，保留 top K
        max_allowed = int(total * max_cross_ratio)
        if max_allowed < 1:
            max_allowed = 1
        
        # 获取跨品种因子的重要性排名
        cross_importance = self.importance_df[
            self.importance_df['Feature'].isin(cross_factors)
        ][['Feature', 'Importance_split']].sort_values('Importance_split', ascending=False)
        
        keep_cross = cross_importance.head(max_allowed)['Feature'].tolist()
        drop_cross = [f for f in cross_factors if f not in keep_cross]
        
        self.factor_to_choose = [f for f in self.factor_to_choose if f not in drop_cross]
        if verbose:
            print(f"  保留跨品种因子 {len(keep_cross)} 个，去掉 {len(drop_cross)} 个")
            print(f"  筛选后: {len(self.factor_to_choose)} 个因子, 跨品种占比 {len(keep_cross)/len(self.factor_to_choose):.1%}")

    # ================== 新增：月度稳定性筛选 ==================

    def calc_monthly_stability(self,
                               factor_df,
                               max_outlier_ratio=0.25,
                               mean_r2_thresh=0.4,
                               std_r2_thresh=0.7,
                               p95_r2_thresh=0.7,
                               p05_r2_thresh=0.7,
                               coverage_r2_thresh=0.6,
                               boundary_drift_thresh=0.15,
                               max_consecutive_outliers=2):
        """
        计算月度分布稳定性。
        主指标：boundary_drift_score（月度边界历史排名漂移）。
        原 R² 指标已保留计算但暂不参与 is_stable 判定。
        返回 DataFrame，index=factor
        """
        df = factor_df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        numeric_df = df.select_dtypes(include=[np.number])

        # 全局分位数（用于覆盖率计算）
        global_p90 = numeric_df.quantile(0.90)
        global_p10 = numeric_df.quantile(0.10)

        # 覆盖率掩码
        high_mask = numeric_df.gt(global_p90, axis=1)
        low_mask = numeric_df.lt(global_p10, axis=1)

        # 按月分组
        monthly_period = numeric_df.index.to_period('M')
        monthly = numeric_df.groupby(monthly_period)
        monthly_high_cov = high_mask.groupby(monthly_period).mean()
        monthly_low_cov = low_mask.groupby(monthly_period).mean()

        results = []

        for col in numeric_df.columns:
            monthly_mean = monthly[col].mean()
            monthly_std = monthly[col].std()
            monthly_p95 = monthly[col].quantile(0.95)
            monthly_p05 = monthly[col].quantile(0.05)

            n_months = len(monthly_mean)
            if n_months < 3:
                continue

            x = np.arange(n_months)

            def _regress_r2(y):
                valid = ~np.isnan(y)
                if valid.sum() < 3:
                    return 0.0
                _, _, r, _, _ = stats.linregress(x[valid], y[valid])
                return r**2 if not np.isnan(r) else 0.0

            mean_r2 = _regress_r2(monthly_mean.values)
            std_r2 = _regress_r2(monthly_std.values)
            p95_r2 = _regress_r2(monthly_p95.values)
            p05_r2 = _regress_r2(monthly_p05.values)
            high_cov_r2 = _regress_r2(monthly_high_cov[col].values)
            low_cov_r2 = _regress_r2(monthly_low_cov[col].values)

            # ========== 新增：boundary_drift_score ==========
            all_values = numeric_df[col].dropna().values
            boundary_drifts = []
            if len(all_values) > 0:
                sorted_values = np.sort(all_values)
                for _, group in monthly[col]:
                    month_values = group.dropna().values
                    if len(month_values) == 0:
                        continue
                    p95 = np.percentile(month_values, 95)
                    p05 = np.percentile(month_values, 5)
                    q95 = np.searchsorted(sorted_values, p95) / len(sorted_values)
                    q05 = np.searchsorted(sorted_values, p05) / len(sorted_values)
                    drift = abs(q95 - 0.95) + abs(q05 - 0.05)
                    boundary_drifts.append(drift)
            boundary_drift_score = np.mean(boundary_drifts) if boundary_drifts else 0.0

            # MAD outlier 检测（基于月度均值）
            y_mean = monthly_mean.values
            median_m = np.nanmedian(y_mean)
            mad_m = np.nanmedian(np.abs(y_mean - median_m))
            if mad_m < 1e-10:
                mad_m = 1e-10
            outlier_mask = np.abs(y_mean - median_m) > 3 * mad_m
            outlier_ratio = outlier_mask.sum() / n_months

            consecutive = 0
            max_consecutive = 0
            for is_out in outlier_mask:
                if is_out:
                    consecutive += 1
                    max_consecutive = max(max_consecutive, consecutive)
                else:
                    consecutive = 0

            # is_stable：R² 硬门槛（boundary_drift_score 已搁置）
            is_stable = (
                (mean_r2 <= mean_r2_thresh) and
                (std_r2 <= std_r2_thresh) and
                (p95_r2 <= p95_r2_thresh) and
                (p05_r2 <= p05_r2_thresh)
            )

            results.append({
                'factor': col,
                'n_months': n_months,
                'mean_r2': mean_r2,
                'std_r2': std_r2,
                'p95_r2': p95_r2,
                'p05_r2': p05_r2,
                'high_coverage_r2': high_cov_r2,
                'low_coverage_r2': low_cov_r2,
                'boundary_drift_score': boundary_drift_score,
                'outlier_ratio': outlier_ratio,
                'max_consecutive_outliers': max_consecutive,
                'is_stable': is_stable,
                'monthly_mean_min': np.nanmin(y_mean),
                'monthly_mean_max': np.nanmax(y_mean),
            })

        return pd.DataFrame(results).set_index('factor')

    def monthly_stability_select(self, factor_df, verbose=True, **kwargs):
        """
        月度稳定性筛选。只保留 is_stable=True 的因子。
        返回 stability_df（供后续可视化使用）
        """
        stability_df = self.calc_monthly_stability(factor_df, **kwargs)
        stable_factors = stability_df[stability_df['is_stable']].index.tolist()

        before = len(self.factor_to_choose)
        removed = [f for f in self.factor_to_choose if f not in stable_factors]
        self.factor_to_choose = [f for f in self.factor_to_choose if f in stable_factors]
        after = len(self.factor_to_choose)

        if verbose:
            print(f"\n[月度稳定性] {before} -> {after} (淘汰 {before-after})")
            # 打印当前 pool 中被淘汰的典型因子
            if removed:
                unstable = stability_df.loc[removed].sort_values('mean_r2', ascending=False).head(5)
                for fac, row in unstable.iterrows():
                    reason = []
                    if row['mean_r2'] > kwargs.get('mean_r2_thresh', 0.8):
                        reason.append(f"均值漂移(R²={row['mean_r2']:.3f})")
                    if row['std_r2'] > kwargs.get('std_r2_thresh', 0.7):
                        reason.append(f"波动漂移(R²={row['std_r2']:.3f})")
                    if row['p95_r2'] > kwargs.get('p95_r2_thresh', 0.7):
                        reason.append(f"P95漂移(R²={row['p95_r2']:.3f})")
                    if row['p05_r2'] > kwargs.get('p05_r2_thresh', 0.7):
                        reason.append(f"P05漂移(R²={row['p05_r2']:.3f})")
                    if reason:
                        print(f"  淘汰 {fac}: {', '.join(reason)}")

        return stability_df

    def run_full_pipeline(self, factor_df, config, verbose=True):
        """
        一键跑完整筛选链，返回 summary DataFrame + stability_df。

        config 示例：
        {
            "info_select": {"nan_rate": 0.8, "mode_rate": 0.9},
            "importance_select_by_group": {"cut_num_1": 300, "cut_num_2": 200, "same_name_cut": 5},
            "ic_select": {"th": 0.0},
            "sp_select": {"th": 0},
            "day_cut": {"num_limit": 5},
            "monthly_stability": {
                "max_outlier_ratio": 0.25,
                "trend_r2_thresh": 0.4,
                "slope_abs_thresh": 0.01,
            },
            "corr_select": {"feature_num_limit": 300, "corr_limit": 0.9},
            "exclude_factors": ['datetime', 'instrument']
        }
        """
        steps = []

        def _record(step_name, before, after, note=""):
            steps.append({
                "step": step_name,
                "before": before,
                "after": after,
                "delta": before - after,
                "note": note
            })

        # 初始
        _record("init", len(self.factor_to_choose), len(self.factor_to_choose), "初始因子池")

        # exclude
        if config.get("exclude_factors"):
            before = len(self.factor_to_choose)
            self.factor_to_choose = [f for f in self.factor_to_choose if f not in config["exclude_factors"]]
            _record("exclude", before, len(self.factor_to_choose), f"排除指定列")

        # info_select
        if config.get("info_select"):
            before = len(self.factor_to_choose)
            self.factor_info_select(**config["info_select"], verbose=False)
            p = config["info_select"]
            _record("info_select", before, len(self.factor_to_choose), 
                    f"nan<{p.get('nan_rate','?')}, mode<{p.get('mode_rate','?')}")

        # importance_select_by_group
        if config.get("importance_select_by_group"):
            before = len(self.factor_to_choose)
            self.importance_select_by_group(**config["importance_select_by_group"], verbose=False)
            p = config["importance_select_by_group"]
            _record("importance", before, len(self.factor_to_choose),
                    f"dir:{p.get('cut_num_1','?')}, undir:{p.get('cut_num_2','?')}")

        # ic_select
        if config.get("ic_select"):
            before = len(self.factor_to_choose)
            self.ic_select(**config["ic_select"], verbose=False)
            _record("ic_select", before, len(self.factor_to_choose), f"abs>={config['ic_select'].get('th','?')}")

        # sp_select
        if config.get("sp_select"):
            before = len(self.factor_to_choose)
            self.sp_select(**config["sp_select"], verbose=False)
            _record("sp_select", before, len(self.factor_to_choose), f"abs>={config['sp_select'].get('th','?')}")

        # day_cut
        if config.get("day_cut"):
            before = len(self.factor_to_choose)
            self.day_cut(**config["day_cut"], verbose=False)
            _record("day_cut", before, len(self.factor_to_choose), f"limit={config['day_cut'].get('num_limit','?')}")

        # monthly_stability
        stability_df = None
        if config.get("monthly_stability"):
            before = len(self.factor_to_choose)
            # 自动剔除非因子列
            fac_df = factor_df.copy()
            drop_cols = ['pred_ret', 'hour', 'datetime', 'instrument', 'rtn_1', 'rtn_5', 'rtn_10', 'rtn_20']
            fac_df = fac_df.drop(columns=[c for c in drop_cols if c in fac_df.columns], errors='ignore')
            stability_df = self.monthly_stability_select(fac_df, verbose=False, **config["monthly_stability"])
            p = config["monthly_stability"]
            _record("monthly_stable", before, len(self.factor_to_choose),
                    f"mean_r2<{p.get('mean_r2_thresh','?')}, std_r2<{p.get('std_r2_thresh','?')}")

        # corr_select
        if config.get("corr_select"):
            before = len(self.factor_to_choose)
            self.corr_select(**config["corr_select"], verbose=False)
            p = config["corr_select"]
            _record("corr_select", before, len(self.factor_to_choose),
                    f"limit={p.get('feature_num_limit','?')}, corr<{p.get('corr_limit','?')}")

        # cross_symbol_select
        if config.get("cross_symbol_select"):
            before = len(self.factor_to_choose)
            self.cross_symbol_select(**config["cross_symbol_select"], verbose=False)
            p = config["cross_symbol_select"]
            _record("cross_symbol", before, len(self.factor_to_choose),
                    f"max_ratio={p.get('max_cross_ratio','?')}, symbol={p.get('symbol','?')}")

        summary = pd.DataFrame(steps)
        return summary, stability_df

    def run_selection(
        self, 
        info_select_params=None, 
        importance_select_params=None,
        importance_select_by_group_params=None,
        ic_select = None,
        sp_select = None,
        permutation_select_params = None,
        permutation_select_params_ascending_False = None,
        corr_select_params=None,
        exclude_factors=None,
        day_cut=None
    ):
        assert not (importance_select_params and importance_select_by_group_params), \
            "特征重要性方式只能选择其中一种"
        if info_select_params:
            self.factor_info_select(**info_select_params)  # 解包字典到命名参数
        if importance_select_params:
            self.importance_select(**importance_select_params)
        if importance_select_by_group_params:
            self.importance_select_by_group(**importance_select_by_group_params)
        if ic_select:
            self.ic_select(**ic_select)
        if sp_select:
            self.sp_select(**sp_select)
        if permutation_select_params:
            self.permutation_select(**permutation_select_params)
        if permutation_select_params_ascending_False:
            self.permutation_select_params_ascending_False(**permutation_select_params_ascending_False)
        if corr_select_params:
            self.corr_select(**corr_select_params)
        if day_cut:
            self.day_cut(**day_cut)
        if exclude_factors:
            self.factor_to_choose = [f for f in self.factor_to_choose if f not in exclude_factors]
            print(f"\n排除指定因子后，剩余特征 {len(self.factor_to_choose)} 个")
