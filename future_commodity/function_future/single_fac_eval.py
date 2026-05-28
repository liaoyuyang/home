import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import os 
import json
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from tqdm import tqdm



def align_and_dropna(factor_df, rtn_df, factor_col):
    aligned_data = pd.DataFrame({
        'factor': factor_df[factor_col],
        # 'rtn': rtn_df['rtn_5']
        'rtn': rtn_df.values

    }).replace([-np.inf, np.nan], np.nan).dropna()
    return aligned_data['factor'].values, aligned_data['rtn'].values

def fast_corr(x, y):
    return np.corrcoef(x, y)[0, 1]

def parallel_calc_ic_optimized(factor_df, rtn_df, factor_cols=None, n_jobs=10, symbol=None):
    if factor_cols is None:
        factor_cols = factor_df.select_dtypes(include='number').columns.tolist()

    results = Parallel(n_jobs=n_jobs)(
        delayed(fast_corr)(*align_and_dropna(factor_df, rtn_df, col))
        for col in tqdm(factor_cols, leave=False)
    )
    
    return pd.Series(results, index=factor_cols, name=symbol)

# def fast_sharpe(x, y):
#     return (x * y).mean() / (x * y).std() * np.sqrt(len(x))

def fast_sharpe_standardized(x, y):
    """计算标准化后的Sharpe比率"""
    # 标准化
    x_std = (x - np.nanmean(x)) / np.nanstd(x)
    y_std = (y - np.nanmean(y)) / np.nanstd(y)
    
    # 乘积
    z = x_std * y_std
    
    # 计算夏普比率
    mean_z = np.nanmean(z)
    std_z = np.nanstd(z)
    n = np.sum(~np.isnan(z))
    
    if std_z == 0 or n <= 1:
        return 0
    
    return mean_z / std_z * np.sqrt(n)

def parallel_calc_sharpe_optimized(factor_df, rtn_df, factor_cols=None, n_jobs=10, symbol=None):
    if factor_cols is None:
        factor_cols = factor_df.select_dtypes(include='number').columns.tolist()

    results = Parallel(n_jobs=n_jobs)(
        delayed(fast_sharpe_standardized)(*align_and_dropna(factor_df, rtn_df, col))
        for col in tqdm(factor_cols, leave=False)
    )
    
    return pd.Series(results, index=factor_cols, name=symbol)

def analyze_factors_wide(wide_df):
    wide_df = wide_df.replace([None], np.nan)
    nan_rates = wide_df.isna().mean()
    
    def calc_rolling_range(s):
        s_clean = s.dropna()
        if len(s_clean) < 60 * 4:  # 确保有足够数据计算滚动窗口
            return np.nan
        
        rolling_range = s_clean.rolling(60 * 4).max() - s_clean.rolling(60 * 4).min()
        valid_range = rolling_range[
            (rolling_range >= rolling_range.quantile(0.05)) & 
            (rolling_range <= rolling_range.quantile(0.95))
        ]
        return valid_range.mean() if not valid_range.empty else np.nan
    
    rolling_means = wide_df.apply(calc_rolling_range)
    abs_mean = wide_df.abs().mean()
    rolling_means = rolling_means / abs_mean.replace(0, np.nan)  # 避免除零
    
    def calc_mode_rate(s):
        s_clean = s.dropna()
        if len(s_clean) == 0:
            return np.nan
        modes = s_clean.mode()
        return (s_clean == modes.iloc[0]).mean() if not modes.empty else np.nan
    
    mode_rates = wide_df.apply(calc_mode_rate)
    
    # 构建结果DataFrame
    result_df = pd.DataFrame({
        'nan_rate': nan_rates,
        'mean_result': rolling_means,
        'mode_rate': mode_rates
    })
    result_df.loc[wide_df.count() < 100, ['mean_result', 'mode_rate']] = np.nan
    
    return result_df

def main():
    for train_end_date in tqdm([
        # '2025-01-01',
        '2025-01-01',
        # '2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01', '2023-06-01', '2023-07-01', '2023-08-01', '2023-09-01', '2023-10-01', '2023-11-01', '2023-12-01',
        # '2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01', '2024-05-01', '2024-06-01', '2024-07-01', '2024-08-01', '2024-09-01', '2024-10-01', '2024-11-01', '2024-12-01',
        # '2025-01-01', '2025-02-01', '2025-03-01', '2025-04-01', '2025-05-01', '2025-06-01', '2025-07-01', '2025-08-01', '2025-09-01', '2025-10-01'
        ]):
        symbol = 'AG'
        fac_df = pd.read_feather(f'/mnt/Data/writable/liaoyuyang/factor/{symbol}/all_fac/all_factor.feather').set_index('datetime')
        fac_df = fac_df.loc[fac_df.index < train_end_date]
        factor_col = [x for x in fac_df.columns if x not in ['datetime', 'instrument']]
        fac_df = fac_df[factor_col]

        rtn_df = pd.read_csv(f'/mnt/Data/writable/liaoyuyang/data/1min/active/main_{symbol}.csv', index_col=0, parse_dates=['ts']).set_index('ts').reindex(index=fac_df.index)
        ic_results = parallel_calc_ic_optimized(fac_df, rtn_df, n_jobs=1)
        sharpe_results = parallel_calc_sharpe_optimized(fac_df, rtn_df, n_jobs=1)
        result_df = analyze_factors_wide(fac_df)

        os.makedirs(f'/mnt/Data/writable/liaoyuyang/factor_eval_commodity/{train_end_date}', exist_ok=True)
        result_df.join(pd.concat([ic_results.rename('ic'), sharpe_results.rename('sharpe')], axis=1)).to_csv(f'/mnt/Data/writable/liaoyuyang/factor_eval_commodity/{train_end_date}/{symbol}_single_factor_eval_5.csv')

if __name__ == "__main__":
    main()