"""
analyze_label_distribution.py
分析各品种收益率标签的分布特征，验证 Huber/MSE 效果分化的原因。
运行: python analyze_label_distribution.py
"""

import sys
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.append('/home/future_commodity')
import function_future.DataLoader as DL
import pipeline_v1 as pl


def analyze_symbol(symbol, train_end_date='2025-07-01', train_label=5):
    """加载单个品种的训练数据并分析标签分布（按实际训练流程剔除边界）"""
    config_loader = DL.InstrumentConfig()
    
    # 读取收益率数据
    rtn_df = pd.read_csv(
        f'/mnt/Data/writable/liaoyuyang/data/1min/active/main_{symbol}.csv',
        index_col=0, parse_dates=['ts']
    ).set_index('ts')
    
    col = f'rtn_{train_label}'
    if col not in rtn_df.columns:
        print(f"  警告: {symbol} 未找到列 {col}")
        return None
    
    # 取训练集
    y_series = rtn_df[col].loc[:pd.Timestamp(train_end_date)]
    
    # 按实际训练流程做 log 变换（和 train_model 一致）
    y_series = np.log1p(y_series)
    
    # 按实际训练流程剔除边界（cut=10）
    # 构造一个 dummy df 传给 df_cut_time
    dummy_df = pd.DataFrame({'dummy': 1}, index=y_series.index)
    dummy_df = config_loader.df_cut_time(
        dummy_df,
        config_loader.get_instrument_config(symbol)['trading_hours'],
        cut=10
    )
    
    # 只保留 cut 后的时间戳
    y = y_series.loc[y_series.index.isin(dummy_df.index)].values
    y = y[~np.isnan(y)]

    # 基础统计
    mean = np.mean(y)
    std = np.std(y)
    skew = stats.skew(y)
    kurt = stats.kurtosis(y)  # 超额峰度（正态=0）

    # 分位数（绝对值）
    q50, q90, q95, q99 = np.quantile(np.abs(y), [0.5, 0.9, 0.95, 0.99])

    # 极端值比例
    extreme_2sigma = np.mean(np.abs(y) > 2 * std)
    extreme_3sigma = np.mean(np.abs(y) > 3 * std)

    # 正态性检验
    jb_stat, jb_pvalue = stats.jarque_bera(y)

    return {
        'symbol': symbol,
        'n_samples': len(y),
        'mean': mean,
        'std': std,
        'skewness': skew,
        'kurtosis': kurt,
        'median_abs': q50,
        'q90_abs': q90,
        'q95_abs': q95,
        'q99_abs': q99,
        'extreme_2sigma_pct': extreme_2sigma,
        'extreme_3sigma_pct': extreme_3sigma,
        'jb_stat': jb_stat,
        'jb_pvalue': jb_pvalue,
    }


def main():
    symbols = ['A', 'B', 'C', 'CS', 'M', 'Y', 'P', 'LH']
    train_end_date = '2025-07-01'
    train_label = 5

    rows = []
    for sym in symbols:
        print(f"分析 {sym}...")
        row = analyze_symbol(sym, train_end_date, train_label)
        rows.append(row)

    df = pd.DataFrame(rows)

    # 排序：峰度从高到低（越厚尾越靠前）
    df = df.sort_values('kurtosis', ascending=False)

    print("\n" + "="*110)
    print("品种收益率标签分布特征对比")
    print("="*110)

    display_cols = [
        'symbol', 'n_samples', 'std', 'skewness', 'kurtosis',
        'median_abs', 'q90_abs', 'q95_abs', 'q99_abs',
        'extreme_2sigma_pct', 'extreme_3sigma_pct',
        'jb_pvalue'
    ]

    display_df = df[display_cols].copy()
    display_df.columns = [
        '品种', '样本数', '标准差', '偏度', '峰度',
        '|rtn|中位数', '|rtn|90%分位', '|rtn|95%分位', '|rtn|99%分位',
        '>2σ比例', '>3σ比例',
        'JB检验p值'
    ]

    pd.set_option('display.float_format', '{:.4f}'.format)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 220)
    print(display_df.to_string(index=False))

    print("\n" + "="*110)
    print("解读")
    print("="*110)
    print("""
【峰度 (kurtosis)】: 超额峰度，正态分布=0。
    > 3: 明显厚尾（极端值比正态多）
    0~3: 中等厚尾
    < 0: 比正态还"平"，极端值少

【>2σ / >3σ 比例】: 偏离均值 2/3 个标准差的样本占比。
    正态分布理论值: 2σ≈4.5%, 3σ≈0.3%
    实际值远高于理论值 → 厚尾特征明显

【JB检验p值】: p < 0.05 表示"显著非正态"。
""")

    print("\n按峰度排序（从高到低 = 从厚尾到薄尾）:")
    print(" -> ".join(df['symbol'].tolist()))

    print("\n按 >2σ 极端值比例排序（从高到低）:")
    df_by_extreme = df.sort_values('extreme_2sigma_pct', ascending=False)
    print(" -> ".join(df_by_extreme['symbol'].tolist()))

    print("\n按标准差排序（从高到低 = 波动率从高到低）:")
    df_by_std = df.sort_values('std', ascending=False)
    print(" -> ".join(df_by_std['symbol'].tolist()))

    return df


if __name__ == '__main__':
    df = main()
