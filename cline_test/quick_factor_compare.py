"""
快速因子对比 - 检查同一时间点盘中 vs 研究环境的因子值
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, '/home/strategy_PAMY_dev')

symbol = "A"
rt_prefix = "/home/strategy_PAMY_dev/save_files"
research_prefix = "/mnt/Data/writable/liaoyuyang/factor"

# 1. 读取盘中因子（最后一分钟）
rt_dir = Path(rt_prefix) / symbol / "factors"
rt_files = sorted(rt_dir.glob("factors_*.csv"))
rt_df = pd.read_csv(rt_files[-1], parse_dates=['datetime']).set_index('datetime')
print(f"盘中因子时间: {rt_df.index[-1]}")
print(f"盘中因子行数: {len(rt_df)}")

# 2. 读取研究环境因子（同一时间点附近）
print("\n正在读取研究环境因子...")
research_path = Path(research_prefix) / symbol / "all_fac" / "all_factor.feather"
meta = pd.read_feather(research_path, columns=['datetime', 'instrument'])
meta['datetime'] = pd.to_datetime(meta['datetime'])

# 找最新日期
latest_date = meta['datetime'].dt.date.max()
print(f"研究环境最新日期: {latest_date}")

# 读取该日期的因子
fac = pd.read_feather(research_path)
fac['datetime'] = pd.to_datetime(fac['datetime'])
fac_date = fac[fac['datetime'].dt.date == latest_date].copy().set_index('datetime')
print(f"研究环境因子行数: {len(fac_date)}")

# 3. 对齐时间
rt_time = rt_df.index[-1]
# 研究环境的时间应该是同一分钟的结束（如 22:15:00 -> 22:16:00）
rs_time = rt_time + pd.Timedelta(minutes=1)
print(f"\n盘中时间: {rt_time}")
print(f"研究环境匹配时间: {rs_time}")

# 4. 对比所有共有列
common_cols = list(set(rt_df.columns) & set(fac_date.columns))
print(f"\n共有因子数: {len(common_cols)}")

rt_row = rt_df.iloc[-1]
rs_row = fac_date.loc[rs_time] if rs_time in fac_date.index else None

if rs_row is None:
    # 尝试找最接近的时间
    closest = fac_date.index[np.abs(fac_date.index - rs_time).argmin()]
    print(f"⚠️ 精确时间不存在，使用最近时间: {closest}")
    rs_row = fac_date.loc[closest]

# 计算差异
print("\n=== 因子差异对比（绝对值排序 Top 30）===")
print(f"{'因子名':<35} {'盘中值':>15} {'研究值':>15} {'差异':>15}")
print("-" * 85)

diff_dict = []
for col in common_cols:
    rt_val = rt_row[col]
    rs_val = rs_row[col]
    
    if pd.isna(rt_val) and pd.isna(rs_val):
        continue
    
    diff = abs(rt_val - rs_val) if not (pd.isna(rt_val) or pd.isna(rs_val)) else np.nan
    diff_dict.append((col, rt_val, rs_val, diff))

diff_df = pd.DataFrame(diff_dict, columns=['因子', '盘中', '研究', '差异'])
diff_df = diff_df.dropna(subset=['差异'])
diff_df = diff_df.sort_values('差异', ascending=False)

for _, row in diff_df.head(30).iterrows():
    print(f"{row['因子']:<35} {row['盘中']:>15.6f} {row['研究']:>15.6f} {row['差异']:>15.6f}")

# 统计
print(f"\n=== 统计 ===")
print(f"总共有因子: {len(diff_df)}")
print(f"差异 < 0.01: {(diff_df['差异'] < 0.01).sum()}")
print(f"差异 0.01-0.1: {((diff_df['差异'] >= 0.01) & (diff_df['差异'] < 0.1)).sum()}")
print(f"差异 > 0.1: {(diff_df['差异'] >= 0.1).sum()}")
