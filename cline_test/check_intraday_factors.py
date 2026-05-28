"""
检查因子文件内部一致性 - 看因子值是否随时间正确变化
"""
import pandas as pd
import numpy as np
from pathlib import Path

symbol = "A"
rt_prefix = "/home/strategy_PAMY_dev/save_files"

# 读取所有盘中因子文件
rt_dir = Path(rt_prefix) / symbol / "factors"
rt_files = sorted(rt_dir.glob("factors_*.csv"))
print(f"找到 {len(rt_files)} 个因子文件")

dfs = []
for f in rt_files[-10:]:  # 最近10个
    df = pd.read_csv(f, parse_dates=['datetime'])
    dfs.append(df)

combined = pd.concat(dfs).set_index('datetime')
print(f"\n时间范围: {combined.index[0]} ~ {combined.index[-1]}")
print(f"总行数: {len(combined)}")

# 检查几个关键因子在连续时间的值
key_factors = ['LR', 'RPP_22D', 'RPP_5D', 'bar5_trend_corr', 'hour']
print("\n=== 关键因子随时间变化 ===")
for fac in key_factors:
    if fac in combined.columns:
        vals = combined[fac].tail(5)
        print(f"\n{fac}:")
        for dt, v in vals.items():
            print(f"  {dt}: {v:.6f}")

# 检查 FAC_KDJ_biddommean 这个差异大的因子
print("\n\n=== 差异大的因子（如 FAC_KDJ_biddommean）===")
if 'FAC_KDJ_biddommean' in combined.columns:
    vals = combined['FAC_KDJ_biddommean'].tail(10)
    for dt, v in vals.items():
        print(f"  {dt}: {v:.6f}")
