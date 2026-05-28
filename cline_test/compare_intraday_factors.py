"""
盘中因子 vs 研究环境因子对比
- 盘中因子: save_files/A/factors/factors_*.csv (拼接所有分钟)
- 研究环境: /mnt/Data/writable/liaoyuyang/factor/{SYMBOL}/all_fac/all_factor.feather
"""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/home/strategy_PAMY_dev')

from data_function import load_config
import lightgbm as lgb
from pathlib import Path

# ========== Config ==========
SYMBOL = "A"
MODEL_ROOT = "/home/strategy_PAMY_dev/models"
RESEARCH_FEATHER = f"/mnt/Data/writable/liaoyuyang/factor/{SYMBOL}/all_fac/all_factor.feather"
FILES_ROOT = "/home/strategy_PAMY_dev/files"

TOL_ABS = 0.01
TOL_REL = 0.05
TAIL_N = 50

# ========== 1. 读取盘中因子 (拼接所有文件) ==========
rt_fac_dir = Path("/home/strategy_PAMY_dev/save_files/A/factors")
rt_files = sorted(rt_fac_dir.glob("factors_*.csv"))
rt_df = pd.concat([pd.read_csv(f, parse_dates=['datetime']) for f in rt_files], ignore_index=True)
rt_df = rt_df.sort_values('datetime').reset_index(drop=True)
print(f"[盘中因子] {len(rt_df)} 行, 时间范围: {rt_df['datetime'].min()} ~ {rt_df['datetime'].max()}")

# ========== 2. 读取研究环境因子 ==========
print("\n加载研究环境因子...")
rs_all = pd.read_feather(RESEARCH_FEATHER)
rs_all['datetime'] = pd.to_datetime(rs_all['datetime'])
rs_all['trade_date_dt'] = rs_all['datetime'].dt.date

# 获取盘中因子对应的交易日
# 夜盘 (21:00+) 属于下个交易日
rt_df['trade_date_dt'] = rt_df['datetime'].apply(
    lambda dt: (dt + pd.Timedelta(days=1)).date() if dt.hour >= 20 else dt.date()
)
target_trade_dates = rt_df['trade_date_dt'].unique()
print(f"[交易日] 盘中因子对应交易日: {target_trade_dates}")

# 获取品种
inst = rs_all[rs_all['trade_date_dt'].isin(target_trade_dates)]['instrument'].iloc[0]
print(f"[品种] {inst}")

# 筛选研究环境因子
rs_df = rs_all[
    (rs_all['instrument'] == inst) &
    (rs_all['trade_date_dt'].isin(target_trade_dates))
].copy()
rs_df = rs_df.set_index('datetime')
print(f"[研究因子] {len(rs_df)} 行")

# ========== 3. 时间对齐 ==========
# 研究环境因子 index 是 datetime
# 盘中因子用相同的 datetime 对齐
fac_idx = rt_df.set_index('datetime').index.intersection(rs_df.index)
print(f"\n[对齐] 共同时间点: {len(fac_idx)}")
print(f"  盘中因子缺失: {len(rs_df.index) - len(fac_idx)}")
print(f"  研究环境缺失: {len(rt_df.set_index('datetime').index) - len(fac_idx)}")

if len(fac_idx) == 0:
    print("\n没有对齐的时间点！检查日期映射...")
    print("盘中因子最后时间:", rt_df['datetime'].max())
    print("研究因子最后时间:", rs_df.index.max())
    print("研究因子 trade_date:", rs_df.index.date[:5] if hasattr(rs_df.index, 'date') else 'N/A')
    sys.exit(1)

# ========== 4. 因子对比 ==========
drop_cols = {"datetime", "instrument", "hour"}
fac_rt = rt_df.set_index('datetime')
rt_cols = set(fac_rt.columns) - drop_cols
rs_cols = set(rs_df.columns) - drop_cols
common_factors = sorted(rt_cols & rs_cols)
print(f"\n[因子] 共同因子数量: {len(common_factors)}")

# 对齐数据
rt_aligned = fac_rt.loc[fac_idx, common_factors]
rs_aligned = rs_df.loc[fac_idx, common_factors]

# 统计差异
diff = (rs_aligned - rt_aligned).abs()
mask_na = rs_aligned.isna() | rt_aligned.isna()
diff = diff.where(~mask_na, np.nan)

with np.errstate(divide="ignore", invalid="ignore"):
    denom = rs_aligned.abs().replace(0, np.nan)
    rel = diff / denom
    rel = rel.fillna(0)

ok = (diff < TOL_ABS) | (rel < TOL_REL)
ok = ok.where(~mask_na, np.nan)

fail = (ok == False).sum(axis=0)
total = ok.notna().sum(axis=0)
ratio = fail / total

summary = pd.DataFrame({
    "total": total,
    "fail": fail,
    "ratio": ratio,
    "median_diff": diff.median(axis=0),
    "p90_diff": diff.quantile(0.90, axis=0),
    "p95_diff": diff.quantile(0.95, axis=0),
}).sort_values("ratio", ascending=False)

fully_bad = summary[summary["ratio"] == 1.0]
perfect = summary[summary["ratio"] == 0.0]

print(f"\n[结果] Perfect: {len(perfect)} | Fully bad: {len(fully_bad)} | Partial: {len(summary)-len(perfect)-len(fully_bad)}")
print()
print("Fully bad Top10:")
print(fully_bad.head(10)[["median_diff"]])
print()
print("Highest fail-ratio Top10:")
print(summary.head(10).round(4))

# ========== 5. 对比最后 N 个 bar ==========
print(f"\n[详细对比] 最后 {TAIL_N} 个 bar")
rt_tail = rt_aligned.tail(TAIL_N)
rs_tail = rs_aligned.reindex_like(rt_tail)
diff_tail = (rt_tail.fillna(142857) - rs_tail.fillna(142857))
print("\n因子差异表 (fillna=142857 trick):")
print(diff_tail.T.round(4).sort_index())
