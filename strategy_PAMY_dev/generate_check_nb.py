import nbformat as nbf

nb = nbf.v4.new_notebook()

nb.cells.append(nbf.v4.new_markdown_cell(r"""\
# History Factor + Market Data Check Notebook — 0317~0323

**Note**: Night session (21:00-23:00) belongs to next trade_date. Filter by `trade_date`, not calendar date.

**Research env**:
- Factor: `/mnt/Data/writable/liaoyuyang/factor/{SYMBOL}/all_fac/all_factor.feather`
- 1min: `/mnt/Data/writable/liaoyuyang/data/1min/{SYMBOL}/{contract}.feather`
- tick: `/mnt/Data/writable/liaoyuyang/data/level2_all/{SYMBOL}/{contract}.feather`

**Realtime env**:
- `files/{contract}_tick.csv` / `{contract}_min.csv` (from `calc_recent_data.py`)

**Key output variables**:
- `fac_rt` / `fac_rs` — realtime/research factor tables
- `mkt_rt` / `mkt_rs` — realtime/research 1min market data
- `mkt_diff` — market data diff
- `summary` — factor diff summary
"""))

nb.cells.append(nbf.v4.new_code_cell(r"""\
# ========== Config ==========
SYMBOL = "P"
CONTRACT = "p2605"
DATES = ["2026-03-17", "2026-03-18", "2026-03-19", "2026-03-20", "2026-03-23"]

FILES_ROOT = "/home/strategy_PAMY_dev/files"
MODEL_ROOT = "/home/strategy_PAMY_dev/models"
RESEARCH_FEATHER = f"/mnt/Data/writable/liaoyuyang/factor/{SYMBOL}/all_fac/all_factor.feather"
RESEARCH_1MIN = f"/mnt/Data/writable/liaoyuyang/data/1min/{SYMBOL}/{CONTRACT}.feather"
RESEARCH_TICK = f"/mnt/Data/writable/liaoyuyang/data/level2_all/{SYMBOL}/{CONTRACT}.feather"

TOL_ABS = 0.01
TOL_REL = 0.05

print(f"Symbol: {SYMBOL} | Contract: {CONTRACT}")
print(f"Trade dates: {DATES}")
"""))

nb.cells.append(nbf.v4.new_code_cell(r"""\
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/strategy_PAMY_dev")

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt

from data_function import load_fac_df_old, load_config
from strategies import (
    generate_factor_dataframe_A, generate_factor_dataframe_B,
    generate_factor_dataframe_C, generate_factor_dataframe_CS,
    generate_factor_dataframe_M, generate_factor_dataframe_P,
    generate_factor_dataframe_Y, generate_factor_dataframe_LH,
)

GENERATE_FUNC_MAP = {
    "A": generate_factor_dataframe_A, "B": generate_factor_dataframe_B,
    "C": generate_factor_dataframe_C, "CS": generate_factor_dataframe_CS,
    "M": generate_factor_dataframe_M, "Y": generate_factor_dataframe_Y,
    "P": generate_factor_dataframe_P, "LH": generate_factor_dataframe_LH,
}

CONTRACT_MAP = {
    "A": "a2605", "B": "b2605", "C": "c2605", "CS": "cs2605",
    "M": "m2605", "Y": "y2605", "P": "p2605", "LH": "lh2605",
}

OTHER_SYMBOLS_MAP = {
    "A": ["B", "C", "CS", "M", "Y", "P", "LH"],
    "B": ["A", "C", "CS", "M", "Y", "P", "LH"],
    "C": ["A", "B", "CS", "M", "Y", "P", "LH"],
    "CS": ["A", "B", "C", "M", "Y", "P", "LH"],
    "M": ["A", "B", "C", "CS", "Y", "P", "LH"],
    "Y": ["A", "B", "C", "CS", "M", "P", "LH"],
    "P": ["A", "B", "C", "CS", "M", "Y", "LH"],
    "LH": ["A", "B", "C", "CS", "M", "Y", "P"],
}

target_dates = [pd.to_datetime(d).date() for d in DATES]
"""))

# ========== Market Data Check ==========

nb.cells.append(nbf.v4.new_code_cell(r"""\
# ========== Load Research 1min (filter by trade_date) ==========
mkt_rs = pd.read_feather(RESEARCH_1MIN)
mkt_rs['datetime'] = pd.to_datetime(mkt_rs['ts'])

if hasattr(mkt_rs['trade_date'], 'dt'):
    mkt_rs['trade_date_dt'] = mkt_rs['trade_date'].dt.date
else:
    mkt_rs['trade_date_dt'] = pd.to_datetime(mkt_rs['trade_date']).dt.date

mkt_rs = mkt_rs[mkt_rs['trade_date_dt'].isin(target_dates)].copy()
mkt_rs = mkt_rs.set_index('datetime')
print(f"Research 1min | shape={mkt_rs.shape}")

# ========== Load Realtime 1min (filter by trade_date) ==========
mkt_rt = pd.read_csv(f"{FILES_ROOT}/{CONTRACT}_min.csv", parse_dates=['datetime'])

if hasattr(mkt_rt['trade_date'], 'dt'):
    mkt_rt['trade_date_dt'] = mkt_rt['trade_date'].dt.date
else:
    mkt_rt['trade_date_dt'] = pd.to_datetime(mkt_rt['trade_date']).dt.date

mkt_rt = mkt_rt[mkt_rt['trade_date_dt'].isin(target_dates)].copy()
mkt_rt = mkt_rt.set_index('datetime')
print(f"Realtime 1min | shape={mkt_rt.shape}")
"""))

nb.cells.append(nbf.v4.new_code_cell(r"""\
# Align by datetime
mkt_idx = mkt_rt.index.intersection(mkt_rs.index)
print(f"Common index: {len(mkt_idx)}")
print(f"Missing in research: {len(mkt_rt.index.difference(mkt_rs.index))}")
print(f"Missing in realtime: {len(mkt_rs.index.difference(mkt_rt.index))}")

if len(mkt_rt.index.difference(mkt_rs.index)) > 0:
    print("Realtime has but research missing:")
    print(sorted(mkt_rt.index.difference(mkt_rs.index))[:10])
if len(mkt_rs.index.difference(mkt_rt.index)) > 0:
    print("Research has but realtime missing:")
    print(sorted(mkt_rs.index.difference(mkt_rt.index))[:10])

price_cols = ['open', 'high', 'low', 'close', 'volume', 'turnover', 'open_interest']
mkt_rt_aligned = mkt_rt.loc[mkt_idx, price_cols]
mkt_rs_aligned = mkt_rs.loc[mkt_idx, price_cols]

mkt_diff = (mkt_rs_aligned - mkt_rt_aligned).abs()
for c in price_cols:
    print(f"{c}: max_diff={mkt_diff[c].max():.6f}")
"""))

nb.cells.append(nbf.v4.new_code_cell(r"""\
# Row-level diff
mkt_diff_full = (mkt_rs_aligned.fillna(142857) - mkt_rt_aligned.fillna(142857)).abs()

print("=== Volume diff Top10 ===")
print(mkt_diff_full.nlargest(10, 'volume')[['volume', 'turnover']])

print()
print("=== Turnover diff Top10 ===")
print(mkt_diff_full.nlargest(10, 'turnover')[['volume', 'turnover']])
"""))

nb.cells.append(nbf.v4.new_code_cell(r"""\
# Research market table
mkt_rs_aligned.tail(8).T.sort_index()
"""))

nb.cells.append(nbf.v4.new_code_cell(r"""\
# Realtime market table
mkt_rt_aligned.tail(8).T.sort_index()
"""))

nb.cells.append(nbf.v4.new_code_cell(r"""\
# Market diff table (fillna 142857 trick)
(mkt_rt_aligned.tail(8).fillna(142857) - mkt_rs_aligned.reindex_like(mkt_rt_aligned.tail(8)).fillna(142857)).T.round(4).sort_index()
"""))

nb.cells.append(nbf.v4.new_code_cell(r"""\
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

axes[0].plot(mkt_idx, mkt_rs_aligned['close'], label='research', alpha=0.8)
axes[0].plot(mkt_idx, mkt_rt_aligned['close'], label='realtime', alpha=0.8)
axes[0].set_title('Close Price Compare')
axes[0].legend()

axes[1].plot(mkt_idx, mkt_rs_aligned['volume'], label='research', alpha=0.8)
axes[1].plot(mkt_idx, mkt_rt_aligned['volume'], label='realtime', alpha=0.8)
axes[1].set_title('Volume Compare')
axes[1].legend()

axes[2].plot(mkt_idx, mkt_diff['volume'], color='red', alpha=0.6)
axes[2].set_title('Volume Abs Diff')

axes[3].plot(mkt_idx, mkt_diff['turnover'], color='red', alpha=0.6)
axes[3].set_title('Turnover Abs Diff')

plt.tight_layout()
plt.show()
"""))

nb.cells.append(nbf.v4.new_code_cell(r"""\
# Drill down to tick level at a specific minute
VIEW_TIME_MKT = "2026-03-23 11:30:00"

# Research tick
tick_rs = pd.read_feather(RESEARCH_TICK)
tick_rs['datetime'] = pd.to_datetime(tick_rs['datetime'])
rs_sub = tick_rs[tick_rs['datetime'] == VIEW_TIME_MKT][['datetime','volume','turnover','last_price','open_interest']].copy()
print(f"Research tick @ {VIEW_TIME_MKT} | {len(rs_sub)} rows | volume_sum={rs_sub['volume'].sum():.0f}")

# Realtime tick
tick_rt = pd.read_csv(f"{FILES_ROOT}/{CONTRACT}_tick.csv", parse_dates=['datetime'])
rt_sub = tick_rt[tick_rt['datetime'] == VIEW_TIME_MKT][['datetime','volume','turnover','last_price','open_interest']].copy()
print(f"Realtime tick @ {VIEW_TIME_MKT} | {len(rt_sub)} rows | volume_sum={rt_sub['volume'].sum():.0f}")

print("Research tick tail:")
print(rs_sub.tail(5).to_string())
print("Realtime tick tail:")
print(rt_sub.tail(5).to_string())
"""))

# ========== Factor Check ==========

nb.cells.append(nbf.v4.new_code_cell(r"""\
# Load research factors (filter by trade_date)
print("Loading research factors...")
meta = pd.read_feather(RESEARCH_FEATHER, columns=["datetime", "instrument"])
meta["datetime"] = pd.to_datetime(meta["datetime"])

meta["trade_date_dt"] = meta["datetime"].dt.date
meta_target = meta[meta["trade_date_dt"].isin(target_dates)]
inst = meta_target["instrument"].iloc[0]
print(f"Research contract: {inst}")

import pyarrow.feather as feather
schema = feather.read_table(RESEARCH_FEATHER).schema
rs_cols = set(str(c) for c in schema.names)

fac_rs_all = pd.read_feather(RESEARCH_FEATHER)
fac_rs_all["datetime"] = pd.to_datetime(fac_rs_all["datetime"])
fac_rs_all["trade_date_dt"] = fac_rs_all["datetime"].dt.date
fac_rs = fac_rs_all[
    (fac_rs_all["instrument"] == inst) &
    (fac_rs_all["trade_date_dt"].isin(target_dates))
].copy()
del fac_rs_all
fac_rs = fac_rs.set_index("datetime")
print(f"Research factors | shape={fac_rs.shape}")
"""))

nb.cells.append(nbf.v4.new_code_cell(r"""\
# Compute realtime factors
print("Computing realtime factors, please wait...")

cfg = load_config()
trade_hours = cfg["symbol_specs"][SYMBOL]["trade_hours"]

model_path = f"{MODEL_ROOT}/{SYMBOL}"
model = lgb.Booster(model_file=f"{model_path}/kfold_fold1_0.lgb")
factor_col = model.feature_name()
print(f"Model features: {len(factor_col)}")

main_contract = CONTRACT_MAP[SYMBOL]
other_symbols = OTHER_SYMBOLS_MAP[SYMBOL]
instrument_list = [main_contract] + [CONTRACT_MAP[s] for s in other_symbols]
dict_keys = [SYMBOL] + other_symbols

fac_rt = load_fac_df_old(
    factor_col=factor_col,
    instrument_list=instrument_list,
    recent_data_path=FILES_ROOT,
    trade_type=trade_hours,
    dict_keys=dict_keys,
    generate_factor_dataframe=GENERATE_FUNC_MAP[SYMBOL],
)

print(f"Realtime factors done | shape={fac_rt.shape}")
fac_rt = fac_rt[fac_rt.index.normalize().isin(pd.to_datetime(DATES).date)].copy()
print(f"After filter: {fac_rt.shape}")
fac_rt.head()
"""))

nb.cells.append(nbf.v4.new_code_cell(r"""\
drop_cols = {"datetime", "instrument", "hour"}
rt_cols = set(fac_rt.columns) - drop_cols
rs_cols_fac = set(fac_rs.columns) - drop_cols
common_factors = sorted(rt_cols & rs_cols_fac)
print(f"Common factors: {len(common_factors)}")

fac_idx = fac_rt.index.intersection(fac_rs.index)
print(f"Common index: {len(fac_idx)}")

rt = fac_rt.loc[fac_idx, common_factors]
rs = fac_rs.loc[fac_idx, common_factors]

print("Core vars ready: rt (realtime), rs (research)")
"""))

nb.cells.append(nbf.v4.new_code_cell(r"""\
# Research factor table
rs.tail(8).T.sort_index()
"""))

nb.cells.append(nbf.v4.new_code_cell(r"""\
# Realtime factor table
rt.tail(8).T.sort_index()
"""))

nb.cells.append(nbf.v4.new_code_cell(r"""\
# Factor diff table (fillna 142857 trick)
(rt.tail(8).fillna(142857) - rs.reindex_like(rt.tail(8)).fillna(142857)).T.round(4).sort_index()
"""))

nb.cells.append(nbf.v4.new_code_cell(r"""\
diff = (rs - rt).abs()
mask_na = rs.isna() | rt.isna()
diff = diff.where(~mask_na, np.nan)

with np.errstate(divide="ignore", invalid="ignore"):
    denom = rs.abs().replace(0, np.nan)
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
    "max_diff": diff.max(axis=0),
}).sort_values("ratio", ascending=False)

fully_bad = summary[summary["ratio"] == 1.0]
perfect = summary[summary["ratio"] == 0.0]

print(f"Perfect: {len(perfect)} | Fully bad: {len(fully_bad)} | Partial: {len(summary)-len(perfect)-len(fully_bad)}")
print()
print("Fully bad Top10:")
print(fully_bad.head(10)[["max_diff"]])
print()
print("Highest diff Top10:")
print(summary.head(10))
"""))

nb.cells.append(nbf.v4.new_code_cell(r"""\
VIEW_FACTOR = "day_jump"

fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

if VIEW_FACTOR in rt.columns:
    axes[0].plot(fac_idx, rt[VIEW_FACTOR], label="realtime", alpha=0.8)
    axes[0].plot(fac_idx, rs[VIEW_FACTOR], label="research", alpha=0.8)
    axes[0].set_title(f"{VIEW_FACTOR} — Time Series Compare")
    axes[0].legend()
    
    axes[1].plot(fac_idx, diff[VIEW_FACTOR], color="red", alpha=0.6)
    axes[1].axhline(TOL_ABS, color="green", linestyle="--", label=f"abs_tol={TOL_ABS}")
    axes[1].set_title(f"{VIEW_FACTOR} — Abs Diff")
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
else:
    print(f"{VIEW_FACTOR} not in common factors")
"""))

nb.cells.append(nbf.v4.new_code_cell(r"""\
VIEW_TIME = "2026-03-17 09:30:00"
contract = CONTRACT_MAP[SYMBOL]
tick_df = pd.read_csv(f"{FILES_ROOT}/{contract}_tick.csv", parse_dates=["datetime"])

target = pd.to_datetime(VIEW_TIME)
mask = (tick_df["datetime"] >= target - pd.Timedelta("1min")) & (tick_df["datetime"] <= target + pd.Timedelta("1min"))
print(f"Tick data around {VIEW_TIME} | {mask.sum()} rows")
tick_df[mask].head(20)
"""))

nb.cells.append(nbf.v4.new_code_cell(r"""\
summary.to_csv(f"/home/strategy_PAMY_dev/factor_diff_summary_{SYMBOL}.csv")
print("Summary CSV saved")
"""))

with open("/home/strategy_PAMY_dev/check_history_factor.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook generated")
