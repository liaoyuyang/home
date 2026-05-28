"""
Prediction Diff Check Script
=============================
只对比实盘环境 files/ 目录下的数据与 research feathers 数据生成的预测值
"""

import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/strategy_PAMY_dev")

import json
import pandas as pd
import numpy as np
import lightgbm as lgb

from data_function import load_fac_df_old, load_config
from strategies import (
    generate_factor_dataframe_A, generate_factor_dataframe_B,
    generate_factor_dataframe_C, generate_factor_dataframe_CS,
    generate_factor_dataframe_M, generate_factor_dataframe_P,
    generate_factor_dataframe_Y, generate_factor_dataframe_LH,
)

# ========== pd.set_options ==========
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 300)
pd.set_option('display.float_format', '{:.6f}'.format)

# ========== Config ==========
SYMBOL = "A"
CONTRACT = "a2605"
DATES = ["2026-03-17", "2026-03-18", "2026-03-19", "2026-03-20", "2026-03-23"]

FILES_ROOT = "/home/strategy_PAMY_dev/files"
MODEL_ROOT = "/home/strategy_PAMY_dev/models"
RESEARCH_FEATHER = f"/mnt/Data/writable/liaoyuyang/factor/{SYMBOL}/all_fac/all_factor.feather"

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

print(f"Symbol: {SYMBOL} | Contract: {CONTRACT}")
print(f"Trade dates: {DATES}")

target_dates = [pd.to_datetime(d).date() for d in DATES]

# ========== Load Research Factors ==========
print("\n=== Loading research factors ===")
import pyarrow.feather as feather
schema = feather.read_table(RESEARCH_FEATHER).schema
rs_cols = set(str(c) for c in schema.names)

fac_rs_all = pd.read_feather(RESEARCH_FEATHER)
fac_rs_all["datetime"] = pd.to_datetime(fac_rs_all["datetime"])
fac_rs_all["trade_date_dt"] = fac_rs_all["datetime"].dt.date

meta = fac_rs_all[["datetime", "instrument"]].copy()
meta["trade_date_dt"] = meta["datetime"].dt.date
meta_target = meta[meta["trade_date_dt"].isin(target_dates)]
inst = meta_target["instrument"].iloc[0]
print(f"Research contract: {inst}")

fac_rs = fac_rs_all[
    (fac_rs_all["instrument"] == inst) &
    (fac_rs_all["trade_date_dt"].isin(target_dates))
].copy()
del fac_rs_all
fac_rs = fac_rs.set_index("datetime")
print(f"Research factors | shape={fac_rs.shape}")

# ========== Compute Realtime Factors ==========
print("\n=== Computing realtime factors ===")

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

# ========== Prediction Check ==========
print("\n=== Prediction Check ===")

# Load all 5 models with log(best_iteration+1) weights
model_lst = []
weight_lst = []
for i in range(1, 6):
    model_file = f"{MODEL_ROOT}/{SYMBOL}/kfold_fold{i}_0.lgb"
    meta_file = f"{MODEL_ROOT}/{SYMBOL}/kfold_fold{i}_0_meta.json"
    m = lgb.Booster(model_file=model_file)
    with open(meta_file, 'r') as f:
        meta_data = json.load(f)
    model_lst.append(m)
    weight_lst.append(float(np.log(meta_data['best_iteration'] + 1)))

print(f"Loaded {len(model_lst)} models | weights: {[round(w, 3) for w in weight_lst]}")

# Align input on common datetime index
pred_idx = fac_rt.index.intersection(fac_rs.index)
print(f"Common index: {len(pred_idx)}")

# Realtime input
rt_input = fac_rt.loc[pred_idx, factor_col]

# Research input
rs_input = fac_rs.loc[pred_idx, [c for c in factor_col if c in fac_rs.columns]].copy()

if 'hour' not in rs_input.columns:
    rs_input['hour'] = rs_input.index.hour

rs_input = rs_input[factor_col]

print(f"Realtime  input shape: {rt_input.shape}")
print(f"Research  input shape: {rs_input.shape}")

# Realtime prediction
pred_rt = pd.DataFrame(
    [m.predict(rt_input) for m in model_lst],
    columns=rt_input.index,
    index=[f'model_{i+1}' for i in range(len(model_lst))]
).T
pred_rt['weighted'] = pred_rt.mul(weight_lst, axis=1).sum(axis=1) / sum(weight_lst)
pred_rt['weighted_s'] = (
    pred_rt['weighted'] * 0.6
    + pred_rt['weighted'].shift(1) * 0.3
    + pred_rt['weighted'].shift(2) * 0.1
)

# Research prediction
pred_rs = pd.DataFrame(
    [m.predict(rs_input) for m in model_lst],
    columns=rs_input.index,
    index=[f'model_{i+1}' for i in range(len(model_lst))]
).T
pred_rs['weighted'] = pred_rs.mul(weight_lst, axis=1).sum(axis=1) / sum(weight_lst)
pred_rs['weighted_s'] = (
    pred_rs['weighted'] * 0.6
    + pred_rs['weighted'].shift(1) * 0.3
    + pred_rs['weighted'].shift(2) * 0.1
)

# Diff stats
diff_w = (pred_rt['weighted'] - pred_rs['weighted']).abs()
diff_ws = (pred_rt['weighted_s'] - pred_rs['weighted_s']).abs()

print(f"\nweighted   max_diff={diff_w.max():.4f}  mean_diff={diff_w.mean():.4f}")
print(f"weighted_s max_diff={diff_ws.max():.4f}  mean_diff={diff_ws.mean():.4f}")

print("\n=== Top 10 weighted diff ===")
print(diff_w.nlargest(10).round(4).to_string())

print("\n=== Top 10 weighted_s diff ===")
print(diff_ws.nlargest(10).round(4).to_string())

# Full comparison table
cmp = pd.DataFrame({
    'rt_w': pred_rt['weighted'],
    'rs_w': pred_rs['weighted'],
    'diff_w': diff_w,
    'rt_ws': pred_rt['weighted_s'],
    'rs_ws': pred_rs['weighted_s'],
    'diff_ws': diff_ws,
}).tail(100)

print(f"\n=== Full comparison ({len(cmp)} rows) ===")
print(cmp.round(6).to_string())

print("\n=== Done ===")