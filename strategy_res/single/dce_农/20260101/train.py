#!/usr/bin/env python3
"""
DCE 农产品单日期训练脚本
========================
顺序训练指定品种 × 两个版本(v0/v1)，不并行。
预训练文件若已存在则自动跳过。
"""

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.append("/home/future_commodity")
sys.path.append("/home/strategy_res/single/dce_农")

import numpy as np
np.seterr(all="ignore")

import pipeline as pl
import function_future.DataLoader as DL

# =====================================================================
# 只改这里
# =====================================================================
TRAIN_END_DATE = "2026-01-01"          # 如 2026-01-01
SYMBOLS = ["A", "B", "C", "CS", "M", "Y", "P", "LH"]
TRAIN_LABEL = 5
VERSIONS = ["v0", "v1"]            # 两个版本都训练

# 因子筛选配置
PIPELINE_CONFIG = {
    "info_select": {"nan_rate": 0.8, "mode_rate": 0.9},
    "importance_select_by_group": {"cut_num_1": 300, "cut_num_2": 200, "same_name_cut": 5},
    "sp_select": {"th": 0},
    "day_cut": {"num_limit": 5},
}

# LightGBM 基础超参（v0/v1 共用）
LGB_BASE = {
    "learning_rate": 0.005,
    "num_leaves": 32,
    "max_depth": 5,
    "min_data_in_leaf": 500,
    "lambda_l1": 1,
    "lambda_l2": 1,
    "feature_fraction": 0.7,
    "bagging_freq": 10,
    "extra_trees": True,
    "max_bin": 32,
    "verbose": -1,
    "seed": 142,
    "num_threads": 20,
    "deterministic": True,
}

# v0 = MSE
LGB_V0 = {**LGB_BASE, "objective": "regression", "metric": "rmse"}
# v1 = Huber
LGB_V1 = {**LGB_BASE, "objective": "huber", "metric": "l1", "alpha": 0.9}

# 是否先删除旧预训练文件（强制重新生成）
CLEAR_PRETRAIN = False

# 是否跳过已存在模型的品种/版本
SKIP_EXISTING_MODEL = False
# =====================================================================


def train_single(symbol: str, version: str):
    folder_name = f"{symbol}_pred{TRAIN_LABEL}_{TRAIN_END_DATE}_{version}"
    model_base_dir = pl.get_model_base_dir(TRAIN_END_DATE)

    print(f"\n{'='*60}")
    print(f"开始训练: {symbol} | {TRAIN_END_DATE} | {version} | {folder_name}")
    print(f"{'='*60}")
    t0 = time.time()

    # ---- 1. 检查是否已存在模型 ----
    model_folder = model_base_dir / folder_name
    if SKIP_EXISTING_MODEL and model_folder.exists() and any(model_folder.iterdir()):
        print(f"  [跳过] 模型已存在: {model_folder}")
        return

    # ---- 2. 清理预训练文件（可选，只清一次，v0 和 v1 共用预训练文件） ----
    if CLEAR_PRETRAIN and version == VERSIONS[0]:
        pl.clear_pretrain_files(symbol, TRAIN_END_DATE, TRAIN_LABEL)

    # ---- 3. 加载数据 ----
    config_loader = DL.InstrumentConfig()
    config_loader.get_instrument_config(symbol)
    fac_df = pl.load_factor_data(symbol, TRAIN_END_DATE, TRAIN_LABEL, config_loader)

    # ---- 4. 因子筛选 ----
    factor_col = [c for c in fac_df.columns if c not in ["datetime", "instrument"]]
    factor_filter, summary, stability_df, cat_df = pl.run_factor_filter(
        symbol, fac_df, TRAIN_END_DATE, TRAIN_LABEL, factor_col, PIPELINE_CONFIG
    )
    print(f"  筛选后因子数: {len(factor_filter.factor_to_choose)}")

    # ---- 5. 预训练 ----
    if version == VERSIONS[0]:
        pl.run_pretrain(symbol, fac_df, TRAIN_END_DATE, TRAIN_LABEL)

    # ---- 6. 模型训练 ----
    lgb_params = LGB_V0 if version == "v0" else LGB_V1
    trainer, metrics_df = pl.train_model(
        symbol=symbol,
        factor_col=factor_filter.factor_to_choose,
        train_end_date=TRAIN_END_DATE,
        config_loader=config_loader,
        train_label=TRAIN_LABEL,
        folder_name=folder_name,
        label_transform=None,
        lgb_params=lgb_params,
        model_base_dir=model_base_dir,
    )
    print(f"  训练指标:\n{metrics_df.to_string(index=False)}")

    elapsed = time.time() - t0
    print(f"  ✓ {symbol} {version} 完成，耗时 {elapsed:.1f}s")


def main():
    total = len(SYMBOLS) * len(VERSIONS)
    start_all = time.time()
    idx = 0

    for symbol in SYMBOLS:
        for version in VERSIONS:
            idx += 1
            print(f"\n[{idx}/{total}] {symbol} | {version}")
            try:
                train_single(symbol, version)
            except Exception as e:
                print(f"  ✗ {symbol} {version} 失败: {e}")
                import traceback
                traceback.print_exc()

    total_elapsed = time.time() - start_all
    print(f"\n{'='*60}")
    print(f"全部完成: {total} 组，总耗时 {total_elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
