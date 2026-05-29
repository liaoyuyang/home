#!/usr/bin/env python3
"""
DCE 农产品单日期回测脚本 — 多参数组对比
=========================================
加载本日期训练的模型，在指定回测窗口上跑回测，
支持多组开平仓阈值参数对比。
输出按月绩效宽表（CSV），同时保存静态图(PNG)。
结果存到 ./backtest/{suffix}/ 目录下。
"""

import sys
import warnings
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")
sys.path.append("/home/future_commodity")
sys.path.append("/home/strategy_res/single/dce_农")

import numpy as np
np.seterr(all="ignore")
import pandas as pd

import pipeline as pl
import function_future.backtest_v3 as bv
import function_future.DataLoader as DL

# =====================================================================
# 只改这里
# =====================================================================
TRAIN_END_DATE = "2025-04-01"          # 如 2026-01-01
SYMBOLS = ["A", "B", "C", "CS", "M", "Y", "P", "LH"]
TRAIN_LABEL = 5
VERSIONS = ["v0", "v1"]            # 两个版本都回测

# 回测窗口
BT_START = TRAIN_END_DATE          # 从训练截止日期开始
BT_END = "2026-05-07"

# 多组回测参数对比
PARAM_SETS = [
    {"th1": 0.9, "th2": 0.5, "suffix": "default"},
    {"th1": 0.8, "th2": 0.4, "suffix": "th08_04"},
]

# 各参数组共用的回测配置
BT_COMMON = {
    "holding_bars": 10,
    "holding_days": 5,
    "fee": 0,
    "v": 2,
}
# =====================================================================


def run_single_backtest(symbol: str, version: str, bt_params: dict):
    """单个品种回测，返回 merged_data"""
    folder_name = f"{symbol}_pred{TRAIN_LABEL}_{TRAIN_END_DATE}_{version}"
    model_base_dir = pl.get_model_base_dir(TRAIN_END_DATE)
    initial_capital = pl.calc_initial_capital(symbol)
    config_loader = DL.InstrumentConfig()
    inst_cfg = config_loader.get_instrument_config(symbol)
    bars_per_day = inst_cfg.get("bars_per_day", 345)

    params = bt_params.copy()
    params["day"] = bt_params["holding_days"] * bars_per_day

    print(f"  [{symbol}] 回测: {BT_START} ~ {BT_END}, money={initial_capital}")
    _bt, merged_data = pl.run_backtest(
        symbol, TRAIN_END_DATE, folder_name, params, initial_capital,
        window_end=BT_END, model_base_dir=model_base_dir
    )

    # 过滤回测窗口
    merged_data = merged_data[
        merged_data["datetime"] >= pd.Timestamp(BT_START)
    ].reset_index(drop=True)

    return merged_data


def calc_monthly_metrics(merged_data: pd.DataFrame) -> pd.DataFrame:
    """
    从 merged_data 提取按月绩效指标。
    返回 DataFrame: index=月份, columns=[品种相关的指标]
    """
    if merged_data.empty:
        return pd.DataFrame()

    md = merged_data.copy()
    md["datetime"] = pd.to_datetime(md["datetime"])
    md["month"] = md["datetime"].dt.to_period("M")

    # 日级汇总
    daily = md.groupby("date").agg({
        "pnl_ret": "sum",
        "pos": "mean",
    }).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["month"] = daily["date"].dt.to_period("M")

    monthly = daily.groupby("month").agg(
        monthly_ret=("pnl_ret", "sum"),
        sharpe=("pnl_ret", lambda x: x.mean() / x.std() * np.sqrt(len(x)) if x.std() > 0 else np.nan),
        win_rate=("pnl_ret", lambda x: (x > 0).mean()),
        avg_pos=("pos", "mean"),
        trade_days=("pnl_ret", "count"),
    )

    # 月内最大回撤（按日累积）
    def _monthly_dd(sub):
        cum = sub["pnl_ret"].cumsum()
        running_max = cum.cummax()
        dd = cum - running_max
        return dd.min()

    monthly["max_dd"] = daily.groupby("month").apply(_monthly_dd)
    return monthly


def run_param_set(param_set: dict):
    """跑一组参数的回测"""
    suffix = param_set.pop("suffix")
    output_dir = Path(f"./backtest/{suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)

    bt_params = BT_COMMON.copy()
    bt_params.update(param_set)

    print(f"\n{'='*60}")
    print(f"参数组: th1={bt_params['th1']}, th2={bt_params['th2']}, suffix={suffix}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*60}")

    for version in VERSIONS:
        print(f"\n--- 版本: {version} ---")

        # ---- 逐个品种回测 + 按月聚合 ----
        all_monthly = {}   # symbol -> monthly_df
        for symbol in SYMBOLS:
            try:
                merged_data = run_single_backtest(symbol, version, bt_params)
                monthly = calc_monthly_metrics(merged_data)
                if not monthly.empty:
                    all_monthly[symbol] = monthly
                    print(f"    {symbol} 回测月份: {len(monthly)}")
            except Exception as e:
                print(f"  ✗ {symbol} 回测失败: {e}")

        if not all_monthly:
            print("无有效回测数据，跳过")
            continue

        # ---- 构建宽表 ----
        metrics = ["monthly_ret", "sharpe", "max_dd", "win_rate", "avg_pos", "trade_days"]

        for metric in metrics:
            wide = pd.DataFrame()
            for symbol, monthly_df in all_monthly.items():
                if metric in monthly_df.columns:
                    wide[symbol] = monthly_df[metric]

            if not wide.empty:
                wide.index.name = "month"
                out_path = output_dir / f"{metric}_wide_{TRAIN_END_DATE}_{version}.csv"
                wide.to_csv(out_path)
                print(f"  宽表已保存: {out_path}")

        # ---- 每个品种的累计收益图 ----
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(all_monthly)
        cols = 4
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

        for idx, (symbol, monthly_df) in enumerate(all_monthly.items()):
            ax = axes[idx]
            cum = monthly_df["monthly_ret"].cumsum()
            ax.plot(cum.index.astype(str), cum.values, linewidth=1.2)
            ax.set_title(symbol, fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="x", rotation=45, labelsize=7)

        for idx in range(len(all_monthly), len(axes)):
            axes[idx].axis("off")

        fig.tight_layout()
        fig_path = output_dir / f"monthly_equity_{TRAIN_END_DATE}_{version}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"  静态图已保存: {fig_path}")
        plt.close(fig)


def main():
    model_base_dir = pl.get_model_base_dir(TRAIN_END_DATE)
    print(f"模型目录: {model_base_dir}")
    print(f"回测窗口: {BT_START} ~ {BT_END}")
    print(f"品种: {SYMBOLS}")
    print(f"参数组数: {len(PARAM_SETS)}")

    # 用副本跑，避免修改原始 PARAM_SETS
    for ps in PARAM_SETS:
        run_param_set(ps.copy())

    print(f"\n全部完成。")


if __name__ == "__main__":
    main()
