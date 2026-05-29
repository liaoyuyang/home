#!/usr/bin/env python3
"""
DCE 农产品跨训练截止日期月度绩效对比
====================================
行 = 训练截止日期（模型迭代时间）
列 = 回测月份（样本外右端点）
值 = 月度收益 / 夏普 / 最大回撤 等

用法：
    cd /home/strategy_res/single/dce_农
    python eval_cross_dates.py
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import pipeline as pl
import function_future.DataLoader as DL

# =====================================================================
# 只改这里
# =====================================================================
TRAIN_END_DATES = ["2025-01-01", "2025-04-01", "2025-07-01", "2025-10-01", "2026-01-01"]
SYMBOLS = ["A", "B", "C", "CS", "M", "Y", "P", "LH"]
TRAIN_LABEL = 5
VERSIONS = ["v0", "v1"]  # 支持多版本对比，可改 ["v0"] 只跑一个

# 回测窗口：从各自训练截止日期开始，到最新数据
# 对比时 eval 脚本会自动截取公共月份
BT_START = None  # None 表示用 TRAIN_END_DATE
BT_END = "2026-05-07"

BT_PARAMS = {
    "th1": 0.9,
    "th2": 0.5,
    "holding_bars": 10,
    "holding_days": 5,
    "fee": 0,
    "v": 2,
}

# 输出目录
OUTPUT_DIR = Path("./eval_cross_dates")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 按品种分别出表，还是等权合成后出一张总表
PER_SYMBOL = True       # True: 每个品种一张表；False: 只出等权合成表
# =====================================================================


def run_single_backtest(symbol: str, train_end_date: str):
    """单个品种 + 单个训练截止日期的回测。"""
    folder_name = f"{symbol}_pred{TRAIN_LABEL}_{train_end_date}_{VERSION}"
    model_base_dir = pl.get_model_base_dir(train_end_date)

    # 检查模型是否存在
    model_folder = model_base_dir / folder_name
    if not model_folder.exists() or not any(model_folder.iterdir()):
        print(f"  [跳过] 模型不存在: {model_folder}")
        return None

    initial_capital = pl.calc_initial_capital(symbol)
    config_loader = DL.InstrumentConfig()
    inst_cfg = config_loader.get_instrument_config(symbol)
    bars_per_day = inst_cfg.get("bars_per_day", 345)

    bt_params = BT_PARAMS.copy()
    bt_params["day"] = BT_PARAMS["holding_days"] * bars_per_day

    _bt, merged_data = pl.run_backtest(
        symbol, train_end_date, folder_name, bt_params, initial_capital,
        window_end=BT_END, model_base_dir=model_base_dir
    )

    # 过滤回测窗口
    bt_start = BT_START if BT_START is not None else train_end_date
    merged_data = merged_data[
        merged_data["datetime"] >= pd.Timestamp(bt_start)
    ].reset_index(drop=True)

    return merged_data


def calc_monthly_metrics(merged_data: pd.DataFrame) -> pd.DataFrame:
    """从 merged_data 提取按月绩效指标。"""
    if merged_data.empty:
        return pd.DataFrame()

    md = merged_data.copy()
    md["datetime"] = pd.to_datetime(md["datetime"])
    md["month"] = md["datetime"].dt.to_period("M")

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
        max_dd=("pnl_ret", lambda x: _calc_dd(x.cumsum())),
        trade_days=("pnl_ret", "count"),
    )
    return monthly


def _calc_dd(cum: pd.Series) -> float:
    """计算序列的最大回撤（负数）。"""
    running_max = cum.cummax()
    dd = cum - running_max
    return dd.min()


def build_wide_table(monthly_results: dict, metric: str) -> dict:
    """
    构建宽表。
    monthly_results: {(train_end_date, symbol, version): monthly_df}
    返回: dict[name] -> wide_df, 其中 index=train_end_date, columns=month
    """
    if PER_SYMBOL:
        tables = {}
        for sym in SYMBOLS:
            sym_wide = pd.DataFrame()
            for ted in TRAIN_END_DATES:
                # 取该版本的数据（已由外层过滤）
                keys = [k for k in monthly_results.keys() if k[0] == ted and k[1] == sym]
                if not keys:
                    continue
                monthly_df = monthly_results[keys[0]]
                if metric in monthly_df.columns:
                    month_strs = monthly_df.index.astype(str)
                    sym_wide.loc[ted, month_strs] = monthly_df[metric].values
            if not sym_wide.empty:
                tables[sym] = sym_wide
        return tables
    else:
        wide = pd.DataFrame()
        for ted in TRAIN_END_DATES:
            monthly_list = []
            for sym in SYMBOLS:
                keys = [k for k in monthly_results.keys() if k[0] == ted and k[1] == sym]
                if not keys:
                    continue
                monthly_list.append(monthly_results[keys[0]][metric])

            if monthly_list:
                combined = pd.concat(monthly_list, axis=1).mean(axis=1)
                wide.loc[ted, combined.index.astype(str)] = combined.values
        return {"combined": wide}


def plot_heatmap(wide_df: pd.DataFrame, title: str, out_path: Path, cmap="RdYlGn"):
    """画热力图。"""
    if wide_df.empty:
        print(f"  [跳过] 空数据，不画热力图: {title}")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(wide_df.columns) * 1.2), max(4, len(wide_df) * 0.6)))

    # 格式化数值为百分比
    annot = wide_df.applymap(lambda x: f"{x:.2%}" if pd.notna(x) else "")

    sns.heatmap(
        wide_df,
        annot=annot,
        fmt="",
        cmap=cmap,
        center=0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "月度收益"},
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("月份")
    ax.set_ylabel("训练截止日期")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  热力图已保存: {out_path}")
    plt.close(fig)


def main():
    print(f"训练截止日期: {TRAIN_END_DATES}")
    print(f"回测窗口: {BT_START} ~ {BT_END}")
    print(f"品种: {SYMBOLS}\n")

    # ---- 1. 跑所有 (train_end_date, symbol) 的回测 ----
    all_results = {}   # (train_end_date, symbol) -> merged_data
    total_tasks = len(TRAIN_END_DATES) * len(SYMBOLS) * len(VERSIONS)
    task_idx = 0

    for ted in TRAIN_END_DATES:
        for sym in SYMBOLS:
            for ver in VERSIONS:
                task_idx += 1
                print(f"[{task_idx}/{total_tasks}] {ted} | {sym} | {ver}")
                try:
                    merged_data = run_single_backtest(sym, ted, ver)
                    if merged_data is not None and not merged_data.empty:
                        all_results[(ted, sym, ver)] = merged_data
                except Exception as e:
                    print(f"  ✗ 失败: {e}")

    if not all_results:
        print("无有效回测数据，退出")
        return

    # ---- 2. 按月算指标 ----
    monthly_results = {}  # (train_end_date, symbol) -> monthly_df
    for key, merged_data in all_results.items():
        monthly_results[key] = calc_monthly_metrics(merged_data)

    # ---- 3. 构建宽表并输出 ----
    metrics = ["monthly_ret", "sharpe", "win_rate", "max_dd"]

    for metric in metrics:
        print(f"\n=== 生成 {metric} 宽表 ===")
        tables = build_wide_table(monthly_results, metric)

        for name, wide_df in tables.items():
            if wide_df.empty:
                continue

            # 保存 CSV
            csv_path = OUTPUT_DIR / f"{metric}_wide_{name}.csv"
            wide_df.to_csv(csv_path)
            print(f"  CSV: {csv_path}")

            # 保存热力图（月度收益 + 夏普）
            if metric == "monthly_ret":
                plot_heatmap(
                    wide_df,
                    title=f"月度收益热力图 ({name})",
                    out_path=OUTPUT_DIR / f"heatmap_{name}.png",
                    cmap="RdYlGn",
                )
            elif metric == "sharpe":
                plot_heatmap(
                    wide_df,
                    title=f"月度夏普热力图 ({name})",
                    out_path=OUTPUT_DIR / f"heatmap_sharpe_{name}.png",
                    cmap="RdYlGn",
                )

    # ---- 4. 等权合成总表（额外） ----
    if PER_SYMBOL:
        print("\n=== 等权合成总表 ===")
        combined_wide = pd.DataFrame()
        for ted in TRAIN_END_DATES:
            monthly_list = []
            for sym in SYMBOLS:
                key = (ted, sym)
                if key not in monthly_results:
                    continue
                monthly_list.append(monthly_results[key]["monthly_ret"])

            if monthly_list:
                combined = pd.concat(monthly_list, axis=1).mean(axis=1)
                combined_wide.loc[ted, combined.index.astype(str)] = combined.values

        if not combined_wide.empty:
            csv_path = OUTPUT_DIR / "combined_monthly_ret_wide.csv"
            combined_wide.to_csv(csv_path)
            print(f"  CSV: {csv_path}")
            plot_heatmap(
                combined_wide,
                title="等权合成月度收益热力图",
                out_path=OUTPUT_DIR / "heatmap_combined.png"
            )

    print(f"\n全部完成。输出目录: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
