"""
FactorRecon — 研究环境 vs 实时测试环境 因子核对核心模块

用法:
    from factor_recon.recon_core import FactorRecon
    recon = FactorRecon("C")
    recon.run_all()          # 一键加载、对齐、计算、画图
    recon.plot()             # 仅画图（需先 run_all）
"""

import json
import warnings
from pathlib import Path

try:
    from IPython.display import display
except ImportError:
    display = print

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# 默认路径（可通过参数覆盖）
_DEFAULT_CONFIG_PATH = "/home/strategy_PAMY_dev/config.json"
_DEFAULT_RT_PREFIX = "/home/strategy_PAMY_dev/save_files"
_DEFAULT_RESEARCH_PREFIX = "/mnt/Data/writable/liaoyuyang/factor"
_DEFAULT_MODEL_ROOT = "/home/strategy_PAMY_dev/models"

_DROP_COLS = {"datetime", "instrument", "hour"}


class FactorRecon:
    """
    封装「研究环境因子 / 实时环境因子 / 预测值」的加载、对齐、核对、可视化。
    """

    def __init__(
        self,
        symbol: str,
        *,
        contract: str | None = None,
        night_date: str | None = None,
        day_date: str | None = None,
        rt_prefix: str | None = None,
        research_prefix: str | None = None,
        model_root: str | None = None,
        config_path: str = _DEFAULT_CONFIG_PATH,
        model_folder: str | None = None,
        tol_abs: float = 0.01,
        tol_rel: float = 0.05,
    ):
        """
        Parameters
        ----------
        symbol : str
            品种代码，如 "C", "P", "M" …
        night_date, day_date : str, optional
            夜盘日期、白天盘日期。未提供时自动读取 config.json 的 replay_mode。
        rt_prefix, research_prefix, model_root : str, optional
            实时数据根目录、研究因子根目录、模型根目录。
        config_path : str
            config.json 路径。
        """
        self.symbol = symbol.upper()
        self.contract = contract or f"{self.symbol.lower()}2605"

        # 读取 config.json
        with open(config_path, "r") as f:
            self._config = json.load(f)

        replay = self._config.get("replay_mode", {})
        self._replay_enabled = replay.get("enabled", False)

        self.night_date = night_date or replay.get("night_date", "2026-03-22")
        self.day_date = day_date or replay.get("day_date", "2026-03-23")

        self.rt_prefix = Path(rt_prefix or _DEFAULT_RT_PREFIX)
        self.research_prefix = Path(research_prefix or _DEFAULT_RESEARCH_PREFIX)
        self.model_root = Path(model_root or _DEFAULT_MODEL_ROOT)
        self.model_folder = Path(model_folder) if model_folder else None

        self.tol_abs = tol_abs
        self.tol_rel = tol_rel

        # ---------- 占位属性（run_all 后填充） ----------
        self.fac_research: pd.DataFrame | None = None
        self.rt_df: pd.DataFrame | None = None
        self.fac_res: pd.DataFrame | None = None
        self.fac_rt: pd.DataFrame | None = None
        self.common_factors: list[str] | None = None

        self.pred_res: pd.DataFrame | None = None
        self.rt_pred: pd.DataFrame | None = None

        self.model_lst: list = []
        self.weight_lst: list[float] = []

    # ------------------------------------------------------------------ #
    # 1. 加载因子
    # ------------------------------------------------------------------ #
    def load_factors(self) -> None:
        """加载研究环境因子 + 实时测试环境因子。"""
        self._load_research_factors()
        self._load_rt_factors()

    def _load_research_factors(self) -> None:
        research_path = self.research_prefix / self.symbol / "all_fac" / "all_factor.feather"
        print(f"[{self.symbol}] 读取研究环境因子: {research_path}")
        fac_all = pd.read_feather(research_path)
        fac_all["datetime"] = pd.to_datetime(fac_all["datetime"])

        # 判断实时数据是否包含夜盘 / 白天盘
        rt_dir = self.rt_prefix / self.symbol / "factors"
        rt_files = sorted(rt_dir.glob("factors_*.csv"))
        if not rt_files:
            raise FileNotFoundError(f"未找到实时因子文件: {rt_dir}")
        hours = [int(f.name.split("_")[2].split("-")[0]) for f in rt_files]
        has_night = any(h >= 20 for h in hours)
        has_day = any(h < 20 for h in hours)
        print(f"[{self.symbol}] 实时文件数={len(rt_files)}, has_night={has_night}, has_day={has_day}")

        fac_parts = []
        if has_night:
            fac_night = fac_all[
                (fac_all["datetime"].dt.date == pd.to_datetime(self.night_date).date())
                & (fac_all["datetime"].dt.hour >= 20)
            ].copy()
            fac_parts.append(fac_night)
            print(f"[{self.symbol}] 研究环境夜盘 {self.night_date}: {len(fac_night)} 根")
        if has_day:
            fac_day = fac_all[
                (fac_all["datetime"].dt.date == pd.to_datetime(self.day_date).date())
                & (fac_all["datetime"].dt.hour < 20)
            ].copy()
            fac_parts.append(fac_day)
            print(f"[{self.symbol}] 研究环境白天盘 {self.day_date}: {len(fac_day)} 根")

        self.fac_research = (
            pd.concat(fac_parts).sort_values("datetime") if fac_parts else pd.DataFrame()
        )
        self.fac_research = self.fac_research.set_index("datetime")
        print(f"[{self.symbol}] 研究因子总行数={len(self.fac_research)}, 列数={len(self.fac_research.columns)}")

    def _load_rt_factors(self) -> None:
        rt_dir = self.rt_prefix / self.symbol / "factors"
        rt_files = sorted(rt_dir.glob("factors_*.csv"))
        if not rt_files:
            raise FileNotFoundError(f"未找到实时因子文件: {rt_dir}")

        rt_dfs = []
        for f in rt_files:
            rt_dfs.append(pd.read_csv(f, parse_dates=["datetime"]))
        rt_df = pd.concat(rt_dfs, ignore_index=True).sort_values("datetime").reset_index(drop=True)
        rt_df = rt_df.drop_duplicates(subset=["datetime"], keep="last")

        rt_time_min = rt_df["datetime"].iloc[0].time()
        rt_time_max = rt_df["datetime"].iloc[-1].time()
        latest_date = rt_df["datetime"].dt.date.max()

        if self._replay_enabled:
            print(f"[{self.symbol}] [Replay] save_files 日期已为 replay 日期，无需还原")
        else:
            def _restore_date(dt):
                if dt.date() != latest_date:
                    return dt
                t = dt.strftime("%H:%M:%S")
                if dt.hour >= 20:
                    return pd.Timestamp(f"{self.night_date} {t}")
                return pd.Timestamp(f"{self.day_date} {t}")

            rt_df["datetime"] = rt_df["datetime"].apply(_restore_date)
            print(f"[{self.symbol}] [日期还原] 夜盘→{self.night_date}, 白天盘→{self.day_date}")

        print(
            f"[{self.symbol}] 实时因子 文件数={len(rt_files)}, "
            f"时间范围={rt_time_min}~{rt_time_max}, 去重后行数={len(rt_df)}"
        )
        self.rt_df = rt_df.set_index("datetime")

    # ------------------------------------------------------------------ #
    # 2. 因子对齐
    # ------------------------------------------------------------------ #
    def align_factors(self) -> None:
        """对齐研究因子与实时因子，提取共有因子列。"""
        if self.fac_research is None or self.rt_df is None:
            raise RuntimeError("请先调用 load_factors()")

        common_index = self.rt_df.index.intersection(self.fac_research.index)
        print(f"[{self.symbol}] [因子对齐] 共有时间点={len(common_index)}")
        if len(common_index) == 0:
            print("⚠️ 无交集！请检查日期或时间范围。")
            print("实时时间样例:", self.rt_df.index[:3].tolist())
            print("研究时间样例:", self.fac_research.index[:3].tolist())
            return

        print(
            f"[{self.symbol}] 时间范围: {common_index.min().strftime('%H:%M')} ~ "
            f"{common_index.max().strftime('%H:%M')}"
        )

        rt_cols = set(self.rt_df.columns) - _DROP_COLS
        research_cols = set(self.fac_research.columns) - _DROP_COLS
        self.common_factors = sorted(rt_cols & research_cols)
        only_rt = sorted(rt_cols - research_cols)
        only_research = sorted(research_cols - rt_cols)

        print(f"[{self.symbol}] 共有因子={len(self.common_factors)}")
        if only_rt:
            print(f"[{self.symbol}] 仅实时有: {only_rt[:10]}{'...' if len(only_rt) > 10 else ''}")
        if only_research:
            print(f"[{self.symbol}] 仅研究有: {only_research[:10]}{'...' if len(only_research) > 10 else ''}")

        self.fac_rt = self.rt_df.loc[common_index, self.common_factors]
        self.fac_res = self.fac_research.loc[common_index, self.common_factors]
        print(f"[{self.symbol}] [对齐后] 实时{self.fac_rt.shape}, 研究{self.fac_res.shape}")

    # ------------------------------------------------------------------ #
    # 3. 预测值
    # ------------------------------------------------------------------ #
    def compute_predictions(self) -> None:
        """计算研究环境预测值 + 加载实时环境预测值 + 对齐。"""
        self._compute_research_pred()
        self._load_rt_pred()
        self._align_predictions()

    def _compute_research_pred(self) -> None:
        if self.fac_res is None:
            raise RuntimeError("请先调用 align_factors()")

        self.model_lst.clear()
        self.weight_lst.clear()
        model_dir = self.model_folder or (self.model_root / self.symbol)
        for i in range(1, 6):
            model_file = model_dir / f"kfold_fold{i}_0.lgb"
            meta_file = model_dir / f"kfold_fold{i}_0_meta.json"
            m = lgb.Booster(model_file=str(model_file))
            with open(meta_file, "r") as f:
                meta_data = json.load(f)
            self.model_lst.append(m)
            self.weight_lst.append(float(np.log(meta_data["best_iteration"] + 1)))

        print(f"[{self.symbol}] 加载模型×{len(self.model_lst)}, weights={np.round(self.weight_lst, 3).tolist()}")

        factor_col = self.model_lst[0].feature_name()
        rs_input = self.fac_res[[c for c in factor_col if c in self.fac_res.columns]].copy()
        if "hour" not in rs_input.columns:
            rs_input["hour"] = rs_input.index.hour
        rs_input = rs_input[factor_col]

        pred_res = pd.DataFrame(
            [m.predict(rs_input) for m in self.model_lst],
            columns=rs_input.index,
            index=[f"model_{i + 1}" for i in range(len(self.model_lst))],
        ).T
        pred_res["weighted"] = pred_res.mul(self.weight_lst, axis=1).sum(axis=1) / sum(self.weight_lst)
        pred_res["weighted_s"] = (
            pred_res["weighted"] * 0.6
            + pred_res["weighted"].shift(1) * 0.3
            + pred_res["weighted"].shift(2) * 0.1
        )
        self.pred_res = pred_res
        print(f"[{self.symbol}] [研究预测] shape={self.pred_res.shape}")

    def _load_rt_pred(self) -> None:
        rt_pred_dir = self.rt_prefix / self.symbol / "predictions"
        rt_pred_files = sorted(rt_pred_dir.glob("predictions_*.csv"))
        if not rt_pred_files:
            raise FileNotFoundError(f"未找到实时预测文件: {rt_pred_dir}")

        rt_pred_dfs = []
        for f in rt_pred_files:
            rt_pred_dfs.append(pd.read_csv(f, parse_dates=["datetime"]))
        rt_pred = pd.concat(rt_pred_dfs, ignore_index=True).sort_values("datetime").reset_index(drop=True)
        rt_pred = rt_pred.drop_duplicates(subset=["datetime"], keep="last")

        if not self._replay_enabled:
            latest_date = rt_pred["datetime"].dt.date.max()

            def _restore_date(dt):
                if dt.date() != latest_date:
                    return dt
                t = dt.strftime("%H:%M:%S")
                if dt.hour >= 20:
                    return pd.Timestamp(f"{self.night_date} {t}")
                return pd.Timestamp(f"{self.day_date} {t}")

            rt_pred["datetime"] = rt_pred["datetime"].apply(_restore_date)

        self.rt_pred = rt_pred.set_index("datetime")
        print(f"[{self.symbol}] [实时预测] 文件数={len(rt_pred_files)}, 去重后行数={len(self.rt_pred)}")

    def _align_predictions(self) -> None:
        pred_common = self.pred_res.index.intersection(self.rt_pred.index)
        print(f"[{self.symbol}] [预测对齐] 共有时间点={len(pred_common)}")
        self.pred_res = self.pred_res.loc[pred_common]
        self.rt_pred = self.rt_pred.loc[pred_common]

        diff_w = (self.rt_pred["weighted"] - self.pred_res["weighted"]).abs()
        diff_ws = (self.rt_pred["weighted_s"] - self.pred_res["weighted_s"]).abs()
        print(f"[{self.symbol}] weighted   max_diff={diff_w.max():.4f}  mean_diff={diff_w.mean():.4f}")
        print(f"[{self.symbol}] weighted_s max_diff={diff_ws.max():.4f}  mean_diff={diff_ws.mean():.4f}")

    # ------------------------------------------------------------------ #
    # 4. 展示 / 检查
    # ------------------------------------------------------------------ #
    # ---------- 分开展示函数 ----------
    def show_pred_tail(self, n: int = 20) -> None:
        """研究环境预测值 tail。"""
        print("\n=== 预测值 tail (研究) ===")
        display(self.pred_res.tail(n))

    def show_rt_pred_tail(self, n: int = 20) -> None:
        """实时环境预测值 tail。"""
        print("\n=== 预测值 tail (实时) ===")
        display(self.rt_pred.tail(n))

    def show_pred_diff(self) -> None:
        """预测值 diff（Research - Realtime）。"""
        print("\n=== 预测值 diff ===")
        display(
            (self.pred_res - self.rt_pred)[
                ["model_1", "model_2", "model_3", "model_4", "model_5", "weighted_s"]
            ].round(4)
        )

    def show_factor_tail(self) -> None:
        """因子完整对比（研究 vs 实时，全部显示）。"""
        print("\n=== 因子 (研究) ===")
        display(self.fac_res.T.sort_index().round(8))
        print("\n=== 因子 (实时) ===")
        display(self.fac_rt.T.sort_index())

    def show_factor_diff(self) -> None:
        """因子 diff（Realtime - Research），全部显示，异常值用 142857 占位。"""
        print("\n=== 因子 diff ===")
        display(
            (
                self.fac_rt.fillna(142857)
                - self.fac_res.reindex_like(self.fac_rt).fillna(142857)
            ).T.round(4).sort_index()
        )

    def show_corr(self) -> None:
        """按天相关系数 + 整体相关系数。"""
        corr_df = self.correlations()
        print("\n=== 按天相关系数 ===")
        display(corr_df.round(4))

        mask = self.get_trade_mask()
        pr = self.pred_res.loc[mask]
        rt = self.rt_pred.loc[mask]
        w_pearson = np.corrcoef(pr["weighted"], rt["weighted"])[0, 1]
        w_spearman = stats.spearmanr(pr["weighted"], rt["weighted"])[0]
        ws_pearson = np.corrcoef(pr["weighted_s"].dropna(), rt["weighted_s"].dropna())[0, 1]
        ws_spearman = stats.spearmanr(pr["weighted_s"].dropna(), rt["weighted_s"].dropna())[0]
        print(f"\nOverall | weighted   pearson={w_pearson:.4f} spearman={w_spearman:.4f}")
        print(f"Overall | weighted_s pearson={ws_pearson:.4f} spearman={ws_spearman:.4f}")

    # 兼容旧接口，仍保留但内部调用分开展示
    def tail_summary(self, tail_num: int = 20) -> None:
        """打印预测值 tail、因子完整对比（一键展示全部）。"""
        self.show_pred_tail(tail_num)
        self.show_rt_pred_tail(tail_num)
        self.show_pred_diff()
        self.show_factor_tail()
        self.show_factor_diff()

    # ------------------------------------------------------------------ #
    # 5. 相关系数 + 画图
    # ------------------------------------------------------------------ #
    def get_trade_mask(self, df: pd.DataFrame | None = None) -> np.ndarray:
        """返回交易时段布尔 mask（numpy array）。"""
        idx = (df or self.pred_res).index
        return (
            ((idx.hour == 9) & (idx.minute >= 11))
            | ((idx.hour >= 10) & (idx.hour <= 14))
            | ((idx.hour == 15) & (idx.minute == 0))
            | ((idx.hour == 21) & (idx.minute >= 10))
            | ((idx.hour == 22) & (idx.minute <= 50))
        )

    def correlations(self) -> pd.DataFrame:
        """按天计算相关系数，返回 DataFrame。"""
        mask = self.get_trade_mask()
        pr = self.pred_res.loc[mask]
        rt = self.rt_pred.loc[mask]

        records = []
        for td in sorted(set(pr.index.date)):
            rs_day = pr[pr.index.date == td]
            rt_day = rt[rt.index.date == td]
            if len(rs_day) < 2:
                continue
            records.append(
                {
                    "date": td,
                    "w_pearson": np.corrcoef(rs_day["weighted"], rt_day["weighted"])[0, 1],
                    "w_spearman": stats.spearmanr(rs_day["weighted"], rt_day["weighted"])[0],
                    "ws_pearson": np.corrcoef(
                        rs_day["weighted_s"].dropna(), rt_day["weighted_s"].dropna()
                    )[0, 1],
                    "ws_spearman": stats.spearmanr(
                        rs_day["weighted_s"].dropna(), rt_day["weighted_s"].dropna()
                    )[0],
                }
            )
        return pd.DataFrame(records)

    def plot(self, figsize=(14, 10), show: bool = False, save_path: str | None = None) -> dict:
        """画出 weighted / weighted_s 的 Research vs Realtime 对比图（仅交易时段）。

        Parameters
        ----------
        show : bool
            是否弹窗显示（批量运行时建议 False）。
        save_path : str, optional
            图片保存路径。未提供时默认保存到 ``factor_recon/output/{symbol}.png``。

        Returns
        -------
        dict
            关键指标：pearson / spearman / max_diff / mean_diff
        """
        if self.pred_res is None or self.rt_pred is None:
            raise RuntimeError("请先调用 compute_predictions()")

        mask = self.get_trade_mask()
        pr = self.pred_res.loc[mask]
        rt = self.rt_pred.loc[mask]

        # 整体相关系数
        w_pearson = np.corrcoef(pr["weighted"], rt["weighted"])[0, 1]
        w_spearman = stats.spearmanr(pr["weighted"], rt["weighted"])[0]
        ws_pearson = np.corrcoef(pr["weighted_s"].dropna(), rt["weighted_s"].dropna())[0, 1]
        ws_spearman = stats.spearmanr(pr["weighted_s"].dropna(), rt["weighted_s"].dropna())[0]

        # x 轴用等距序号，避免非交易时段留白
        x_idx = np.arange(mask.sum())
        x_labels = self.pred_res.loc[mask].index.strftime("%m-%d %H:%M")

        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # weighted
        axes[0].plot(x_idx, pr["weighted"].values, label="Research", alpha=0.9, linewidth=1.2)
        axes[0].plot(x_idx, rt["weighted"].values, label="Realtime", alpha=0.9, linewidth=1.2)
        axes[0].plot(
            x_idx,
            (pr["weighted"].values - rt["weighted"].values),
            label="Research - Realtime",
            color="red",
            linestyle="--",
            alpha=0.8,
            linewidth=1.0,
        )
        axes[0].set_title(
            f"{self.symbol} weighted   | pearson={w_pearson:.4f}  spearman={w_spearman:.4f}"
        )
        axes[0].legend(loc="upper left")
        axes[0].grid(True, alpha=0.3)

        # weighted_s
        axes[1].plot(x_idx, pr["weighted_s"].values, label="Research", alpha=0.9, linewidth=1.2)
        axes[1].plot(x_idx, rt["weighted_s"].values, label="Realtime", alpha=0.9, linewidth=1.2)
        axes[1].plot(
            x_idx,
            (pr["weighted_s"].values - rt["weighted_s"].values),
            label="Research - Realtime",
            color="red",
            linestyle="--",
            alpha=0.8,
            linewidth=1.0,
        )
        axes[1].set_title(
            f"{self.symbol} weighted_s | pearson={ws_pearson:.4f}  spearman={ws_spearman:.4f}"
        )
        axes[1].legend(loc="upper left")
        axes[1].set_xlabel("datetime")
        axes[1].grid(True, alpha=0.3)

        step = max(1, len(x_idx) // 12)
        axes[1].set_xticks(x_idx[::step])
        axes[1].set_xticklabels(x_labels[::step], rotation=45, ha="right")

        plt.tight_layout()

        if save_path is None:
            save_path = f"/home/strategy_PAMY_dev/factor_recon/output/{self.symbol}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[{self.symbol}] 图片已保存: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        diff_w = (pr["weighted"] - rt["weighted"]).abs()
        diff_ws = (pr["weighted_s"] - rt["weighted_s"]).abs()

        return {
            "symbol": self.symbol,
            "w_pearson": w_pearson,
            "w_spearman": w_spearman,
            "ws_pearson": ws_pearson,
            "ws_spearman": ws_spearman,
            "w_max_diff": diff_w.max(),
            "w_mean_diff": diff_w.mean(),
            "ws_max_diff": diff_ws.max(),
            "ws_mean_diff": diff_ws.mean(),
        }

    # ------------------------------------------------------------------ #
    # 6. 一键执行
    # ------------------------------------------------------------------ #
    def run_all(self, show: bool = False) -> dict:
        """顺序执行：加载因子 → 对齐因子 → 计算预测值 → 画图 → 返回指标。

        Returns
        -------
        dict
            汇总指标，方便批量收集。
        """
        self.load_factors()
        self.align_factors()
        self.compute_predictions()
        return self.plot(show=show)
