"""
回测适配器

复用 /home/future_commodity/function_future/backtest_v3.py 中的回测逻辑，
适配 Strat Lab 的路径体系和分组概念。
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/future_commodity")
import function_future.backtest_v3 as bv

warnings.filterwarnings("ignore")

from src.config_manager import ConfigManager
from src.data_loader import DataLoader
from src.path_resolver import PathResolver


class BacktestPipeline:
    """
    回测流水线

    封装加载模型、生成预测、回测、统计的完整流程
    """

    def __init__(self, group_name: str, symbol: str):
        self.group_name = group_name
        self.symbol = symbol
        self.cm = ConfigManager()
        self.dl = DataLoader()
        self.cfg = self.cm.get_pipeline()
        self.pr = PathResolver()

    def run(
        self,
        model_folder_name: str,
        test_start_date: str | None = None,
        test_end_date: str | None = None,
        th1: float | None = None,
        th2: float | None = None,
        holding_period: int | None = None,
    ) -> pd.DataFrame:
        """
        运行回测

        Parameters
        ----------
        model_folder_name : str
            模型文件夹名，如 "A_pred5_2025-01-01_v0"
        test_start_date : str, optional
            回测开始日期，默认训练截止日次日
        test_end_date : str, optional
            回测结束日期，默认全部数据
        th1, th2 : float, optional
            信号阈值，默认从 pipeline.yaml 读取
        holding_period : int, optional
            最大持仓分钟数，默认从 pipeline.yaml 读取

        Returns
        -------
        pd.DataFrame
            回测结果明细
        """
        th1 = th1 or self.cfg["backtest"]["th1"]
        th2 = th2 or self.cfg["backtest"]["th2"]
        holding_period = holding_period or self.cfg["backtest"]["holding_period"]

        # 加载 all_factor（DataLoader 已统一返回 ts 为索引）
        df = self.dl.load_all_factor(self.group_name, self.symbol)
        df = df.reset_index()
        df["ts"] = pd.to_datetime(df["ts"])

        # 加载收益率标签
        train_label = self.cfg["training"]["train_label"]
        rtn_df = self.dl.load_main_1min(self.symbol, from_local=True)
        rtn_col = f"rtn_{train_label}"
        df = df.merge(
            rtn_df[[rtn_col]].reset_index().rename(columns={"ts": "ts"}),
            on="ts",
            how="left",
        )
        df = df.rename(columns={rtn_col: "pred_ret"})

        # 加载模型并生成预测值
        preds = self._load_models_and_predict(model_folder_name, df)
        df["factor"] = preds["weighted"]
        df["weighted_s"] = preds["weighted_s"]

        # 构造回测所需字段
        df["trade_date"] = df["ts"].dt.strftime("%Y-%m-%d")

        # 截取测试区间
        train_end = pd.to_datetime(self.cfg["training"]["train_end_date"])
        if test_start_date is None:
            test_start = train_end + pd.Timedelta(days=1)
        else:
            test_start = pd.to_datetime(test_start_date)
        if test_end_date:
            test_end = pd.to_datetime(test_end_date)
            df = df[(df["ts"] >= test_start) & (df["ts"] <= test_end)]
        else:
            df = df[df["ts"] >= test_start]

        # 运行回测
        instrument_cfg = self.cm.get_instrument_config(self.symbol)
        trading_hours = instrument_cfg["trade_hours"]

        merged = bv.process_signals_v2(
            df,
            th1=th1,
            th2=th2,
            holding_period=holding_period,
            warmup=100,
            day=self.cfg["backtest"]["day"],
            date_max_trade=self.cfg["backtest"]["date_max_trade"],
            ts_col="datetime",
            factor_col="factor",
            open_drop=self.cfg["backtest"]["open_drop"],
            close_drop=self.cfg["backtest"]["close_drop"],
            trading_hours=trading_hours,
            mask_hours=[],
        )

        # 保存结果
        save_dir = self.pr.ensure_dir(
            self.pr.resolve("local", "group", self.group_name, "backtest")
        )
        save_path = save_dir / f"{self.symbol}_{model_folder_name}_backtest.feather"
        merged.reset_index(drop=True).to_feather(save_path)
        print(f"Backtest result saved: {save_path}")

        return merged

    def _load_models_and_predict(self, model_folder_name: str, df: pd.DataFrame) -> pd.DataFrame:
        """加载 5-fold 模型并生成加权预测值"""
        import lightgbm as lgb

        model_dir = self.pr.resolve("mnt", "model") / model_folder_name
        # 如果 mnt 没有，尝试本地
        if not model_dir.exists():
            model_dir = (
                self.pr.resolve("local", "group", self.group_name, "models")
                / model_folder_name
            )

        models = []
        weights = []

        for i in range(1, 6):
            model_file = model_dir / f"kfold_fold{i}_0.lgb"
            meta_file = model_dir / f"kfold_fold{i}_0_meta.json"

            if not model_file.exists():
                raise FileNotFoundError(f"模型文件不存在: {model_file}")

            m = lgb.Booster(model_file=str(model_file))
            models.append(m)

            with open(meta_file, "r") as f:
                meta = json.load(f)
            weights.append(float(np.log(meta["best_iteration"] + 1)))

        # 构造输入特征
        feature_names = models[0].feature_name()
        X = df[[c for c in feature_names if c in df.columns]].copy()
        if "hour" not in X.columns:
            X["hour"] = df["datetime"].dt.hour
        X = X[feature_names]

        # 预测
        preds = pd.DataFrame(
            [m.predict(X) for m in models],
            columns=df.index,
            index=[f"model_{i}" for i in range(1, 6)],
        ).T

        preds["weighted"] = preds.mul(weights, axis=1).sum(axis=1) / sum(weights)
        preds["weighted_s"] = (
            preds["weighted"] * 0.6
            + preds["weighted"].shift(1) * 0.3
            + preds["weighted"].shift(2) * 0.1
        )

        return preds

    def analyze(self, merged: pd.DataFrame) -> dict:
        """快速统计回测结果"""
        stats = bv.analyze_pos_distribution(merged)
        return stats.to_dict()
