"""
数据加载封装
提供从 mnt 或本地加载主力合约数据、因子数据的标准接口
"""

import os
import warnings

import pandas as pd

from src.path_resolver import PathResolver

warnings.filterwarnings("ignore")


class DataLoader:
    """
    数据加载器

    支持从 mnt 或本地加载数据，自动处理路径解析
    """

    def __init__(self):
        self.pr = PathResolver()

    # ------------------------------------------------------------------ #
    # 主力合约 1min 数据
    # ------------------------------------------------------------------ #
    def load_main_1min(self, symbol: str, from_local: bool = True) -> pd.DataFrame:
        """
        加载主力合约 1 分钟数据

        Parameters
        ----------
        symbol : str
            品种代码，如 "A"
        from_local : bool
            True 从本地加载，False 从 mnt 加载
        """
        if from_local:
            path = self.pr.resolve("local", "raw", symbol) / "main_1min.csv"
        else:
            path = self.pr.resolve("mnt", "data_1min_active") / f"main_{symbol}.csv"

        if not path.exists():
            raise FileNotFoundError(f"主力合约 1min 数据不存在: {path}")

        df = pd.read_csv(path, parse_dates=["ts"])
        df = df.set_index("ts")
        # 分钟级数据统一用 ts 作为时间索引名
        df.index.name = "ts"
        return df

    # ------------------------------------------------------------------ #
    # 合约切换表
    # ------------------------------------------------------------------ #
    def load_contract_info(self, symbol: str, from_local: bool = True) -> pd.DataFrame:
        """加载主力合约切换表"""
        if from_local:
            path = self.pr.resolve("local", "raw", symbol) / "contract_info.csv"
        else:
            path = self.pr.resolve("mnt", "future_info", symbol) / "main_instrument.csv"

        if not path.exists():
            raise FileNotFoundError(f"合约切换表不存在: {path}")

        df = pd.read_csv(path, dtype=str)
        df["trade_date"] = df["trade_date"].map(lambda x: x.split(" ")[0])
        return df

    # ------------------------------------------------------------------ #
    # 单品种因子（从 mnt 或本地）
    # ------------------------------------------------------------------ #
    def load_single_factor(
        self,
        symbol: str,
        factor_name: str,
        mode: str = "t2t",
        instrument: str | None = None,
        from_local: bool = False,
    ) -> pd.DataFrame:
        """
        加载单品种因子

        Parameters
        ----------
        symbol : str
        factor_name : str
            因子文件名（不含后缀）
        mode : str
            'm2m' | 't2m' | 't2t'
        instrument : str, optional
            合约代码，t2t 模式下需要
        from_local : bool
        """
        if from_local:
            base = self.pr.resolve("local", "processed", symbol) / mode
        else:
            base = self.pr.resolve("mnt", "factor", symbol, mode)

        if mode == "t2t" and instrument:
            path = base / instrument / f"{factor_name}@{instrument}.feather"
        else:
            path = base / f"{factor_name}@{instrument or symbol}.feather"

        if not path.exists():
            raise FileNotFoundError(f"因子文件不存在: {path}")

        return pd.read_feather(path)

    # ------------------------------------------------------------------ #
    # 分组产物（all_factor、预测值、回测结果）
    # ------------------------------------------------------------------ #
    def load_all_factor(self, group_name: str, symbol: str) -> pd.DataFrame:
        """加载指定分组下某品种的完整因子表"""
        path = (
            self.pr.resolve("local", "group", group_name, "all_factor")
            / f"{symbol}_all_fac.feather"
        )
        if not path.exists():
            raise FileNotFoundError(f"all_factor 不存在: {path}")
        df = pd.read_feather(path)
        # mnt 侧的 all_factor 时间列叫 datetime，统一重命名为 ts
        if "datetime" in df.columns:
            df = df.rename(columns={"datetime": "ts"})
        df["ts"] = pd.to_datetime(df["ts"])
        return df.set_index("ts")

    def load_predictions(self, group_name: str, symbol: str, model_folder: str) -> pd.DataFrame:
        """加载预测值"""
        path = (
            self.pr.resolve("local", "group", group_name, "predictions")
            / f"{symbol}_{model_folder}.csv"
        )
        if not path.exists():
            raise FileNotFoundError(f"预测值文件不存在: {path}")
        return pd.read_csv(path, parse_dates=["datetime"])

    def load_backtest_result(self, group_name: str, symbol: str, model_folder: str) -> pd.DataFrame:
        """加载回测结果"""
        path = (
            self.pr.resolve("local", "group", group_name, "backtest")
            / f"{symbol}_{model_folder}_backtest.feather"
        )
        if not path.exists():
            raise FileNotFoundError(f"回测结果不存在: {path}")
        return pd.read_feather(path)
