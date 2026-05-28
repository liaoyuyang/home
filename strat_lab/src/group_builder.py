"""
分组构建器
根据 groups.yaml 管理跨品种因子的生成和 all_factor 的拼接
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.config_manager import ConfigManager
from src.data_loader import DataLoader
from src.factor_assembler import FactorAssembler
from src.path_resolver import PathResolver


class GroupBuilder:
    """
    分组构建器

    负责：
    1. 按分组计算跨品种因子
    2. 将单品种因子 + 跨品种因子拼接为 all_factor.feather
    """

    def __init__(self, group_name: str):
        self.group_name = group_name
        self.cm = ConfigManager()
        self.pr = PathResolver()
        self.dl = DataLoader()

        self.symbols = self.cm.get_group_symbols(group_name)
        self.group_root = self.pr.resolve("local", "group", group_name)
        self.pr.ensure_dir(self.group_root / "all_factor")

    def build_all_factor(
        self,
        single_factor_source: str = "mnt",  # 'mnt' | 'local'
        cross_factor_source: str = "local",  # 'local' 为本地已计算的跨品种因子
        n_jobs: int = 1,
    ) -> dict[str, Path]:
        """
        为分组内所有品种构建 all_factor.feather

        Parameters
        ----------
        single_factor_source : str
            单品种因子来源：'mnt' 从 mnt 读取，'local' 从本地读取
        cross_factor_source : str
            跨品种因子来源

        Returns
        -------
        dict[str, Path]
            各品种的 all_factor 保存路径
        """
        # 1. 加载跨品种因子（如果本地已存在）
        cross_fac_path = self.group_root / "cross_factors.feather"
        if cross_fac_path.exists() and cross_factor_source == "local":
            cross_fac = pd.read_feather(cross_fac_path)
            # 统一时间列名为 ts
            if "datetime" in cross_fac.columns:
                cross_fac = cross_fac.rename(columns={"datetime": "ts"})
            cross_fac["ts"] = pd.to_datetime(cross_fac["ts"])
            cross_fac = cross_fac.set_index("ts")
        else:
            cross_fac = None

        saved_paths = {}

        for symbol in tqdm(self.symbols, desc=f"Build {self.group_name} all_factor"):
            # 2. 加载单品种因子（从底层 m2m/t2m/t2t 重新拼，分组隔离）
            single_fac = self._load_single_factors(
                symbol, source=single_factor_source, n_jobs=n_jobs
            )

            # 3. 拼接跨品种因子
            if cross_fac is not None:
                related_cols = [
                    c for c in cross_fac.columns
                    if c.startswith(f"{symbol}_") or f"_{symbol}_" in c
                ]
                if related_cols:
                    mix = cross_fac[related_cols].copy()
                    mix.index = pd.to_datetime(mix.index)
                    single_fac = single_fac.join(mix, how="left")

            # 4. 清理并保存
            single_fac = single_fac.replace([np.inf, -np.inf], np.nan)
            save_path = self.group_root / "all_factor" / f"{symbol}_all_fac.feather"
            single_fac = single_fac.reset_index()
            # 统一时间列名为 ts
            if "datetime" in single_fac.columns:
                single_fac = single_fac.rename(columns={"datetime": "ts"})
            single_fac.sort_values("ts").to_feather(save_path)
            saved_paths[symbol] = save_path
            tqdm.write(f"[{symbol}] all_factor saved: {save_path} | shape={single_fac.shape}")

        return saved_paths

    def _load_single_factors(
        self, symbol: str, source: str = "mnt", n_jobs: int = 1
    ) -> pd.DataFrame:
        """
        加载某品种的所有单品种因子并拼接

        source='mnt': 从底层 m2m/t2m/t2t 重新拼，分组隔离（不直接读 mnt all_factor）
        source='local': 从本地 processed/ 读取已拼接好的 all_factor.feather
        """
        if source == "mnt":
            assembler = FactorAssembler(symbol=symbol, n_jobs=n_jobs)
            return assembler.assemble()

        # fallback：从本地 processed 读取（需预先同步/拼表）
        path = self.pr.resolve("local", "processed", symbol) / "all_factor.feather"
        if path.exists():
            df = pd.read_feather(path)
            if "datetime" in df.columns:
                df = df.rename(columns={"datetime": "ts"})
            df["ts"] = pd.to_datetime(df["ts"])
            return df.set_index("ts")

        raise FileNotFoundError(f"未找到 {symbol} 的单品种因子数据")

    def compute_cross_factors(self, method: str = "default") -> pd.DataFrame:
        """
        计算跨品种因子

        默认实现与现有 factor_cross_index.py 逻辑一致：
        - 两两品种的 close 收益率差（5/20周期）
        - 成交量比差（5/20周期）
        - 成交量滚动相关性（10周期）
        - 量价相关性差（10周期）
        - 持仓量变动差（5周期）

        可通过继承此类并重写该方法来实现自定义跨品种逻辑
        """
        # 加载分组内所有品种的 1min active 数据
        mkt_data_lst = []
        for symbol in self.symbols:
            df = self.dl.load_main_1min(symbol, from_local=True)
            df = df.drop_duplicates()
            mkt_data_lst.append(df)

        n = len(self.symbols)
        df_cross = pd.DataFrame()

        for i in range(n - 1):
            for j in range(i + 1, n):
                vi, vj = self.symbols[i], self.symbols[j]
                di, dj = mkt_data_lst[i], mkt_data_lst[j]

                # close 收益率差
                df_cross[f"{vi}_{vj}_closepctchg5_sub"] = di["close"].pct_change(5).sub(
                    dj["close"].pct_change(5)
                )
                df_cross[f"{vi}_{vj}_closepctchg20_sub"] = di["close"].pct_change(20).sub(
                    dj["close"].pct_change(20)
                )

                # 成交量比差
                df_cross[f"{vi}_{vj}_volumediv5_diff5"] = (
                    di["volume"].rolling(5).sum().div(dj["volume"].rolling(5).sum())
                ).diff(5)
                df_cross[f"{vi}_{vj}_volumediv20_diff5"] = (
                    di["volume"].rolling(20).sum().div(dj["volume"].rolling(20).sum())
                ).diff(5)

                # 成交量滚动相关性
                df_cross[f"{vi}_{vj}_vcorr10"] = (
                    di["volume"].rolling(10).corr(dj["volume"])
                )

                # 量价相关性差
                df_cross[f"{vi}_{vj}_cvcorr10_diff"] = (
                    di["close"].rolling(10).corr(di["volume"])
                ).sub(dj["close"].rolling(10).corr(dj["volume"]))

                # 持仓量变动差
                df_cross[f"{vi}_{vj}_oi5_diff"] = (
                    di["open_interest"].pct_change(5)
                ).sub(dj["open_interest"].pct_change(5))

        df_cross = df_cross.replace([-np.inf, np.inf], np.nan)

        # 保存：分钟级统一用 ts 作为时间列名
        save_path = self.group_root / "cross_factors.feather"
        self.pr.ensure_dir(save_path.parent)
        df_cross = df_cross.reset_index()
        if "datetime" in df_cross.columns:
            df_cross = df_cross.rename(columns={"datetime": "ts"})
        df_cross.to_feather(save_path)
        print(f"Cross factors saved: {save_path}")
        return df_cross.set_index("ts")
