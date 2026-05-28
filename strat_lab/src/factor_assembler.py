"""
单品种因子拼表器

从 mnt 读取底层的 m2m / t2m / t2t 因子 feather，
按主力合约切换表拼接成单品种因子表（不含跨品种因子）。

与 make_main_factor_dataframe_dce农.py 逻辑一致，但：
- 并行度严格可控（默认 n_jobs=1）
- 所有 mnt 数据只读不写
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from src.config_manager import ConfigManager
from src.data_loader import DataLoader
from src.path_resolver import PathResolver


class FactorAssembler:
    """
    单品种因子拼表器

    用法:
        asm = FactorAssembler("A", n_jobs=1)
        single_fac = asm.assemble()
        # single_fac 索引为 datetime，列包含所有 m2m/t2m/t2t 因子（不含跨品种因子）
    """

    def __init__(self, symbol: str, n_jobs: int = 1):
        self.symbol = symbol
        self.n_jobs = n_jobs
        self.pr = PathResolver()
        self.dl = DataLoader()
        self.cm = ConfigManager()

        # 加载主力合约切换表
        self.contract_info = self.dl.load_contract_info(symbol, from_local=True)
        self.instruments_lst = self._get_instruments()

        # 加载因子列表（复刻 DataLoader.load_f_lst / load_tf_lst 逻辑）
        fac_cfg = self.cm._load("factor_config")["FAC"]

        # m2m：无 modifiers
        self.mf_fac_lst = fac_cfg["min_min_fac"]["fac_func_lst"]

        # t2m：无 modifiers
        self.tmf_fac_lst = fac_cfg["tick_min_fac"]["fac_func_lst"]

        # t2t：有 modifiers，需笛卡尔积
        tf_cfg = fac_cfg["tick_tick_fac"]
        tf_base = tf_cfg["fac_func_lst"]
        tf_mods = tf_cfg.get("modifiers", [None])
        from itertools import product

        self.tf_fac_lst = [
            f"{func}_{mod}" if mod else func for func, mod in product(tf_base, tf_mods)
        ]

    def _get_instruments(self) -> list[str]:
        """获取该品种的所有主力合约列表"""
        return sorted(self.contract_info["instrument"].dropna().unique().tolist())

    def assemble(self) -> pd.DataFrame:
        """
        拼表主入口

        Returns
        -------
        pd.DataFrame
            索引为 datetime（分钟右界，统一命名为 ts），列为单品种因子
        """
        print(f"[{self.symbol}] 开始拼表，合约数={len(self.instruments_lst)}, n_jobs={self.n_jobs}")

        # 1. t2t
        t2tfac = self._assemble_mode("t2t", self.tf_fac_lst)
        if t2tfac.empty:
            raise ValueError(f"{self.symbol} t2t 为空，无法拼表")

        # 2. t2m（只读到 t2t 成功处理的合约数）
        valid_instruments = t2tfac.index.get_level_values("instrument").unique().tolist()
        t2mfac = self._assemble_mode("t2m", self.tmf_fac_lst, instruments=valid_instruments)

        # 3. m2m
        m2mfac = self._assemble_mode("m2m", self.mf_fac_lst, instruments=valid_instruments)

        # 4. 拼接单品种因子
        print(f"[{self.symbol}] 拼接 t2t + t2m + m2m ...")
        single_fac = t2tfac.join(t2mfac, how="left").join(m2mfac, how="left")
        single_fac = single_fac.reset_index().set_index("datetime")
        single_fac.index.name = "ts"

        print(f"[{self.symbol}] 单品种因子拼表完成: {single_fac.shape}")
        return single_fac

    def _assemble_mode(
        self,
        mode: str,
        fac_lst: list[str],
        instruments: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        按模式（m2m/t2m/t2t）读取并拼接因子

        Parameters
        ----------
        mode : str
            'm2m' | 't2m' | 't2t'
        fac_lst : list[str]
            该模式下的因子函数名列表
        instruments : list[str], optional
            指定合约列表，None 则用全部
        """
        instruments = instruments or self.instruments_lst
        dfs = []

        for instrument in tqdm(instruments, desc=f"{self.symbol} {mode}", leave=False):
            fac_df = self._read_factors_for_instrument(instrument, mode, fac_lst)
            if fac_df.empty:
                print(f"[{self.symbol}] {instrument} {mode} empty, skip")
                continue
            fac_df["instrument"] = instrument
            dfs.append(fac_df)

        if not dfs:
            return pd.DataFrame()

        # 纵向拼接所有合约
        combined = pd.concat(dfs)
        # 设置 (datetime, instrument) 双重索引
        combined = combined.reset_index(names="datetime").set_index(["datetime", "instrument"])
        return combined

    def _read_factors_for_instrument(
        self,
        instrument: str,
        mode: str,
        fac_lst: list[str],
    ) -> pd.DataFrame:
        """
        读取单个合约的所有指定因子，横向拼接

        支持并行读取（当 n_jobs > 1 时）
        """
        if self.n_jobs == 1:
            results = []
            for fac_name in tqdm(fac_lst, desc=f"{instrument}", leave=False):
                df = self._read_single_factor(fac_name, instrument, mode)
                if not df.empty:
                    results.append(df)
        else:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._read_single_factor)(fac_name, instrument, mode)
                for fac_name in fac_lst
            )
            results = [r for r in results if not r.empty]

        if not results:
            return pd.DataFrame()

        # 横向拼接：同一个合约的所有因子按 datetime 对齐
        fac_df = pd.concat(results, axis=1)
        return fac_df

    def _read_single_factor(
        self,
        factor_name: str,
        instrument: str,
        mode: str,
    ) -> pd.DataFrame:
        """
        读取单个因子 feather 文件

        与 make_main_factor_dataframe_dce农.py 的 read_factor 逻辑一致：
        - 只保留该合约作为主力合约期间的数据
        - 返回 (datetime, factor_value) 的单列 DataFrame
        """
        try:
            if mode == "t2t":
                path = (
                    self.pr.resolve("mnt", "factor", self.symbol, mode)
                    / instrument
                    / f"{factor_name}@{instrument}.feather"
                )
            else:
                path = (
                    self.pr.resolve("mnt", "factor", self.symbol, mode)
                    / f"{factor_name}@{instrument}.feather"
                )

            if not path.exists():
                # 静默跳过不存在的因子（和原代码一致）
                return pd.DataFrame()

            df = pd.read_feather(path)
            df["trade_date"] = df["datetime"].dt.strftime("%Y-%m-%d")

            # 列名统一：t2t 加 FAC_ 前缀，m2m/t2m 不加
            factor_col = f"FAC_{factor_name}" if mode == "t2t" else factor_name
            df.columns = ["datetime", "instrument", "factor_name", "factor_value", "trade_date"]

            # 只保留该合约在主力合约表中的时间段
            df = (
                df.merge(
                    self.contract_info[["instrument", "trade_date"]].dropna(),
                    on=["instrument", "trade_date"],
                    how="inner",
                )
                .set_index("datetime")[["factor_value"]]
            )
            df.columns = [factor_col]
            return df

        except Exception:
            return pd.DataFrame()
