"""
训练适配器

复用 /home/future_commodity/function_future/ 中的训练逻辑，
但适配 Strat Lab 的路径体系和分组概念。

如果未来迁移到新机器且 /home/future_commodity 不可用，
需要将 pre_train.py / train_model.py 复制到 src/ 并替换此适配器。
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# 复用现有训练逻辑
sys.path.insert(0, "/home/future_commodity")
import function_future.pre_train as pt
import function_future.train_model as tm

warnings.filterwarnings("ignore")

from src.config_manager import ConfigManager
from src.data_loader import DataLoader


class TrainingPipeline:
    """
    训练流水线

    将预训练、因子筛选、KFold 训练封装为按分组/品种批量执行的接口
    """

    def __init__(self, group_name: str, symbol: str):
        self.group_name = group_name
        self.symbol = symbol
        self.cm = ConfigManager()
        self.dl = DataLoader()
        self.cfg = self.cm.get_pipeline()

    def load_data(self, train_end_date: str | None = None, train_label: int | None = None) -> pd.DataFrame:
        """
        加载训练数据

        Parameters
        ----------
        train_end_date : str, optional
            训练截止日期，默认从 pipeline.yaml 读取
        train_label : int, optional
            预测标签周期，默认从 pipeline.yaml 读取
        """
        train_end_date = train_end_date or self.cfg["training"]["train_end_date"]
        train_label = train_label or self.cfg["training"]["train_label"]

        # 加载分组产物 all_factor
        df = self.dl.load_all_factor(self.group_name, self.symbol)
        df = df.set_index("datetime")
        df.index = pd.to_datetime(df.index)

        # 截取训练截止日前
        df = df.loc[:train_end_date]

        # 切割交易时段（去掉开盘前10分钟等）
        instrument_cfg = self.cm.get_instrument_config(self.symbol)
        df = self._cut_time(df, instrument_cfg["trade_hours"])

        # 加载收益率标签
        rtn_df = self.dl.load_main_1min(self.symbol, from_local=True)
        rtn_col = f"rtn_{train_label}"
        if rtn_col not in rtn_df.columns:
            raise KeyError(f"{self.symbol} 的 1min 数据中缺少 {rtn_col}")

        rtn_df = rtn_df[[rtn_col]].reindex(df.index)
        df["pred_ret"] = rtn_df[rtn_col]
        df = df.replace([np.inf, -np.inf], np.nan)
        df["hour"] = df.index.hour

        return df

    def run_pretraining(self, df: pd.DataFrame, train_end_date: str, train_label: int) -> pt.Pretrainer:
        """运行预训练，输出重要性表、相关性表、分组表、单因子评估表"""
        pretrainer = pt.Pretrainer(
            variety=self.symbol,
            data=df,
            train_end_date=train_end_date,
            train_label=train_label,
        )
        pretrainer.run_full_pretraining(type_lgb="reg")
        return pretrainer

    def run_training(
        self,
        df: pd.DataFrame,
        factor_col: list[str],
        train_end_date: str,
        model_folder_name: str,
        n_splits: int = 5,
    ) -> dict:
        """
        运行 KFold 训练

        Parameters
        ----------
        df : pd.DataFrame
            包含特征和 label 的数据
        factor_col : list[str]
            筛选后的因子列表
        train_end_date : str
        model_folder_name : str
            模型保存文件夹名，如 "A_pred5_2025-01-01_v0"
        n_splits : int

        Returns
        -------
        dict
            fold 训练结果
        """
        # 适配 TimeSeriesAnalyzer
        analyzer = tm.TimeSeriesAnalyzer(
            symbol=self.symbol,
            factor_col=factor_col,
            train_end_date=train_end_date,
            config_loader=self._mock_config_loader(),
        )

        # 手动填充数据（绕过原有的文件加载逻辑）
        df_reset = df.reset_index()
        # 统一时间列名为 ts
        if "datetime" in df_reset.columns:
            df_reset = df_reset.rename(columns={"datetime": "ts"})
        analyzer.full_data = df_reset
        analyzer.train_data = df_reset[df_reset["ts"] <= train_end_date]
        analyzer.test_data = df_reset[df_reset["ts"] > train_end_date]
        analyzer.label_col = f"rtn_{self.cfg['training']['train_label']}"
        analyzer.target_col = "pred_ret"
        analyzer.category_col = self.cfg["training"]["category_col"]
        analyzer.ts_col = "ts"

        trainer = tm.LGBMTrainer(analyzer)
        results = trainer.train_kfold_v0(
            n_splits=n_splits,
            model_folder_name=model_folder_name,
            custom_params={"verbose": -1},
            plot_train=False,
        )
        return results

    def _cut_time(self, df: pd.DataFrame, trading_hours: list[str]) -> pd.DataFrame:
        """按交易时段过滤"""
        from src.factor_utils import time_scale_df
        df_reset = df.reset_index()
        # 分钟级统一用 ts
        time_col = "ts" if "ts" in df_reset.columns else "datetime"
        df_filtered = time_scale_df(df_reset, time_col, trading_hours)
        return df_filtered.set_index(time_col)

    def _mock_config_loader(self):
        """构造一个与现有 train_model 兼容的 config_loader 对象"""
        class MockLoader:
            def df_cut_time(self, df, trading_hours, drop_minutes=10):
                from src.factor_utils import time_scale_df
                df = df.reset_index()
                # 简单实现：去掉开盘前 drop_minutes 分钟
                # 完整逻辑应与 backtest_v3 中的 is_in_exclude_period 一致
                df = time_scale_df(df, "datetime", trading_hours)
                return df.set_index("datetime")

            def get_instrument_config(self, symbol):
                return {"trading_hours": ["09:00-11:30", "13:30-15:00", "21:00-23:00"]}

        return MockLoader()


def build_model_folder_name(symbol: str, train_label: int, train_end_date: str, version: str = "v0") -> str:
    """构造模型文件夹名"""
    return f"{symbol}_pred{train_label}_{train_end_date}_{version}"
