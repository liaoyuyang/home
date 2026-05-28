#!/usr/bin/env python3
"""
Factor Trigger Logger — 实时因子计算数据捕获

每分钟 trigger 存一个文件夹，内含该次计算用到的所有源数据表。
文件夹名 = 市场时间（valid_index[-1]）
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

LOG_ROOT = Path('/home/strategy_PAMY_dev/logs/factor_debug')


class FactorTriggerLogger:
    """
    单品种单因子实时数据捕获器。
    """

    def __init__(self, symbol: str = None, target_factor: str = None):
        self.symbol = (symbol or '').upper()
        self.target_factor = target_factor or ''
        self.enabled = bool(self.symbol and self.target_factor)

    def log(self, factor_name: str, fac_generator, result, data_dict=None):
        """
        在 generate_factor_dataframe 中调用。

        参数:
            factor_name: 当前因子列名（如 RPP_5D）
            fac_generator: Factor_generator 实例
            result: 该因子本次计算的结果
            data_dict: strategies.py 中的 data_dict（含跨品种 min）
        """
        if not self.enabled:
            return
        if factor_name != self.target_factor:
            return

        # 市场时间 = valid_index 最后一个点
        if hasattr(fac_generator, 'valid_index') and len(fac_generator.valid_index) > 0:
            trigger_time = pd.Timestamp(fac_generator.valid_index[-1])
        else:
            trigger_time = pd.Timestamp.now()

        ts_str = trigger_time.strftime('%Y%m%d_%H%M%S')
        log_dir = LOG_ROOT / self.symbol / ts_str
        log_dir.mkdir(parents=True, exist_ok=True)

        # 1. 极简 info.json（仅市场时间和基本配置）
        info = {
            'trigger_time': trigger_time.isoformat(),
            'symbol': self.symbol,
            'factor': self.target_factor,
            'dict_keys': getattr(fac_generator, 'dict_keys', None),
        }
        with open(log_dir / 'info.json', 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False, default=str)

        # 2. 保存主品种数据
        if hasattr(fac_generator, 'tick_data') and fac_generator.tick_data is not None:
            fac_generator.tick_data.to_parquet(log_dir / 'tick.parquet')

        if hasattr(fac_generator, 'min_data') and fac_generator.min_data is not None:
            fac_generator.min_data.to_parquet(log_dir / 'min.parquet')

        if hasattr(fac_generator, 'valid_index') and fac_generator.valid_index is not None:
            pd.Series(fac_generator.valid_index).to_csv(log_dir / 'valid_index.csv', index=False, header=['datetime'])

        # 3. 保存跨品种 min 数据
        if data_dict is not None and hasattr(fac_generator, 'dict_keys'):
            for sym in fac_generator.dict_keys:
                key = f'{sym}_min_data_concat'
                if key in data_dict and data_dict[key] is not None:
                    data_dict[key].to_parquet(log_dir / f'min_{sym}.parquet')

        # 4. 保存结果
        if isinstance(result, np.ndarray):
            pd.Series(result, name=factor_name).to_csv(log_dir / 'result.csv', index=False)
        elif hasattr(result, 'to_csv'):
            result.to_csv(log_dir / 'result.csv')
        else:
            pd.Series([result], name=factor_name).to_csv(log_dir / 'result.csv', index=False)


def make_logger(symbol: str, target_factor: str):
    """创建一个已配置好的 logger 实例。"""
    return FactorTriggerLogger(symbol=symbol, target_factor=target_factor)
