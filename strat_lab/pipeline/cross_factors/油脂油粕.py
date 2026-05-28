"""
油脂油粕分组 — 跨品种因子计算脚本

此脚本为模板，展示如何为特定分组自定义跨品种因子逻辑。
默认逻辑与 GroupBuilder.compute_cross_factors 一致，可在此扩展。
"""

import numpy as np
import pandas as pd

# 品种列表（必须与 groups.yaml 中一致）
SYMBOLS = ["A", "M", "Y", "P"]


def compute(data_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    计算跨品种因子

    Parameters
    ----------
    data_dict : dict[str, pd.DataFrame]
        品种 -> 主力合约 1min 数据的字典

    Returns
    -------
    pd.DataFrame
        索引为时间，列为跨品种因子
    """
    n = len(SYMBOLS)
    df_cross = pd.DataFrame()

    for i in range(n - 1):
        for j in range(i + 1, n):
            vi, vj = SYMBOLS[i], SYMBOLS[j]
            di, dj = data_dict[vi], data_dict[vj]

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
            df_cross[f"{vi}_{vj}_vcorr10"] = di["volume"].rolling(10).corr(dj["volume"])

            # 量价相关性差
            df_cross[f"{vi}_{vj}_cvcorr10_diff"] = (
                di["close"].rolling(10).corr(di["volume"])
            ).sub(dj["close"].rolling(10).corr(dj["volume"]))

            # 持仓量变动差
            df_cross[f"{vi}_{vj}_oi5_diff"] = (
                di["open_interest"].pct_change(5)
            ).sub(dj["open_interest"].pct_change(5))

    return df_cross.replace([-np.inf, np.inf], np.nan)
