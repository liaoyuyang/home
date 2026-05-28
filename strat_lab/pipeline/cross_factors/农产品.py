"""
农产品分组 — 跨品种因子计算脚本

品种列表: A, B, C, CS, M, Y, P, LH
默认逻辑与油脂油粕一致，可在此扩展农产品特有的跨品种因子。
"""

import numpy as np
import pandas as pd

SYMBOLS = ["A", "B", "C", "CS", "M", "Y", "P", "LH"]


def compute(data_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """计算跨品种因子"""
    n = len(SYMBOLS)
    df_cross = pd.DataFrame()

    for i in range(n - 1):
        for j in range(i + 1, n):
            vi, vj = SYMBOLS[i], SYMBOLS[j]
            di, dj = data_dict[vi], data_dict[vj]

            df_cross[f"{vi}_{vj}_closepctchg5_sub"] = di["close"].pct_change(5).sub(
                dj["close"].pct_change(5)
            )
            df_cross[f"{vi}_{vj}_closepctchg20_sub"] = di["close"].pct_change(20).sub(
                dj["close"].pct_change(20)
            )
            df_cross[f"{vi}_{vj}_volumediv5_diff5"] = (
                di["volume"].rolling(5).sum().div(dj["volume"].rolling(5).sum())
            ).diff(5)
            df_cross[f"{vi}_{vj}_volumediv20_diff5"] = (
                di["volume"].rolling(20).sum().div(dj["volume"].rolling(20).sum())
            ).diff(5)
            df_cross[f"{vi}_{vj}_vcorr10"] = di["volume"].rolling(10).corr(dj["volume"])
            df_cross[f"{vi}_{vj}_cvcorr10_diff"] = (
                di["close"].rolling(10).corr(di["volume"])
            ).sub(dj["close"].rolling(10).corr(dj["volume"]))
            df_cross[f"{vi}_{vj}_oi5_diff"] = (
                di["open_interest"].pct_change(5)
            ).sub(dj["open_interest"].pct_change(5))

    return df_cross.replace([-np.inf, np.inf], np.nan)
