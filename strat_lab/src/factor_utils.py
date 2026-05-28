"""
因子工具函数
降频、对齐、缺失处理等通用操作
"""

import numpy as np
import pandas as pd


def time_scale_df(
    df: pd.DataFrame,
    time_col: str = "ts",
    trading_hours: list[str] | None = None,
) -> pd.DataFrame:
    """
    按交易时段过滤数据
    （复用现有 DataLoader.time_scale_df 的逻辑，简化版）

    Parameters
    ----------
    df : pd.DataFrame
    time_col : str
        时间列名，分钟级数据默认用 'ts'
    trading_hours : list[str]
        如 ["09:00-11:30", "13:30-15:00", "21:00-23:00"]
    """
    if trading_hours is None:
        trading_hours = ["09:00-11:30", "13:30-15:00", "21:00-23:00"]

    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])

    masks = []
    for period in trading_hours:
        start, end = period.split("-")
        start_t = pd.to_datetime(start).time()
        end_t = pd.to_datetime(end).time()
        t = df[time_col].dt.time
        if start_t < end_t:
            masks.append((t >= start_t) & (t <= end_t))
        else:
            # 跨午夜（如 21:00-01:00）
            masks.append((t >= start_t) | (t <= end_t))

    if masks:
        df = df[np.any(masks, axis=0)]

    return df.reset_index(drop=True)


def align_factor_index(
    *dfs: pd.DataFrame,
    index_col: str = "datetime",
) -> list[pd.DataFrame]:
    """
    对齐多个 DataFrame 的索引，取交集

    Returns
    -------
    list[pd.DataFrame]
        对齐后的 DataFrame 列表
    """
    common_index = None
    for df in dfs:
        idx = df.index if index_col in [df.index.name, None] else df.set_index(index_col).index
        if common_index is None:
            common_index = idx
        else:
            common_index = common_index.intersection(idx)

    result = []
    for df in dfs:
        if index_col in df.columns:
            df = df.set_index(index_col)
        result.append(df.loc[common_index].reset_index())

    return result


def drop_extreme_values(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    method: str = "mad",
    n: float = 3.0,
) -> pd.DataFrame:
    """
    极端值处理

    Parameters
    ----------
    df : pd.DataFrame
    cols : list[str], optional
        要处理的列，None 则处理所有数值列
    method : str
        'mad' | 'std' | 'percentile'
    n : float
        阈值倍数
    """
    df = df.copy()
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in cols:
        if method == "mad":
            median = df[col].median()
            mad = (df[col] - median).abs().median()
            mask = (df[col] - median).abs() <= n * mad
        elif method == "std":
            mean = df[col].mean()
            std = df[col].std()
            mask = (df[col] - mean).abs() <= n * std
        elif method == "percentile":
            low, high = df[col].quantile([0.01, 0.99])
            mask = (df[col] >= low) & (df[col] <= high)
        else:
            raise ValueError(f"不支持的方法: {method}")

        df.loc[~mask, col] = np.nan

    return df


def compute_forward_return(
    price: pd.Series,
    periods: list[int] = [1, 5, 10],
    price_col: str = "tick6t60avg",
) -> pd.DataFrame:
    """
    计算未来收益率（与 prepare_mkt_data 中的逻辑一致）

    rtn_period = price.shift(-(1 + period)) / price.shift(-1) - 1
    """
    results = {}
    for period in periods:
        results[f"rtn_{period}"] = price.shift(-(1 + period)) / price.shift(-1) - 1
    return pd.DataFrame(results)
