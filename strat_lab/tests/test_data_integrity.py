"""
数据完整性测试
同步后运行，检查主力合约数据是否存在、时间是否连续、合约切换是否正确
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.config_manager import ConfigManager
from src.data_loader import DataLoader


def test_main_1min_integrity(symbol: str) -> dict:
    """检查主力合约 1min 数据完整性"""
    dl = DataLoader()
    errors = []
    warnings_list = []

    try:
        df = dl.load_main_1min(symbol, from_local=True)
    except FileNotFoundError as e:
        return {"symbol": symbol, "status": "FAIL", "errors": [str(e)], "warnings": []}

    # 检查时间列
    if df.index.name != "datetime":
        errors.append("索引名不是 datetime")

    # 检查必要列
    required_cols = {"open", "high", "low", "close", "volume", "turnover"}
    missing = required_cols - set(df.columns)
    if missing:
        errors.append(f"缺失必要列: {missing}")

    # 检查时间连续性（同一天内不应有缺失分钟）
    df["date"] = df.index.date
    for date, group in df.groupby("date"):
        # 简单检查：同一天内分钟数是否异常少
        if len(group) < 100:
            warnings_list.append(f"{date} 只有 {len(group)} 条数据，可能缺失")

    # 检查 close 是否为 0 或 NaN
    zero_close = (df["close"] == 0).sum()
    nan_close = df["close"].isna().sum()
    if zero_close > 0:
        warnings_list.append(f"close=0 的行数: {zero_close}")
    if nan_close > 0:
        warnings_list.append(f"close=NaN 的行数: {nan_close}")

    status = "FAIL" if errors else ("WARN" if warnings_list else "PASS")
    return {
        "symbol": symbol,
        "status": status,
        "shape": df.shape,
        "errors": errors,
        "warnings": warnings_list,
    }


def test_contract_info(symbol: str) -> dict:
    """检查合约切换表"""
    dl = DataLoader()
    try:
        df = dl.load_contract_info(symbol, from_local=True)
    except FileNotFoundError as e:
        return {"symbol": symbol, "status": "FAIL", "errors": [str(e)], "warnings": []}

    errors = []
    if "instrument" not in df.columns:
        errors.append("缺少 instrument 列")
    if "trade_date" not in df.columns:
        errors.append("缺少 trade_date 列")

    # 检查合约切换是否单调不递减
    if "instrument" in df.columns:
        instruments = df["instrument"].tolist()
        for i in range(1, len(instruments)):
            if instruments[i] < instruments[i - 1]:
                errors.append(f"合约切换回退: {instruments[i-1]} -> {instruments[i]} at row {i}")

    status = "FAIL" if errors else "PASS"
    return {
        "symbol": symbol,
        "status": status,
        "shape": df.shape,
        "errors": errors,
        "warnings": [],
    }


def main():
    cm = ConfigManager()
    symbols = cm.list_symbols()

    print("=" * 60)
    print("数据完整性测试")
    print("=" * 60)

    for symbol in symbols:
        r1 = test_main_1min_integrity(symbol)
        r2 = test_contract_info(symbol)

        status = "PASS"
        if r1["status"] == "FAIL" or r2["status"] == "FAIL":
            status = "FAIL"
        elif r1["status"] == "WARN" or r2["status"] == "WARN":
            status = "WARN"

        print(f"\n[{symbol}] {status}")
        print(f"  main_1min: {r1.get('shape', 'N/A')}, errors={len(r1['errors'])}, warns={len(r1['warnings'])}")
        print(f"  contract_info: {r2.get('shape', 'N/A')}, errors={len(r2['errors'])}")

        for e in r1["errors"] + r2["errors"]:
            print(f"    ERROR: {e}")
        for w in r1["warnings"]:
            print(f"    WARN: {w}")


if __name__ == "__main__":
    main()
