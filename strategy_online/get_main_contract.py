"""
主力合约查询工具

用法示例（作为脚本运行）：
    python3 get_main_contract.py

用法示例（作为模块导入）：
    from get_main_contract import get_main_contract

    # 查询单个日期、多个品种
    result = get_main_contract("2024-03-15", ["A", "P", "IC"])
    print(result["A"]["main_contract"])   # 输出: a2405
    print(result["A"]["end_date"])        # 输出: 2024-04-19

日期支持多种格式：
    - "2024-03-15"
    - "20240315"
    - "2024/03/15"
    - datetime 对象
"""

import os
import csv
from datetime import datetime
from typing import List, Dict, Optional, Union

# 主力合约表存放目录（每个品种一个子文件夹）
FUTURE_INFO_DIR = "/mnt/Data/writable/liaoyuyang/data/future_info"


def parse_date(date_input: Union[str, datetime]) -> str:
    """统一将日期输入转为 'YYYY-MM-DD' 字符串。"""
    if isinstance(date_input, datetime):
        return date_input.strftime("%Y-%m-%d")
    date_str = str(date_input).strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return date_str


def _norm_header(name: str) -> str:
    """表头标准化：去空格、去引号、变小写。"""
    return name.strip().strip('"').strip("'").lower()


def load_main_contract(variety: str) -> Optional[Dict[str, dict]]:
    """
    加载单个品种的主力合约表，返回以 trade_date 为 key 的字典。
    每个 value 包含 instrument 和 end_date。
    """
    file_path = os.path.join(FUTURE_INFO_DIR, variety.upper(), "main_instrument.csv")
    if not os.path.exists(file_path):
        return None

    data = {}
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            raw_header = next(reader)
        except StopIteration:
            return None

        # 标准化表头
        header = [_norm_header(h) for h in raw_header]

        # 查找关键列索引
        def idx(col_name: str, alternatives: Optional[List[str]] = None) -> int:
            candidates = [col_name]
            if alternatives:
                candidates += alternatives
            for c in candidates:
                if c in header:
                    return header.index(c)
            return -1

        # 日期列：可能是第一列（unnamed index），也可能是明确的 trade_date / date
        date_idx = idx("trade_date", ["date"])
        if date_idx == -1:
            # 兜底：如果第一列没名字（空字符串），且后面有 instrument，就把第一列当日期
            if header[0] == "" and len(header) > 1:
                date_idx = 0
            else:
                return None

        # 主力合约列
        inst_idx = idx("instrument", ["contract", "main_contract"])
        if inst_idx == -1:
            return None

        # 结束日期列（非必须）
        end_idx = idx("end_date", ["enddate"])

        for row in reader:
            if not row:
                continue
            if date_idx >= len(row):
                continue

            trade_date_raw = row[date_idx].strip()
            if not trade_date_raw:
                continue

            # 统一日期格式
            trade_date = parse_date(trade_date_raw)

            instrument = row[inst_idx].strip() if inst_idx < len(row) else None
            end_date = None
            if end_idx != -1 and end_idx < len(row):
                end_raw = row[end_idx].strip()
                if end_raw:
                    end_date = parse_date(end_raw)

            data[trade_date] = {
                "instrument": instrument,
                "end_date": end_date,
            }

    return data


def get_main_contract(trade_date: Union[str, datetime], varieties: List[str]) -> Dict[str, dict]:
    """
    查询指定日期和品种列表的主力合约信息。

    Parameters
    ----------
    trade_date : str or datetime
        查询日期，例如 "2024-03-15"、20240315、datetime 对象。
    varieties : List[str]
        品种代码列表，例如 ["A", "P", "IC"]。

    Returns
    -------
    Dict[str, dict]
        每个品种对应一个字典：
        - variety       : 品种代码
        - trade_date    : 查询日期
        - main_contract : 当天主力合约代码（查不到为 None）
        - end_date      : 该主力合约结束当主力的日期（None 表示尚未结束或数据缺失）
        - error         : 出错时包含错误信息
    """
    query_date = parse_date(trade_date)
    results = {}

    for v in varieties:
        variety = v.upper().strip()
        cache = load_main_contract(variety)
        if cache is None:
            results[variety] = {
                "variety": variety,
                "trade_date": query_date,
                "error": f"未找到品种 {variety} 的主力合约数据"
            }
            continue

        record = cache.get(query_date)
        if record is None:
            results[variety] = {
                "variety": variety,
                "trade_date": query_date,
                "error": f"品种 {variety} 在日期 {query_date} 无数据"
            }
            continue

        results[variety] = {
            "variety": variety,
            "trade_date": query_date,
            "main_contract": record["instrument"],
            "end_date": record["end_date"],
        }

    return results


def print_results(results: Dict[str, dict]):
    """美观地打印查询结果。"""
    print(f"\n{'品种':<8}{'查询日期':<12}{'主力合约':<14}{'结束日期':<12}{'备注'}")
    print("-" * 70)
    for info in results.values():
        if "error" in info:
            print(f"{info['variety']:<8}{info['trade_date']:<12}{'N/A':<14}{'N/A':<12}{info['error']}")
        else:
            end = info['end_date'] if info['end_date'] else "至今/未记录"
            print(f"{info['variety']:<8}{info['trade_date']:<12}{info['main_contract']:<14}{end:<12}")
    print()


# ==================== 示例用法 ====================
if __name__ == "__main__":
    # 示例 1：查几个商品期货+股指
    date = "2025-03-24"
    varieties = ["A", "B", "C", "CS", "M", "Y", "P", "LH"]
    results = get_main_contract(date, varieties)
    print_results(results)

    # # 示例 2：查另一个日期
    # date2 = "2021-04-21"
    # varieties2 = ["A", "P"]
    # results2 = get_main_contract(date2, varieties2)
    # print_results(results2)
