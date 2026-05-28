"""
保证金计算器
根据品种代码计算保证金，价格使用最近一年数据的95分位数
"""

import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import numpy as np

# 配置文件路径
CONFIG_PATH = Path("/home/future_config/basic_config/config_info.yaml")
DATA_DIR = Path("/mnt/Data/writable/liaoyuyang/data/1min/active")

# 缓存配置数据
_config_cache: Optional[Dict] = None
_price_cache: Dict[str, float] = {}
_first_trade_date_cache: Dict[str, str] = {}
_activity_cache: Dict[str, Dict] = {}


def _load_config() -> Dict:
    """加载配置文件"""
    global _config_cache
    if _config_cache is None:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            _config_cache = yaml.safe_load(f)
    return _config_cache


def _get_price_95th(symbol: str) -> float:
    """
    获取指定品种最近一年收盘价的95分位数
    
    Args:
        symbol: 品种代码，如 'AL', 'CU' 等
    
    Returns:
        95分位数价格
    """
    global _price_cache
    
    if symbol in _price_cache:
        return _price_cache[symbol]
    
    # CSV文件路径
    csv_file = DATA_DIR / f"main_{symbol}.csv"
    
    if not csv_file.exists():
        raise FileNotFoundError(f"找不到品种 {symbol} 的数据文件: {csv_file}")
    
    # 读取数据
    df = pd.read_csv(csv_file)
    
    # 确保ts列是datetime类型
    df['ts'] = pd.to_datetime(df['ts'])
    
    # 获取最近一年的数据
    end_date = df['ts'].max()
    start_date = end_date - pd.DateOffset(years=1)
    recent_data = df[df['ts'] >= start_date]
    
    if len(recent_data) == 0:
        raise ValueError(f"品种 {symbol} 没有最近一年的数据")
    
    # 计算收盘价的95分位数
    price_95th = recent_data['close'].quantile(0.95)
    
    # 缓存结果
    _price_cache[symbol] = price_95th
    
    return price_95th


def calculate_margin(symbol: str, lots: int = 1) -> float:
    """
    计算指定品种的保证金
    
    公式: 保证金 = 价格 × 乘数 × 保证金率 × 手数
    
    Args:
        symbol: 品种代码，如 'AL', 'CU', 'MA' 等
        lots: 手数，默认为1
    
    Returns:
        保证金金额（元）
    
    Example:
        >>> margin = calculate_margin('AL')  # 计算1手铝的保证金
        >>> margin = calculate_margin('CU', lots=2)  # 计算2手铜的保证金
    """
    # 加载配置
    config = _load_config()
    
    # 检查品种是否存在
    if symbol not in config['instruments']:
        raise ValueError(f"未知的品种代码: {symbol}")
    
    instrument = config['instruments'][symbol]
    
    # 获取参数
    multiplier = instrument['contract_multiplier']  # 合约乘数
    margin_rate = instrument['margin_rate']  # 保证金率
    
    # 获取95分位数价格
    price = _get_price_95th(symbol)
    
    # 计算保证金
    margin = price * multiplier * margin_rate * lots
    
    return margin


def get_instrument_info(symbol: str) -> Dict:
    """
    获取品种详细信息
    
    Args:
        symbol: 品种代码
    
    Returns:
        品种信息字典
    """
    config = _load_config()
    
    if symbol not in config['instruments']:
        raise ValueError(f"未知的品种代码: {symbol}")
    
    info = config['instruments'][symbol].copy()
    info['price_95th'] = _get_price_95th(symbol)
    info['margin_per_lot'] = calculate_margin(symbol)

    return info


def get_first_trade_date(symbol: str) -> str:
    """
    获取指定品种的第一个交易日期
    
    Args:
        symbol: 品种代码，如 'AL', 'CU' 等
    
    Returns:
        第一个交易日期，格式 'YYYY-MM-DD'
    
    Example:
        >>> first_date = get_first_trade_date('AL')  # 返回 '2021-02-08'
    """
    global _first_trade_date_cache
    
    if symbol in _first_trade_date_cache:
        return _first_trade_date_cache[symbol]
    
    # CSV文件路径
    csv_file = DATA_DIR / f"main_{symbol}.csv"
    
    if not csv_file.exists():
        raise FileNotFoundError(f"找不到品种 {symbol} 的数据文件: {csv_file}")
    
    # 读取数据（只读取ts列以提高效率）
    df = pd.read_csv(csv_file, usecols=['ts'])
    
    # 确保ts列是datetime类型
    df['ts'] = pd.to_datetime(df['ts'])
    
    # 获取第一个交易日期
    first_date = df['ts'].min().strftime('%Y-%m-%d')
    
    # 缓存结果
    _first_trade_date_cache[symbol] = first_date
    
    return first_date


def get_recent_activity(symbol: str) -> Dict:
    """
    获取指定品种最近一年的平均持仓量和成交量（用于判断活跃度）
    
    Args:
        symbol: 品种代码，如 'AL', 'CU' 等
    
    Returns:
        活跃度信息字典，包含:
        - avg_open_interest: 平均持仓量
        - avg_volume: 平均成交量
        - avg_turnover: 平均成交额
    
    Example:
        >>> activity = get_recent_activity('AL')
        >>> print(activity['avg_open_interest'])  # 平均持仓量
        >>> print(activity['avg_volume'])         # 平均成交量
    """
    global _activity_cache
    
    if symbol in _activity_cache:
        return _activity_cache[symbol]
    
    # CSV文件路径
    csv_file = DATA_DIR / f"main_{symbol}.csv"
    
    if not csv_file.exists():
        raise FileNotFoundError(f"找不到品种 {symbol} 的数据文件: {csv_file}")
    
    # 读取数据（只读取需要的列）
    df = pd.read_csv(csv_file, usecols=['ts', 'open_interest', 'volume', 'turnover'])
    
    # 确保ts列是datetime类型
    df['ts'] = pd.to_datetime(df['ts'])
    
    # 获取最近一年的数据
    end_date = df['ts'].max()
    start_date = end_date - pd.DateOffset(years=1)
    recent_data = df[df['ts'] >= start_date]
    
    if len(recent_data) == 0:
        raise ValueError(f"品种 {symbol} 没有最近一年的数据")
    
    # 计算平均值
    activity = {
        'avg_open_interest': recent_data['open_interest'].mean(),
        'avg_volume': recent_data['volume'].mean(),
        'avg_turnover': recent_data['turnover'].mean(),
        'data_points': len(recent_data)
    }
    
    # 缓存结果
    _activity_cache[symbol] = activity
    
    return activity


def clear_cache():
    """清除缓存，用于重新加载数据"""
    global _config_cache, _price_cache, _first_trade_date_cache, _activity_cache
    _config_cache = None
    _price_cache = {}
    _first_trade_date_cache = {}
    _activity_cache = {}


if __name__ == "__main__":
    # 测试示例
    test_symbols = ['AL', 'CU', 'MA', 'RB']
    
    print("=" * 60)
    print("保证金计算测试")
    print("=" * 60)
    
    for symbol in test_symbols:
        try:
            margin = calculate_margin(symbol)
            info = get_instrument_info(symbol)
            first_date = get_first_trade_date(symbol)
            activity = get_recent_activity(symbol)
            print(f"\n品种: {symbol} ({info.get('name', 'N/A')})")
            print(f"  首个交易日: {first_date}")
            print(f"  95分位数价格: {info['price_95th']:.2f}")
            print(f"  合约乘数: {info['contract_multiplier']}")
            print(f"  保证金率: {info['margin_rate']:.2%}")
            print(f"  1手保证金: {margin:,.2f} 元")
            print(f"  交易所: {info['exchange']}")
            print(f"  --- 活跃度指标 (最近一年) ---")
            print(f"  平均持仓量: {activity['avg_open_interest']:,.0f}")
            print(f"  平均成交量: {activity['avg_volume']:,.0f}")
            print(f"  平均成交额: {activity['avg_turnover']:,.0f} 元")
        except Exception as e:
            print(f"\n品种: {symbol}")
            print(f"  错误: {e}")
    
    print("\n" + "=" * 60)
