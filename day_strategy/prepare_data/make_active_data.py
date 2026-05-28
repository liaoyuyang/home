#!/usr/bin/env python3
"""
生成主力合约日线数据
- 按照主力合约表拼接日线数据
- 计算日收益率（close to close, open to open），在原始合约内计算避免拼接处问题
  - close_to_close: 当天收益率 (close_today - close_yesterday) / close_yesterday
  - open_to_open: 昨天的收益率，需要移动 (open_today - open_yesterday) / open_yesterday
- 输出到active文件夹
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 路径配置
DATA_DIR = Path('/mnt/Data/writable/liaoyuyang/data/1day')
FUTURE_INFO_DIR = Path('/mnt/Data/writable/liaoyuyang/data/future_info')
OUTPUT_DIR = DATA_DIR / 'active'

# 创建输出目录
OUTPUT_DIR.mkdir(exist_ok=True)


def get_all_symbols():
    """获取所有品种代码"""
    symbols = []
    for item in DATA_DIR.iterdir():
        if item.is_dir() and item.name != 'active':
            symbols.append(item.name)
    return sorted(symbols)


def load_main_instrument(symbol):
    """加载主力合约表"""
    main_file = FUTURE_INFO_DIR / symbol / 'main_instrument.csv'
    if not main_file.exists():
        return None
    
    df = pd.read_csv(main_file)
    df['trade_date'] = pd.to_datetime(df['trade_date']).dt.date
    return df


def load_contract_data(symbol, contract):
    """加载单个合约的日线数据"""
    contract_file = DATA_DIR / symbol / f'{contract}.feather'
    if not contract_file.exists():
        return None
    
    df = pd.read_feather(contract_file)
    df['trade_date'] = pd.to_datetime(df['trade_date']).dt.date
    return df


def calculate_returns(df):
    """
    计算日收益率
    - 在原始合约内计算，避免拼接处产生问题
    - close_to_close: (close_today - close_yesterday) / close_yesterday
    - open_to_open: (open_today - open_yesterday) / open_yesterday，需要移动到前一天
    - 过滤价格为0的数据，避免产生inf
    """
    df = df.copy()
    df = df.sort_values('trade_date').reset_index(drop=True)
    
    # 将价格为0的数据替换为NaN，避免计算inf
    df['close'] = df['close'].replace(0, np.nan)
    df['open'] = df['open'].replace(0, np.nan)
    
    # 按合约分组计算收益率
    df['close_to_close'] = df.groupby('instrument')['close'].pct_change()
    df['open_to_open'] = df.groupby('instrument')['open'].pct_change()
    
    # open_to_open是当天的open相对于昨天open的收益率，需要移动到前一天
    # 即：今天的open_to_open收益率对应的是昨天的收益
    df['open_to_open'] = df.groupby('instrument')['open_to_open'].shift(-1)
    
    return df


def build_active_data(symbol):
    """
    构建主力合约数据
    按照主力合约表，每天取对应的主力合约数据
    """
    print(f"Processing {symbol}...")
    
    # 加载主力合约表
    main_df = load_main_instrument(symbol)
    if main_df is None:
        print(f"  Warning: No main instrument file for {symbol}")
        return None
    
    # 加载所有合约数据
    contracts = main_df['instrument'].unique()
    contract_data = {}
    
    for contract in contracts:
        df = load_contract_data(symbol, contract)
        if df is not None:
            contract_data[contract] = df
    
    if not contract_data:
        print(f"  Warning: No contract data for {symbol}")
        return None
    
    # 按日期拼接主力数据
    active_rows = []
    
    for _, row in main_df.iterrows():
        trade_date = row['trade_date']
        instrument = row['instrument']
        
        if instrument not in contract_data:
            continue
        
        contract_df = contract_data[instrument]
        day_data = contract_df[contract_df['trade_date'] == trade_date]
        
        if len(day_data) > 0:
            active_rows.append(day_data.iloc[0].to_dict())
    
    if not active_rows:
        print(f"  Warning: No active data built for {symbol}")
        return None
    
    # 创建主力合约数据框
    active_df = pd.DataFrame(active_rows)
    active_df = active_df.sort_values('trade_date').reset_index(drop=True)
    
    # 计算收益率（在原始合约内计算）
    active_df = calculate_returns(active_df)
    
    return active_df


def main():
    """主函数"""
    symbols = get_all_symbols()
    print(f"Found {len(symbols)} symbols")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    success_count = 0
    fail_count = 0
    
    for symbol in symbols:
        try:
            active_df = build_active_data(symbol)
            
            if active_df is not None and len(active_df) > 0:
                # 保存为feather格式
                output_file = OUTPUT_DIR / f'{symbol}_active.feather'
                active_df.to_feather(output_file)
                print(f"  Saved {len(active_df)} rows to {output_file}")
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"  Error processing {symbol}: {e}")
            fail_count += 1
        print()
    
    print(f"\nCompleted: {success_count} succeeded, {fail_count} failed")


if __name__ == '__main__':
    main()
