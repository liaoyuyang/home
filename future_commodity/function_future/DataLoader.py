import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union
from function_future.date_selection import get_trading_days
from datetime import time,datetime,timedelta
from datetime import datetime
import json
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed  # 用于并行处理
from functools import lru_cache
from itertools import product
import yaml
import pandas as pd

# ===============================一些工具函数========================================
def round_datetime_to_minute(df, datetime_col):
    df[f"{datetime_col}_rounded"] = df[datetime_col].dt.ceil('min')
    return df[f'{datetime_col}_rounded']

def is_valid_date(date_str: str) -> bool:
    """检查日期字符串是否有效"""
    try:
        datetime.strptime(date_str, '%Y%m%d')
        return True
    except ValueError:
        return False

def convert_timestamp_with_offset(timestamp_series):
    """
    转换时间戳，并仅对秒级（s）添加 500ms 偏移量
    """
    max_value = timestamp_series.max()
    
    if max_value < 1e10:  # 秒级（s）
        base_time = pd.to_datetime(timestamp_series, unit='s')
        # 仅对秒级数据添加 500ms 偏移
        offset = pd.to_timedelta((timestamp_series.groupby(timestamp_series).cumcount() - 1) * 500, 'ms')
        return base_time + offset
    elif max_value < 1e13:  # 毫秒级（ms）
        return pd.to_datetime(timestamp_series, unit='ms')
    elif max_value < 1e16:  # 微秒级（us）
        return pd.to_datetime(timestamp_series, unit='us')
    else:  # 纳秒级（ns）
        return pd.to_datetime(timestamp_series, unit='ns')

# 商品期货数据聚合函数
def process_time_column_vectorized_floor(df, time_column='trade_time'):
    """
    向量化版本，将毫秒向上取整到最近的0.25秒（向前调整）
    """
    # 分离秒和毫秒部分
    time_parts = df[time_column].str.split('.', expand=True)
    
    # 检查是否有毫秒部分（即分割后是否有第二列）
    if time_parts.shape[1] < 2:
        # 没有时间小数部分，所有时间都是整秒
        seconds = time_parts[0]
        milliseconds = pd.Series(0, index=df.index)
    else:
        seconds = time_parts[0]
        milliseconds = time_parts[1].fillna('000').astype(int)
    
    # 向上取整到最近的250毫秒（只向后调整）
    rounded_ms = np.floor(milliseconds / 250) * 250
    
    # 处理进位（如果向上取整后达到1000毫秒）
    carry_over = (rounded_ms >= 1000).astype(int)
    rounded_ms = np.where(rounded_ms >= 1000, rounded_ms - 1000, rounded_ms)
    
    # 处理秒的进位
    def adjust_seconds(sec_str, carry):
        if carry == 0:
            return sec_str
        
        # 解析时间并增加一秒
        time_obj = datetime.strptime(sec_str, '%H:%M:%S')
        time_obj += timedelta(seconds=1)
        return time_obj.strftime('%H:%M:%S')
    
    # 应用进位调整
    adjusted_seconds = [adjust_seconds(sec, carry) for sec, carry in zip(seconds, carry_over)]
    
    # 重新组合时间字符串
    df['rounded_time'] = [
        f"{sec}.{int(ms):03d}" if ms > 0 else sec 
        for sec, ms in zip(adjusted_seconds, rounded_ms)
    ]
    
    return df

def process_datetime(df, dates=None):
    if not dates:
        dates = sorted([str(x) for x in get_trading_days("2021-02-01")])
    
    date_to_prev_date = {dates[i]: dates[i-1] for i in range(1, len(dates))}
    
    def get_adjusted_date(row):
        if (row['period'] == 'night') and (row['trade_time'] > '15:00:00.000'):
            return date_to_prev_date.get(row['trade_date'], row['trade_date'])
        return row['trade_date']
    
    df['adjusted_date'] = df.apply(get_adjusted_date, axis=1)
    
    def combine_datetime(row):
        time_part = row['rounded_time']
        if '.' not in time_part:
            time_part = time_part + '.000'
        return f"{row['adjusted_date']} {time_part}"
    
    df['datetime'] = pd.to_datetime(
        df.apply(combine_datetime, axis=1),
        format='%Y-%m-%d %H:%M:%S.%f',
        errors='coerce'  # 如果格式不匹配则转为 NaT
    )
    df.drop(['adjusted_date', 'rounded_time'], axis=1, inplace=True)
    df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-4]

    return df

def df_is_trading_time(df: pd.DataFrame, tscol_name: str, trade_type: list):
    df[tscol_name] = pd.to_datetime(df[tscol_name])
    df['is_trading'] = False

    if trade_type == ["09:30-11:30", "13:00-15:00"]:
        df['is_trading'] = (
            df[tscol_name].dt.time.between(time(9, 30, 0, 1), time(11, 30, 0)) |
            df[tscol_name].dt.time.between(time(13, 0, 0, 1), time(15, 0, 0))
        )

    elif trade_type == ["09:00-11:30", "13:30-15:00", "21:00-23:00"]:
        df['is_trading'] = (
            df[tscol_name].dt.time.between(time(9, 0, 0, 1), time(10, 15, 0)) |
            df[tscol_name].dt.time.between(time(10, 30, 0, 1), time(11, 30, 0)) |
            df[tscol_name].dt.time.between(time(13, 30, 0, 1), time(15, 0, 0)) |
            df[tscol_name].dt.time.between(time(21, 0, 0, 1), time(23, 0, 0))
        )

    elif trade_type == ["09:00-11:30", "13:30-15:00"]:
        df['is_trading'] = (
            df[tscol_name].dt.time.between(time(9, 0, 0, 1), time(10, 15, 0)) |
            df[tscol_name].dt.time.between(time(10, 30, 0, 1), time(11, 30, 0)) |
            df[tscol_name].dt.time.between(time(13, 30, 0, 1), time(15, 0, 0)) 
        )

    elif trade_type == ["09:00-11:30", "13:30-15:00", "21:00-01:00"]:
        df['is_trading'] = (
            df[tscol_name].dt.time.between(time(9, 0, 0, 1), time(10, 15, 0)) |
            df[tscol_name].dt.time.between(time(10, 30, 0, 1), time(11, 30, 0)) |
            df[tscol_name].dt.time.between(time(13, 30, 0, 1), time(15, 0, 0)) |
            df[tscol_name].dt.time.between(time(21, 0, 0, 1), time(23, 59, 59, 9999)) |
            df[tscol_name].dt.time.between(time(0, 0, 0, 0), time(1, 0, 0))
        )


    elif trade_type == ["09:00-11:30", "13:30-15:00", "21:00-02:30"]:
        df['is_trading'] = (
            df[tscol_name].dt.time.between(time(9, 0, 0, 1), time(10, 15, 0)) |
            df[tscol_name].dt.time.between(time(10, 30, 0, 1), time(11, 30, 0)) |
            df[tscol_name].dt.time.between(time(13, 30, 0, 1), time(15, 0, 0)) |
            (df[tscol_name].dt.time >= time(21, 0, 0, 1)) |  # 21:00之后
            (df[tscol_name].dt.time <= time(2, 30, 0))       # 02:30之前
        )

    else: 
        print('time not typical, pass')
        
    return df

def time_scale_df(df: pd.DataFrame, tscol_name: str, trade_type: list):
    df[tscol_name] = pd.to_datetime(df[tscol_name])
    if trade_type == ["09:30-11:30", "13:00-15:00"]:
        df = df[
            (df[tscol_name].dt.time.between(time(9, 30, 0, 1000), time(11, 30, 0))) |
            (df[tscol_name].dt.time.between(time(13, 0, 0, 1000), time(15, 0, 0)))
        ]

    elif trade_type == ["09:00-11:30", "13:30-15:00", "21:00-23:00"]:
        df = df[
            (df[tscol_name].dt.time.between(time(9, 0, 0, 1000), time(10, 15, 0))) |
            (df[tscol_name].dt.time.between(time(10, 30, 0, 1000), time(11, 30, 0))) |
            (df[tscol_name].dt.time.between(time(13, 30, 0, 1000), time(15, 0, 0))) |
            (df[tscol_name].dt.time.between(time(21, 0, 0, 1000), time(23, 0, 0)))
        ]

    elif trade_type == ["09:00-11:30", "13:30-15:00"]:
        df = df[
            (df[tscol_name].dt.time.between(time(9, 0, 0, 1000), time(10, 15, 0))) |
            (df[tscol_name].dt.time.between(time(10, 30, 0, 1000), time(11, 30, 0))) |
            (df[tscol_name].dt.time.between(time(13, 30, 0, 1000), time(15, 0, 0))) 
        ]

    elif trade_type == ["09:00-11:30", "13:30-15:00", "21:00-01:00"]:
        df = df[
            (df[tscol_name].dt.time.between(time(9, 0, 0, 1000), time(10, 15, 0))) |
            (df[tscol_name].dt.time.between(time(10, 30, 0, 1000), time(11, 30, 0))) |
            (df[tscol_name].dt.time.between(time(13, 30, 0, 1000), time(15, 0, 0))) |
            (df[tscol_name].dt.time.between(time(21, 0, 0, 1000), time(23, 59, 59, 9999))) |
            (df[tscol_name].dt.time.between(time(0, 0, 0, 0), time(1, 0, 0)))
        ]

    elif trade_type == ["09:00-11:30", "13:30-15:00", "21:00-02:30"]:
        df = df[
            (df[tscol_name].dt.time.between(time(9, 0, 0, 1000), time(10, 15, 0))) |
            (df[tscol_name].dt.time.between(time(10, 30, 0, 1000), time(11, 30, 0))) |
            (df[tscol_name].dt.time.between(time(13, 30, 0, 1000), time(15, 0, 0))) |
            (df[tscol_name].dt.time.between(time(21, 0, 0, 1000), time(23, 59, 59, 9999))) |
            (df[tscol_name].dt.time.between(time(0, 0, 0, 0), time(2, 30, 0)))
        ]
    
    else: print('time not typical, pass')
    
    return df

class InstrumentConfig:
    def __init__(self, config_path: str = "/home/future_config/basic_config/config_info.yaml"):
        """
        初始化配置加载器
        :param config_path: YAML 配置文件路径
        """
        self.config_path = config_path
        self.data = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """加载 YAML 配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"配置文件 {self.config_path} 不是有效的 YAML 格式: {e}")

    def get_instrument_config(self, symbol: Union[str, List[str]]) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        获取指定品种的配置
        
        参数:
        ----------
        symbol : str 或 List[str]
            品种代码（如 "IC"）或品种代码列表（如 ["IC", "IF", "IH"]）
        
        返回:
        ----------
        Union[Dict[str, Any], pd.DataFrame]:
            - 如果symbol是字符串: 返回包含配置的字典
            - 如果symbol是列表: 返回包含所有品种配置的DataFrame
        """
        if isinstance(symbol, str):
            instruments = self.data.get("instruments", {})
            if symbol not in instruments:
                raise KeyError(f"品种 {symbol} 不存在于配置文件中")
        
            # 合并默认配置和品种特定配置（品种配置优先）
            default_config = self.data.get("default_config", {})
            instrument_config = instruments[symbol]
            return {**default_config, **instrument_config}
        
        elif isinstance(symbol, list):
            # 为每个品种获取配置
            configs = []
            for sym in symbol:
                try:
                    config = self.get_instrument_config(sym)  # 递归调用自身
                    # 添加品种代码到配置中
                    config['symbol'] = sym
                    configs.append(config)
                except KeyError as e:
                    # 如果品种不存在，可以选择跳过或记录错误
                    print(f"警告: {e}")
                    continue
            
            if not configs:
                raise ValueError("没有找到任何有效的品种配置")
            
            return configs
            # 转换为DataFrame
            df = pd.DataFrame(configs)
            
            # 将'symbol'列放在第一列
            cols = ['symbol'] + [col for col in df.columns if col != 'symbol']
            df = df[cols].set_index('symbol')
            
            return df.T
        
        else:
            raise TypeError(f"symbol参数必须是str或list，但传入的是{type(symbol).__name__}")

    def list_all_instruments(self) -> list:
        """返回所有可用的品种列表"""
        return list(self.data.get("instruments", {}).keys())
    
    def df_cut_time(self, df, trading_hours, cut=10):
        if trading_hours == ['09:00-11:30', '13:30-15:00', '21:00-23:00']:
            # 直接硬编码要移除的时间
            mask = pd.Series(True, index=df.index)
            
            for date in df.index.normalize().unique():
                # 上午开盘后cut分钟
                t1 = pd.Timestamp.combine(date, time(9, 0))
                t2 = t1 + pd.Timedelta(minutes=cut)
                mask = mask & ~((df.index >= t1) & (df.index <= t2))
                
                # # 上午收盘前cut分钟
                # t3 = pd.Timestamp.combine(date, time(11, 30)) - pd.Timedelta(minutes=cut)
                # t4 = pd.Timestamp.combine(date, time(11, 30))
                # mask = mask & ~((df.index >= t3) & (df.index <= t4))
                
                # # 下午开盘后cut分钟
                # t5 = pd.Timestamp.combine(date, time(13, 30))
                # t6 = t5 + pd.Timedelta(minutes=cut)
                # mask = mask & ~((df.index >= t5) & (df.index <= t6))
                
                # 下午收盘前cut分钟
                t7 = pd.Timestamp.combine(date, time(15, 0)) - pd.Timedelta(minutes=cut)
                t8 = pd.Timestamp.combine(date, time(15, 0))
                mask = mask & ~((df.index >= t7) & (df.index <= t8))
                
                # 晚上开盘后cut分钟
                t9 = pd.Timestamp.combine(date, time(21, 0))
                t10 = t9 + pd.Timedelta(minutes=cut)
                mask = mask & ~((df.index >= t9) & (df.index <= t10))
                
                # 晚上收盘前cut分钟
                t11 = pd.Timestamp.combine(date, time(23, 0)) - pd.Timedelta(minutes=cut)
                t12 = pd.Timestamp.combine(date, time(23, 0))
                mask = mask & ~((df.index >= t11) & (df.index <= t12))
            
            return df[mask]

        elif trading_hours == ['09:00-11:30', '13:30-15:00']:
            # 直接硬编码要移除的时间
            mask = pd.Series(True, index=df.index)
            
            for date in df.index.normalize().unique():
                # 上午开盘后cut分钟
                t1 = pd.Timestamp.combine(date, time(9, 0))
                t2 = t1 + pd.Timedelta(minutes=cut)
                mask = mask & ~((df.index >= t1) & (df.index <= t2))
                
                # # 上午收盘前cut分钟
                # t3 = pd.Timestamp.combine(date, time(11, 30)) - pd.Timedelta(minutes=cut)
                # t4 = pd.Timestamp.combine(date, time(11, 30))
                # mask = mask & ~((df.index >= t3) & (df.index <= t4))
                
                # # 下午开盘后cut分钟
                # t5 = pd.Timestamp.combine(date, time(13, 30))
                # t6 = t5 + pd.Timedelta(minutes=cut)
                # mask = mask & ~((df.index >= t5) & (df.index <= t6))
                
                # 下午收盘前cut分钟
                t7 = pd.Timestamp.combine(date, time(15, 0)) - pd.Timedelta(minutes=cut)
                t8 = pd.Timestamp.combine(date, time(15, 0))
                mask = mask & ~((df.index >= t7) & (df.index <= t8))
            
            return df[mask]

        elif trading_hours == ['09:00-11:30', '13:30-15:00', '21:00-02:30']:
            # 直接硬编码要移除的时间
            mask = pd.Series(True, index=df.index)
            
            start_date = df.index.min().normalize()
            end_date = df.index.max().normalize()
            
            all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            for date in all_dates:
                # 上午开盘后cut分钟
                t1 = pd.Timestamp.combine(date.date(), time(9, 0))
                t2 = t1 + pd.Timedelta(minutes=cut)
                mask = mask & ~((df.index >= t1) & (df.index <= t2))
                
                # 上午收盘前cut分钟
                t3 = pd.Timestamp.combine(date.date(), time(11, 30)) - pd.Timedelta(minutes=cut)
                t4 = pd.Timestamp.combine(date.date(), time(11, 30))
                mask = mask & ~((df.index >= t3) & (df.index <= t4))
                
                # 下午开盘后cut分钟
                t5 = pd.Timestamp.combine(date.date(), time(13, 30))
                t6 = t5 + pd.Timedelta(minutes=cut)
                mask = mask & ~((df.index >= t5) & (df.index <= t6))
                
                # 下午收盘前cut分钟
                t7 = pd.Timestamp.combine(date.date(), time(15, 0)) - pd.Timedelta(minutes=cut)
                t8 = pd.Timestamp.combine(date.date(), time(15, 0))
                mask = mask & ~((df.index >= t7) & (df.index <= t8))
                
                # 晚上开盘后cut分钟（当天21:00-21:10）
                t9 = pd.Timestamp.combine(date.date(), time(21, 0))
                t10 = t9 + pd.Timedelta(minutes=cut)
                mask = mask & ~((df.index >= t9) & (df.index <= t10))
                
                # 次日凌晨收盘前cut分钟（次日02:20-02:30）
                next_date = date + pd.Timedelta(days=1)
                t11 = pd.Timestamp.combine(next_date.date(), time(2, 30)) - pd.Timedelta(minutes=cut)
                t12 = pd.Timestamp.combine(next_date.date(), time(2, 30))
                mask = mask & ~((df.index >= t11) & (df.index <= t12))
            
            return df[mask]

        elif trading_hours == ['09:00-11:30', '13:30-15:00', '21:00-01:00']:
            # 直接硬编码要移除的时间
            mask = pd.Series(True, index=df.index)
            
            start_date = df.index.min().normalize()
            end_date = df.index.max().normalize()
            
            all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            for date in all_dates:
                # 上午开盘后cut分钟
                t1 = pd.Timestamp.combine(date.date(), time(9, 0))
                t2 = t1 + pd.Timedelta(minutes=cut)
                mask = mask & ~((df.index >= t1) & (df.index <= t2))
                
                # 上午收盘前cut分钟
                t3 = pd.Timestamp.combine(date.date(), time(11, 30)) - pd.Timedelta(minutes=cut)
                t4 = pd.Timestamp.combine(date.date(), time(11, 30))
                mask = mask & ~((df.index >= t3) & (df.index <= t4))
                
                # 下午开盘后cut分钟
                t5 = pd.Timestamp.combine(date.date(), time(13, 30))
                t6 = t5 + pd.Timedelta(minutes=cut)
                mask = mask & ~((df.index >= t5) & (df.index <= t6))
                
                # 下午收盘前cut分钟
                t7 = pd.Timestamp.combine(date.date(), time(15, 0)) - pd.Timedelta(minutes=cut)
                t8 = pd.Timestamp.combine(date.date(), time(15, 0))
                mask = mask & ~((df.index >= t7) & (df.index <= t8))
                
                # 晚上开盘后cut分钟（当天21:00-21:10）
                t9 = pd.Timestamp.combine(date.date(), time(21, 0))
                t10 = t9 + pd.Timedelta(minutes=cut)
                mask = mask & ~((df.index >= t9) & (df.index <= t10))
                
                # 次日凌晨收盘前cut分钟（次日02:20-02:30）
                next_date = date + pd.Timedelta(days=1)
                t11 = pd.Timestamp.combine(next_date.date(), time(1, 0)) - pd.Timedelta(minutes=cut)
                t12 = pd.Timestamp.combine(next_date.date(), time(1, 0))
                mask = mask & ~((df.index >= t11) & (df.index <= t12))
            
            return df[mask]              

        else: print('trading hours not typical, pass')

class FutureDataLoader:
    def __init__(self, symbol: str):

        self.column_names = [
            'trade_time', 'contract', 
            'volume', 'turnover',             
            'open_interest', 
            'last_price', 
            'bid_price_1', 'bid_volume_1',         
            'bid_price_2', 'bid_volume_2', 
            'bid_price_3', 'bid_volume_3',         
            'bid_price_4', 'bid_volume_4',         
            'bid_price_5', 'bid_volume_5',         
            'ask_price_1', 'ask_volume_1',         
            'ask_price_2', 'ask_volume_2',         
            'ask_price_3', 'ask_volume_3',         
            'ask_price_4', 'ask_volume_4',         
            'ask_price_5', 'ask_volume_5'          
        ]

        self.symbol = symbol
        config_loader = InstrumentConfig()
        config_future = config_loader.get_instrument_config(symbol)
        exchange = config_future["exchange"]
        self.trade_type = config_future["trading_hours"]

        self.commodity = True
        self.base_path = f"/mnt/Data/future/decode_csv_{exchange}"
    
    @lru_cache(maxsize=1000)
    def get_contracts(self, symbol: str, date: str) -> List[str]:
        """
        获取指定品种和日期的所有合约列表
        
        参数:
            product: 品种代码 (如 'IC')
            date: 日期 (格式 'YYYYMMDD')
            
        返回:
            合约代码列表 (如 ['IC2401', 'IC2402'])
        """
        year, month, day = date.split('-')
        date_path = os.path.join(self.base_path, year, str(int(month)), str(int(day)))
        if not os.path.exists(date_path):
            # print(f"日期路径不存在: {date_path}")
            return []
        
        light_path = os.path.join(date_path, 'light')
        night_path = os.path.join(date_path, 'night')
        contracts = []

        import re
        def get_letters(s):
            return re.sub(r'\d+$', '', s)  # 只移除末尾的连续数字

        if os.path.exists(light_path):
            for filename in os.listdir(light_path):
                if get_letters(filename).upper() == symbol.upper():
                    contracts.append(filename)
        else:
            pass
            # print(f'{date} 无白天盘交易数据') 

        if os.path.exists(night_path):
            for filename in os.listdir(night_path):
                if get_letters(filename).upper() == symbol.upper():
                    contracts.append(filename) 
        else:
            pass
            # print(f'{date} 无夜盘交易数据')         
                
        return sorted(set(contracts))

    def load_contract_data(self, date: str, contract: str) -> pd.DataFrame:
        """
        加载指定日期和合约的CSV数据
        
        参数:
            date: 日期 (格式 'YYYYMMDD')
            contract: 合约代码 (如 'IC2401')
            
        返回:
            pandas DataFrame
        """
        year, month, day = date.split('-')
        light_file_path = os.path.join(self.base_path, year, str(int(month)), str(int(day)), 'light', contract)
        night_file_path = os.path.join(self.base_path, year, str(int(month)), str(int(day)), 'night', contract)
    
        if os.path.exists(light_file_path):
            df_light = pd.read_csv(light_file_path, header=None, low_memory=False)
            df_light.columns = self.column_names
            df_light['period'] = 'light'
        else: df_light = pd.DataFrame()

        if os.path.exists(night_file_path):
            df_night = pd.read_csv(night_file_path, header=None, low_memory=False)
            df_night.columns = self.column_names
            df_night['period'] = 'night'
        else: df_night = pd.DataFrame()

        df = pd.concat([df_night, df_light])  
        df['trade_date'] = date
        return df

    def pre_resample_data(self, df, date, last_trade_date_map=None):
        """
        预处理tick数据，与calc_recent_data.py保持一致
        
        Args:
            df: tick数据DataFrame
            date: 日期
            last_trade_date_map: 最后交易日映射（可选）
        """
        df = process_time_column_vectorized_floor(df)
        df = process_datetime(df)
        df = df_is_trading_time(df, 'datetime', trade_type=self.trade_type)
        df = df.sort_values('datetime')
        df['contract'] = df['contract'].str.strip()

        if isinstance(date, int):
            date = str(date)
        if len(date) == 8: 
            trade_date = pd.to_datetime(date, format='%Y%m%d').date()
        else:
            trade_date = pd.to_datetime(date).date()
        
        # 数值列处理（与calc_recent_data.py一致）
        numeric_cols = ['last_price',
                'open_interest', 'turnover', 'volume', 'bid_price_1', 'bid_price_2',
                'bid_price_3', 'bid_price_4', 'bid_price_5', 'ask_price_1',
                'ask_price_2', 'ask_price_3', 'ask_price_4', 'ask_price_5',
                'bid_volume_1', 'bid_volume_2', 'bid_volume_3', 'bid_volume_4',
                'bid_volume_5', 'ask_volume_1', 'ask_volume_2', 'ask_volume_3',
                'ask_volume_4', 'ask_volume_5']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                mask = df[col].abs()>1e20
                df.loc[mask, col] = pd.NA

        bool_cols = df.select_dtypes(include=['bool']).columns
        for col in bool_cols:
            df[col] = df[col].astype('boolean')

        for col in df.columns:
            str_lengths = df[col].astype(str).str.len()
            df.loc[str_lengths > 50, col] = pd.NA
            
        df['ts'] = pd.to_datetime(df['datetime']).dt.ceil('1min')
        
        df['trade_date'] = trade_date

        # 保存原始累计值
        df['TotalTradeVolume'] = df['volume'].copy()
        df['TotalTradeValue'] = df['turnover'].copy()

        # 计算差值（当前tick - 上一tick）
        volume_diff = df['volume'] - df['volume'].shift(1)
        turnover_diff = df['turnover'] - df['turnover'].shift(1)
        
        # 只更新交易时间的行
        df.loc[df.is_trading, 'volume'] = volume_diff.loc[df.is_trading]
        df.loc[df.is_trading, 'turnover'] = turnover_diff.loc[df.is_trading]

        # OHLC价格
        df['high'] = df['last_price']
        df['low'] = df['last_price']
        df['open'] = df['last_price']
        df['close'] = df['last_price']
        
        # 买卖价差和中间价
        df['spread'] = df['ask_price_1'] - df['bid_price_1']
        total_volume = df['bid_volume_1'] + df['ask_volume_1']
        df['mid_price'] = (
            (df['ask_price_1'] * df['bid_volume_1'] + 
                df['bid_price_1'] * df['ask_volume_1']) / total_volume
        ).round(4)
        
        df['last_twap'] = df['last_price']
        df['bar_count'] = df.groupby('ts').cumcount() + 1
        
        # 添加秒数列（用于计算tick均价）
        df['second'] = pd.to_datetime(df.datetime).dt.ceil('1s').dt.second
        
        # 添加最高最低价（日内累计）
        df['highest_price'] = df.groupby('trade_date')['last_price'].cummax()
        df['lowest_price'] = df.groupby('trade_date')['last_price'].cummin()
        
        # 添加最后交易日（如果提供）
        if last_trade_date_map:
            contract = df['contract'].iloc[0] if not df.empty else ""
            df['LAST_TRADE_DATE'] = last_trade_date_map.get(contract, "")
        else:
            df['LAST_TRADE_DATE'] = ""
        
        df = df.replace([np.inf, -np.inf], np.nan)

        return df

    # pd.set_option('future.no_silent_downcasting', True)


class FutureStockIndexDataLoader:
    def __init__(self, symbol: str):

        self.column_names = ['market_symbol', 'instrument', 'datetime', 'last_price', 'open_interest', 'open_interest_diff', 'turnover', 'volume', 'opening', 'closing',
       'trading_type', 'direction', 'bid_price_1', 'bid_price_2', 'bid_price_3', 'bid_price_4', 'bid_price_5', 'ask_price_1', 'ask_price_2', 'ask_price_3',
       'ask_price_4', 'ask_price_5', 'bid_volume_1', 'bid_volume_2', 'bid_volume_3', 'bid_volume_4', 'bid_volume_5', 'ask_volume_1', 'ask_volume_2', 'ask_volume_3',
       'ask_volume_4', 'ask_volume_5']

        self.symbol = symbol
        self.base_path = r"/mnt/Data/future/stock_index"
    
    @staticmethod
    def get_codes_by_symbol(symbol: str, start_contract: str, end_contract: str) -> List[str]:
        def extract_year_month(contract: str) -> tuple:
            year = int(contract[len(symbol):len(symbol)+2])
            month = int(contract[-2:])
            return year, month

        start_year, start_month = extract_year_month(start_contract)
        end_year, end_month = extract_year_month(end_contract)
        
        valid_months = {3, 6, 9, 12}
        if start_month not in valid_months or end_month not in valid_months:
            raise ValueError("合约月份必须是03/06/09/12之一")

        contracts = []
        current_year, current_month = start_year, start_month
        
        while (current_year < end_year) or (current_year == end_year and current_month <= end_month):
            contracts.append(f"{symbol}{current_year:02d}{current_month:02d}")
            if current_month == 12:
                current_year += 1
                current_month = 3
            else:
                current_month += 3
        return contracts

    def load_contract_data_cffex(self, contract: str) -> pd.DataFrame:
        extract_month = int(contract[len(self.symbol):])
        begin_month = extract_month - 100
        month_range = [x for x in range(begin_month, extract_month+1) if x%100 in range(1,13)]
        # print(f"加载{self.symbol}品种从20{begin_month}到20{extract_month}的所有数据...")
        data_lst = []
        for month in month_range:
            base_file_path = os.path.join(self.base_path, f"Fut_SF_Tick5_PanKouKZ_Daily_20{month}", f"Fut_SF_Tick5_PanKouKZ_Daily_20{month}")
            if not os.path.exists(base_file_path):
                continue

            files = [f for f in os.listdir(base_file_path) if f.startswith(contract) and f.endswith('.csv')]
            for file in files:
                file_path = os.path.join(base_file_path, file)
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path, encoding='GBK')
                    df.columns = self.column_names
                    df['trade_date'] = pd.to_datetime(df['datetime']).dt.date
                    data_lst.append(df)
        if data_lst:
            df_all = pd.concat(data_lst, ignore_index=True)
            return df_all
        else:        
            return pd.DataFrame()

class DataLoader:
    def __init__(self):

        self.main_config_dir = r'/mnt/Data/writable/liaoyuyang/data/future_info'
        self.factor_config_dir = r'/home/future_config/basic_config/config_factor.yaml'

    def make_main_instrument(self, symbol):
        config_loader = InstrumentConfig()
        config_future = config_loader.get_instrument_config(symbol)
        exchange = config_future["exchange"]

        if exchange in ["cffex"]:
            lst = os.listdir(f'/mnt/Data/writable/liaoyuyang/data/1day/{symbol}')
            df_lst = [pd.read_feather(f'/mnt/Data/writable/liaoyuyang/data/1day/{symbol}/{file}').rename(columns=str.lower) 
                        for file in lst]
            df_all = pd.concat(df_lst).reset_index(drop=True)
            df_all = df_all[df_all.instrument.str.endswith(('3', '6', '9', '12'))].reset_index(drop=True)
            df_all.trade_date = df_all.trade_date.astype(str)
            s = df_all.groupby('instrument', group_keys=False)['trade_date'].apply(lambda x: x.min())
            e = df_all.groupby('instrument', group_keys=False)['trade_date'].apply(lambda x: x.max())
            s.name = 'start_date'
            e.name = 'end_date'

            mr = config_future['margin_rate']
            m = config_future['contract_multiplier']
            se = pd.DataFrame(s).join(e)
            se.marin_rate = mr
            se.multiplier = m
            se.index = se.index +'.CFE'
            se.index.name = 'instrument'
            OI = pd.pivot_table(df_all, index='trade_date', columns='instrument', values='open_interest', aggfunc='sum')
            daily_max_contract = OI.idxmax(axis=1).shift().bfill()

            dominant_contracts = []
            current_contract = None

            for date, contract in daily_max_contract.items():
                if current_contract is None:
                    current_contract = contract
                else:
                    # 提取合约的到期年月（假设格式 ICYYMM）
                    current_year = int(current_contract[2:4])
                    current_month = int(current_contract[4:6])
                    new_year = int(contract[2:4])
                    new_month = int(contract[4:6])
                    
                    # 比较到期日（按年月）
                    if (new_year > current_year) or (new_year == current_year and new_month >= current_month):
                        current_contract = contract
                dominant_contracts.append(current_contract)

            dominant_series = pd.Series(dominant_contracts, index=daily_max_contract.index, name='dominant_contract')

            def get_next_3month_contract(contract):
                if not isinstance(contract, str):
                    return None
                prefix = contract[:2]  # 如 'IC'
                year = int(contract[2:4])  # '21' → 21
                month = int(contract[4:6])  # '03' → 3
                
                # 计算3个月后的年月
                new_month = month + 3
                new_year = year
                if new_month > 12:
                    new_month -= 12
                    new_year += 1
                
                return f"{prefix}{new_year:02d}{new_month:02d}"

            next_3month_contracts = dominant_series.apply(get_next_3month_contract)

            result_df = pd.DataFrame({
                'instrument': dominant_series,
                'instrument_next': next_3month_contracts
            }).reset_index()

            result_df['instrument'] = result_df['instrument']+'.CFE'
            result_df['instrument_next'] = result_df['instrument_next']+'.CFE'
            result_df = result_df.merge(se, on='instrument', how='left').set_index('trade_date')
            result_df.instrument = result_df.instrument.map(lambda x:x.split('.')[0])
            result_df.instrument_next = result_df.instrument_next.map(lambda x:x.split('.')[0])

            os.makedirs(f'/mnt/Data/writable/liaoyuyang/data/future_info/{symbol}', exist_ok=True)
            result_df.to_csv(f'/mnt/Data/writable/liaoyuyang/data/future_info/{symbol}/main_instrument.csv')

    def load_main_instrument(self, instrument):
        path = os.path.join(self.main_config_dir, instrument, 'main_instrument.csv')
        main_config = pd.read_csv(path, dtype=str)
        main_config['trade_date'] = main_config['trade_date'].map(lambda x :x.split(' ')[0])
        if instrument not in ['IF', 'IC', 'IH', 'IM']:
            main_config = main_config[main_config.trade_date>='2021-02-01']
        else:
            main_config = main_config[main_config.trade_date>='2018-01-01']

        return main_config
    
    def load_main_min(self, symbol):
        main_config = self.load_main_instrument(symbol)
        date_ranges = main_config.groupby('instrument').agg(
            start_date=('trade_date', 'min'),  # 合约最早出现日期
            end_date=('trade_date', 'max')     # 合约最后出现日期
        ).reset_index()

        df_lst = []
        for instrument in date_ranges['instrument']:
            start_date = date_ranges.loc[date_ranges['instrument'] == instrument, 'start_date'].values[0]
            end_date = date_ranges.loc[date_ranges['instrument'] == instrument, 'end_date'].values[0]
            
            df = pd.read_feather(f'/mnt/Data/writable/liaoyuyang/data/1min/{symbol}/{instrument}.feather')
            
            if not pd.api.types.is_datetime64_any_dtype(df['trade_date']):
                df['trade_date'] = pd.to_datetime(df['trade_date'])
            
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            
            mask = (df['trade_date'] >= start_date) & (df['trade_date'] <= end_date)
            df = df[mask]
            df_lst.append(df)
        
        df_all = pd.concat(df_lst, ignore_index=True).drop_duplicates()
        return df_all

    def load_main_daily(self, symbol):
        main_config = self.load_main_instrument(symbol)
        date_ranges = main_config.groupby('instrument').agg(
            start_date=('trade_date', 'min'),  # 合约最早出现日期
            end_date=('trade_date', 'max')     # 合约最后出现日期
        ).reset_index()

        df_lst = []
        for instrument in date_ranges['instrument']:
            if not os.path.exists(f'/mnt/Data/writable/liaoyuyang/data/1day/{symbol}/{instrument}.feather'):
                continue

            start_date = date_ranges.loc[date_ranges['instrument'] == instrument, 'start_date'].values[0]
            end_date = date_ranges.loc[date_ranges['instrument'] == instrument, 'end_date'].values[0]
            
            df = pd.read_feather(f'/mnt/Data/writable/liaoyuyang/data/1day/{symbol}/{instrument}.feather')
            
            if not pd.api.types.is_datetime64_any_dtype(df['trade_date']):
                df['trade_date'] = pd.to_datetime(df['trade_date'])
            
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            
            mask = (df['trade_date'] >= start_date) & (df['trade_date'] <= end_date)
            df = df[mask]
            df.trade_date = df.trade_date.dt.strftime('%Y-%m-%d')
            df_lst.append(df)
        df_all = pd.concat(df_lst, ignore_index=True)
        return df_all

    def load_valid_codes(self, symbol):
        return  sorted(set(list(self.load_main_instrument(symbol).instrument.dropna().to_list() + list(self.load_main_instrument(symbol).instrument_next.dropna().to_list()))))

    # def load_f_lst(self, fac_cls):
    #     from itertools import product

    #     def load_config(path):
    #         with open(path) as f:
    #             return json.load(f)
        
    #     kw_config = load_config(f'/mnt/Data/writable/liaoyuyang/basic_config/config_factor.json')
    #     base_functions = kw_config["FAC"][f"{fac_cls}_fac"]["fac_func_lst"]
    #     return base_functions

    # def load_tf_lst(self, fac_cls):
    #     from itertools import product

    #     def load_config(path):
    #         with open(path) as f:
    #             return json.load(f)
        
    #     kw_config = load_config(f'/mnt/Data/writable/liaoyuyang/basic_config/config_factor.json')
    #     base_functions = kw_config["FAC"][f"{fac_cls}_fac"]["fac_func_lst"]
    #     modifiers = kw_config["FAC"][f"{fac_cls}_fac"].get("modifiers", [None])
        
    #     # 生成笛卡尔积
    #     return [f"{func}_{mod}" if mod else func 
    #         for func, mod in product(base_functions, modifiers)]

    def load_f_lst(self, fac_cls):

        def load_config(path):
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        
        kw_config = load_config(self.factor_config_dir)
        base_functions = kw_config["FAC"][f"{fac_cls}_fac"]["fac_func_lst"]
        return base_functions

    def load_tf_lst(self, fac_cls):


        def load_config(path):
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        
        kw_config = load_config(self.factor_config_dir)
        base_functions = kw_config["FAC"][f"{fac_cls}_fac"]["fac_func_lst"]
        modifiers = kw_config["FAC"][f"{fac_cls}_fac"].get("modifiers", [None])
        
        # 生成笛卡尔积
        return [f"{func}_{mod}" if mod else func 
            for func, mod in product(base_functions, modifiers)]
