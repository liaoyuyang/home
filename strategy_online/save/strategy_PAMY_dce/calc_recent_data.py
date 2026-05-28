import os
import json
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from datetime import datetime, date, timedelta, time
from chinese_calendar import is_workday
from joblib import Parallel, delayed
import pytz

def load_config(config_path=None):
    """加载配置文件"""
    if config_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config.json")
    with open(config_path) as f:
        return json.load(f)

def get_exchange_from_symbol(symbol):
    """从品种代码获取交易所"""
    symbol_upper = symbol.upper()
    # DCE 大连商品交易所
    if any(symbol_upper.startswith(x) for x in ['A', 'B', 'M', 'Y', 'P', 'L', 'V', 'PP', 'J', 'JM', 'I', 'C', 'CS']):
        return 'dce'
    # CZCE 郑州商品交易所
    elif any(symbol_upper.startswith(x) for x in ['CF', 'SR', 'TA', 'OI', 'MA', 'FG', 'RM', 'SF', 'SM', 'CY', 'AP', 'UR', 'SA', 'PF', 'PK', 'LH', 'CJ']):
        return 'czce'
    # SHFE 上海期货交易所
    elif any(symbol_upper.startswith(x) for x in ['CU', 'AL', 'ZN', 'PB', 'NI', 'SN', 'AU', 'AG', 'RB', 'WR', 'HC', 'FU', 'BU', 'RU', 'SC']):
        return 'shfe'
    # INE 上海国际能源交易中心
    elif any(symbol_upper.startswith(x) for x in ['NR', 'BC', 'LU', 'EC']):
        return 'ine'
    # CFFEX 中国金融期货交易所
    elif any(symbol_upper.startswith(x) for x in ['IF', 'IC', 'IH', 'IM', 'TS', 'TF', 'T', 'TL']):
        return 'cffex'
    else:
        raise ValueError(f"无法识别品种 {symbol} 的交易所")

def get_trading_days(start_date='2021-01-01', end_date=None, exclude_days=None):
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    if end_date is None:
        end_date = date.today()
    elif isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    
    if exclude_days is None:
        exclude_days = []
    
    all_weekdays = pd.date_range(start_date, end_date, freq='B').date
    exclude_days_set = set(exclude_days)
    
    trading_days = [
        day.strftime('%Y-%m-%d') for day in all_weekdays 
        if is_workday(day) and day.strftime('%Y-%m-%d') not in exclude_days_set
    ]
    
    return trading_days

def parse_trade_hours(trade_hours_list):
    """解析交易时段字符串列表为时间对象对"""
    parsed = []
    for period in trade_hours_list:
        start_str, end_str = period.split('-')
        start_parts = start_str.strip().split(':')
        end_parts = end_str.strip().split(':')
        
        start_time = time(int(start_parts[0]), int(start_parts[1]), 0, 1)
        end_time = time(int(end_parts[0]), int(end_parts[1]), 0, 0)
        parsed.append((start_time, end_time))
    return parsed

def time_scale_df(df: pd.DataFrame, tscol_name: str, trade_type: list):
    df[tscol_name] = pd.to_datetime(df[tscol_name])
    
    conditions = []
    for start_time, end_time in parse_trade_hours(trade_type):
        conditions.append(df[tscol_name].dt.time.between(start_time, end_time))
    
    if conditions:
        mask = conditions[0]
        for cond in conditions[1:]:
            mask = mask | cond
        df = df[mask]
    
    return df

def df_is_trading_time(df: pd.DataFrame, tscol_name: str, trade_type: list):
    df[tscol_name] = pd.to_datetime(df[tscol_name])
    df['is_trading'] = False

    conditions = []
    for start_time, end_time in parse_trade_hours(trade_type):
        conditions.append(df[tscol_name].dt.time.between(start_time, end_time))
    
    if conditions:
        mask = conditions[0]
        for cond in conditions[1:]:
            mask = mask | cond
        df['is_trading'] = mask
    
    return df

def process_time_column_vectorized_floor(df, time_column='update_time'):
    """
    向量化版本，将毫秒向上取整到最近的0.25秒（向前调整）
    """
    time_parts = df[time_column].str.split('.', expand=True)
    seconds = time_parts[0]
    milliseconds = time_parts[1].fillna('000').astype(int)
    
    rounded_ms = np.floor(milliseconds / 250) * 250
    carry_over = (rounded_ms >= 1000).astype(int)
    rounded_ms = np.where(rounded_ms >= 1000, rounded_ms - 1000, rounded_ms)
    
    def adjust_seconds(sec_str, carry):
        if carry == 0:
            return sec_str
        time_obj = datetime.strptime(sec_str, '%H:%M:%S')
        time_obj += timedelta(seconds=1)
        return time_obj.strftime('%H:%M:%S')
    
    adjusted_seconds = [adjust_seconds(sec, carry) for sec, carry in zip(seconds, carry_over)]
    
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
        if row['period'] == 'night':
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
        errors='coerce'
    )
    df.drop(['adjusted_date', 'rounded_time'], axis=1, inplace=True)
    df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-4]

    return df

def load_data(instrument, date, trade_type, data_sources, last_trade_date_map=None):
    """
    加载数据 - 通用版本
    
    参数:
        instrument: 合约代码
        date: 日期
        trade_type: 交易时段列表
        data_sources: 数据源配置
        last_trade_date_map: 最后交易日映射
    """
    exchange = get_exchange_from_symbol(instrument)
    
    def pre_resample_data(df, date, instrument):
        df = process_time_column_vectorized_floor(df)
        df = process_datetime(df)
        df = df_is_trading_time(df, 'datetime', trade_type=trade_type)
        df = df.sort_values('datetime').copy()
        df['instrument'] = df['instrument'].str.strip()

        if isinstance(date, int):
            date = str(date)
        if len(date) == 8: 
            trade_date = pd.to_datetime(date, format='%Y%m%d').date()
        else:
            trade_date = pd.to_datetime(date).date()
        
        numeric_cols = ['last_price',
                'open_interest', 'turnover', 'volume', 'bid_price1', 'bid_price2',
                'bid_price3', 'bid_price4', 'bid_price5', 'ask_price1',
                'ask_price2', 'ask_price3', 'ask_price4', 'ask_price5',
                'bid_volume1', 'bid_volume2', 'bid_volume3', 'bid_volume4',
                'bid_volume5', 'ask_volume1', 'ask_volume2', 'ask_volume3',
                'ask_volume4', 'ask_volume5']
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

        df['TotalTradeVolume'] = df['volume'].copy()
        df['TotalTradeValue'] = df['turnover'].copy()

        volume_diff = df['volume'] - df['volume'].shift(1)
        turnover_diff = df['turnover'] - df['turnover'].shift(1)
        df.loc[df.is_trading, 'volume'] = volume_diff.loc[df.is_trading]
        df.loc[df.is_trading, 'turnover'] = turnover_diff.loc[df.is_trading]
        # 去重：先差分后去重，保留最后一条，与研究环境保持一致
        df = df.drop_duplicates(subset=['datetime'], keep='last')

        df['high'] = df['last_price']
        df['low'] = df['last_price']
        df['open'] = df['last_price']
        df['close'] = df['last_price']
        
        df['spread'] = df['ask_price1'] - df['bid_price1']
        total_volume = df['bid_volume1'] + df['ask_volume1']
        df['mid_price'] = (
            (df['ask_price1'] * df['bid_volume1'] + 
                df['bid_price1'] * df['ask_volume1']) / total_volume
        ).round(4)
        
        df['last_twap'] = df['last_price']
        df['bar_count'] = df.groupby('ts').cumcount() + 1
        
        df['highest_price'] = df.groupby('trade_date')['last_price'].cummax()
        df['lowest_price'] = df.groupby('trade_date')['last_price'].cummin()

        df['second'] = pd.to_datetime(df.datetime).dt.ceil('1s').dt.second
        
        # 从 last_trade_date_map 获取最后交易日
        if last_trade_date_map:
            df['LAST_TRADE_DATE'] = last_trade_date_map.get(instrument, "")
        else:
            df['LAST_TRADE_DATE'] = ""

        df = df.replace([np.inf, -np.inf], np.nan)

        return df
    
    year, month, day = date.split('-')
    column_names = [
                'update_time', 'instrument', 
                'volume', 'turnover',             
                'open_interest', 
                'last_price', 
                'bid_price1', 'bid_volume1',         
                'bid_price2', 'bid_volume2', 
                'bid_price3', 'bid_volume3',         
                'bid_price4', 'bid_volume4',         
                'bid_price5', 'bid_volume5',         
                'ask_price1', 'ask_volume1',         
                'ask_price2', 'ask_volume2',         
                'ask_price3', 'ask_volume3',         
                'ask_price4', 'ask_volume4',         
                'ask_price5', 'ask_volume5'          
            ]

    tick_data_root = data_sources.get('tick_data_root', '/mnt/Data/future')
    decode_csv_prefix = data_sources.get('decode_csv_prefix', 'decode_csv_')
    
    light_file_path = os.path.join(tick_data_root, f"{decode_csv_prefix}{exchange}", year, str(int(month)), str(int(day)), 'light', instrument)
    night_file_path = os.path.join(tick_data_root, f"{decode_csv_prefix}{exchange}", year, str(int(month)), str(int(day)), 'night', instrument)
    
    if os.path.exists(light_file_path):
        df_light = pd.read_csv(light_file_path, header=None)
        df_light.columns = column_names
        df_light['period'] = 'light'
    else: 
        df_light = pd.DataFrame()

    if os.path.exists(night_file_path):
        df_night = pd.read_csv(night_file_path, header=None)
        df_night.columns = column_names
        df_night['period'] = 'night'
    else: 
        df_night = pd.DataFrame()
        
    df = pd.concat([df_night, df_light])  
    df['trade_date'] = date

    df = pre_resample_data(df, date, instrument)
    return df
    
agg_dict = {
    'instrument': 'last',
    'open': 'first',
    'high': 'max',  
    'low': 'min',
    'close': 'last',
    'last_twap': 'mean',
    'mid_price': 'mean',
    'volume': 'sum',
    'turnover': 'sum',
    'open_interest': 'last',
    'spread': 'mean',
    'bar_count': 'last',
    'trade_date': 'last',
    'LAST_TRADE_DATE': 'last'
}

def process_single_instrument(instrument, date_lst, recent_data_path, trade_hours, data_sources, last_trade_date_map, tick_cache_days=5):
    """处理单个合约的数据"""
    min_data_lst = []
    tick_data_lst = []
    
    for date in tqdm(date_lst, desc=instrument):
        df = load_data(instrument, date, trade_type=trade_hours, data_sources=data_sources, last_trade_date_map=last_trade_date_map)
        
        if not df.empty:
            min_data = df.groupby('ts').agg(agg_dict).reset_index()
            min_data = time_scale_df(min_data, 'ts', trade_hours)
            min_data.rename(columns={"ts": "datetime"}, inplace=True)
            min_data = min_data[min_data.bar_count>1]
            min_data_lst.append(min_data)
            
            if date in date_lst[-tick_cache_days:]:
                tick_data_lst.append(df)
    
    if min_data_lst:
        min_data = pd.concat(min_data_lst).reset_index(drop=True).round(6)
        os.makedirs(recent_data_path, exist_ok=True)
        min_data.to_csv(f'{recent_data_path}/{instrument}_min.csv', index=False)
    
    if tick_data_lst:
        tick_data = pd.concat(tick_data_lst).reset_index(drop=True).round(6)
        tick_data.to_csv(f'{recent_data_path}/{instrument}_tick.csv', index=False)
    
    return instrument, len(min_data_lst), len(tick_data_lst)

def get_instruments_to_process(config):
    """根据配置获取需要处理的合约列表"""
    return list(set(config['instruments'].values()))

def get_trade_hours_for_symbol(symbol, config):
    """获取品种的交易时段"""
    symbol_specs = config.get('symbol_specs', {})
    if symbol in symbol_specs:
        return symbol_specs[symbol].get('trade_hours', ["09:00-11:30", "13:30-15:00", "21:00-23:00"])
    return ["09:00-11:30", "13:30-15:00", "21:00-23:00"]

if __name__ == "__main__":
    config = load_config()
    
    paths = config['paths']
    calculation_params = config['calculation_params']

    recent_data_path = paths['load_recent_data_path']
    
    # 获取需要处理的合约列表
    instruments = get_instruments_to_process(config)
    print(f"需要处理的合约: {instruments}")
    
    # 从第一个合约获取交易时段（假设所有合约交易时段相同）
    main_symbol = instruments[0][0].upper() if instruments else 'Y'
    trade_hours = get_trade_hours_for_symbol(main_symbol, config)
    
    # 使用东八区时间
    tz = pytz.timezone('Asia/Shanghai')
    now = datetime.now(tz)
    print("当前时间：", now)
    
    # 计算结束日期
    end_date = date.today() if now.time() >= time(16, 0) else date.today() - timedelta(days=1)
    # end_date = "2026-03-17"  # 测试时固定日期
    print(f"结束日期: {end_date}")
    
    start_date = calculation_params.get('start_date')
    recent_days = calculation_params.get('recent_days')
    tick_cache_days = calculation_params.get('tick_cache_days')
    parallel_jobs = calculation_params.get('parallel_jobs')
    
    date_lst = get_trading_days(start_date, end_date)[-recent_days:]
    print(f"处理日期范围: {date_lst[0]} 至 {date_lst[-1]}, 共 {len(date_lst)} 个交易日")
    
    data_sources = config['data_sources']
    last_trade_date_map = config.get('last_trade_date_map', {})
    
    # 并行处理所有合约
    results = Parallel(n_jobs=parallel_jobs)(
        delayed(process_single_instrument)(
            instrument, 
            date_lst, 
            recent_data_path,
            trade_hours,
            data_sources,
            last_trade_date_map,
            tick_cache_days
        )
        for instrument in instruments
    )

    # 输出结果统计
    print("\n处理完成统计：")
    for instrument, min_days, tick_days in results:
        print(f"{instrument}: 分钟数据 {min_days} 天, tick数据 {tick_days} 天")
