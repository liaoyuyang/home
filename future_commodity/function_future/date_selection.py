import chinese_calendar as calendar
import pandas as pd
from datetime import datetime, date, timedelta

def generate_trading_bars(trading_days, trading_hours):
    from datetime import datetime, time
    """生成标准交易时间序列"""
    timestamps = []
    if trading_hours == ["09:30-11:30", "13:00-15:00"]:

        for trade_date in trading_days:
            date_obj = datetime.strptime(trade_date, "%Y-%m-%d").date() if isinstance(trade_date, str) else trade_date
            
            # 上午: 09:31 - 11:30 (120根bar)
            timestamps.extend(pd.date_range(
                f"{date_obj} 09:31:00", f"{date_obj} 11:30:00", freq='1min'
            ))
            
            # 下午: 13:01 - 15:00 (120根bar)
            timestamps.extend(pd.date_range(
                f"{date_obj} 13:01:00", f"{date_obj} 15:00:00", freq='1min'
            ))

    if trading_hours == ["09:00-11:30", "13:30-15:00", "21:00-23:00"]:
        # 确保交易日按顺序排列
        sorted_days = sorted(trading_days)
        
        for i, trade_date in enumerate(sorted_days):
            date_obj = datetime.strptime(trade_date, "%Y-%m-%d").date() if isinstance(trade_date, str) else trade_date
            
            # 获取前一天日期
            if i == 0:  # 第一天，使用前一个自然日
                prev_date_obj = date_obj - timedelta(days=1)
            else:  # 使用前一个交易日
                prev_date = sorted_days[i-1]
                prev_date_obj = datetime.strptime(prev_date, "%Y-%m-%d").date() if isinstance(prev_date, str) else prev_date
            
            # 前一天的夜盘: 21:00 - 23:00 (120根bar)
            timestamps.extend(pd.date_range(
                f"{prev_date_obj} 21:00:00", f"{prev_date_obj} 23:00:00", freq='1min'
            ))
            
            # 当天的日盘上午: 09:00 - 11:30 (150根bar)
            timestamps.extend(pd.date_range(
                f"{date_obj} 09:00:00", f"{date_obj} 11:30:00", freq='1min'
            ))
            
            # 当天的日盘下午: 13:30 - 15:00 (90根bar)
            timestamps.extend(pd.date_range(
                f"{date_obj} 13:30:00", f"{date_obj} 15:00:00", freq='1min'
            ))

    if trading_hours == ['09:00-11:30', '13:30-15:00', '21:00-02:30']:
        # 确保交易日按顺序排列
        sorted_days = sorted(trading_days)
        
        for i, trade_date in enumerate(sorted_days):
            date_obj = datetime.strptime(trade_date, "%Y-%m-%d").date() if isinstance(trade_date, str) else trade_date
            
            # 获取前一天日期
            if i == 0:  # 第一天，使用前一个自然日
                prev_date_obj = date_obj - timedelta(days=1)
            else:  # 使用前一个交易日
                prev_date = sorted_days[i-1]
                prev_date_obj = datetime.strptime(prev_date, "%Y-%m-%d").date() if isinstance(prev_date, str) else prev_date
            
            # 前一天的夜盘: 21:00 - 23:00 (120根bar)
            timestamps.extend(pd.date_range(
                f"{prev_date_obj} 21:00:00", f"{prev_date_obj} 23:59:00", freq='1min'
            ))

            timestamps.extend(pd.date_range(
                f"{date_obj} 00:00:00", f"{date_obj} 02:30:00", freq='1min'
            ))

            # 当天的日盘上午: 09:00 - 11:30 (150根bar)
            timestamps.extend(pd.date_range(
                f"{date_obj} 09:00:00", f"{date_obj} 11:30:00", freq='1min'
            ))
            
            # 当天的日盘下午: 13:30 - 15:00 (90根bar)
            timestamps.extend(pd.date_range(
                f"{date_obj} 13:30:00", f"{date_obj} 15:00:00", freq='1min'
            ))

    if trading_hours == ['09:00-11:30', '13:30-15:00', '21:00-01:00']:
        # 确保交易日按顺序排列
        sorted_days = sorted(trading_days)
        
        for i, trade_date in enumerate(sorted_days):
            date_obj = datetime.strptime(trade_date, "%Y-%m-%d").date() if isinstance(trade_date, str) else trade_date
            
            # 获取前一天日期
            if i == 0:  # 第一天，使用前一个自然日
                prev_date_obj = date_obj - timedelta(days=1)
            else:  # 使用前一个交易日
                prev_date = sorted_days[i-1]
                prev_date_obj = datetime.strptime(prev_date, "%Y-%m-%d").date() if isinstance(prev_date, str) else prev_date
            
            # 前一天的夜盘: 21:00 - 23:00 (120根bar)
            timestamps.extend(pd.date_range(
                f"{prev_date_obj} 21:00:00", f"{prev_date_obj} 23:59:00", freq='1min'
            ))

            timestamps.extend(pd.date_range(
                f"{date_obj} 00:00:00", f"{date_obj} 01:00:00", freq='1min'
            ))

            # 当天的日盘上午: 09:00 - 11:30 (150根bar)
            timestamps.extend(pd.date_range(
                f"{date_obj} 09:00:00", f"{date_obj} 11:30:00", freq='1min'
            ))
            
            # 当天的日盘下午: 13:30 - 15:00 (90根bar)
            timestamps.extend(pd.date_range(
                f"{date_obj} 13:30:00", f"{date_obj} 15:00:00", freq='1min'
            ))

    # 处理无夜盘的品种，如尿素(UR): ['09:00-11:30', '13:30-15:00']
    if trading_hours == ['09:00-11:30', '13:30-15:00']:
        # 确保交易日按顺序排列
        sorted_days = sorted(trading_days)
        
        for trade_date in sorted_days:
            date_obj = datetime.strptime(trade_date, "%Y-%m-%d").date() if isinstance(trade_date, str) else trade_date
            
            # 当天的日盘上午: 09:00 - 11:30 (150根bar)
            timestamps.extend(pd.date_range(
                f"{date_obj} 09:00:00", f"{date_obj} 11:30:00", freq='1min'
            ))
            
            # 当天的日盘下午: 13:30 - 15:00 (90根bar)
            timestamps.extend(pd.date_range(
                f"{date_obj} 13:30:00", f"{date_obj} 15:00:00", freq='1min'
            ))

    return pd.DatetimeIndex(timestamps)

def get_trading_days(start_date='2021-01-01', end_date=None, exclude_days=None, return_str=False):
    """
    获取指定日期范围内的所有交易日（周一到周五，并剔除指定的非交易日）
    
    参数:
        start_date (str/datetime.
        date): 起始日期，格式 'YYYY-MM-DD' 或 date 对象，默认 '2021-01-01'
        end_date (str/datetime.date): 结束日期，格式 'YYYY-MM-DD' 或 date 对象，默认今天
        exclude_days (list[str]): 要排除的非交易日列表，格式 ['YYYY-MM-DD', ...]
    
    返回:
        list[date]: 所有交易日的日期列表
    """
    # 处理日期格式
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    if end_date is None:
        end_date = date.today()
    elif isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    
    # 默认要排除的非交易日（可自定义修改）
    if exclude_days is None:
        exclude_days = [
            # 这里放你要排除的非交易日，例如：
            '2024-02-09',  # 2024年除夕（周五）
            '2024-02-12',  # 2024年春节
            '2024-02-13',  # 2024年春节
            '2024-02-14',  # 2024年春节
            '2024-02-15',  # 2024年春节
            '2024-02-16',  # 2024年春节
            '2024-04-04',  # 2024年清明节

            # 可以继续添加其他需要排除的日期...
        ]
    
    # 生成所有工作日（周一到周五）
    all_weekdays = pd.date_range(start_date, end_date, freq='B').date
    
    # 转换为字符串格式用于比较
    exclude_days_set = set(exclude_days)
    
    if return_str:
        trading_days = [
            day.strftime('%Y-%m-%d') for day in all_weekdays 
            if calendar.is_workday(day) and day.strftime('%Y-%m-%d') not in exclude_days_set
        ]     
    else:
        trading_days = [
            day for day in all_weekdays 
            if (calendar.is_workday(day)) & (day.strftime('%Y-%m-%d') not in exclude_days_set)
        ]
    
    return trading_days


