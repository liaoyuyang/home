from pyexpat import model
import pandas as pd
import numpy as np
from datetime import datetime, time
from pathlib import Path 
import os
import json
import sys
from io import StringIO
from contextlib import redirect_stdout
from concurrent.futures import ThreadPoolExecutor
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display, clear_output
import ipywidgets as widgets
import threading
from tqdm.auto import tqdm
import warnings
import lightgbm as lgb

warnings.filterwarnings("ignore", category=UserWarning, module="numpy.core.getlimits")
pd.set_option('future.no_silent_downcasting', True)
def load_config(config_path: str):
    try:
        # 检查文件是否存在
        if not Path(config_path).exists():
            raise FileNotFoundError(f"配置文件 {config_path} 不存在")
        
        # 读取文件内容
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        return config
        
    except json.JSONDecodeError as e:
        raise ValueError(f"配置文件格式错误: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"读取配置文件失败: {str(e)}")

def analyze_pos_distribution(merged_data):
    """
    统计仓位分布情况
    :param pos_series: 仓位序列（包含+1, -1, 0）
    :return: 分布统计字典
    """

    pos_series = merged_data['pos'].fillna(0)
    merged_data['trade_count'] = pos_series.diff().abs().groupby(merged_data['date']).cumsum()
    trade_count = merged_data[['trade_count', 'date']].groupby('date').last().sum().iloc[0] / 2
    ret_all = merged_data.groupby('date').date_cum_ret.last().sum()
    margin = ret_all / trade_count
    date_count = len(merged_data.date.unique())
    sharpe = merged_data.groupby('date').date_cum_ret.last().mean() / merged_data.groupby('date').date_cum_ret.last().std() * np.sqrt(252)
    max_dd = (merged_data.date_cum_ret - merged_data.groupby('date').date_cum_ret.cummax()).min()
    calmar = merged_data.groupby('date').date_cum_ret.last().mean() * 252 / abs(max_dd)

    count = pos_series.value_counts()
    total = len(pos_series)

    max_loss_date = merged_data.groupby('date').date_cum_ret.last().idxmin()
    max_earn_date = merged_data.groupby('date').date_cum_ret.last().idxmax()
    
    stats = {
        '多头(+1) 数量': count.get(1, 0),
        '空头(-1) 数量': count.get(-1, 0),
        '空仓(0) 数量': count.get(0, 0),
        '多头占比': f"{count.get(1, 0)/total:.2%}",
        '空头占比': f"{count.get(-1, 0)/total:.2%}",
        '空仓占比': f"{count.get(0, 0)/total:.2%}",
        "交易日数量": f"{date_count}",
        "总收益(非年化%)": f"{ret_all * 100:.2f}",
        "总交易次数（开平算一次）": f"{trade_count:.0f}",
        "每笔收益(%%)": f"{margin * 10000:.2f}",
        "年化日度夏普": f"{sharpe:.2f}",
        "最大回撤(%)": f"{max_dd * 100:.2f}",
        "年化卡玛比率": f'{calmar:.2f}',
        "最赚钱的一天": f"{max_earn_date}",
        "最亏钱的一天": f"{max_loss_date}",

    }
    return pd.DataFrame.from_dict(stats, orient='index', columns=['统计值'])

# ====================== Model Backtesting ======================
def process_signals_v2(df, th1, th2,
                     holding_period, warmup=100, day=None, date_max_trade=100, 
                     vol_controller=None, 
                     ts_col='datetime', factor_col='factor', 
                     open_drop=True, close_drop=True,
                     trading_hours=None,
                     mask_hours=[]
                     ):
    
    if not vol_controller:
        vol_controller = pd.Series(1, index=df.index)

    df = df.copy()
    window = day
    df['factor_val'] = df[factor_col]
    df['date'] = df.trade_date

    if trading_hours == ["09:30-11:30", "13:00-15:00"]:
        df['factor_val'] = np.where(
            df[ts_col].dt.time > time(14, 50, 0),
            0,
            df['factor_val']
        )
        df['factor_val'] = np.where(
            df[ts_col].dt.time < time(9, 10, 0),
            0,
            df['factor_val']
        )
    elif  trading_hours == ["09:00-11:30", "13:30-15:00", "21:00-23:00"]:
        df['factor_val'] = np.where(
            ((df[ts_col].dt.time >= time(21, 0, 0)) & (df[ts_col].dt.time <= time(21, 10, 0))) |  # 21:00-21:10
            ((df[ts_col].dt.time >= time(22, 50, 0)) & (df[ts_col].dt.time <= time(23, 0, 0))) |  # 22:50-23:00
            ((df[ts_col].dt.time >= time(9, 0, 0)) & (df[ts_col].dt.time <= time(9, 10, 0))) |   # 9:00-9:10
            ((df[ts_col].dt.time >= time(14, 50, 0)) & (df[ts_col].dt.time <= time(15, 0, 0))),   # 14:50-15:00
            0,
            df['factor_val']
        )
    elif trading_hours == ["09:00-11:30", "13:30-15:00", "21:00-02:30"]:
        df['factor_val'] = np.where(
            ((df[ts_col].dt.time >= time(21, 0, 0)) & (df[ts_col].dt.time <= time(21, 10, 0))) |  # 21:00-21:10
            ((df[ts_col].dt.time >= time(2, 20, 0)) & (df[ts_col].dt.time <= time(2, 30, 0))) |   # 02:20-02:30（次日）
            ((df[ts_col].dt.time >= time(9, 0, 0)) & (df[ts_col].dt.time <= time(9, 10, 0))) |    # 09:00-09:10
            ((df[ts_col].dt.time >= time(14, 50, 0)) & (df[ts_col].dt.time <= time(15, 0, 0))),    # 14:50-15:00
            0,
            df['factor_val']
        )

    print("trading_hours", trading_hours)
    df['factor_val'] = df['factor_val'].fillna(0)
    thresholds = {
        'open_long': th1,
        'close_long': th2,
        'open_short': 1 - th1,
        'close_short': 1 - th2
    }

    for name, th in thresholds.items():
        df[f'th_{name}'] = df['factor_val'].shift(1).rolling(
            window=window, min_periods=warmup
        ).quantile(th)

    # ================================
    # 信号生成逻辑
    def _generate_signals(df):
        in_long = False
        in_short = False
        holding_bars = 0
        today_trade = 0
        pos = pd.Series(0, index=df.index, dtype=np.float32)
        time_series = df[ts_col].dt.time

        for i in range(warmup, len(df)):
 
            if df['date'].iloc[i]!=df['date'].iloc[i-1]:
                today_trade = 0

            current_val = df['factor_val'].iloc[i]
            th_ol = max(df['th_open_long'].iloc[i], 0)

            th_cl = df['th_close_long'].iloc[i]
            th_os = min(df['th_open_short'].iloc[i], -0)

            th_cs = df['th_close_short'].iloc[i]

            if time_series.iloc[i].hour in mask_hours:
                pos.iloc[i] = 0
                in_long = False
                in_short = False
                holding_bars = 0
                continue

            if open_drop:
                if trading_hours == ["09:30-11:30", "13:00-15:00"]:
                    # 股票交易时间段：9:40之前不交易
                    if time_series.iloc[i] < time(9, 40):
                        pos.iloc[i] = 0
                        in_long = False
                        in_short = False
                        holding_bars = 0
                        continue
                elif trading_hours == ["09:00-11:30", "13:30-15:00", "21:00-23:00"]:
                    # 期货交易时间段：早上9:10之前和夜盘21:10之前不交易
                    if (time_series.iloc[i] < time(9, 10)) or \
                       (time_series.iloc[i] >= time(21, 0) and time_series.iloc[i] < time(21, 10)):
                        pos.iloc[i] = 0
                        in_long = False
                        in_short = False
                        holding_bars = 0
                        continue
                elif trading_hours == ["09:00-11:30", "13:30-15:00", "21:00-02:30"]:
                    # 期货交易时间段：早上9:10之前和夜盘21:10之前不交易
                    if (time_series.iloc[i] < time(9, 10)) or \
                       (time_series.iloc[i] >= time(21, 0) and time_series.iloc[i] < time(21, 10)):
                        pos.iloc[i] = 0
                        in_long = False
                        in_short = False
                        holding_bars = 0
                        continue

            if close_drop:
                if trading_hours == ["09:30-11:30", "13:00-15:00"]:
                    # 股票交易时间段：14:50之后不交易
                    if time_series.iloc[i] > time(14, 50):
                        pos.iloc[i] = 0        
                        in_long = False
                        in_short = False
                        holding_bars = 0
                        continue
                elif trading_hours == ["09:00-11:30", "13:30-15:00", "21:00-23:00"]:
                    # 期货交易时间段：下午14:50之后和夜盘22:50之后不交易
                    if (time_series.iloc[i] > time(14, 50) and time_series.iloc[i] <= time(15, 0)) or \
                       (time_series.iloc[i] >= time(22, 50) and time_series.iloc[i] <= time(23, 0)):
                        pos.iloc[i] = 0        
                        in_long = False
                        in_short = False
                        holding_bars = 0
                        continue 

                elif trading_hours == ["09:00-11:30", "13:30-15:00", "21:00-02:30"]:
                    # 期货交易时间段：下午14:50之后和夜盘22:50之后不交易
                    if (time_series.iloc[i] > time(14, 50) and time_series.iloc[i] <= time(15, 0)) or \
                       (time_series.iloc[i] >= time(2, 20) and time_series.iloc[i] <= time(2, 30)):
                        pos.iloc[i] = 0        
                        in_long = False
                        in_short = False
                        holding_bars = 0
                        continue 

            if not in_long and not in_short:

                if current_val >= th_ol:
                    # 当前没有仓位的时候， 如果满足波动率开仓条件，并且今日开仓次数小于date_max_trade就可以开仓
                    if (vol_controller.iloc[i] == 1) & (today_trade<date_max_trade):
                        pos.iloc[i] = 1
                        in_long = True
                        holding_bars = 1
                        today_trade += 1

                elif current_val <= th_os:
                    # 当前没有仓位的时候， 如果满足波动率开仓条件，并且今日开仓次数小于date_max_trade就可以开仓
                    if (vol_controller.iloc[i] == 1) & (today_trade<date_max_trade):
                        pos.iloc[i] = -1
                        in_short = True
                        holding_bars = 1
                        today_trade += 1

            elif in_long:
                # 持多时继续达到开多阈值，刷新当前已持仓时间
                if current_val >= th_ol:
                    pos.iloc[i] = 1
                    holding_bars = 1
                # 持多时跌落到开空阈值，判断是否达到波动率阈值和日开仓次数上限，满足条件反向开仓
                elif current_val <= th_os:
                    if (vol_controller.iloc[i] == 1) & (today_trade<date_max_trade):
                        pos.iloc[i] = -1
                        in_long = False
                        in_short = True
                        holding_bars = 1
                        today_trade += 1

                # 持多时跌落到平多阈值并且没有达到反向开空阈值，或者没有达到任何阈值但是达到持仓时间上限
                elif (holding_bars > holding_period) or (current_val <= th_cl):
                    pos.iloc[i] = 0
                    in_long = False
                    holding_bars = 0

                # 持多并且没有达到任何阈值，继续拿
                else:
                    pos.iloc[i] = 1
                    holding_bars += 1
            
            elif in_short:
                # 持空时继续达到开空阈值，刷新当前已持仓时间
                if current_val <= th_os:
                    pos.iloc[i] = -1
                    holding_bars = 1

                # 持空时上涨到开多阈值，判断是否达到波动率阈值和日开仓次数上限，满足条件反向开仓
                elif current_val >= th_ol:
                    if (vol_controller.iloc[i] == 1) & (today_trade<date_max_trade):
                        pos.iloc[i] = 1
                        in_long = True
                        in_short = False
                        holding_bars = 1
                        today_trade += 1

                # 持空时上涨到平空阈值并且没有达到反向开空阈值，或者没有达到任何阈值但是达到持仓时间上限
                elif (holding_bars > holding_period) or (current_val >= th_cs):
                    pos.iloc[i] = 0
                    in_short = False
                    holding_bars = 0

                # 持空并且没有达到任何阈值，继续拿
                else:
                    pos.iloc[i] = -1
                    holding_bars += 1
        
        return pos

    df['pos'] = _generate_signals(df).fillna(0)
    return df

def process_signals_v3(df, th1, th2,
                     holding_period, warmup=100, day=3, date_max_trade=100, 
                     vol_controller=None, 
                     ts_col='datetime', factor_col='factor', 
                     open_drop=True, close_drop=True, print_wrong=False, color_th=0.8,
                     trading_hours=None):
    
    if not vol_controller:
        vol_controller = pd.Series(1, index=df.index)

    df = df.copy()
    window = 60 * 4 * day
    df['factor_val'] = df[factor_col]
    df['date'] = df.trade_date

    df['factor_val'] = np.where(
        df[ts_col].dt.time > time(14, 50, 0),
        0,
        df['factor_val']
    )
    df['factor_val'] = np.where(
        df[ts_col].dt.time < time(9, 40, 0),
        0,
        df['factor_val']
    )

    df['factor_val'] = df['factor_val'].fillna(0)
    thresholds = {
        'open_long': th1,
        'close_long': th2,
        'open_short': 1 - th1,
        'close_short': 1 - th2
    }

    for name, th in thresholds.items():
        df[f'th_{name}'] = df['factor_val'].shift(1).rolling(
            window=window, min_periods=warmup
        ).quantile(th)

    # for n_diff in [0,1,2,3,4,5]:
    #     df[f'close_diff_lag{n_diff}'] = df['close'].diff().shift(n_diff).fillna(0) >= 0
    # df['color_rate'] = df[[f'close_diff_lag{n}' for n in range(0,6)]].sum(axis=1) / 6

    window_size = 6
    close_diff = df['close'].diff().fillna(0)

    # 计算涨的幅度总和
    positive_sum = close_diff.rolling(window=window_size, min_periods=1).apply(
        lambda x: np.sum(np.maximum(x, 0)), raw=True
    )

    # 计算总幅度（绝对值）总和
    abs_sum = close_diff.abs().rolling(window=window_size, min_periods=1).sum()

    # 计算占比
    df['color_rate'] = positive_sum / (abs_sum + 1e-10)

    # ================================
    # 信号生成逻辑
    def _generate_signals(df):
        in_long = False
        in_short = False
        holding_bars = 0
        today_trade = 0
        pos = pd.Series(0, index=df.index, dtype=np.float32)

        time_series = df[ts_col].dt.time

        for i in range(warmup, len(df)):
 
            if df['date'].iloc[i]!=df['date'].iloc[i-1]:
                today_trade = 0

            current_val = df['factor_val'].iloc[i]
            th_ol = max(df['th_open_long'].iloc[i], 0)

            th_cl = df['th_close_long'].iloc[i]
            th_os = min(df['th_open_short'].iloc[i], -0)

            th_cs = df['th_close_short'].iloc[i]

            if open_drop:
                if trading_hours == ["09:30-11:30", "13:00-15:00"]:
                    # 股票交易时间段：9:40之前不交易
                    if time_series.iloc[i] < time(9, 40):
                        pos.iloc[i] = 0
                        in_long = False
                        in_short = False
                        holding_bars = 0
                        continue
                elif trading_hours == ["09:00-11:30", "13:30-15:00", "21:00-23:00"]:
                    # 期货交易时间段：早上9:10之前和夜盘21:10之前不交易
                    if (time_series.iloc[i] < time(9, 10)) or \
                       (time_series.iloc[i] >= time(21, 0) and time_series.iloc[i] < time(21, 10)):
                        pos.iloc[i] = 0
                        in_long = False
                        in_short = False
                        holding_bars = 0
                        continue
            
            if close_drop:
                if trading_hours == ["09:30-11:30", "13:00-15:00"]:
                    # 股票交易时间段：14:50之后不交易
                    if time_series.iloc[i] > time(14, 50):
                        pos.iloc[i] = 0        
                        in_long = False
                        in_short = False
                        holding_bars = 0
                        continue
                elif trading_hours == ["09:00-11:30", "13:30-15:00", "21:00-23:00"]:
                    # 期货交易时间段：下午14:50之后和夜盘22:50之后不交易
                    if (time_series.iloc[i] > time(14, 50) and time_series.iloc[i] <= time(15, 0)) or \
                       (time_series.iloc[i] >= time(22, 50) and time_series.iloc[i] <= time(23, 0)):
                        pos.iloc[i] = 0        
                        in_long = False
                        in_short = False
                        holding_bars = 0
                        continue   

            if not in_long and not in_short:

                if current_val >= th_ol:
                    # 当前没有仓位的时候， 如果满足波动率开仓条件，并且今日开仓次数小于date_max_trade就可以开仓
                    if (vol_controller.iloc[i] == 1) & (today_trade<date_max_trade):
                        pos.iloc[i] = 1
                        in_long = True
                        holding_bars = 1
                        today_trade += 1

                elif current_val <= th_os:
                    # 当前没有仓位的时候， 如果满足波动率开仓条件，并且今日开仓次数小于date_max_trade就可以开仓
                    if (vol_controller.iloc[i] == 1) & (today_trade<date_max_trade):
                        pos.iloc[i] = -1
                        in_short = True
                        holding_bars = 1
                        today_trade += 1

            elif in_long:
                # 持多时继续达到开多阈值，刷新当前已持仓时间
                if current_val >= th_ol:
                    pos.iloc[i] = 1
                    holding_bars = 1
                # 持多时跌落到开空阈值，判断是否达到波动率阈值和日开仓次数上限，满足条件反向开仓
                elif current_val <= th_os:
                    if (vol_controller.iloc[i] == 1) & (today_trade<date_max_trade):
                        pos.iloc[i] = -1
                        in_long = False
                        in_short = True
                        holding_bars = 1
                        today_trade += 1

                # 持多时跌落到平多阈值并且没有达到反向开空阈值，或者没有达到任何阈值但是达到持仓时间上限
                elif (holding_bars > holding_period) or (current_val <= th_cl):
                    pos.iloc[i] = 0
                    in_long = False
                    holding_bars = 0

                # 开了多头，拿到超过3分钟的时候发现超过80%的绿线，直接平仓
                elif (holding_bars > 3) and df['color_rate'].iloc[i]<1-color_th:
                    pos.iloc[i] = 0
                    in_long = False
                    holding_bars = 0
                    if print_wrong:
                        print(f"多头止损平仓 at datetime {df[ts_col].iloc[i]}, color_rate {df['color_rate'].iloc[i]:.2f}") 

                # 持多并且没有达到任何阈值，继续拿
                else:
                    pos.iloc[i] = 1
                    holding_bars += 1
            
            elif in_short:
                # 持空时继续达到开空阈值，刷新当前已持仓时间
                if current_val <= th_os:
                    pos.iloc[i] = -1
                    holding_bars = 1

                # 持空时上涨到开多阈值，判断是否达到波动率阈值和日开仓次数上限，满足条件反向开仓
                elif current_val >= th_ol:
                    if (vol_controller.iloc[i] == 1) & (today_trade<date_max_trade):
                        pos.iloc[i] = 1
                        in_long = True
                        in_short = False
                        holding_bars = 1
                        today_trade += 1

                # 持空时上涨到平空阈值并且没有达到反向开空阈值，或者没有达到任何阈值但是达到持仓时间上限
                elif (holding_bars > holding_period) or (current_val >= th_cs):
                    pos.iloc[i] = 0
                    in_short = False
                    holding_bars = 0

                # 开了空头，拿到超过3分钟的时候发现超过80%的红线，直接平仓
                elif (holding_bars > 3) and df['color_rate'].iloc[i]>color_th:
                    pos.iloc[i] = 0
                    in_short = False
                    holding_bars = 0
                    if print_wrong:
                        print(f"空头止损平仓 at datetime {df[ts_col].iloc[i]}, color_rate {df['color_rate'].iloc[i]:.2f}")

                # 持空并且没有达到任何阈值，继续拿
                else:
                    pos.iloc[i] = -1
                    holding_bars += 1
        
        return pos

    df['pos'] = _generate_signals(df).fillna(0)
    return df

def calc_dd(cum_returns):
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max * 100
    return drawdown

def calc_yearly_eval(merged_data):
    stats_df = merged_data.copy()
    stats_df['pos_chg'] = stats_df.pos.diff().abs() 
    stats_date = stats_df.groupby('date').agg({
        'pnl_ret': 'sum',
        'close' : 'last',
        'year': 'last',
        'pos_chg': 'sum'
    })
    year_lst = []
    for year in sorted(stats_df.year.unique()):
        stats_date_y = stats_date[stats_date.year == year]
        drawdown = calc_dd(stats_date_y.pnl_ret.cumsum()+1).min()
        ann_ret = stats_date_y.pnl_ret.mean() * 252
        sharpe = stats_date_y.pnl_ret.mean() / stats_date_y.pnl_ret.std() * np.sqrt(252)
        turnover_rate = stats_date_y['pos_chg'].mean()
        year_lst.append([year, ann_ret * 100, sharpe, drawdown, turnover_rate * 100])
    df_stat = pd.DataFrame(year_lst, columns=['year', 'ann_ret(%)', 'sharpe', 'max_drawdown', 'turnover_rate(%)']).set_index('year')
    return df_stat.round(2)

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def max_zero_duration_per_day(df):
    daily_max_zeros = []
    
    for date, day_data in df.groupby('date'):
        # 计算当日内连续0的序列
        day_data = day_data.copy()
        day_data['zero_group'] = (day_data['pos'] != 0).cumsum()  # 当pos不为0时分组变化
        
        # 只关注pos=0的序列
        zero_sequences = day_data[day_data['pos'] == 0].groupby('zero_group').size()
        
        # 取当日内最长的连续0序列长度
        max_zero_duration = zero_sequences.max() if not zero_sequences.empty else 0
        daily_max_zeros.append({'date': date, 'max_zero_duration': max_zero_duration})
    
    return pd.DataFrame(daily_max_zeros)


def plot_eval(variety, merged_data, mode):
    stats_df = merged_data.copy()
    stats_df = stats_df.set_index('datetime')
    stats_df['hour'] = stats_df.index.hour

    if mode == "锁仓":
        # 计算绩效指标
        daily_returns = stats_df[['pnl_ret', 'date']].groupby('date').sum()
        ret = daily_returns.mean() * 252
        sp = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)).iloc[0]
        maxdd = calc_dd(daily_returns.cumsum() + 1).min().iloc[0]
        
        # 创建包含3个子图的画布
        fig, axes = plt.subplots(4, 2, figsize=(18, 9))
        
        # 图表1: 累计收益和成本滑点
        ax1 = axes[0, 0]

        cum_data = stats_df[['pnl_ret', 'cost&slippage_rate', 'date']].groupby('date').sum().cumsum()

        # 在同一个y轴上绘制两条线
        ax1.plot(cum_data.index, cum_data['pnl_ret'], color='blue', label='Cumulative Return', linewidth=2)
        ax1.plot(cum_data.index, cum_data['cost&slippage_rate'], color='red', label='Cost & Slippage', linestyle='--', linewidth=2)

        ax1.set_title(f'{variety} - (Ret: {ret[0]*100:.2f}%, Sharpe: {sp:.2f}, MaxDD: {maxdd:.2f}%)', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')  # 自动选择最佳位置
        ax1.grid(True, alpha=0.3)
        ax1.set_ylabel('Cumulative Rate')

        # 图表2: 持仓柱状图
        ax2 = axes[1, 0]
        pos_data = stats_df[['date', 'pos']].replace(0, np.nan).groupby('date').mean()
        ax2.bar(pos_data.index, pos_data['pos'], color='green', alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax2.set_title(f'Average Daily Position', fontsize=14, fontweight='bold')
        # ax2.set_ylabel('Position')
        # ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        
        # 设置x轴刻度数量（避免过于拥挤）
        ax2.xaxis.set_major_locator(MaxNLocator(nbins=8))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0)
        
        # 图表3: 多头/空头和保险率
        ax3 = axes[2 ,0]
        ax3_right = ax3.twinx()
        
        trade_data = stats_df[['date', 'long_today', 'short_today', 'Margin_rate']].groupby('date').max()
        ax3.plot(trade_data.index, trade_data['long_today'], color='blue', label='Long Today', linewidth=2)
        ax3.plot(trade_data.index, trade_data['short_today'], color='red', label='Short Today', linewidth=2)
        ax3_right.plot(trade_data.index, trade_data['Margin_rate'], color='orange', label='Margin Rate', linestyle='--')
        
        ax3.set_title(f'Trading Activity & Margin Rate', fontsize=14, fontweight='bold')
        # ax3.set_ylabel('Trade Count')
        # ax3_right.set_ylabel('Margin Rate')
        ax3.legend(loc='upper left')
        ax3_right.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)

        # 图表4: 多头/空头收益率拆解
        ax4 = axes[0, 1]

        # 分别计算多头和空头的累计收益
        long_cum = stats_df[stats_df['pos'] > 0].groupby('date')['pnl_ret'].sum().cumsum()
        short_cum = stats_df[stats_df['pos'] < 0].groupby('date')['pnl_ret'].sum().cumsum()

        # 绘制多头和空头收益曲线
        ax4.plot(long_cum.index, long_cum.values, color='green', label='long_cum', linewidth=2)
        ax4.plot(short_cum.index, short_cum.values, color='red', label='short_cum', linewidth=2)

        # 设置图表属性
        ax4.set_title('long/short_split', fontsize=14, fontweight='bold')
        ax4.legend(loc='upper left')
        ax4.grid(True, alpha=0.3)

        # 图表5: 多头和空头持仓连续性统计（简洁版）
        ax5 = axes[1, 1]

        stats_df['pos_group'] = (stats_df['pos'] != stats_df['pos'].shift()).cumsum()
        grouped = stats_df.groupby('pos_group')
        position_durations = grouped.size()
        position_types = grouped['pos'].first()

        position_df = pd.DataFrame({
            'position': position_types,
            'duration': position_durations
        })

        trade_df = position_df[position_df['position'].isin([1, -1])]
        long_durations = trade_df[trade_df['position'] == 1]['duration']
        short_durations = trade_df[trade_df['position'] == -1]['duration']

        if not long_durations.empty:
            ax5.hist(long_durations, bins=20, alpha=0.6, color='lightgreen', edgecolor='darkgreen')
        if not short_durations.empty:
            ax5.hist(short_durations, bins=20, alpha=0.6, color='lightcoral', edgecolor='darkred')
        stats_text = ""
        if not long_durations.empty:
            stats_text += f"Long: n={len(long_durations)}\nμ={long_durations.mean():.1f}\nσ={long_durations.std():.1f}\n\n"
        if not short_durations.empty:
            stats_text += f"Short: n={len(short_durations)}\nμ={short_durations.mean():.1f}\nσ={short_durations.std():.1f}"

        ax5.text(0.75, 1, stats_text, transform=ax5.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax5.set_title('Position Holding Duration Distribution', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)

        # Chart 6: Maximum Intraday Zero Position Duration
        ax6 = axes[2, 1]  # 假设这是第三行第一列的图表


        # 计算每日最大连续不开仓长度
        daily_zero_stats = max_zero_duration_per_day(stats_df)
        zero_days = len(daily_zero_stats[daily_zero_stats['max_zero_duration'] == 240])
        non_zero_stats = daily_zero_stats[daily_zero_stats['max_zero_duration'] != 240]

        if len(non_zero_stats) > 0:
            # 创建主坐标系
            ax6_main = ax6
            
            # 绘制非零值分布
            n, bins, patches = ax6_main.hist(non_zero_stats['max_zero_duration'], bins=20,
                                        color='blue', alpha=0.7, edgecolor='darkblue')
            
            # 设置y轴上限（为零值天数留出空间）
            y_max = max(n) * 1.2
            ax6_main.set_ylim(0, y_max)
            
            # 添加零值天数标注
            ax6_main.annotate(f'No Trade Days: {zero_days}', 
                            xy=(bins[len(bins)-1], y_max * 0.95),
                            xytext=(0, 20), textcoords='offset points',
                            ha='center', va='bottom', fontsize=10,
                            bbox=dict(boxstyle='round', facecolor='red', alpha=0.3),
                            arrowprops=dict(arrowstyle='->', color='red'))
            
            # 添加统计信息
            stats_text = f"Trade Days: {len(non_zero_stats)}\n"
            stats_text += f"Max no trade Duration: {non_zero_stats['max_zero_duration'].max()}\n"
            stats_text += f"Mean no trade Duration: {non_zero_stats['max_zero_duration'].mean():.1f}"
            
            ax6_main.text(0.75, 0.85, stats_text, transform=ax6_main.transAxes, fontsize=9,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax6_main.set_title('Daily Max Zero Duration Distribution', fontsize=14, fontweight='bold')
            ax6_main.grid(True, alpha=0.3)

        else:
            ax6.text(0.5, 0.5, f'All {len(daily_zero_stats)} days have zero positions', 
                    transform=ax6.transAxes, ha='center', va='center', fontsize=12)
            ax6.set_title('Daily Maximum Zero Position Duration', fontsize=14, fontweight='bold')
        ax7 = axes[3, 0]  
        hourly_pnl_pivot = stats_df.groupby(['date', 'hour'])['pnl_ret'].sum().unstack('hour')
        desired_hours = [9, 10, 11, 13, 14, 21, 22]
        hourly_pnl_pivot = hourly_pnl_pivot.reindex(columns=desired_hours)
        hourly_pnl_pivot.index = hourly_pnl_pivot.index.astype(str)
        # 绘制每条小时线并指定标签
        for hour in hourly_pnl_pivot.columns:
            ax7.plot(hourly_pnl_pivot.index, hourly_pnl_pivot[hour].fillna(0).cumsum(), label=f'{hour}:00')

        # 设置图表属性
        ax7.set_title('Average PnL by Hour of Day', fontsize=14, fontweight='bold')
        ax7.legend(bbox_to_anchor=(1, 1), loc='upper left')  # 将图例放在图表右侧
        ax7.grid(True, alpha=0.3)
        # 设置x轴刻度 - 只显示10个均匀分布的日期标签
        n = len(hourly_pnl_pivot.index)
        step = max(1, n // 10)  # 计算步长，确保至少有1个间隔
        xticks = hourly_pnl_pivot.index[::step]
        ax7.set_xticks(xticks)
        ax7.set_xticklabels(xticks, rotation=45)  # 旋转45度避免重叠

        ax8 = axes[3, 1]  # 假设这是第八个图的位置

        # 计算每日position变化的绝对值之和
        daily_pos_change = stats_df.groupby('date')['pos'].apply(lambda x: x.diff().abs().sum())

        # 绘制柱状图
        ax8.bar(range(len(daily_pos_change)), daily_pos_change.values, color='blue', alpha=0.7)

        # 设置图表属性
        ax8.set_title(f'Daily Position Changes: {daily_pos_change.mean()}', fontsize=14, fontweight='bold')
        ax8.grid(True, alpha=0.3)

        # 设置x轴刻度 - 只显示10个均匀分布的日期标签
        n = len(daily_pos_change)
        step = max(1, n // 10)  # 计算步长，确保至少有1个间隔
        xticks_indices = list(range(0, n, step))
        xtick_labels = [daily_pos_change.index[i] for i in xticks_indices]

        ax8.set_xticks(xticks_indices)
        ax8.set_xticklabels(xtick_labels, rotation=45, ha='right')
    
    elif mode=="无":
        # 计算绩效指标
        daily_returns = stats_df[['pnl_ret', 'date']].groupby('date').sum()
        ret = daily_returns.mean() * 252
        sp = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)).iloc[0]
        maxdd = calc_dd(daily_returns.cumsum() + 1).min().iloc[0]
        
        # 创建包含3个子图的画布
        fig, axes = plt.subplots(4, 2, figsize=(18, 9))
        
        # 图表1: 累计收益和成本滑点
        ax1 = axes[0, 0]
        
        cum_data = stats_df[['pnl_ret', 'cost&slippage_rate', 'date']].groupby('date').sum().cumsum()
        # 在同一个y轴上绘制两条线
        ax1.plot(cum_data.index, cum_data['pnl_ret'], color='blue', label='Cumulative Return', linewidth=2)
        ax1.plot(cum_data.index, cum_data['cost&slippage_rate'], color='red', label='Cost & Slippage', linestyle='--', linewidth=2)

        ax1.set_title(f'{variety} - (Ret: {ret[0]*100:.2f}%, Sharpe: {sp:.2f}, MaxDD: {maxdd:.2f}%)', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')  # 自动选择最佳位置
        ax1.grid(True, alpha=0.3)
        ax1.set_ylabel('Cumulative Rate')
        
        # 图表2: 持仓柱状图
        ax2 = axes[1, 0]
        pos_data = stats_df[['date', 'pos']].replace(0, np.nan).groupby('date').mean()
        ax2.bar(pos_data.index, pos_data['pos'], color='green', alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax2.set_title(f'Average Daily Position', fontsize=14, fontweight='bold')
        # ax2.set_ylabel('Position')
        # ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        
        # 设置x轴刻度数量（避免过于拥挤）
        ax2.xaxis.set_major_locator(MaxNLocator(nbins=8))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0)
        
        # 图表3: 多头/空头和保险率
        ax3 = axes[2 ,0]
        ax3_right = ax3.twinx()
        
        trade_data = stats_df[['date', 'long_pos', 'short_pos', 'Margin_rate']].groupby('date').max()
        ax3.plot(trade_data.index, trade_data['long_pos'], color='blue', label='Long', linewidth=2)
        ax3.plot(trade_data.index, trade_data['short_pos'], color='red', label='Short', linewidth=2)
        ax3_right.plot(trade_data.index, trade_data['Margin_rate'], color='orange', label='Margin Rate', linestyle='--')
        
        ax3.set_title(f'Trading Activity & Margin Rate', fontsize=14, fontweight='bold')
        ax3.legend(loc='upper left')
        ax3_right.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)

        # 图表4: 多头/空头收益率拆解
        ax4 = axes[0, 1]

        # 分别计算多头和空头的累计收益
        long_cum = stats_df[stats_df['pos'] > 0].groupby('date')['pnl_ret'].sum().cumsum()
        short_cum = stats_df[stats_df['pos'] < 0].groupby('date')['pnl_ret'].sum().cumsum()

        # 绘制多头和空头收益曲线
        ax4.plot(long_cum.index, long_cum.values, color='green', label='long_cum', linewidth=2)
        ax4.plot(short_cum.index, short_cum.values, color='red', label='short_cum', linewidth=2)

        # 设置图表属性
        ax4.set_title('long/short_split', fontsize=14, fontweight='bold')
        ax4.legend(loc='upper left')
        ax4.grid(True, alpha=0.3)

        # 图表5: 多头和空头持仓连续性统计（简洁版）
        ax5 = axes[1, 1]

        stats_df['pos_group'] = (stats_df['pos'] != stats_df['pos'].shift()).cumsum()
        grouped = stats_df.groupby('pos_group')
        position_durations = grouped.size()
        position_types = grouped['pos'].first()

        position_df = pd.DataFrame({
            'position': position_types,
            'duration': position_durations
        })

        trade_df = position_df[position_df['position'].isin([1, -1])]
        long_durations = trade_df[trade_df['position'] == 1]['duration']
        short_durations = trade_df[trade_df['position'] == -1]['duration']

        if not long_durations.empty:
            ax5.hist(long_durations, bins=20, alpha=0.6, color='lightgreen', edgecolor='darkgreen')
        if not short_durations.empty:
            ax5.hist(short_durations, bins=20, alpha=0.6, color='lightcoral', edgecolor='darkred')
        stats_text = ""
        if not long_durations.empty:
            stats_text += f"Long: n={len(long_durations)}\nμ={long_durations.mean():.1f}\nσ={long_durations.std():.1f}\n\n"
        if not short_durations.empty:
            stats_text += f"Short: n={len(short_durations)}\nμ={short_durations.mean():.1f}\nσ={short_durations.std():.1f}"

        ax5.text(0.75, 1, stats_text, transform=ax5.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax5.set_title('Position Holding Duration Distribution', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)

        # Chart 6: Maximum Intraday Zero Position Duration
        ax6 = axes[2, 1]  # 假设这是第三行第一列的图表


        # 计算每日最大连续不开仓长度
        daily_zero_stats = max_zero_duration_per_day(stats_df)
        zero_days = len(daily_zero_stats[daily_zero_stats['max_zero_duration'] == 240])
        non_zero_stats = daily_zero_stats[daily_zero_stats['max_zero_duration'] != 240]

        if len(non_zero_stats) > 0:
            # 创建主坐标系
            ax6_main = ax6
            
            # 绘制非零值分布
            n, bins, patches = ax6_main.hist(non_zero_stats['max_zero_duration'], bins=20,
                                        color='blue', alpha=0.7, edgecolor='darkblue')
            
            # 设置y轴上限（为零值天数留出空间）
            y_max = max(n) * 1.2
            ax6_main.set_ylim(0, y_max)
            
            # 添加零值天数标注
            ax6_main.annotate(f'No Trade Days: {zero_days}', 
                            xy=(bins[len(bins)-1], y_max * 0.95),
                            xytext=(0, 20), textcoords='offset points',
                            ha='center', va='bottom', fontsize=10,
                            bbox=dict(boxstyle='round', facecolor='red', alpha=0.3),
                            arrowprops=dict(arrowstyle='->', color='red'))
            
            # 添加统计信息
            stats_text = f"Trade Days: {len(non_zero_stats)}\n"
            stats_text += f"Max no trade Duration: {non_zero_stats['max_zero_duration'].max()}\n"
            stats_text += f"Mean no trade Duration: {non_zero_stats['max_zero_duration'].mean():.1f}"
            
            ax6_main.text(0.75, 0.85, stats_text, transform=ax6_main.transAxes, fontsize=9,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax6_main.set_title('Daily Max Zero Duration Distribution', fontsize=14, fontweight='bold')
            ax6_main.grid(True, alpha=0.3)

        else:
            ax6.text(0.5, 0.5, f'All {len(daily_zero_stats)} days have zero positions', 
                    transform=ax6.transAxes, ha='center', va='center', fontsize=12)
            ax6.set_title('Daily Maximum Zero Position Duration', fontsize=14, fontweight='bold')

        ax7 = axes[3, 0]  
        hourly_pnl_pivot = stats_df.groupby(['date', 'hour'])['pnl_ret'].sum().unstack('hour')
        desired_hours = [9, 10, 11, 13, 14, 21, 22]
        hourly_pnl_pivot = hourly_pnl_pivot.reindex(columns=desired_hours)
        hourly_pnl_pivot.index = hourly_pnl_pivot.index.astype(str)
        # 绘制每条小时线并指定标签
        for hour in hourly_pnl_pivot.columns:
            ax7.plot(hourly_pnl_pivot.index, hourly_pnl_pivot[hour].fillna(0).cumsum(), label=f'{hour}:00')

        # 设置图表属性
        ax7.set_title('Average PnL by Hour of Day', fontsize=14, fontweight='bold')
        ax7.legend(bbox_to_anchor=(1, 1), loc='upper left')  # 将图例放在图表右侧
        ax7.grid(True, alpha=0.3)
        # 设置x轴刻度 - 只显示10个均匀分布的日期标签
        n = len(hourly_pnl_pivot.index)
        step = max(1, n // 10)  # 计算步长，确保至少有1个间隔
        xticks = hourly_pnl_pivot.index[::step]
        ax7.set_xticks(xticks)
        ax7.set_xticklabels(xticks, rotation=45)  # 旋转45度避免重叠

        ax8 = axes[3, 1]  # 假设这是第八个图的位置

        # 计算每日position变化的绝对值之和
        daily_pos_change = stats_df.groupby('date')['pos'].apply(lambda x: x.diff().abs().sum())

        # 绘制柱状图
        ax8.bar(range(len(daily_pos_change)), daily_pos_change.values, color='blue', alpha=0.7)

        # 设置图表属性
        ax8.set_title(f'Daily Position Changes: {daily_pos_change.mean()}', fontsize=14, fontweight='bold')
        ax8.grid(True, alpha=0.3)

        # 设置x轴刻度 - 只显示10个均匀分布的日期标签
        n = len(daily_pos_change)
        step = max(1, n // 10)  # 计算步长，确保至少有1个间隔
        xticks_indices = list(range(0, n, step))
        xtick_labels = [daily_pos_change.index[i] for i in xticks_indices]

        ax8.set_xticks(xticks_indices)
        ax8.set_xticklabels(xtick_labels, rotation=45, ha='right')

    plt.tight_layout()
    plt.show()
    
    return None



def plot_yearly_eval(merged_data):
    import plotly.graph_objects as go

    df = calc_yearly_eval(merged_data)

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Year'] + df.columns.tolist(),
            fill_color='navy',
            font=dict(color='white', size=10),
            align='center',
            height=25  # 固定表头高度
        ),
        cells=dict(
            values=[df.index.astype(str)] + [
                [f'{x:.1f}%' for x in df['ann_ret(%)']],
                [f'{x:.2f}' for x in df['sharpe']],
                [f'{x:.1f}%' for x in df['max_drawdown']],
                [f'{x:.2f}' for x in df['turnover_rate(%)']]
            ],
            fill_color=['white', 'lightgrey']*len(df),
            font=dict(size=9),
            align='center',
            height=25  # 固定行高
        ))
    ])

    # 关键调整：标题居中加粗
    fig.update_layout(
        title={
            'text': '<b>Annual Performance Metrics （年化）</b>',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=14)
        },
        margin=dict(l=10, r=10, b=10, t=30),  # 边距保持不变
        height=(len(df)+1)*50 ,  # 高度计算保持不变
        width=max(1200, len(df.columns)*150),  # 宽度保持不变
        autosize=True  # 保持自动调整
    )

    fig.show()

class ModelBacktester:
    def __init__(self, train_end_date, config):
        self.config = config
        self.window_end = '2026-01-01'

        self.MODEL_DIR_LST = config['MODEL_DIR_LST']
        self.model_group_name = config['model_group_name']

        self.max_trade_inday = 10000
        self.hand_per_trade = 10
        self.money_lst = []

        self.ts_col = 'datetime'
        self.instrument_col = 'instrument'
        self.factor_col = 'factor'
        self.open_drop = True
        self.holding_price_col = config.get('holding_price_col', "")
        self.trading_price_col = config.get('trading_price_col', "")
        self.train_end_date = train_end_date

    def load_config(self, symbol):
        import function_future.DataLoader as DL
        config_loader = DL.InstrumentConfig()
        symbol_config = config_loader.get_instrument_config(symbol)
        self.margin_rate = symbol_config["margin_rate"]
        self.fee_way = symbol_config["fee_way"]
        self.fee = symbol_config["fee"]
        self.fee_mode = symbol_config["fee_comment"]
        self.multiplier = symbol_config["contract_multiplier"]
        self.price_tick = symbol_config["price_tick"]
        self.trading_hours = symbol_config["trading_hours"]
        self.slippage = 0
        print(f"Loaded config for {symbol}: margin_rate={self.margin_rate}, fee_way={self.fee_way}, fee={self.fee}, multiplier={self.multiplier}, price_tick={self.price_tick}")

    def load_factor(self, variety, start_date='20180101', end_date='20260101'):
        factor_single = pd.read_feather(f'/mnt/Data/writable/liaoyuyang/factor/{variety}/all_fac/all_factor.feather')
        factor_single['hour'] = factor_single.datetime.dt.hour
        factor_single = factor_single[factor_single.datetime.astype('datetime64[ns]').between(start_date, end_date)]
        factor_single = factor_single.sort_values('datetime').drop_duplicates('datetime', keep='last')
        self.factor_single = factor_single.round(8)

    def load_mktdata(self, variety, start_date='20180101', end_date ='20260101'):
        data = pd.read_csv(f'/mnt/Data/writable/liaoyuyang/data/1min/active/main_{variety}.csv', index_col=0).rename(columns={'ts': 'datetime'})
        data = data[data.datetime.astype('datetime64[ns]').between(start_date, end_date)]
        data = data.rename(columns={'code': 'instrument'})
        data = data.sort_values('datetime').drop_duplicates('datetime', keep='last')
        self.mkt_data = data

    def prepare_data(self, factor_df, price_df):        
        factor_df[self.ts_col] = pd.to_datetime(factor_df[self.ts_col])
        price_df[self.ts_col] = pd.to_datetime(price_df[self.ts_col])
        
        merged = pd.merge(
            price_df.sort_values(self.ts_col),
            factor_df.sort_values(self.ts_col),
            on=[self.ts_col,self.instrument_col], 
            how='right'
        )
        return merged.reset_index(drop=True)

    def load_models(self):
        """加载所有 .lgb 模型文件和 .json 元数据文件"""

        self.model_group = {}
        self.model_group_metadata = {}  
        
        for n, group_name in enumerate(self.model_group_name):
            self.model_group[group_name] = {}
            self.model_group_metadata[group_name] = {}

            model_dir = self.MODEL_DIR_LST[n]
            print(model_dir)

            for lgb_file in model_dir.glob("*.lgb"):
                try:
                    model = lgb.Booster(model_file=str(lgb_file))
                    
                    meta_file = lgb_file.with_name(lgb_file.stem + '_meta.json')
                    if meta_file.exists():
                        with open(meta_file, 'r') as f:
                            metadata = json.load(f)
                    else:
                        metadata = {}
                    
                    self.model_group[group_name][lgb_file.stem] = model
                    self.model_group[group_name][lgb_file.stem].best_iteration = metadata["best_iteration"]
                    self.model_group_metadata[group_name][lgb_file.stem] = metadata
                    
                    # print(f"✅ 加载模型: {lgb_file.stem}")
                    
                except Exception as e:
                    print(f"❌ 加载模型失败 {lgb_file.stem}: {e}")
                    continue
            
            # print(f"已加载 {len(self.model_group[group_name])} 个模型")
        
            if self.model_group[group_name]:
                first_model = list(self.model_group[group_name].values())[0]
                all_features = first_model.feature_name()
                print(f"总共用到了 {len(all_features)} 个特征")
                self.all_features = list(all_features)
                
                # for name, metadata in sorted(self.model_group_metadata[group_name].items()):
                #     best_iter = metadata.get('best_iteration', '未知')
                #     test_corr = metadata.get('test_corr', '未知')
                    # print(f"模型 {name}: best_iteration={best_iter}, test_corr={test_corr}")
            else:
                print("警告：未加载到任何模型")
                self.all_features = []
        
        return None

    def generate_predictions(self):
        """Generate predictions for all models"""
        self.predictions = {}

        for n, group_name in enumerate(self.model_group_name):
            self.predictions[group_name] = {}  
            for model_name, model in sorted(self.model_group[group_name].items()):
                feature_name = self.model_group[group_name][model_name].feature_name()
                X_tmp = self.factor_single[feature_name].values 
                preds = model.predict(X_tmp)
                
                pred_df = pd.DataFrame({
                    self.ts_col: self.factor_single[self.ts_col],
                    self.instrument_col: self.factor_single[self.instrument_col],
                    model_name: preds,
                    'best_round' :model.best_iteration
                })

                pred_df = pred_df[pred_df[self.ts_col]>=group_name]
                self.predictions[group_name][model_name] = pred_df.reset_index(drop=True)
                

    def combine_models(self, by, avg=True):
        if by == 'best_iteration_log_weighted':
            for n, group_name in enumerate(self.model_group_name):
                pred_lst = []
                weight_lst = []

                for model_name, prediction in self.predictions[group_name].items():
                    pred_df = prediction
                    best_iteration = self.model_group_metadata[group_name][model_name]['best_iteration']
                    pred_lst.append(pred_df[model_name])
                    weight_lst.append(np.log(best_iteration + 1))

                pred_df = pd.concat(pred_lst, axis=1)  
                self.pred_all = pred_df.copy()[sorted(pred_df.columns)]
                self.pred_all.index = prediction[self.ts_col]

                weighted_avg = pred_df.mul(weight_lst).sum(axis=1) / sum(weight_lst)

                pred_df = pd.DataFrame({
                    self.ts_col: prediction[self.ts_col],
                    self.instrument_col: prediction[self.instrument_col],
                    by: weighted_avg
                })
                if avg:
                    pred_df[by] = pred_df[by] * 0.6 + pred_df[by].shift() * 0.3 + pred_df[by].shift(2) * 0.1
                self.predictions[group_name][by] = pred_df

            return None

    def calc_pos_table_fee_rate_suocang(self, merged_data, th1, th2, holding_bars):
        print("使用的交易价格", self.trading_price_col)
        df = merged_data[['datetime', 'pos', 'close', self.trading_price_col, 'trade_date']].copy()
        df[self.trading_price_col] = df[self.trading_price_col].ffill()
        df[['hand', 'cash', 'market_value', 'init_price', 'cost&slippage', 'margin_occupied', 'equity',
            'Floating P&L', 'long_today', 'long_yesturday', 'short_today', 'short_yesturday', 'margin_used', 'Margin_rate']] = np.nan

        df.loc[[0,1], ['hand', 'market_value', 'cost&slippage', 'margin_occupied', 'long_today', 'long_yesturday', 'short_today', 'short_yesturday', 'margin_used']] = 0
        df.loc[[0,1], ['cash', 'equity']] = self.money

        date_dict = {
            'trade_today': 0,
            'margin_used': 0,
            'long_pos_today': 0,
            'long_pos_yesturday': 0,
            'short_pos_today': 0,
            'short_pos_yesturday': 0
                     }

        for ind in tqdm(range(2, len(merged_data)), desc=f'{th1}_{th2}_{holding_bars}'):

            if df.loc[ind, 'datetime'].date() != df.loc[ind-1, 'datetime'].date():
                date_dict['trade_today'] = 0
                date_dict['long_pos_yesturday'] += date_dict['long_pos_today']
                date_dict['long_pos_today'] = 0
                date_dict['short_pos_yesturday'] += date_dict['short_pos_today']
                date_dict['short_pos_today'] = 0

            # --------------------------当前仓位与上一根的仓位不变的情况--------------------------------------
            if df.loc[ind-1, 'pos'] == df.loc[ind-2, 'pos']:
                df.loc[ind, 'init_price'] = df.loc[ind-1, 'init_price']
                df.loc[ind, 'hand'] = df.loc[ind-1, 'hand']
                df.loc[ind, 'cash'] = df.loc[ind-1, 'cash']
                df.loc[ind, 'market_value'] = df.loc[ind, 'hand'] * self.multiplier * df.loc[ind, 'close']
                df.loc[ind, 'margin_occupied'] = np.abs(df.loc[ind, 'market_value']) * self.margin_rate
                df.loc[ind, 'equity'] = df.loc[ind, 'cash'] + df.loc[ind, 'market_value']
                df.loc[ind, 'Floating P&L'] = (df.loc[ind, 'close'] - df.loc[ind, 'init_price']) * df.loc[ind, 'hand'] * self.multiplier 

            # --------------------------当前仓位与上一根的仓位不同的情况（上一根无仓位）-----------------------
            elif df.loc[ind-2, 'pos'] == 0 : # 操作前为0， 操作后为1或者-1, 此刻要买卖
                df.loc[ind, 'init_price'] = df.loc[ind, self.trading_price_col]
                if df.loc[ind-1, 'pos'] == 1:
                    df.loc[ind, 'hand'] = self.hand_per_trade

                    # 更新当前成交状态
                    if date_dict['short_pos_yesturday'] > 0: date_dict['short_pos_yesturday'] -= self.hand_per_trade
                    else: date_dict['long_pos_today'] += self.hand_per_trade
                    date_dict['trade_today'] += 1

                    trade_val = df.loc[ind, 'hand'] * self.multiplier * df.loc[ind, self.trading_price_col] 
                    cost = trade_val * self.fee
                    slippage = trade_val * self.slippage             

                    df.loc[ind, 'cost&slippage'] = cost + slippage
                    df.loc[ind, 'cash'] = df.loc[ind-1, 'cash'] - trade_val - cost - slippage
                    df.loc[ind, 'market_value'] = df.loc[ind, 'hand'] * self.multiplier * df.loc[ind, 'close']
                    df.loc[ind, 'margin_occupied'] = np.abs(df.loc[ind, 'market_value']) * self.margin_rate
                    df.loc[ind, 'equity'] = df.loc[ind, 'cash'] + df.loc[ind, 'market_value']
                    df.loc[ind, 'Floating P&L'] = (df.loc[ind, 'close'] - df.loc[ind, 'init_price']) * df.loc[ind, 'hand'] * self.multiplier 

                else:
                    df.loc[ind, 'hand'] = -self.hand_per_trade

                    # 更新当前成交状态
                    if date_dict['long_pos_yesturday'] > 0: date_dict['long_pos_yesturday'] -= self.hand_per_trade
                    else: date_dict['short_pos_today'] += self.hand_per_trade
                    date_dict['trade_today'] += 1

                    trade_val = -df.loc[ind, 'hand'] * self.multiplier * df.loc[ind, self.trading_price_col] 
                    cost = trade_val * self.fee
                    slippage = trade_val * self.slippage                       

                    df.loc[ind, 'cost&slippage'] = cost + slippage
                    df.loc[ind, 'cash'] = df.loc[ind-1, 'cash'] + trade_val  - cost - slippage
                    df.loc[ind, 'market_value'] = df.loc[ind, 'hand'] * self.multiplier * df.loc[ind, 'close']
                    df.loc[ind, 'margin_occupied'] = np.abs(df.loc[ind, 'market_value']) * self.margin_rate
                    df.loc[ind, 'equity'] = df.loc[ind, 'cash'] + df.loc[ind, 'market_value']
                    df.loc[ind, 'Floating P&L'] = (df.loc[ind, 'close'] - df.loc[ind, 'init_price']) * df.loc[ind, 'hand'] * self.multiplier 
            
            # --------------------------当前仓位与上一根的仓位不同的情况（上一根多头）-----------------------
            elif df.loc[ind-2, 'pos'] == 1 : # 操作前为1， 操作后为0或者-1, 此刻要平多或者平多后反向开空
                if df.loc[ind-1, 'pos'] == 0:
                    df.loc[ind, 'hand'] = 0

                    # 更新当前成交状态
                    if date_dict['long_pos_yesturday'] > 0: date_dict['long_pos_yesturday'] -= self.hand_per_trade
                    else: date_dict['short_pos_today'] += self.hand_per_trade

                    trade_val = df.loc[ind-1, 'hand'] * self.multiplier * df.loc[ind, self.trading_price_col] 
                    cost = trade_val * self.fee
                    slippage = trade_val * self.slippage     

                    df.loc[ind, 'cost&slippage'] = cost + slippage
                    df.loc[ind, 'cash'] = df.loc[ind-1, 'cash'] + trade_val - cost - slippage
                    df.loc[ind, 'market_value'] = 0
                    df.loc[ind, 'margin_occupied'] = 0
                    df.loc[ind, 'equity'] = df.loc[ind, 'cash'] + df.loc[ind, 'market_value']
                    df.loc[ind, 'Floating P&L'] = (df.loc[ind, self.trading_price_col] - df.loc[ind-1, 'init_price']) * df.loc[ind, 'hand'] * self.multiplier 

                if df.loc[ind-1, 'pos'] == -1:
                    df.loc[ind, 'hand'] = -self.hand_per_trade

                    # 更新当前成交状态
                    if date_dict['long_pos_yesturday'] >= self.hand_per_trade * 2: date_dict['long_pos_yesturday'] -= self.hand_per_trade * 2 
                    elif date_dict['long_pos_yesturday'] == self.hand_per_trade: 
                        date_dict['long_pos_yesturday'] -= self.hand_per_trade
                        date_dict['short_pos_today'] += self.hand_per_trade
                    else: date_dict['short_pos_today'] += self.hand_per_trade * 2
                    date_dict['trade_today'] += 1

                    df.loc[ind, 'init_price'] = df.loc[ind, self.trading_price_col]
                    trade_val = df.loc[ind-1, 'hand'] * self.multiplier * df.loc[ind, self.trading_price_col] * 2
                    cost = trade_val * self.fee
                    slippage = trade_val * self.slippage     

                    df.loc[ind, 'cost&slippage'] = cost + slippage
                    
                    df.loc[ind, 'cash'] = df.loc[ind-1, 'cash'] + trade_val  - cost - slippage
                    df.loc[ind, 'market_value'] = df.loc[ind, 'hand'] * self.multiplier * df.loc[ind, 'close']
                    df.loc[ind, 'margin_occupied'] = np.abs(df.loc[ind, 'market_value']) * self.margin_rate
                    df.loc[ind, 'equity'] = df.loc[ind, 'cash'] + df.loc[ind, 'market_value']
                    df.loc[ind, 'Floating P&L'] = (df.loc[ind, 'close'] - df.loc[ind, 'init_price']) * df.loc[ind, 'hand'] * self.multiplier 

            # --------------------------当前仓位与上一根的仓位不变的情况（上一根空头）-----------------------
            elif df.loc[ind-2, 'pos'] == -1 : # 操作前为-1， 操作后为0或者1, 此刻要平空或者平空后反向开多
                if df.loc[ind-1, 'pos'] == 0:
                    df.loc[ind, 'hand'] = 0

                    # 更新当前成交状态
                    if date_dict['short_pos_yesturday'] > 0: date_dict['short_pos_yesturday'] -= self.hand_per_trade
                    else: date_dict['long_pos_today'] += self.hand_per_trade                    

                    trade_val = -df.loc[ind-1, 'hand'] * self.multiplier * df.loc[ind, self.trading_price_col] 
                    cost = trade_val * self.fee
                    slippage = trade_val * self.slippage     

                    df.loc[ind, 'cost&slippage'] = cost + slippage
                    
                    df.loc[ind, 'cash'] = df.loc[ind-1, 'cash'] - trade_val - cost - slippage
                    df.loc[ind, 'market_value'] = 0
                    df.loc[ind, 'margin_occupied'] = 0
                    df.loc[ind, 'equity'] = df.loc[ind, 'cash'] + df.loc[ind, 'market_value']
                    df.loc[ind, 'Floating P&L'] = (df.loc[ind, self.trading_price_col] - df.loc[ind-1, 'init_price']) * df.loc[ind, 'hand'] * self.multiplier 

                if df.loc[ind-1, 'pos'] == 1:
                    df.loc[ind, 'hand'] = self.hand_per_trade

                    # 更新当前成交状态
                    if date_dict['short_pos_yesturday'] >= self.hand_per_trade: date_dict['short_pos_yesturday'] -= self.hand_per_trade * 2 
                    elif date_dict['short_pos_yesturday'] == self.hand_per_trade: 
                        date_dict['short_pos_yesturday'] -= self.hand_per_trade
                        date_dict['long_pos_today'] += self.hand_per_trade
                    else: date_dict['long_pos_today'] += self.hand_per_trade * 2
                    date_dict['trade_today'] += 1

                    df.loc[ind, 'init_price'] = df.loc[ind, self.trading_price_col]
                    trade_val = -df.loc[ind-1, 'hand'] * self.multiplier * df.loc[ind, self.trading_price_col] * 2
                    cost = trade_val * self.fee
                    slippage = trade_val * self.slippage     

                    df.loc[ind, 'cost&slippage'] = cost + slippage
                    df.loc[ind, 'cash'] = df.loc[ind-1, 'cash'] - trade_val - cost - slippage
                    df.loc[ind, 'market_value'] = df.loc[ind, 'hand'] * self.multiplier * df.loc[ind, 'close']
                    df.loc[ind, 'margin_occupied'] = np.abs(df.loc[ind, 'market_value']) * self.margin_rate
                    df.loc[ind, 'equity'] = df.loc[ind, 'cash'] + df.loc[ind, 'market_value']
                    df.loc[ind, 'Floating P&L'] = (df.loc[ind, 'close'] - df.loc[ind, 'init_price']) * df.loc[ind, 'hand'] * self.multiplier 

            else:
                print(df.loc[ind, 'datetime'])
                break
            
            df.loc[ind, 'long_today'] = date_dict['long_pos_today']
            df.loc[ind, 'long_yesturday'] = date_dict['long_pos_yesturday']
            df.loc[ind, 'short_today'] = date_dict['short_pos_today']
            df.loc[ind, 'short_yesturday'] = date_dict['short_pos_yesturday']
            df.loc[ind, 'margin_used'] = (df.loc[ind, ['long_today', 'long_yesturday', 'short_today', 'short_yesturday']].sum() * 
                                        (df.loc[ind, 'close'] if pd.notna(df.loc[ind, 'close']) else 0) * 
                                        self.multiplier * self.margin_rate)            
            df.loc[ind, 'Margin_rate'] = df.loc[ind, 'margin_used'] / self.money 
        df['pnl_ret'] = (df['equity'].diff() / self.money).fillna(0)
        df['pnl_ret_cum'] = df['pnl_ret'].cumsum().ffill()
        df['cost&slippage_cum'] = df['cost&slippage'].cumsum().ffill()
        df['cost&slippage_rate'] = df['cost&slippage'] / self.money
        df['date'] = df.datetime.dt.date
        df['month'] = df.datetime.dt.to_period('1M')
        df['year'] = df.datetime.dt.year
        return df

    def calc_pos_table_fee_rate_normal(self, merged_data):
        print("使用的交易价格", self.trading_price_col)
        df = merged_data[['datetime', 'pos', 'close', self.trading_price_col, 'trade_date']].copy()
        df[self.trading_price_col] = df[self.trading_price_col].ffill()
        df[['hand', 'cash', 'market_value', 'init_price', 'cost&slippage', 'margin_occupied', 'equity',
            'Floating P&L', 'long_pos', 'short_pos', 'margin_used', 'Margin_rate']] = np.nan

        df.loc[[0,1], ['hand', 'market_value', 'cost&slippage', 'margin_occupied', 'long_pos', 'short_pos', 'margin_used']] = 0
        df.loc[[0,1], ['cash', 'equity']] = self.money

        date_dict = {
            'trade_today': 0,
            'margin_used': 0,
            'long_pos': 0,
            'short_pos': 0
                     }

        for ind in tqdm(range(2, len(merged_data)), desc="计算持仓表"):

            # --------------------------当前仓位与上一根的仓位不变的情况--------------------------------------
            if df.loc[ind-1, 'pos'] == df.loc[ind-2, 'pos']:
                df.loc[ind, 'init_price'] = df.loc[ind-1, 'init_price']
                df.loc[ind, 'hand'] = df.loc[ind-1, 'hand']
                df.loc[ind, 'cash'] = df.loc[ind-1, 'cash']
                df.loc[ind, 'market_value'] = df.loc[ind, 'hand'] * self.multiplier * df.loc[ind, 'close']
                df.loc[ind, 'margin_occupied'] = np.abs(df.loc[ind, 'market_value']) * self.margin_rate
                df.loc[ind, 'equity'] = df.loc[ind, 'cash'] + df.loc[ind, 'market_value']
                df.loc[ind, 'Floating P&L'] = (df.loc[ind, 'close'] - df.loc[ind, 'init_price']) * df.loc[ind, 'hand'] * self.multiplier 

            # --------------------------当前仓位与上一根的仓位不同的情况（上一根无仓位）-----------------------
            elif df.loc[ind-2, 'pos'] == 0 : # 操作前为0， 操作后为1或者-1, 此刻要买卖
                df.loc[ind, 'init_price'] = df.loc[ind, self.trading_price_col]
                if df.loc[ind-1, 'pos'] == 1:
                    df.loc[ind, 'hand'] = self.hand_per_trade

                    # 更新当前成交状态
                    if date_dict['short_pos'] > 0: date_dict['short_pos'] -= self.hand_per_trade
                    else: date_dict['long_pos'] += self.hand_per_trade
                    date_dict['trade_today'] += 1

                    trade_val = df.loc[ind, 'hand'] * self.multiplier * df.loc[ind, self.trading_price_col] 
                    cost = trade_val * self.fee
                    slippage = trade_val * self.slippage             

                    df.loc[ind, 'cost&slippage'] = cost + slippage
                    df.loc[ind, 'cash'] = df.loc[ind-1, 'cash'] - trade_val - cost - slippage
                    df.loc[ind, 'market_value'] = df.loc[ind, 'hand'] * self.multiplier * df.loc[ind, 'close']
                    df.loc[ind, 'margin_occupied'] = np.abs(df.loc[ind, 'market_value']) * self.margin_rate
                    df.loc[ind, 'equity'] = df.loc[ind, 'cash'] + df.loc[ind, 'market_value']
                    df.loc[ind, 'Floating P&L'] = (df.loc[ind, 'close'] - df.loc[ind, 'init_price']) * df.loc[ind, 'hand'] * self.multiplier 

                else:
                    df.loc[ind, 'hand'] = -self.hand_per_trade

                    # 更新当前成交状态
                    if date_dict['long_pos'] > 0: date_dict['long_pos'] -= self.hand_per_trade
                    else: date_dict['short_pos'] += self.hand_per_trade
                    date_dict['trade_today'] += 1  

                    trade_val = -df.loc[ind, 'hand'] * self.multiplier * df.loc[ind, self.trading_price_col] 
                    cost = trade_val * self.fee
                    slippage = trade_val * self.slippage                       

                    df.loc[ind, 'cost&slippage'] = cost + slippage
                    df.loc[ind, 'cash'] = df.loc[ind-1, 'cash'] + trade_val  - cost - slippage
                    df.loc[ind, 'market_value'] = df.loc[ind, 'hand'] * self.multiplier * df.loc[ind, 'close']
                    df.loc[ind, 'margin_occupied'] = np.abs(df.loc[ind, 'market_value']) * self.margin_rate
                    df.loc[ind, 'equity'] = df.loc[ind, 'cash'] + df.loc[ind, 'market_value']
                    df.loc[ind, 'Floating P&L'] = (df.loc[ind, 'close'] - df.loc[ind, 'init_price']) * df.loc[ind, 'hand'] * self.multiplier 
            
            # --------------------------当前仓位与上一根的仓位不同的情况（上一根多头）-----------------------
            elif df.loc[ind-2, 'pos'] == 1 : # 操作前为1， 操作后为0或者-1, 此刻要平多或者平多后反向开空
                if df.loc[ind-1, 'pos'] == 0:
                    df.loc[ind, 'hand'] = 0

                    # 更新当前成交状态
                    if date_dict['long_pos'] > 0: date_dict['long_pos'] -= self.hand_per_trade
                    else: date_dict['short_pos'] += self.hand_per_trade  

                    trade_val = df.loc[ind-1, 'hand'] * self.multiplier * df.loc[ind, self.trading_price_col] 
                    cost = trade_val * self.fee
                    slippage = trade_val * self.slippage     

                    df.loc[ind, 'cost&slippage'] = cost + slippage
                    df.loc[ind, 'cash'] = df.loc[ind-1, 'cash'] + trade_val - cost - slippage
                    df.loc[ind, 'market_value'] = 0
                    df.loc[ind, 'margin_occupied'] = 0
                    df.loc[ind, 'equity'] = df.loc[ind, 'cash'] + df.loc[ind, 'market_value']
                    df.loc[ind, 'Floating P&L'] = (df.loc[ind, self.trading_price_col] - df.loc[ind-1, 'init_price']) * df.loc[ind, 'hand'] * self.multiplier 

                if df.loc[ind-1, 'pos'] == -1:
                    df.loc[ind, 'hand'] = -self.hand_per_trade

                    # 更新当前成交状态
                    if date_dict['long_pos'] >= self.hand_per_trade * 2: date_dict['long_pos'] -= self.hand_per_trade * 2 
                    elif date_dict['long_pos'] == self.hand_per_trade: 
                        date_dict['long_pos'] -= self.hand_per_trade
                        date_dict['short_pos'] += self.hand_per_trade
                    else: date_dict['short_pos'] += self.hand_per_trade * 2
                    date_dict['trade_today'] += 1

                    df.loc[ind, 'init_price'] = df.loc[ind, self.trading_price_col]
                    trade_val = df.loc[ind-1, 'hand'] * self.multiplier * df.loc[ind, self.trading_price_col] * 2
                    cost = trade_val * self.fee
                    slippage = trade_val * self.slippage     

                    df.loc[ind, 'cost&slippage'] = cost + slippage
                    
                    df.loc[ind, 'cash'] = df.loc[ind-1, 'cash'] + trade_val  - cost - slippage
                    df.loc[ind, 'market_value'] = df.loc[ind, 'hand'] * self.multiplier * df.loc[ind, 'close']
                    df.loc[ind, 'margin_occupied'] = np.abs(df.loc[ind, 'market_value']) * self.margin_rate
                    df.loc[ind, 'equity'] = df.loc[ind, 'cash'] + df.loc[ind, 'market_value']
                    df.loc[ind, 'Floating P&L'] = (df.loc[ind, 'close'] - df.loc[ind, 'init_price']) * df.loc[ind, 'hand'] * self.multiplier 

            # --------------------------当前仓位与上一根的仓位不变的情况（上一根空头）-----------------------
            elif df.loc[ind-2, 'pos'] == -1 : # 操作前为-1， 操作后为0或者1, 此刻要平空或者平空后反向开多
                if df.loc[ind-1, 'pos'] == 0:
                    df.loc[ind, 'hand'] = 0

                    # 更新当前成交状态
                    if date_dict['short_pos'] > 0: date_dict['short_pos'] -= self.hand_per_trade
                    else: date_dict['long_pos'] += self.hand_per_trade                   

                    trade_val = -df.loc[ind-1, 'hand'] * self.multiplier * df.loc[ind, self.trading_price_col] 
                    cost = trade_val * self.fee
                    slippage = trade_val * self.slippage     

                    df.loc[ind, 'cost&slippage'] = cost + slippage
                    
                    df.loc[ind, 'cash'] = df.loc[ind-1, 'cash'] - trade_val - cost - slippage
                    df.loc[ind, 'market_value'] = 0
                    df.loc[ind, 'margin_occupied'] = 0
                    df.loc[ind, 'equity'] = df.loc[ind, 'cash'] + df.loc[ind, 'market_value']
                    df.loc[ind, 'Floating P&L'] = (df.loc[ind, self.trading_price_col] - df.loc[ind-1, 'init_price']) * df.loc[ind, 'hand'] * self.multiplier 

                if df.loc[ind-1, 'pos'] == 1:
                    df.loc[ind, 'hand'] = self.hand_per_trade

                    # 更新当前成交状态
                    if date_dict['short_pos'] >= self.hand_per_trade: date_dict['short_pos'] -= self.hand_per_trade * 2 
                    elif date_dict['short_pos'] == self.hand_per_trade: 
                        date_dict['short_pos'] -= self.hand_per_trade
                        date_dict['long_pos'] += self.hand_per_trade
                    else: date_dict['long_pos'] += self.hand_per_trade * 2
                    date_dict['trade_today'] += 1


                    df.loc[ind, 'init_price'] = df.loc[ind, self.trading_price_col]
                    trade_val = -df.loc[ind-1, 'hand'] * self.multiplier * df.loc[ind, self.trading_price_col] * 2
                    cost = trade_val * self.fee
                    slippage = trade_val * self.slippage     

                    df.loc[ind, 'cost&slippage'] = cost + slippage
                    df.loc[ind, 'cash'] = df.loc[ind-1, 'cash'] - trade_val - cost - slippage
                    df.loc[ind, 'market_value'] = df.loc[ind, 'hand'] * self.multiplier * df.loc[ind, 'close']
                    df.loc[ind, 'margin_occupied'] = np.abs(df.loc[ind, 'market_value']) * self.margin_rate
                    df.loc[ind, 'equity'] = df.loc[ind, 'cash'] + df.loc[ind, 'market_value']
                    df.loc[ind, 'Floating P&L'] = (df.loc[ind, 'close'] - df.loc[ind, 'init_price']) * df.loc[ind, 'hand'] * self.multiplier 

            else:
                print(df.loc[ind, 'datetime'])
                break
            
            df.loc[ind, 'long_pos'] = date_dict['long_pos']
            df.loc[ind, 'short_pos'] = date_dict['short_pos']
            df.loc[ind, 'margin_used'] = (df.loc[ind, ['long_pos','short_pos']].sum() * 
                                        (df.loc[ind, 'close'] if pd.notna(df.loc[ind, 'close']) else 0) * 
                                        self.multiplier * self.margin_rate)            
            df.loc[ind, 'Margin_rate'] = df.loc[ind, 'margin_used'] / self.money 

        df['pnl_ret'] = (df['equity'].diff() / self.money).fillna(0)
        df['pnl_ret_cum'] = df['pnl_ret'].cumsum().ffill()
        df['cost&slippage_cum'] = df['cost&slippage'].cumsum().ffill()
        df['cost&slippage_rate'] = df['cost&slippage'] / self.money
        df['month'] = df.datetime.dt.to_period('1M')
        df['year'] = df.datetime.dt.year
        return df

    def calc_pos_table_fee_number_normal(self, merged_data):
        df = merged_data[['datetime', 'pos', 'close', self.trading_price_col, 'trade_date']].copy()
        df[self.trading_price_col] = df[self.trading_price_col].ffill()
        df[['hand', 'cash', 'market_value', 'init_price', 'cost&slippage', 'margin_occupied', 'equity',
            'Floating P&L', 'long_pos', 'short_pos', 'margin_used', 'Margin_rate']] = np.nan

        df.loc[[0,1], ['hand', 'market_value', 'cost&slippage', 'margin_occupied', 'long_pos', 'short_pos', 'margin_used']] = 0
        df.loc[[0,1], ['cash', 'equity']] = self.money

        date_dict = {
            'trade_today': 0,
            'margin_used': 0,
            'long_pos': 0,
            'short_pos': 0
                     }

        cost = self.fee * self.hand_per_trade
        slippage = self.slippage * self.hand_per_trade

        for ind in tqdm(range(2, len(merged_data))):
            
            # --------------------------当前仓位与上一根的仓位不变的情况--------------------------------------
            if df.loc[ind-1, 'pos'] == df.loc[ind-2, 'pos']:
                df.loc[ind, 'init_price'] = df.loc[ind-1, 'init_price']
                df.loc[ind, 'hand'] = df.loc[ind-1, 'hand']
                df.loc[ind, 'cash'] = df.loc[ind-1, 'cash']
                df.loc[ind, 'market_value'] = df.loc[ind, 'hand'] * self.multiplier * df.loc[ind, 'close']
                df.loc[ind, 'margin_occupied'] = np.abs(df.loc[ind, 'market_value']) * self.margin_rate
                df.loc[ind, 'equity'] = df.loc[ind, 'cash'] + df.loc[ind, 'market_value']
                df.loc[ind, 'Floating P&L'] = (df.loc[ind, 'close'] - df.loc[ind, 'init_price']) * df.loc[ind, 'hand'] * self.multiplier 

            # --------------------------当前仓位与上一根的仓位不同的情况（上一根无仓位）-----------------------
            elif df.loc[ind-2, 'pos'] == 0 : # 操作前为0， 操作后为1或者-1, 此刻要买卖
                df.loc[ind, 'init_price'] = df.loc[ind, self.trading_price_col]
                if df.loc[ind-1, 'pos'] == 1:
                    df.loc[ind, 'hand'] = self.hand_per_trade

                    # 更新当前成交状态
                    if date_dict['short_pos'] > 0: date_dict['short_pos'] -= self.hand_per_trade
                    else: date_dict['long_pos'] += self.hand_per_trade
                    date_dict['trade_today'] += 1          

                    trade_val = abs(df.loc[ind, 'hand'] - df.loc[ind-1, 'hand']) * self.multiplier * df.loc[ind, self.trading_price_col]
                    df.loc[ind, 'cost&slippage'] = cost + slippage
                    
                    df.loc[ind, 'cash'] = df.loc[ind-1, 'cash'] - trade_val - cost - slippage
                    df.loc[ind, 'market_value'] = df.loc[ind, 'hand'] * self.multiplier * df.loc[ind, 'close']
                    df.loc[ind, 'margin_occupied'] = np.abs(df.loc[ind, 'market_value']) * self.margin_rate
                    df.loc[ind, 'equity'] = df.loc[ind, 'cash'] + df.loc[ind, 'market_value']
                    df.loc[ind, 'Floating P&L'] = (df.loc[ind, 'close'] - df.loc[ind, 'init_price']) * df.loc[ind, 'hand'] * self.multiplier 

                else:
                    df.loc[ind, 'hand'] = -self.hand_per_trade

                    # 更新当前成交状态
                    if date_dict['long_pos'] > 0: date_dict['long_pos'] -= self.hand_per_trade
                    else: date_dict['short_pos'] += self.hand_per_trade
                    date_dict['trade_today'] += 1          

                    trade_val = abs(df.loc[ind, 'hand'] - df.loc[ind-1, 'hand']) * self.multiplier * df.loc[ind, self.trading_price_col]     
                    df.loc[ind, 'cost&slippage'] = cost + slippage

                    df.loc[ind, 'cash'] = df.loc[ind-1, 'cash'] + trade_val  - cost - slippage
                    df.loc[ind, 'market_value'] = df.loc[ind, 'hand'] * self.multiplier * df.loc[ind, 'close']
                    df.loc[ind, 'margin_occupied'] = np.abs(df.loc[ind, 'market_value']) * self.margin_rate
                    df.loc[ind, 'equity'] = df.loc[ind, 'cash'] + df.loc[ind, 'market_value']
                    df.loc[ind, 'Floating P&L'] = (df.loc[ind, 'close'] - df.loc[ind, 'init_price']) * df.loc[ind, 'hand'] * self.multiplier 
            
            # --------------------------当前仓位与上一根的仓位不同的情况（上一根多头）-----------------------
            elif df.loc[ind-2, 'pos'] == 1 : # 操作前为1， 操作后为0或者-1, 此刻要平多或者平多后反向开空
                if df.loc[ind-1, 'pos'] == 0:
                    df.loc[ind, 'hand'] = 0

                    # 更新当前成交状态
                    if date_dict['long_pos'] > 0: date_dict['long_pos'] -= self.hand_per_trade
                    else: date_dict['short_pos'] += self.hand_per_trade  

                    trade_val = abs(df.loc[ind, 'hand'] - df.loc[ind-1, 'hand']) * self.multiplier * df.loc[ind, self.trading_price_col]
                    df.loc[ind, 'cost&slippage'] = cost + slippage

                    df.loc[ind, 'cash'] = df.loc[ind-1, 'cash'] + trade_val - cost - slippage
                    df.loc[ind, 'market_value'] = 0
                    df.loc[ind, 'margin_occupied'] = 0
                    df.loc[ind, 'equity'] = df.loc[ind, 'cash'] + df.loc[ind, 'market_value']
                    df.loc[ind, 'Floating P&L'] = (df.loc[ind, self.trading_price_col] - df.loc[ind-1, 'init_price']) * df.loc[ind, 'hand'] * self.multiplier 

                if df.loc[ind-1, 'pos'] == -1:
                    df.loc[ind, 'hand'] = -self.hand_per_trade

                    # 更新当前成交状态
                    if date_dict['long_pos'] >= self.hand_per_trade * 2: date_dict['long_pos'] -= self.hand_per_trade * 2 
                    elif date_dict['long_pos'] == self.hand_per_trade: 
                        date_dict['long_pos'] -= self.hand_per_trade
                        date_dict['short_pos'] += self.hand_per_trade
                    else: date_dict['short_pos'] += self.hand_per_trade * 2
                    date_dict['trade_today'] += 1

                    df.loc[ind, 'init_price'] = df.loc[ind, self.trading_price_col]   
                    trade_val = abs(df.loc[ind, 'hand'] - df.loc[ind-1, 'hand']) * self.multiplier * df.loc[ind, self.trading_price_col]
                    df.loc[ind, 'cost&slippage'] = cost + slippage
                    
                    df.loc[ind, 'cash'] = df.loc[ind-1, 'cash'] + trade_val  - cost - slippage
                    df.loc[ind, 'market_value'] = df.loc[ind, 'hand'] * self.multiplier * df.loc[ind, 'close']
                    df.loc[ind, 'margin_occupied'] = np.abs(df.loc[ind, 'market_value']) * self.margin_rate
                    df.loc[ind, 'equity'] = df.loc[ind, 'cash'] + df.loc[ind, 'market_value']
                    df.loc[ind, 'Floating P&L'] = (df.loc[ind, 'close'] - df.loc[ind, 'init_price']) * df.loc[ind, 'hand'] * self.multiplier 

            # --------------------------当前仓位与上一根的仓位不变的情况（上一根空头）-----------------------
            elif df.loc[ind-2, 'pos'] == -1 : # 操作前为-1， 操作后为0或者1, 此刻要平空或者平空后反向开多
                if df.loc[ind-1, 'pos'] == 0:
                    df.loc[ind, 'hand'] = 0

                    # 更新当前成交状态
                    if date_dict['short_pos'] > 0: date_dict['short_pos'] -= self.hand_per_trade
                    else: date_dict['long_pos'] += self.hand_per_trade                    

                    trade_val = abs(df.loc[ind, 'hand'] - df.loc[ind-1, 'hand']) * self.multiplier * df.loc[ind, self.trading_price_col]     
                    df.loc[ind, 'cost&slippage'] = cost + slippage
                    
                    df.loc[ind, 'cash'] = df.loc[ind-1, 'cash'] - trade_val - cost - slippage
                    df.loc[ind, 'market_value'] = 0
                    df.loc[ind, 'margin_occupied'] = 0
                    df.loc[ind, 'equity'] = df.loc[ind, 'cash'] + df.loc[ind, 'market_value']
                    df.loc[ind, 'Floating P&L'] = (df.loc[ind, self.trading_price_col] - df.loc[ind-1, 'init_price']) * df.loc[ind, 'hand'] * self.multiplier 

                if df.loc[ind-1, 'pos'] == 1:
                    df.loc[ind, 'hand'] = self.hand_per_trade

                    # 更新当前成交状态
                    if date_dict['short_pos'] >= self.hand_per_trade: date_dict['short_pos'] -= self.hand_per_trade * 2 
                    elif date_dict['short_pos'] == self.hand_per_trade: 
                        date_dict['short_pos'] -= self.hand_per_trade
                        date_dict['long_pos'] += self.hand_per_trade
                    else: date_dict['long_pos'] += self.hand_per_trade * 2
                    date_dict['trade_today'] += 1

                    df.loc[ind, 'init_price'] = df.loc[ind, self.trading_price_col]
                    trade_val = abs(df.loc[ind, 'hand'] - df.loc[ind-1, 'hand']) * self.multiplier * df.loc[ind, self.trading_price_col]
                    df.loc[ind, 'cost&slippage'] = cost + slippage

                    df.loc[ind, 'cash'] = df.loc[ind-1, 'cash'] - trade_val - cost - slippage
                    df.loc[ind, 'market_value'] = df.loc[ind, 'hand'] * self.multiplier * df.loc[ind, 'close']
                    df.loc[ind, 'margin_occupied'] = np.abs(df.loc[ind, 'market_value']) * self.margin_rate
                    df.loc[ind, 'equity'] = df.loc[ind, 'cash'] + df.loc[ind, 'market_value']
                    df.loc[ind, 'Floating P&L'] = (df.loc[ind, 'close'] - df.loc[ind, 'init_price']) * df.loc[ind, 'hand'] * self.multiplier 

            else:
                print(df.loc[ind, 'datetime'])
                break
            
            df.loc[ind, 'long_pos'] = date_dict['long_pos']
            df.loc[ind, 'short_pos'] = date_dict['short_pos']
            df.loc[ind, 'margin_used'] = (df.loc[ind, ['long_pos','short_pos']].sum() * 
                                        (df.loc[ind, 'close'] if pd.notna(df.loc[ind, 'close']) else 0) * 
                                        self.multiplier * self.margin_rate)            
            df.loc[ind, 'Margin_rate'] = df.loc[ind, 'margin_used'] / self.money 

        df['pnl_ret'] = (df['equity'].diff() / self.money).fillna(0)
        df['pnl_ret_cum'] = df['pnl_ret'].cumsum().ffill()
        df['cost&slippage_cum'] = df['cost&slippage'].cumsum().ffill()
        df['cost&slippage_rate'] = df['cost&slippage'] / self.money
        df['month'] = df.datetime.dt.to_period('1M')
        df['year'] = df.datetime.dt.year
        return df

    def backtest(self,
                    th1,
                    th2=0.5,
                    model_name="best_iteration_log_weighted",
                    save=False, 
                    open_drop=True,
                    close_drop=True,
                    holding_bars=5, 
                    day=3,
                    v=3,
                    color_th=0.9,
                    mask_hours = [],
                    merge_th = 0.8
                    ):
        
        from joblib import Parallel, delayed
        def process_group(group_name):
            if model_name not in self.predictions[group_name]:
                return group_name, None
            
            factor_df = self.predictions[group_name][model_name].rename(
                columns={model_name: 'factor'}
            )[[self.instrument_col, 'factor', self.ts_col]]
            
            factor_df = self.prepare_data(factor_df, self.mkt_data)
            
            if v == 2:
                merged_data = process_signals_v2(
                    factor_df,
                    th1=th1,
                    th2=th2,
                    factor_col=self.factor_col,
                    open_drop=open_drop,
                    close_drop=close_drop,
                    holding_period=holding_bars,
                    day=day,
                    trading_hours=self.trading_hours,
                    mask_hours=mask_hours
                )
                merged_data = merged_data[merged_data.datetime >= group_name]
                return group_name, merged_data
            
            return group_name, None

        results = Parallel(n_jobs=10)(
            delayed(process_group)(group_name) 
            for group_name in self.model_group_name
        )
        merged_data_dict = {k: v for k, v in results if v is not None}
        merged_data = pd.DataFrame(index = self.factor_single.datetime)
        for group_name, pos_df in merged_data_dict.items():
            merged_data[group_name] = pos_df.set_index('datetime').reindex(index=merged_data.index)['pos']

        merged_data['pos'] = merged_data.mean(axis=1)
        merged_data.index = merged_data.index.astype(str)

        merged_data = pd.merge(
            self.mkt_data.sort_values(self.ts_col),
            merged_data['pos'].reset_index(),
            on=[self.ts_col], 
            how='right'
        )

        merged_data[self.ts_col] = pd.to_datetime(merged_data[self.ts_col])

        merged_data['pos'] = merged_data['pos'].apply(lambda x: 1 if x > merge_th else (-1 if x < -merge_th else 0))

        if self.fee_way == "number" and self.fee_mode != "无":
            merged_data = self.calc_pos_table_fee_number_normal(merged_data)

        if self.fee_way == "number" and self.fee_mode != "锁仓":
            merged_data = self.calc_pos_table_fee_number_normal(merged_data)

        if self.fee_way == "rate" and self.fee_mode == "锁仓":
            merged_data = self.calc_pos_table_fee_rate_suocang(merged_data, th1, th2, holding_bars)

        if self.fee_way == "rate" and self.fee_mode == "无":
            merged_data = self.calc_pos_table_fee_rate_normal(merged_data)

        merged_data['th1'] = th1
        merged_data['th2'] = th2
        merged_data['date'] = pd.to_datetime(merged_data.trade_date).dt.date
        merged_data['date_cum_ret'] = merged_data.groupby('date')['pnl_ret'].cumsum()
        return merged_data



