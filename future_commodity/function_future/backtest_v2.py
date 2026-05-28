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
warnings.filterwarnings("ignore", category=UserWarning, module="numpy.core.getlimits")

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

def analyze_pos_distribution(pos_series):
    """
    统计仓位分布情况
    :param pos_series: 仓位序列（包含+1, -1, 0）
    :return: 分布统计字典
    """
    count = pos_series.value_counts()
    total = len(pos_series)
    
    stats = {
        '多头(+1) 数量': count.get(1, 0),
        '空头(-1) 数量': count.get(-1, 0),
        '空仓(0) 数量': count.get(0, 0),
        '多头占比': f"{count.get(1, 0)/total:.2%}",
        '空头占比': f"{count.get(-1, 0)/total:.2%}",
        '空仓占比': f"{count.get(0, 0)/total:.2%}"
    }
    return pd.DataFrame.from_dict(stats, orient='index', columns=['统计值'])

# ====================== Model Backtesting ======================
def process_signals_v2(df, th1, th2,
                     holding_period, warmup=100, day=3, date_max_trade=100, 
                     vol_controller=None, 
                     ts_col='datetime', factor_col='factor', 
                     open_drop=True, close_drop=True):
    
    df = df.copy()
    window = 60 * 4 * day
    df['factor_val'] = df[factor_col]
    df['date'] = df.datetime.dt.date

    # df['factor_val'] = np.where(
    #     df[ts_col].dt.time > time(14, 50, 0),
    #     np.nan,
    #     df['factor_val']
    # )
    # df['factor_val'] = np.where(
    #     df[ts_col].dt.time < time(9, 40, 0),
    #     np.nan,
    #     df['factor_val']
    # )

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
                if time_series.iloc[i] < time(9, 40):
                    pos.iloc[i] = 0
                    in_long = False
                    in_short = False
                    holding_bars = 0
                    continue
            
            if close_drop:
                if time_series.iloc[i] > time(14, 50):
                    pos.iloc[i] = 0        
                    in_long = False
                    in_short = False
                    holding_bars = 0
                    continue   

            if not in_long and not in_short:

                if current_val >= th_ol:
                    # 当前没有仓位的时候， 如果满足波动率开仓条件，并且今日开仓次数小于date_max_trade就可以开仓
                    if (vol_controller.iloc[i] == 1) & (today_trade<date_max_trade):
                        pos.iloc[i] = 1.0
                        in_long = True
                        holding_bars = 1
                        today_trade += 1

                elif current_val <= th_os:
                    # 当前没有仓位的时候， 如果满足波动率开仓条件，并且今日开仓次数小于date_max_trade就可以开仓
                    if (vol_controller.iloc[i] == 1) & (today_trade<date_max_trade):
                        pos.iloc[i] = -1.0
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
        ax1_right = ax1.twinx()
        
        cum_data = stats_df[['pnl_ret', 'cost&slippage', 'date']].groupby('date').sum().cumsum()
        ax1.plot(cum_data.index, cum_data['pnl_ret'], color='blue', label='Cumulative Return')
        ax1_right.plot(cum_data.index, cum_data['cost&slippage'], color='red', label='Cost & Slippage', linestyle='--')
        
        ax1.set_title(f'{variety} - (Ret: {ret[0]*100:.2f}%, Sharpe: {sp:.2f}, MaxDD: {maxdd:.2f}%)', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1_right.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        
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
        self.window_end = '2025-01-01'

        self.model_dir = Path(config.get('MODEL_DIR', ""))
        self.backtest_output_dir = Path(config.get('OUTPUT_DIR', ""))
        self.pic_dir = Path(config.get('pic_dir', ""))
        self.max_trade_inday = 10000
        self.hand_per_trade = 10
        self.money = 10_000_000

        self.feature_dir = config.get('feature_dir', "")
        self.ts_col = 'datetime'
        self.instrument_col = 'instrument'
        self.backtest_col = 'rtn_1'
        self.factor_col = 'factor'
        self.model_save_name = ''
        self.open_drop = True
        self.vol_pred = pd.DataFrame()
        self.holding_price_col = config.get('holding_price_col', "")
        self.trading_price_col = config.get('trading_price_col', "")

        # 交易参数
        # self.start_date = start_date
        # self.end_date = end_date
        self.train_end_date = train_end_date

    def load_config(self, symbol):
        import function.DataLoader as DL
        config_loader = DL.InstrumentConfig()
        symbol_config = config_loader.get_instrument_config(symbol)
        self.margin_rate = symbol_config["margin_rate"]
        self.fee_way = symbol_config["fee_way"]
        self.fee = symbol_config["fee"]
        self.fee_mode = symbol_config["fee_comment"]
        self.multiplier = symbol_config["contract_multiplier"]
        self.price_tick = symbol_config["price_tick"]
        self.slippage = 0
        print(f"Loaded config for {symbol}: margin_rate={self.margin_rate}, fee_way={self.fee_way}, fee={self.fee}, multiplier={self.multiplier}, price_tick={self.price_tick}")

    def load_factor(self, variety, start_date='20180101', end_date='20241231'):
        factor_single = pd.read_feather(f'/mnt/Data/writable/liaoyuyang/factor/{variety}/all_fac/all_factor.feather')
        factor_single['hour'] = factor_single.datetime.dt.hour
        factor_single = factor_single[factor_single.datetime.astype('datetime64[ns]').between(start_date, end_date)]
        factor_single = factor_single.sort_values('datetime').drop_duplicates('datetime', keep='last')
        self.factor_single = factor_single.round(8)

    def load_mktdata(self, variety, start_date = '20180101', end_date = '20241231'):
        data = pd.read_csv(f'/mnt/Data/writable/liaoyuyang/data/1min/active/main_{variety}.csv', index_col=0).rename(columns={'ts': 'datetime'})
        data = data[data.datetime.astype('datetime64[ns]').between(start_date, end_date)]
        data = data.rename(columns={'code': 'instrument'})
        data = data.sort_values('datetime').drop_duplicates('datetime', keep='last')
        # data['mid10avg'] = data['mid10avg'].ffill(limit=120)
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

        merged = merged.dropna(subset='close')
        merged['HoldingPnl'] = merged.groupby(self.instrument_col)[self.holding_price_col].pct_change()
        merged['TradingPnl'] = (merged[self.holding_price_col] - merged[self.trading_price_col]) / merged.groupby(self.instrument_col)['close'].shift(1)
        merged['HoldingPnl']  = merged['HoldingPnl'].fillna(0.0)
        merged['TradingPnl']  = merged['TradingPnl'].fillna(0.0)
        return merged.reset_index(drop=True)

    # def load_models(self):
    #     """加载所有.pkl模型文件"""
    #     import joblib
    #     self.models = {}
        
    #     for pkl_file in self.model_dir.glob("*.joblib"):
    #         with open(pkl_file, 'rb') as f:
    #             self.models[pkl_file.stem] = joblib.load(f)['model']
        
    #     print(f"已加载 {len(self.models)} 个模型")
        
    #     if self.models:
    #         # 随便取第一个模型获取特征名称
    #         first_model = list(self.models.values())[0]
    #         all_features = first_model.feature_name()  # 注意这里应该是 feature_name() 方法调用
    #         print(f"总共用到了{len(all_features)}个特征")
    #         self.all_features = list(all_features)
    #     else:
    #         print("警告：未加载到任何模型")
    #         self.all_features = []
        
    #     return self.models

    def load_models(self):
        """加载所有 .lgb 模型文件和 .json 元数据文件"""
        import lightgbm as lgb
        import json
        
        self.models = {}
        self.models_metadata = {}  
        
        for lgb_file in self.model_dir.glob("*.lgb"):
            try:
                model = lgb.Booster(model_file=str(lgb_file))
                
                meta_file = lgb_file.with_name(lgb_file.stem + '_meta.json')
                if meta_file.exists():
                    with open(meta_file, 'r') as f:
                        metadata = json.load(f)
                else:
                    metadata = {}
                
                self.models[lgb_file.stem] = model
                self.models_metadata[lgb_file.stem] = metadata
                
                print(f"✅ 加载模型: {lgb_file.stem}")
                
            except Exception as e:
                print(f"❌ 加载模型失败 {lgb_file.stem}: {e}")
                continue
        
        print(f"已加载 {len(self.models)} 个模型")
        
        if self.models:
            first_model = list(self.models.values())[0]
            all_features = first_model.feature_name()
            print(f"总共用到了 {len(all_features)} 个特征")
            self.all_features = list(all_features)
            
            for name, metadata in self.models_metadata.items():
                best_iter = metadata.get('best_iteration', '未知')
                test_corr = metadata.get('test_corr', '未知')
                print(f"模型 {name}: best_iteration={best_iter}, test_corr={test_corr}")
        else:
            print("警告：未加载到任何模型")
            self.all_features = []
        
        return self.models

    def generate_predictions(self):
        """Generate predictions for all models"""
        self.predictions = {}

        for model_name, model in self.models.items():
            feature_name = self.models[model_name].feature_name()
            X_tmp = self.factor_single[feature_name].values 
            preds = model.predict(X_tmp)
            
            pred_df = pd.DataFrame({
                self.ts_col: self.factor_single[self.ts_col],
                self.instrument_col: self.factor_single[self.instrument_col],
                model_name: preds,
                'best_round' :model.best_iteration
            })
            self.predictions[model_name] = pred_df

    def combine_models(self, by, avg=True):
        if by == 'best_iteration_log_weighted':
            pred_lst = []
            weight_lst = []

            for model_name, prediction in self.predictions.items():
                pred_df = prediction
                best_iteration = self.models_metadata[model_name]['best_iteration']
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
            self.predictions[by] = pred_df
            return self.predictions[by]

    def calc_pos_table_fee_rate_suocang(self, merged_data, th1, th2, holding_bars):
        print("使用的交易价格", self.trading_price_col)
        df = merged_data[['datetime', 'factor', 'pos', 'close', self.trading_price_col]].copy()
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

    def calc_pos_table_fee_number_normal(self, merged_data):
        df = merged_data[['datetime', 'factor', 'pos', 'close', self.trading_price_col]].copy()
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

        cost = self.fee
        slippage = self.slippage       

        for ind in range(2, len(merged_data)):
            
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
        df['cost&slippage_cum_rate'] = df['cost&slippage_cum'] / self.money
        df['date'] = df.datetime.dt.date
        df['month'] = df.datetime.dt.to_period('1M')
        df['year'] = df.datetime.dt.year
        return df

    def backtest(self,
                    th1,
                    th2=0.5,
                    save=False, 
                    vol_control=pd.Series(),
                    open_drop=True,
                    close_drop=True,
                    holding_bars=5, 
                    day=3):

        """Run backtests for all models"""
        self.backtest_results = {}
        for model_name, pred_df in sorted(self.predictions.items()):
            if not model_name.startswith('best_iteration_log_weighted'):
                continue

            # print(f"\n=== Backtesting model: {model_name} ===")
            pd.set_option('future.no_silent_downcasting', True)

            vol_control = vol_control.reindex(index=pred_df[self.ts_col]).fillna(1)
            factor_df = pred_df.rename(columns={model_name: 'factor'})
            factor_df = factor_df[[self.instrument_col, 'factor', self.ts_col]]
            factor_df = process_signals_v2(factor_df,
                                            th1=th1,
                                            th2=th2,
                                            factor_col=self.factor_col,
                                            vol_controller=vol_control,
                                            open_drop=open_drop,
                                            close_drop=close_drop,
                                            holding_period=holding_bars,
                                            day=day
                                            )
            merged_data = self.prepare_data(
                        factor_df,
                        self.mkt_data
                    )
            merged_data = merged_data[merged_data.datetime>=self.train_end_date].reset_index(drop=True)
            
            df_th = merged_data.set_index('datetime')[['th_open_long', 'th_close_long', 'th_open_short', 'th_close_short']].join(vol_control, how='left')
            if self.fee_way == "number" and self.fee_mode != "锁仓":
                merged_data = self.calc_pos_table_fee_number_normal(merged_data)
            if self.fee_way == "rate" and self.fee_mode == "锁仓":
                merged_data = self.calc_pos_table_fee_rate_suocang(merged_data, th1, th2, holding_bars)
            merged_data['th1'] = th1
            merged_data['th2'] = th2
            merged_data = merged_data.set_index('datetime').join(df_th, how='left').reset_index()
            return merged_data

import function.date_selection as ds
class TradingVisualizationPager:
    def __init__(self, data, rows_per_page=240, date_format='%H:%M', skip_weekends=True):
        """
        交易数据分页可视化工具
        
        参数:
            data: 必须包含以下列的DataFrame:
                  - datetime: 时间戳
                  - factor: 因子值
                  - pos: 仓位
                  - close: 收盘价
                  - equity: 权益曲线
                  - pnl_ret: 收益率
                  - pnl_ret_cum: 累积收益率
            rows_per_page: 每页显示行数
            date_format: 时间显示格式
            skip_weekends: 是否跳过周末
        """
        # 初始化参数
        data['date_cum_ret'] = data.groupby('date')['pnl_ret'].cumsum()
        self.rows_per_page = rows_per_page
        self.current_page = 0
        self.data = data.copy()
        trade_bars = ds.generate_trading_bars(sorted(data.date.unique()))
        self.data = self.data.set_index('datetime').reindex(trade_bars).reset_index(names='datetime')
        self.date_format = date_format
        self.skip_weekends = skip_weekends
        self.output_lock = threading.Lock()
        
        # 数据预处理
        self._prepare_data()
        self.total_pages = max(1, (len(self.data) - 1) // self.rows_per_page + 1)
        
        # 创建控件
        self._create_widgets()
        self._create_manual_input()
    
    def _prepare_data(self):
        """数据预处理"""
        # 确保时间列是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(self.data['datetime']):
            self.data['datetime'] = pd.to_datetime(self.data['datetime'])
        
        # 创建时间字符串列（用于显示）
        self.data['ts_str'] = self.data['datetime'].dt.strftime(self.date_format)
        
        # 计算10个tick的平均价格
        if 'tick10avg' not in self.data.columns:
            self.data['tick10avg'] = self.data['close'].rolling(10).mean()
    
    def _create_widgets(self):
        """创建所有交互控件"""
        # 页码显示
        self.page_label = widgets.Label(value="页码: 1/1")
        
        # 导航按钮
        self.prev_button = widgets.Button(
            description="← 上一页",
            layout=widgets.Layout(width='100px')
        )
        self.next_button = widgets.Button(
            description="下一页 →",
            layout=widgets.Layout(width='100px')
        )
        self.exit_button = widgets.Button(
            description="退出",
            button_style='danger',
            layout=widgets.Layout(width='80px')
        )
        
        # 绑定事件
        self.prev_button.on_click(self._on_prev)
        self.next_button.on_click(self._on_next)
        self.exit_button.on_click(self._on_exit)
        
        # 控件布局
        self.controls = widgets.HBox([
            self.prev_button,
            self.page_label,
            self.next_button,
            self.exit_button
        ], layout=widgets.Layout(justify_content='center'))
        
        # 输出区域
        self.output = widgets.Output()
    
    def _create_manual_input(self):
        """创建手动页码输入控件"""
        self.page_input = widgets.IntText(
            value=1,
            min=1,
            max=self.total_pages,
            description='跳至页码:',
            layout=widgets.Layout(width='200px')
        )
        self.jump_button = widgets.Button(
            description="跳转",
            layout=widgets.Layout(width='80px')
        )
        self.jump_button.on_click(self._on_jump)
        
        # 添加到控件栏
        self.controls.children = tuple(list(self.controls.children) + [
            self.page_input,
            self.jump_button
        ])
    
    def _on_prev(self, b):
        """上一页事件"""
        if self.current_page > 0:
            self.current_page -= 1
            self._update_display()
    
    def _on_next(self, b):
        """下一页事件"""
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self._update_display()
    
    def _on_jump(self, b):
        """跳转到指定页码"""
        try:
            target_page = self.page_input.value - 1
            if 0 <= target_page < self.total_pages:
                self.current_page = target_page
                self._update_display()
            else:
                with self.output:
                    print(f"⚠️ 请输入1到{self.total_pages}之间的页码")
        except Exception as e:
            with self.output:
                print(f"跳转错误: {str(e)}")
    
    def _on_exit(self, b):
        """退出可视化"""
        with self.output:
            clear_output()
            print("可视化工具已关闭")
    
    def _get_current_page_data(self):
        """获取当前页数据"""
        start = self.current_page * self.rows_per_page
        end = start + self.rows_per_page
        return self.data.iloc[start:end].copy()
    
    def _calculate_symmetric_range(self, data, padding=0.1):
        """计算对称Y轴范围"""
        if data.empty:
            return [-1, 1]
        abs_max = max(abs(data.max()), abs(data.min()))
        return [-abs_max*(1+padding), abs_max*(1+padding)]
    
    def _create_chart(self):
        """创建包含所有信号线的图表"""
        page_data = self._get_current_page_data()

        # ========== 计算价格波动幅度 ==========
        if not page_data.empty and 'close' in page_data.columns:
            high = page_data['close'].max()
            low = page_data['close'].min()
            mean = page_data['close'].mean()
            price_volatility = (high - low) / mean if mean != 0 else 0
            volatility_text = f"价格波动: {(price_volatility*100):.2f}%"
        else:
            volatility_text = "价格波动: N/A"
        
        # 创建包含三个子图的图表
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}]],
            subplot_titles=[
                "收益率和仓位", 
                f"价格和因子 | {volatility_text}",
                "账户权益和浮动盈亏"
            ]
        )
        
        # ========== 第一子图：收益率和仓位 ==========
        # 收益率线 (蓝色实线，左侧Y轴)
        fig.add_trace(
            go.Scatter(
                x=page_data['ts_str'] if self.skip_weekends else page_data['datetime'],
                y=page_data['pnl_ret'],
                name='收益率',
                line=dict(color='#3B82F6', width=1.5),
                opacity=0.8
            ),
            row=1, col=1, secondary_y=False
        )

        # 累积收益率线 (橙色实线，右侧Y轴)
        fig.add_trace(
            go.Scatter(
                x=page_data['ts_str'] if self.skip_weekends else page_data['datetime'],
                y=page_data['date_cum_ret'],
                name='累积收益率',
                line=dict(color='#F97316', width=1.5),
                opacity=0.8
            ),
            row=1, col=1, secondary_y=False
        )
        # 仓位线 (绿色实线，右侧Y轴)
        fig.add_trace(
            go.Scatter(
                x=page_data['ts_str'] if self.skip_weekends else page_data['datetime'],
                y=page_data['pos'],
                name='仓位',
                line=dict(color='#10B981', width=2),
                opacity=0.8
            ),
            row=1, col=1, secondary_y=True
        )

        # ========== 第二子图：价格和因子 ==========
        # 收盘价线 (紫色实线)
        fig.add_trace(
            go.Scatter(
                x=page_data['ts_str'] if self.skip_weekends else page_data['datetime'],
                y=page_data['close'],
                name='收盘价',
                line=dict(color='#8B5CF6', width=2),  # 紫色
                opacity=0.8
            ),
            row=2, col=1, secondary_y=False
        )

        # 开多阈值线 (深绿色实线)
        fig.add_trace(
            go.Scatter(
                x=page_data['ts_str'] if self.skip_weekends else page_data['datetime'],
                y=page_data['th_open_long'],
                name='开多',
                line=dict(color='#047857', width=1.5, dash='dash'),  # 深绿色
                opacity=0.8
            ),
            row=2, col=1, secondary_y=True
        )

        # 平多阈值线 (浅绿色虚线)
        fig.add_trace(
            go.Scatter(
                x=page_data['ts_str'] if self.skip_weekends else page_data['datetime'],
                y=page_data['th_close_long'],
                name='平多',
                line=dict(color='#10B981', width=1.5, dash='dash'),  # 浅绿色虚线
                opacity=0.8
            ),
            row=2, col=1, secondary_y=True
        )

        # 开空阈值线 (深红色实线)
        fig.add_trace(
            go.Scatter(
                x=page_data['ts_str'] if self.skip_weekends else page_data['datetime'],
                y=page_data['th_open_short'],
                name='开空',
                line=dict(color='#DC2626', width=1.5, dash='dash'),  # 深红色
                opacity=0.8
            ),
            row=2, col=1, secondary_y=True
        )

        # 平空阈值线 (浅红色虚线)
        fig.add_trace(
            go.Scatter(
                x=page_data['ts_str'] if self.skip_weekends else page_data['datetime'],
                y=page_data['th_close_short'],
                name='平空',
                line=dict(color='#F87171', width=1.5, dash='dash'),  # 浅红色虚线
                opacity=0.8
            ),
            row=2, col=1, secondary_y=True
        )

        # 10tick平均价线 (浅蓝色实线)
        fig.add_trace(
            go.Scatter(
                x=page_data['ts_str'] if self.skip_weekends else page_data['datetime'],
                y=page_data['tick10avg'],
                name='10Tick均价',
                line=dict(color='#0EA5E9', width=1.5),
                opacity=0.7
            ),
            row=2, col=1, secondary_y=False
        )
        
        # 因子值线 (粉色实线，右侧Y轴)
        fig.add_trace(
            go.Scatter(
                x=page_data['ts_str'] if self.skip_weekends else page_data['datetime'],
                y=page_data['factor'],
                name='因子值',
                line=dict(color='#EC4899', width=2),
                opacity=0.8
            ),
            row=2, col=1, secondary_y=True
        )
        
        # ========== 第三子图：账户权益和浮动盈亏 ==========
        # 账户权益线 (深绿色实线)
        fig.add_trace(
            go.Scatter(
                x=page_data['ts_str'] if self.skip_weekends else page_data['datetime'],
                y=page_data['equity'],
                name='账户权益',
                line=dict(color='#047857', width=2),
                opacity=0.8
            ),
            row=3, col=1, secondary_y=False
        )
        
        # 浮动盈亏线 (红色实线)
        fig.add_trace(
            go.Scatter(
                x=page_data['ts_str'] if self.skip_weekends else page_data['datetime'],
                y=page_data['Margin_rate'],
                name='保证金率',
                line=dict(color='#EF4444', width=2),
                opacity=0.8
            ),
            row=3, col=1, secondary_y=True
        )

        # 浮动盈亏线 (红色实线)
        fig.add_trace(
            go.Scatter(
                x=page_data['ts_str'] if self.skip_weekends else page_data['datetime'],
                y=page_data['vol_pred'],
                name='波动率信号',
                line=dict(color='darkgray', width=2),
                opacity=0.8
            ),
            row=3, col=1, secondary_y=True
        )

        # ========== 图表布局配置 ==========
        # 时间范围标题
        if not page_data.empty:
            start_time = page_data['datetime'].iloc[0].strftime('%Y-%m-%d %H:%M')
            end_time = page_data['datetime'].iloc[-1].strftime('%Y-%m-%d %H:%M')
            time_range = f"{start_time} 至 {end_time}"
        else:
            time_range = "无数据"
        
        fig.update_layout(
            title=f"第 {self.current_page+1}/{self.total_pages} 页 | 时间范围: {time_range}",
            height=800,  # 增加高度以适应三行子图
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=100),
        )
        
        # ========== Y轴范围设置 ==========
        # 第一子图左侧Y轴（收益率）
        fig.update_yaxes(
            title_text="收益率",
            range=self._calculate_symmetric_range(page_data['date_cum_ret']),
            row=1, col=1, secondary_y=False
        )
        
        # 第一子图右侧Y轴（仓位）
        fig.update_yaxes(
            title_text="仓位",
            range=[-1.5, 1.5],
            row=1, col=1, secondary_y=True
        )
        
        # 第二子图左侧Y轴（价格）
        price_min = page_data['close'].min()
        price_max = page_data['close'].max()
        price_padding = (price_max - price_min) * 0.1
        fig.update_yaxes(
            title_text="价格",
            range=[price_min - price_padding, price_max + price_padding],
            row=2, col=1, secondary_y=False
        )
        
        # 第二子图右侧Y轴（因子值）
        fig.update_yaxes(
            title_text="因子值",
            range=self._calculate_symmetric_range(page_data['factor']),
            row=2, col=1, secondary_y=True
        )
        
        # 第三子图左侧Y轴（权益和盈亏）
        equity_min = page_data['equity'].min()
        equity_max = page_data['equity'].max()
        equity_padding = (equity_max - equity_min) * 0.01
        fig.update_yaxes(
            title_text="权益/盈亏",
            range=[equity_min - equity_padding, equity_max + equity_padding],
            row=3, col=1, secondary_y=False
        )        
        # 第三子图右侧Y轴（保证金占用）
        fig.update_yaxes(
            title_text="保证金占用",
            range=[0, 1.2],
            row=3, col=1, secondary_y=True
        )
        
        # X轴标签（仅底部子图显示）
        fig.update_xaxes(title_text="时间", row=3, col=1)
        
        return fig
    
    def _update_display(self):
        """更新所有显示内容"""
        with self.output_lock:
            self.page_label.value = f"页码: {self.current_page+1}/{self.total_pages}"
            self.page_input.value = self.current_page + 1
            self.prev_button.disabled = (self.current_page == 0)
            self.next_button.disabled = (self.current_page >= self.total_pages - 1)
            
            with self.output:
                clear_output(wait=True) 
                fig = self._create_chart()
                display(fig) 

    def run(self):
        """启动交互式可视化"""
        display(self.controls)
        display(self.output)
        self._update_display()
