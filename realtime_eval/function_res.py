import os
import numpy as np
import pandas as pd
import json
from tqdm.auto import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def json_files_to_dataframe(folder_path):
    """读取文件夹中所有 JSON 文件并合并为 DataFrame"""
    file_lst = [x for x in os.listdir(folder_path) if x.endswith('.json')]
    
    if not file_lst:
        print("❌ 没有找到 JSON 文件")
        return pd.DataFrame()
    
    data_list = []
    
    for file_name in file_lst:
        file_path = os.path.join(folder_path, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                # data['source_file'] = file_name
                data_list.append(data)

        except Exception as e:
            print(f"❌ 读取文件失败 {file_name}: {e}")
            continue    
    if not data_list:
        print("❌ 没有成功读取任何数据")
        return pd.DataFrame()
    
    # 创建 DataFrame
    df = pd.DataFrame(data_list)
    df['time_recently'] = df['time_recently'].str.replace('.500000', '')
    df['time_recently'] = pd.to_datetime(df['time_recently'], format='mixed').dt.strftime('%Y-%m-%d %H:%M:%S')
    print(f"✅ 成功读取 {len(data_list)} 条记录，来自 {len(file_lst)} 个文件")
    return df.set_index('time_recently').sort_index()

def parse_tick_to_min(main_data):
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
    'trade_date': 'last'
    }

    df = main_data.copy()

    df['ts'] = pd.to_datetime(df['datetime']).dt.ceil('1min')
    df['high'] = df['last_price']
    df['low'] = df['last_price']
    df['open'] = df['last_price']
    df['close'] = df['last_price']

    # 买卖价差和中间价
    df['spread'] = df['ask_price1'] - df['bid_price1']
    total_volume = df['bid_volume1'] + df['ask_volume1']
    df['mid_price'] = (
        (df['ask_price1'] * df['bid_volume1'] + 
            df['bid_price1'] * df['ask_volume1']) / total_volume
    ).round(4)

    df['last_twap'] = df['last_price']
    df['bar_count'] = df.groupby('ts').cumcount() + 1

    # 添加秒数列（用于计算tick均价）
    df['second'] = pd.to_datetime(df.datetime).dt.ceil('1s').dt.second

    df.volume = df.volume.diff()

    # 添加最高最低价（日内累计）
    df['highest_price'] = df.groupby('trade_date')['last_price'].cummax()
    df['lowest_price'] = df.groupby('trade_date')['last_price'].cummin()

    df = df[df['last_price'] > 0]

    min_data = df.groupby('ts').agg(agg_dict)

    min_data = min_data[min_data.bar_count>1]

    tick6t60avg = df.groupby('ts').apply(
        lambda x: x.loc[(x['second'] >= 6) & (x['second'] <= 60), 'last_price'].mean() 
        if len(x) > 9 else np.nan, include_groups=False  
    )

    min_data = min_data.join(tick6t60avg.rename('tick6t60avg'), how='left').reset_index()
    min_data.ts = min_data.ts.dt.strftime('%Y-%m-%d %H:%M:%S')
    return min_data.set_index('ts')

def calc_trade_lst(df):
    trade = []
    for n, time in enumerate(df.index[:-5]):
        if n>=9:
            if n==9:
                if df.loc[time, 'now_pos'] != 0:
                    trade_price = df.loc[time, 'tick6t60avg']
                    open_time = time

            now_pos = df.loc[time, 'now_pos']
            last_pos = df.loc[df.index[n-1], 'now_pos']

            trade_price = df.loc[df.index[n+1], 'tick6t60avg']
            if now_pos != last_pos:
                if now_pos == 1:
                    if last_pos == 0:
                        open_price = trade_price
                        open_time = time
                    if last_pos == -1:
                        trade.append([-1, open_time, time, open_price, trade_price])
                        open_price = trade_price
                        open_time = time

                if now_pos == -1:
                    if last_pos == 0:
                        open_price = trade_price
                        open_time = time
                    if last_pos == 1:
                        trade.append([1, open_time, time, open_price, trade_price])    
                        open_price = trade_price
                        open_time = time               

                if now_pos == 0:
                    if last_pos == -1:
                        trade.append([-1, open_time, time, open_price, trade_price])
                    if last_pos == 1:
                        trade.append([1, open_time, time, open_price, trade_price])

    return trade

def plot_trading_chart_simple(df, trade_df, symbol, date):
    """简化版本 - 所有标签右移一根bar"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3]
    )
    
    # 蜡烛图 - 红色涨绿色跌
    fig.add_trace(go.Candlestick(
        x=df.index, 
        open=df['open'], 
        high=df['high'], 
        low=df['low'], 
        close=df['close'], 
        name='价格',
        increasing_line_color='red',    # 涨为红色
        decreasing_line_color='green',  # 跌为绿色
        increasing_fillcolor='red',     # 涨的填充色
        decreasing_fillcolor='green'    # 跌的填充色
    ), row=1, col=1)
    
    # 获取df的时间索引列表
    time_indices = list(df.index)
    
    # 交易标记
    if not trade_df.empty:
        for _, row in trade_df.iterrows():
            time_o_str = str(row['time_o'])
            time_c_str = str(row['time_c'])
            
            # 找到开仓时间在df中的位置
            if time_o_str in time_indices:
                idx = time_indices.index(time_o_str)
                # 右移一根bar
                if idx < len(time_indices) - 1:
                    time_o_shifted = time_indices[idx + 1]
                else:
                    time_o_shifted = time_o_str
            else:
                time_o_shifted = time_o_str
            
            # 找到平仓时间在df中的位置
            if time_c_str in time_indices:
                idx = time_indices.index(time_c_str)
                # 右移一根bar
                if idx < len(time_indices) - 1:
                    time_c_shifted = time_indices[idx + 1]
                else:
                    time_c_shifted = time_c_str
            else:
                time_c_shifted = time_c_str
            
            if row['flag'] == 1:  # 做多
                y_pos = df.loc[time_o_str, 'low'] - 5 if time_o_str in df.index else row['price_o'] - 5
                fig.add_annotation(x=time_o_shifted, y=y_pos, text="▲", 
                                 font=dict(size=20, color="green"), showarrow=False)
            elif row['flag'] == -1:  # 做空
                y_pos = df.loc[time_o_str, 'high'] + 5 if time_o_str in df.index else row['price_o'] + 5
                fig.add_annotation(x=time_o_shifted, y=y_pos, text="▼", 
                                 font=dict(size=20, color="red"), showarrow=False)
            
            # 平仓标记
            fig.add_annotation(x=time_c_shifted, y=row['price_c'], text="✕", 
                             font=dict(size=18, color="black", family="Arial Black"), showarrow=False)
    
    
    if 'volume' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['volume'],
            mode='lines', name='volume',
            line=dict(color='blue', width=1)
        ), row=2, col=1)
    
    fig.update_layout(
        title=f'{symbol} - {date}',
        height=700,
        xaxis_rangeslider_visible=False,
        showlegend=False
    )
    fig.update_xaxes(type='category')

    return fig

def get_trade_df(symbol, date, name):
    folder_path = f'/home/realtime_eval/{name}/save_files/{date}/{symbol}'
    df = json_files_to_dataframe(folder_path).set_index('time_recently').sort_index()
    df.index = df.index.map(lambda x: x[:-1] + '0')

    main_data = pd.read_feather(f'/home/realtime_eval/{name}/save_files/{date}/{symbol}.feather')

    # # 修改索引中的13:00:00为11:30:00
    # df.index = df.index.map(lambda x: x.replace("13:00:00", "11:30:00"))
    # df = df[~df.index.duplicated(keep='first')]
    # 之后merge
    # df = df.join(main_data[['tick6t60avg', 'open', 'high', 'low', 'close', 'volume']])

    # trade_df = pd.DataFrame(calc_trade_lst(df), columns=['flag', 'time_o', 'time_c', 'price_o', 'price_c'])
    # trade_df['profit'] = trade_df.flag * (trade_df['price_c'] - trade_df['price_o'])
    # return trade_df