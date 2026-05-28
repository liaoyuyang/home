"""
DCE农产品盘后分析模块
参考 if 品种方案，使用 twap（avg_price_from_5s）作为成交参考价

支持品种: A, B, C, CS, M, Y, P, LH
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# 配置
# =============================================================================
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    CONFIG = json.load(f)

SAVE_ROOT = CONFIG['data_paths']['save_files_root']
DATA_DIR = os.path.join(SAVE_ROOT, "data")
SYMBOLS = CONFIG['symbols']
SPECS = CONFIG['symbol_specs']


def get_spec(symbol: str) -> Dict:
    symbol = symbol.upper()
    if symbol not in SPECS:
        raise ValueError(f"不支持的品种: {symbol}")
    return SPECS[symbol]


# =============================================================================
# 数据加载
# =============================================================================
def get_min_files(symbol: str, date_str: str) -> List[str]:
    """获取某交易日所有 min CSV（含前一日夜盘）"""
    symbol = symbol.upper()
    files = []
    # 日盘
    files.extend(glob.glob(os.path.join(DATA_DIR, f"{symbol}_min_{date_str}_*.csv")))
    # 夜盘
    prev = (pd.to_datetime(date_str) - timedelta(days=1)).strftime('%Y-%m-%d')
    for f in glob.glob(os.path.join(DATA_DIR, f"{symbol}_min_{prev}_*.csv")):
        basename = os.path.basename(f).replace('.csv', '')
        hour = int(basename.split('_')[3].split('-')[0])
        if hour >= 21:
            files.append(f)
    return sorted(files)


def load_min_data(symbol: str, date_str: str) -> pd.DataFrame:
    """加载分钟 K 线"""
    files = get_min_files(symbol, date_str)
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_csv(f) for f in files if pd.read_csv(f).shape[0] > 0]
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').drop_duplicates('datetime', keep='last')
    return df.set_index('datetime')


def load_json_status(symbol: str, date_str: str) -> pd.DataFrame:
    """加载交易状态 json"""
    symbol = symbol.upper()
    json_dir = os.path.join(SAVE_ROOT, symbol, 'json')
    files = []
    files.extend(glob.glob(os.path.join(json_dir, f'trading_status_{date_str}_*.json')))
    prev = (pd.to_datetime(date_str) - timedelta(days=1)).strftime('%Y-%m-%d')
    for f in glob.glob(os.path.join(json_dir, f'trading_status_{prev}_*.json')):
        hour = int(os.path.basename(f).replace('trading_status_', '').replace('.json', '').split('_')[1].split('-')[0])
        if hour >= 21:
            files.append(f)
    if not files:
        return pd.DataFrame()
    rows = []
    for f in sorted(files):
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                rows.append(json.load(fp))
        except Exception:
            continue
    df = pd.DataFrame(rows)
    if 'time_recently' not in df.columns:
        return pd.DataFrame()
    df['time_recently'] = df['time_recently'].astype(str).str.replace('.500000', '')
    df['time_recently'] = pd.to_datetime(df['time_recently'])
    df = df.sort_values('time_recently').drop_duplicates('time_recently', keep='last')
    return df.set_index('time_recently')


# =============================================================================
# 交易计算（参考 if 方案）
# =============================================================================
def calc_trade_lst(df: pd.DataFrame) -> List[list]:
    """
    成交价 = 下一根 bar 的 avg_price_from_5s（右移一根 bar）
    """
    trade = []
    for n, time in enumerate(df.index[:-5]):
        if n >= 9:
            if n == 9 and df.loc[time, 'now_pos'] != 0:
                trade_price = df.loc[time, 'avg_price_from_5s']
                open_time = time

            now_pos = df.loc[time, 'now_pos']
            last_pos = df.loc[df.index[n - 1], 'now_pos']
            trade_price = df.loc[df.index[n + 1], 'avg_price_from_5s']

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


def get_trade_df(symbol: str, date_str: str) -> pd.DataFrame:
    """返回 trade DataFrame: [flag, time_o, time_c, price_o, price_c, profit]"""
    symbol = symbol.upper()

    df_status = load_json_status(symbol, date_str)
    if df_status.empty:
        return pd.DataFrame()
    df_status.index = df_status.index.map(lambda x: x.replace(second=0, microsecond=0))
    df_status = df_status[~df_status.index.duplicated(keep='first')]

    df_min = load_min_data(symbol, date_str)
    if df_min.empty or 'avg_price_from_5s' not in df_min.columns:
        return pd.DataFrame()

    # 修改 13:00:00 -> 11:30:00（兼容 if 方案中的映射，dce 实际没有这个问题，保留无害）
    df_status.index = df_status.index.map(lambda x: x.replace(hour=11, minute=30) if x.hour == 13 and x.minute == 0 else x)

    df = df_status[['now_pos']].join(
        df_min[['avg_price_from_5s', 'open', 'high', 'low', 'close', 'volume']], how='inner'
    )
    df = df.dropna(subset=['now_pos', 'avg_price_from_5s'])
    if len(df) < 11:
        return pd.DataFrame()

    trades = calc_trade_lst(df)
    trade_df = pd.DataFrame(trades, columns=['flag', 'time_o', 'time_c', 'price_o', 'price_c'])
    if trade_df.empty:
        return trade_df
    trade_df['profit'] = trade_df['flag'] * (trade_df['price_c'] - trade_df['price_o'])
    trade_df['cum_profit'] = trade_df['profit'].cumsum()
    return trade_df


# =============================================================================
# 绘图（参考 if 方案）
# =============================================================================
def plot_trading_chart(df: pd.DataFrame, trade_df: pd.DataFrame, symbol: str, date: str):
    """plotly 蜡烛图 + 交易标记"""
    if df.empty:
        return None
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])

    fig.add_trace(go.Candlestick(
        x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name='价格', increasing_line_color='red', decreasing_line_color='green',
        increasing_fillcolor='red', decreasing_fillcolor='green'
    ), row=1, col=1)

    times = list(df.index)
    if not trade_df.empty:
        for _, row in trade_df.iterrows():
            time_o = pd.to_datetime(row['time_o'])
            time_c = pd.to_datetime(row['time_c'])

            idx_o = times.index(time_o) if time_o in times else -1
            idx_c = times.index(time_c) if time_c in times else -1
            time_o_shift = times[idx_o + 1] if idx_o >= 0 and idx_o + 1 < len(times) else time_o
            time_c_shift = times[idx_c + 1] if idx_c >= 0 and idx_c + 1 < len(times) else time_c

            if row['flag'] == 1:
                y = df.loc[time_o, 'low'] - 5 if time_o in df.index else row['price_o'] - 5
                fig.add_annotation(x=time_o_shift, y=y, text="▲", font=dict(size=20, color="green"), showarrow=False)
            elif row['flag'] == -1:
                y = df.loc[time_o, 'high'] + 5 if time_o in df.index else row['price_o'] + 5
                fig.add_annotation(x=time_o_shift, y=y, text="▼", font=dict(size=20, color="red"), showarrow=False)

            fig.add_annotation(x=time_c_shift, y=row['price_c'], text="✕",
                             font=dict(size=18, color="black", family="Arial Black"), showarrow=False)

    if 'volume' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['volume'], mode='lines', name='volume',
                                 line=dict(color='blue', width=1)), row=2, col=1)

    spec = get_spec(symbol)
    fig.update_layout(title=f'{symbol} ({spec["name"]}) - {date}', height=700,
                      xaxis_rangeslider_visible=False, showlegend=False)
    fig.update_xaxes(type='category')
    return fig


# =============================================================================
# 主入口
# =============================================================================
def analyze_trading_day(symbol: str, date: str, save_plot: bool = True, show_plot: bool = False):
    """分析单日单品种"""
    symbol = symbol.upper()
    date_fmt = date.replace('-', '') if '-' in date else date
    date = f"{date[:4]}-{date[4:6]}-{date[6:8]}" if '-' not in date else date

    spec = get_spec(symbol)
    print(f"\n{'='*60}")
    print(f"分析 {symbol} ({spec['name']}) {date}")
    print(f"{'='*60}")

    df_status = load_json_status(symbol, date)
    df_min = load_min_data(symbol, date)

    if df_status.empty:
        print(f"❌ 未找到 {symbol} 交易状态")
        return None
    if df_min.empty:
        print(f"❌ 未找到 {symbol} 分钟 K 线")
        return None
    if 'avg_price_from_5s' not in df_min.columns:
        print(f"❌ {symbol} min 数据缺少 avg_price_from_5s 字段")
        return None

    print(f"✅ 状态 {len(df_status)} 条 | K线 {len(df_min)} 条")

    df_status.index = df_status.index.map(lambda x: x.replace(second=0, microsecond=0))
    df_status = df_status[~df_status.index.duplicated(keep='first')]
    df_status.index = df_status.index.map(lambda x: x.replace(hour=11, minute=30) if x.hour == 13 and x.minute == 0 else x)

    df = df_status[['now_pos']].join(
        df_min[['avg_price_from_5s', 'open', 'high', 'low', 'close', 'volume']], how='inner'
    )
    df = df.dropna(subset=['now_pos', 'avg_price_from_5s'])
    print(f"✅ 合并有效 {len(df)} 条")

    if len(df) < 11:
        print("❌ 有效数据不足")
        return None

    trades = calc_trade_lst(df)
    trade_df = pd.DataFrame(trades, columns=['flag', 'time_o', 'time_c', 'price_o', 'price_c'])

    result = {'symbol': symbol, 'date': date, 'num_trades': 0, 'total_profit': 0}
    if not trade_df.empty:
        trade_df['profit'] = trade_df['flag'] * (trade_df['price_c'] - trade_df['price_o'])
        trade_df['cum_profit'] = trade_df['profit'].cumsum()
        result.update({
            'num_trades': len(trade_df),
            'total_profit': trade_df['profit'].sum(),
            'max_profit': trade_df['profit'].max(),
            'max_loss': trade_df['profit'].min(),
        })
        print(f"\n交易 {result['num_trades']} 笔 | 总盈亏 {result['total_profit']:.2f} | 最大盈利 {result['max_profit']:.2f} | 最大亏损 {result['max_loss']:.2f}")
    else:
        print("\n⚠️ 未识别到交易")

    fig = plot_trading_chart(df_min, trade_df, symbol, date)
    if fig and save_plot:
        out_dir = os.path.join(SAVE_ROOT, date_fmt, 'analysis')
        os.makedirs(out_dir, exist_ok=True)
        fig.write_html(os.path.join(out_dir, f"{symbol}_trading_signals.html"))
        if not trade_df.empty:
            trade_df.to_csv(os.path.join(out_dir, f"{symbol}_trades.csv"), index=False)
        print(f"📊 已保存到 {out_dir}")
    if fig and show_plot:
        fig.show()

    result['trade_df'] = trade_df
    result['df'] = df
    return result


def analyze_all_symbols(date: str, save_plot: bool = True, show_plot: bool = False) -> List[Dict]:
    """分析所有品种"""
    results = []
    for symbol in SYMBOLS:
        try:
            res = analyze_trading_day(symbol, date, save_plot=save_plot, show_plot=show_plot)
            if res:
                results.append(res)
        except Exception as e:
            print(f"❌ {symbol} 失败: {e}")
    return results


if __name__ == "__main__":
    import sys
    symbol = sys.argv[1] if len(sys.argv) >= 2 else 'C'
    date = sys.argv[2] if len(sys.argv) >= 3 else '2026-05-19'
    analyze_trading_day(symbol, date, save_plot=True, show_plot=False)
