"""
Investigate market data (1min) differences between research and realtime env.
Focus on volume/turnover diffs at boundary times (11:30, 23:00).
"""
import pandas as pd
import numpy as np
import sys

# ========== Config ==========
SYMBOL = "P"
CONTRACT = "p2605"
DATES = ["2026-03-17", "2026-03-18", "2026-03-19", "2026-03-20", "2026-03-23"]

FILES_ROOT = "/home/strategy_PAMY_dev/files"
RESEARCH_1MIN = f"/mnt/Data/writable/liaoyuyang/data/1min/{SYMBOL}/{CONTRACT}.feather"
RESEARCH_TICK = f"/mnt/Data/writable/liaoyuyang/data/level2_all/{SYMBOL}/{CONTRACT}.feather"
REALTIME_1MIN = f"{FILES_ROOT}/{CONTRACT}_min.csv"
REALTIME_TICK = f"{FILES_ROOT}/{CONTRACT}_tick.csv"

TOL_ABS = 0.01

# ========== Helper ==========
def load_1min(path, name):
    if path.endswith('.feather'):
        df = pd.read_feather(path)
    else:
        df = pd.read_csv(path, parse_dates=['datetime'])
    # research uses 'ts', realtime uses 'datetime'
    time_col = 'ts' if 'ts' in df.columns else 'datetime'
    df['datetime'] = pd.to_datetime(df[time_col])
    if hasattr(df['trade_date'], 'dt'):
        df['trade_date_dt'] = df['trade_date'].dt.date
    else:
        df['trade_date_dt'] = pd.to_datetime(df['trade_date']).dt.date
    print(f"{name} 1min loaded | shape={df.shape} | time_col={time_col} | cols={df.columns.tolist()[:10]}...")
    return df


def load_tick(path, name):
    if path.endswith('.feather'):
        df = pd.read_feather(path)
    else:
        df = pd.read_csv(path, parse_dates=['datetime'])
    df['datetime'] = pd.to_datetime(df['datetime'])
    print(f"{name} tick loaded | shape={df.shape}")
    return df


def filter_by_trade_dates(df, dates):
    target_dates = [pd.to_datetime(d).date() for d in dates]
    if 'trade_date_dt' not in df.columns:
        if hasattr(df['trade_date'], 'dt'):
            df['trade_date_dt'] = df['trade_date'].dt.date
        else:
            df['trade_date_dt'] = pd.to_datetime(df['trade_date']).dt.date
    return df[df['trade_date_dt'].isin(target_dates)].copy()


def inspect_minute(research_tick, realtime_tick, minute_dt, label=""):
    """Compare tick-level details for a specific minute."""
    rs_min = research_tick[research_tick['ts'] == minute_dt].copy()
    rt_min = realtime_tick[realtime_tick['ts'] == minute_dt].copy()

    print(f"\n{'='*60}")
    print(f"Inspecting {label} | ts = {minute_dt}")
    print(f"{'='*60}")

    # Basic stats
    print(f"Research ticks: {len(rs_min)} | volume_sum={rs_min['volume'].sum():.2f} | turnover_sum={rs_min['turnover'].sum():.2f}")
    print(f"Realtime ticks: {len(rt_min)} | volume_sum={rt_min['volume'].sum():.2f} | turnover_sum={rt_min['turnover'].sum():.2f}")

    if len(rs_min) == 0 and len(rt_min) == 0:
        print("No ticks in either side.")
        return

    # Check for duplicates
    rs_dups = rs_min.duplicated(subset=['datetime', 'last_price', 'volume', 'turnover', 'open_interest'], keep=False)
    rt_dups = rt_min.duplicated(subset=['datetime', 'last_price', 'volume', 'turnover', 'open_interest'], keep=False)
    print(f"Research duplicate ticks (all fields): {rs_dups.sum()}")
    print(f"Realtime duplicate ticks (all fields):  {rt_dups.sum()}")

    rs_dups_dt = rs_min.duplicated(subset=['datetime'], keep=False)
    rt_dups_dt = rt_min.duplicated(subset=['datetime'], keep=False)
    print(f"Research duplicate ticks (datetime only): {rs_dups_dt.sum()}")
    print(f"Realtime duplicate ticks (datetime only):  {rt_dups_dt.sum()}")

    # Show first/last tick
    if len(rs_min) > 0:
        print("\nResearch first tick:")
        print(rs_min.head(1)[['datetime', 'last_price', 'volume', 'turnover', 'open_interest', 'TotalTradeVolume', 'TotalTradeValue']].to_string())
        print("Research last tick:")
        print(rs_min.tail(1)[['datetime', 'last_price', 'volume', 'turnover', 'open_interest', 'TotalTradeVolume', 'TotalTradeValue']].to_string())
    if len(rt_min) > 0:
        print("\nRealtime first tick:")
        print(rt_min.head(1)[['datetime', 'last_price', 'volume', 'turnover', 'open_interest', 'TotalTradeVolume', 'TotalTradeValue']].to_string())
        print("Realtime last tick:")
        print(rt_min.tail(1)[['datetime', 'last_price', 'volume', 'turnover', 'open_interest', 'TotalTradeVolume', 'TotalTradeValue']].to_string())

    # Show tick count per second
    if len(rs_min) > 0:
        rs_min['sec'] = rs_min['datetime'].dt.floor('1s')
        print("\nResearch tick count per second (top 10):")
        print(rs_min.groupby('sec').size().sort_values(ascending=False).head(10).to_string())
    if len(rt_min) > 0:
        rt_min['sec'] = rt_min['datetime'].dt.floor('1s')
        print("\nRealtime tick count per second (top 10):")
        print(rt_min.groupby('sec').size().sort_values(ascending=False).head(10).to_string())

    # Diff in volume/turnover by tick
    if len(rs_min) > 0 and len(rt_min) > 0:
        merged = pd.merge(
            rs_min[['datetime', 'volume', 'turnover']].rename(columns={'volume': 'rs_vol', 'turnover': 'rs_to'}),
            rt_min[['datetime', 'volume', 'turnover']].rename(columns={'volume': 'rt_vol', 'turnover': 'rt_to'}),
            on='datetime',
            how='outer',
            indicator=True
        )
        only_rs = merged[merged['_merge'] == 'left_only']
        only_rt = merged[merged['_merge'] == 'right_only']
        both = merged[merged['_merge'] == 'both']
        diff_rows = both[(both['rs_vol'] != both['rt_vol']) | (both['rs_to'] != both['rt_to'])]

        print(f"\nMerged tick comparison:")
        print(f"  Both sides:     {len(both)}")
        print(f"  Only research:  {len(only_rs)}")
        print(f"  Only realtime:  {len(only_rt)}")
        print(f"  Same datetime but diff values: {len(diff_rows)}")
        if len(diff_rows) > 0:
            print(diff_rows.head(10).to_string())
        if len(only_rs) > 0:
            print("\nOnly in research (first 5):")
            print(only_rs.head().to_string())
        if len(only_rt) > 0:
            print("\nOnly in realtime (first 5):")
            print(only_rt.head().to_string())


def inspect_prev_minute(research_tick, realtime_tick, minute_dt, label=""):
    """Inspect the previous minute to understand boundary carry-over."""
    prev = minute_dt - pd.Timedelta(minutes=1)
    rs_prev = research_tick[research_tick['ts'] == prev].copy()
    rt_prev = realtime_tick[realtime_tick['ts'] == prev].copy()
    print(f"\n--- Previous minute {prev} ---")
    print(f"Research ticks: {len(rs_prev)} | last volume={rs_prev['volume'].iloc[-1] if len(rs_prev)>0 else 'N/A'} | last turnover={rs_prev['turnover'].iloc[-1] if len(rs_prev)>0 else 'N/A'}")
    print(f"Realtime ticks: {len(rt_prev)} | last volume={rt_prev['volume'].iloc[-1] if len(rt_prev)>0 else 'N/A'} | last turnover={rt_prev['turnover'].iloc[-1] if len(rt_prev)>0 else 'N/A'}")
    if len(rs_prev) > 0:
        print(f"Research last tick: {rs_prev.tail(1)[['datetime','volume','turnover','TotalTradeVolume','TotalTradeValue']].to_string()}")
    if len(rt_prev) > 0:
        print(f"Realtime last tick: {rt_prev.tail(1)[['datetime','volume','turnover','TotalTradeVolume','TotalTradeValue']].to_string()}")


# ========== Main ==========
def main():
    print("Loading 1min data...")
    rs_1min = load_1min(RESEARCH_1MIN, "Research")
    rt_1min = load_1min(REALTIME_1MIN, "Realtime")

    rs_1min = filter_by_trade_dates(rs_1min, DATES)
    rt_1min = filter_by_trade_dates(rt_1min, DATES)
    print(f"After trade_date filter -> Research: {rs_1min.shape}, Realtime: {rt_1min.shape}")

    # Align
    rs_1min = rs_1min.set_index('datetime')
    rt_1min = rt_1min.set_index('datetime')
    common_idx = rs_1min.index.intersection(rt_1min.index)
    print(f"Common 1min bars: {len(common_idx)}")

    cols = ['open', 'high', 'low', 'close', 'volume', 'turnover', 'open_interest']
    rs_aligned = rs_1min.loc[common_idx, cols]
    rt_aligned = rt_1min.loc[common_idx, cols]
    diff = (rs_aligned - rt_aligned).abs()

    # Find problematic minutes
    vol_diff = diff['volume']
    to_diff = diff['turnover']
    bad = diff[(vol_diff > TOL_ABS) | (to_diff > TOL_ABS)].copy()
    bad['vol_diff'] = vol_diff
    bad['to_diff'] = to_diff
    bad = bad.sort_values('vol_diff', ascending=False)

    print(f"\n>>> Diff summary: {len(bad)} minutes have volume/turnover diffs")
    print(bad[['vol_diff', 'to_diff']].head(20).to_string())

    # Load tick data
    print("\nLoading tick data...")
    rs_tick = load_tick(RESEARCH_TICK, "Research")
    rt_tick = load_tick(REALTIME_TICK, "Realtime")

    # Filter tick data to target window (with some padding)
    target_dates = [pd.to_datetime(d).date() for d in DATES]
    if hasattr(rs_tick['trade_date'], 'dt'):
        rs_tick['trade_date_dt'] = rs_tick['trade_date'].dt.date
    else:
        rs_tick['trade_date_dt'] = pd.to_datetime(rs_tick['trade_date']).dt.date
    if hasattr(rt_tick['trade_date'], 'dt'):
        rt_tick['trade_date_dt'] = rt_tick['trade_date'].dt.date
    else:
        rt_tick['trade_date_dt'] = pd.to_datetime(rt_tick['trade_date']).dt.date

    # Keep trade_dates in target + previous day (for night session padding)
    all_dates = set(target_dates)
    for d in target_dates:
        all_dates.add(d - pd.Timedelta(days=1))
    all_dates = sorted(all_dates)

    rs_tick = rs_tick[rs_tick['trade_date_dt'].isin(all_dates)].copy()
    rt_tick = rt_tick[rt_tick['trade_date_dt'].isin(all_dates)].copy()
    print(f"Tick after date filter -> Research: {rs_tick.shape}, Realtime: {rt_tick.shape}")

    # Ensure ts column exists and is datetime
    if 'ts' not in rs_tick.columns:
        rs_tick['ts'] = rs_tick['datetime'].dt.ceil('1min')
    else:
        rs_tick['ts'] = pd.to_datetime(rs_tick['ts'])
    if 'ts' not in rt_tick.columns:
        rt_tick['ts'] = rt_tick['datetime'].dt.ceil('1min')
    else:
        rt_tick['ts'] = pd.to_datetime(rt_tick['ts'])

    # Inspect top diffs
    top_diff_minutes = bad.index[:10]
    for ts in top_diff_minutes:
        minute_dt = pd.to_datetime(ts)
        label = f"vol_diff={bad.loc[ts, 'vol_diff']:.2f} to_diff={bad.loc[ts, 'to_diff']:.2f}"
        inspect_minute(rs_tick, rt_tick, minute_dt, label=label)
        inspect_prev_minute(rs_tick, rt_tick, minute_dt, label=label)

    print("\nDone.")


if __name__ == "__main__":
    main()
