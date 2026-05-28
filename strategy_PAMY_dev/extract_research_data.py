#!/usr/bin/env python3
"""
从研究环境提取与实时 logger 同结构的源数据，用于对比排查。

用法：
    python extract_research_data.py 2026-03-23_225700

输出到 logs/factor_debug/research/20260323_225700/：
    tick.parquet      ← 从 level2 提取的 tick
    min.parquet       ← 从 1min feather 提取的 min
    valid_index.csv   ← 该分钟的时间戳
    info.json         ← 品种、因子、时间
"""

import sys
import json
import pandas as pd
from pathlib import Path

RESEARCH_PREFIX = Path('/mnt/Data/writable/liaoyuyang')
LOG_ROOT = Path('/home/strategy_PAMY_dev/logs/factor_debug')


def extract(symbol: str, trigger_str: str):
    """
    trigger_str: '2026-03-23_22:57:00' 或 '20260323_225700'
    """
    # 统一解析为 datetime
    try:
        dt = pd.to_datetime(trigger_str, format='%Y%m%d_%H%M%S')
    except ValueError:
        dt = pd.to_datetime(trigger_str)

    ts_str = dt.strftime('%Y%m%d_%H%M%S')
    out_dir = LOG_ROOT / 'research' / ts_str
    out_dir.mkdir(parents=True, exist_ok=True)

    # 合约映射（简化，实际需要根据品种查）
    contract_map = {
        'A': 'a2605', 'B': 'b2605', 'C': 'c2605', 'CS': 'cs2605',
        'M': 'm2605', 'Y': 'y2605', 'P': 'p2605', 'LH': 'lh2605'
    }
    contract = contract_map.get(symbol, f'{symbol.lower()}2605')

    # 1. 读取研究环境 1min
    min_path = RESEARCH_PREFIX / 'data' / '1min' / symbol / f'{contract}.feather'
    if min_path.exists():
        min_df = pd.read_feather(min_path)
        min_df['datetime'] = pd.to_datetime(min_df['ts'])
        # 取 trigger 前后 5 天的数据（RPP_5D 需要 5 天历史）
        start_dt = dt - pd.Timedelta(days=7)
        sub = min_df[(min_df['datetime'] >= start_dt) & (min_df['datetime'] <= dt)]
        sub.to_parquet(out_dir / 'min.parquet')
        print(f'[min] {len(sub)} rows → {out_dir / "min.parquet"}')
    else:
        print(f'⚠️ 未找到 {min_path}')

    # 2. 从 all_factor.feather 提取该分钟的因子值
    fac_path = RESEARCH_PREFIX / 'factor' / symbol / 'all_fac' / 'all_factor.feather'
    if fac_path.exists():
        fac_df = pd.read_feather(fac_path)
        fac_df['datetime'] = pd.to_datetime(fac_df['datetime'])
        row = fac_df[fac_df['datetime'] == dt]
        if not row.empty:
            row.to_parquet(out_dir / 'factor_row.parquet')
            print(f'[factor] 1 row → {out_dir / "factor_row.parquet"}')
        else:
            print(f'⚠️ 未找到 {dt} 的因子数据')
    else:
        print(f'⚠️ 未找到 {fac_path}')

    # 3. info.json
    info = {
        'trigger_time': dt.isoformat(),
        'symbol': symbol,
        'contract': contract,
    }
    with open(out_dir / 'info.json', 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False, default=str)

    print(f'完成: {out_dir}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('用法: python extract_research_data.py <trigger_time> [symbol]')
        print('  trigger_time: 2026-03-23_22:57:00 或 20260323_225700')
        print('  symbol: 默认 P')
        sys.exit(1)

    trigger_str = sys.argv[1]
    symbol = sys.argv[2] if len(sys.argv) > 2 else 'P'
    extract(symbol, trigger_str)
