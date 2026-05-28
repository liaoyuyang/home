"""
因子计算性能 benchmark（安全版，不测多进程避免序列化卡死）
"""
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_function import load_config, parse_df, time_scale_df, Factor_generator
from strategies import _FACTOR_CALLS

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
config = load_config(CONFIG_PATH)
recent_data_path = config['paths']['load_recent_data_path']
trade_hours = config['symbol_specs']['P']['trade_hours']

TEST_SYMBOL = 'B'
TEST_FUNC_NAME = f'generate_factor_dataframe_{TEST_SYMBOL}'

print(f"[Benchmark] 加载 {TEST_SYMBOL} 数据 ...")
other_symbols = [s for s in config['symbols'] if s != TEST_SYMBOL]
main_inst = config['instruments'][TEST_SYMBOL]['contract']
other_insts = [config['instruments'][s]['contract'] for s in other_symbols]

tick_data_main = pd.read_csv(f'{recent_data_path}/{main_inst}_tick.csv', parse_dates=['datetime'])
tick_data_main = parse_df(tick_data_main, trade_type=trade_hours, local=True)
tick_data_main = time_scale_df(tick_data_main, 'ts', trade_hours)
tick_data_main = tick_data_main.replace([np.inf, -np.inf], np.nan)

min_data_main = pd.read_csv(f'{recent_data_path}/{main_inst}_min.csv', parse_dates=['datetime']).reindex(
    columns=['datetime', 'open', 'high', 'low', 'close', 'volume', 'turnover',
             'instrument', 'open_interest', 'trade_date', 'LAST_TRADE_DATE']
)

other_min_dfs = []
for inst in other_insts:
    df = pd.read_csv(f'{recent_data_path}/{inst}_min.csv', parse_dates=['datetime']).reindex(
        columns=['datetime', 'open', 'high', 'low', 'close', 'volume', 'turnover',
                 'instrument', 'open_interest', 'trade_date', 'LAST_TRADE_DATE']
    )
    other_min_dfs.append(df)

fac_generator = Factor_generator(tick_data_main, min_data_main, *other_min_dfs)
fac_generator.dict_keys = [TEST_SYMBOL] + other_symbols
fac_generator.load_df_names()

valid_index = fac_generator.valid_index
factor_col = list({name for name, _ in _FACTOR_CALLS[TEST_FUNC_NAME]})

print(f"[Benchmark] valid_index={len(valid_index)}, 因子数={len(factor_col)}")


def _compute_one_worker(name_expr, fg=None):
    name, expr = name_expr
    try:
        val = eval(expr, {
            'fac_generator': fg,
            'np': np, 'pd': pd,
            '__builtins__': __builtins__
        })
        return name, val
    except Exception as e:
        print(f"  ⚠️ 因子 {name} 失败: {e}")
        return name, np.full(len(fg.valid_index), np.nan)


def run_factors(max_workers, fg):
    matches = _FACTOR_CALLS[TEST_FUNC_NAME]
    t0 = time.time()

    if max_workers == 1:
        factor_dict = {}
        for name_expr in matches:
            name, val = _compute_one_worker(name_expr, fg)
            factor_dict[name] = val
    else:
        def _worker(name_expr):
            return _compute_one_worker(name_expr, fg)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            factor_dict = dict(executor.map(_worker, matches))

    fac_df = pd.DataFrame(factor_dict, index=fg.valid_index)
    fac_df = fac_df.replace([np.inf, -np.inf], np.nan)
    _ = fac_df[factor_col].round(8)
    return time.time() - t0


configs = [1, 2, 4, 8, 16, 32]
results = []
print("\n[Benchmark] 开始测试 ...\n")

for workers in configs:
    try:
        t = run_factors(workers, fac_generator)
        label = "单线程" if workers == 1 else f"Thread({workers})"
        results.append((label, t))
        print(f"  {label:20s}  {t:.3f}s")
    except Exception as e:
        print(f"  Thread({workers}) 失败: {e}")

print("\n" + "=" * 50)
print("汇总（按耗时排序）")
print("=" * 50)
results.sort(key=lambda x: x[1])
for name, t in results:
    print(f"{name:20s}  {t:.3f}s")
