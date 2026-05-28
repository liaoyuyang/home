#!/usr/bin/env python3
"""
独立 benchmark：验证 ProcessPoolExecutor 8 策略并行速度
进程池常驻，复用策略对象缓存
"""
import json, os, sys, time, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd

# ------------------------------------------------------------------
# 1. 加载配置
# ------------------------------------------------------------------
with open('config.json') as f:
    config = json.load(f)

# ------------------------------------------------------------------
# 2. 初始化 DataService 获取一个真实数据包
# ------------------------------------------------------------------
from data_service import DataService
import multiprocessing

data_queue = multiprocessing.Queue()
ds = DataService(config, data_queue, poll_interval=0.5, tick_limit=50000)
ds.initialize()

current_time = ds._get_latest_datetime()
if current_time is None:
    print("[Benchmark] 无数据，退出")
    sys.exit(1)

pkg = ds._package_data(current_time)
print(f"[Benchmark] DataPackage: timestamp={current_time}")
for sym, df in pkg['tick'].items():
    print(f"  tick[{sym}] = {len(df)} rows")
for sym, df in pkg['min'].items():
    print(f"  min [{sym}] = {len(df)} rows")

# ------------------------------------------------------------------
# 3. 构建 strategy_configs
# ------------------------------------------------------------------
import orchestrator as orch

strategy_configs = []
for sym in config['symbols']:
    inst = config['instruments'][sym]['contract']
    if inst not in pkg['tick'] or pkg['tick'][inst].empty:
        continue
    func_name = f'generate_factor_dataframe_{sym}'
    if func_name not in orch._IMPORTED_FACTOR_FUNCS:
        continue
    other_symbols = [s for s in config['symbols'] if s != sym]
    strategy_configs.append({
        'main_symbol': sym,
        'other_symbols': other_symbols,
        'config': config,
        'generate_factor_func': orch._IMPORTED_FACTOR_FUNCS[func_name],
    })

print(f"\n[Benchmark] 策略数: {len(strategy_configs)}")

# ------------------------------------------------------------------
# 4. 创建常驻进程池，连续跑 N 轮（第一轮含模型加载/JIT）
# ------------------------------------------------------------------
readable_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
N = 5
print(f"\n===== Benchmark {N} rounds (常驻进程池 + 策略缓存) =====")
executor = ProcessPoolExecutor(max_workers=8)
round_times = []
for i in range(N):
    t0 = time.time()
    futures = {
        executor.submit(orch.run_strategy, cfg, pkg, current_time, readable_time): cfg['main_symbol']
        for cfg in strategy_configs
    }
    results = {}
    for future in as_completed(futures):
        sym = futures[future]
        results[sym] = future.result()
    total = time.time() - t0
    round_times.append(total)
    max_single = max(r['elapsed'] for r in results.values())
    speedup = max_single / total * len(strategy_configs) if total > 0 else 0
    print(f"Round {i+1}: total={total:.3f}s, max_single={max_single:.3f}s, speedup={speedup:.1f}x")

executor.shutdown()

print(f"\n===== Summary =====")
print(f"Avg total: {sum(round_times)/len(round_times):.3f}s")
print(f"Min total: {min(round_times):.3f}s")
print(f"Max total: {max(round_times):.3f}s")
print(f"Steady-state (last 3 avg): {sum(round_times[-3:])/3:.3f}s")
