"""
策略编排器 (StrategyOrchestrator)
====================================
统一入口，协调 DataService 与多个策略的执行。

核心约束：
- 4个策略必须并行执行，总耗时 < 5秒
- 使用 ThreadPoolExecutor（多线程），避免进程间大数据序列化开销
- numpy/pandas 操作会释放 GIL，多线程可有效利用多核

运行方式：
    python orchestrator.py
"""

import argparse
import logging
import multiprocessing
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from queue import Queue, Empty

# 北京时间 (UTC+8)
BEIJING_TZ = timezone(timedelta(hours=8))

import pandas as pd

from data_function import load_config
from data_service import DataService
from strategies import BaseStrategy, _IMPORTED_FACTOR_FUNCS

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# 日志配置：同时输出到控制台和 logs/orchestrator_YYYYMMDD.log
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
logs_dir = os.path.join(current_dir, "logs")
os.makedirs(logs_dir, exist_ok=True)

log_file = os.path.join(logs_dir, f"orchestrator_{datetime.now(BEIJING_TZ).strftime('%Y%m%d')}.log")

# 创建logger
logger = logging.getLogger("orchestrator")
logger.setLevel(logging.INFO)
logger.handlers.clear()

# FileHandler - 写入日志文件
fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))

# StreamHandler - 同时输出到stdout（屏幕）
sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.INFO)
sh.setFormatter(logging.Formatter('%(message)s'))

logger.addHandler(fh)
logger.addHandler(sh)

# 拦截所有print，改为logger.info
_original_print = print
def _logged_print(*args, **kwargs):
    logger.info(' '.join(str(a) for a in args))
print = _logged_print


# 子进程全局缓存：每个 worker 进程只创建一次 BaseStrategy
_WORKER_STRATEGIES = {}


def run_strategy(strategy_config, data_package, time_recently, readable_time):
    """
    包装函数：在进程池中执行单个策略的纯计算部分（因子生成 + 模型预测）。
    子进程内缓存 BaseStrategy 对象，避免每轮重复加载模型。
    子进程不再维护 pred_df/fac_df，也不保存缓存，避免多进程竞争写入和状态漂移。
    返回结果字典，包含 pred / fac_df / data_dict / timings。
    """
    import os
    import sys
    import time
    import pandas as pd
    from strategies import BaseStrategy

    # suppress 子进程中的 stdout 噪音（错误信息通过 result['err'] 回传）
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    try:
        sym = strategy_config['main_symbol']
        st = _WORKER_STRATEGIES.get(sym)
        if st is None:
            st = BaseStrategy(
                strategy_config['main_symbol'],
                strategy_config['other_symbols'],
                strategy_config['config'],
                strategy_config['generate_factor_func']
            )
            _WORKER_STRATEGIES[sym] = st

        t0 = time.time()
        err = None
        pred = None
        fac_df = None
        data_dict = None
        timings = {}
        try:
            pred, fac_df, data_dict, timings = st.compute(data_package, time_recently)
        except Exception as e:
            import traceback
            err = f"{e}\n{traceback.format_exc()}"
        elapsed = time.time() - t0
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout

    return {
        'sym': st.main_symbol,
        'err': err,
        'elapsed': elapsed,
        'pred': pred,
        'fac_df': fac_df,
        'data_dict': data_dict,
        'timings': timings,
    }


def main():
    # --------------------------------------------------------------------
    # 0. 解析命令行参数
    # --------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="策略编排器")
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="指定本次运行的品种白名单（如 --symbols C LH），默认运行 config 中所有已有模型的品种",
    )
    args = parser.parse_args()

    # --------------------------------------------------------------------
    # 1. 加载配置
    # --------------------------------------------------------------------
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "config.json")
    config = load_config(config_path)
    print("[Orchestrator] 配置加载完成")

    # --------------------------------------------------------------------
    # 2. 启动 DataService
    # --------------------------------------------------------------------
    data_queue = Queue(maxsize=5)
    data_service = DataService(
        config=config,
        data_queue=data_queue,
        poll_interval=0.5,
        tick_limit=50000,
    )
    data_service.initialize()

    # 创建 ProcessPoolExecutor（spawn 模式避免 fork 死锁）
    mp_ctx = multiprocessing.get_context('spawn')
    executor = ProcessPoolExecutor(max_workers=8, mp_context=mp_ctx)

    data_service.start()

    # --------------------------------------------------------------------
    # 3. 初始化策略
    # --------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[Orchestrator] 初始化策略...")
    print("=" * 70)

    # 白名单优先级：命令行参数 > config['active_symbols'] > config['symbols']
    if args.symbols:
        active_symbols = args.symbols
    else:
        active_symbols = config.get('active_symbols', config.get('symbols', []))

    # 启动前清理上一次的实时结果文件（只在这里执行一次，子进程不再清理）
    print("[Orchestrator] 清理上一次实时结果文件...")
    for sym in active_symbols:
        save_path = os.path.join(current_dir, "save_files", sym)
        if os.path.isdir(save_path):
            # 清理根目录下旧格式文件（兼容旧版本）
            for f in os.listdir(save_path):
                if f.startswith(('predictions_', 'factors_', 'trading_status_',
                                 'pred_df.csv', 'fac_old.csv')):
                    os.remove(os.path.join(save_path, f))
            # 清理子目录中的文件
            for subdir in ['factors', 'predictions', 'json', 'data']:
                sub_path = os.path.join(save_path, subdir)
                if os.path.isdir(sub_path):
                    for f in os.listdir(sub_path):
                        if f.startswith(('predictions_', 'factors_', 'trading_status_', 'main_min_', '_min_')) or f.startswith(f'{sym}_min_'):
                            os.remove(os.path.join(sub_path, f))

    strategies = []
    for sym in active_symbols:
        func_name = f'generate_factor_dataframe_{sym}'
        if func_name not in _IMPORTED_FACTOR_FUNCS:
            print(f"  ⚠️ 跳过 {sym}: 未找到 {func_name}（模型可能尚未训练）")
            continue

        # 检查模型文件是否存在，避免初始化时崩溃
        model_path = os.path.join(config['paths']['models_root'], sym)
        if not os.path.isdir(model_path) or not any(f.endswith('.lgb') for f in os.listdir(model_path)):
            print(f"  ⚠️ 跳过 {sym}: 模型目录不存在或为空 {model_path}")
            continue

        other_symbols = [s for s in config['symbols'] if s != sym]
        st = BaseStrategy(sym, other_symbols, config, _IMPORTED_FACTOR_FUNCS[func_name])
        strategies.append(st)

    if not strategies:
        print("[Orchestrator] 错误: 没有可运行的策略，退出")
        return

    print("=" * 70)
    print("[Orchestrator] 所有策略就绪，进入主循环")
    print("提示: 按 Ctrl+C 停止")
    print("=" * 70 + "\n")

    time_config = config['time_config']
    loop_count = 0
    first_data_received = False

    # --------------------------------------------------------------------
    # 4. 主循环：阻塞等待数据包 → 并行执行策略
    # --------------------------------------------------------------------
    try:
        while True:
            try:
                pkg = data_queue.get(timeout=60)
            except Empty:
                continue

            time_recently = pkg['timestamp']
            if not first_data_received:
                print("[Orchestrator] 🎉 收到第一条数据包，开始运行")
                first_data_received = True
            readable_time = datetime.now(BEIJING_TZ).strftime("%Y-%m-%d %H:%M:%S")
            loop_count += 1

            # 15点收盘退出
            if time_recently.hour == 15:
                print("\n[Orchestrator] 检测到15点收盘，退出主循环")
                break

            # 只打印关键信息
            if loop_count <= 3 or loop_count % 10 == 0:
                print(f"[Orchestrator] ⏰ 新分钟 #{loop_count} | 行情:{time_recently} | 系统:{readable_time}")

            # ---------------------------------------------------------------
            # 数据齐备性检查：交易时段内的品种长度不齐则等待重试
            # ---------------------------------------------------------------
            ready_strategies = strategies.copy()
            for retry in range(10):
                not_ready = []
                for st in ready_strategies:
                    data_dict = st._extract_data(pkg, time_recently)
                    if not st._align_and_check(data_dict, time_recently):
                        not_ready.append(st.main_symbol)
                if not not_ready:
                    break
                print(f"[Orchestrator] 数据不齐 {not_ready}，等待 0.5s 重试 ({retry+1}/10)")
                time.sleep(0.5)
                # 强制刷新所有品种数据后再打包
                for inst in data_service.instruments:
                    data_service._refresh_instrument(inst)
                pkg = data_service._package_data(time_recently)
            else:
                # 10 次重试后仍不齐，打印警告但不剔除策略
                print(f"[Orchestrator] ⚠️ 重试 10 次后数据仍不齐: {not_ready}，继续执行（强制对齐）")

                    # ---------------------------------------------------------------
            # 构建 strategy_configs（轻量字典，供子进程创建 BaseStrategy）
            # ---------------------------------------------------------------
            strategy_configs = []
            for st in ready_strategies:
                strategy_configs.append({
                    'main_symbol': st.main_symbol,
                    'other_symbols': st.other_symbols,
                    'config': config,
                    'generate_factor_func': _IMPORTED_FACTOR_FUNCS[f'generate_factor_dataframe_{st.main_symbol}'],
                })

            # ---------------------------------------------------------------
            # 并行执行所有策略（ProcessPoolExecutor）
            # ---------------------------------------------------------------
            total_t0 = time.time()

            results = {}
            max_workers = min(len(ready_strategies), 8)
            futures = {
                executor.submit(
                    run_strategy, cfg, pkg, time_recently, readable_time
                ): cfg['main_symbol']
                for cfg, st in zip(strategy_configs, ready_strategies)
            }

            for future in as_completed(futures):
                sym = futures[future]
                result = future.result()
                results[sym] = result
                if result.get('err'):
                    print(f"[{sym}] ❌ 异常 ({result['elapsed']:.2f}s): {result['err'][:200]}")
                    continue

                # 主进程维护状态：合并子进程返回的计算结果，执行策略逻辑
                for st in ready_strategies:
                    if st.main_symbol != sym:
                        continue
                    pred = result.get('pred')
                    fac_df = result.get('fac_df')
                    data_dict = result.get('data_dict')
                    timings = result.get('timings', {})

                    if pred is None or fac_df is None:
                        print(f"[{sym}] ⚠️ 计算返回空，跳过策略逻辑")
                        st._last_signal = '-'
                        st._last_timings = timings
                        st._last_fac_shape = '-'
                        st._last_weighted_s = []
                        st._last_quantile = '-'
                        st._last_pred_t2 = st._last_pred_t1 = st._last_pred_t0 = None
                        break

                    # 更新 pred_df / fac_df（主进程统一维护，避免子进程漂移丢数据）
                    pred_tail = pred.iloc[-3:]
                    new_mask = ~pred_tail.index.isin(st.pred_df.index)
                    if new_mask.any():
                        st.pred_df = pd.concat([st.pred_df, pred_tail[new_mask]])
                        st.pred_df = st.pred_df[~st.pred_df.index.duplicated(keep='last')]

                    fac_new_mask = ~fac_df.index.isin(st.fac_df.index)
                    if fac_new_mask.any():
                        st.fac_df = pd.concat([st.fac_df, fac_df[fac_new_mask]])
                        st.fac_df = st.fac_df[~st.fac_df.index.duplicated(keep='last')]

                    # 在主进程中执行策略逻辑（run_345 + 保存）
                    st.run_logic(pred, fac_df, data_dict, time_recently, readable_time)
                    break

            # 打印各品种 weighted_s（按 symbol 排序，方便对比）
            for st in ready_strategies:
                ws = getattr(st, '_last_weighted_s', [])
                if ws:
                    ws_str = ', '.join(f'{x:+.4f}' for x in ws)
                    print(f"{st.main_symbol:>2s}: [{ws_str}]")

            # 打印汇总表格（按 symbol 排序）
            rows = []
            for st in ready_strategies:
                sym = st.main_symbol
                tm = getattr(st, '_last_timings', {})
                rows.append({
                    'symbol': sym,
                    'pos': st.now_pos,
                    'hold': st.now_holding,
                    'elapsed': round(results.get(sym, {}).get('elapsed', 0), 2),
                    'gen': tm.get('generate_factor', 0),
                    'pred_t1': getattr(st, '_last_pred_t1', '-'),
                    'pred_t0': getattr(st, '_last_pred_t0', '-'),
                    'pct': getattr(st, '_last_quantile', '-'),
                    'signal': getattr(st, '_last_signal', '-'),
                })
            summary = pd.DataFrame(rows).set_index('symbol')
            # 对数值列做格式化，避免科学计数法
            def _fmt(v):
                if isinstance(v, float):
                    return f"{v:+.6f}" if v != 0 else "0.000000"
                return v
            for col in ['pred_t1', 'pred_t0']:
                if col in summary.columns:
                    summary[col] = summary[col].apply(_fmt)
            print(f"\n{'='*100}")
            print(f"[Orchestrator] #{loop_count} | market: {time_recently} | sys: {readable_time}")
            print(f"{'='*100}")
            print(summary.to_string())
            print(f"{'='*100}")

            total_elapsed = time.time() - total_t0
            if total_elapsed > 5.0:
                print(f"⚠️ 警告: 总耗时 {total_elapsed:.2f}s 超过5秒阈值")
            elif total_elapsed > 3.0:
                print(f"⚡ 提醒: 总耗时 {total_elapsed:.2f}s 接近阈值")

            # if time_recently.hour == 21 and time_recently.minute == 15:
            #     print("[Orchestrator] 21:15 快照完成，退出程序")
            #     sys.exit(0)

    except KeyboardInterrupt:
        print("\n[Orchestrator] 收到用户中断信号 (Ctrl+C)")
    finally:
        print("[Orchestrator] 关闭进程池...")
        executor.shutdown(wait=True)
        # 保存各策略的因子缓存（按交易日拆分）
        print("[Orchestrator] 保存策略缓存...")
        for st in strategies:
            try:
                st._save_fac_cache()
            except Exception as e:
                print(f"[{st.main_symbol}] 缓存保存失败: {e}")
        data_service.stop()
        print("[Orchestrator] 程序结束")


if __name__ == "__main__":
    main()
