#!/usr/bin/env python3
"""
Master 脚本：顺序训练 + 顺序回测（双参数组对比）
===============================================
1. 等 20250101 训练完成（监控已有进程）
2. 依次跑 20250401/20250701/20251001/20260101 训练
3. 全部训练完成后，依次跑 5 个日期的回测

用法（后台运行）:
    cd /home/strategy_res/single/dce_农
    nohup python master_run.py > master_run.log 2>&1 &
"""

import subprocess
import time
import sys
import os
from pathlib import Path
import datetime

BASE_DIR = Path("/home/strategy_res/single/dce_农")
DATES = ["20250101", "20250401", "20250701", "20251001", "20260101"]
MONITOR_PID = 61384  # 当前正在跑的 20250101/train.py


def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    sys.stdout.flush()


def wait_for_pid(pid):
    """等指定 PID 的进程结束"""
    log(f"等待进程 {pid} (20250101/train.py) 结束...")
    check_interval = 60  # 每分钟检查一次
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            log(f"进程 {pid} 已结束")
            return True
        except PermissionError:
            log(f"进程 {pid} 无权限访问，假设已结束")
            return True
        time.sleep(check_interval)


def run_train(date_str):
    """跑指定日期的训练"""
    script = BASE_DIR / date_str / "train.py"
    work_dir = BASE_DIR / date_str
    log(f"{'='*60}")
    log(f"[训练开始] {date_str}")
    log(f"{'='*60}")
    start = time.time()
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(work_dir),
        capture_output=False,
    )
    elapsed = time.time() - start
    success = result.returncode == 0
    status = "✓ 成功" if success else f"✗ 失败 (code={result.returncode})"
    log(f"[训练结束] {date_str} {status}，耗时 {elapsed/60:.1f} 分钟")
    return success


def run_backtest(date_str):
    """跑指定日期的回测"""
    script = BASE_DIR / date_str / "backtest.py"
    work_dir = BASE_DIR / date_str
    log(f"{'='*60}")
    log(f"[回测开始] {date_str}")
    log(f"{'='*60}")
    start = time.time()
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(work_dir),
        capture_output=False,
    )
    elapsed = time.time() - start
    success = result.returncode == 0
    status = "✓ 成功" if success else f"✗ 失败 (code={result.returncode})"
    log(f"[回测结束] {date_str} {status}，耗时 {elapsed/60:.1f} 分钟")
    return success


def main():
    log("=" * 60)
    log("Master Run 启动")
    log(f"监控 PID: {MONITOR_PID}")
    log(f"训练日期: {DATES}")
    log(f"回测参数组: default (0.9/0.5) + th08_04 (0.8/0.4)")
    log("=" * 60)

    # 1. 等 20250101 训练完成
    if not wait_for_pid(MONITOR_PID):
        log("监控进程异常退出")
        return

    # 2. 依次跑其余日期的训练
    train_success = []
    for date_str in DATES[1:]:
        ok = run_train(date_str)
        train_success.append((date_str, ok))

    log(f"{'='*60}")
    log("训练阶段总结:")
    for date_str, ok in [(DATES[0], True)] + train_success:
        log(f"  {date_str}: {'✓' if ok else '✗'}")
    log(f"{'='*60}")

    # 3. 依次回测（所有日期）
    log("全部训练完成，开始回测阶段")
    for date_str in DATES:
        run_backtest(date_str)

    log(f"{'='*60}")
    log("全部完成！")
    log(f"回测结果保存在各日期目录的 backtest/ 子目录下")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
