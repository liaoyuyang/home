"""
独立数据服务 (DataService)
==============================
设计目标：
1. 与 sql_writer_dce.py 完全解耦，策略端不直接读DB
2. 统一维护所有合约的tick缓存和分钟线缓存
3. 增量更新，避免全表扫描
4. 新分钟到来时，推送完整数据包供策略使用

架构位置：
    sql_writer_dce (写DB)  ←→  tick_data.db  ←→  DataService (读DB+缓存)
                                                          ↓
                                              数据包 (Queue)
                                                          ↓
                                               StrategyOrchestrator
"""

import os
import sqlite3
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from queue import Queue

import pandas as pd
import numpy as np

from data_function import (
    parse_df, aggregate_ticks, update_concat,
    read_table, parse_time
)

warnings.filterwarnings('ignore')


class DataService:
    """
    行情数据服务
    
    职责：
    - 启动时加载历史分钟CSV到内存
    - 首次全量读取当前DB中的tick数据
    - 运行期增量读取新tick（通过rowid或limit）
    - 聚合tick为分钟线，合并到历史分钟线
    - 检测新分钟，推送数据包
    """

    def __init__(
        self,
        config: dict,
        data_queue: Queue,
        poll_interval: float = 0.5,
        tick_limit: int = 50000,
    ):
        self.config = config
        self.data_queue = data_queue
        self.poll_interval = poll_interval
        self.tick_limit = tick_limit  # 每次从DB读多少条tick

        # 所有合约列表（去重）
        self.instruments = list(dict.fromkeys(
            v["contract"] for v in config["instruments"].values()
        ))
        
        # 主品种列表（用于新分钟检测）
        self.main_instruments = [
            config["instruments"][s]["contract"]
            for s in config.get("symbols", [])
        ]

        # 内存缓存
        self.tick_cache: Dict[str, pd.DataFrame] = {}
        self.min_cache: Dict[str, pd.DataFrame] = {}
        self.last_rowid: Dict[str, int] = {}

        # 路径与配置
        self.db_path = config["paths"]["db_path"]
        self.recent_data_path = config["paths"]["load_recent_data_path"]
        self.trade_hours = config["symbol_specs"]["P"]["trade_hours"]
        self.time_config = config["time_config"]

        # 运行控制
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ #
    # 初始化
    # ------------------------------------------------------------------ #
    def initialize(self):
        """启动初始化：加载历史CSV + 全量读取当前DB状态"""
        print("=" * 70)
        print("[DataService] 初始化中...")
        print("=" * 70)

        for inst in self.instruments:
            # 1) 加载历史分钟线CSV
            self.min_cache[inst] = self._load_historical_min(inst)

            # 2) 全量读取当前tick数据（仅启动时一次）
            df = read_table(inst, db_path=self.db_path, word=False, trade_type=self.trade_hours)
            if df is not None and not df.empty:
                self.tick_cache[inst] = df
                self._update_last_rowid(inst)

                # 3) 聚合已有tick到分钟线，合并到历史
                new_min = aggregate_ticks(df, time_config=self.time_config)
                self.min_cache[inst] = update_concat(self.min_cache[inst], new_min)

                print(f"  ✅ {inst:10s} | tick={len(df):>6,} | min={len(self.min_cache[inst]):>6,}")
            else:
                self.tick_cache[inst] = pd.DataFrame()
                self.last_rowid[inst] = 0
                print(f"  ⚠️ {inst:10s} | 无数据")

        print("=" * 70)
        print("[DataService] 初始化完成")
        print("=" * 70)

    def _load_historical_min(self, instrument: str) -> pd.DataFrame:
        """加载历史分钟数据CSV"""
        path = f'{self.recent_data_path}/{instrument}_min.csv'
        cols = [
            'datetime', 'open', 'high', 'low', 'close', 'volume',
            'turnover', 'instrument', 'open_interest', 'bar_count',
            'trade_date', 'LAST_TRADE_DATE'
        ]
        if not os.path.exists(path):
            print(f"    ⚠️ CSV不存在: {path}")
            return pd.DataFrame(columns=cols)

        df = pd.read_csv(path, parse_dates=['datetime'])
        return df.reindex(columns=cols)

    def _update_last_rowid(self, instrument: str):
        """记录当前表的最大rowid"""
        table = f"tick_data_{instrument}"
        try:
            conn = sqlite3.connect(f'{self.db_path}/tick_data.db', check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute(f"SELECT MAX(rowid) FROM {table}")
            max_rid = cursor.fetchone()[0]
            self.last_rowid[instrument] = max_rid or 0
            conn.close()
        except Exception as e:
            print(f"    ⚠️ 获取rowid失败 {instrument}: {e}")
            self.last_rowid[instrument] = 0

    # ------------------------------------------------------------------ #
    # 运行时更新
    # ------------------------------------------------------------------ #
    def _refresh_instrument(self, instrument: str) -> bool:
        """
        刷新单个合约的tick缓存（取最近N条）
        返回是否有新数据
        """
        try:
            # 用 limit 读最近N条，避免全表扫描
            df = read_table(
                instrument,
                db_path=self.db_path,
                word=False,
                trade_type=self.trade_hours,
                limit=self.tick_limit,
            )
            if df is None or df.empty:
                return False

            # 简单判断：如果行数或最后时间没变，认为无新数据
            old_df = self.tick_cache.get(instrument)
            if old_df is not None and not old_df.empty:
                if len(df) == len(old_df):
                    if df['datetime'].iloc[-1] == old_df['datetime'].iloc[-1]:
                        return False

            self.tick_cache[instrument] = df
            self._update_last_rowid(instrument)

            # 重新聚合分钟线
            new_min = aggregate_ticks(df, time_config=self.time_config)
            self.min_cache[instrument] = update_concat(self.min_cache[instrument], new_min)
            return True

        except Exception as e:
            print(f"❌ 刷新{instrument}失败: {e}")
            return False

    def _get_latest_datetime(self) -> Optional[pd.Timestamp]:
        """取所有主品种中最新的时间"""
        latest = None
        for inst in self.main_instruments:
            df = self.tick_cache.get(inst)
            if df is not None and not df.empty:
                dt = pd.to_datetime(df['datetime'].iloc[-1])
                if latest is None or dt > latest:
                    latest = dt
        return latest

    def _package_data(self, current_time: pd.Timestamp) -> dict:
        """
        打包数据供策略使用
        
        结构:
        {
            'tick':  {instrument: pd.DataFrame},
            'min':   {instrument: pd.DataFrame},
            'timestamp': pd.Timestamp
        }
        """
        pkg = {
            'tick': {},
            'min': {},
            'timestamp': current_time.replace(second=0, microsecond=0),
        }
        for inst in self.instruments:
            tick_df = self.tick_cache.get(inst)
            min_df = self.min_cache.get(inst)

            if tick_df is not None and not tick_df.empty:
                # 只传当前时间前最近1000条tick（最大窗口约120 tick，1000条足够覆盖滚动窗口）
                mask = tick_df['datetime'] <= current_time
                pkg['tick'][inst] = tick_df[mask].iloc[-1000:].copy()
            else:
                pkg['tick'][inst] = pd.DataFrame()

            pkg['min'][inst] = min_df.copy() if min_df is not None else pd.DataFrame()

        return pkg

    # ------------------------------------------------------------------ #
    # 主循环
    # ------------------------------------------------------------------ #
    def run(self):
        self._running = True
        time_before = pd.Timestamp('2026-01-25 20:59:00.000000')
        print(f"[DataService] 主循环开始，轮询间隔 {self.poll_interval}s")

        while self._running:
            updated_any = False
            for inst in self.instruments:
                if self._refresh_instrument(inst):
                    updated_any = True

            # 检测新分钟（以任意主品种最新时间为准）
            time_recently = self._get_latest_datetime()
            if time_recently is not None:
                trigger = (
                    time_recently.minute != time_before.minute
                    and not (time_before.hour == 20 and time_before.minute == 59)
                    and not (time_before.hour == 8 and time_before.minute == 59)
                    and not (time_recently.hour == 8 and time_recently.minute == 59)
                )

                if trigger:
                    raw_time = time_recently  # 保存原始时间（含秒），用于 tick 数据过滤
                    # 截断到整分钟，避免秒级数据被resample归入下一分钟
                    time_recently = time_recently.replace(second=0, microsecond=0)
                    print(f"[DataService] ⏰ 新分钟触发: {time_recently}")
                    time.sleep(1.0)  # 给DB写入留缓冲
                    # 重新刷新所有品种，确保该分钟 tick 已完整写入 DB 后再打包
                    for inst in self.instruments:
                        self._refresh_instrument(inst)
                    pkg = self._package_data(time_recently)
                    self.data_queue.put(pkg)

                # 无论是否触发，都要更新时间基准（否则初始值20:59会永远卡住）
                time_before = time_recently

            time.sleep(self.poll_interval)

    def start(self):
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()
        print("[DataService] 后台线程已启动")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        print("[DataService] 已停止")


if __name__ == "__main__":
    # 独立测试DataService
    from data_function import load_config

    cfg = load_config("/home/strategy_online/strategy_PAMY_dce/config.json")
    q = Queue(maxsize=10)

    ds = DataService(cfg, q, poll_interval=1.0)
    ds.initialize()
    ds.start()

    try:
        while True:
            pkg = q.get(timeout=10)
            print(f"[TEST] 收到数据包，timestamp={pkg['timestamp']}")
            for inst, df in pkg['min'].items():
                print(f"       {inst}: min={len(df)}", end="")
            print()
    except KeyboardInterrupt:
        ds.stop()
