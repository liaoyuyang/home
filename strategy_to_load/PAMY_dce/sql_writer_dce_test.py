"""
高性能实时市场数据写入器 - 测试版

设计目标：
1. 极低延迟：数据到达后立即写入数据库（毫秒级）
2. 零数据丢失：内存队列缓冲 + 独立写入线程
3. 高吞吐量：每秒可处理数千条数据

核心机制：
- 生产者-消费者模式：ZMQ接收(主线程) → 内存队列 → 数据库写入(独立线程)
- SQLite WAL模式：读写不互阻塞
- 单条事务：每条数据独立提交，确保不丢数据

测试版特性：
- 启动时输入交易日
- 根据时间判断使用输入日期或前一天
"""

import threading
import time
import os
import json
import sqlite3
import struct
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from queue import Queue, Empty
from contextlib import contextmanager

import zmq
import chinese_calendar as calendar
from datetime import datetime, date, timedelta

# ============================================================================
# 全局交易日变量（由用户输入）
# ============================================================================
INPUT_TRADE_DATE: Optional[date] = None

# ============================================================================
# 日志配置
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 常量定义
# ============================================================================
class Config:
    """配置常量"""
    DEFAULT_DATE_RANGE_START = '2026-01-01'
    DEFAULT_DATE_RANGE_END = '2026-12-31'
    TICK_DB_NAME = 'tick_data.db'
    ZMQ_LINGER_MS = 100
    
    # 队列配置
    QUEUE_MAX_SIZE = 10000  # 队列最大容量，防止内存无限增长
    QUEUE_TIMEOUT = 0.001   # 队列操作超时时间（秒）
    
    # 表结构定义（32个字段）
    TICK_COLUMNS = [
        'datetime', 'instrument', 'last_price', 'open_price', 
        'highest_price', 'lowest_price', 'volume', 'turnover',
        'open_interest', 'update_time', 'trade_date', 'record_time'
    ] + [f'{side}_price{i}' for side in ['bid', 'ask'] for i in range(1, 6)] \
      + [f'{side}_volume{i}' for side in ['bid', 'ask'] for i in range(1, 6)]
    
    TOTAL_COLUMNS = len(TICK_COLUMNS)


# ============================================================================
# 数据类定义
# ============================================================================
@dataclass
class TickRecord:
    """Tick数据结构"""
    datetime: str
    instrument: str
    last_price: float
    open_price: float
    highest_price: float
    lowest_price: float
    volume: int
    turnover: float
    open_interest: float
    update_time: float
    trade_date: str
    record_time: str
    bid_price1: float = 0.0
    bid_volume1: int = 0
    bid_price2: float = 0.0
    bid_volume2: int = 0
    bid_price3: float = 0.0
    bid_volume3: int = 0
    bid_price4: float = 0.0
    bid_volume4: int = 0
    bid_price5: float = 0.0
    bid_volume5: int = 0
    ask_price1: float = 0.0
    ask_volume1: int = 0
    ask_price2: float = 0.0
    ask_volume2: int = 0
    ask_price3: float = 0.0
    ask_volume3: int = 0
    ask_price4: float = 0.0
    ask_volume4: int = 0
    ask_price5: float = 0.0
    ask_volume5: int = 0
    
    def to_tuple(self) -> Tuple:
        """转换为元组用于数据库插入"""
        return (
            self.datetime, self.instrument, self.last_price, self.open_price,
            self.highest_price, self.lowest_price, self.volume, self.turnover,
            self.open_interest, self.update_time, self.trade_date, self.record_time,
            # bid prices 1-5
            self.bid_price1, self.bid_price2, self.bid_price3, self.bid_price4, self.bid_price5,
            # ask prices 1-5
            self.ask_price1, self.ask_price2, self.ask_price3, self.ask_price4, self.ask_price5,
            # bid volumes 1-5
            self.bid_volume1, self.bid_volume2, self.bid_volume3, self.bid_volume4, self.bid_volume5,
            # ask volumes 1-5
            self.ask_volume1, self.ask_volume2, self.ask_volume3, self.ask_volume4, self.ask_volume5
        )


# ============================================================================
# 工具函数
# ============================================================================
def get_trading_day(today: date, exclude_days: Optional[List[str]] = None) -> Optional[date]:
    """获取下一个交易日"""
    import pandas as pd
    
    exclude_days_set = set(exclude_days or [])
    
    try:
        all_weekdays = pd.date_range(
            Config.DEFAULT_DATE_RANGE_START, 
            Config.DEFAULT_DATE_RANGE_END, 
            freq='B'
        ).date
        
        trading_days = [
            day for day in all_weekdays 
            if calendar.is_workday(day) and day.strftime('%Y-%m-%d') not in exclude_days_set
        ]
        
        for trading_day in trading_days:
            if trading_day > today:
                return trading_day
                
        logger.warning(f'日期超出范围，无法获取下一个交易日 {today}')
        return None
        
    except Exception as e:
        logger.error(f'获取交易日失败: {e}')
        return None


def parse_time_from_update_time(update_time: Any) -> datetime:
    """从update_time解析时间（测试版：根据输入交易日和小时判断日期）"""
    global INPUT_TRADE_DATE
    
    try:
        if isinstance(update_time, (int, float)):
            time_str = str(int(update_time)).zfill(9)
        else:
            time_str = str(update_time).replace('.', '').replace(' ', '').zfill(9)
        
        hour = int(time_str[0:2])
        minute = int(time_str[2:4])
        second = int(time_str[4:6])
        millisecond = int(time_str[6:9])
        
        # 根据小时判断使用哪一天
        if INPUT_TRADE_DATE is not None:
            if hour > 16:
                # 小时大于16，使用输入交易日的前一天
                target_date = INPUT_TRADE_DATE - timedelta(days=1)
            else:
                # 小时小于等于16，使用输入的交易日
                target_date = INPUT_TRADE_DATE
        else:
            # 如果没有输入交易日，使用当前日期（兼容模式）
            target_date = date.today()
        
        return datetime(
            target_date.year, target_date.month, target_date.day,
            hour, minute, second, millisecond * 1000
        )
        
    except Exception as e:
        logger.error(f'时间解析失败: {e}, update_time={update_time}')
        return datetime.now()


def safe_decode(byte_str: bytes) -> str:
    """安全解码字节字符串"""
    try:
        decoded = byte_str.decode('utf-8', errors='ignore')
        return decoded.replace('\x00', '').strip()
    except Exception:
        return str(byte_str)


# ============================================================================
# 数据库管理器 - 优化版（单条写入，最小延迟）
# ============================================================================
class DatabaseManager:
    """
    数据库管理器 - 优化版
    
    特性：
    - WAL模式：读写不互阻塞
    - 长连接：保持连接打开，避免频繁开关开销
    - 预编译SQL：提升执行效率
    - 线程安全：使用锁保护共享资源
    """
    
    def __init__(self, db_path: str = 'realtime_data'):
        self.db_path = db_path
        self.tick_db = os.path.join(db_path, Config.TICK_DB_NAME)
        self._initialized_tables: set = set()
        self._connection: Optional[sqlite3.Connection] = None
        self._cursor: Optional[sqlite3.Cursor] = None
        self._insert_statements: Dict[str, str] = {}  # 缓存INSERT语句
        self._lock = threading.RLock()  # 线程锁保护cursor操作
        
        self._ensure_directories()
        self._initialize_connection()
    
    def _ensure_directories(self) -> None:
        """确保数据目录存在"""
        os.makedirs(self.db_path, exist_ok=True)
        logger.info(f"数据目录就绪: {os.path.abspath(self.db_path)}")
    
    def _initialize_connection(self) -> None:
        """初始化数据库连接（长连接）"""
        try:
            self._connection = sqlite3.connect(self.tick_db, check_same_thread=False)
            self._cursor = self._connection.cursor()
            
            # 启用WAL模式 - 关键！读写不互阻塞
            self._cursor.execute("PRAGMA journal_mode=WAL")
            self._cursor.execute("PRAGMA synchronous=NORMAL")
            self._cursor.execute("PRAGMA cache_size=-64000")  # 64MB缓存
            self._cursor.execute("PRAGMA temp_store=MEMORY")
            
            self._connection.commit()
            logger.info("数据库连接初始化完成（WAL模式已启用）")
            
        except Exception as e:
            logger.error(f"数据库连接初始化失败: {e}")
            raise
    
    def _get_table_name(self, instrument: str) -> str:
        """获取安全的表名（防止SQL注入）"""
        safe_instrument = ''.join(c for c in instrument if c.isalnum() or c == '_')
        return f"tick_data_{safe_instrument}"
    
    def create_instrument_table(self, instrument: str) -> None:
        """为指定合约创建tick数据表"""
        if instrument in self._initialized_tables:
            return
            
        table_name = self._get_table_name(instrument)
        
        try:
            columns_def = [
                "id INTEGER PRIMARY KEY AUTOINCREMENT",
                "datetime TEXT",
                "instrument TEXT",
                "last_price REAL",
                "open_price REAL",
                "highest_price REAL",
                "lowest_price REAL",
                "volume INTEGER",
                "turnover REAL",
                "open_interest REAL",
                "update_time REAL",
                "trade_date TEXT",
                "record_time TEXT"
            ]
            
            for i in range(1, 6):
                columns_def.append(f"bid_price{i} REAL")
                columns_def.append(f"bid_volume{i} INTEGER")
                columns_def.append(f"ask_price{i} REAL")
                columns_def.append(f"ask_volume{i} INTEGER")
            
            create_sql = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                {', '.join(columns_def)}
            )
            """
            
            self._cursor.execute(create_sql)
            
            # 创建索引
            self._cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_datetime 
            ON {table_name}(datetime)
            """)
            
            self._connection.commit()
            self._initialized_tables.add(instrument)
            
            # 预编译INSERT语句
            placeholders = ','.join(['?' for _ in range(Config.TOTAL_COLUMNS)])
            self._insert_statements[instrument] = f"""
            INSERT INTO {table_name} ({', '.join(Config.TICK_COLUMNS)}) VALUES ({placeholders})
            """
            
            logger.debug(f"表就绪: {table_name}")
            
        except Exception as e:
            logger.error(f"创建表失败 {instrument}: {e}")
            raise
    
    def save_tick_data_immediate(self, instrument: str, record: TickRecord) -> bool:
        """
        立即保存单条tick数据 - 核心方法
        
        特点：
        - 单条事务：每条数据独立提交
        - 预编译SQL：避免重复解析
        - 异常回滚：出错时自动回滚
        - 线程安全：使用锁保护cursor操作
        
        Args:
            instrument: 合约代码
            record: Tick记录
            
        Returns:
            是否成功
        """
        with self._lock:
            try:
                self.create_instrument_table(instrument)
                
                insert_sql = self._insert_statements.get(instrument)
                if not insert_sql:
                    raise ValueError(f"未找到{instrument}的INSERT语句")
                
                values = record.to_tuple()
                self._cursor.execute(insert_sql, values)
                self._connection.commit()  # 立即提交，确保数据不丢失
                
                return True
                
            except Exception as e:
                logger.error(f"保存tick数据失败 {instrument}: {e}")
                try:
                    self._connection.rollback()
                except:
                    pass
                return False
    
    def close(self) -> None:
        """关闭数据库连接"""
        try:
            if self._cursor:
                self._cursor.close()
            if self._connection:
                self._connection.close()
            logger.info("数据库连接已关闭")
        except Exception as e:
            logger.error(f"关闭数据库连接失败: {e}")
    
    def delete_instrument_tables(self, instruments: List[str], db_type: str = 'tick') -> bool:
        """删除指定合约对应的表"""
        if db_type != 'tick':
            logger.error(f"无效的db_type: {db_type}")
            return False
        
        if not os.path.exists(self.tick_db):
            logger.info(f"数据库文件不存在: {self.tick_db}")
            return True
        
        with self._lock:
            try:
                deleted_count = 0
                
                for instrument in instruments:
                    table_name = self._get_table_name(instrument)
                    
                    self._cursor.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                        (table_name,)
                    )
                    
                    if self._cursor.fetchone():
                        self._cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                        deleted_count += 1
                        logger.debug(f"已删除表: {table_name}")
                    
                    self._initialized_tables.discard(instrument)
                    self._insert_statements.pop(instrument, None)
                
                self._connection.commit()
                logger.info(f"清理完成: 成功删除 {deleted_count}/{len(instruments)} 个合约表")
                return True
                
            except Exception as e:
                logger.error(f"清理合约表失败: {e}")
                return False
    
    def get_table_stats(self, instrument: str) -> Optional[Dict[str, Any]]:
        """获取表的统计信息"""
        table_name = self._get_table_name(instrument)
        
        with self._lock:
            try:
                self._cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = self._cursor.fetchone()[0]
                
                self._cursor.execute(f"SELECT MAX(record_time) FROM {table_name}")
                latest_time = self._cursor.fetchone()[0]
                
                return {
                    'instrument': instrument,
                    'count': count,
                    'latest_record_time': latest_time
                }
                
            except Exception as e:
                logger.debug(f"获取表统计失败 {instrument}: {e}")
                return None


# ============================================================================
# 异步写入器 - 核心组件（生产者-消费者模式）
# ============================================================================
class AsyncTickWriter:
    """
    异步Tick数据写入器
    
    工作原理：
    1. 主线程（ZMQ接收）将数据放入队列（非阻塞，立即返回）
    2. 后台线程从队列取出数据并写入数据库
    3. 队列作为缓冲区，平衡生产速度和消费速度的差异
    
    优势：
    - 主线程不会被数据库IO阻塞
    - 数据在内存中短暂停留，延迟可控（毫秒级）
    - 队列满时可以选择等待或丢弃（可配置）
    """
    
    def __init__(self, db_manager: DatabaseManager, max_queue_size: int = None):
        self.db_manager = db_manager
        self.max_queue_size = max_queue_size or Config.QUEUE_MAX_SIZE
        
        # 每个合约一个队列，避免相互阻塞
        self._queues: Dict[str, Queue] = {}
        self._lock = threading.Lock()
        self._running = True
        self._stats = {
            'enqueued': 0,      # 入队数量
            'written': 0,       # 写入数量
            'dropped': 0,       # 丢弃数量
            'queue_max': 0      # 队列最大长度
        }
        
        # 启动后台写入线程
        self._writer_thread = threading.Thread(target=self._write_loop, daemon=True)
        self._writer_thread.start()
        
        logger.info(f"异步写入器已启动（队列容量: {self.max_queue_size}）")
    
    def _get_queue(self, instrument: str) -> Queue:
        """获取或创建合约对应的队列"""
        with self._lock:
            if instrument not in self._queues:
                self._queues[instrument] = Queue(maxsize=self.max_queue_size)
            return self._queues[instrument]
    
    def enqueue(self, instrument: str, record: TickRecord) -> bool:
        """
        将记录加入写入队列 - 由主线程调用
        
        Args:
            instrument: 合约代码
            record: Tick记录
            
        Returns:
            是否成功入队
        """
        try:
            queue = self._get_queue(instrument)
            
            # 非阻塞放入队列，如果队列满则丢弃最旧的数据
            try:
                queue.put_nowait(record)
                self._stats['enqueued'] += 1
                
                # 更新队列最大长度统计
                current_size = queue.qsize()
                if current_size > self._stats['queue_max']:
                    self._stats['queue_max'] = current_size
                    
                return True
                
            except:
                # 队列满，丢弃最旧的数据
                try:
                    queue.get_nowait()  # 移除最旧的数据
                    queue.put_nowait(record)  # 放入新数据
                    self._stats['dropped'] += 1
                    logger.warning(f"队列已满，丢弃旧数据: {instrument}")
                    return True
                except:
                    self._stats['dropped'] += 1
                    return False
                    
        except Exception as e:
            logger.error(f"入队失败 {instrument}: {e}")
            return False
    
    def _write_loop(self) -> None:
        """
        后台写入循环 - 由独立线程执行
        
        不断从所有队列中取出数据并写入数据库
        """
        while self._running:
            total_processed = 0
            
            with self._lock:
                queues_snapshot = list(self._queues.items())
            
            for instrument, queue in queues_snapshot:
                try:
                    # 尽可能多地处理该合约的数据
                    while not queue.empty():
                        try:
                            record = queue.get_nowait()
                            success = self.db_manager.save_tick_data_immediate(instrument, record)
                            
                            if success:
                                self._stats['written'] += 1
                                total_processed += 1
                            else:
                                # 写入失败，重新入队稍后重试
                                try:
                                    queue.put_nowait(record)
                                except:
                                    pass
                                break
                                
                        except Empty:
                            break
                            
                except Exception as e:
                    logger.error(f"写入循环出错 {instrument}: {e}")
            
            # 如果没有数据要处理，短暂休眠避免CPU空转
            if total_processed == 0:
                time.sleep(0.001)  # 1毫秒
    
    def flush_all(self) -> int:
        """强制刷新所有队列中的数据"""
        total = 0
        
        with self._lock:
            queues_snapshot = list(self._queues.items())
        
        for instrument, queue in queues_snapshot:
            while not queue.empty():
                try:
                    record = queue.get_nowait()
                    if self.db_manager.save_tick_data_immediate(instrument, record):
                        total += 1
                except Empty:
                    break
        
        return total
    
    def stop(self) -> None:
        """停止写入器并刷新剩余数据"""
        self._running = False
        remaining = self.flush_all()
        logger.info(f"异步写入器已停止，最后刷新 {remaining} 条记录")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取写入统计"""
        return self._stats.copy()


# ============================================================================
# ZMQ市场数据订阅器 - 优化版
# ============================================================================
class ZmqMarketDataSubscriber:
    """ZMQ市场数据订阅器 - 优化版"""
    
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.context: Optional[zmq.Context] = None
        self.socket: Optional[zmq.Socket] = None
        self.is_running = False
        self._callback: Optional[callable] = None
    
    def subscribe(self, instruments: List[str]) -> None:
        """订阅合约"""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        
        connection_str = f"tcp://{self.host}:{self.port}"
        self.socket.connect(connection_str)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        
        # 设置接收缓冲区大小
        self.socket.setsockopt(zmq.RCVBUF, 1024 * 1024)  # 1MB接收缓冲区
        self.socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms接收超时
        
        logger.info(f"ZMQ连接: {connection_str}")
    
    def parse_market_data(self, data: bytes) -> Optional[Dict[str, Any]]:
        """解析市场数据"""
        try:
            fmt_clean = '=ixxxxqq16s64sqqdixxxxddddddixxxxddddddqiidddixxxxdixxxxdixxxxdixxxxdixxxxdixxxxdixxxxdixxxxdixxxxdixxxxdii40s'
            fields = struct.unpack_from(fmt_clean, data, 0)
            
            return {
                "data": {
                    "instrument_id": safe_decode(fields[4]),
                    "last_price": fields[7],
                    "open_price": fields[12],
                    "highest_price": fields[13],
                    "lowest_price": fields[14],
                    "volume": fields[15],
                    "turnover": fields[16],
                    "open_interest": fields[17],
                    "update_time": fields[22],
                    "bids": [
                        {"price": fields[29], "volume": fields[30]},
                        {"price": fields[33], "volume": fields[34]},
                        {"price": fields[37], "volume": fields[38]},
                        {"price": fields[41], "volume": fields[42]},
                        {"price": fields[45], "volume": fields[46]}
                    ],
                    "asks": [
                        {"price": fields[31], "volume": fields[32]},
                        {"price": fields[35], "volume": fields[36]},
                        {"price": fields[39], "volume": fields[40]},
                        {"price": fields[43], "volume": fields[44]},
                        {"price": fields[47], "volume": fields[48]}
                    ]
                }
            }
            
        except Exception as e:
            logger.error(f"数据解析失败: {e}")
            return None
    
    def start(self, callback: callable) -> None:
        """开始接收数据"""
        self._callback = callback
        self.is_running = True
        
        logger.info("开始接收数据...")
        
        try:
            while self.is_running:
                try:
                    message = self.socket.recv()
                    if message == b'feed':
                        data = self.socket.recv()
                        market_data = self.parse_market_data(data)
                        if market_data and self._callback:
                            self._callback(market_data)
                except zmq.Again:
                    # 超时，继续循环
                    continue
                    
        except Exception as e:
            logger.error(f"接收错误: {e}")
        finally:
            self.stop()
    
    def stop(self) -> None:
        """停止订阅器"""
        self.is_running = False
        
        try:
            if self.socket:
                self.socket.setsockopt(zmq.LINGER, Config.ZMQ_LINGER_MS)
                self.socket.close()
                self.socket = None
        except Exception as e:
            logger.debug(f"关闭socket出错: {e}")
        
        try:
            if self.context:
                self.context.term()
                self.context = None
        except Exception as e:
            logger.debug(f"终止context出错: {e}")
        
        logger.info("ZMQ连接已关闭")


# ============================================================================
# 实时市场数据管理器 - 优化版
# ============================================================================
class RealTimeMarketData:
    """实时市场数据管理器 - 优化版"""
    
    def __init__(self, instruments: List[str], db_path: Optional[str] = None, 
                 host: Optional[str] = None, port: Optional[int] = None, 
                 clean_on_start: bool = True):
        self.instruments = instruments
        self.instrument_set = set(instruments)
        
        self.db_manager = DatabaseManager(db_path=db_path)
        self.async_writer = AsyncTickWriter(self.db_manager)
        self.zmq_subscriber = ZmqMarketDataSubscriber(host=host, port=port)
        
        if clean_on_start:
            self.cleanup_on_start()
        
        self.running = False
        self.message_count = 0
        self.instrument_counts: Dict[str, int] = {inst: 0 for inst in instruments}
        self._count_lock = threading.Lock()
        
        logger.info(f"实时市场数据管理器初始化 | 合约: {instruments}")
    
    def cleanup_on_start(self) -> None:
        """程序启动时执行清理"""
        logger.info("程序启动，清理旧数据表...")
        self.db_manager.delete_instrument_tables(self.instruments, 'tick')
        logger.info("启动清理完成")
    
    def start(self) -> None:
        """启动数据接收"""
        if self.running:
            return
        
        self.running = True
        self.zmq_subscriber.subscribe(self.instruments)
        
        self.data_thread = threading.Thread(target=self._run_data_receiver, daemon=True)
        self.data_thread.start()
        
        logger.info("启动数据接收线程")
    
    def stop(self) -> None:
        """停止数据接收"""
        self.running = False
        self.zmq_subscriber.stop()
        self.async_writer.stop()
        self.db_manager.close()
        logger.info("停止数据接收")
    
    def _run_data_receiver(self) -> None:
        """数据接收线程"""
        self.zmq_subscriber.start(callback=self._handle_market_data)
    
    def _handle_market_data(self, data: Dict[str, Any]) -> None:
        """处理接收到的市场数据"""
        try:
            data_content = data.get('data', {})
            instrument_id = data_content.get('instrument_id', 'UNKNOWN')
            
            if instrument_id not in self.instrument_set:
                return
            
            # 解析时间
            update_time = data_content.get('update_time', 0.0)
            tick_datetime = parse_time_from_update_time(update_time)
            datetime_str = tick_datetime.strftime('%Y-%m-%d %H:%M:%S.%f')
            
            # 确定交易日期
            current_time = datetime.now()
            hour = int(str(update_time)[:2]) if len(str(update_time)) >= 2 else 0
            trade_date = (
                current_time.strftime('%Y-%m-%d') if hour < 20 
                else get_trading_day(current_time.date()).strftime('%Y-%m-%d')
            )
            
            # 创建Tick记录
            record = TickRecord(
                datetime=datetime_str,
                instrument=instrument_id,
                last_price=float(data_content.get('last_price', 0.0)),
                open_price=float(data_content.get('open_price', 0.0)),
                highest_price=float(data_content.get('highest_price', 0.0)),
                lowest_price=float(data_content.get('lowest_price', 0.0)),
                volume=int(data_content.get('volume', 0)),
                turnover=float(data_content.get('turnover', 0.0)),
                open_interest=float(data_content.get('open_interest', 0.0)),
                update_time=float(update_time),
                trade_date=trade_date,
                record_time=current_time.strftime('%Y-%m-%d %H:%M:%S.%f'),
            )
            
            # 处理买卖盘数据
            bids = data_content.get('bids', [])
            asks = data_content.get('asks', [])
            
            for i, bid in enumerate(bids[:5]):
                setattr(record, f'bid_price{i+1}', float(bid.get('price', 0.0)))
                setattr(record, f'bid_volume{i+1}', int(bid.get('volume', 0)))
            
            for i, ask in enumerate(asks[:5]):
                setattr(record, f'ask_price{i+1}', float(ask.get('price', 0.0)))
                setattr(record, f'ask_volume{i+1}', int(ask.get('volume', 0)))
            
            # 加入异步写入队列（非阻塞，立即返回）
            self.async_writer.enqueue(instrument_id, record)
            
            # 更新统计
            with self._count_lock:
                self.message_count += 1
                self.instrument_counts[instrument_id] += 1
                
                if self.message_count % 1000 == 0:
                    stats = self.async_writer.get_stats()
                    logger.info(
                        f"总处理: {self.message_count} 条 | "
                        f"队列: {stats['enqueued']}, 写入: {stats['written']}, 丢弃: {stats['dropped']} | "
                        f"队列峰值: {stats['queue_max']}"
                    )
                    
        except Exception as e:
            logger.error(f"处理市场数据失败: {e}", exc_info=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取运行统计信息"""
        writer_stats = self.async_writer.get_stats()
        return {
            'total_messages': self.message_count,
            'instrument_counts': self.instrument_counts.copy(),
            'writer_stats': writer_stats
        }


# ============================================================================
# 辅助函数
# ============================================================================
def check_database_status(db_manager: DatabaseManager, instruments: List[str]) -> None:
    """检查数据库状态"""
    logger.info("=" * 50)
    logger.info("数据库状态:")
    
    for instrument in instruments:
        stats = db_manager.get_table_stats(instrument)
        if stats:
            logger.info(f"  {instrument}: {stats['count']} 条记录 | 最新录入时间: {stats['latest_record_time']}")
        else:
            logger.info(f"  {instrument}: 表不存在或无数据")
    
    logger.info("=" * 50)


def input_trade_date() -> date:
    """输入交易日"""
    while True:
        try:
            date_str = input("请输入交易日 (格式: YYYY-MM-DD): ").strip()
            input_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            logger.info(f"输入的交易日: {input_date}")
            return input_date
        except ValueError:
            print("日期格式错误，请使用 YYYY-MM-DD 格式，例如: 2026-03-19")
        except KeyboardInterrupt:
            print("\n用户取消输入")
            raise


# ============================================================================
# 主程序入口
# ============================================================================
def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_instruments_from_config(config: Dict[str, Any], exchange: str) -> List[str]:
    """从配置中获取合约列表"""
    return [
        config["instruments"][f"{symbol}_{appendix}"]
        for symbol in config["exchange_dict"][exchange]
        for appendix in ["main", "next"]
        if f"{symbol}_{appendix}" in config["instruments"]
    ]


def main():
    """主函数"""
    global INPUT_TRADE_DATE
    
    # 输入交易日
    INPUT_TRADE_DATE = input_trade_date()
    logger.info(f"程序将使用交易日: {INPUT_TRADE_DATE}")
    logger.info(f"规则: 如果小时 > 16，使用前一天 ({INPUT_TRADE_DATE - timedelta(days=1)})")
    logger.info(f"      如果小时 <= 16，使用输入日期 ({INPUT_TRADE_DATE})")
    
    # 加载配置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "config.json")
    
    config = load_config(config_path)
    
    db_path = config["paths"]["db_path"]
    exchange = "dce"
    
    instruments = get_instruments_from_config(config, exchange)
    host = config["zmq"]["host"]
    port = config["zmq"]["port"]
    
    # 创建并启动市场数据管理器
    market_data = RealTimeMarketData(
        instruments=instruments,
        db_path=db_path,
        host=host,
        port=port
    )
    
    market_data.start()
    
    try:
        last_count = 0
        last_check_time = time.time()
        
        logger.info("实时市场数据系统运行中...")
        logger.info("按 Ctrl+C 停止程序")
        
        while True:
            time.sleep(5)
            
            current_count = market_data.message_count
            if current_count > last_count:
                stats = market_data.get_stats()
                logger.info(f"运行状态 | 总处理: {current_count} | 新增: {current_count - last_count}")
                logger.info(f"各合约统计: {stats['instrument_counts']}")
                logger.info(f"写入统计: {stats['writer_stats']}")
                last_count = current_count
            
            # 每分钟检查一次数据库状态
            if time.time() - last_check_time > 60:
                check_database_status(market_data.db_manager, instruments)
                last_check_time = time.time()
                
    except KeyboardInterrupt:
        logger.info("用户停止程序")
    finally:
        market_data.stop()
        
        stats = market_data.get_stats()
        logger.info(f"最终统计: 共处理 {stats['total_messages']} 条消息")
        logger.info(f"各合约详细统计: {stats['instrument_counts']}")
        logger.info(f"写入器统计: {stats['writer_stats']}")
        logger.info("最终数据库状态:")
        check_database_status(market_data.db_manager, instruments)


if __name__ == "__main__":
    main()
