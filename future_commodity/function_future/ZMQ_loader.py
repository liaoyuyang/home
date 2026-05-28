import zmq
import struct
from typing import Any, Dict, List, Callable, Optional
from datetime import datetime, timedelta, timezone
from multiprocessing import Manager
import pandas as pd
import time
import numpy as np

class ZmqMarketDataSubscriber:
    """
    ZMQ 市场数据订阅器，支持按 instrument_id 筛选数据
    """
    
    def __init__(self, host: str = "192.168.2.239", port: int = 7778):
        """
        初始化订阅器
        
        Args:
            host: ZMQ 服务器地址
            port: ZMQ 服务器端口
        """
        self.host = host
        self.port = port
        self.subscribed_instruments = set()  # 订阅的合约集合
        self.is_running = False
        
        # 初始化 ZMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        
    def subscribe(self, instruments: List[str]) -> None:
        """
        订阅指定的合约列表
        
        Args:
            instruments: 合约代码列表，如 ['IC2512', 'IF2512']
        """
        self.subscribed_instruments.update(instruments)
        print(f"已订阅合约: {list(self.subscribed_instruments)}")
    
    def unsubscribe(self, instruments: List[str]) -> None:
        """
        取消订阅指定的合约
        
        Args:
            instruments: 要取消订阅的合约代码列表
        """
        for instrument in instruments:
            if instrument in self.subscribed_instruments:
                self.subscribed_instruments.remove(instrument)
        print(f"当前订阅合约: {list(self.subscribed_instruments)}")
    
    def early_filter_by_instrument(self, data: bytes, instrument_prefix: bytes) -> bool:
        """
        早期过滤：在解析前检查是否为指定前缀的合约
        
        Args:
            data: 二进制数据
            instrument_prefix: 合约前缀，如 b'TL2606'
            
        Returns:
            bool: 是否匹配前缀
        """
        # 在数据中查找 instrument_id 字段的位置
        # 根据您的数据结构，instrument_id 在数据中的位置
        # 假设 instrument_id 在偏移量 40 字节处开始
        offset = 40
        
        if len(data) >= offset + len(instrument_prefix):
            instrument_start = data[offset:offset+len(instrument_prefix)]
            return instrument_start == instrument_prefix
        return False

    def parse_market_data(self, data: bytes) -> Dict[str, Any]:
        """
        解析市场数据 - 使用您提供的正确格式
        """
        try:
            # 使用您提供的格式字符串
            fmt = (
                "=ixxxx"  # item_head head（4字节int + 4字节填充）
                "qq"      # long type（8字节） + long time（8字节）
                "16s"     # char feedsource_name[16]（16字节）
                "64s"     # instrument_id[64]（64字节）
                "qq"      # long fqr_type + long product_class（各8字节）
                "d"       # double last_price（8字节）
                "ixxxx"   # int last_volume（4字节） + 4字节填充
                "dddddd"  # pre_settlement_price ~ lowest_price（6个double，各8字节）
                "ixxxx"   # int volume（4字节） + 4字节填充
                "dddddd"  # turnover ~ lower_limit_price（6个double，各8字节）
                "q"       # long update_time（8字节）
                "ii"      # int tot_buy_num + int tot_sell_num（共8字节，无需填充）
                "ddd"     # tot_buy_avg_w ~ theoretical_open_price（3个double，各8字节）
                "ixxxx"   # int level（4字节） + 4字节填充
                # 五档买卖盘（每档：double(8) + int(4)+填充(4) + double(8) + int(4)+填充(4)）
                "dixxxxdixxxx"  # 第1档
                "dixxxxdixxxx"  # 第2档
                "dixxxxdixxxx"  # 第3档
                "dixxxxdixxxx"  # 第4档
                "dixxxxdi"  # 第5档
                "i"   # int fqr_time（4字节） + 4字节填充
                "40s"     # char fqr_id[40]（40字节）
            )
            
            # 移除格式字符串中的空白字符
            fmt_clean = ''.join(fmt.split())
            
            # 计算结构体大小
            struct_size = struct.calcsize(fmt_clean)
            
            if len(data) < struct_size:
                raise ValueError(f"数据长度不足，需要至少{struct_size}字节，实际{len(data)}字节")
            
            # 解析数据
            fields = struct.unpack_from(fmt_clean, data, 0)
            
            # 安全解码字符串字段
            def safe_decode(byte_str, field_name):
                try:
                    return byte_str.decode('utf-8').rstrip('\x00')
                except UnicodeDecodeError:
                    try:
                        # 尝试使用latin1编码作为备选
                        return byte_str.decode('latin1').rstrip('\x00')
                    except:
                        print(f"⚠️ 无法解码 {field_name} 字段，使用原始字节")
                        return str(byte_str)
            
            # 映射字段到字典
            result = {
                "head": {"status": fields[0]},
                "type": fields[1],
                "time": fields[2],
                "feedsource_name": safe_decode(fields[3], "feedsource_name"),
                "data": {
                    "instrument_id": safe_decode(fields[4], "instrument_id"),
                    "fqr_type": fields[5],
                    "product_class": fields[6],
                    "last_price": fields[7],
                    "last_volume": fields[8],
                    "pre_settlement_price": fields[9],
                    "pre_close_price": fields[10],
                    "pre_open_interest": fields[11],
                    "open_price": fields[12],
                    "highest_price": fields[13],
                    "lowest_price": fields[14],
                    "volume": fields[15],
                    "turnover": fields[16],
                    "open_interest": fields[17],
                    "close_price": fields[18],
                    "settlement_price": fields[19],
                    "upper_limit_price": fields[20],
                    "lower_limit_price": fields[21],
                    "update_time": fields[22],
                    "tot_buy_num": fields[23],
                    "tot_sell_num": fields[24],
                    "tot_buy_avg_w": fields[25],
                    "tot_sell_avg_w": fields[26],
                    "theoretical_open_price": fields[27],
                    "level": fields[28],
                    # 五档买卖盘
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
                    ],
                    "fqr_time": fields[49],
                    "fqr_id": safe_decode(fields[50], "fqr_id")
                }
            }
            return result
            
        except Exception as e:
            print(f"❌ 数据解析失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                "head": {"status": -1},
                "data": {
                    "instrument_id": "ERROR",
                    "last_price": 0.0,
                    "bids": [{"price": 0.0, "volume": 0} for _ in range(5)],
                    "asks": [{"price": 0.0, "volume": 0} for _ in range(5)]
                }
            }

    def start(self, 
              callback: Optional[Callable] = None, 
              early_filter: bool = True) -> None:
        """
        开始接收数据
        
        Args:
            callback: 数据回调函数，接收解析后的数据字典
            early_filter: 是否启用早期过滤（性能优化）
        """
        # 连接 ZMQ 服务器
        connection_str = f"tcp://{self.host}:{self.port}"
        self.socket.connect(connection_str)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")  # 订阅所有消息
        
        print(f"ZMQ连接成功: {connection_str}")
        print("开始接收数据..." + ("（启用早期过滤）" if early_filter else ""))
        
        self.is_running = True
        
        try:
            while self.is_running:
                # 接收消息标识
                tt = self.socket.recv()
                if tt != b'feed':
                    continue
                
                # 接收实际数据
                data = self.socket.recv()
                
                # 早期过滤：如果不是目标合约，直接跳过
                if early_filter and self.subscribed_instruments:
                    # 检查数据是否以订阅的任意合约前缀开头
                    matched = any(
                        self.early_filter_by_instrument(data, instrument.encode('utf-8'))
                        for instrument in self.subscribed_instruments
                    )
                    if not matched:
                        continue
                
                try:
                    market_data = self.parse_market_data(data)
                    
                    if callback:
                        callback(market_data)
                    else:
                        # 打印详细的买卖盘信息
                        bids = market_data['data']['bids']
                        asks = market_data['data']['asks']
                        print(f"📊 最新价: {market_data['data']['last_price']}")
                        print(f"📈 买一: {bids[0]['price']}@{bids[0]['volume']} | 卖一: {asks[0]['price']}@{asks[0]['volume']}")
                        print(f"🕒 时间: {market_data['time']}")
                        print("-" * 50)
                        
                except Exception as e:
                    print(f"数据解析错误: {e}")
                    continue
                    
        except KeyboardInterrupt:
            print("用户中断接收")
        except Exception as e:
            print(f"接收数据错误: {e}")
        finally:
            self.stop()
    
    def stop(self) -> None:
        """停止接收数据"""
        self.is_running = False
        self.socket.close()
        self.context.term()
        print("ZMQ连接已关闭")

class RealtimeDataFrame:
    _instance = None
    
    def __new__(cls, window_minutes=2):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            manager = Manager()
            cls._instance.shared = manager.Namespace()
            cls._instance.shared.df = manager.list()  # 共享列表存储数据
            cls._instance.shared.columns = [
                # 核心数据区
                'instrument_id',      # 合约代码
                'last_price',         # 最新价
                'open_price',          # 开盘价
                'highest_price',       # 最高价
                'lowest_price',        # 最低价
                'volume',              # 成交量
                'turnover',            # 成交额
                'open_interest',       # 持仓量
                'update_time',         # 更新时间戳
                
                # 买卖盘深度
                'bid1_price', 'bid1_volume',   
                'bid2_price', 'bid2_volume',   
                'bid3_price', 'bid3_volume',   
                'bid4_price', 'bid4_volume',   
                'bid5_price', 'bid5_volume',   
                
                'ask1_price', 'ask1_volume',   
                'ask2_price', 'ask2_volume',   
                'ask3_price', 'ask3_volume',   
                'ask4_price', 'ask4_volume',   
                'ask5_price', 'ask5_volume',
                
                # 时间字段
                'datetime_str'         # 日期时间字符串
            ]
            cls._instance.window_minutes = window_minutes
            cls._instance.window_seconds = window_minutes * 60
            cls._instance.lock = manager.Lock()
            print(f"✅ RealtimeDataFrame 初始化完成 | 时间窗口: {window_minutes}分钟")
        return cls._instance

    def add_data(self, data: dict):
        """线程安全的共享数据添加 - 修复版本"""
        with self.lock:
            try:
                # 提取数据内容
                data_content = data.get('data', {})
                
                # 构建新记录
                new_record = {}
                
                # 核心数据
                new_record['instrument_id'] = data_content.get('instrument_id', 'UNKNOWN')
                new_record['last_price'] = float(data_content.get('last_price', 0.0))
                new_record['open_price'] = float(data_content.get('open_price', 0.0))
                new_record['highest_price'] = float(data_content.get('highest_price', 0.0))
                new_record['lowest_price'] = float(data_content.get('lowest_price', 0.0))
                new_record['volume'] = int(data_content.get('volume', 0))
                new_record['turnover'] = float(data_content.get('turnover', 0.0))
                new_record['open_interest'] = float(data_content.get('open_interest', 0.0))
                
                # 时间处理
                raw_update_time = data_content.get('update_time', 0)
                new_record['update_time'] = self.parse_time(raw_update_time)
                
                # 买卖盘深度
                bids = data_content.get('bids', [{} for _ in range(5)])
                asks = data_content.get('asks', [{} for _ in range(5)])
                
                for i in range(5):
                    bid = bids[i] if i < len(bids) else {}
                    new_record[f'bid{i+1}_price'] = float(bid.get('price', 0.0))
                    new_record[f'bid{i+1}_volume'] = int(bid.get('volume', 0))
                
                for i in range(5):
                    ask = asks[i] if i < len(asks) else {}
                    new_record[f'ask{i+1}_price'] = float(ask.get('price', 0.0))
                    new_record[f'ask{i+1}_volume'] = int(ask.get('volume', 0))
                
                # 添加日期时间字符串
                new_record['datetime_str'] = datetime.fromtimestamp(
                    new_record['update_time']
                ).strftime('%Y-%m-%d %H:%M:%S.%f') if new_record['update_time'] > 0 else ''
                
                # 添加到共享列表
                self.shared.df.append(new_record)
                
                # 自动清理过期数据
                self._cleanup_old_data()
                
                print(f"✅ 数据添加成功: {new_record['instrument_id']} | 价格: {new_record['last_price']} | 时间: {new_record['datetime_str']}")
                
            except Exception as e:
                print(f"❌ 数据添加失败: {str(e)}")

    def _cleanup_old_data(self):
        """清理超过时间窗口的旧数据 - 修复版本"""
        try:
            current_time = time.time()
            threshold_time = current_time - self.window_seconds
            
            # 创建新列表，只保留在时间窗口内的数据
            valid_records = []
            for record in self.shared.df:
                record_time = record.get('update_time', 0)
                if record_time >= threshold_time:
                    valid_records.append(record)
            
            # 更新共享列表
            self.shared.df[:] = valid_records
            
            cleaned_count = len(self.shared.df) - len(valid_records)
            if cleaned_count > 0:
                print(f"🧹 清理了 {cleaned_count} 条旧数据 | 剩余 {len(valid_records)} 条有效数据")
                
        except Exception as e:
            print(f"❌ 数据清理失败: {e}")

    def get_all_data(self):
        """返回所有合约的完整数据（按datetime和instrument_id排序）"""
        with self.lock:
            try:
                if not self.shared.df:
                    return pd.DataFrame(columns=self.shared.columns)
                
                # 转换为DataFrame
                df = pd.DataFrame(list(self.shared.df))
                
                if df.empty:
                    return df
                
                # 确保所有列都存在
                for col in self.shared.columns:
                    if col not in df.columns:
                        df[col] = None
                
                # 创建datetime列用于排序
                df['datetime'] = pd.to_datetime(df['datetime_str'], errors='coerce')
                df = df.dropna(subset=['datetime'])
                
                if df.empty:
                    return pd.DataFrame(columns=self.shared.columns)
                
                # 按datetime和instrument_id排序
                df = df.sort_values(['datetime', 'instrument_id']).reset_index(drop=True)
                
                print(f"📊 获取所有数据: {len(df)} 条记录 | 合约: {df['instrument_id'].unique().tolist()}")
                return df
                
            except Exception as e:
                print(f"❌ 获取所有数据失败: {e}")
                return pd.DataFrame(columns=self.shared.columns)

    def get_recent_data(self, instrument=None, n=200):
        """获取最近n条数据（线程安全） - 支持按合约筛选"""
        with self.lock:
            try:
                # 先获取所有数据
                df = self.get_all_data()
                
                if df.empty:
                    return df
                
                # 按合约筛选
                if instrument:
                    df = df[df['instrument_id'] == instrument]
                
                # 返回最近n条（按时间倒序）
                result = df.sort_values('datetime', ascending=False).head(n)
                result = result.sort_values('datetime', ascending=True)  # 最终按时间正序返回
                
                print(f"📈 获取最近数据: {len(result)} 条 | 合约: {instrument or 'ALL'}")
                return result
                
            except Exception as e:
                print(f"❌ 获取最近数据失败: {e}")
                return pd.DataFrame(columns=self.shared.columns)

    def get_instruments(self):
        """返回当前数据中所有的合约代码"""
        with self.lock:
            try:
                if not self.shared.df:
                    return []
                
                instruments = list(set(record['instrument_id'] for record in self.shared.df))
                return sorted(instruments)
                
            except Exception as e:
                print(f"❌ 获取合约列表失败: {e}")
                return []

    def get_data_stats(self):
        """返回数据统计信息"""
        with self.lock:
            try:
                df = self.get_all_data()
                if df.empty:
                    return {"total_records": 0, "instruments": [], "time_range": "无数据"}
                
                stats = {
                    "total_records": len(df),
                    "instruments": df['instrument_id'].unique().tolist(),
                    "time_range": f"{df['datetime'].min()} 至 {df['datetime'].max()}",
                    "records_per_instrument": df.groupby('instrument_id').size().to_dict()
                }
                return stats
                
            except Exception as e:
                print(f"❌ 获取统计信息失败: {e}")
                return {}

    @staticmethod
    def parse_time(time_str):
        """时间解析方法（增强版）"""
        try:
            # 如果是 Unix 时间戳（通常 >= 1e9），直接返回
            if isinstance(time_str, (int, float)) and time_str >= 1e9:
                return float(time_str)
            
            # 处理数字格式的时间（如 101827400 → HHMMSSmmm）
            time_str = str(int(time_str)).zfill(9)  # 确保是 9 位数字
            if len(time_str) >= 6:
                hour = int(time_str[:2])
                minute = int(time_str[2:4])
                second = int(time_str[4:6])
                millisecond = int(time_str[6:9]) if len(time_str) > 6 else 0
                
                if 0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59:
                    # 转换为当天的时间戳（秒数）
                    today = datetime.now().date()
                    dt = datetime(
                        today.year, today.month, today.day,
                        hour, minute, second, millisecond * 1000
                    )
                    return dt.timestamp()
            
            return pd.NaT
        except:
            return pd.NaT
