import zmq
import struct


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

def calculate_size(fmt):
    # 移除格式字符串中的空白字符
    fmt_clean = ''.join(fmt.split())
    return struct.calcsize(fmt_clean)

# 计算结构体大小（用于验证）
struct_size = calculate_size(fmt)
print(f"结构体总大小: {struct_size} 字节")

# 解析函数示例
def parse_my_shm_feed_item(data):
    fmt_clean = ''.join(fmt.split())
    if len(data) < struct_size:
        raise ValueError(f"数据长度不足，需要至少{struct_size}字节，实际{len(data)}字节")
    
    # 从偏移量0开始解析（如果需要跳过前8字节，可将第三个参数改为8）
    fields = struct.unpack_from(fmt_clean, data, 0)
    
    # 映射字段到字典（按顺序对应结构体成员）
    result = {
        "head": {"status": fields[0]},
        "type": fields[1],
        "time": fields[2],
        "feedsource_name": fields[3].decode('utf-8').rstrip('\x00'),
        "data": {
            "instrument_id": fields[4].decode('utf-8').rstrip('\x00'),
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
            "fqr_id": fields[50].decode('utf-8').rstrip('\x00')
        }
    }
    return result
            

# 设置ZMQ
context = zmq.Context()
socket = context.socket(zmq.SUB)
# socket.connect("tcp://192.168.2.238:77718")
socket.connect("tcp://172.17.0.6:7779")
socket.setsockopt_string(zmq.SUBSCRIBE, "")

print("ZMQ连接成功，等待接收数据...")

# 主循环解析数据
while True:
        tt = socket.recv()
        if tt != b'feed':
            continue
        data = socket.recv()
        print(f"接收到数据，长度: {len(data)} 字节")
        
        fields = parse_my_shm_feed_item(data)
        print(fields)