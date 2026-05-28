"""
测试 DataService 的 trigger 逻辑
直接读取 DB，模拟轮询，打印 trigger 判定过程
"""
import sys
import time
import pandas as pd

sys.path.insert(0, '/home/strategy_PAMY_dev')
from data_function import load_config, read_table

config = load_config('/home/strategy_PAMY_dev/config.json')
db_path = config['paths']['db_path']
trade_hours = config['symbol_specs']['P']['trade_hours']

inst = 'a2605'

# 模拟 DataService 的初始状态
time_before = pd.Timestamp('2026-01-25 20:59:00.000000')
tick_cache = None

print(f"{'='*60}")
print("[TEST] 开始测试 trigger 逻辑")
print(f"{'='*60}")

for i in range(20):
    df = read_table(inst, db_path=db_path, word=False, trade_type=trade_hours, limit=50000)
    
    if df is None or df.empty:
        print(f"#{i:02d} | read_table 返回空，跳过")
        time.sleep(1)
        continue
    
    tick_cache = df
    time_recently = pd.to_datetime(df['datetime'].iloc[-1])
    
    trigger = (
        time_recently.minute != time_before.minute
        and not (time_before.hour == 20 and time_before.minute == 59)
        and not (time_before.hour == 8 and time_before.minute == 59)
        and not (time_recently.hour == 8 and time_recently.minute == 59)
    )
    
    status = "🚨 TRIGGER" if trigger else "-"
    print(f"#{i:02d} | DB最新: {time_recently} | 前次基准: {time_before:%H:%M:%S} | tick数: {len(df):>5} | {status}")
    
    if trigger:
        print(f"       >>> 新分钟触发: {time_recently.replace(second=0, microsecond=0)}")
        time_before = time_recently.replace(second=0, microsecond=0)
    else:
        time_before = time_recently
    
    time.sleep(1)

print(f"{'='*60}")
print("[TEST] 测试结束")
print(f"{'='*60}")
