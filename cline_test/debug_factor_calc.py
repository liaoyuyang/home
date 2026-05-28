"""
单因子调试脚本 - 手动验证因子计算是否正确
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, '/home/strategy_PAMY_dev')

# 1. 读取实时 min_data
rt_dir = Path("/home/strategy_PAMY_dev/save_files/A/data")
rt_files = sorted(rt_dir.glob("main_min_*.csv"))
print(f"找到 {len(rt_files)} 个 min_data 文件")

# 读取最近的数据（假设这是回放 3月24日的数据）
rt_mkt = pd.read_csv(rt_files[-1], parse_dates=['datetime'])
print(f"\n实时行情: {len(rt_mkt)} 行")
print(f"时间范围: {rt_mkt['datetime'].iloc[0]} ~ {rt_mkt['datetime'].iloc[-1]}")
print(f"品种: {rt_mkt['instrument'].iloc[0]}")

# 2. 初始化 Factor_generator
from data_function import Factor_generator
fac_gen = Factor_generator(tick_data=rt_mkt, min_data=rt_mkt)
print(f"\nvalid_index: {len(fac_gen.valid_index)} 条")
print(f"首尾时间: {fac_gen.valid_index[0]} ~ {fac_gen.valid_index[-1]}")

# 3. 计算几个简单因子并打印中间过程
print("\n" + "="*60)
print("因子计算验证")
print("="*60)

# 用最后5个 valid_index 计算因子
test_times = fac_gen.valid_index[-5:]
for t in test_times:
    print(f"\n--- 时间点: {t} ---")
    
    # 计算 LR (线性回归斜率)
    lr = fac_gen.LR()
    lr_last = lr[-1] if hasattr(lr, '__len__') else lr
    print(f"LR = {lr_last:.6f}")
    
    # 计算 RPP_22D
    rpp = fac_gen.RPP_22D()
    rpp_last = rpp[-1] if hasattr(rpp, '__len__') else rpp
    print(f"RPP_22D = {rpp_last:.6f}")
    
    # 计算 bar5_trend_corr
    bar5 = fac_gen.bar5_trend_corr()
    bar5_last = bar5[-1] if hasattr(bar5, '__len__') else bar5
    print(f"bar5_trend_corr = {bar5_last:.6f}")

# 4. 对比盘中因子文件
print("\n" + "="*60)
print("对比盘中因子文件")
print("="*60)

rt_fac_dir = Path("/home/strategy_PAMY_dev/save_files/A/factors")
rt_fac_files = sorted(rt_fac_dir.glob("factors_*.csv"))
rt_fac_df = pd.read_csv(rt_fac_files[-1], parse_dates=['datetime']).set_index('datetime')

fac_time = rt_fac_df.index[-1]
print(f"\n盘中因子时间: {fac_time}")
print(f"盘中因子 LR = {rt_fac_df.loc[fac_time, 'LR']:.6f}")
print(f"盘中因子 RPP_22D = {rt_fac_df.loc[fac_time, 'RPP_22D']:.6f}")
print(f"盘中因子 bar5_trend_corr = {rt_fac_df.loc[fac_time, 'bar5_trend_corr']:.6f}")

# 5. 检查数据完整性
print("\n" + "="*60)
print("数据完整性检查")
print("="*60)

# 检查当前 tick_data 的状态
print(f"\n当前 tick_data 行数: {len(fac_gen.tick_data)}")
print(f"当前 valid_index 行数: {len(fac_gen.valid_index)}")

# 打印最后几行 tick_data
print("\ntick_data 最后 3 行:")
print(fac_gen.tick_data.tail(3))

# 打印最后几个 valid_index
print("\nvalid_index 最后 3 个:")
for idx in fac_gen.valid_index[-3:]:
    print(f"  {idx}")
