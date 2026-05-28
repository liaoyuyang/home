"""
因子核对脚本 - 诊断盘中 vs 历史因子不匹配问题

问题定位思路：
1. 对比盘中与历史的 valid_index
2. 检查 tick 数据边界
3. 单因子手动验证
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from datetime import datetime, time
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/strategy_PAMY_dev')
from data_function import Factor_generator, load_config
from strategies import load_fac_df_old

# ==================== 配置区 ====================
symbol = "A"  # 品种
rt_prefix = "/home/strategy_PAMY_dev/save_files"  # 实时环境因子目录
research_prefix = "/mnt/Data/writable/liaoyuyang/factor"  # 研究环境因子目录
research_date = None  # None=自动匹配，指定如 "2026-03-24"

TAIL_N = 10
TOL_ABS = 0.01
TOL_REL = 0.05
# =================================================

def load_research_factors(symbol, research_prefix, research_date=None):
    """从研究环境加载因子数据"""
    research_path = Path(research_prefix) / symbol / "all_fac" / "all_factor.feather"
    if not research_path.exists():
        print(f"❌ 研究环境因子文件不存在: {research_path}")
        return None
    
    meta = pd.read_feather(research_path, columns=['datetime', 'instrument'])
    meta['datetime'] = pd.to_datetime(meta['datetime'])
    meta['date'] = meta['datetime'].dt.date
    meta['time'] = meta['datetime'].dt.time
    
    all_dates = sorted(meta['date'].unique())
    
    if research_date:
        target_date = pd.to_datetime(research_date).date()
    else:
        target_date = all_dates[-1]
    
    fac_all = pd.read_feather(research_path)
    fac_all['datetime'] = pd.to_datetime(fac_all['datetime'])
    fac = fac_all[
        (fac_all['datetime'].dt.date == target_date)
    ].copy()
    
    return fac.set_index('datetime')

def load_rt_factors(symbol, rt_prefix):
    """从实时环境加载因子数据"""
    rt_dir = Path(rt_prefix) / symbol
    rt_files = sorted((rt_dir / "factors").glob("factors_*.csv"))
    if not rt_files:
        print(f"❌ 实时因子文件不存在: {rt_dir}/factors/")
        return None
    
    rt_dfs = []
    for f in rt_files:
        rt_dfs.append(pd.read_csv(f, parse_dates=['datetime']))
    
    rt_df = pd.concat(rt_dfs, ignore_index=True).sort_values('datetime').reset_index(drop=True)
    return rt_df.set_index('datetime')

def compare_valid_index(tick_data_realtime, tick_data_history, min_data):
    """
    对比盘中与历史的 valid_index
    
    关键点：
    - 盘中模式：tick_data < 5000 条时，valid_index 只保留最后一个点
    - 历史模式：tick_data 完整，valid_index 包含所有交易分钟
    """
    print("\n" + "="*60)
    print("步骤1: 对比 valid_index 生成过程")
    print("="*60)
    
    # 模拟盘中模式（短 tick 数据）
    tick_short = tick_data_realtime.copy()
    tick_short_indexed = tick_short.set_index('datetime')
    time_index = (tick_short_indexed
                  .resample('1min', label='right', closed='right').last().index.time)
    
    is_trading = (
        ((time_index >= time(21, 0, 0, 1)) & (time_index <= time(23, 0, 0, 0))) |
        ((time_index >= time(9, 0, 0, 1)) & (time_index <= time(10, 15, 0, 0))) |
        ((time_index >= time(10, 30, 0, 1)) & (time_index <= time(11, 30, 0, 0))) |
        ((time_index >= time(13, 0, 0, 1)) & (time_index <= time(15, 0, 0, 0)))
    )
    
    valid_index_realtime = (tick_short_indexed
                            .resample('1min', label='right', closed='right').last()
                            .loc[is_trading].dropna(how='all').index)
    
    # 运行时优化：tick < 5000 时只保留最后一个点
    if len(tick_short) < 5000 and len(valid_index_realtime) > 1:
        valid_index_realtime = valid_index_realtime[-1:]
        print(f"⚠️ 盘中模式: tick数={len(tick_short)} < 5000, valid_index 截断为最后一个点")
    
    # 模拟历史模式（完整 tick 数据）
    tick_full = tick_data_history.copy()
    tick_full_indexed = tick_full.set_index('datetime')
    time_index_full = (tick_full_indexed
                       .resample('1min', label='right', closed='right').last().index.time)
    
    is_trading_full = (
        ((time_index_full >= time(21, 0, 0, 1)) & (time_index_full <= time(23, 0, 0, 0))) |
        ((time_index_full >= time(9, 0, 0, 1)) & (time_index_full <= time(10, 15, 0, 0))) |
        ((time_index_full >= time(10, 30, 0, 1)) & (time_index_full <= time(11, 30, 0, 0))) |
        ((time_index_full >= time(13, 0, 0, 1)) & (time_index_full <= time(15, 0, 0, 0)))
    )
    
    valid_index_history = (tick_full_indexed
                           .resample('1min', label='right', closed='right').last()
                           .loc[is_trading_full].dropna(how='all').index)
    
    print(f"\n盘中模式 valid_index: {len(valid_index_realtime)} 个点")
    print(f"历史模式 valid_index: {len(valid_index_history)} 个点")
    
    if len(valid_index_realtime) == 1:
        print(f"\n🔴 问题确认: 盘中模式只保留最后1个点!")
        print(f"   盘中最后时间: {valid_index_realtime[-1]}")
        print(f"   历史最后时间: {valid_index_history[-1] if len(valid_index_history) > 0 else 'N/A'}")
    
    return valid_index_realtime, valid_index_history

def check_tick_boundary(tick_data, min_data):
    """
    检查 tick 数据边界
    
    问题点：
    - resample('1min', label='right', closed='right') 使用 tick 数据
    - 但因子计算依赖 min_data 的 OHLCV
    - 如果 tick 数据不完整，会导致 resample 结果不一致
    """
    print("\n" + "="*60)
    print("步骤2: 检查 tick 数据边界")
    print("="*60)
    
    tick_idx = tick_data['datetime']
    min_idx = min_data['datetime']
    
    print(f"tick 数据: {len(tick_data)} 条")
    print(f"  时间范围: {tick_idx.min()} ~ {tick_idx.max()}")
    print(f"min 数据: {len(min_data)} 条")
    print(f"  时间范围: {min_idx.min()} ~ {min_idx.max()}")
    
    # 检查 tick 数据最后一条的时间
    last_tick_time = tick_idx.max()
    last_min_time = min_idx.max()
    
    # 计算最后一条 tick 到整分钟的差距
    tick_minute_end = pd.Timestamp(f"{last_tick_time.date()} {last_tick_time.hour}:{last_tick_time.minute}:00")
    if last_tick_time.second > 0 or last_tick_time.microsecond > 0:
        tick_minute_end += pd.Timedelta(minutes=1)
    
    gap_seconds = (tick_minute_end - last_tick_time).total_seconds()
    print(f"\n最后 tick 时间: {last_tick_time}")
    print(f"距下一分钟: {gap_seconds:.1f} 秒")
    
    if gap_seconds < 30:
        print("⚠️ 最后 tick 距离下一分钟不足 30 秒，可能导致 resample 结果不完整")
    
    return last_tick_time, last_min_time

def test_single_factor(fac_gen_realtime, fac_gen_history, factor_name="SPREAD"):
    """
    单因子手动验证
    
    选择一个简单因子，手动计算并对比
    """
    print("\n" + "="*60)
    print(f"步骤3: 单因子验证 - {factor_name}")
    print("="*60)
    
    # 盘中模式因子计算
    try:
        fac_realtime = fac_gen_realtime.SPREAD() if hasattr(fac_gen_realtime, 'SPREAD') else None
        print(f"盘中因子计算成功: {len(fac_realtime)} 个值" if fac_realtime is not None else "盘中因子计算失败")
    except Exception as e:
        fac_realtime = None
        print(f"盘中因子计算异常: {e}")
    
    # 历史模式因子计算
    try:
        fac_history = fac_gen_history.SPREAD() if hasattr(fac_gen_history, 'SPREAD') else None
        print(f"历史因子计算成功: {len(fac_history)} 个值" if fac_history is not None else "历史因子计算失败")
    except Exception as e:
        fac_history = None
        print(f"历史因子计算异常: {e}")
    
    if fac_realtime is not None and fac_history is not None:
        # 对齐到相同时间点
        rt_idx = fac_gen_realtime.valid_index
        hist_idx = fac_gen_history.valid_index
        common_idx = rt_idx.intersection(hist_idx)
        
        if len(common_idx) > 0:
            rt_vals = pd.Series(fac_realtime, index=rt_idx).loc[common_idx]
            hist_vals = pd.Series(fac_history, index=hist_idx).loc[common_idx]
            
            diff = (rt_vals - hist_vals).abs()
            max_diff = diff.max()
            mean_diff = diff.mean()
            
            print(f"\n共有时间点: {len(common_idx)} 个")
            print(f"最大差异: {max_diff:.6f}")
            print(f"平均差异: {mean_diff:.6f}")
            
            if max_diff < TOL_ABS:
                print("✅ 因子一致性通过")
            else:
                print("❌ 因子一致性不通过")
        else:
            print("⚠️ 无共有时间点")
    
    return fac_realtime, fac_history

def main():
    print("="*60)
    print("因子核对工具 - 盘中 vs 历史")
    print("="*60)
    
    config = load_config('/home/strategy_PAMY_dev/config.json')
    main_inst = config['instruments'][symbol]['contract']
    
    # 尝试读取实时 tick 数据
    rt_dir = Path(rt_prefix) / symbol / "data"
    tick_files = sorted(rt_dir.glob("main_tick_*.csv"))
    
    if tick_files:
        tick_data = pd.read_csv(tick_files[-1], parse_dates=['datetime'])
        print(f"\n读取实时 tick 数据: {tick_files[-1].name}")
    else:
        print(f"\n❌ 未找到实时 tick 数据: {rt_dir}")
        # 创建模拟数据
        print("创建模拟 tick 数据...")
        n = 3000  # < 5000，模拟盘中模式
        times = pd.date_range('2026-03-24 21:00', periods=n, freq='1s')
        tick_data = pd.DataFrame({
            'datetime': times,
            'last_price': np.random.uniform(5000, 5100, n),
            'volume': np.random.randint(1, 100, n),
            'turnover': np.random.uniform(50000, 60000, n),
            'open_interest': np.random.randint(1000, 5000, n),
            'bid_price1': np.random.uniform(4990, 5000, n),
            'ask_price1': np.random.uniform(5000, 5010, n),
            'bid_volume1': np.random.randint(10, 100, n),
            'ask_volume1': np.random.randint(10, 100, n),
        })
    
    # 读取 min 数据
    min_files = sorted(rt_dir.glob("main_min_*.csv"))
    if min_files:
        min_data = pd.read_csv(min_files[-1], parse_dates=['datetime'])
        print(f"读取实时 min 数据: {min_files[-1].name}")
    else:
        # 创建模拟 min 数据
        print("创建模拟 min 数据...")
        n = 100
        times = pd.date_range('2026-03-24 21:00', periods=n, freq='1min')
        min_data = pd.DataFrame({
            'datetime': times,
            'open': np.random.uniform(5000, 5100, n),
            'high': np.random.uniform(5000, 5100, n),
            'low': np.random.uniform(5000, 5100, n),
            'close': np.random.uniform(5000, 5100, n),
            'volume': np.random.randint(100, 1000, n),
            'turnover': np.random.uniform(500000, 600000, n),
            'open_interest': np.random.randint(1000, 5000, n),
        })
    
    # 检查 tick 数据边界
    last_tick, last_min = check_tick_boundary(tick_data, min_data)
    
    # 对比 valid_index
    valid_index_realtime, valid_index_history = compare_valid_index(tick_data, tick_data, min_data)
    
    # 创建 Factor_generator 实例
    print("\n" + "="*60)
    print("步骤4: 创建 Factor_generator 实例")
    print("="*60)
    
    # 盘中模式（短 tick）
    fac_gen_realtime = Factor_generator(tick_data, min_data)
    print(f"盘中模式 Factor_generator: valid_index={len(fac_gen_realtime.valid_index)} 个点")
    print(f"  tick_data 长度: {len(fac_gen_realtime.tick_data)}")
    
    # 历史模式（完整 tick，需要更多数据模拟）
    tick_full = pd.concat([tick_data] * 10)  # 复制10倍模拟完整数据
    tick_full = tick_full.reset_index(drop=True)
    fac_gen_history = Factor_generator(tick_full, min_data)
    print(f"\n历史模式 Factor_generator: valid_index={len(fac_gen_history.valid_index)} 个点")
    print(f"  tick_data 长度: {len(fac_gen_history.tick_data)}")
    
    # 单因子验证
    fac_realtime, fac_history = test_single_factor(fac_gen_realtime, fac_gen_history, "SPREAD")
    
    # 汇总
    print("\n" + "="*60)
    print("诊断结论")
    print("="*60)
    
    print("""
可能的问题原因及排查建议：

1. 【valid_index 截断】
   代码位置: data_function.py 第 733-734 行
   问题: if len(self.tick_data) < 5000 and len(self.valid_index) > 1:
             self.valid_index = self.valid_index[-1:]
   说明: 盘中模式 tick < 5000 时，valid_index 只保留最后一个点
   对比: 研究环境加载完整历史数据，valid_index 包含所有分钟

2. 【tick 数据边界】
   问题: resample('1min', label='right', closed='right') 使用 tick 最后时间
   对比: 盘中 tick 可能不包含完整的最后一分钟数据
   排查: 检查最后 tick 时间距离整分钟是否 < 30 秒

3. 【因子 reindex 对齐】
   问题: 每个因子最后都会 .reindex(index=self.valid_index)
   对比: 盘中 reindex 到 1 个点，历史 reindex 到所有点
   结果: 盘中因子只有 1 个非 NaN 值，历史有完整序列

4. 【跨品种因子窗口】
   问题: 涉及多品种对齐时 reindex 可能引入 NaN
   排查: 检查 _symbol_data_indexed 的时间对齐

建议下一步：
- 修改 data_function.py 第 733-734 行，添加日志打印截断前后 valid_index
- 在 strategies.py on_new_minute() 中打印传入的 tick 数据长度
- 对比同一时间点盘中与历史计算的中间变量
""")

if __name__ == "__main__":
    main()
