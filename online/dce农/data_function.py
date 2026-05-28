# 标准库
import functools
import json
import os
import sqlite3
import struct
import sys
import time as time_module
from datetime import date, datetime, time, timedelta
from multiprocessing import shared_memory
from typing import Any, Dict, List, Optional

# 第三方库
import chinese_calendar as calendar
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zmq
from joblib import Parallel, delayed
from numba import njit

##-------------------------------------------------------------------------------------------------------------------------##
def load_config(config_path=None):
    """加载配置文件"""
    if config_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config.json")
    with open(config_path) as f:
        return json.load(f)

def get_strategy_from_filename():
    """
    从当前运行的文件名提取策略名称
    例如: main_PAMY_dce.py -> PAMY_dce
          main_MA_shfe.py -> MA_shfe
    """
    # 获取当前执行的文件路径
    import sys
    main_file = sys.argv[0] if sys.argv else __file__
    
    # 提取文件名（不含路径和扩展名）
    base_name = os.path.basename(main_file)
    file_name, _ = os.path.splitext(base_name)
    return file_name

def get_symbol_from_instrument(instrument):
    """从合约代码提取品种代码"""
    # 提取字母部分（如 p2605 -> P）
    symbol = ''.join([c for c in instrument if c.isalpha()]).upper()
    return symbol

def parse_time_str(time_str):
    """将时间字符串解析为time对象"""
    parts = time_str.split(':')
    return time(int(parts[0]), int(parts[1]))

def is_in_no_trade_period(current_time, time_config):
    """
    检查当前时间是否在不交易时间段内
    
    参数:
        current_time: datetime.time 对象
        time_config: 时间配置字典
    
    返回:
        bool: 如果在不交易时间段内返回True
    """
    day_session = time_config['day_session']
    night_session = time_config['night_session']
    
    # 白天盘不交易时间段1: 09:00-09:10
    no_trade_start1 = parse_time_str(day_session['no_trade_start'])
    no_trade_end1 = parse_time_str(day_session['no_trade_end'])
    
    # 白天盘不交易时间段2: 14:50-15:00
    no_trade_start2 = parse_time_str(day_session['no_trade_start2'])
    no_trade_end2 = parse_time_str(day_session['no_trade_end2'])
    
    # 检查是否在白天盘的不交易时间段
    if (no_trade_start1 <= current_time <= no_trade_end1) or \
       (no_trade_start2 <= current_time <= no_trade_end2):
        return True
    
    # 如果有夜盘，检查夜盘的不交易时间段
    if night_session.get('enabled', False):
        night_start = parse_time_str(night_session['start_time'])
        night_end = parse_time_str(night_session['end_time'])
        
        # 判断是否在夜盘时间段内
        if night_start <= current_time <= night_end or \
           (night_start > night_end and (current_time >= night_start or current_time <= night_end)):
            night_no_trade_start1 = parse_time_str(night_session['no_trade_start'])
            night_no_trade_end1 = parse_time_str(night_session['no_trade_end'])
            night_no_trade_start2 = parse_time_str(night_session['no_trade_start2'])
            night_no_trade_end2 = parse_time_str(night_session['no_trade_end2'])
            
            # 夜盘不交易时间段1: 21:00-21:10
            if night_no_trade_start1 <= current_time <= night_no_trade_end1:
                return True
            # 夜盘不交易时间段2: 22:50-23:00
            if night_no_trade_start2 <= current_time <= night_no_trade_end2:
                return True
    
    return False

def get_bar_in_day(time_config):
    """
    根据交易时间配置计算bar_in_day
    
    参数:
        time_config: 时间配置字典
    
    返回:
        int: 一天内的分钟数
    """
    # 如果配置了bar_in_day，直接使用
    if 'bar_in_day' in time_config:
        return time_config['bar_in_day']
    
    # 否则根据交易时段计算
    total_minutes = 0
    
    # 上午: 9:00-11:30 = 150分钟
    # 减去休息时间 10:15-10:30 = 15分钟
    morning_minutes = 150 - 15
    
    # 下午: 13:30-15:00 = 90分钟
    afternoon_minutes = 90
    
    total_minutes += morning_minutes + afternoon_minutes
    
    # 如果有夜盘
    night_session = time_config.get('night_session', {})
    if night_session.get('enabled', False):
        night_start = parse_time_str(night_session['start_time'])
        night_end = parse_time_str(night_session['end_time'])
        
        # 计算夜盘分钟数
        night_minutes = (night_end.hour * 60 + night_end.minute) - (night_start.hour * 60 + night_start.minute)
        if night_minutes < 0:  # 跨天的情况
            night_minutes += 24 * 60
        
        total_minutes += night_minutes
    
    return total_minutes

def load_config_for_symbols(config, main_symbol, other_symbols):
    """
    根据指定主品种和其他品种加载配置
    
    参数:
        config: 配置字典
        main_symbol: 主要品种代码（如 'P', 'M' 等）
        other_symbols: 其他品种代码列表（如 ['M', 'A', 'Y']）
    
    返回:
        包含所有必要变量的字典
    """
    instruments = config.get("instruments", {})
    if main_symbol not in instruments:
        raise ValueError(f"配置中找不到主品种 {main_symbol}")
    
    # 构建 instrument_list
    instrument_list = [instruments[main_symbol]["contract"]]
    for symbol in other_symbols:
        if symbol in instruments:
            instrument_list.append(instruments[symbol]["contract"])
        else:
            print(f"警告: 品种 {symbol} 的配置不存在，已跳过")
    
    # 路径与模型配置
    paths = config.get("paths", {})
    model_path = os.path.join(paths.get("models_root", ""), main_symbol)
    save_path = os.path.join(paths.get("save_files_root", ""), main_symbol)
    publisher_port = config.get("model_config", {}).get(main_symbol, {}).get("port")
    
    # 获取交易时段
    symbol_specs = config.get("symbol_specs", {})
    main_spec = symbol_specs.get(main_symbol, {})
    trade_hours = main_spec.get("trade_hours", {})
    
    db_path = paths.get("db_path")
    recent_data_path = paths.get("load_recent_data_path")
    
    # 获取交易参数
    trading_params = config.get("trading_params", {})
    th1 = trading_params.get("th1")
    th2 = trading_params.get("th2")
    holding_period_max = trading_params.get("holding_period_max")
    
    # 获取时间配置
    time_config = config.get("time_config", {})
    bar_in_day = get_bar_in_day(time_config)
    
    return {
        "instrument_list": instrument_list,
        "model_path": model_path,
        "save_path": save_path,
        "trade_hours": trade_hours,
        "db_path": db_path,
        "recent_data_path": recent_data_path,
        "th1": th1,
        "th2": th2,
        "holding_period_max": holding_period_max,
        "bar_in_day": bar_in_day,
        "time_config": time_config,
        "main_symbol": main_symbol,
        "main_contract": instruments[main_symbol]["contract"],
        "other_symbols": other_symbols,
        "publisher_port": publisher_port
    }

def load_model_lst(config_loaded):
    """加载模型列表"""
    model_lst = []
    weight_lst = []
    factor_col = None
    
    for i in range(1, 6):
        try:
            model_file = f'{config_loaded["model_path"]}/kfold_fold{i}_0.lgb'
            model = lgb.Booster(model_file=model_file)
            
            meta_file = f'{config_loaded["model_path"]}/kfold_fold{i}_0_meta.json'
            with open(meta_file, 'r') as f:
                meta_data = json.load(f)
        
            model_lst.append(model)
            weight_lst.append(float(np.log(meta_data['best_iteration'] + 1)))
            
            factor_col = model.feature_name()
            
            # print(f"✅ 模型 {i} 加载成功")
            
        except FileNotFoundError as e:
            print(f"❌ 模型文件不存在: {e}")
            break
        except Exception as e:
            print(f"❌ 加载模型 {i} 失败: {e}")
            break

    if not model_lst:
        raise Exception("❌ 没有成功加载任何模型！")

    # print(f"✅ 成功加载 {len(model_lst)} 个模型")
    # print(f"✅ 权重列表: {weight_lst}")
    # if factor_col:
    #     print(f"✅ 特征数量: {len(factor_col)}")
    return model_lst, weight_lst, factor_col

def check_minute_data_consistency(instruments_to_load, data_dict):
    """
    检查分钟数据的一致性和非空性
    
    参数:
        instruments_to_load: 合约加载列表
        data_dict: 数据字典，键为变量名，值为对应的DataFrame
        
    返回:
        bool: 是否所有检查都通过
    """
    min_data_list = []
    empty_data = []
    
    for ins_key, tick_var, min_var in instruments_to_load:
        data = data_dict.get(min_var)
        min_data_list.append((ins_key, data))
        if data is None or len(data) == 0:
            empty_data.append(ins_key)
    
    # 检查空数据
    if empty_data:
        print(f"❌ 以下数据为空: {empty_data}")
        return False
    else:
        print("✅ 所有分钟数据非空")
    
    # 检查数据长度一致性
    lengths = {name: len(data) for name, data in min_data_list if data is not None}
    if len(set(lengths.values())) > 1:
        print(f"❌ 数据长度不一致: {lengths}")
        return False
    else:
        print(f"✅ 所有分钟数据长度一致: {list(lengths.values())[0] if lengths else 0}")
        return True

def load_instrument_parallel(instrument_key, tick_var, min_var, config, config_loaded, current_time):
    """并行加载单个合约数据"""
    instrument = config["instruments"][instrument_key]
    tick_data, min_data = load_tick_min(
        instrument, 
        current_time, 
        config_loaded['recent_data_path'],
        config_loaded['db_path'],
        trade_type=config_loaded.get('trade_hours'),
        time_config=config_loaded.get('time_config')
    )
    return tick_var, tick_data, min_var, min_data
##----------------------------------------------------------------------------------------------------------------##

def parse_time_str(time_str):
    """将时间字符串解析为time对象"""
    parts = time_str.split(':')
    return time(int(parts[0]), int(parts[1]))

class ResampleMethods:
    """重采样方法统一管理"""
    
    def __init__(self, tick_data, valid_index):
        self.tick_data = tick_data
        self.valid_index = valid_index
    
    def _prepare_data(self, tick_fac, extra_cols):
        """准备数据：重命名列并合并额外列（使用预缓存的 _tick_indexed 避免重复 set_index）"""
        tick_fac = tick_fac.copy()
        if tick_fac.columns[0] != 'factor_value':
            tick_fac = tick_fac.rename(columns={tick_fac.columns[0]: 'factor_value'})
        
        if extra_cols:
            merged_data = pd.concat([
                tick_fac,
                self._tick_indexed[extra_cols]
            ], axis=1)
        else:
            merged_data = tick_fac
        
        return merged_data
    
    def _apply_resample(self, tick_fac, compute_func, extra_cols=None):
        """应用重采样（fallback，向量化方法未覆盖时调用）"""
        merged_data = self._prepare_data(tick_fac, extra_cols)
        result = merged_data.resample('1min', label='right', closed='right').apply(compute_func)
        result = result.reindex(index=self.valid_index)
        return result.squeeze()  # 确保返回Series
    def resample_agg(self, tick_fac, agg_method):
        """
        统一的降频接口，支持所有重采样方法
        
        参数:
        tick_fac: 因子DataFrame
        agg_method: 降频方法，可以是:
            - 简单聚合: 'mean', 'sum', 'first', 'last', 'max', 'min', 'std', 'var'
            - 特殊统计: 'kurtosis', 'skewness'
            - 复杂方法: 'vwap', 'askdommean', 'biddommean', 'corrAsk', 'corrBid',
                       'Mstd', 'upmean', 'downmean', 'Volraise', 'MADmean', 'trend_rev'
        
        返回:
        numpy.ndarray: 一维数组
        """
        tick_fac = tick_fac.replace([-np.inf, np.inf], np.nan)
        if len(tick_fac) <= 60 * 4 * 2:
            return np.array([np.nan] * len(self.valid_index))

        def ensure_1d_array(result):
            """确保返回一维数组"""
            if isinstance(result, pd.DataFrame):
                # DataFrame转换为Series再取values
                result = result.squeeze()
                if isinstance(result, pd.Series):
                    return result.values
                else:
                    # squeeze后变成标量（如numpy.float64，当只有1个点时）
                    return np.array([result])
            elif isinstance(result, pd.Series):
                # Series直接取values
                return result.values
            elif isinstance(result, np.ndarray):
                # numpy数组
                if result.ndim == 2:
                    return result.ravel()
                return result
            else:
                # 标量或其他类型
                return np.array([result])
        
        # 1. 简单聚合方法
        simple_agg_methods = ['mean', 'sum', 'first', 'last', 'max', 'min', 'std', 'var']
        if agg_method in simple_agg_methods:
            result = tick_fac.resample('1min', label='right', closed='right').agg(agg_method)
            result = result.reindex(index=self.valid_index)

            return ensure_1d_array(result)
        
        # 2. 特殊统计方法
        elif agg_method == 'kurtosis':
            result = self._resample_kurtosis(tick_fac)
            return ensure_1d_array(result)
        
        elif agg_method == 'skewness':
            grouper = pd.Grouper(freq='1min', label='right', closed='right')
            result = tick_fac.groupby(grouper)['factor_value'].skew()
            result = result.reindex(index=self.valid_index)
            return ensure_1d_array(result)
        
        # 3. 复杂重采样方法
        elif agg_method == 'vwap':
            result = self.resample_vwap(tick_fac)
            return ensure_1d_array(result)
        
        elif agg_method == 'askdommean':
            result = self.resample_askdommean(tick_fac)
            return ensure_1d_array(result)
        
        elif agg_method == 'biddommean':
            result = self.resample_biddommean(tick_fac)
            return ensure_1d_array(result)
        
        elif agg_method == 'corrAsk':
            result = self.resample_corrAsk(tick_fac)
            return ensure_1d_array(result)
        
        elif agg_method == 'corrBid':
            result = self.resample_corrBid(tick_fac)
            return ensure_1d_array(result)
        
        elif agg_method == 'Mstdwap':
            result = self.resample_Mstd(tick_fac)
            return ensure_1d_array(result)
        
        elif agg_method == 'upmean':
            result = self.resample_upmean(tick_fac)
            return ensure_1d_array(result)
        
        elif agg_method == 'downmean':
            result = self.resample_downmean(tick_fac)
            return ensure_1d_array(result)
        
        elif agg_method == 'Volraise':
            result = self.resample_Volraise(tick_fac)
            return ensure_1d_array(result)
        
        elif agg_method == 'MADmean':
            result = self.resample_MADmean(tick_fac)
            return ensure_1d_array(result)
        
        elif agg_method == 'trend_rev':
            result = self.resample_trend_rev(tick_fac)
            return ensure_1d_array(result)
        
        else:
            raise ValueError(f"不支持的降频方法: {agg_method}")
        
    # 2. 加权平均类方法
    def resample_vwap(self, tick_fac):
        """成交量加权平均"""
        def compute_vwap(group):
            if len(group) == 0:
                return np.nan
            weights = group['TotalTradeVolume'].diff().fillna(0)
            weighted_value = (group['factor_value'] * weights).sum() / weights.sum() if weights.sum() != 0 else np.nan
            return weighted_value
        
        return self._apply_resample(tick_fac, compute_vwap, ['TotalTradeVolume'])
    
    def resample_askdommean(self, tick_fac):
        """卖盘主导时的加权平均（向量化）"""
        merged = self._prepare_data(tick_fac, ['bvall', 'avall'])
        grouper = pd.Grouper(freq='1min', label='right', closed='right')
        weights = (merged['bvall'] < merged['avall']).astype(int)
        weighted_sum = (merged['factor_value'] * weights).groupby(grouper).sum()
        weight_sum = weights.groupby(grouper).sum()
        result = weighted_sum / weight_sum
        result = result.reindex(index=self.valid_index)
        return result.squeeze()
    
    def resample_biddommean(self, tick_fac):
        """买盘主导时的加权平均（向量化）"""
        merged = self._prepare_data(tick_fac, ['bvall', 'avall'])
        grouper = pd.Grouper(freq='1min', label='right', closed='right')
        weights = (merged['bvall'] > merged['avall']).astype(int)
        weighted_sum = (merged['factor_value'] * weights).groupby(grouper).sum()
        weight_sum = weights.groupby(grouper).sum()
        result = weighted_sum / weight_sum
        result = result.reindex(index=self.valid_index)
        return result.squeeze()
    
    def resample_corrAsk(self, tick_fac):
        """与ask相关性的加权平均（向量化）"""
        merged = self._prepare_data(tick_fac, ['corrAskwap'])
        grouper = pd.Grouper(freq='1min', label='right', closed='right')
        weights = 1 - merged['corrAskwap']
        weighted_sum = (merged['factor_value'] * weights).groupby(grouper).sum()
        weight_sum = weights.groupby(grouper).sum()
        result = weighted_sum / weight_sum
        result = result.reindex(index=self.valid_index)
        return result.squeeze()
    
    def resample_corrBid(self, tick_fac):
        """与bid相关性的加权平均（向量化）"""
        merged = self._prepare_data(tick_fac, ['corrBidwap'])
        grouper = pd.Grouper(freq='1min', label='right', closed='right')
        weights = 1 - merged['corrBidwap']
        weighted_sum = (merged['factor_value'] * weights).groupby(grouper).sum()
        weight_sum = weights.groupby(grouper).sum()
        result = weighted_sum / weight_sum
        result = result.reindex(index=self.valid_index)
        return result.squeeze()
    
    def resample_Mstd(self, tick_fac):
        """M_std加权平均（向量化）"""
        merged = self._prepare_data(tick_fac, ['M_std'])
        grouper = pd.Grouper(freq='1min', label='right', closed='right')
        weighted_sum = (merged['factor_value'] * merged['M_std']).groupby(grouper).sum()
        weight_sum = merged['M_std'].groupby(grouper).sum()
        result = weighted_sum / weight_sum
        result = result.reindex(index=self.valid_index)
        return result.squeeze()
    
    # 3. 条件平均类方法
    def resample_upmean(self, tick_fac):
        """价格上涨时的平均值（向量化）"""
        merged = self._prepare_data(tick_fac, ['mid_price'])
        grouper = pd.Grouper(freq='1min', label='right', closed='right')
        diff2 = merged['mid_price'] - merged['mid_price'].groupby(grouper).shift(2)
        result = merged['factor_value'].where(diff2 >= 0, 0).groupby(grouper).agg(np.mean)
        result = result.reindex(index=self.valid_index)
        return result.squeeze()
    
    def resample_downmean(self, tick_fac):
        """价格下跌时的平均值（向量化）"""
        merged = self._prepare_data(tick_fac, ['mid_price'])
        grouper = pd.Grouper(freq='1min', label='right', closed='right')
        diff2 = merged['mid_price'] - merged['mid_price'].groupby(grouper).shift(2)
        result = merged['factor_value'].where(diff2 <= 0, 0).groupby(grouper).agg(np.mean)
        result = result.reindex(index=self.valid_index)
        return result.squeeze()
    
    def resample_Volraise(self, tick_fac):
        """成交量突增时的加权平均"""
        def compute_Volraise(group):
            weights = (group['TotalTradeVolume'].diff() > 1.5 * group.volume_avg).astype(int)
            weighted_value = (group['factor_value'] * weights).sum() / weights.sum() if weights.sum() != 0 else np.nan
            return weighted_value
        
        return self._apply_resample(tick_fac, compute_Volraise, ['TotalTradeVolume', 'volume_avg'])
    
    # 4. 特殊统计方法
    def resample_MADmean(self, tick_fac):
        """基于MAD的稳健平均值（向量化）"""
        merged = self._prepare_data(tick_fac, ['mid_price'])
        grouper = pd.Grouper(freq='1min', label='right', closed='right')
        groups = merged.groupby(grouper)['factor_value']
        median = groups.transform('median')
        mad = (merged['factor_value'] - median).abs().groupby(grouper).transform('median')
        mask = (merged['factor_value'] - median).abs() <= 3 * mad
        result = merged.loc[mask, 'factor_value'].groupby(grouper).mean()
        result = result.reindex(index=self.valid_index)
        return result.squeeze()
    
    def _resample_kurtosis(self, tick_fac):
        """基于 pandas 底层 nankurt 的向量化 kurtosis"""
        from pandas.core.nanops import nankurt
        tick_fac = tick_fac.copy()
        if tick_fac.columns[0] != 'factor_value':
            tick_fac = tick_fac.rename(columns={tick_fac.columns[0]: 'factor_value'})
        grouper = pd.Grouper(freq='1min', label='right', closed='right')
        def _fast_kurt(x):
            if len(x) <= 3:
                return np.nan
            return nankurt(x.values, skipna=True)
        result = tick_fac.groupby(grouper)['factor_value'].apply(_fast_kurt)
        result = result.reindex(index=self.valid_index)
        return result.squeeze()
    
    def resample_trend_rev(self, tick_fac):
        """趋势反转时的加权平均（向量化）"""
        merged = self._prepare_data(tick_fac, ['mid_price'])
        grouper = pd.Grouper(freq='1min', label='right', closed='right')
        short_trend = merged['mid_price'] - merged['mid_price'].groupby(grouper).shift(10)
        long_trend = merged['mid_price'] - merged['mid_price'].groupby(grouper).shift(30)
        prod = short_trend * long_trend
        weights = prod.mask(prod > 0, 0).abs()
        weighted_sum = (merged['factor_value'] * weights).groupby(grouper).sum()
        weight_sum = weights.groupby(grouper).sum()
        result = weighted_sum / weight_sum
        result = result.reindex(index=self.valid_index)
        return result.squeeze()


# ==============================================================================
# 生产环境 min 因子截断优化：按 trade_date 数量截断，减少全历史重复遍历
# ==============================================================================
MIN_FACTOR_TRADE_DATES = {
    'bar3_trend_corr': 1,   # 纯 rolling，窗口 3，当天足够
    'bar5_trend_corr': 1,   # 纯 rolling，窗口 5，当天足够
    'zigzag': 6,            # volatility_rg 需要 240*5=1200 bar（约 5 天），留 1 天缓冲
    'day_jump': 2,          # 需要当天 open + 前一天 close
}


# ==============================================================================
# 模块级 numba 函数（避免 bar3/bar5_trend_corr 每次调用都重复 JIT 编译）
# ==============================================================================
@njit
def _spearman_rank(trend, ranks):
    n = len(trend)
    sum_tr = (trend * ranks).sum()
    sum_t = trend.sum()
    sum_r = ranks.sum()
    sum_t2 = (trend**2).sum()
    sum_r2 = (ranks**2).sum()
    numerator = n*sum_tr - sum_t*sum_r
    denom1 = np.sqrt(n*sum_t2 - sum_t**2)
    denom2 = np.sqrt(n*sum_r2 - sum_r**2)
    return numerator / (denom1 * denom2 + 1e-10)


@njit
def _wma_numba(values, window):
    n = len(values)
    result = np.full(n, np.nan)
    weights = np.arange(window, 0, -1)
    wsum = weights.sum()
    for i in range(window - 1, n):
        result[i] = np.sum(values[i - window + 1:i + 1] * weights) / wsum
    return result


@njit
def _rolling_corr_3(close):
    n = len(close)
    result = np.full(n, np.nan)
    trend = np.arange(3)
    for i in range(2, n):
        window = close[i-2:i+1]
        ranks = np.argsort(np.argsort(window)) + 1
        result[i] = round(_spearman_rank(trend, ranks), 2)
    return result


@njit
def _rolling_corr_5(close):
    n = len(close)
    result = np.full(n, np.nan)
    trend = np.arange(5)
    for i in range(4, n):
        window = close[i-4:i+1]
        ranks = np.argsort(np.argsort(window)) + 1
        result[i] = round(_spearman_rank(trend, ranks), 2)
    return result


def _truncate_by_trade_date(func):
    """装饰器：按 trade_date 数量截断 min_data，不影响 self.min_data 本身"""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        n_dates = MIN_FACTOR_TRADE_DATES.get(func.__name__)
        if n_dates and hasattr(self, 'min_data') and 'trade_date' in self.min_data.columns:
            unique_dates = self.min_data['trade_date'].unique()
            if len(unique_dates) > n_dates:
                keep_dates = unique_dates[-n_dates:]
                original_min_data = self.min_data
                self.min_data = self.min_data[
                    self.min_data['trade_date'].isin(keep_dates)
                ].copy()
                try:
                    return func(self, *args, **kwargs)
                finally:
                    self.min_data = original_min_data
        return func(self, *args, **kwargs)
    return wrapper


class Factor_generator(ResampleMethods):
    def __init__(self, tick_data, min_data, *other_mins):

        self._load_data_tick(tick_data)
        self._load_data_min(min_data, *other_mins)
        super().__init__(tick_data, self.valid_index)
        
    def load_df_names(self):
        self.symbol_data_map = dict(
            zip(self.dict_keys, [self.min_data] + list(self.other_mins))
        )
        # 预缓存 min_data / other_mins 的 set_index 结果，减少跨品种因子重复计算
        self._symbol_data_indexed = {}
        for key, df in self.symbol_data_map.items():
            if df is not None and not df.empty and 'datetime' in df.columns:
                self._symbol_data_indexed[key] = df.set_index('datetime')
            else:
                self._symbol_data_indexed[key] = pd.DataFrame()

    def get_symbol_data(self, symbol):
        return self.symbol_data_map.get(symbol)

    def _load_data_tick(self, tick_data):
        self.tick_data = tick_data
        
    def _load_data_min(self, *min_datas):
        self.min_data = min_datas[0]
        self.other_mins = min_datas[1:]
        
        # 从 tick_data 中获取合约代码，读取 config 获取对应品种的 multiplier
        try:
            instrument = self.tick_data['instrument'].iloc[0]
            symbol = get_symbol_from_instrument(instrument)
            config = load_config()
            self.main_multiplier = config.get('symbol_specs', {}).get(symbol, {}).get('multiplier', 10)
        except Exception:
            self.main_multiplier = 10
        
        # 预缓存 tick_data set_index，避免每个因子重复做
        self._tick_indexed = self.tick_data.set_index('datetime')
        
        time_index = (self._tick_indexed
                    .resample('1min', label='right', closed='right').last().index.time)

        evening_start = time(21, 00, 0, 1)
        evening_end = time(23, 00, 0, 0)
        morning1_start = time(9, 00, 0, 1)      
        morning1_end = time(10, 15, 0, 0)     
        morning2_start = time(10, 30, 0, 1)      
        morning2_end = time(11, 30, 0, 0)  
        afternoon_start = time(13, 0, 0, 1)    
        afternoon_end = time(15, 0, 0, 0)    
        
        self.is_trading = (
            # 夜盘时间段: 21:00-23:00
            ((time_index >= evening_start) & (time_index <= evening_end)) |
            
            # 早盘第一节: 9:00-10:15
            ((time_index >= morning1_start) & (time_index <= morning1_end)) |
            
            # 早盘第二节: 10:30-11:30
            ((time_index >= morning2_start) & (time_index <= morning2_end)) |
            
            # 下午盘: 13:00-15:00
            ((time_index >= afternoon_start) & (time_index <= afternoon_end))
        )
        
        self.valid_index = (self._tick_indexed
                            .resample('1min', label='right', closed='right').last()
                            .loc[self.is_trading].dropna(how='all').index)
        
        # 运行时优化：如果 tick_data 很短（<5000条），说明是实时模式，
        # 只保留 valid_index 的最后一个点，减少 resample 和 predict 的计算量。
        # tick 级别 rolling 的历史数据仍保留在 tick_data 中，不受影响。
        if len(self.tick_data) < 5000 and len(self.valid_index) > 1:
            self.valid_index = self.valid_index[-1:]
    
    # -----------------------------------工具函数-------------------------------------------------------------------
    @staticmethod
    def wma(series, window):
        result = _wma_numba(series.values, window)
        return pd.Series(result, index=series.index)

    # -----------------------------------t2t原始因子-------------------------------------------------------------------
    def FAC_atr(self, agg_method='mean', window=14):
        """ATR因子"""
        data = self.tick_data.copy(deep=False)
        tr = np.maximum(data.highest_price - data.lowest_price, 
                       (data.highest_price - data.mid_price.shift(1)).abs(), 
                       (data.lowest_price - data.mid_price.shift(1)).abs())
        fac = tr.rolling(window=window).mean().diff(60 * 2).clip(lower=0, upper=20)
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_RVar(self, agg_method='mean', window=120):
        """RVar因子"""
        data = self.tick_data.copy(deep=False)
        Rvar = (data.mid_price.pct_change()**2).rolling(window=window).sum()
        fac = (Rvar * 1e5).clip(lower=0, upper=1)
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_RSkew(self, agg_method='mean', window=120):
        """RSkew因子"""
        data = self.tick_data.copy(deep=False)
        RSkew = (data.mid_price.diff()**3).rolling(window=window, min_periods=100).sum() * np.sqrt(window) / \
                (data.mid_price.diff()**2).rolling(window=window, min_periods=100).sum()**1.5
        fac = RSkew
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_RKurt(self, agg_method='mean', window=120):
        """RKurt因子"""
        data = self.tick_data.copy(deep=False)
        RKurt = (data.mid_price.diff()**4).rolling(window=window, min_periods=100).sum() * window / \
                (data.mid_price.diff()**2).rolling(window=window, min_periods=100).sum()**2
        fac = RKurt
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_RVar_down_rate(self, agg_method='mean', window=120):
        """RVar_down_rate因子"""
        data = self.tick_data.copy(deep=False)
        RVar_down = (data.mid_price.diff().apply(lambda x: 0 if x > 0 else x)**2).rolling(window=window).sum()
        RVar = (data.mid_price.diff()**2).rolling(window=window).sum()
        RVar_down_rate = RVar_down / RVar
        fac = RVar_down_rate
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_ask1_vmean_20(self, agg_method='mean', window=20):
        """ask1_vmean_20因子"""
        data = self.tick_data.copy(deep=False)
        ask1_vmean_20 = data.ask_volume1.rolling(window=window).sum()
        fac = ask1_vmean_20 / data.mid_price.rolling(window=window).mean()
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_ask2_vmean_20(self, agg_method='mean', window=20):
        """ask2_vmean_20因子"""
        data = self.tick_data.copy(deep=False)
        ask2_vmean_20 = data.ask_volume2.rolling(window=window).sum()
        fac = ask2_vmean_20 / data.mid_price.rolling(window=window).mean()
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_ask3_vmean_20(self, agg_method='mean', window=20):
        """ask3_vmean_20因子"""
        data = self.tick_data.copy(deep=False)
        ask3_vmean_20 = data.ask_volume3.rolling(window=window).sum()
        fac = ask3_vmean_20 / data.mid_price.rolling(window=window).mean()
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_ask4_vmean_20(self, agg_method='mean', window=20):
        """ask4_vmean_20因子"""
        data = self.tick_data.copy(deep=False)
        ask4_vmean_20 = data.ask_volume4.rolling(window=window).sum()
        fac = ask4_vmean_20 / data.mid_price.rolling(window=window).mean()
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_ask5_vmean_20(self, agg_method='mean', window=20):
        """ask5_vmean_20因子"""
        data = self.tick_data.copy(deep=False)
        ask5_vmean_20 = data.ask_volume5.rolling(window=window).sum()
        fac = ask5_vmean_20 / data.mid_price.rolling(window=window).mean()
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_bid1_vmean_20(self, agg_method='mean', window=20):
        """bid1_vmean_20因子"""
        data = self.tick_data.copy(deep=False)
        bid1_vmean_20 = data.bid_volume1.rolling(window=window).sum()
        fac = bid1_vmean_20 / data.mid_price.rolling(window=window).mean()
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_bid2_vmean_20(self, agg_method='mean', window=20):
        """bid2_vmean_20因子"""
        data = self.tick_data.copy(deep=False)
        bid2_vmean_20 = data.bid_volume2.rolling(window=window).sum()
        fac = bid2_vmean_20 / data.mid_price.rolling(window=window).mean()
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_bid3_vmean_20(self, agg_method='mean', window=20):
        """bid3_vmean_20因子"""
        data = self.tick_data.copy(deep=False)
        bid3_vmean_20 = data.bid_volume3.rolling(window=window).sum()
        fac = bid3_vmean_20 / data.mid_price.rolling(window=window).mean()
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_bid4_vmean_20(self, agg_method='mean', window=20):
        """bid4_vmean_20因子"""
        data = self.tick_data.copy(deep=False)
        bid4_vmean_20 = data.bid_volume4.rolling(window=window).sum()
        fac = bid4_vmean_20 / data.mid_price.rolling(window=window).mean()
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_bid5_vmean_20(self, agg_method='mean', window=20):
        """bid5_vmean_20因子"""
        data = self.tick_data.copy(deep=False)
        bid5_vmean_20 = data.bid_volume5.rolling(window=window).sum()
        fac = bid5_vmean_20 / data.mid_price.rolling(window=window).mean()
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_sub_a1b1_vmean_20(self, agg_method='mean', window=20):
        """sub_a1b1_vmean_20因子"""
        data = self.tick_data.copy(deep=False)
        sub_a1b1_vmean_20 = data.ask_volume1.rolling(window=window).sum() - data.bid_volume1.rolling(window=window).sum()
        fac = sub_a1b1_vmean_20
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_sub_a2b2_vmean_20(self, agg_method='mean', window=20):
        """sub_a2b2_vmean_20因子"""
        data = self.tick_data.copy(deep=False)
        sub_a2b2_vmean_20 = data.ask_volume2.rolling(window=window).sum() - data.bid_volume2.rolling(window=window).sum()
        fac = sub_a2b2_vmean_20
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_sub_a3b3_vmean_20(self, agg_method='mean', window=20):
        """sub_a3b3_vmean_20因子"""
        data = self.tick_data.copy(deep=False)
        sub_a3b3_vmean_20 = data.ask_volume3.rolling(window=window).sum() - data.bid_volume3.rolling(window=window).sum()
        fac = sub_a3b3_vmean_20
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_sub_a4b4_vmean_20(self, agg_method='mean', window=20):
        """sub_a4b4_vmean_20因子"""
        data = self.tick_data.copy(deep=False)
        sub_a4b4_vmean_20 = data.ask_volume4.rolling(window=window).sum() - data.bid_volume4.rolling(window=window).sum()
        fac = sub_a4b4_vmean_20
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_sub_a5b5_vmean_20(self, agg_method='mean', window=20):
        """sub_a5b5_vmean_20因子"""
        data = self.tick_data.copy(deep=False)
        sub_a5b5_vmean_20 = data.ask_volume5.rolling(window=window).sum() - data.bid_volume5.rolling(window=window).sum()
        fac = sub_a5b5_vmean_20
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_money_flow_power(self, agg_method='mean', window=20):
        """money_flow_power因子"""
        data = self.tick_data.copy(deep=False)
        money_flow_a = (data.ask_volume1 * data.ask_price1 + data.ask_volume2 * data.ask_price2 + 
                       data.ask_volume3 * data.ask_price3 + data.ask_volume4 * data.ask_price4 + 
                       data.ask_volume5 * data.ask_price5)
        money_flow_b = (data.bid_volume1 * data.bid_price1 + data.bid_volume2 * data.bid_price2 + 
                       data.bid_volume3 * data.bid_price3 + data.bid_volume4 * data.bid_price4 + 
                       data.bid_volume5 * data.bid_price5)
        money_flow_power = (money_flow_a - money_flow_b).rolling(window=window).sum() / \
                          (money_flow_a - money_flow_b).abs().rolling(window=window).sum()
        fac = money_flow_power
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_MPC(self, agg_method='mean', window=5):
        """MPC因子"""
        data = self.tick_data.copy(deep=False)
        MPC = (data.ask_price1 + data.bid_price1).pct_change(window)
        fac = MPC
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_MPB(self, agg_method='mean'):
        """MPB因子"""
        data = self.tick_data.copy(deep=False)
        M = data.mid_price
        
        contract_multiplier = self.main_multiplier
        
        TP = data['turnover'] / (data['volume'] * contract_multiplier)
        fac = TP - M
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_PVcorrsub_a1b1(self, agg_method='mean', window=20):
        """PVcorrsub_a1b1因子"""
        data = self.tick_data.copy(deep=False)
        PVcorrsub_a1b1 = data.ask_price1.rolling(window=window).corr(data.ask_volume1) - \
                        data.bid_price1.rolling(window=window).corr(data.bid_volume1)
        fac = PVcorrsub_a1b1
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_PVcorrsub_a2b2(self, agg_method='mean', window=20):
        """PVcorrsub_a2b2因子"""
        data = self.tick_data.copy(deep=False)
        PVcorrsub_a2b2 = data.ask_price2.rolling(window=window).corr(data.ask_volume2) - \
                        data.bid_price2.rolling(window=window).corr(data.bid_volume2)
        fac = PVcorrsub_a2b2
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_PVcorrsub_a3b3(self, agg_method='mean', window=20):
        """PVcorrsub_a3b3因子"""
        data = self.tick_data.copy(deep=False)
        PVcorrsub_a3b3 = data.ask_price3.rolling(window=window).corr(data.ask_volume3) - \
                        data.bid_price3.rolling(window=window).corr(data.bid_volume3)
        fac = PVcorrsub_a3b3
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_PVcorrsub_a4b4(self, agg_method='mean', window=20):
        """PVcorrsub_a4b4因子"""
        data = self.tick_data.copy(deep=False)
        acorr = data.ask_price4.rolling(window=window).corr(data.ask_volume4)
        bcorr = data.bid_price4.rolling(window=window).corr(data.bid_volume4)
        fac = acorr - bcorr
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_PVcorrsub_a5b5(self, agg_method='mean', window=20):
        """PVcorrsub_a5b5因子"""
        data = self.tick_data.copy(deep=False)
        PVcorrsub_a5b5 = data.ask_price5.rolling(window=window).corr(data.ask_volume5) - \
                        data.bid_price5.rolling(window=window).corr(data.bid_volume5)
        fac = PVcorrsub_a5b5
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_ask_amount_sub20(self, agg_method='mean'):
        """ask_amount_sub20因子"""
        data = self.tick_data.copy(deep=False)
        money_flow_a = (data.ask_volume1 * data.ask_price1 + data.ask_volume2 * data.ask_price2 + 
                       data.ask_volume3 * data.ask_price3 + data.ask_volume4 * data.ask_price4 + 
                       data.ask_volume5 * data.ask_price5)
        fac = money_flow_a.diff()
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_bid_amount_sub20(self, agg_method='mean'):
        """bid_amount_sub20因子"""
        data = self.tick_data.copy(deep=False)
        money_flow_b = (data.bid_volume1 * data.bid_price1 + data.bid_volume2 * data.bid_price2 + 
                       data.bid_volume3 * data.bid_price3 + data.bid_volume4 * data.bid_price4 + 
                       data.bid_volume5 * data.bid_price5)
        fac = money_flow_b.diff()
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_ADTM(self, agg_method='mean', window=20):
        """ADTM因子 - 与ADTMMA相同但使用不同计算公式"""
        data = self.tick_data.copy(deep=False)
        data['prev_price'] = data['mid_price'].shift(window)
        mask_up = data['mid_price'] > data['prev_price']
        mask_down = data['mid_price'] < data['prev_price']
        
        DTM = np.where(mask_up, data['volume'] * (data['mid_price'] - data['prev_price']), 0)
        DBM = np.where(mask_down, data['volume'] * (data['prev_price'] - data['mid_price']), 0)
        
        data['DTM'] = DTM
        data['DBM'] = DBM
        STM = data['DTM'].rolling(window=window).sum()
        SBM = data['DBM'].rolling(window=window).sum()
        
        fac = (STM - SBM) / np.maximum(STM, SBM)
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df

    def FAC_ADTMMA(self, agg_method='mean', window=20):
        data = self.tick_data.copy(deep=False)
        data['prev_price'] = data['mid_price'].shift(window)
        mask_up = data['mid_price'] > data['prev_price']
        mask_down = data['mid_price'] < data['prev_price']
        
        DTM = np.where(mask_up, data['volume'] * (data['mid_price'] - data['prev_price']), 0)
        DBM = np.where(mask_down, data['volume'] * (data['prev_price'] - data['mid_price']), 0)
        
        data['DTM'] = DTM
        data['DBM'] = DBM

        STM = data['DTM'].rolling(window=window).sum()
        SBM = data['DBM'].rolling(window=window).sum()
        
        fac = (STM - SBM) / (STM + SBM + 1e-9)
        fac = fac.rolling(window=8).sum()
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df

    def FAC_RSJ(self, agg_method='mean', window=120):
        """RSJ因子"""
        data = self.tick_data.copy(deep=False)
        RVar_down = (data.mid_price.diff().apply(lambda x: 0 if x > 0 else x)**2).rolling(window=window).sum()
        RVar_up = (data.mid_price.diff().apply(lambda x: 0 if x < 0 else x)**2).rolling(window=window).sum()
        RVar = (data.mid_price.diff()**2).rolling(window=window).sum()
        RSJ = (RVar_up - RVar_down) / (RVar + 0.001)
        fac = RSJ
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_VOI(self, agg_method='mean'):
        """VOI因子"""
        data = self.tick_data.copy(deep=False)
        
        def wavg(df, n=6):
            weights = np.array([1 - (i - 1) / (n - 1) for i in range(1, n)])
            return (df.iloc[:, :n-1].values * weights).sum(axis=1) / weights.sum()
        
        VWA = wavg(data[['ask_volume1', 'ask_volume2', 'ask_volume3', 'ask_volume4', 'ask_volume5']])
        VWB = wavg(data[['bid_volume1', 'bid_volume2', 'bid_volume3', 'bid_volume4', 'bid_volume5']])
        VWA_series = pd.Series(VWA, index=data.index)
        VWB_series = pd.Series(VWB, index=data.index)
        
        dVWA = np.where(data['ask_price1'].diff() < 0, 0,
                       np.where(data['ask_price1'].diff() == 0, VWA_series.diff(), VWA_series))
        dVWB = np.where(data['bid_price1'].diff() > 0, 0,
                       np.where(data['bid_price1'].diff() == 0, VWB_series.diff(), VWB_series))
        
        fac = dVWB - dVWA
        factor_df = pd.DataFrame(fac, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_DBCD(self, agg_method='mean', n1=5, n2=16, n3=17):
        """DBCD因子"""
        data = self.tick_data.copy(deep=False)
        data['SMA'] = data['mid_price'].rolling(window=n1).mean()
        data['BIAS'] = (data['mid_price'] - data['SMA']) / data['SMA']
        data['DIF'] = data['BIAS'] - data['BIAS'].shift(n2)
        fac = data['DIF'].rolling(window=n3).mean()
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_OIR(self, agg_method='mean'):
        """OIR因子"""
        data = self.tick_data.copy(deep=False)
        
        def wavg(df, n=6):
            weights = np.array([1 - (i - 1) / (n - 1) for i in range(1, n)])
            return (df.iloc[:, :n-1].values * weights).sum(axis=1) / weights.sum()
        
        VWA = wavg(data[['ask_volume1', 'ask_volume2', 'ask_volume3', 'ask_volume4', 'ask_volume5']])
        VWB = wavg(data[['bid_volume1', 'bid_volume2', 'bid_volume3', 'bid_volume4', 'bid_volume5']])
        fac = (VWB - VWA) / (VWB + VWA)
        
        factor_df = pd.DataFrame(fac, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_SOIR(self, agg_method='mean'):
        """SOIR因子"""
        data = self.tick_data.copy(deep=False)
            
        def wavg(df, n=6):
            # 计算权重
            weights = np.array([1 - (i - 1) / (n - 1) for i in range(1, n)])
            weights_sum = weights.sum()

            # 计算加权平均
            weighted_avg = (df * weights).sum(axis=1) / weights_sum
            return weighted_avg
        
        dfa = data[['ask_volume1', 'ask_volume2', 'ask_volume3', 'ask_volume4', 'ask_volume5']]
        dfb = data[['bid_volume1', 'bid_volume2', 'bid_volume3', 'bid_volume4', 'bid_volume5']]
        fac = wavg((dfa.values-dfb.values)/(dfa.values+dfb.values))    

        factor_df = pd.DataFrame(fac, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_MOFI(self, agg_method='mean'):
        """MOFI因子"""
        data = self.tick_data.copy(deep=False)
        MOFI = pd.Series(0, index=data.index)
        for i in range(1, 6):
            dVWA = np.where(data[f'ask_price{i}'].diff() > 0, -data[f'ask_volume{i}'].shift(),
                           np.where(data[f'ask_price{i}'].diff() == 0, data[f'ask_volume{i}'].diff(), data[f'ask_volume{i}']))
            dVWB = np.where(data[f'bid_price{i}'].diff() < 0, -data[f'bid_volume{i}'].shift(),
                           np.where(data[f'bid_price{i}'].diff() == 0, data[f'bid_volume{i}'].diff(), data[f'bid_volume{i}']))
            MOFI += (dVWA - dVWB) * i
        fac = MOFI / 15  # 1+2+3+4+5 = 15
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_PIR(self, agg_method='mean'):
        """PIR因子"""
        data = self.tick_data.copy(deep=False)
        pwa = pd.Series(0, index=data.index)
        pwb = pd.Series(0, index=data.index)
        divd = 0
        for i in range(1, 6):
            w = 1 - (i - 1) / 5
            pwb += w * data[f'bid_price{i}']
            pwa += w * data[f'ask_price{i}']
            divd += w
        pwa /= divd
        pwb /= divd
        fac = (pwb - pwa) / (pwb + pwa)
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_MAX(self, agg_method='mean', window=120):
        """MAX因子"""
        data = self.tick_data.copy(deep=False)
        MAX = data.groupby('trade_date')['mid_price'].transform(lambda x: x.diff(window).rolling(window=window).max()).values
        factor_df = pd.DataFrame(MAX, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_MLQSweight(self, agg_method='mean'):
        """MLQSweight因子"""
        data = self.tick_data.copy(deep=False)
        MLQSweight = pd.Series(0, index=data.index)
        divd = 0
        for i in range(1, 6):
            w = i / 5
            divd += w
            LS = w * (np.log(data[f'ask_price{i}']) - np.log(data[f'bid_price{i}'])) / \
                 (np.log(data[f'ask_volume{i}'] + 1) + np.log(data[f'bid_volume{i}'] + 1) + 1)
            MLQSweight += LS
        fac = MLQSweight / divd
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_MCIA(self, agg_method='mean'):
        """MCIA因子"""
        data = self.tick_data.copy(deep=False)
        M = data.mid_price
        DolVolA = (data.ask_price1 * data.ask_volume1 + data.ask_price2 * data.ask_volume2 + 
                  data.ask_price3 * data.ask_volume3 + data.ask_price4 * data.ask_volume4 + 
                  data.ask_price5 * data.ask_volume5)
        QA = data[['ask_volume1', 'ask_volume2', 'ask_volume3', 'ask_volume4', 'ask_volume5']].sum(axis=1)
        fac = ((DolVolA / QA) - M) / M
        fac = fac.clip(lower=0, upper=0.01)
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_MCIB(self, agg_method='mean'):
        """MCIB因子"""
        data = self.tick_data.copy(deep=False)
        M = data.mid_price
        DolVolB = (data.bid_price1 * data.bid_volume1 + data.bid_price2 * data.bid_volume2 + 
                  data.bid_price3 * data.bid_volume3 + data.bid_price4 * data.bid_volume4 + 
                  data.bid_price5 * data.bid_volume5)
        QB = data[['bid_volume1', 'bid_volume2', 'bid_volume3', 'bid_volume4', 'bid_volume5']].sum(axis=1)
        fac = ((DolVolB / QB) - M) / M
        fac = fac.clip(lower=-0.01, upper=0)
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_CORR_PVOL_RET(self, agg_method='mean'):
        """CORR_PVOL_RET因子"""
        data = self.tick_data.copy(deep=False)
        RET = (data['ask_price1']/2 + data['bid_price1']/2).diff()
        sig_trade_value = data['TotalTradeVolume'].diff()
        CORR_PVOL_RET = (sig_trade_value - sig_trade_value.rolling(window=120).mean()) / \
                       (RET.rolling(window=120).std() * sig_trade_value.rolling(window=120).std())
        fac = CORR_PVOL_RET
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_RTN_JUMP(self, agg_method='mean'):
        """RTN_JUMP因子"""
        data = self.tick_data.copy(deep=False)
        RET = data['mid_price'].pct_change()
        RTN_JUMP = ((RET - np.log(RET + 1))**2 - np.log(RET + 1)**2) * 1e6
        fac = RTN_JUMP.clip(lower=-1, upper=1)
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_NR(self, agg_method='mean', window=120):
        """NR因子"""
        data = self.tick_data.copy(deep=False)
        RETCO = data.mid_price / data.mid_price.shift(20) - 1
        RETOC = data.mid_price / data.mid_price.shift(240) - 1
        RETCO[RETCO < 0] = 0
        RETOC[RETOC > 0] = 0
        data['RETCOOC'] = RETCO * RETOC
        fac = data.groupby('trade_date')['RETCOOC'].transform(lambda x: x.rolling(window=window).sum() / (1 + np.arange(len(x)))) * 1e6
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_STREN(self, agg_method='mean'):
        """STREN因子"""
        data = self.tick_data.copy(deep=False)
        vol_buy = data['TotalTradeVolume'].diff(10)
        vol_buy.loc[data['mid_price'] < data['ask_price1'].shift()] = 0
        vol_sell = data['TotalTradeVolume'].diff(10)
        vol_sell.loc[data['mid_price'] > data['bid_price1'].shift()] = 0
        fac = (vol_buy - vol_sell).clip(-1, 1)
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_SPREAD(self, agg_method='mean', window=20):
        """SPREAD因子"""
        data = self.tick_data.copy(deep=False)
        SPREAD = -(data.ask_price1 - data.bid_price1) / (data.ask_price1 + data.bid_price1) * 2
        fac = SPREAD.rolling(window=window).mean()
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_ASKDEPTH(self, agg_method='mean'):
        """ASKDEPTH因子"""
        data = self.tick_data.copy(deep=False)
        ASKDEPTH = pd.Series(0, index=data.index)
        divd = 0
        for i in range(1, 6):
            w = (7 - i)
            divd += w
            ASKDEPTH += w * (data[f'ask_volume{i}'] * (data.ask_price1 + data.bid_price1)/2) / \
                       (data[f'ask_price{i}'] - data['ask_price1'].shift(1)/2 - data['bid_price1'].shift(1)/2 + 100).abs()
        fac = ASKDEPTH / divd
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_BBI(self, agg_method='mean'):
        """BBI因子"""
        data = self.tick_data.copy(deep=False)
        fac = data['mid_price'].rolling(window=4).mean() - data['mid_price'].rolling(window=8).mean()
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_resiliency(self, agg_method='mean'):
        """resiliency因子"""
        data = self.tick_data.copy(deep=False)
        fac = (data['highest_price'] - data['lowest_price']) / (data.TotalTradeVolume / data.open_interest)
        fac = fac.clip(lower=-1, upper=1000)
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_BIDDEPTH(self, agg_method='mean'):
        """BIDDEPTH因子"""
        data = self.tick_data.copy(deep=False)
        BIDDEPTH = pd.Series(0, index=data.index)
        divd = 0
        for i in range(1, 6):
            w = (7 - i)
            divd += w
            BIDDEPTH += w * (data[f'bid_volume{i}'] * (data.ask_price1 + data.bid_price1)/2) / \
                       (data[f'bid_price{i}'] - data['ask_price1'].shift(1)/2 - data['bid_price1'].shift(1)/2 + 100).abs()
        fac = BIDDEPTH / divd
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_AVGDEPTH(self, agg_method='mean'):
        """AVGDEPTH因子"""
        data = self.tick_data.copy(deep=False)
        fac = data.ask_volume1/2 + data.bid_volume1/2
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_VOL_FLU(self, agg_method='mean'):
        """VOL_FLU因子"""
        data = self.tick_data.copy(deep=False)
        VOL_FLU = data['TotalTradeVolume'].diff().rolling(window=240).std()
        fac = VOL_FLU.rolling(window=20).std() / VOL_FLU.rolling(window=20).mean()
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_QUA(self, agg_method='mean'):
        """QUA因子"""
        data = self.tick_data.copy(deep=False)
        sig_trade_value = data['TotalTradeVolume'].diff(10)
        QUA = (sig_trade_value.rolling(window=120, min_periods=100).quantile(0.1) - 
              sig_trade_value.rolling(window=120, min_periods=100).min()) / \
             (sig_trade_value.rolling(window=120, min_periods=100).max() - 
              sig_trade_value.rolling(window=120, min_periods=100).min())
        fac = QUA
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_shortQUA(self, agg_method='mean'):
        """shortQUA因子"""
        data = self.tick_data.copy(deep=False)
        sig_trade_value = data['TotalTradeVolume'].diff(10)
        QUA = (sig_trade_value.rolling(window=20).quantile(0.1) - 
              sig_trade_value.rolling(window=20).min()) / \
             (sig_trade_value.rolling(window=20).max() - 
              sig_trade_value.rolling(window=20).min())
        fac = QUA
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_midQUA(self, agg_method='mean'):
        """midQUA因子"""
        data = self.tick_data.copy(deep=False)
        sig_trade_value = data['TotalTradeVolume'].diff(10)
        QUA = (sig_trade_value.rolling(window=60).quantile(0.1) - 
              sig_trade_value.rolling(window=60).min()) / \
             (sig_trade_value.rolling(window=60).max() - 
              sig_trade_value.rolling(window=60).min())
        fac = QUA
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_PRICE_PRESSURE(self, agg_method='mean'):
        """PRICE_PRESSURE因子"""
        data = self.tick_data.copy(deep=False)
        data['last_price'] = data['mid_price'].shift(1)
        PRICE_PRESSURE = (data.ask_price1 - data.last_price) / (data.ask_price1 - data.bid_price1)
        fac = PRICE_PRESSURE - 0.5
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_ORDER_SLOPE_A(self, agg_method='mean'):
        """ORDER_SLOPE_A因子"""
        data = self.tick_data.copy(deep=False)
        ORDER_SLOPE_A = (data.ask_price5 - data.ask_price1) / \
                       (data.ask_volume1 + data.ask_volume2 + data.ask_volume3 + data.ask_volume4 + data.ask_volume5)
        fac = ORDER_SLOPE_A
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_ORDER_SLOPE_B(self, agg_method='mean'):
        """ORDER_SLOPE_B因子"""
        data = self.tick_data.copy(deep=False)
        ORDER_SLOPE_B = (data.bid_price5 - data.bid_price1) / \
                       (data.bid_volume1 + data.bid_volume2 + data.bid_volume3 + data.bid_volume4 + data.bid_volume5)
        fac = ORDER_SLOPE_B
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_PRICE_STD_B(self, agg_method='mean'):
        """PRICE_STD_B因子"""
        data = self.tick_data.copy(deep=False)
        fac = data[['bid_price1','bid_price2', 'bid_price3', 'bid_price4', 'bid_price5']].std(axis=1)
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_PRICE_STD_A(self, agg_method='mean'):
        """PRICE_STD_A因子"""
        data = self.tick_data.copy(deep=False)
        fac = data[['ask_price1','ask_price2', 'ask_price3', 'ask_price4', 'ask_price5']].std(axis=1)
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_OI_CHG(self, agg_method='mean'):
        """OI_CHG因子"""
        data = self.tick_data.copy(deep=False)
        fac = data['open_interest'].pct_change().clip(-0.01, 0.01)
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_OI_GP_CH(self, agg_method='mean'):
        """OI_GP_CH因子"""
        data = self.tick_data.copy(deep=False)
        OI_GP_CH = data['open_interest'].pct_change() * data['mid_price'].pct_change()
        fac = (OI_GP_CH * 1e6).clip(-1, 1)
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_OI_V_DIV(self, agg_method='mean'):
        """OI_V_DIV因子"""
        data = self.tick_data.copy(deep=False)
        OI_V_DIV = data['TotalTradeVolume'].pct_change() / data['open_interest'] * 1e6
        OI_V_DIV = OI_V_DIV.replace([-np.inf, np.inf], np.nan)
        fac = OI_V_DIV.clip(-1, 1)
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_Imbalance_OI(self, agg_method='mean'):
        """Imbalance_OI因子"""
        data = self.tick_data.copy(deep=False)
        Imbalance_OI = (data.bid_volume1 - data.ask_volume1) / (data.bid_volume1 + data.ask_volume1) * data['open_interest'].diff().values
        fac = Imbalance_OI
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_Depth_OI_Pressure(self, agg_method='mean'):
        """Depth_OI_Pressure因子"""
        data = self.tick_data.copy(deep=False)
        SUMAV = data[['ask_volume1', 'ask_volume2', 'ask_volume3', 'ask_volume4', 'ask_volume5']].sum(axis=1)
        SUMBV = data[['bid_volume1', 'bid_volume2', 'bid_volume3', 'bid_volume4', 'bid_volume5']].sum(axis=1)
        DOP = (SUMBV - SUMAV) / (SUMAV + SUMBV) * np.log(data.open_interest)
        fac = DOP
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_VWAP_Deviation(self, agg_method='mean'):
        """VWAP_Deviation因子"""
        data = self.tick_data.copy(deep=False)
        VWAP_Deviation = data.mid_price / (data.TotalTradeValue / data.TotalTradeVolume)
        fac = VWAP_Deviation
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_OI_Price_Accel(self, agg_method='mean'):
        """OI_Price_Accel因子"""
        data = self.tick_data.copy(deep=False)
        OI_Price_Accel = (data['open_interest'].diff() / data['open_interest']) * \
                        (data['mid_price'].diff() / data['mid_price']) * 1e6
        fac = OI_Price_Accel.clip(-1, 1)
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_OFI1(self, agg_method='mean'):
        """OFI1因子"""
        data = self.tick_data.copy(deep=False)
        i = 1
        dVWA = np.where(data[f'ask_price{i}'].diff() > 0, -data[f'ask_volume{i}'].shift(),
                       np.where(data[f'ask_price{i}'].diff() == 0, data[f'ask_volume{i}'].diff(), data[f'ask_volume{i}']))
        dVWB = np.where(data[f'bid_price{i}'].diff() < 0, -data[f'bid_volume{i}'].shift(),
                       np.where(data[f'bid_price{i}'].diff() == 0, data[f'bid_volume{i}'].diff(), data[f'bid_volume{i}']))
        fac = dVWA - dVWB
        
        factor_df = pd.DataFrame(fac, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_OFI2(self, agg_method='mean'):
        """OFI2因子"""
        data = self.tick_data.copy(deep=False)
        i = 2
        dVWA = np.where(data[f'ask_price{i}'].diff() > 0, -data[f'ask_volume{i}'].shift(),
                       np.where(data[f'ask_price{i}'].diff() == 0, data[f'ask_volume{i}'].diff(), data[f'ask_volume{i}']))
        dVWB = np.where(data[f'bid_price{i}'].diff() < 0, -data[f'bid_volume{i}'].shift(),
                       np.where(data[f'bid_price{i}'].diff() == 0, data[f'bid_volume{i}'].diff(), data[f'bid_volume{i}']))
        fac = dVWA - dVWB
        
        factor_df = pd.DataFrame(fac, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_OFI3(self, agg_method='mean'):
        """OFI3因子"""
        data = self.tick_data.copy(deep=False)
        i = 3
        dVWA = np.where(data[f'ask_price{i}'].diff() > 0, -data[f'ask_volume{i}'].shift(),
                       np.where(data[f'ask_price{i}'].diff() == 0, data[f'ask_volume{i}'].diff(), data[f'ask_volume{i}']))
        dVWB = np.where(data[f'bid_price{i}'].diff() < 0, -data[f'bid_volume{i}'].shift(),
                       np.where(data[f'bid_price{i}'].diff() == 0, data[f'bid_volume{i}'].diff(), data[f'bid_volume{i}']))
        fac = dVWA - dVWB
        
        factor_df = pd.DataFrame(fac, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_OFI4(self, agg_method='mean'):
        """OFI4因子"""
        data = self.tick_data.copy(deep=False)
        i = 4
        dVWA = np.where(data[f'ask_price{i}'].diff() > 0, -data[f'ask_volume{i}'].shift(),
                       np.where(data[f'ask_price{i}'].diff() == 0, data[f'ask_volume{i}'].diff(), data[f'ask_volume{i}']))
        dVWB = np.where(data[f'bid_price{i}'].diff() < 0, -data[f'bid_volume{i}'].shift(),
                       np.where(data[f'bid_price{i}'].diff() == 0, data[f'bid_volume{i}'].diff(), data[f'bid_volume{i}']))
        fac = dVWA - dVWB
        
        factor_df = pd.DataFrame(fac, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_OFI5(self, agg_method='mean'):
        """OFI5因子"""
        data = self.tick_data.copy(deep=False)
        i = 5
        dVWA = np.where(data[f'ask_price{i}'].diff() > 0, -data[f'ask_volume{i}'].shift(),
                       np.where(data[f'ask_price{i}'].diff() == 0, data[f'ask_volume{i}'].diff(), data[f'ask_volume{i}']))
        dVWB = np.where(data[f'bid_price{i}'].diff() < 0, -data[f'bid_volume{i}'].shift(),
                       np.where(data[f'bid_price{i}'].diff() == 0, data[f'bid_volume{i}'].diff(), data[f'bid_volume{i}']))
        fac = dVWA - dVWB
        
        factor_df = pd.DataFrame(fac, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_ILLIQ(self, agg_method='mean'):
        """ILLIQ因子"""
        data = self.tick_data.copy(deep=False)
        ILLIQ = data['mid_price'].diff() / data['TotalTradeValue'].diff() * 1e8
        fac = ILLIQ.rolling(window=20, min_periods=10).mean()
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_Hurst(self, agg_method='mean'):
        """Hurst因子"""
        data = self.tick_data.copy(deep=False)
        data['lma'] = data.mid_price - data['mid_price'].rolling(window=20).mean().values
        data['Z'] = data['lma'].rolling(window=20).sum()
        data['rn'] = data['Z'].rolling(window=20).max() - data['Z'].rolling(window=20).min()
        fac = data['rn'] / data['mid_price'].rolling(window=20).std()
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_Bolling(self, agg_method='mean'):
        """Bolling因子"""
        data = self.tick_data.copy(deep=False)

        data['M'] = data.mid_price
        fac = (data['M'] - data['M'].rolling(window=20).mean()) / (1+data['M'].rolling(window=20).std())
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_RSI(self, agg_method='mean', window=60):
        """RSI因子"""
        data = self.tick_data.copy(deep=False)
        delta = data['mid_price'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        fac = 100 - (100 / (1 + rs))
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_MACD(self, agg_method='mean'):
        """MACD因子"""
        data = self.tick_data.copy(deep=False)
        EMA12oi = self.wma(data['mid_price'].diff(), window=12)
        EMA26am = self.wma(data['mid_price'].diff(), window=26)
        data['MACD'] = EMA12oi - EMA26am
        fac = self.wma(data['MACD'], window=9)
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_Price_Divergence(self, agg_method='mean'):
        """Price_Divergence因子"""
        data = self.tick_data.copy(deep=False)
        high_diff = data['highest_price'].diff(30)
        low_diff = data['lowest_price'].diff(30)
        volume_diff_20 = data['TotalTradeVolume'].diff(30)
        volume_diff_60 = data['TotalTradeVolume'].diff(60)
        
        def strict_mask(series, min_val=0):
            masked = series.where(series >= min_val, np.nan)
            masked.replace([np.inf, -np.inf], np.nan, inplace=True)
            return masked
        
        high_diff = (strict_mask(high_diff) > 0).astype(int)
        low_diff = (strict_mask(-low_diff) > 0).astype(int)
        volume_diff_20 = strict_mask(volume_diff_20)
        volume_diff_60 = strict_mask(volume_diff_60)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            volume_ratio = volume_diff_60 / (volume_diff_20 + 1e-8)
            volume_ratio = strict_mask(volume_ratio)
        
        fac = (volume_ratio * (high_diff.astype(float) - low_diff.astype(float))).clip(-10, 10)
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_Momentum_Decay(self, agg_method='mean'):
        """Momentum_Decay因子"""
        data = self.tick_data.copy(deep=False)
        data['num'] = range(1, len(data) + 1)
        Slope = data['mid_price'].rolling(window=20).corr(data['num'])
        fac = np.sign(Slope) * (abs(Slope) - (abs(Slope).rolling(window=20).mean()))
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_Depth_Reversal(self, agg_method='mean', window=50):
        """Depth_Reversal因子"""
        data = self.tick_data.copy(deep=False)
        sbv = data[['bid_volume1', 'bid_volume2', 'bid_volume3', 'bid_volume4', 'bid_volume5']].sum(axis=1)
        sav = data[['ask_volume1', 'ask_volume2', 'ask_volume3', 'ask_volume4', 'ask_volume5']].sum(axis=1)
        spread = data['ask_price1'] - data['bid_price1']
        mask = (spread > spread.rolling(window).mean())
        m = data.mid_price
        fac = np.where(mask, sbv/sav, np.nan) * m.pct_change(window)
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_PRICE_VOL_CORR_A(self, agg_method='mean', window=20):
        """PRICE_VOL_CORR_A因子"""
        data = self.tick_data.copy(deep=False)
        fac = data['ask_price1'].rolling(window=window).corr(data['ask_volume1'])
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_PRICE_VOL_CORR_B(self, agg_method='mean', window=20):
        """PRICE_VOL_CORR_B因子"""
        data = self.tick_data.copy(deep=False)
        fac = data['bid_price1'].rolling(window=window).corr(data['bid_volume1'])
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_KDJ(self, agg_method='mean'):
        """KDJ因子"""
        data = self.tick_data.copy(deep=False)
        data['L_n'] = data['lowest_price'].rolling(window=20).min()
        data['H_n'] = data['highest_price'].rolling(window=20).max()
        data['RSV'] = (data['mid_price'] - data['L_n']) / (data['H_n'] - data['L_n']) * 100
        data['K'] = self.wma(data['RSV'], window=20)
        data['D'] = self.wma(data['K'], window=20)
        fac = 3 * data.K - 2 * data.D
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_MFI(self, agg_method='mean', period=14):
        """MFI因子"""
        data = self.tick_data.copy(deep=False)
        data['TP'] = (data['highest_price'] + data['lowest_price'] + data['mid_price']) / 3
        data['MF'] = data['TP'] * data['TotalTradeVolume'].diff()
        data['Positive_MF'] = np.where(data['TP'] > data['TP'].shift(1), data['MF'], 0)
        data['Negative_MF'] = np.where(data['TP'] <= data['TP'].shift(1), data['MF'], 0)
        data['Positive_MF'] = data['Positive_MF'].rolling(window=period).sum()
        data['Negative_MF'] = data['Negative_MF'].rolling(window=period).sum()
        fac = 100 - (100 / (1 + data['Positive_MF'] / data['Negative_MF']))
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_fibonacci_retracement(self, agg_method='mean'):
        """fibonacci_retracement因子"""
        data = self.tick_data.copy(deep=False)
        fibonacci_retracement = data['mid_price'] / ((data['mid_price'].rolling(window=20).max() - 
                                                    (data['mid_price'].rolling(window=20).max() - 
                                                     data['mid_price'].rolling(window=20).min())*0.5))
        fac = fibonacci_retracement - 1
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_LVC(self, agg_method='mean'):
        """LVC因子"""
        data = self.tick_data.copy(deep=False)
        LVC = (data[['bid_volume1', 'bid_volume2', 'bid_volume3', 'bid_volume4', 'bid_volume5']].sum(axis=1) * 
              data['mid_price'].rolling(window=20).std() / data['mid_price'].rolling(window=100).std())
        fac = LVC
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_tsi(self, agg_method='mean', long_period=25, short_period=13):
        """tsi因子"""
        data = self.tick_data.copy(deep=False)
        price_diff = data['mid_price'].diff()
        smoothed_price_diff = self.wma(price_diff, window=short_period)
        smoothed_abs_price_diff = self.wma(abs(price_diff), window=short_period)
        fac = 100 * self.wma(smoothed_price_diff, window=long_period) / self.wma(smoothed_abs_price_diff, window=long_period)
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_CMF(self, agg_method='mean', period=20):
        """CMF因子"""
        data = self.tick_data.copy(deep=False)
        typical_price = (data['highest_price'] + data['lowest_price'] + data['mid_price']) / 3
        volume_diff = data['TotalTradeVolume'].diff()
        volume_diff = volume_diff.where(volume_diff >= 0, np.nan)
        money_flow = typical_price * volume_diff
        numerator = money_flow.rolling(window=period, min_periods=15).mean()
        denominator = volume_diff.rolling(window=period, min_periods=15).mean()
        cmf = numerator / denominator.where(denominator > 0, np.nan)
        fac = cmf.pct_change(60 * 2) * 10000
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_RUSHSTOPASK(self, agg_method='mean'):
        """RUSHSTOPASK因子"""
        data = self.tick_data.copy(deep=False)
        fac = data.mid_price.pct_change(60 * 2) * data[['ask_price2', 'ask_price3', 'ask_price4', 'ask_price5']].max(axis=1).diff().rolling(window=4).mean()
        fac = fac.clip(-0.01, 0.01)
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    def FAC_RUSHSTOPBID(self, agg_method='mean'):
        """RUSHSTOPBID因子"""
        data = self.tick_data.copy(deep=False)
        fac = data.mid_price.pct_change(60 * 2) * data[['bid_price2', 'bid_price3', 'bid_price4', 'bid_price5']].max(axis=1).diff().rolling(window=4).mean()
        fac = fac.clip(-0.01, 0.01)
        
        factor_df = pd.DataFrame(fac.values, index=data.datetime, columns=['factor_value'])
        factor_df = self.resample_agg(factor_df, agg_method)
        return factor_df
    
    # -----------------------------------m2m因子生成-------------------------------------------------------------------
    def ASI(self, window=5):
        data = self.min_data.copy()
        A = (data.high - data.close.shift(1)).abs()
        B = (data.low - data.close.shift(1)).abs()
        C = (data.high - data.close.shift(1)).abs()
        D = (data.close.shift(1) - data.open.shift(1)).abs()
        E = data.close - data.close.shift(1)
        F = data.close - data.open
        G = data.close.shift(1) - data.open.shift(1)
        X = E + F/2 + G
        K = np.maximum(A, B)
        R = np.where((A>B)&(A>C), A+B/2+D/4, np.where((B>A)&(B>C), B+A/2+D/4, C+D/4))
        SI = 16 * X / R * K
        ASI = SI.rolling(window=window).sum()
        FAC = pd.DataFrame(ASI.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def BOLLING(self, window=5):
        data = self.min_data.copy()
        MA = data['close'].rolling(window=window).mean()
        STD = data['close'].rolling(window=window).std()
        FACTOR = (data['close'] - MA) / STD
        FAC = pd.DataFrame(FACTOR.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def MAdiff_Vol_div(self, window1=5, window2=20):
        data = self.min_data.copy()
        MAdiff = (data.close.rolling(window=window1).mean() - data.close.rolling(window=window2).mean()) / data.close
        FACTOR = MAdiff * data.open_interest.diff(window2)
        FAC = pd.DataFrame(FACTOR.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def drawback(self, window=30):
        data = self.min_data.copy()
        HIGH = data.high.rolling(window=window).max()
        LOW = data.low.rolling(window=window).min()
        ratio = (data.close - LOW) / (HIGH - LOW)
        th = 0.8
        FACTOR = np.where(ratio>th, ((HIGH - data.close) / HIGH), np.where(ratio<1-th, -((data.close-LOW) / LOW), np.nan))
        FAC = pd.DataFrame(FACTOR, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def PVcorr(self, window=16):
        data = self.min_data.copy()
        FACTOR = data.close.rolling(window=window).corr(data.volume) * data.close.pct_change(window)
        FAC = pd.DataFrame(FACTOR.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def KnifeReversal(self, window=5):
        data = self.min_data.copy()
        FACTOR = data.close.pct_change(window) * (1-data.volume / data.volume.rolling(window=window).mean())
        FAC = pd.DataFrame(FACTOR.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def IntradayEntropy(self, window1=5, window2=20):
        data = self.min_data.copy()
        FACTOR = (data.close - data.low.rolling(window=window1).min()) / (data.high.rolling(window=window1).max() - data.low.rolling(window=window1).min()) * data.close.diff(window2)
        FAC = pd.DataFrame(FACTOR.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def CapitalEfficiency(self, iqr_k=1.5, ret_window=10, rolling_window=60, th=0.75):
        data = self.min_data.copy()
        ret_close = data.close.pct_change(ret_window)
        ret_oi = data.open_interest.pct_change().rolling(ret_window).mean()
        factor = ret_close / ret_oi
        rolling_q1 = factor.shift(1).rolling(rolling_window, min_periods=10).quantile(1-th)  
        rolling_q3 = factor.shift(1).rolling(rolling_window, min_periods=10).quantile(th)
        rolling_iqr = rolling_q3 - rolling_q1
        lb = rolling_q1 - iqr_k * rolling_iqr
        ub = rolling_q3 + iqr_k * rolling_iqr
        FACTOR = factor.clip(lb, ub)
        FAC = pd.DataFrame(FACTOR.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def BuzzResonance(self, window1=5, window2=5, price_window=30):
        data = self.min_data.copy()
        mul1 = data.close.rolling(window=window1).corr(data.volume)
        mul2 = data.close.rolling(window=window2).corr(data.volume)
        FACTOR = - mul1 * mul2 * data.close.pct_change(price_window)
        FAC = pd.DataFrame(FACTOR.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def QST(self, window=16):
        data = self.min_data.copy()
        O = data.open.rolling(window=window).sum()
        C = data.close.rolling(window=window).sum()
        FACTOR = (C-O)
        FAC = pd.DataFrame(FACTOR.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def AR(self, window=4):
        data = self.min_data.copy()
        H = data.high.rolling(window=window).sum()
        L = data.low.rolling(window=window).sum()
        FACTOR = H - L
        FAC = pd.DataFrame(FACTOR.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def CMFmin(self, window=20):
        data = self.min_data.copy()
        TP = data.high/3 + data.close/3 + data.low/3
        CCI = (TP-TP.rolling(window=window).mean()) / (1+(TP-(TP.rolling(window=window).median())).abs())
        FACTOR = CCI
        FAC = pd.DataFrame(FACTOR.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def VRSI(self, window=10):
        data = self.min_data.copy()
        U = pd.Series(np.where(data.close.diff()>0, data.volume, np.where(data.close.diff()==0, data.volume/2, 0)))
        D = pd.Series(np.where(data.close.diff()<0, data.volume, np.where(data.close.diff()==0, data.volume/2, 0)))
        UU = (window-U.shift()+U)/window
        DD = (window-D.shift()+D)/window
        FACTOR = 100*DD/(UU+DD)
        FAC = pd.DataFrame(FACTOR.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def KVO(self, window1=5, window2=20):
        data = self.min_data.copy()
        data['VWAP'] = (data['volume'] * data['close']).rolling(window=window1).sum() / data['volume'].rolling(window=window1).sum()
        data['KVO_short'] = data['VWAP'].rolling(window=window1).mean()
        data['KVO_long'] = data['VWAP'].rolling(window=window2).mean()
        FACTOR = data['KVO_short'] - data['KVO_long']
        FAC = pd.DataFrame(FACTOR.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def DDI(self, window=20):
        data = self.min_data.copy()
        DN = data.high.diff() + data.low.diff()  
        DM = np.maximum(data.high.diff().abs(), data.low.diff().abs())  
        DMZ = pd.Series(np.where(DN <= 0, 0, DM), index=data.index)  
        DMF = pd.Series(np.where(DN > 0, 0, DM), index=data.index)   
        sum_DMZ = DMZ.rolling(window, min_periods=1).sum()
        sum_DMF = DMF.rolling(window, min_periods=1).sum()
        total_DM = sum_DMZ + sum_DMF
        total_DM = total_DM.replace(0, np.nan)
        DIZ = sum_DMZ / total_DM  
        DIF = sum_DMF / total_DM  
        FACTOR = DIZ - DIF
        FAC = pd.DataFrame(FACTOR.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def PV_corr_std(self, window=5):
        data = self.min_data.copy()
        CORR = data.close.rolling(window=window).corr(data.volume)
        FACTOR = CORR.rolling(window=window).std()
        FAC = pd.DataFrame(FACTOR.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def TS(self, window_ret=2, window_sum=14):
        data = self.min_data.copy()
        TS = np.where(data.close.rolling(window=window_ret).mean().diff()>0, 1, -1)
        FACTOR = pd.Series(TS).rolling(window=window_sum).sum()
        FAC = pd.DataFrame(FACTOR.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def VHF(self, window=10):
        data = self.min_data.copy()
        TR = np.maximum(data.high-data.low, np.maximum((data.high-data.close.shift()).abs(), (data.low-data.close.shift()).abs()))
        FACTOR = pd.Series(TR).rolling(window=window).mean()
        FAC = pd.DataFrame(FACTOR.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def RPP_5D(self, window=60 * 4 * 5):
        data = self.min_data.copy()
        H = data.high.rolling(window=window).max()
        L = data.low.rolling(window=window).min()
        FACTOR = (data.close - L) / (H - L)
        FAC = pd.DataFrame(FACTOR.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def RPP_22D(self, window=60 * 4 * 22):
        data = self.min_data.copy()
        H = data.high.rolling(window=window).max()
        L = data.low.rolling(window=window).min()
        RPP = (data.close - L) / (H - L)
        FACTOR = RPP
        FAC = pd.DataFrame(FACTOR.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def RPP_4H(self, window=60 * 4):
        data = self.min_data.copy()
        H = data.high.rolling(window=window).max()
        L = data.low.rolling(window=window).min()
        RPP = (data.close - L) / (H - L)
        FACTOR = RPP
        FAC = pd.DataFrame(FACTOR.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def RPP_2H(self, window=60 * 2):
        data = self.min_data.copy()
        H = data.high.rolling(window=window).max()
        L = data.low.rolling(window=window).min()
        RPP = (data.close - L) / (H - L)
        FACTOR = RPP
        FAC = pd.DataFrame(FACTOR.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def RPP_30M(self, window=30):
        data = self.min_data.copy()
        H = data.high.rolling(window=window).max()
        L = data.low.rolling(window=window).min()
        RPP = (data.close - L) / (H - L)
        FACTOR = RPP
        FAC = pd.DataFrame(FACTOR.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def RPP_10M(self, window=10):
        data = self.min_data.copy()
        H = data.high.rolling(window=window).max()
        L = data.low.rolling(window=window).min()
        RPP = (data.close - L) / (H - L)
        FACTOR = RPP
        FAC = pd.DataFrame(FACTOR.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def RPP_5M(self, window=5):
        data = self.min_data.copy()
        H = data.high.rolling(window=window).max()
        L = data.low.rolling(window=window).min()
        RPP = (data.close - L) / (H - L)
        FACTOR = RPP
        FAC = pd.DataFrame(FACTOR.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def RPP_3M(self, window=3):
        data = self.min_data.copy()
        H = data.high.rolling(window=window).max()
        L = data.low.rolling(window=window).min()
        RPP = (data.close - L) / (H - L)
        FACTOR = RPP
        FAC = pd.DataFrame(FACTOR.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def XSMOM1M(self, window=60 * 4 * 22):
        data = self.min_data.copy()
        FACTOR = data.close.pct_change(window)
        FAC = pd.DataFrame(FACTOR.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def CCI(self, window=120):
        data = self.min_data.copy()
        HLC = data.high/3 + data.low/3 + data.close/3
        HLCma = HLC.rolling(window=window).sum() / window
        AVEDEV = (HLC - HLCma).abs().rolling(window=window).sum() / window
        CCI = (HLC - HLCma) / (0.015*AVEDEV + 1)
        FACTOR = CCI
        FAC = pd.DataFrame(FACTOR.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def HP(self, window=20):
        data = self.min_data.copy()
        FACTOR = data.open_interest.diff(window) / data.volume.rolling(window=window).sum()
        FAC = pd.DataFrame(FACTOR.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def SNR(self, window=10):
        data = self.min_data.copy()
        FACTOR = np.log(data.close / data.open).abs().rolling(window=window).sum()
        FAC = pd.DataFrame(FACTOR.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def dOI(self, iqr_k=1.5, window=20, rolling_window=20, th=0.75):
        data = self.min_data.copy()
        absRetnight = np.log(data.open_interest) - np.log(data.open_interest.shift(window))
        factor = absRetnight  
        rolling_q1 = factor.shift(1).rolling(rolling_window).quantile(1-th)
        rolling_q3 = factor.shift(1).rolling(rolling_window).quantile(th)
        rolling_iqr = rolling_q3 - rolling_q1
        lb = rolling_q1 - iqr_k * rolling_iqr
        ub = rolling_q3 + iqr_k * rolling_iqr
        FACTOR = factor.clip(lb, ub)
        FAC = pd.DataFrame(FACTOR.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def RSImin(self, window=20):
        data = self.min_data.copy()
        FACTOR = pd.Series(np.maximum(data.close.diff(), 0)).rolling(window=window).mean() / data.close.diff().abs().rolling(window=window).mean()
        FAC = pd.DataFrame(FACTOR.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def CR(self, window=240):
        data = self.min_data.copy()
        HLCC = data.high/3 + data.low/3 + data.close/3
        CR1 = pd.Series(np.maximum(0, data.high - HLCC)).rolling(window=window).mean()
        CR2 = pd.Series(np.maximum(0, HLCC - data.low)).rolling(window=window).mean()
        FACTOR = CR1 / CR2
        FAC = pd.DataFrame(FACTOR.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def CV(self, window=40, iqr_k=1.5, rolling_window=60, th=0.75):
        data = self.min_data.copy()
        factor = data.close.pct_change().rolling(window=window).std() / data.close.pct_change().rolling(window=window).mean().abs()
        rolling_q1 = factor.shift(1).rolling(rolling_window).quantile(1-th)  
        rolling_q3 = factor.shift(1).rolling(rolling_window).quantile(th)
        rolling_iqr = rolling_q3 - rolling_q1
        lb = rolling_q1 - iqr_k * rolling_iqr
        ub = rolling_q3 + iqr_k * rolling_iqr
        FACTOR = factor.clip(lb, ub)
        FAC = pd.DataFrame(FACTOR.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def LR(self, window=60):
        data = self.min_data.copy()
        FACTOR = (data.close.diff().abs() / data.volume).rolling(window=window).mean()
        FACTOR = (FACTOR-FACTOR.rolling(window=window).mean()) / FACTOR.rolling(window=window).std()
        FAC = pd.DataFrame(FACTOR.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def signmom(self, window=60):
        data = self.min_data.copy()
        sign = pd.Series(np.maximum(0, np.sign(data.close.diff()))).rolling(window=window).mean()
        FACTOR = sign
        FAC = pd.DataFrame(FACTOR.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def VR(self, window=240):
        data = self.min_data.copy()
        FACTOR = pd.Series(np.where(data.close.diff()>0, data.volume, 0)).rolling(window=window).sum() / pd.Series(np.where(data.close.diff()<=0, data.volume, 0)).rolling(window=window).sum()
        FACTOR = (FACTOR-FACTOR.rolling(window=window).mean()) / FACTOR.rolling(window=window).std()
        FAC = pd.DataFrame(FACTOR.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def Chande(self, window=30):
        data = self.min_data.copy()
        SU = pd.Series(np.where(data.close.diff()>0, data.close.diff(), 0)).rolling(window=window).sum()
        SD = pd.Series(np.where(data.close.diff()<0, -data.close.diff(), 0)).rolling(window=window).sum()
        FACTOR = (SU-SD) / (SU+SD) * 100
        FAC = pd.DataFrame(FACTOR.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def masign(self, window=60):
        data = self.min_data.copy()
        tmp = np.zeros(data.shape[0])
        for i in range(1, window):
            mean_i = data.close.rolling(window=i).mean()
            mean_i_plus_1 = data.close.rolling(window=i+1).mean()
            tmp += np.sign(mean_i - mean_i_plus_1).fillna(0)  
        FACTOR = tmp
        FAC = pd.DataFrame(FACTOR.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def ATR(self, window=60):
        data = self.min_data.copy()
        TR = np.maximum((data.high-data.close.shift()).abs(), np.maximum((data.low-data.close.shift()).abs(), data.high - data.close))
        FACTOR = pd.Series(TR).rolling(window=window).mean()
        FACTOR = (FACTOR-FACTOR.rolling(window=window).mean()) / FACTOR.rolling(window=window).std()
        FAC = pd.DataFrame(FACTOR.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def skew(self, window=60, iqr_k=1.5, rolling_window=20, th=0.75):
        data = self.min_data.copy()
        factor = (((data.close.pct_change() - data.close.pct_change().shift().rolling(window=window).mean()) / data.close.pct_change().shift().rolling(window=window).std()) ** 3)
        rolling_q1 = factor.shift(1).rolling(rolling_window).quantile(1-th)
        rolling_q3 = factor.shift(1).rolling(rolling_window).quantile(th)
        rolling_iqr = rolling_q3 - rolling_q1
        lb = rolling_q1 - iqr_k * rolling_iqr
        ub = rolling_q3 + iqr_k * rolling_iqr
        FACTOR = factor.clip(lb, ub)
        FAC = pd.DataFrame(FACTOR.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FACTOR
    
    def trendstrength(self, window=20):
        data = self.min_data.copy()
        FACTOR = data.close.diff(window) / (data.close.diff().abs()).rolling(window=window).sum()
        FAC = pd.DataFrame(FACTOR.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def DDI(self, window=20):
        data = self.min_data.copy()
        DMZ = np.where(data.high.diff()+data.low.diff()<=0, 0, np.maximum(data.high.diff().abs(), data.low.diff().abs()))
        DMF = np.where(data.high.diff()+data.low.diff()>0, 0, np.maximum(data.high.diff().abs(), data.low.diff().abs()))
        DIZ = pd.Series(DMZ).rolling(window=window).sum() / (pd.Series(DMZ).rolling(window=window).sum() + pd.Series(DMF).rolling(window=window).sum())
        DIF = pd.Series(DMF).rolling(window=window).sum() / (pd.Series(DMZ).rolling(window=window).sum() + pd.Series(DMF).rolling(window=window).sum())
        FACTOR = DIZ - DIF
        FAC = pd.DataFrame(FACTOR.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def kurt(self, window=20, iqr_k=3, rolling_window=60):
        data = self.min_data.copy()
        factor = (((data.close.pct_change() - data.close.pct_change().shift().rolling(window=window).mean()) / data.close.pct_change().shift().rolling(window=window).std()) ** 4)
        rolling_q1 = factor.shift(1).rolling(rolling_window).quantile(0.25)
        rolling_q3 = factor.shift(1).rolling(rolling_window).quantile(0.75)
        rolling_iqr = rolling_q3 - rolling_q1
        lb = rolling_q1 - iqr_k * rolling_iqr
        ub = rolling_q3 + iqr_k * rolling_iqr
        factor = factor.clip(lb, ub) 
        factor = (factor-factor.rolling(window=window).mean()) / factor.rolling(window=window).std()
        FAC = pd.DataFrame(factor.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def ACD(self, window=30, iqr_k=3, rolling_window=60):
        data = self.min_data.copy()
        DIF = np.where(data.close.diff()>0, np.minimum(data.low, data.close.shift()), np.maximum(data.high, data.close.shift()))
        factor = data.close.diff().values * pd.Series(np.where(data.close.diff()==0, 0, DIF)).rolling(window=window).sum().diff()
        rolling_q1 = factor.shift(1).rolling(rolling_window).quantile(0.25)
        rolling_q3 = factor.shift(1).rolling(rolling_window).quantile(0.75)
        rolling_iqr = rolling_q3 - rolling_q1
        lb = rolling_q1 - iqr_k * rolling_iqr
        ub = rolling_q3 + iqr_k * rolling_iqr
        factor = factor.clip(lb, ub) 
        factor = (factor-factor.rolling(window=window).mean()) / factor.rolling(window=window).std()
        FAC = pd.DataFrame(factor.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def up_shadow_5mean(self, window=5):
        data = self.min_data.copy()
        up_shadow = (data.high - np.maximum(data.open, data.close)) / (np.maximum(data.open, data.close) - np.minimum(data.open, data.close) + 1)
        FACTOR = up_shadow.rolling(window=window).mean()
        FAC = pd.DataFrame(FACTOR.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def down_shadow_5mean(self, window=5):
        data = self.min_data.copy()
        down_shadow = -(data.low - np.minimum(data.open, data.close)) / (np.maximum(data.open, data.close) - np.minimum(data.open, data.close) + 1)
        FACTOR = down_shadow.rolling(window=window).mean()
        FAC = pd.DataFrame(FACTOR.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def up_shadow_5std(self, window=5):
        data = self.min_data.copy()
        up_shadow = (data.high - np.maximum(data.open, data.close)) / (np.maximum(data.open, data.close) - np.minimum(data.open, data.close) + 1)
        FACTOR = up_shadow.rolling(window=window).std()
        FAC = pd.DataFrame(FACTOR.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def down_shadow_5std(self, window=5):
        data = self.min_data.copy()
        down_shadow = -(data.low - np.minimum(data.open, data.close)) / (np.maximum(data.open, data.close) - np.minimum(data.open, data.close) + 1)
        FACTOR = down_shadow.rolling(window=window).std()
        FAC = pd.DataFrame(FACTOR.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    @_truncate_by_trade_date
    def day_jump(self):
        data = self.min_data.copy()
        def day_open(group):
            if group.shape[0]<3:
                return np.nan
            else:
                first_price = group.open.iloc[0]
                return first_price
        
        def day_close(group):
            if group.shape[0]<3:
                return np.nan
            else:
                last_price = group.close.iloc[-3:].mean()
                return last_price
        jump_today = data.groupby('trade_date').apply(day_open) - data.groupby('trade_date').apply(day_close).shift()
        FACTOR = jump_today.reset_index(name='fac').merge(data, on='trade_date',how='right').fac
        FAC = pd.DataFrame(FACTOR.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def day_first3power(self, use_info_bar=3):
        data = self.min_data.copy()
        def calc_fac(group):
            if len(group)<use_info_bar:
                return np.nan
            else:
                first_price = (group.close - group.open) 
                return first_price.iloc[:use_info_bar].mean()
        FACTOR = data.groupby('trade_date').apply(calc_fac).reset_index(name='fac').merge(data, on='trade_date',how='right')
        def mask_first(group, use_info_bar):
            group.loc[group.index[:use_info_bar], 'fac'] = np.nan  
            return group
        FACTOR = FACTOR.groupby('trade_date', group_keys=False).apply(lambda group: mask_first(group,use_info_bar))
        FAC = pd.DataFrame(FACTOR.fac.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def day_first6power(self, use_info_bar=6):
        data = self.min_data.copy()
        def calc_fac(group):
            if len(group)<use_info_bar:
                return np.nan
            else:
                first_price = (group.close - group.open) 
                return first_price.iloc[:use_info_bar].mean()
        FACTOR = data.groupby('trade_date').apply(calc_fac).reset_index(name='fac').merge(data, on='trade_date',how='right')
        def mask_first(group, use_info_bar):
            group.loc[group.index[:use_info_bar], 'fac'] = np.nan  
            return group
        FACTOR = FACTOR.groupby('trade_date', group_keys=False).apply(lambda group: mask_first(group,use_info_bar))
        FAC = pd.DataFrame(FACTOR.fac.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def day_first10power(self, use_info_bar=10):
        data = self.min_data.copy()
        def calc_fac(group):
            if len(group)<use_info_bar:
                return np.nan
            else:
                first_price = (group.close - group.open) 
                return first_price.iloc[:use_info_bar].mean()
        FACTOR = data.groupby('trade_date').apply(calc_fac).reset_index(name='fac').merge(data, on='trade_date',how='right')
        def mask_first(group, use_info_bar):
            group.loc[group.index[:use_info_bar], 'fac'] = np.nan  
            return group
        FACTOR = FACTOR.groupby('trade_date', group_keys=False).apply(lambda group: mask_first(group,use_info_bar))
        FAC = pd.DataFrame(FACTOR.fac.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def day_first10colarrate(self, use_info_bar=10):
        data = self.min_data.copy()
        def calc_fac(group):
            if len(group)<use_info_bar:
                return np.nan
            else:
                first_N = group.head(use_info_bar)
                is_positive = (first_N.close > first_N.open)
                positive_ratio = is_positive.mean()
                return positive_ratio
        FACTOR = data.groupby('trade_date').apply(calc_fac).reset_index(name='fac').merge(data, on='trade_date',how='right')
        def mask_first(group, use_info_bar):
            group.loc[group.index[:use_info_bar], 'fac'] = np.nan  
            return group
        FACTOR = FACTOR.groupby('trade_date', group_keys=False).apply(lambda group: mask_first(group,use_info_bar))
        FAC = pd.DataFrame(FACTOR.fac.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def day_first10rev(self, use_info_bar=10):
        data = self.min_data.copy()
        def calc_fac(group):
            if len(group)<use_info_bar:
                return np.nan
            else:
                first_N = group.head(use_info_bar)
                if len(first_N) < use_info_bar:
                    return np.nan
                max_up = 0
                max_down = 0
                max_close = first_N.close.iloc[0]
                min_close = first_N.close.iloc[0]
                for i in range(1, len(first_N)):
                    if first_N.close.iloc[i] > max_close:
                        max_close = first_N.close.iloc[i]
                    if first_N.close.iloc[i] < min_close:
                        min_close = first_N.close.iloc[i]
                    max_up = max(max_up, first_N.close.iloc[i] - min_close)
                    max_down = min(max_down, first_N.close.iloc[i] - max_close)
                factor = np.log(abs(max_up * max_down) / (first_N.close.mean()) ** 2 * 1e6 + 1)
                return factor
        FACTOR = data.groupby('trade_date').apply(calc_fac).reset_index(name='fac').merge(data, on='trade_date',how='right')
        def mask_first(group, use_info_bar):
            group.loc[group.index[:use_info_bar], 'fac'] = np.nan  
            return group
        FACTOR = FACTOR.groupby('trade_date', group_keys=False).apply(lambda group: mask_first(group,use_info_bar))
        FAC = pd.DataFrame(FACTOR.fac.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def day_first4redcorr(self, use_info_bar=4):
        data = self.min_data.copy()
        def calc_fac(group):
            if group.shape[0]<use_info_bar:
                return np.nan
            else:
                first_price = np.maximum(group.close - group.open,0).iloc[:use_info_bar].corr(group.volume.iloc[:use_info_bar])
                return first_price
        FACTOR = data.groupby('trade_date').apply(calc_fac).reset_index(name='fac').merge(data, on='trade_date',how='right')
        def mask_first(group, use_info_bar):
            group.loc[group.index[:use_info_bar], 'fac'] = np.nan 
            return group
        FACTOR = FACTOR.groupby('trade_date', group_keys=False).apply(lambda group: mask_first(group,use_info_bar))
        FAC = pd.DataFrame(FACTOR.fac.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def day_first4greencorr(self, use_info_bar=4):
        data = self.min_data.copy()
        def calc_fac(group):
            if group.shape[0]<use_info_bar:
                return np.nan
            else:
                first_price = -np.minimum(group.close - group.open,0).iloc[:use_info_bar].corr(group.volume.iloc[:use_info_bar])
                return first_price
        FACTOR = data.groupby('trade_date').apply(calc_fac).reset_index(name='fac').merge(data, on='trade_date',how='right')
        def mask_first(group, use_info_bar):
            group.loc[group.index[:use_info_bar], 'fac'] = np.nan  
            return group
        FACTOR = FACTOR.groupby('trade_date', group_keys=False).apply(lambda group: mask_first(group,use_info_bar))
        FAC = pd.DataFrame(FACTOR.fac.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def OFI5(self, window=5, iqr_k=3, rolling_window=60):
        data = self.min_data.copy()
        factor = (data.buy_volume.rolling(window=window).sum() - data.sell_volume.rolling(window=window).sum()) * data.close.pct_change(window)
        rolling_q1 = factor.shift(1).rolling(rolling_window, min_periods=10).quantile(0.25)
        rolling_q3 = factor.shift(1).rolling(rolling_window, min_periods=10).quantile(0.75)
        rolling_iqr = rolling_q3 - rolling_q1
        lb = rolling_q1 - iqr_k * rolling_iqr
        ub = rolling_q3 + iqr_k * rolling_iqr
        factor = factor.clip(lb, ub) 
        factor = (factor-factor.rolling(window=window).mean()) / (0.001 + factor.rolling(window=window).std())
        FAC = pd.DataFrame(factor.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def OFI60(self, window=60, iqr_k=3, rolling_window=60):
        data = self.min_data.copy()
        factor = (data.buy_volume.rolling(window=window).sum() - data.sell_volume.rolling(window=window).sum()) * data.close.pct_change(window)
        rolling_q1 = factor.shift(1).rolling(rolling_window, min_periods=10).quantile(0.25)
        rolling_q3 = factor.shift(1).rolling(rolling_window, min_periods=10).quantile(0.75)
        rolling_iqr = rolling_q3 - rolling_q1
        lb = rolling_q1 - iqr_k * rolling_iqr
        ub = rolling_q3 + iqr_k * rolling_iqr
        factor = factor.clip(lb, ub) 
        factor = (factor-factor.rolling(window=window).mean()) / (0.001 + factor.rolling(window=window).std())
        FAC = pd.DataFrame(factor.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    @_truncate_by_trade_date
    def zigzag(self):
        data = self.min_data.copy()
        data = data.sort_values(['datetime', 'instrument'])
        data['count_bar'] = 1
        data['count_bar'] = data.groupby(['instrument', 'trade_date'])['count_bar'].cumsum()
        data['prev_close'] = data.groupby('instrument')['close'].shift(1)
        
        def rolling_percentile(s, p, window, min_periods):
            return s.rolling(window, min_periods=min_periods).quantile(p/100)
        
        data['volatility_rg'] = data.groupby('instrument')['close'].transform(
            lambda x: (rolling_percentile(x, 95, 240 * 5, 240 * 2) - 
                       rolling_percentile(x, 5, 240 * 5, 240 * 2)) / 24.0
        )
        
        data['pre_vol'] = data.groupby('instrument')['volatility_rg'].shift(1)
        data['pre_close'] = data.groupby('instrument')['close'].shift(1)
        data['pre2_close'] = data.groupby('instrument')['close'].shift(2)
        data['pre_turnover'] = data.groupby('instrument')['turnover'].shift(1)
        
        def price_filter_f5(df):
            results = []
            for (instrument, date), group in df.groupby(['instrument', 'trade_date']):
                pre_close = group['pre_close'].values
                pre2_close = group['pre2_close'].values
                volatility = group['volatility_rg'].values
                
                plist = []
                plast = []
                
                for i in range(len(group)):
                    if pd.isna(pre2_close[i]):
                        plast.append(np.nan)
                        continue
                        
                    if not plist:
                        plist.extend([pre2_close[i], pre_close[i]])
                        plast.append(np.nan)
                        continue
                        
                    vol_use = 3 * volatility[i]
                    last_p, prev_p = plist[-1], plist[-2]
                    
                    if (pre_close[i] - last_p) * (last_p - prev_p) > 0:
                        plist[-1] = pre_close[i]
                    elif abs(pre_close[i] - last_p) > vol_use:
                        plist.append(pre_close[i])
                        
                    plast.append(plist[-2] if len(plist) >= 2 else np.nan)
                
                group['plast'] = plast
                results.append(group)
            
            return pd.concat(results)['plast']
        
        data['plast'] = price_filter_f5(data)
        factor = (data['close'] - data['plast']) / data['volatility_rg']
        FAC = pd.DataFrame(factor.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def volatility_rg(self):
        data = self.min_data.copy()
        data = data.sort_values(['datetime', 'instrument'])
        data['count_bar'] = 1
        data['count_bar'] = data.groupby(['instrument', 'trade_date'])['count_bar'].cumsum()
        data['prev_close'] = data.groupby('instrument')['close'].shift(1)
        
        def rolling_percentile(s, p, window, min_periods):
            return s.rolling(window, min_periods=min_periods).quantile(p/100)
        
        factor = data.groupby(['instrument'])['close'].transform(
            lambda x: (rolling_percentile(x, 95, 15, 10) - 
                       rolling_percentile(x, 5, 15, 10)) / 24.0
        )
        FAC = pd.DataFrame(factor.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def before_bot_price_diff(self):
        data = self.min_data.copy()
        def _calculate_group(group):
            close = group['close'].values
            volume = group['volume'].values
            n = len(close)
            if n < 4 or np.all(volume <= 0):
                return np.full(n, np.nan)
            valley_pos = np.zeros(n, dtype=int)
            for i in range(1, n):
                mask = volume[:i] > 0
                if np.any(mask):
                    valley_pos[i] = np.argmin(volume[:i][mask])
                else:
                    valley_pos[i] = i
            starts = np.maximum(0, valley_pos - 3)
            means = np.array([
                np.mean(close[s:i]) if i > s else np.nan
                for s, i in zip(starts, valley_pos)
            ])
            global_means = np.cumsum(close) / np.arange(1, n+1)
            return means / global_means - 1.0
        
        factor = data.groupby('trade_date', group_keys=False).apply(
            lambda g: pd.Series(_calculate_group(g), index=g.index)
        )
        # 多线程下 apply 可能返回 DataFrame，统一处理为 Series
        if isinstance(factor, pd.DataFrame):
            factor = factor.squeeze()
        FAC = pd.DataFrame({'factor_value': factor.values}, index=data.datetime).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    @_truncate_by_trade_date
    def bar3_trend_corr(self):
        data = self.min_data.copy()
        factor = _rolling_corr_3(data['close'].values)
        FAC = pd.DataFrame(factor, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    @_truncate_by_trade_date
    def bar5_trend_corr(self):
        data = self.min_data.copy()
        factor = _rolling_corr_5(data['close'].values)
        FAC = pd.DataFrame(factor, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index).values.reshape(-1,)
        return FAC
    
    def hour(self):

        data = self.min_data.copy()

        FAC = data.datetime.dt.hour

        FAC = pd.DataFrame(FAC.values, index=data.datetime, columns=['factor_value']).reindex(index=self.valid_index) 
        return FAC.values.flatten() 
    
    def buy_trend(self):
        """
        买盘趋势因子
        5档买盘总量 > 5档卖盘总量的时间占比
        """
        tick_data = self.tick_data.copy(deep=False)
        
        def calc_buy_tend(group):
            if group.shape[0] < 5:
                return np.nan
            else:
                # 计算5档买盘和卖盘总量
                buy_amount = (group['bid_volume1'] + group['bid_volume2'] + group['bid_volume3'] + 
                             group['bid_volume4'] + group['bid_volume5'])
                sell_amount = (group['ask_volume1'] + group['ask_volume2'] + group['ask_volume3'] + 
                              group['ask_volume4'] + group['ask_volume5'])
                return (buy_amount > sell_amount).mean()
        
        # 按分钟分组计算
        FAC = tick_data.set_index(['datetime']).resample('1min', label='right', closed='right').apply(calc_buy_tend)
        FAC = FAC.reindex(self.valid_index)
        FAC = pd.DataFrame(FAC.values, index=self.valid_index, columns=['factor_value'])
        return FAC.values.flatten()
    
    def buy_trend1(self):
        """
        买盘趋势因子（1档）
        1档买盘量 > 1档卖盘量的时间占比
        """
        tick_data = self.tick_data.copy(deep=False)
        
        def calc_fac(group):
            if group.shape[0] < 5:
                return np.nan
            else:
                buy_amount = group['bid_volume1']
                sell_amount = group['ask_volume1']
                return (buy_amount > sell_amount).mean()
        
        FAC = tick_data.set_index(['datetime']).resample('1min', label='right', closed='right').apply(calc_fac)
        FAC = FAC.reindex(self.valid_index)
        FAC = pd.DataFrame(FAC.values, index=self.valid_index, columns=['factor_value'])
        return FAC.values.flatten()
    
    def buy_trend_rol10(self):
        """
        滚动买盘趋势因子
        1档买盘10秒滚动总量 > 1档卖盘10秒滚动总量的时间占比
        """
        tick_data = self.tick_data.copy(deep=False)
        
        def calc_fac(group):
            if group.shape[0] < 5:
                return np.nan
            else:
                buy_amount = group['bid_volume1'].rolling(window=10).sum()
                sell_amount = group['ask_volume1'].rolling(window=10).sum()
                return (buy_amount > sell_amount).mean()
        
        FAC = tick_data.set_index(['datetime']).resample('1min', label='right', closed='right').apply(calc_fac)
        FAC = FAC.reindex(self.valid_index)
        FAC = pd.DataFrame(FAC.values, index=self.valid_index, columns=['factor_value'])
        return FAC.values.flatten()
    
    def lastprice_bias1(self):
        """
        最新价偏置因子
        基于VWAP与买卖一价的距离判断价格偏向
        """
        tick_data = self.tick_data.copy(deep=False)
        multiplier = self.main_multiplier
        
        def calc_fac(group):
            if group.shape[0] < 5:
                return np.nan
            else:
                # 计算VWAP
                group['vwap'] = (group['turnover'] / group['volume']) / multiplier
                
                # 计算VWAP与买卖一价的绝对距离
                buy_amount = (group['vwap'] - group['ask_price1']).abs().rolling(window=1).sum()
                sell_amount = (group['vwap'] - group['bid_price1']).abs().rolling(window=1).sum()
                
                return (buy_amount > sell_amount).mean()
        
        FAC = tick_data.set_index(['datetime']).resample('1min', label='right', closed='right').apply(calc_fac)
        FAC = FAC.reindex(self.valid_index)
        FAC = pd.DataFrame(FAC.values, index=self.valid_index, columns=['factor_value'])
        return FAC.values.flatten()
    
    def tickvol10(self, window=20, std_window=60, min_periods=50):
        """
        tick波动率因子（20tick窗口）
        基于中间价变化的已实现波动率
        """
        tick_data = self.tick_data.copy(deep=False)
        
        def calc_fac(group):
            if group.shape[0] < 5:
                return np.nan
            else:
                # 计算已实现波动率
                realized_vol = group.mid_price.diff().apply(lambda x: x*x).rolling(window=window).sum()
                return realized_vol.mean()
        
        # 计算原始因子
        FAC_raw = tick_data.set_index(['datetime']).resample('1min', label='right', closed='right').apply(calc_fac)
        FAC_raw = FAC_raw.reindex(self.valid_index)
        
        # 标准化
        FAC_std = (FAC_raw - FAC_raw.rolling(window=std_window, min_periods=min_periods).mean()
                  ).div(FAC_raw.rolling(window=std_window, min_periods=min_periods).std())
        
        # 处理无穷值
        FAC_std = FAC_std.replace([-np.inf, np.inf], np.nan)
        
        FAC = pd.DataFrame(FAC_std.values, index=self.valid_index, columns=['factor_value'])
        return FAC.values.flatten()
    
    def tickvol20(self, window=10, std_window=60, min_periods=50):
        """
        tick波动率因子（10tick窗口）
        基于中间价变化的已实现波动率
        """
        tick_data = self.tick_data.copy(deep=False)
        
        def calc_fac(group):
            if group.shape[0] < 5:
                return np.nan
            else:
                # 计算已实现波动率
                realized_vol = group.mid_price.diff().apply(lambda x: x*x).rolling(window=window).sum()
                return realized_vol.mean()
        
        # 计算原始因子
        FAC_raw = tick_data.set_index(['datetime']).resample('1min', label='right', closed='right').apply(calc_fac)
        FAC_raw = FAC_raw.reindex(self.valid_index)
        
        # 标准化
        FAC_std = (FAC_raw - FAC_raw.rolling(window=std_window, min_periods=min_periods).mean()
                  ).div(FAC_raw.rolling(window=std_window, min_periods=min_periods).std())
        
        # 处理无穷值
        FAC_std = FAC_std.replace([-np.inf, np.inf], np.nan)
        
        FAC = pd.DataFrame(FAC_std.values, index=self.valid_index, columns=['factor_value'])
        return FAC.values.flatten()
    
    def TMB(self):
        """
        趋势动量偏度因子
        基于多项式回归的趋势斜率与拟合优度的乘积
        """
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.metrics import r2_score
        
        tick_data = self.tick_data.copy(deep=False)
        
        def calc_fac(group):
            if group.shape[0] < 5:
                return np.nan
            else:
                M = group['mid_price'].values
                X = np.arange(len(M)).reshape(-1, 1)
                
                # 二阶多项式特征
                poly = PolynomialFeatures(degree=2)
                X_poly = poly.fit_transform(X)
                
                # 有效数据掩码
                valid_mask = ~np.isnan(M)
                if valid_mask.sum() < 30:
                    return np.nan
                
                # 多项式回归
                model = LinearRegression()
                model.fit(X_poly[valid_mask], M[valid_mask])
                
                # 预测
                y_pred = model.predict(X_poly)
                
                # 线性系数和R方
                linear_coefficient = model.coef_[1]
                r_squared = r2_score(M[valid_mask], y_pred[valid_mask])
                
                return linear_coefficient * r_squared
        
        FAC = tick_data.set_index(['datetime']).resample('1min', label='right', closed='right').apply(calc_fac)
        FAC = FAC.reindex(self.valid_index)
        FAC = pd.DataFrame(FAC.values, index=self.valid_index, columns=['factor_value'])
        return FAC.values.flatten()
    
    def inflectionpoint(self):
        """
        拐点检测因子
        基于中间价符号变化的拐点数量
        """
        tick_data = self.tick_data.copy(deep=False)
        
        def calc_fac(group):
            if group.shape[0] < 5:
                return np.nan
            else:
                # 计算中间价
                m = (group['ask_price1']/2 + group['bid_price1']/2).fillna(method='ffill')
                
                # 计算符号变化
                sign_series = pd.Series(np.sign(m.diff()))
                sign_series = sign_series.replace(0, np.nan).fillna(method='ffill')
                
                # 检测拐点
                point = sign_series.diff().abs()
                
                return point.sum()
        
        FAC = tick_data.set_index(['datetime']).resample('1min', label='right', closed='right').apply(calc_fac)
        FAC = FAC.reindex(self.valid_index)
        FAC = pd.DataFrame(FAC.values, index=self.valid_index, columns=['factor_value'])
        return FAC.values.flatten()
    
    def inflectionpoint5(self, diff_window=5):
        """
        拐点检测因子（5期差分）
        基于中间价5期差分符号变化的拐点数量
        """
        tick_data = self.tick_data.copy(deep=False)
        
        def calc_fac(group):
            if group.shape[0] < 5:
                return np.nan
            else:
                # 计算中间价
                m = (group['ask_price1']/2 + group['bid_price1']/2).fillna(method='ffill')
                
                # 计算5期差分的符号变化
                sign_series = pd.Series(np.sign(m.diff(diff_window)))
                sign_series = sign_series.replace(0, np.nan).fillna(method='ffill')
                
                # 检测拐点
                point = sign_series.diff().abs()
                
                return point.sum()
        
        FAC = tick_data.set_index(['datetime']).resample('1min', label='right', closed='right').apply(calc_fac)
        FAC = FAC.reindex(self.valid_index)
        FAC = pd.DataFrame(FAC.values, index=self.valid_index, columns=['factor_value'])
        return FAC.values.flatten()

    # -----------------------------------跨所因子生成------------------------------------------------------------------
    def cvcorr10_diff(self, main_symbol, symbol1, symbol2, window=10):
        data1 = self._symbol_data_indexed.get(symbol1, pd.DataFrame())
        data2 = self._symbol_data_indexed.get(symbol2, pd.DataFrame())

        # 按时间索引对齐计算（使用预缓存的 set_index 结果）
        corr1 = data1['close'].rolling(window=window).corr(data1['volume'])
        corr2 = data2['close'].rolling(window=window).corr(data2['volume'])
        FAC = corr1.sub(corr2, fill_value=np.nan)
        FAC = FAC.reindex(self.valid_index)
        return FAC.values.flatten()

    def closepctchg_sub(self, main_symbol, symbol1, symbol2, window=20):
        data1 = self._symbol_data_indexed.get(symbol1, pd.DataFrame())
        data2 = self._symbol_data_indexed.get(symbol2, pd.DataFrame())
        
        # 按时间索引对齐计算（使用预缓存的 set_index 结果）
        pct1 = data1['close'].pct_change(window)
        pct2 = data2['close'].pct_change(window)
        FAC = pct1.sub(pct2, fill_value=np.nan)
        FAC = FAC.reindex(self.valid_index)
        return FAC.values.flatten()
    
    def oi5_diff(self, main_symbol, symbol1, symbol2, window=5):
        """持仓量百分比变化差值: symbol1 - symbol2"""
        data1 = self._symbol_data_indexed.get(symbol1, pd.DataFrame())
        data2 = self._symbol_data_indexed.get(symbol2, pd.DataFrame())
        
        # 按时间索引对齐计算（使用预缓存的 set_index 结果）
        oi1 = data1['open_interest'].pct_change(window)
        oi2 = data2['open_interest'].pct_change(window)
        FAC = oi1.sub(oi2, fill_value=np.nan)
        FAC = FAC.reindex(self.valid_index)
        return FAC.values.flatten()
    
    def vcorr10(self, main_symbol, symbol1, symbol2, window=10):
        """成交量相关性: corr(symbol1.volume, symbol2.volume)"""
        data1 = self._symbol_data_indexed.get(symbol1, pd.DataFrame())
        data2 = self._symbol_data_indexed.get(symbol2, pd.DataFrame())
        
        # 取两个品种都有数据的共同索引
        common_idx = data1.index.intersection(data2.index)
        if len(common_idx) < window:
            return np.full(len(self.valid_index), np.nan)
        
        # 实时单点模式：切小段数据算 rolling，避免在 10000+ 行上浪费计算
        # 历史模式：直接用完整数据
        if len(self.valid_index) == 1:
            subset_idx = common_idx[common_idx <= self.valid_index[0]]
            need_rows = window + 5
            if len(subset_idx) > need_rows:
                subset_idx = subset_idx[-need_rows:]
            v1 = data1['volume'].reindex(subset_idx)
            v2 = data2['volume'].reindex(subset_idx)
            df = pd.DataFrame({'v1': v1, 'v2': v2})
            corr = df['v1'].rolling(window=window).corr(df['v2'])
            return pd.Series(corr.values, index=subset_idx).reindex(self.valid_index).values.flatten()
        else:
            v1 = data1['volume']
            v2 = data2['volume']
            df = pd.DataFrame({'v1': v1, 'v2': v2})
            corr = df['v1'].rolling(window=window).corr(df['v2'])
            return corr.reindex(self.valid_index).values.flatten()
    
    def volumediv_diff(self, main_symbol, symbol1, symbol2, window1=20, window2=5):
        """成交量比值的差分: (symbol1.vol_sum / symbol2.vol_sum).diff(window2)"""
        data1 = self._symbol_data_indexed.get(symbol1, pd.DataFrame())
        data2 = self._symbol_data_indexed.get(symbol2, pd.DataFrame())
        
        # 取两个品种都有数据的共同索引
        common_idx = data1.index.intersection(data2.index)
        if len(common_idx) < max(window1, window2):
            return np.full(len(self.valid_index), np.nan)
        
        # 实时单点模式：切小段数据算 rolling
        # 历史模式：直接用完整数据
        if len(self.valid_index) == 1:
            subset_idx = common_idx[common_idx <= self.valid_index[0]]
            need_rows = max(window1, window2) + 10
            if len(subset_idx) > need_rows:
                subset_idx = subset_idx[-need_rows:]
            v1 = data1["volume"].reindex(subset_idx)
            v2 = data2["volume"].reindex(subset_idx)
            df = pd.DataFrame({"v1": v1, "v2": v2})
            vol_sum1 = df["v1"].rolling(window=window1).sum()
            vol_sum2 = df["v2"].rolling(window=window1).sum()
            vol_ratio = vol_sum1 / (vol_sum2 + 1e-8)
            vol_ratio_diff = vol_ratio.diff(window2)
            return pd.Series(vol_ratio_diff.values, index=subset_idx).reindex(self.valid_index).values.flatten()
        else:
            v1 = data1["volume"]
            v2 = data2["volume"]
            df = pd.DataFrame({"v1": v1, "v2": v2})
            vol_sum1 = df["v1"].rolling(window=window1).sum()
            vol_sum2 = df["v2"].rolling(window=window1).sum()
            vol_ratio = vol_sum1 / (vol_sum2 + 1e-8)
            vol_ratio_diff = vol_ratio.diff(window2)
            return vol_ratio_diff.reindex(self.valid_index).values.flatten()

class ZMQDataFramePublisher:
    def __init__(self, port=11123):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{port}")
        self.port = port
        
    def publish_dataframe_json(self, df):
        """以 JSON 格式发送 DataFrame"""
        print(f"📡 发布者启动，端口 {self.port}")
        
        try:
            json_data = df.to_json(orient='records', date_format='iso')            
            self.socket.send_string(json_data)
            print(f"✅ 发送 JSON DataFrame: {df.shape}, 记录数: {len(df)}")
                
        except KeyboardInterrupt:
            print("\n⏹️ 发布者停止")

class ZMQPublisher:
    def __init__(self, port):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        # 设置 LINGER=0，进程退出时立即释放端口，避免残留占用
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(f"tcp://*:{port}")
        self.port = port

        time_module.sleep(0.1)

    def publish_data(self, messages):
        """发布数据 - 发送二进制数据"""
        print(f"发布者启动，监听端口 {self.port}")

        try:
            for msg_str, msg_float, msg_float2 in messages:
                float_bytes = struct.pack('d', msg_float)
                float_bytes += struct.pack('d', msg_float2)

                self.socket.send(msg_str.encode('utf-8'), zmq.SNDMORE)
                self.socket.send(float_bytes)
                print(f"发送二进制: 字符串={msg_str}, 浮点数={msg_float:.2f}, 总长度={len(float_bytes)}字节")
        except KeyboardInterrupt:
            print("====")

# 全局连接缓存，用于复用数据库连接
_db_connections = {}

def get_db_connection(db_path):
    """获取或创建数据库连接（复用连接）"""
    if db_path not in _db_connections:
        _db_connections[db_path] = sqlite3.connect(f'{db_path}/tick_data.db', check_same_thread=False)
    return _db_connections[db_path]

def close_all_db_connections():
    """关闭所有缓存的数据库连接"""
    for conn in _db_connections.values():
        conn.close()
    _db_connections.clear()

def read_table(instrument, db_path=None, word=True, trade_type=None, limit=None, only_datetime=False):
    """
    读取数据库表
    
    参数:
        instrument: 合约代码
        db_path: 数据库路径
        word: 是否打印日志
        trade_type: 交易时段类型
        limit: 限制读取行数
        only_datetime: 是否只读取最后一行的datetime（用于快速检查新数据，大幅减少CPU占用）
    """
    table_name = f"tick_data_{instrument}"
    try:
        # 如果只读取datetime，直接读取最后一行，无需排序
        if only_datetime:
            conn = get_db_connection(db_path)
            # 使用更可靠的方式获取最后一行的datetime
            query = f"SELECT datetime FROM {table_name} ORDER BY rowid DESC LIMIT 1"
            cursor = conn.cursor()
            if word:
                print(f"[DEBUG] Executing query: {query}")
            cursor.execute(query)
            result = cursor.fetchone()
            if word:
                print(f"[DEBUG] Query result: {result}")
            if result:
                datetime_val = result[0]
                if word:
                    print(f"[DEBUG] Returning datetime: {datetime_val}, type: {type(datetime_val)}")
                return datetime_val
            if word:
                print(f"[DEBUG] No result returned, returning None")
            return None
        
        # 正常读取流程
        if limit:
            conn = get_db_connection(db_path)
            query = f"SELECT * FROM {table_name} WHERE rowid > (SELECT MAX(rowid) FROM {table_name}) - {limit}"
        else:
            conn = get_db_connection(db_path)
            query = f"SELECT * FROM {table_name}"
            
        df = pd.read_sql_query(query, conn)

        if df.empty:
            if word:
                print('📭 表存在但为空')
            return pd.DataFrame()

        df = parse_df(df, trade_type=trade_type, word=word)
        
        if word:
            print(f"✅ 读取成功: {len(df)} 行数据")
        return df
        
    except sqlite3.OperationalError as e:
        if "no such table" in str(e).lower():
            if word:
                print(f"📋 表不存在: {table_name}")
            return None
        else:
            if word:
                print(f"❌ 数据库操作错误: {e}")
            return pd.DataFrame()
            
    except Exception as e:
        if word:
            print(f"❌ 读取失败: {e}")
        return pd.DataFrame()

def df_is_trading_time(df: pd.DataFrame, tscol_name: str, trade_type: list, word: str=True):
    """
    根据trade_type判断交易时间
    
    参数:
        df: DataFrame
        tscol_name: 时间列名
        trade_type: 交易时段列表，如 ["09:00-11:30", "13:30-15:00", "21:00-23:00"]
    """
    df[tscol_name] = pd.to_datetime(df[tscol_name])
    df['is_trading'] = False

    if trade_type == ["09:30-11:30", "13:00-15:00"]:
        # 股票类交易时间
        df['is_trading'] = (
            df[tscol_name].dt.time.between(time(9, 30, 0, 1), time(11, 30, 0)) |
            df[tscol_name].dt.time.between(time(13, 0, 0, 1), time(15, 0, 0))
        )

    elif trade_type == ["09:00-11:30", "13:30-15:00", "21:00-23:00"]:
        # 期货标准交易时间（夜盘到23点）
        df['is_trading'] = (
            df[tscol_name].dt.time.between(time(9, 0, 0, 1), time(10, 15, 0)) |
            df[tscol_name].dt.time.between(time(10, 30, 0, 1), time(11, 30, 0)) |
            df[tscol_name].dt.time.between(time(13, 30, 0, 1), time(15, 0, 0)) |
            df[tscol_name].dt.time.between(time(21, 0, 0, 1), time(23, 0, 0))
        )

    elif trade_type == ["09:00-11:30", "13:30-15:00"]:
        # 期货无夜盘
        df['is_trading'] = (
            df[tscol_name].dt.time.between(time(9, 0, 0, 1), time(10, 15, 0)) |
            df[tscol_name].dt.time.between(time(10, 30, 0, 1), time(11, 30, 0)) |
            df[tscol_name].dt.time.between(time(13, 30, 0, 1), time(15, 0, 0))
        )

    elif trade_type == ["09:00-11:30", "13:30-15:00", "21:00-01:00"]:
        # 期货夜盘跨天到01:00
        df['is_trading'] = (
            df[tscol_name].dt.time.between(time(9, 0, 0, 1), time(10, 15, 0)) |
            df[tscol_name].dt.time.between(time(10, 30, 0, 1), time(11, 30, 0)) |
            df[tscol_name].dt.time.between(time(13, 30, 0, 1), time(15, 0, 0)) |
            df[tscol_name].dt.time.between(time(21, 0, 0, 1), time(23, 59, 59, 999999)) |
            df[tscol_name].dt.time.between(time(0, 0, 0, 1), time(1, 0, 0))
        )

    elif trade_type == ["09:00-11:30", "13:30-15:00", "21:00-02:30"]:
        # 期货夜盘跨天到02:30
        df['is_trading'] = (
            df[tscol_name].dt.time.between(time(9, 0, 0, 1), time(10, 15, 0)) |
            df[tscol_name].dt.time.between(time(10, 30, 0, 1), time(11, 30, 0)) |
            df[tscol_name].dt.time.between(time(13, 30, 0, 1), time(15, 0, 0)) |
            df[tscol_name].dt.time.between(time(21, 0, 0, 1), time(23, 59, 59, 999999)) |
            df[tscol_name].dt.time.between(time(0, 0, 0), time(2, 30, 0))
        )
    
    else: 
        if word:
            print(f'time not typical: {trade_type}, pass')
        
    return df

def _process_time_column_vectorized_floor(df, time_column='update_time'):
    """
    向量化版本，将毫秒向上取整到最近的0.25秒（向前调整）
    输入格式: 210000313 表示 21:00:00.313
    """
    # 将整数转换为 HH:MM:SS.mmm 格式的字符串
    time_str = df[time_column].astype(int).astype(str).str.zfill(9)  # 确保9位数字
    
    # 格式化为 HH:MM:SS.mmm
    formatted_times = (
        time_str.str[0:2] + ':' +  # 小时
        time_str.str[2:4] + ':' +  # 分钟
        time_str.str[4:6] + '.' +  # 秒 + 小数点
        time_str.str[6:9]          # 毫秒
    )
    
    # 使用原来的逻辑
    time_parts = formatted_times.str.split('.', expand=True)
    seconds = time_parts[0]
    milliseconds = time_parts[1].fillna('000').astype(int)
    
    # 向上取整到最近的250毫秒（只向后调整）
    rounded_ms = np.floor(milliseconds / 250) * 250
    
    # 处理进位（如果向上取整后达到1000毫秒）
    carry_over = (rounded_ms >= 1000).astype(int)
    rounded_ms = np.where(rounded_ms >= 1000, rounded_ms - 1000, rounded_ms)
    
    # 处理秒的进位
    def adjust_seconds(sec_str, carry):
        if carry == 0:
            return sec_str
        
        # 解析时间并增加一秒
        time_obj = datetime.strptime(sec_str, '%H:%M:%S')
        time_obj += timedelta(seconds=1)
        return time_obj.strftime('%H:%M:%S')
    
    # 应用进位调整
    adjusted_seconds = [adjust_seconds(sec, carry) for sec, carry in zip(seconds, carry_over)]
    
    # 重新组合时间字符串
    df['rounded_time'] = [
        f"{sec}.{int(ms):03d}" 
        for sec, ms in zip(adjusted_seconds, rounded_ms)
    ]
    return df

def _process_time_column_vectorized_floor_local(df, time_column='update_time'):
    """
    向量化版本，将毫秒向上取整到最近的0.25秒（向前调整）
    """
    # 分离秒和毫秒部分
    time_parts = df[time_column].str.split('.', expand=True)
    seconds = time_parts[0]
    milliseconds = time_parts[1].fillna('000').astype(int)
    
    # 向上取整到最近的250毫秒（只向后调整）
    rounded_ms = np.floor(milliseconds / 250) * 250
    
    # 处理进位（如果向上取整后达到1000毫秒）
    carry_over = (rounded_ms >= 1000).astype(int)
    rounded_ms = np.where(rounded_ms >= 1000, rounded_ms - 1000, rounded_ms)
    
    # 处理秒的进位
    def adjust_seconds(sec_str, carry):
        if carry == 0:
            return sec_str
        
        # 解析时间并增加一秒
        time_obj = datetime.strptime(sec_str, '%H:%M:%S')
        time_obj += timedelta(seconds=1)
        return time_obj.strftime('%H:%M:%S')
    
    # 应用进位调整
    adjusted_seconds = [adjust_seconds(sec, carry) for sec, carry in zip(seconds, carry_over)]
    
    # 重新组合时间字符串
    df['rounded_time'] = [
        f"{sec}.{int(ms):03d}" if ms > 0 else sec 
        for sec, ms in zip(adjusted_seconds, rounded_ms)
    ]
    
    return df

def _process_datetime(df):
    date_s = pd.to_datetime(df.datetime).dt.date
    df_datetime_std = date_s.astype(str) + ' ' + df['rounded_time']
    df['datetime'] = df_datetime_std

    return df

def get_trading_days(start_date='2021-01-01', end_date='2026-12-31', exclude_days=None, return_str=False):

    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()

    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    
    # 默认要排除的非交易日（可自定义修改）
    if exclude_days is None:
        exclude_days = [
        ]
    
    # 生成所有工作日（周一到周五）
    all_weekdays = pd.date_range(start_date, end_date, freq='B').date
    
    # 转换为字符串格式用于比较
    exclude_days_set = set(exclude_days)
    
    if return_str:
        trading_days = [
            day.strftime('%Y-%m-%d') for day in all_weekdays 
            if calendar.is_workday(day) and day.strftime('%Y-%m-%d') not in exclude_days_set
        ]     
    else:
        trading_days = [
            day for day in all_weekdays 
            if (calendar.is_workday(day)) & (day.strftime('%Y-%m-%d') not in exclude_days_set)
        ]
    
    return trading_days

def _process_datetime_local(df):
    dates = sorted([str(x) for x in get_trading_days()])
    
    date_to_prev_date = {dates[i]: dates[i-1] for i in range(1, len(dates))}
    
    def get_adjusted_date(row):
        if row['period'] == 'night':
            return date_to_prev_date.get(row['trade_date'], row['trade_date'])
        return row['trade_date']
    
    df['adjusted_date'] = df.apply(get_adjusted_date, axis=1)
    
    def combine_datetime(row):
        time_part = row['rounded_time']
        if '.' not in time_part:
            time_part = time_part + '.000'
        return f"{row['adjusted_date']} {time_part}"
    
    df['datetime'] = pd.to_datetime(
        df.apply(combine_datetime, axis=1),
        format='%Y-%m-%d %H:%M:%S.%f',
        errors='coerce'  # 如果格式不匹配则转为 NaT
    )
    df.drop(['adjusted_date', 'rounded_time'], axis=1, inplace=True)
    df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-4]

    return df

def parse_df(df, trade_type, local=False, word=True):

    if local:
        # print('local time parse')
        df = _process_time_column_vectorized_floor_local(df)
        df = _process_datetime_local(df)
    else:
        df = _process_time_column_vectorized_floor(df)
        df = _process_datetime(df)

    df = df_is_trading_time(df, 'datetime', trade_type=trade_type, word=word)
    df = df.sort_values('datetime')
    df['instrument'] = df['instrument'].str.strip()
    numeric_cols = ['last_price',
            'open_interest', 'turnover', 'volume', 'bid_price1', 'bid_price2',
            'bid_price3', 'bid_price4', 'bid_price5', 'ask_price1',
            'ask_price2', 'ask_price3', 'ask_price4', 'ask_price5',
            'bid_volume1', 'bid_volume2', 'bid_volume3', 'bid_volume4',
            'bid_volume5', 'ask_volume1', 'ask_volume2', 'ask_volume3',
            'ask_volume4', 'ask_volume5']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            mask = df[col].abs()>1e20
            df.loc[mask, col] = pd.NA

    bool_cols = df.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df[col] = df[col].astype('boolean')

    for col in df.columns:
        str_lengths = df[col].astype(str).str.len()
        df.loc[str_lengths > 50, col] = pd.NA
        
    df['ts'] = pd.to_datetime(df['datetime']).dt.ceil('1min')
    
    if local:
        pass
    else:
        df['TotalTradeVolume'] = df['volume'].copy()
        df['TotalTradeValue'] = df['turnover'].copy()
        volume_diff = df['volume'] - df['volume'].shift(1)
        turnover_diff = df['turnover'] - df['turnover'].shift(1)

        df.loc[df.is_trading, 'volume'] = volume_diff.loc[df.is_trading]
        df.loc[df.is_trading, 'turnover'] = turnover_diff.loc[df.is_trading]

    # 只在交易时间内计算最高最低价（与研究环境保持一致）
    # 过滤掉非交易时间的数据
    # df = df[df['is_trading']].copy()
    df['highest_price'] = df.groupby('trade_date')['last_price'].cummax()
    df['lowest_price'] = df.groupby('trade_date')['last_price'].cummin()
    df = df[df['is_trading']].copy()

    df['high'] = df['last_price']
    df['low'] = df['last_price']
    df['open'] = df['last_price']
    df['close'] = df['last_price']
    
    df['spread'] = df['ask_price1'] - df['bid_price1']
    total_volume = df['bid_volume1'] + df['ask_volume1']
    df['mid_price'] = (
        (df['ask_price1'] * df['bid_volume1'] + 
            df['bid_price1'] * df['ask_volume1']) / total_volume
    ).round(4)

    bid_cols = [f'bid_volume{i}' for i in range(1, 6)]
    ask_cols = [f'ask_volume{i}' for i in range(1, 6)]
    df['bvall'] = df[bid_cols].sum(axis=1)
    df['avall'] = df[ask_cols].sum(axis=1)
    df['corrAskwap'] = df['avall'].diff().rolling(120, min_periods=100).corr(df['mid_price'])
    df['corrBidwap'] = df['bvall'].diff().rolling(120, min_periods=100).corr(df['mid_price'])
    df['volume_avg'] = df['TotalTradeVolume'].diff().rolling(120, min_periods=100).mean().shift(1)
    df['M_std'] = df['mid_price'].rolling(120, min_periods=100).std()

    df['second'] = pd.to_datetime(df.datetime).dt.ceil('1s').dt.second
    df['last_twap'] = np.where(df['second'] > 5, df['last_price'], np.nan)
    df['bar_count'] = df.groupby('ts').cumcount() + 1
    df = df.replace([np.inf, -np.inf], np.nan)

    return df

def parse_time(df, tscol_name, time_config):
    """
    根据time_config配置过滤交易时间 - 只剔除真正的非交易时段
    （10:15-10:30休息时间和午间休市），保留开盘前和收盘前10分钟数据
    
    参数:
        df: DataFrame
        tscol_name: 时间列名
        time_config: 时间配置字典，包含day_session和night_session
    """
    df[tscol_name] = pd.to_datetime(df[tscol_name])
    
    day_session = time_config.get('day_session', {})
    night_session = time_config.get('night_session', {})
    
    # 白天盘时间段（完整交易时段，包括开盘前和收盘前10分钟）
    # 使用1微秒确保严格大于整点时间，避免9:00:00的数据被包含
    morning1_start = time(9, 0, 0, 1)   # 9:00:00.000001
    morning1_end = time(10, 15, 0)      # 10:15:00
    morning2_start = time(10, 30, 0, 1) # 10:30:00.000001
    morning2_end = time(11, 30, 0)      # 11:30:00
    afternoon_start = time(13, 30, 0, 1) # 13:30:00.000001
    afternoon_end = time(15, 0, 0)      # 15:00:00
    
    conditions = [
        df[tscol_name].dt.time.between(morning1_start, morning1_end),
        df[tscol_name].dt.time.between(morning2_start, morning2_end),
        df[tscol_name].dt.time.between(afternoon_start, afternoon_end)
    ]
    
    # 夜盘时间段（完整时段）
    if night_session.get('enabled', False):
        night_start = time(21, 0, 0, 1)  # 21:00:00.000001
        night_end = parse_time_str(night_session.get('end_time'))  # 如 23:00
        
        # 处理跨天的情况
        if night_start > night_end:
            # 跨天情况，如 21:00-02:30
            conditions.append(
                (df[tscol_name].dt.time >= night_start) | (df[tscol_name].dt.time <= night_end)
            )
        else:
            # 不跨天情况，如 21:00-23:00
            conditions.append(df[tscol_name].dt.time.between(night_start, night_end))
    
    df = df[np.logical_or.reduce(conditions)]
    return df

def update_concat(current, new_data):
    if new_data.empty:
        print("No new data to update.")
        return current
    new_data = new_data.reindex(columns=current.columns)
    # print(f"Updating data: current {current.shape}, new rows{new_data.shape}")
    updated = pd.concat([current, new_data], axis=0) if not current.empty else new_data
    return updated.sort_values('datetime').drop_duplicates('datetime', keep='last').reset_index(drop=True)

def aggregate_ticks(df: pd.DataFrame, last_trade_date_map: dict = None, time_config=None) -> pd.DataFrame:
    """
    聚合tick数据为分钟数据
    
    参数:
        df: tick数据DataFrame
        last_trade_date_map: 合约到最后交易日的映射字典，如 {'p2605': '2026-04-01'}
        time_config: 时间配置字典
    """
    if df.empty:
        return df
        
    df = df.copy()
    df['bar_count'] = 1

    # 如果提供了最后交易日映射，则添加该列
    if last_trade_date_map:
        df['LAST_TRADE_DATE'] = df['instrument'].map(last_trade_date_map)
    else:
        df['LAST_TRADE_DATE'] = ""

    result = (
        df
        .groupby('ts')
        .agg({
            'last_price': ['first', 'max', 'min', 'last'],
            'volume': 'sum',
            'turnover': 'sum',
            'instrument': 'last',
            'open_interest': 'last',
            'bar_count': 'sum',
            'trade_date': 'last',
            'LAST_TRADE_DATE': 'last',
        })
    )
    result = parse_time(result.reset_index(), 'ts', time_config).set_index('ts')
    result.index.name = 'datetime'

    result.columns = [
        'open', 'high', 'low', 'close',
        'volume', 'turnover', 'instrument', 'open_interest', 'bar_count', 'trade_date', 'LAST_TRADE_DATE']
    
    return result.reset_index()

def time_scale_df(df: pd.DataFrame, tscol_name: str, trade_type: list):
    df[tscol_name] = pd.to_datetime(df[tscol_name])
    if trade_type == ["09:30-11:30", "13:00-15:00"]:
        df = df[
            (df[tscol_name].dt.time.between(time(9, 30, 0, 1000), time(11, 30, 0))) |
            (df[tscol_name].dt.time.between(time(13, 0, 0, 1000), time(15, 0, 0)))
        ]

    elif trade_type == ["09:00-11:30", "13:30-15:00", "21:00-23:00"]:
        df = df[
            (df[tscol_name].dt.time.between(time(9, 0, 0, 1000), time(11, 30, 0))) |
            (df[tscol_name].dt.time.between(time(13, 30, 0, 1000), time(15, 0, 0))) |
            (df[tscol_name].dt.time.between(time(21, 0, 0, 1000), time(23, 0, 0)))
        ]

    elif trade_type == ["09:00-11:30", "13:30-15:00"]:
        df = df[
            (df[tscol_name].dt.time.between(time(9, 0, 0, 1000), time(11, 30, 0))) |
            (df[tscol_name].dt.time.between(time(13, 30, 0, 1000), time(15, 0, 0))) 
        ]

    elif trade_type == ["09:00-11:30", "13:30-15:00", "21:00-01:00"]:
        df = df[
            (df[tscol_name].dt.time.between(time(9, 0, 0, 1000), time(11, 30, 0))) |
            (df[tscol_name].dt.time.between(time(13, 30, 0, 1000), time(15, 0, 0))) |
            (df[tscol_name].dt.time.between(time(21, 0, 0, 1000), time(23, 59, 59, 9999))) |
            (df[tscol_name].dt.time.between(time(0, 0, 0, 0), time(1, 0, 0)))
        ]

    elif trade_type == ["09:00-11:30", "13:30-15:00", "21:00-02:30"]:
        df = df[
            (df[tscol_name].dt.time.between(time(9, 0, 0, 1000), time(11, 30, 0))) |
            (df[tscol_name].dt.time.between(time(13, 30, 0, 1000), time(15, 0, 0))) |
            (df[tscol_name].dt.time.between(time(21, 0, 0, 1000), time(23, 59, 59, 9999))) |
            (df[tscol_name].dt.time.between(time(0, 0, 0, 0), time(2, 30, 0)))
        ]
    
    else: print('time not typical, pass')
    
    return df

def load_fac_df_old(factor_col, instrument_list, recent_data_path, trade_type, dict_keys, generate_factor_dataframe):
    """
    因子加载函数 - 通用版本
    
    参数:
        factor_col: 因子列名列表
        instrument_list: 品种列表，格式为:
            [主合约, 其他品种1合约, 其他品种2合约, ..., 下一个合约]
        recent_data_path: 数据路径
        trade_type: 交易时段列表
        
    返回:
        fac_df: 生成的因子DataFrame
    """
    instrument_main = instrument_list[0]  # 主合约（第一个位置）
    other_instruments = instrument_list[1:]  # 其他品种
    
    tick_data_main = pd.read_csv(f'{recent_data_path}/{instrument_main}_tick.csv', parse_dates=['datetime'])
    tick_data_main = parse_df(tick_data_main, trade_type=trade_type, local=True)
    tick_data_main = time_scale_df(tick_data_main, 'ts', trade_type)
    tick_data_main = tick_data_main.replace([np.inf, -np.inf], np.nan)
    # return tick_data_main

    min_data_main = pd.read_csv(f'{recent_data_path}/{instrument_main}_min.csv', parse_dates=['datetime']).reindex(
        columns=['datetime', 'open', 'high', 'low', 'close', 'volume', 'turnover', 
                'instrument', 'open_interest', 'trade_date', 'LAST_TRADE_DATE']
    )

    other_min_dfs = []
    for i, instrument in enumerate(other_instruments, 1):
        min_data = pd.read_csv(f'{recent_data_path}/{instrument}_min.csv', parse_dates=['datetime']).reindex(
            columns=['datetime', 'open', 'high', 'low', 'close', 'volume', 'turnover', 
                    'instrument', 'open_interest', 'trade_date', 'LAST_TRADE_DATE']
        )
        other_min_dfs.append(min_data)
    
    fac_generator_args = [tick_data_main, min_data_main] + other_min_dfs

    fac_generator = Factor_generator(*fac_generator_args)
    fac_generator.dict_keys = dict_keys
    fac_generator.load_df_names()

    fac_df = generate_factor_dataframe(fac_generator, fac_generator.valid_index, factor_col)
    
    return fac_df

def load_tick_min(instrument, current_time, recent_data_path, dp_path, trade_type=None, time_config=None):
    min_data_concat = pd.read_csv(f'{recent_data_path}/{instrument}_min.csv', parse_dates=['datetime'])
    min_data_concat = min_data_concat[['datetime', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'instrument', 'open_interest', 'bar_count', 'trade_date', 'LAST_TRADE_DATE']]

    df = read_table(instrument, db_path=dp_path, trade_type=trade_type)

    # print(current_time)
    if (current_time.second == 0) and (current_time.microsecond == 0):
        pass
    else:
        current_time = current_time.replace(second=0, microsecond=0)

    tick = df[df.datetime <= current_time]
    
    min = aggregate_ticks(tick, time_config=time_config)
    min_data_concat = update_concat(min_data_concat, min)
    return tick.iloc[-1000:], min_data_concat
    # return tick, min_data_concat

def run_345(pred_df, th_open, th_close, now_pos, now_holding, max_holding, bar_in_day, window=5, time_config=None):
    """
    运行交易策略
    
    参数:
        pred_df: 预测数据框
        th_open: 开仓阈值
        th_close: 平仓阈值
        now_pos: 当前仓位
        now_holding: 当前持仓时间
        max_holding: 最大持仓时间
        bar_in_day: 一天内的分钟数
        time_config: 时间配置字典，包含交易时间段信息
    """
    df = pred_df.copy()
    
    time_cond = df.index.time
    # 按位置 shift：数据每分钟连续到达，直接用前1根/前2根 bar 的值
    w = df['weighted']
    df['weighted_s'] = w * 0.6 + w.shift(1) * 0.3 + w.shift(2) * 0.1
    
    day_session = time_config.get('day_session', {})
    night_session = time_config.get('night_session', {})
    
    # 白天盘交易时间段
    day_start = parse_time_str(day_session.get('no_trade_end'))
    day_end = parse_time_str(day_session.get('no_trade_start2'))
    
    condition = ((time_cond > day_start) & (time_cond <= day_end))
    
    # 如果有夜盘且启用
    if night_session.get('enabled', False):
        night_start = parse_time_str(night_session.get('no_trade_end'))
        night_end = parse_time_str(night_session.get('no_trade_start2'))
        condition = condition | ((time_cond >= night_start) & (time_cond <= night_end))

    df[~condition] = 0 
    thresholds_df = pd.DataFrame(index=df.columns)
    df = df[~df.index.duplicated(keep='last')].iloc[-window * bar_in_day - 1:]
    
    thresholds_df['long_open'] = df.iloc[:-1].quantile(th_open)
    thresholds_df['long_close'] = df.iloc[:-1].quantile(th_close)

    thresholds_df['short_close'] = df.iloc[:-1].quantile(1 - th_close)
    thresholds_df['short_open'] = df.iloc[:-1].quantile(1 - th_open)

    thresholds_df['long_open'] = thresholds_df['long_open'].mask(thresholds_df['long_open']<0, 0)
    thresholds_df['short_open'] = thresholds_df['short_open'].mask(thresholds_df['short_open']>0, 0)

    thresholds_df['now'] = df.iloc[-1]

    weighted = thresholds_df.T.astype(float)['weighted_s'].to_dict()

    signal = '等待行情'
    if now_pos == 0:
        if weighted['now'] > weighted['long_open']:
            signal = '开多'
            now_holding += 1
            now_pos = 1
        elif weighted['now'] < weighted['short_open']:  
            signal = '开空'
            now_holding += 1
            now_pos = -1
        else:
            signal = '等待行情'

    elif now_pos == 1:
        if weighted['now'] < weighted['short_open']:
            signal = '平多开空'
            now_holding = 1
            now_pos = -1
        elif weighted['now'] < weighted['long_close']:
            signal = '平多'
            now_holding = 0
            now_pos = 0
        elif weighted['now'] > weighted['long_open']:
            signal = '继续看多'
            now_holding = 1
            now_pos = 1
        elif now_holding >= max_holding:
            signal = '过长时间无合适信号，平仓'
            now_holding = 0
            now_pos = 0 
        else:
            signal = '持多观望'
            now_holding += 1
            now_pos = 1

    elif now_pos == -1:
        if weighted['now'] > weighted['long_open']:
            signal = '平空开多'
            now_holding = 1
            now_pos = 1
        elif weighted['now'] > weighted['short_close']:
            signal = '平空'
            now_holding = 0
            now_pos = 0
        elif weighted['now'] < weighted['short_open']:
            signal = '继续看空'
            now_holding = 1
            now_pos = -1
        elif now_holding >= max_holding:
            signal = '过长时间无合适信号，平仓'
            print(signal)
            now_holding = 0
            now_pos = 0         
        else:
            signal = '持空观望'
            now_holding += 1
            now_pos = -1          

    hist_data = df.iloc[:-1]['weighted_s']  # 历史数据（排除当前值）
    current_value = df.iloc[-1]['weighted_s']  # 当前值
    
    quantile_position = (hist_data < current_value).mean()
    pass

    current_time = df.index.time[-1]
    
    # 从time_config读取交易时间配置
    day_session = time_config.get('day_session', {})
    night_session = time_config.get('night_session', {})
    
    # 白天盘交易时间（去掉前后各10分钟）
    day_start = parse_time_str(day_session.get('no_trade_end'))  # 09:10
    day_end = parse_time_str(day_session.get('no_trade_start2'))  # 14:50
    
    # 检查是否在白天盘交易时间段
    in_day_session = (day_start < current_time <= day_end)
    
    # 夜盘交易时间（去掉前后各10分钟）
    in_night_session = False
    if night_session.get('enabled', False):
        night_start = parse_time_str(night_session.get('no_trade_end'))  # 21:10
        night_end = parse_time_str(night_session.get('no_trade_start2'))  # 22:50
        in_night_session = (night_start < current_time <= night_end)
    
    if in_day_session or in_night_session:
        return now_pos, now_holding, weighted, thresholds_df, df, signal
    else:
        signal = '未开仓'
        return 0, 0, {}, pd.DataFrame(), df, signal
