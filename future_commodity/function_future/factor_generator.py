import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class FactorGenerator:
    def __init__(self):
        pass

    def load_tick_min_data(self, data, data1m, info, instrument, multiplier):
        self.data = data
        self.data1m = data1m
        self.info = info
        self.instrument = instrument
        self.multiplier = multiplier

    def load_min_min_data(self, data1m, info, instrument, data1m_next, LAST_TRADE_DATE):
        self.data1m = data1m
        self.info = info
        self.instrument = instrument
        # self.instrument_next = info.set_index('instrument').loc[instrument, 'instrument_2'].iloc[0]
        self.data1m_next = data1m_next
        self.LAST_TRADE_DATE = LAST_TRADE_DATE

    def load_tick_tick_data(self, data, info, instrument, multiplier):
        # 兼容：calc_factors_dce农.py 的 tick_name_dict 把 turnover 重命名为 amount
        if 'amount' in data.columns and 'turnover' not in data.columns:
            data = data.copy()
            data['turnover'] = data['amount']
        if 'volume' in data.columns and 'TotalTradeVolume' not in data.columns:
            data = data.copy()
            data['TotalTradeVolume'] = data['volume']
        self.data = data
        self.info = info
        self.instrument = instrument
        self.multiplier = multiplier
        
    def calculate_tick_tick_factor(self, factor_function):
        return self._calculate_factor(factor_function, type='tick_tick')

    def calculate_tick_minute_factor(self, factor_function):
        return self._calculate_factor(factor_function, type='tick_min')

    def calculate_minute_minute_factor(self, factor_function):
        return self._calculate_factor(factor_function, type='min_min')

    def _calculate_factor(self, factor_function, type):

        if type=='tick_tick':
            if factor_function.__name__ == "MPB": res = factor_function(self.data, instrument=self.instrument)
            else: res = factor_function(self.data)
                
            factor_name = factor_function.__name__
            fac = pd.DataFrame(index=self.data.index, columns=['datetime', 'symbol', 'factor_name', 'factor_value'])
            fac['datetime'] = self.data.index
            fac['symbol'] = self.data.instrument
            fac['factor_name'] = factor_name
            
            if factor_name in res.columns:
                fac['factor_value'] = res[factor_name]
            else:
                raise KeyError(f"因子列 '{factor_name}' 不存在于计算结果中。请检查因子函数 '{factor_function.__name__}' 的输出。")
            return fac.reset_index(drop=True)
        
        if type=='min_min':

            self.factor_params_map = {
                    'term_rtn': {'data': self.data1m, 'data_next': self.data1m_next},
                    'ptvol5': {'data': self.data1m, 'data1m_next': self.data1m_next, 'LAST_TRADE_DATE': self.LAST_TRADE_DATE},
                    'JC1D': {'data': self.data1m, 'data1m_next': self.data1m_next, 'LAST_TRADE_DATE': self.LAST_TRADE_DATE},
                    'JC2H': {'data': self.data1m, 'data1m_next': self.data1m_next, 'LAST_TRADE_DATE': self.LAST_TRADE_DATE},
                    'ZCpriceinterval': {'data1m': self.data1m, 'data1m_next': self.data1m_next, 'LAST_TRADE_DATE': self.LAST_TRADE_DATE}
                }
            factor_name = factor_function.__name__
        
            params = self.factor_params_map.get(factor_name, {'data': self.data1m})
            res = factor_function(**params)

            fac = pd.DataFrame(columns=['datetime', 'symbol', 'factor_name', 'factor_value'])
            fac['datetime'] = self.data1m.ts
            fac['symbol'] = self.data1m.instrument
            fac['factor_name'] = factor_name
            
            if len(res) == len(fac):

                if factor_name == 'calculate_factors':
                    return res

                if isinstance(res, np.ndarray):  # 检查是否为 NumPy 数组
                    fac['factor_value'] = res
                else:
                    fac['factor_value'] = res.values  # 强制转换为 NumPy 数组
            else:
                raise KeyError(f"因子列 '{factor_name}'长度不对。请检查函数 '{factor_function.__name__}' 的输出。")
            return fac
        
        if type=='tick_min':
            if factor_function.__name__ == "lastprice_bias1": res = factor_function(self.data, self.data1m, self.multiplier)
            else: res = factor_function(self.data, self.data1m)

            factor_name = factor_function.__name__
            fac = pd.DataFrame(columns=['datetime', 'symbol', 'factor_name', 'factor_value'])
            fac['datetime'] = self.data1m.ts
            fac['symbol'] = self.data1m.instrument
            fac['factor_name'] = factor_name
            
            if len(res) == len(fac):
                fac['factor_value'] = res.values
            else:
                raise KeyError(f"因子列 '{factor_name}'长度不对。请检查函数 '{factor_function.__name__}' 的输出。")
            return fac


# VWAP 计算函数
def compute_vwap(group):
    if len(group) == 0:
        return np.nan
    weights = group['TotalTradeVolume'].diff().fillna(0)  # 计算权重
    weighted_value = (group['factor_value'] * weights).sum() / weights.sum() if weights.sum() != 0 else np.nan
    return weighted_value

# Up Mean 计算函数
# 修复：用 pd.Series.mean() 跳过 NaN（numpy mean 不跳过 NaN 会导致大量 NaN 输出）
def compute_up_mean(factor_series, mid_price):
    fac_up_value = pd.Series(np.where(mid_price.diff(2)>=0, factor_series, 0))
    return fac_up_value.mean()

def compute_down_mean(factor_series, mid_price):
    fac_down_value = pd.Series(np.where(mid_price.diff(2)<=0, factor_series, 0))
    return fac_down_value.mean()

#  （1 - 价格和卖量相关系数做加权）的mean
def compute_corrAskwap(group):
    weights = 1 - group.corrAskwap
    weighted_value = (group['factor_value'] * weights).sum() / weights.sum() if weights.sum() != 0 else np.nan
    return weighted_value

# （1 + 价格和买量相关系数做加权）的mean
def compute_corrBidwap(group):
    weights = 1 - group.corrBidwap
    weighted_value = (group['factor_value'] * weights).sum() / weights.sum() if weights.sum() != 0 else np.nan
    return weighted_value

# （中间价波动率加权）的mean
def compute_Mstdwap(group):
    weighted_value = (group['factor_value'] * group.M_std).sum() / group.M_std.sum() if group.M_std.sum() != 0 else np.nan
    return weighted_value

# 成交量激增
def compute_Volraiseap(group):
    weights = (group['TotalTradeVolume'].diff() > 1.5 * group.volume_avg).astype(int)
    weighted_value = (group['factor_value'] * weights).sum() / weights.sum() if weights.sum() != 0 else np.nan
    return weighted_value

# 买盘主导
def compute_biddommean(group):
    weights = (group['bvall'] > group['avall']).astype(int)
    weighted_value = (group['factor_value'] * weights).sum() / weights.sum() if weights.sum() != 0 else np.nan
    return weighted_value

# 卖盘主导
def compute_askdommean(group):
    weights = (group['bvall'] < group['avall']).astype(int)
    weighted_value = (group['factor_value'] * weights).sum() / weights.sum() if weights.sum() != 0 else np.nan
    return weighted_value

# MAD去均值
def compute_MADmean(group):
    median = group['factor_value'].median()
    mad = (group['factor_value'] - median).abs().median()
    filtered = group[(group['factor_value'] - median).abs() <= 3 * mad]
    return filtered['factor_value'].mean()

# 反转期间均值
def compute_trend_rev(group):
    short_trend = group['mid_price'].diff(10)
    long_trend = group['mid_price'].diff(30)
    valid_period = short_trend * long_trend < -1  # 趋势不一致的时期
    weights = (short_trend * long_trend).mask(short_trend * long_trend>0, 0).abs()

    weighted_value = (group['factor_value'] * weights).sum() / weights.sum() if weights.sum() != 0 else np.nan
    return weighted_value

def compute_statistics(group):
    if len(group) == 0:
        return pd.Series({
            'vwap': np.nan,
            'upmean': np.nan,
            'downmean': np.nan,
            'corrAskwap': np.nan,
            'corrBidwap': np.nan,
            'Mstdwap': np.nan,
            'Volraiseap': np.nan,
            'biddommean': np.nan,
            'askdommean': np.nan,
            'MADmean': np.nan,
            'kurtosis': np.nan,
            'skewness': np.nan,
            'TrendRevmean': np.nan,
            'symbol': np.nan,
            'factor_name': np.nan
        }) 
    
    vwap_value = compute_vwap(group)
    corrAskwap_value = compute_corrAskwap(group)
    corrBidwap_value = compute_corrBidwap(group)
    Mstdwap_value = compute_Mstdwap(group)
    Volraiseap_value = compute_Volraiseap(group)
    biddommean_value = compute_biddommean(group)
    askdommean_value = compute_askdommean(group)
    MADmean_value = compute_MADmean(group)
    TrendRevap_value = compute_trend_rev(group)

    up_mean_value = compute_up_mean(
        group['factor_value'], 
        group['mid_price'], 
    )
    down_mean_value = compute_down_mean(
        group['factor_value'], 
        group['mid_price'], 
    )

    return pd.Series({
        'vwap': vwap_value,
        'upmean': up_mean_value,
        'downmean': down_mean_value,
        'corrAskwap': corrAskwap_value,
        'corrBidwap': corrBidwap_value,
        'Mstdwap': Mstdwap_value,
        'Volraiseap': Volraiseap_value,
        'biddommean': biddommean_value,
        'askdommean': askdommean_value,
        'MADmean': MADmean_value,
        'TrendRevmean': TrendRevap_value,
        'kurtosis': group['factor_value'].kurt(),
        'skewness': group['factor_value'].skew(),
        'symbol': group['symbol'].iloc[-1],  
        'factor_name': group['factor_name'].iloc[-1] 
    })  

from joblib import Parallel, delayed
def parallel_compute_statistics_optimized(df_grouped, n_jobs=20):
    """
    最终版解决方案：
    1. 执行resample生成完整时间序列
    2. 并行计算时自动跳过空分组
    3. 精确重建只含交易时间的索引
    """

    # 并行计算（跳过空分组）
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_statistics)(group)
        for time_key, group in df_grouped
        if not group.empty  # 关键过滤
    )
    
    # 精确重建索引（只保留实际有数据的时间点）
    result_df = pd.concat(
        [pd.DataFrame(r).T for r in results],
        axis=0
    ).set_index(
        pd.to_datetime([time_key for time_key, group in df_grouped if not group.empty])
    )
    return result_df