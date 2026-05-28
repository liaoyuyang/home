import pandas as pd
import numpy as np
from numba import njit

# min data min factors
#-------------------------------------------------------------------------------------------------------------------------------------
# fac 101
def ASI(data, window = 5):
    """
    趋势因子
    技术指标ASI
    """
    data = data.copy()
    A = (data.high - data.close.shift(1)).abs()
    B = (data.low - data.close.shift(1)).abs()
    C = (data.high - data.close.shift(1)).abs()
    D = (data.close.shift(1) - data.open.shift(1)).abs()
    E = data.close - data.close.shift(1)
    F = data.close - data.open
    G = data.close.shift(1) - data.open.shift(1)
    X = E + F/2 + G
    K = np.maximum(A,B)
    R = np.where((A>B)&(A>C), A+B/2+D/4, np.where((B>A)&(B>C), B+A/2+D/4, C+D/4))
    SI = 16 * X / R * K
    ASI = SI.rolling(window=window).sum()
    return ASI

# fac 102
def BOLLING(data, window = 5):
    """
    反转因子
    技术指标布林带
    """
    data = data.copy()

    MA = data['close'].rolling(window=window).mean()
    STD = data['close'].rolling(window=window).std()

    FACTOR = (data['close'] - MA) / STD    
    return FACTOR

# fac 103
def MAdiff_Vol_div(data, window1 = 5,window2 = 20):
    """
    趋势因子
    双均线短期偏离度 * 持仓量之差
    """
    data = data.copy()
    MAdiff = (data.close.rolling(window=window1).mean() - data.close.rolling(window=window2).mean()) / data.close
    FACTOR = MAdiff * data.OpenInterest.diff(window2)
    return FACTOR

# fac 104
def drawback(data, window = 30, th=0.8):
    """
    反转因子
    衡量高位下行情回落程度，低位下行情回弹程度
    """
    data = data.copy()
    
    HIGH = data.high.rolling(window=window).max()
    LOW = data.low.rolling(window=window).min()
    ratio = (data.close - LOW) / (HIGH - LOW)

    FACTOR = np.where(ratio>th, ((HIGH - data.close) / HIGH), np.where(ratio<1-th, -((data.close-LOW) / LOW), np.nan))
    return FACTOR

# fac 105
def PVcorr(data, window = 16):
    """
    趋势因子
    收盘价与成交量的量价相关性
    """
    data = data.copy()
    
    FACTOR = data.close.rolling(window=window).corr(data.volume) * data.close.pct_change(window) 
    return FACTOR

# fac 106
def term_rtn(data, data_next, window=5):
    """
    期限结构因子
    五根线上两个合约收益率之差
    """
    data = data.copy()
    data_czl = data_next.copy()
    data = data.merge(data_czl, on=['ts'],suffixes=('', '_czl'), how='left')
    FACTOR = -(data.close.diff(window) - data.close_czl.diff(window)) 
    return FACTOR

# fac 107
def KnifeReversal(data, window = 5):
    """
    反转因子
    捕捉价格快速下跌且成交量异常放大的"恐慌性抛售"机会
    """
    data = data.copy()
    FACTOR = data.close.pct_change(window) * (1-data.volume / data.volume.rolling(window=window).mean())
    return FACTOR

# fac 108
def IntradayEntropy(data,window1 = 5,window2 = 20):
    """
    行情标注因子
    判断当前收盘价在行情区域的位置
    """
    data = data.copy()
    
    FACTOR = (data.close - data.low.rolling(window=window1).min()) / (data.high.rolling(window=window1).max() 
                                                                      - data.low.rolling(window=window1).min()) * data.close.diff(window2)
    return FACTOR

# fac 109
def CapitalEfficiency(data, iqr_k=1.5, ret_window=10, rolling_window=60, th=0.75):
    """
    趋势因子
    衡量单位持仓量变动承担的收盘价变动
    """
    ret_close = data.close.pct_change(ret_window)
    ret_oi = data.OpenInterest.pct_change().rolling(ret_window).mean()

    factor = ret_close / ret_oi
    
    rolling_q1 = factor.shift(1).rolling(rolling_window, min_periods=10).quantile(1-th)  
    rolling_q3 = factor.shift(1).rolling(rolling_window, min_periods=10).quantile(th)

    rolling_iqr = rolling_q3 - rolling_q1
    lb = rolling_q1 - iqr_k * rolling_iqr
    ub = rolling_q3 + iqr_k * rolling_iqr
    
    return factor.clip(lb, ub)

# fac 110
def BuzzResonance(data, window1 = 5, window2 = 5, price_window=30):
    """
    反转因子
    长短期两家相关性相乘，如果趋势一致下值相反则发生反转
    """
    data = data.copy()
    mul1 = data.close.rolling(window = window1).corr(data.volume)
    mul2 = data.close.rolling(window = window2).corr(data.volume)
    FACTOR =  - mul1 * mul2 * data.close.pct_change(price_window)

    return FACTOR

# fac 111
def QST(data, window = 16):
    """
    行情特征因子
    开盘价收盘价滚动距离差之和
    """
    data = data.copy()
    
    O = data.open.rolling(window=window).sum()
    C = data.close.rolling(window=window).sum()
    FACTOR = (C-O)
    return FACTOR

# fac 112
def AR(data, window = 4):
    """
    行情特征因子
    滚动最高价减去最低价之和
    """
    data = data.copy()
    
    H = data.high.rolling(window=window).sum()
    L = data.low.rolling(window=window).sum()
    FACTOR = H - L
    return FACTOR

# fac 113
def CMFmin(data, window = 20):
    """
    趋势因子
    衡量中间价对于中位数价格的偏离程度
    """
    data = data.copy()

    TP = data.high/3 + data.close/3 + data.low/3
    CCI = (TP-TP.rolling(window=window).mean()) / (1+(TP-(TP.rolling(window=window).median())).abs())
    factor = CCI

    return factor

# fac 114
def VRSI(data, window=10):
    """
    趋势因子
    衡量量价一致性
    """
    data = data.copy()
    U = pd.Series(np.where(data.close.diff()>0, data.volume, np.where(data.close.diff()==0, data.volume/2, 0)))
    D = pd.Series(np.where(data.close.diff()<0, data.volume, np.where(data.close.diff()==0, data.volume/2, 0)))
    UU = (window-U.shift()+U)/window
    DD = (window-D.shift()+D)/window

    FACTOR = 100*DD/(UU+DD)
    return FACTOR

# fac 115
def KVO(data, window1 = 5, window2 = 20):
    """
    趋势因子
    长短期vwap之差
    """
    data = data.copy()
    data['VWAP'] = (data['volume'] * data['close']).rolling(window=window1).sum() / data['volume'].rolling(window=window1).sum()

    data['KVO_short'] = data['VWAP'].rolling(window=window1).mean()
    data['KVO_long'] = data['VWAP'].rolling(window=window2).mean()

    FACTOR = data['KVO_short'] - data['KVO_long'] 
    return FACTOR

# fac 116
def DDI(data, window=20):
    """
    趋势因子
    用于识别价格趋势强度和方向
    """
    data = data.copy()
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

    return FACTOR

# fac 117
def PV_corr_std(data, window=5):
    """
    波动因子
    量价相关性的波动率
    """
    data = data.copy()

    CORR = data.close.rolling(window=window).corr(data.volume)
    FACTOR = CORR.rolling(window=window).std()
    return FACTOR

# fac 118
def TS(data, window_ret=2, window_sum=14):
    """
    趋势因子
    离散化收盘价滚动和
    """
    data = data.copy()
    TS = np.where(data.close.rolling(window=window_ret).mean().diff()>0, 1, -1)
    FACTOR = pd.Series(TS).rolling(window=window_sum).sum()
    return FACTOR

# fac 119
def VHF(data):
    """
    波动率因子
    ATR滚动均值
    """
    data = data.copy()
    window = 10
    TR = np.maximum(data.high-data.low, np.maximum((data.high-data.close.shift()).abs(), (data.low-data.close.shift()).abs()))
    FACTOR = pd.Series(TR).rolling(window=window).mean()
    return FACTOR

# fac 120
def RPP_5D(data, window=60*4*5):
    """
    行情标注因子
    判断因子在五天内的位置
    """
    data = data.copy()
    H = data.high.rolling(window=window).max()
    L = data.low.rolling(window=window).min()
    FACTOR = (data.close - L) / (H - L)

    return FACTOR

# fac 121
def RPP_22D(data, window = 60*4*22):
    """
    行情标注因子
    判断因子在二十二天内的位置
    """
    data = data.copy()
    H = data.high.rolling(window=window).max()
    L = data.low.rolling(window=window).min()
    RPP = (data.close - L) / (H - L)
    FACTOR = RPP
    return FACTOR

# fac 122
def RPP_4H(data, window = 60*4):
    """
    行情标注因子
    判断因子在四小时内的位置
    """
    data = data.copy()

    H = data.high.rolling(window=window).max()
    L = data.low.rolling(window=window).min()
    RPP = (data.close - L) / (H - L)

    FACTOR = RPP
    return FACTOR

# fac 123
def RPP_2H(data, window = 60*2):
    """
    行情标注因子
    判断因子在2小时内的位置
    """
    data = data.copy()

    H = data.high.rolling(window=window).max()
    L = data.low.rolling(window=window).min()
    RPP = (data.close - L) / (H - L)

    FACTOR = RPP
    return FACTOR

# fac 124
def RPP_30M(data, window=30):
    """
    行情标注因子
    判断因子在30分钟内的位置
    """
    data = data.copy()

    H = data.high.rolling(window=window).max()
    L = data.low.rolling(window=window).min()
    RPP = (data.close - L) / (H - L)

    FACTOR = RPP
    return FACTOR

# fac 124
def RPP_10M(data, window=10):
    """
    行情标注因子
    判断因子在30分钟内的位置
    """
    data = data.copy()

    H = data.high.rolling(window=window).max()
    L = data.low.rolling(window=window).min()
    RPP = (data.close - L) / (H - L)

    FACTOR = RPP
    return FACTOR

# fac 124
def RPP_5M(data, window=5):
    """
    行情标注因子
    判断因子在5分钟内的位置
    """
    data = data.copy()

    H = data.high.rolling(window=window).max()
    L = data.low.rolling(window=window).min()
    RPP = (data.close - L) / (H - L)

    FACTOR = RPP
    return FACTOR

# fac 124
def RPP_3M(data, window=3):
    """
    行情标注因子
    判断因子在3分钟内的位置
    """
    data = data.copy()

    H = data.high.rolling(window=window).max()
    L = data.low.rolling(window=window).min()
    RPP = (data.close - L) / (H - L)

    FACTOR = RPP
    return FACTOR

# fac 131
def ptvol5(data, data1m_next, LAST_TRADE_DATE, window=60):
    """
    期限结构因子
    用主力合约和非主力合约估计现货价格的方差
    """
    data1m = data.merge(LAST_TRADE_DATE[['instrument', 'LAST_TRADE_DATE']], on=['instrument'])
    data1m_next = data1m_next.merge(LAST_TRADE_DATE[['instrument', 'LAST_TRADE_DATE']], on=['instrument'])
    data1m['days_to_trade'] = (pd.to_datetime(data1m.LAST_TRADE_DATE) - pd.to_datetime(data1m.date)).dt.days
    data1m_next['days_to_trade'] = (pd.to_datetime(data1m_next.LAST_TRADE_DATE) - pd.to_datetime(data1m_next.date)).dt.days

    data1m = data1m.merge(data1m_next[['ts', 'close', 'days_to_trade']], on=['ts'], how='left', suffixes=('', 'CZL'))
    data1m['n'] = data1m.ts.diff()
    data1m['n'] = data1m.n.dt.total_seconds() // 60
    data1m['pt'] = data1m['close'] * (data1m['close'] / 
                                    data1m['closeCZL']) ** (data1m.days_to_trade / (data1m.days_to_tradeCZL - data1m.days_to_trade))
    data1m['rtn'] = (np.log(data1m.pt) - np.log(data1m.pt.shift())) / (data1m.n)
    FACTOR = data1m['rtn'].rolling(window).std()    
    return FACTOR.round(6)

# fac 132
def JC1D(data, data1m_next, LAST_TRADE_DATE, window=60*4):
    """
    期限结构因子
    期限一日滚动溢价率
    """
    data1m = data.merge(LAST_TRADE_DATE[['instrument', 'LAST_TRADE_DATE']], on=['instrument'])
    data1m_next = data1m_next.merge(LAST_TRADE_DATE[['instrument', 'LAST_TRADE_DATE']], on=['instrument'])
    data1m['days_to_trade'] = (pd.to_datetime(data1m.LAST_TRADE_DATE) - pd.to_datetime(data1m.date)).dt.days
    data1m_next['days_to_trade'] = (pd.to_datetime(data1m_next.LAST_TRADE_DATE) - pd.to_datetime(data1m_next.date)).dt.days

    data1m = data1m.merge(data1m_next[['ts', 'close', 'days_to_trade']], on=['ts'], how='left', suffixes=('', 'CZL'))
    data1m['n'] = data1m.ts.diff()
    data1m['n'] = data1m.n.dt.total_seconds() // 60
    data1m['pt'] = data1m['close'] * (data1m['close'] / 
                                    data1m['closeCZL']) ** (data1m.days_to_trade / (data1m.days_to_tradeCZL - data1m.days_to_trade))
    FACTOR = ((data1m.close/data1m.pt - 1)*365/data1m.days_to_trade).rolling(window=window).sum()
    return FACTOR

# fac 132
def JC2H(data, data1m_next, LAST_TRADE_DATE, window=60*2):
    """
    期限结构因子
    期限一日滚动溢价率
    """
    data1m = data.merge(LAST_TRADE_DATE[['instrument', 'LAST_TRADE_DATE']], on=['instrument'])
    data1m_next = data1m_next.merge(LAST_TRADE_DATE[['instrument', 'LAST_TRADE_DATE']], on=['instrument'])
    data1m['days_to_trade'] = (pd.to_datetime(data1m.LAST_TRADE_DATE) - pd.to_datetime(data1m.date)).dt.days
    data1m_next['days_to_trade'] = (pd.to_datetime(data1m_next.LAST_TRADE_DATE) - pd.to_datetime(data1m_next.date)).dt.days

    data1m = data1m.merge(data1m_next[['ts', 'close', 'days_to_trade']], on=['ts'], how='left', suffixes=('', 'CZL'))
    data1m['n'] = data1m.ts.diff()
    data1m['n'] = data1m.n.dt.total_seconds() // 60
    data1m['pt'] = data1m['close'] * (data1m['close'] / 
                                    data1m['closeCZL']) ** (data1m.days_to_trade / (data1m.days_to_tradeCZL - data1m.days_to_trade))
    FACTOR = ((data1m.close/data1m.pt - 1)*365/data1m.days_to_trade).rolling(window=window).sum()
    return FACTOR

# fac 131
# def ptvol5(data, data1m_next, LAST_TRADE_DATE, window=60):
#     """
#     期限结构因子
#     用主力合约和非主力合约估计现货价格的方差
#     """
#     data1m = data.copy()
#     data1m['days_to_trade'] = (pd.to_datetime(data1m.LAST_TRADE_DATE) - pd.to_datetime(data1m.date)).dt.days
#     data1m_next['days_to_trade'] = (pd.to_datetime(data1m_next.LAST_TRADE_DATE) - pd.to_datetime(data1m_next.date)).dt.days

#     data1m = data1m.merge(data1m_next[['ts', 'close', 'days_to_trade']], on=['ts'], how='left', suffixes=('', 'CZL'))
#     data1m['n'] = data1m.ts.diff()
#     data1m['n'] = data1m.n.dt.total_seconds() // 60
#     data1m['pt'] = data1m['close'] * (data1m['close'] / 
#                                     data1m['closeCZL']) ** (data1m.days_to_trade / (data1m.days_to_tradeCZL - data1m.days_to_trade))
#     data1m['rtn'] = (np.log(data1m.pt) - np.log(data1m.pt.shift())) / (data1m.n)
#     FACTOR = data1m['rtn'].rolling(window).std()    
#     return FACTOR.round(6)

# fac 132
# def JC1D(data, data1m_next, LAST_TRADE_DATE, window=60*4):
#     """
#     期限结构因子
#     期限一日滚动溢价率
#     """
#     data1m = data.copy()
#     data1m['days_to_trade'] = (pd.to_datetime(data1m.LAST_TRADE_DATE) - pd.to_datetime(data1m.date)).dt.days
#     data1m_next['days_to_trade'] = (pd.to_datetime(data1m_next.LAST_TRADE_DATE) - pd.to_datetime(data1m_next.date)).dt.days

#     data1m = data1m.merge(data1m_next[['ts', 'close', 'days_to_trade']], on=['ts'], how='left', suffixes=('', 'CZL'))
#     data1m['n'] = data1m.ts.diff()
#     data1m['n'] = data1m.n.dt.total_seconds() // 60
#     data1m['pt'] = data1m['close'] * (data1m['close'] / 
#                                     data1m['closeCZL']) ** (data1m.days_to_trade / (data1m.days_to_tradeCZL - data1m.days_to_trade))
#     FACTOR = ((data1m.close/data1m.pt - 1)*365/data1m.days_to_trade).rolling(window=window).sum()
#     return FACTOR

# # fac 132
# def JC2H(data, data1m_next, LAST_TRADE_DATE, window=60*2):
#     """
#     期限结构因子
#     期限一日滚动溢价率
#     """
#     data1m = data.copy()
#     data1m['days_to_trade'] = (pd.to_datetime(data1m.LAST_TRADE_DATE) - pd.to_datetime(data1m.date)).dt.days
#     data1m_next['days_to_trade'] = (pd.to_datetime(data1m_next.LAST_TRADE_DATE) - pd.to_datetime(data1m_next.date)).dt.days

#     data1m = data1m.merge(data1m_next[['ts', 'close', 'days_to_trade']], on=['ts'], how='left', suffixes=('', 'CZL'))
#     data1m['n'] = data1m.ts.diff()
#     data1m['n'] = data1m.n.dt.total_seconds() // 60
#     data1m['pt'] = data1m['close'] * (data1m['close'] / 
#                                     data1m['closeCZL']) ** (data1m.days_to_trade / (data1m.days_to_tradeCZL - data1m.days_to_trade))
#     FACTOR = ((data1m.close/data1m.pt - 1)*365/data1m.days_to_trade).rolling(window=window).sum()
#     return FACTOR

# fac 133
def XSMOM1M(data, window = 60*4*22):
    """
    趋势因子
    收盘价计算的月度收益率
    """
    data = data.copy()
    
    FACTOR = data.close.pct_change(window)
    return FACTOR

# fac 135
def CCI(data):
    """
    趋势因子
    用typical价格衡量的均线偏离度
    """
    data = data.copy()
    HLC = data.high/3 + data.low/3 + data.close/3
    HLCma = HLC.rolling(window=120).sum() / 120
    AVEDEV = (HLC - HLCma).abs().rolling(window=120).sum() / 120
    CCI = (HLC - HLCma) / (0.015*AVEDEV + 1)
    FACTOR = CCI
    return FACTOR

# fac 136
def HP(data):
    """
    持仓量因子
    持仓量之差除以滚动成交量之和
    """
    data = data.copy()
    window = 20
    FACTOR = data.OpenInterest.diff(window) / data.volume.rolling(window=window).sum()
    return FACTOR

# fac 137
def ZCpriceinterval(data1m, data1m_next, LAST_TRADE_DATE):
    """
    期限结构因子
    展期收益率
    """
    data1m = data1m.merge(LAST_TRADE_DATE[['instrument', 'LAST_TRADE_DATE']], on=['instrument'])
    data1m_next = data1m_next.merge(LAST_TRADE_DATE[['instrument', 'LAST_TRADE_DATE']], on=['instrument'])
    data1m['days_to_trade'] = (pd.to_datetime(data1m.LAST_TRADE_DATE) - pd.to_datetime(data1m.date)).dt.days
    data1m_next['days_to_trade'] = (pd.to_datetime(data1m_next.LAST_TRADE_DATE) - pd.to_datetime(data1m_next.date)).dt.days

    data1m = data1m.merge(data1m_next[['ts', 'close', 'days_to_trade']], on=['ts'], how='left', suffixes=('', 'CZL'))
    FACTOR = (data1m.closeCZL / data1m.close - 1) * 365 / (data1m.days_to_tradeCZL - data1m.days_to_trade)
    return FACTOR

# fac 138
def SNR(data, window = 10):
    """
    波动率因子
    每分钟收盘对开盘的收益率绝对值滚动和
    """
    data = data.copy()
    
    FACTOR = np.log(data.close / data.open).abs().rolling(window=window).sum()
    return FACTOR

# fac 139
def dOI(data, iqr_k=1.5, window = 20, rolling_window=20, th=0.75):
    """
    持仓量因子
    持仓量趋势变化
    """
    data = data.copy()
    
    absRetnight = np.log(data.OpenInterest) - np.log(data.OpenInterest.shift(window))
    factor = absRetnight  
    rolling_q1 = factor.shift(1).rolling(rolling_window).quantile(1-th)  # 关键：shift(1)
    rolling_q3 = factor.shift(1).rolling(rolling_window).quantile(th)
    rolling_iqr = rolling_q3 - rolling_q1
    lb = rolling_q1 - iqr_k * rolling_iqr
    ub = rolling_q3 + iqr_k * rolling_iqr
    return factor.clip(lb, ub)

# fac 140
def RSImin(data, window = 20):
    """
    反转因子
    RSI技术指标
    """
    data = data.copy()
    
    FACTOR = pd.Series(np.maximum(data.close.diff(), 0)).rolling(window=window).mean() / data.close.diff().abs().rolling(window=window).mean()
    return FACTOR

# fac 143
def CR(data, window = 240):
    """
    行情标注因子
    typical价格的上下界空间之比
    """
    data = data.copy()
    
    HLCC = data.high/3 + data.low/3 + data.close/3
    CR1 = pd.Series(np.maximum(0, data.high - HLCC)).rolling(window=window).mean()
    CR2 = pd.Series(np.maximum(0, HLCC - data.low)).rolling(window=window).mean()

    FACTOR = CR1 / CR2
    return FACTOR

# fac 144
def CV(data, window = 40, iqr_k=1.5, rolling_window=60, th=0.75):
    """
    反转因子
    收益率夏普分之一
    """
    data = data.copy()
    
    factor = data.close.pct_change().rolling(window=window).std() / data.close.pct_change().rolling(window=window).mean().abs()
    rolling_q1 = factor.shift(1).rolling(rolling_window).quantile(1-th)  
    rolling_q3 = factor.shift(1).rolling(rolling_window).quantile(th)
    rolling_iqr = rolling_q3 - rolling_q1
    lb = rolling_q1 - iqr_k * rolling_iqr
    ub = rolling_q3 + iqr_k * rolling_iqr
    return factor.clip(lb, ub)

# fac 145
def LR(data, window=60):
    """
    趋势因子
    单位成交量的收盘价变化，均值变化率
    """
    data = data.copy()
    FACTOR = (data.close.diff().abs() / data.volume).rolling(window=window).mean()
    FACTOR = (FACTOR-FACTOR.rolling(window=window).mean()) / FACTOR.rolling(window=window).std()
    return FACTOR

# fac 146
def signmom(data, window = 60):
    """
    趋势因子
    离散化涨跌
    """
    data = data.copy()
    sign = pd.Series(np.maximum(0,np.sign(data.close.diff()))).rolling(window=window).mean()
    FACTOR = sign
    return FACTOR

# fac 147
def VR(data, window = 240):
    """
    量价因子
    上行成交量除以下行成交量
    """
    data = data.copy()
    FACTOR = pd.Series(np.where(data.close.diff()>0, data.volume, 0)).rolling(window=window).sum() / pd.Series(np.where(data.close.diff()<=0, data.volume, 0)).rolling(window=window).sum()
    FACTOR = (FACTOR-FACTOR.rolling(window=window).mean()) / FACTOR.rolling(window=window).std()
    return FACTOR

# fac 148
def Chande(data,window = 30):
    """
    动量因子
    钱德动量
    """
    data = data.copy()
    
    SU = pd.Series(np.where(data.close.diff()>0, data.close.diff(), 0)).rolling(window=window).sum()
    SD = pd.Series(np.where(data.close.diff()<0, -data.close.diff(), 0)).rolling(window=window).sum()
    FACTOR = (SU-SD) / (SU+SD) * 100
    return FACTOR

# fac 149
def masign(data, window = 60):
    """
    动量因子
    统计相邻均线上穿下穿次数
    """
    data = data.copy()
    
    tmp = np.zeros(data.shape[0])
    for i in range(1, window):
        mean_i = data.close.rolling(window=i).mean()
        mean_i_plus_1 = data.close.rolling(window=i+1).mean()
        
        tmp += np.sign(mean_i - mean_i_plus_1).fillna(0)  

    FACTOR = tmp
    return FACTOR

# fac 150
def ATR(data, window = 60):
    """
    波动率指标
    滚动波动率偏离度
    """
    data = data.copy()
    TR = np.maximum((data.high-data.close.shift()).abs(), np.maximum((data.low-data.close.shift()).abs(), data.high - data.close))
    FACTOR = pd.Series(TR).rolling(window=window).mean()
    FACTOR = (FACTOR-FACTOR.rolling(window=window).mean()) / FACTOR.rolling(window=window).std()

    return FACTOR

# fac 151
def skew(data, window = 60, iqr_k=1.5, rolling_window=20, th=0.75):
    """
    收益率结构因子
    收益率偏度
    """
    data = data.copy()
    factor = (((data.close.pct_change() - data.close.pct_change().shift().rolling(window=window).mean()) / data.close.pct_change().shift().rolling(window=window).std()) ** 3)
    rolling_q1 = factor.shift(1).rolling(rolling_window).quantile(1-th)  # 关键：shift(1)
    rolling_q3 = factor.shift(1).rolling(rolling_window).quantile(th)
    rolling_iqr = rolling_q3 - rolling_q1
    lb = rolling_q1 - iqr_k * rolling_iqr
    ub = rolling_q3 + iqr_k * rolling_iqr
    return factor.clip(lb, ub)   

# fac 152
def trendstrength(data, window = 20):
    """
    波动率因子
    趋势强度
    """
    data = data.copy()
    FACTOR = data.close.diff(window) / (data.close.diff().abs()).rolling(window=window).sum()
    return FACTOR

# fac 153
def DDI(data, window = 20):
    """
    趋势因子
    趋势强度与方向识别
    """
    data = data.copy()
    
    DMZ = np.where(data.high.diff()+data.low.diff()<=0, 0, np.maximum(data.high.diff().abs(), data.low.diff().abs()))
    DMF = np.where(data.high.diff()+data.low.diff()>0, 0, np.maximum(data.high.diff().abs(), data.low.diff().abs()))
    DIZ = pd.Series(DMZ).rolling(window=window).sum() / (pd.Series(DMZ).rolling(window=window).sum() + pd.Series(DMF).rolling(window=window).sum())
    DIF = pd.Series(DMF).rolling(window=window).sum() / (pd.Series(DMZ).rolling(window=window).sum() + pd.Series(DMF).rolling(window=window).sum())

    FACTOR = DIZ - DIF
    return FACTOR

# fac 154
def PriceFilter_f5(data, window=20, rg_ratio = 1.0):
    """
    反转因子
    动态滤波
    """
    # 计算波动率
    if 'volatility' not in data.columns:
        data['volatility'] = data['close'].rolling(window).std()
    
    # 初始化
    plist = []
    plist_each = []
    
    for i in range(len(data)):
        p = data['close'].iloc[i]
        pre_p = data['pre_close'].iloc[i]
        pre2_p = data['pre2_close'].iloc[i]
        vol = data['volatility'].iloc[i]
        
        # 初始处理
        if pd.isna(pre2_p):
            plist_each.append(np.nan)
            continue
            
        # 初始化关键点序列
        if i == 0:
            plist = [pre2_p, pre_p]
            plist_each.append(np.nan)
            continue
            
        # 趋势判断
        vol_threshold = rg_ratio * vol
        last_p = plist[-1]
        prev_p = plist[-2]
        
        # 同方向运动
        if (pre_p - last_p) * (last_p - prev_p) > 0:
            plist[-1] = pre_p  # 更新终点
        # 反转判定
        elif abs(pre_p - last_p) > vol_threshold:
            plist.append(pre_p)
            
        # 记录前一个关键点
        plist_each.append(plist[-2] if len(plist) >= 2 else np.nan)
    
    return pd.Series(plist_each, index=data.index)

# fac 154
def kurt(data, window = 20, iqr_k=3, rolling_window=60):
    """
    收益率特征因子
    收益率峰度
    """
    data = data.copy()
    factor = (((data.close.pct_change() - data.close.pct_change().shift().rolling(window=window).mean()) / data.close.pct_change().shift().rolling(window=window).std()) ** 4)
    rolling_q1 = factor.shift(1).rolling(rolling_window).quantile(0.25)  # 关键：shift(1)
    rolling_q3 = factor.shift(1).rolling(rolling_window).quantile(0.75)
    rolling_iqr = rolling_q3 - rolling_q1
    lb = rolling_q1 - iqr_k * rolling_iqr
    ub = rolling_q3 + iqr_k * rolling_iqr
    factor = factor.clip(lb, ub) 
    factor = (factor-factor.rolling(window=window).mean()) / factor.rolling(window=window).std()

    return factor  

# fac 155
def ACD(data,window= 30,iqr_k=3, rolling_window=60):
    """
    趋势因子
    """
    data = data.copy()
    DIF = np.where(data.close.diff()>0, np.minimum(data.low, data.close.shift()), np.maximum(data.high, data.close.shift()))
    factor = data.close.diff().values * pd.Series(np.where(data.close.diff()==0, 0, DIF)).rolling(window=window).sum().diff()

    rolling_q1 = factor.shift(1).rolling(rolling_window).quantile(0.25)  # 关键：shift(1)
    rolling_q3 = factor.shift(1).rolling(rolling_window).quantile(0.75)
    rolling_iqr = rolling_q3 - rolling_q1
    lb = rolling_q1 - iqr_k * rolling_iqr
    ub = rolling_q3 + iqr_k * rolling_iqr
    factor = factor.clip(lb, ub) 
    factor = (factor-factor.rolling(window=window).mean()) / factor.rolling(window=window).std()

    return factor.clip(lb, ub) 

# fac 156
def up_shadow_5mean(data, window = 5):
    """
    技术指标
    五根线上影线均值
    """
    data = data.copy()
    up_shadow = (data.high - np.maximum(data.open, data.close)) / (np.maximum(data.open, data.close) - np.minimum(data.open, data.close) + 1)
    FACTOR = up_shadow.rolling(window=window).mean()

    return FACTOR

# fac 158
def down_shadow_5mean(data, window = 5):
    """
    技术指标
    五根线下影线均值
    """
    data = data.copy()
    down_shadow = -(data.low - np.minimum(data.open, data.close)) / (np.maximum(data.open, data.close) - np.minimum(data.open, data.close) + 1)
    FACTOR = down_shadow.rolling(window=window).mean()

    return FACTOR

# fac 159
def up_shadow_5std(data, window = 5):
    """
    技术指标
    五根线上影线标准差
    """
    up_shadow = (data.high - np.maximum(data.open, data.close)) / (np.maximum(data.open, data.close) - np.minimum(data.open, data.close) + 1)
    FACTOR = up_shadow.rolling(window=window).std()

    return FACTOR

# fac 160
def down_shadow_5std(data, window = 5):
    """
    技术指标
    五根线下影线标准差
    """
    data = data.copy()
    down_shadow = -(data.low - np.minimum(data.open, data.close)) / (np.maximum(data.open, data.close) - np.minimum(data.open, data.close) + 1)
    FACTOR = down_shadow.rolling(window=window).std()

    return FACTOR

# fac 161
def day_jump(data):
    """
    日线特征因子
    开盘跳价判断
    """
    def day_open(group):
        if group.shape[0]<60:
            return np.nan
        else:
            first_price = group.open.iloc[0]

            return first_price

    def day_close(group):
        if group.shape[0]<60:
            return np.nan
        else:
            last_price = group.close.iloc[-3:].mean()

            return last_price
    jump_today = data.groupby('date').apply(day_open) - data.groupby('date').apply(day_close).shift()
    FACTOR = jump_today.reset_index(name='fac').merge(data, on='date',how='right').fac
    return FACTOR

# fac 161
def day_first3power(data, use_info_bar = 3):
    """
    日线特征因子
    开盘前三根分钟线特征
    """

    def calc_fac(group):
        if len(group)<60:
            return np.nan
        else:
            first_price = (group.close - group.open) 
            return first_price.iloc[:use_info_bar].mean()
        
    FACTOR = data.groupby('date').apply(calc_fac).reset_index(name='fac').merge(data, on='date',how='right')
    def mask_first(group, use_info_bar):
        group.loc[group.index[:use_info_bar], 'fac'] = np.nan  
        return group
    
    FACTOR = FACTOR.groupby('date', group_keys=False).apply(lambda group: mask_first(group,use_info_bar))
    
    return FACTOR.fac

# fac 161 新
def day_first6power(data, use_info_bar = 6):
    """
    日线特征因子
    开盘前六根分钟线特征
    """

    def calc_fac(group):
        if len(group)<60:
            return np.nan
        else:
            first_price = (group.close - group.open) 
            return first_price.iloc[:use_info_bar].mean()
        
    FACTOR = data.groupby('date').apply(calc_fac).reset_index(name='fac').merge(data, on='date',how='right')
    def mask_first(group, use_info_bar):
        group.loc[group.index[:use_info_bar], 'fac'] = np.nan  
        return group
    
    FACTOR = FACTOR.groupby('date', group_keys=False).apply(lambda group: mask_first(group,use_info_bar))
    
    return FACTOR.fac

# fac 161 新
def day_first10power(data, use_info_bar = 10):
    """
    日线特征因子
    开盘前十根分钟线特征
    """

    def calc_fac(group):
        if len(group)<60:
            return np.nan
        else:
            first_price = (group.close - group.open) 
            return first_price.iloc[:use_info_bar].mean()
        
    FACTOR = data.groupby('date').apply(calc_fac).reset_index(name='fac').merge(data, on='date',how='right')
    def mask_first(group, use_info_bar):
        group.loc[group.index[:use_info_bar], 'fac'] = np.nan  
        return group
    
    FACTOR = FACTOR.groupby('date', group_keys=False).apply(lambda group: mask_first(group,use_info_bar))
    
    return FACTOR.fac

# fac 161 新
def day_first10colarrate(data, use_info_bar = 10):
    """
    日线特征因子
    开盘前十根分钟线特征
    """

    def calc_fac(group):
        if len(group)<60:
            return np.nan
        else:
            first_N = group.head(use_info_bar)
            is_positive = (first_N.close > first_N.open)
            positive_ratio = is_positive.mean()
            return positive_ratio
    FACTOR = data.groupby('date').apply(calc_fac).reset_index(name='fac').merge(data, on='date',how='right')
    def mask_first(group, use_info_bar):
        group.loc[group.index[:use_info_bar], 'fac'] = np.nan  
        return group
    
    FACTOR = FACTOR.groupby('date', group_keys=False).apply(lambda group: mask_first(group,use_info_bar))
    
    return FACTOR.fac

# fac 161 新
def day_first10rev(data, use_info_bar = 10):
    """
    日线特征因子
    开盘前十根分钟线特征
    """

    def calc_fac(group):
        if len(group)<60:
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
        
    FACTOR = data.groupby('date').apply(calc_fac).reset_index(name='fac').merge(data, on='date',how='right')
    def mask_first(group, use_info_bar):
        group.loc[group.index[:use_info_bar], 'fac'] = np.nan  
        return group
    
    FACTOR = FACTOR.groupby('date', group_keys=False).apply(lambda group: mask_first(group,use_info_bar))
    
    return FACTOR.fac

# fac 162
def day_first4redcorr(data, use_info_bar = 4):
    """
    日线特征因子
    开盘前四根分钟线特征
    """
    def calc_fac(group):
        if group.shape[0]<60:
            return np.nan
        else:
            first_price = np.maximum(group.close - group.open,0).iloc[:use_info_bar].corr(group.volume.iloc[:use_info_bar])
            return first_price
        
    FACTOR = data.groupby('date').apply(calc_fac).reset_index(name='fac').merge(data, on='date',how='right')
    
    def mask_first(group, use_info_bar):
        group.loc[group.index[:use_info_bar], 'fac'] = np.nan 
        return group
    
    FACTOR = FACTOR.groupby('date', group_keys=False).apply(lambda group: mask_first(group,use_info_bar))
    
    return FACTOR.fac

# fac 163
def day_first4greencorr(data, use_info_bar = 4):
    """
    日线特征因子
    开盘前四根分钟线特征
    """
    
    def calc_fac(group):
        if group.shape[0]<60:
            return np.nan
        else:
            first_price = -np.minimum(group.close - group.open,0).iloc[:use_info_bar].corr(group.volume.iloc[:use_info_bar])
            return first_price
        
    FACTOR = data.groupby('date').apply(calc_fac).reset_index(name='fac').merge(data, on='date',how='right')
    def mask_first(group, use_info_bar):
        group.loc[group.index[:use_info_bar], 'fac'] = np.nan  
        return group
    
    FACTOR = FACTOR.groupby('date', group_keys=False).apply(lambda group: mask_first(group,use_info_bar))
    
    return FACTOR.fac

# fac 164
def OFI5(data, window=5, iqr_k=3, rolling_window=60):
    """
    买卖结构因子
    """
    data = data.copy()
    factor = (data.buy_volume.rolling(window=window).sum() - data.sell_volume.rolling(window=window).sum()) * data.close.pct_change(window)

    rolling_q1 = factor.shift(1).rolling(rolling_window, min_periods=10).quantile(0.25)  # 关键：shift(1)
    rolling_q3 = factor.shift(1).rolling(rolling_window, min_periods=10).quantile(0.75)
    rolling_iqr = rolling_q3 - rolling_q1
    lb = rolling_q1 - iqr_k * rolling_iqr
    ub = rolling_q3 + iqr_k * rolling_iqr
    factor = factor.clip(lb, ub) 
    factor = (factor-factor.rolling(window=window).mean()) / (0.001 + factor.rolling(window=window).std())
    return factor

# fac 165
def OFI60(data, window=60, iqr_k=3, rolling_window=60):
    """
    买卖结构因子
    """
    data = data.copy()
    factor = (data.buy_volume.rolling(window=window).sum() - data.sell_volume.rolling(window=window).sum()) * data.close.pct_change(window)

    rolling_q1 = factor.shift(1).rolling(rolling_window, min_periods=10).quantile(0.25)  # 关键：shift(1)
    rolling_q3 = factor.shift(1).rolling(rolling_window, min_periods=10).quantile(0.75)
    rolling_iqr = rolling_q3 - rolling_q1
    lb = rolling_q1 - iqr_k * rolling_iqr
    ub = rolling_q3 + iqr_k * rolling_iqr
    factor = factor.clip(lb, ub) 
    factor = (factor-factor.rolling(window=window).mean()) / (0.001 + factor.rolling(window=window).std())

    return factor

def zigzag(data):
    """
    转换DDB因子计算到Python实现
    输入需包含列：ts, instrument, date, open, close, high, low, volume, turnover
    """
    data = data.sort_values(['ts', 'instrument'])
    data['count_bar'] = 1
    data['count_bar'] = data.groupby(['instrument', 'date'])['count_bar'].cumsum()
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

    def price_filter_f5(data):
        results = []
        for (instrument, date), group in data.groupby(['instrument', 'date']):
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
    
    # 这里可能有误
    data['plast'] = data.groupby(['instrument', 'date']).apply(price_filter_f5).reset_index(level=[0,1], drop=True)
    factor = (data['close'] - data['plast']) / data['volatility_rg']
    
    return factor

def volatility_rg(data):
    """
    转换DDB因子计算到Python实现
    输入需包含列：ts, instrument, date, open, close, high, low, volume, turnover
    """
    data = data.sort_values(['ts', 'instrument'])
    data['count_bar'] = 1
    data['count_bar'] = data.groupby(['instrument', 'date'])['count_bar'].cumsum()
    data['prev_close'] = data.groupby('instrument')['close'].shift(1)
    
    def rolling_percentile(s, p, window, min_periods):
        return s.rolling(window, min_periods=min_periods).quantile(p/100)
    
    factor = data.groupby(['instrument'])['close'].transform(
        lambda x: (rolling_percentile(x, 95, 15, 10) - 
                   rolling_percentile(x, 5, 15, 10)) / 24.0
    )
    
    return factor

def before_bot_price_diff(data: pd.DataFrame) -> pd.Series:
    """
    按结算日分组计算价格偏离度因子
    参数:
        data: 包含close, volume, date的DataFrame
    返回:
        与输入同长度的因子值Series
    """
    def _calculate_group(group: pd.DataFrame) -> np.ndarray:
        close = group['close'].values
        volume = group['volume'].values
        n = len(close)
        
        # 快速路径：数据不足时返回NaN
        if n < 4 or np.all(volume <= 0):
            return np.full(n, np.nan)
        
        # 计算每个时点的历史最低成交量位置
        valley_pos = np.zeros(n, dtype=int)
        for i in range(1, n):
            mask = volume[:i] > 0
            if np.any(mask):
                valley_pos[i] = np.argmin(volume[:i][mask])
            else:
                valley_pos[i] = i
        
        # 向量化计算前3日均值
        starts = np.maximum(0, valley_pos - 3)

        means = np.array([
            np.mean(close[s:i]) if i > s else np.nan
            for s, i in zip(starts, valley_pos)
        ])
        # 计算全局均值（使用expanding mean优化）
        global_means = np.cumsum(close) / np.arange(1, n+1)
        return means / global_means - 1.0

    # 按结算日分组计算
    return data.groupby('date', group_keys=False).apply(
        lambda g: pd.Series(_calculate_group(g), index=g.index)
    )

def bar3_trend_corr(data):
    @njit
    def spearman_rank(trend, ranks):
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
    def _rolling_corr(close):
        n = len(close)
        result = np.full(n, np.nan)
        trend = np.arange(3)
        
        for i in range(2, n):
            window = close[i-2:i+1]
            ranks = np.argsort(np.argsort(window)) + 1
            result[i] = round(spearman_rank(trend, ranks),2)
        
        return result
    return _rolling_corr(data['close'].values
    )

def bar5_trend_corr(data):
    @njit
    def spearman_rank(trend, ranks):
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
    def _rolling_corr(close):
        n = len(close)
        result = np.full(n, np.nan)
        trend = np.arange(5)
        
        for i in range(4, n):
            window = close[i-4:i+1]
            ranks = np.argsort(np.argsort(window)) + 1
            result[i] = round(spearman_rank(trend, ranks),2)
        
        return result
    return _rolling_corr(data['close'].values
    )
