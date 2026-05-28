import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

def wma(series, window):
    weights = np.arange(window, 0, -1)  # 线性递减权重（如12,11,...,1）
    return series.rolling(window).apply(lambda x: np.sum(x * weights) / weights.sum(), raw=True)


# tick data tick factors
#-------------------------------------------------------------------------------------------------------------------------------------
# fac1
def atr(data, window=14, name='atr'):
    data = data.copy()
    data[name] = np.nan

    tr = np.maximum(data.HighPrice - data.LowPrice, (data.HighPrice - data.mid_price.shift(1)).abs(), (data.LowPrice - data.mid_price.shift(1)).abs())

    data[name] = tr.rolling(window=window).mean().diff(60*2).clip(lower=0, upper=20) 
    data['time'] = data.datetime
    return data[['time', 'date', name]]

# fac2
def RVar(data, window=120, name='RVar'):
    data = data.copy()
    data[name] = np.nan

    data['M'] = data.mid_price
    Rvar = (data.M.pct_change()**2).rolling(window=window).sum()

    data[name] = (Rvar * 1e5).clip(lower=0, upper=1)  
    data['time'] = data.datetime
    return data[['time', 'date', name]]

# fac3
def RSkew(data, window=120, name='RSkew'):
    data = data.copy()
    data[name] = np.nan

    data['M'] = data.mid_price
    RSkew = (data.M.diff()**3).rolling(window=window, min_periods=100).sum() * np.sqrt(window) / (data.M.diff()**2).rolling(window=window, min_periods=100).sum()**1.5

    data[name] = RSkew
    data['time'] = data.datetime
    return data[['time', 'date', name]]

# fac4
def RKurt(data, window=120, name='RKurt'):
    data = data.copy()
    data[name] = np.nan

    data['M'] = data.mid_price
    RKurt = (data.M.diff()**4).rolling(window=window, min_periods=100).sum() * window / (data.M.diff()**2).rolling(window=window, min_periods=100).sum()**2

    data[name] = RKurt
    data['time'] = data.datetime
    return data[['time', 'date', name]]

# fac5
def RVar_down_rate(data, window=120, name='RVar_down_rate'):
    data = data.copy()
    data[name] = np.nan

    data['M'] = data.mid_price
    RVar_down = (data.M.diff().apply(lambda x: 0 if x > 0 else x)**2).rolling(window=window).sum()    
    RVar = (data.M.diff()**2).rolling(window=window).sum() 
    RVar_down_rate = RVar_down / RVar

    data[name] = RVar_down_rate
    data['time'] = data.datetime
    return data[['time', 'date', name]]

# fac6
def ask1_vmean_20(data, window=20, name='ask1_vmean_20'):
    data = data.copy()
    data[name] = np.nan

    ask1_vmean_20 = data.AskVolume1.rolling(window=window).sum()    

    data[name] = ask1_vmean_20 / data.mid_price.rolling(window=window).mean()
    data['time'] = data.datetime
    return data[['time', 'date', name]]

# fac7
def ask2_vmean_20(data, window=20, name='ask2_vmean_20'):
    data = data.copy()
    data[name] = np.nan

    ask2_vmean_20 = data.AskVolume2.rolling(window=window).sum()    
    data[name] = ask2_vmean_20 / data.mid_price.rolling(window=window).mean()
    
    data['time'] = data.datetime
    return data[['time', 'date', name]]

# fac8
def ask3_vmean_20(data, window=20, name='ask3_vmean_20'):
    data = data.copy()
    data[name] = np.nan

    ask3_vmean_20 = data.AskVolume3.rolling(window=window).sum()    

    data[name] = ask3_vmean_20 / data.mid_price.rolling(window=window).mean()
    data['time'] = data.datetime
    return data[['time', 'date', name]]

# fac9
def ask4_vmean_20(data, window=20, name='ask4_vmean_20'):
    data = data.copy()
    data[name] = np.nan

    ask4_vmean_20 = data.AskVolume4.rolling(window=window).sum()    

    data[name] = ask4_vmean_20 / data.mid_price.rolling(window=window).mean()
    data['time'] = data.datetime
    return data[['time', 'date', name]]

# fac10
def ask5_vmean_20(data, window=20, name='ask5_vmean_20'):
    data = data.copy()
    data[name] = np.nan

    ask5_vmean_20 = data.AskVolume5.rolling(window=window).sum()    

    data[name] = ask5_vmean_20 / data.mid_price.rolling(window=window).mean()
    data['time'] = data.datetime
    return data[['time', 'date', name]]

# fac11
def bid1_vmean_20(data, window=20, name='bid1_vmean_20'):
    data = data.copy()
    data[name] = np.nan

    bid1_vmean_20 = data.BidVolume1.rolling(window=window).sum()    

    data[name] = bid1_vmean_20 / data.mid_price.rolling(window=window).mean()
    data['time'] = data.datetime
    return data[['time', 'date', name]]

# fac12
def bid2_vmean_20(data, window=20, name='bid2_vmean_20'):
    data = data.copy()
    data[name] = np.nan

    bid2_vmean_20 = data.BidVolume2.rolling(window=window).sum()    

    data[name] = bid2_vmean_20 / data.mid_price.rolling(window=window).mean()
    data['time'] = data.datetime
    return data[['time', 'date', name]]

# fac13
def bid3_vmean_20(data, window=20, name='bid3_vmean_20'):
    data = data.copy()
    data[name] = np.nan

    bid3_vmean_20 = data.BidVolume3.rolling(window=window).sum()    

    data[name] = bid3_vmean_20 / data.mid_price.rolling(window=window).mean()
    data['time'] = data.datetime
    return data[['time', 'date', name]]

# fac14
def bid4_vmean_20(data, window=20, name='bid4_vmean_20'):
    data = data.copy()
    data[name] = np.nan

    bid4_vmean_20 = data.BidVolume4.rolling(window=window).sum()    

    data[name] = bid4_vmean_20 / data.mid_price.rolling(window=window).mean()
    data['time'] = data.datetime
    return data[['time', 'date', name]]

# fac15
def bid5_vmean_20(data, window=20, name='bid5_vmean_20'):
    data = data.copy()
    data[name] = np.nan

    bid5_vmean_20 = data.BidVolume5.rolling(window=window).sum()    

    data[name] = bid5_vmean_20 / data.mid_price.rolling(window=window).mean()
    data['time'] = data.datetime
    return data[['time', 'date', name]]

# fac16
def sub_a1b1_vmean_20(data, window=20, name='sub_a1b1_vmean_20'):
    data = data.copy()
    data[name] = np.nan

    sub_a1b1_vmean_20 = data.AskVolume1.rolling(window=window).sum() - data.BidVolume1.rolling(window=window).sum()

    data[name] = sub_a1b1_vmean_20
    data['time'] = data.datetime
    return data[['time', 'date', name]]

# fac17
def sub_a2b2_vmean_20(data, window=20, name='sub_a2b2_vmean_20'):
    data = data.copy()
    data[name] = np.nan

    sub_a2b2_vmean_20 = data.AskVolume2.rolling(window=window).sum() - data.BidVolume2.rolling(window=window).sum()

    data[name] = sub_a2b2_vmean_20
    data['time'] = data.datetime
    return data[['time', 'date', name]]

# fac18
def sub_a3b3_vmean_20(data, window=20, name='sub_a3b3_vmean_20'):
    data = data.copy()
    data[name] = np.nan

    sub_a3b3_vmean_20 = data.AskVolume3.rolling(window=window).sum() - data.BidVolume3.rolling(window=window).sum()

    data[name] = sub_a3b3_vmean_20
    data['time'] = data.datetime
    return data[['time', 'date', name]]

# fac19
def sub_a4b4_vmean_20(data, window=20, name='sub_a4b4_vmean_20'):
    data = data.copy()
    data[name] = np.nan

    sub_a4b4_vmean_20 = data.AskVolume4.rolling(window=window).sum() - data.BidVolume4.rolling(window=window).sum()

    data[name] = sub_a4b4_vmean_20
    data['time'] = data.datetime
    return data[['time', 'date', name]]

# fac20
def sub_a5b5_vmean_20(data, window=20, name='sub_a5b5_vmean_20'):
    data = data.copy()
    data[name] = np.nan

    sub_a5b5_vmean_20 = data.AskVolume5.rolling(window=window).sum() - data.BidVolume5.rolling(window=window).sum()

    data[name] = sub_a5b5_vmean_20
    data['time'] = data.datetime
    return data[['time', 'date', name]]

# fac21
def money_flow_power(data, window=20, name='money_flow_power'):
    data = data.copy()
    data[name] = np.nan

    money_flow_a = data.AskVolume1 * data.AskPrice1\
    + data.AskVolume2 * data.AskPrice2\
    + data.AskVolume3 * data.AskPrice3\
    + data.AskVolume4 * data.AskPrice4\
    + data.AskVolume5 * data.AskPrice5

    money_flow_b = data.BidVolume1 * data.BidPrice1\
    + data.BidVolume2 * data.BidPrice2\
    + data.BidVolume3 * data.BidPrice3\
    + data.BidVolume4 * data.BidPrice4\
    + data.BidVolume5 * data.BidPrice5

    money_flow_power = (money_flow_a - money_flow_b).rolling(window=window).sum() / (money_flow_a - money_flow_b).abs().rolling(window=window).sum()

    data[name] = money_flow_power
    data['time'] = data.datetime
    return data[['time', 'date', name]]

# fac22
def MPC(data, window=5, name='MPC'):
    data = data.copy()
    data[name] = np.nan

    MPC = (data.AskPrice1 + data.BidPrice1).pct_change(window)

    data[name] = MPC
    data['time'] = data.datetime
    return data[['time', 'date', name]]

# fac23
def MPB(data, instrument=None, name="MPB"):
    data = data.copy()
    data[name] = np.nan

    import re
    import function_future.DataLoader as DL
    M = data.mid_price
    
    symbol = re.sub(r'\d+$', '', instrument).upper()
    contract_multiplier = DL.InstrumentConfig().get_instrument_config(symbol)['contract_multiplier']

    TP = data['turnover'] / (data['volume'] * contract_multiplier)
    data[name] = TP - M
    data['time'] = data.datetime
    return data[['time', 'date', name]]

# fac24
def PVcorrsub_a1b1(data, window=20, name="PVcorrsub_a1b1"):
    data = data.copy()
    data[name] = np.nan

    PVcorrsub_a1b1 = data.AskPrice1.rolling(window=window).corr(data.AskVolume1) - data.BidPrice1.rolling(window=window).corr(data.BidVolume1)

    data[name] = PVcorrsub_a1b1
    data['time'] = data.datetime
    return data[['time', 'date', name]]

# fac25
def PVcorrsub_a2b2(data, window=20, name="PVcorrsub_a2b2"):
    data = data.copy()
    data[name] = np.nan

    PVcorrsub_a2b2 = data.AskPrice2.rolling(window=window).corr(data.AskVolume2) - data.BidPrice2.rolling(window=window).corr(data.BidVolume2)

    data[name] = PVcorrsub_a2b2
    data['time'] = data.datetime
    return data[['time', 'date', name]]

# fac26
def PVcorrsub_a3b3(data, window=20, name="PVcorrsub_a3b3"):
    data = data.copy()
    data[name] = np.nan

    PVcorrsub_a3b3 = data.AskPrice3.rolling(window=window).corr(data.AskVolume3) - data.BidPrice3.rolling(window=window).corr(data.BidVolume3)

    data[name] = PVcorrsub_a3b3
    data['time'] = data.datetime
    return data[['time', 'date', name]]

# fac27
def PVcorrsub_a4b4(data, window=20, name="PVcorrsub_a4b4"):
    data = data.copy()
    data[name] = np.nan

    PVcorrsub_a4b4 = data.AskPrice4.rolling(window=window).corr(data.AskVolume4) - data.BidPrice4.rolling(window=window).corr(data.BidVolume4)

    data[name] = PVcorrsub_a4b4
    data['time'] = data.datetime
    return data[['time', 'date', name]]

# fac28
def PVcorrsub_a5b5(data, window=20, name="PVcorrsub_a5b5"):
    data = data.copy()
    data[name] = np.nan

    PVcorrsub_a5b5 = data.AskPrice5.rolling(window=window).corr(data.AskVolume5) - data.BidPrice5.rolling(window=window).corr(data.BidVolume5)

    data[name] = PVcorrsub_a5b5
    data['time'] = data.datetime
    return data[['time', 'date', name]]

# fac29
def ask_amount_sub20(data, name='ask_amount_sub20'):
    data = data.copy()
    data[name] = np.nan

    money_flow_a = data.AskVolume1 * data.AskPrice1 \
    + data.AskVolume2 * data.AskPrice2 \
    + data.AskVolume3 * data.AskPrice3 \
    + data.AskVolume4 * data.AskPrice4 \
    + data.AskVolume5 * data.AskPrice5

    data[name] = money_flow_a.diff()
    data['time'] = data.datetime
    return data[['time', 'date', name]]

# fac30
def bid_amount_sub20(data, name='bid_amount_sub20'):
    data = data.copy()
    data[name] = np.nan

    money_flow_b = data.BidVolume1 * data.BidPrice1 \
    + data.BidVolume2 * data.BidPrice2 \
    + data.BidVolume3 * data.BidPrice3 \
    + data.BidVolume4 * data.BidPrice4 \
    + data.BidVolume5 * data.BidPrice5

    data[name] = money_flow_b.diff()
    data['time'] = data.datetime
    return data[['time', 'date', name]]

# fac 31 
def ADTMMA(data, window=(23, 8), name='ADTMMA'):
    data = data.copy()
    data[name] = np.nan
    
    # 计算前一笔价格（使用shift(1)而不是原代码的shift(20)）
    data['prev_price'] = data['mid_price'].shift(20)
    
    # 新的DTM/DBM计算方式（基于Tick数据）
    mask_up = data['mid_price'] > data['prev_price']
    mask_down = data['mid_price'] < data['prev_price']
    
    # 计算DTM和DBM（使用成交量加权）
    DTM = np.where(mask_up, 
                  data['volume'] * (data['mid_price'] - data['prev_price']),
                  0)
    DBM = np.where(mask_down,
                  data['volume'] * (data['prev_price'] - data['mid_price']),
                  0)
    
    data['DTM'] = DTM
    data['DBM'] = DBM
    
    STM = data['DTM'].rolling(window=20).sum()
    SBM = data['DBM'].rolling(window=20).sum()
    
    ADTM = (STM - SBM) / (STM + SBM + 1e-9)

    data[name] = ADTM.rolling(window=window[1]).sum()
    data['time'] = data.datetime
    return data[['time', 'date', name]]

# fac 32 
def RSJ(data, window=120, name='RSJ'):
    data = data.copy()
    data[name] = np.nan

    data['M'] = data.mid_price
    RVar_down = (data.M.diff().apply(lambda x: 0 if x > 0 else x)**2).rolling(window=window).sum()    
    RVar_up = (data.M.diff().apply(lambda x: 0 if x < 0 else x)**2).rolling(window=window).sum()
    RVar = (data.M.diff()**2).rolling(window=window).sum()
    RSJ = (RVar_up - RVar_down) / (RVar + 0.001)

    data[name] = RSJ
    data['time'] = data.datetime
    return data[['time', 'date', name]]

# fac 33
def VOI(data, name="VOI"):
    data = data.copy()
    data[name] = np.nan

    def wavg(df, n=6):
        # 计算权重
        weights = np.array([1 - (i - 1) / (n - 1) for i in range(1, n)])
        weights_sum = weights.sum()

        # 计算加权平均
        weighted_avg = (df.iloc[:, :n-1].values * weights).sum(axis=1) / weights_sum
        return weighted_avg
    
    # 计算买入和卖出量的加权平均
    VWA = wavg(data[['AskVolume1', 'AskVolume2', 'AskVolume3', 'AskVolume4', 'AskVolume5']])
    VWB = wavg(data[['BidVolume1', 'BidVolume2', 'BidVolume3', 'BidVolume4', 'BidVolume5']])
    # 将结果转换为 Series
    VWA_series = pd.Series(VWA, index=data.index)
    VWB_series = pd.Series(VWB, index=data.index)
    dVWA = np.where(data['AskPrice1'].diff() < 0, 0,
                    np.where(data['AskPrice1'].diff() == 0, VWA_series.diff(), VWA_series))

    dVWB = np.where(data['BidPrice1'].diff() > 0, 0,
                    np.where(data['BidPrice1'].diff() == 0, VWB_series.diff(), VWB_series))

    # 将结果赋值回 DataFrame
    data[name] = dVWB - dVWA
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

# fac 34
def DBCD(data, n1=5, n2=16, n3=17, name='DBCD'):
    data = data.copy()
    
    data['SMA'] = data['mid_price'].rolling(window=n1).mean()
    data['BIAS'] = (data['mid_price'] - data['SMA']) / data['SMA']
    
    # 计算 DI F
    data['DIF'] = data['BIAS'] - data['BIAS'].shift(n2)

    data[name] = data['DIF'].rolling(window=n3).mean()
    data['time'] = data.datetime
    return data[['time', 'date', name]]

# fac35
def OIR(data, name='OIR'):
    data = data.copy()
    data[name] = np.nan
    
    def wavg(df, n=6):
        # 计算权重
        weights = np.array([1 - (i - 1) / (n - 1) for i in range(1, n)])
        weights_sum = weights.sum()

        # 计算加权平均
        weighted_avg = (df.iloc[:, :n-1].values * weights).sum(axis=1) / weights_sum
        return weighted_avg
    # 计算买入和卖出量的加权平均
    VWA = wavg(data[['AskVolume1', 'AskVolume2', 'AskVolume3', 'AskVolume4', 'AskVolume5']])
    VWB = wavg(data[['BidVolume1', 'BidVolume2', 'BidVolume3', 'BidVolume4', 'BidVolume5']])

    # 将结果赋值回 DataFrame
    data[name] = (VWB - VWA) / (VWB + VWA)
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

# fac36
def SOIR(data, name='SOIR'):
    data = data.copy()
    data[name] = np.nan
    
    def wavg(df, n=6):
        # 计算权重
        weights = np.array([1 - (i - 1) / (n - 1) for i in range(1, n)])
        weights_sum = weights.sum()

        # 计算加权平均
        weighted_avg = (df * weights).sum(axis=1) / weights_sum
        return weighted_avg
    # 计算买入和卖出量的加权平均
    dfa = data[['AskVolume1', 'AskVolume2', 'AskVolume3', 'AskVolume4', 'AskVolume5']]
    dfb = data[['BidVolume1', 'BidVolume2', 'BidVolume3', 'BidVolume4', 'BidVolume5']]

    SOIR = wavg((dfa.values-dfb.values)/(dfa.values+dfb.values))    

    # 将结果赋值回 DataFrame
    data[name] = SOIR
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

# fac37
def MOFI(data, name="MOFI"):
    data = data.copy()
    data[name] = np.nan
    MOFI = pd.Series(0, index=data.index)
    for i in range(1, 6):

        dVWA = np.where(data[f'AskPrice{i}'].diff() > 0, -data[f'AskVolume{i}'].shift(),
                        np.where(data[f'AskPrice{i}'].diff() == 0, data[f'AskVolume{i}'].diff(), data[f'AskVolume{i}']))

        dVWB = np.where(data[f'BidPrice{i}'].diff() < 0, -data[f'BidVolume{i}'].shift(),
                        np.where(data[f'BidPrice{i}'].diff() == 0, data[f'BidVolume{i}'].diff(), data[f'BidVolume{i}']))
        MOFI += (dVWA-dVWB) * i 
    MOFI/=(1+2+3+4+5)

    # 将结果赋值回 DataFrame
    data[name] = MOFI
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

# fac38
def PIR(data, name='PIR'):
    data = data.copy()
    data[name] = np.nan
    pwa = pd.Series(0, index=data.index)
    pwb = pd.Series(0, index=data.index)

    divd = 0
    for i in range(1,6):
        w = 1-(i-1)/5
        pwb += w*data[f'BidPrice{i}']
        pwa += w*data[f'AskPrice{i}']
        divd+=w
    pwa/=divd
    pwb/=divd
    PIR = (pwb - pwa) / (pwb + pwa)
    # 将结果赋值回 DataFrame
    data[name] = PIR
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

 # fac39
def MAX(data, window=120, name='MAX'):
    data = data.copy()
    data[name] = np.nan

    MAX = data.groupby('date')['mid_price'].transform(lambda x: x.diff(window).rolling(window=window).max()).values
    # 将结果赋值回 DataFrame
    data[name] = MAX
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

# fac40
def MLQSweight(data, name='MLQSweight'):
    data = data.copy()
    data[name] = np.nan

    MLQSweight = pd.Series(0, index=data.index)
    divd = 0
    for i in range(1,6):
        w = i/5
        divd += w
        LS = w * (np.log(data[f'AskPrice{i}'])-np.log(data[f'BidPrice{i}'])) / (np.log(data[f'AskVolume{i}']+1)+np.log(data[f'BidVolume{i}']+1)+1)
        MLQSweight += LS
    MLQSweight /= divd
    # 将结果赋值回 DataFrame
    data[name] = MLQSweight
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

# # fac41
# def LCVOL_B(data, name='LCVOL_B'):
#     data = data.copy()
#     data[name] = np.nan

#     LCVOL_B = pd.Series(0, index=data.index)
#     for i in range(1,6):
#         I = (data[f'BidPrice{i}'] > data['PreSettlePrice']).astype(int)
#         VS = data[f'BidVolume{i}'] * I
#         LCVOL_B += VS

#     # 将结果赋值回 DataFrame
#     data[name] = LCVOL_B
#     data['time'] = data.datetime
#     return data[['time', 'date', name]]

# # fac42
# def LCVOL_A(data, name='LCVOL_A'):
#     data = data.copy()
#     data[name] = np.nan

#     LCVOL_A = pd.Series(0, index=data.index)
#     for i in range(1,6):
#         I = (data[f'AskPrice{i}'] < data['PreSettlePrice']).astype(int)
#         VS = data[f'AskVolume{i}'] * I
#         LCVOL_A += VS

#     # 将结果赋值回 DataFrame
#     data[name] = LCVOL_A
#     data['time'] = data.datetime
    
#     return data[['time', 'date', name]]

# fac43
# def RTNOVERNIGHT(data, name='RTNOVERNIGHT'):
#     data = data.copy()
#     data[name] = np.nan

#     data['M'] = data.mid_price
#     RTNOVERNIGHT = data.M / data.PreSettlePrice - 1

#     # 将结果赋值回 DataFrame
#     data[name] = RTNOVERNIGHT
#     data['time'] = data.datetime
    
#     return data[['time', 'date', name]]

# fac44
def MCIA(data, name='MCIA'):
    data = data.copy()
    data[name] = np.nan

    MCIA = pd.Series(0, index=data.index)
    M = data.mid_price
    DolVolA = data.AskPrice1 * data.AskVolume1 + \
            data.AskPrice2 * data.AskVolume2 + \
            data.AskPrice3 * data.AskVolume3 + \
            data.AskPrice4 * data.AskVolume4 + \
            data.AskPrice5 * data.AskVolume5 
    QA = data.AskVolume1 + data.AskVolume2 + data.AskVolume3 + data.AskVolume4 + data.AskVolume5
    MCIA = ((DolVolA/QA)-M) / M 
    # 将结果赋值回 DataFrame
    data[name] = MCIA.clip(lower=0, upper=0.01)
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

# fac45
def MCIB(data, name='MCIB'):
    data = data.copy()
    data[name] = np.nan

    MCIB = pd.Series(0, index=data.index)
    M = data.mid_price
    DolVolB = data.BidPrice1 * data.BidVolume1 + \
            data.BidPrice2 * data.BidVolume2 + \
            data.BidPrice3 * data.BidVolume3 + \
            data.BidPrice4 * data.BidVolume4 + \
            data.BidPrice5 * data.BidVolume5 
    QB = data.BidVolume1 + data.BidVolume2 + data.BidVolume3 + data.BidVolume4 + data.BidVolume5
    MCIB = ((DolVolB/QB)-M) / M 
    # 将结果赋值回 DataFrame
    data[name] = MCIB.clip(lower=-0.01, upper=0)
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

# fac46
def CORR_PVOL_RET(data, name='CORR_PVOL_RET'):
    data = data.copy()
    data[name] = np.nan

    RET = (data['AskPrice1']/2 + data['BidPrice1']/2).diff()
    CORR_PVOL_RET = (data['volume'] - data['volume'].rolling(window=120).mean())\
        /(RET.rolling(window=120).std() * data['volume'].rolling(window=120).std())

    # 将结果赋值回 DataFrame
    data[name] = CORR_PVOL_RET
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

# fac47
def RTN_JUMP(data, name='RTN_JUMP'):
    data = data.copy()
    data[name] = np.nan

    RET = data['mid_price'].pct_change()
    RTN_JUMP = ((RET-np.log(RET+1))**2 - np.log(RET+1)**2) * 1e6

    # 将结果赋值回 DataFrame
    data[name] = RTN_JUMP.clip(lower=-1, upper=1)
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

# fac48
def NR(data, window=120, name='NR'):
    data = data.copy()
    data[name] = np.nan

    RETCO = data.mid_price / data.mid_price.shift(20) - 1
    RETOC = data.mid_price / data.mid_price.shift(240) - 1
    RETCO[RETCO<0] = 0
    RETOC[RETOC>0] = 0
    data['RETCOOC'] = RETCO * RETOC
    NR = data.groupby('date')['RETCOOC'].transform(lambda x: x.rolling(window=window).sum() / (1 + np.arange(len(x)))) * 1e6

    # 将结果赋值回 DataFrame
    data[name] = NR
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

# fac49
def STREN(data, name='STREN'):
    data = data.copy()
    data[name] = np.nan

    vol_buy = data['volume'].rolling(window=10, min_periods=1).sum()
    vol_buy.loc[data['mid_price'] < data['AskPrice1'].shift()] = 0

    vol_sell = data['volume'].rolling(window=10, min_periods=1).sum()
    vol_sell.loc[data['mid_price'] > data['BidPrice1'].shift()] = 0

    # 将结果赋值回 DataFrame
    data[name] = (vol_buy - vol_sell).clip(-1,1)
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

# fac50
def SPREAD(data, name='SPREAD'):
    data = data.copy()
    data[name] = np.nan

    SPREAD = -(data.AskPrice1 - data.BidPrice1) / (data.AskPrice1 + data.BidPrice1) * 2 

    # 将结果赋值回 DataFrame
    data[name] = SPREAD.rolling(window=20).mean()
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

# fac51
def ASKDEPTH(data, name='ASKDEPTH'):
    data = data.copy()
    data[name] = np.nan

    ASKDEPTH = pd.Series(0,index=data.index)
    divd = 0
    for i in range(1,6):
        w = (7-i)
        divd += w
        ASKDEPTH += w * (data[f'AskVolume{i}'] * (data.AskPrice1 + data.BidPrice1)/2) / (data[f'AskPrice{i}'] - data['AskPrice1'].shift(1)/2 - data['BidPrice1'].shift(1)/2 +100).abs()

    ASKDEPTH /= divd
    # 将结果赋值回 DataFrame
    data[name] = ASKDEPTH
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

# fac52
def BBI(data, name='BBI'):
    data = data.copy()
    data[name] = np.nan

    data[name] = data['mid_price'].rolling(window=4).mean() - data['mid_price'].rolling(window=8).mean()
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

# fac53
def resiliency(data, name='resiliency'):
    data = data.copy()
    data[name] = np.nan

    data[name] = (data['HighPrice'] - data['LowPrice']) / (data.TotalTradeVolume / data.OpenInterest)
    data[name] = data[name].clip(lower=-1, upper=1000)
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

# fac54
def BIDDEPTH(data, name='BIDDEPTH'):
    data = data.copy()
    data[name] = np.nan

    BIDDEPTH = pd.Series(0,index=data.index)
    divd = 0
    for i in range(1,6):
        w = (7-i)
        divd += w
        BIDDEPTH += w * (data[f'BidVolume{i}'] * (data.AskPrice1 + data.BidPrice1)/2) / (data[f'BidPrice{i}'] - data['AskPrice1'].shift(1)/2 - data['BidPrice1'].shift(1)/2 +100).abs()

    BIDDEPTH /= divd
    # 将结果赋值回 DataFrame
    data[name] = BIDDEPTH
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

# fac55
def AVGDEPTH(data, name='AVGDEPTH'):
    data = data.copy()
    data[name] = np.nan

    AVGDEPTH = data.AskVolume1/2 + data.BidVolume1/2

    # 将结果赋值回 DataFrame
    data[name] = AVGDEPTH
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

# fac56
def VOL_FLU(data, name='VOL_FLU'):
    data = data.copy()
    data[name] = np.nan

    VOL_FLU = data['volume'].rolling(window=240, min_periods=1).std()
    VOL_FLU = VOL_FLU.rolling(window=20).std() / VOL_FLU.rolling(window=20).mean()

    # 将结果赋值回 DataFrame
    data[name] = VOL_FLU
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

# fac57
def QUA(data, name='QUA'):
    data = data.copy()
    data[name] = np.nan

    sig_trade_value = data['volume'].rolling(window=10, min_periods=1).sum()
    QUA = (sig_trade_value.rolling(window=120,min_periods=100).quantile(0.1) - sig_trade_value.rolling(window=120,min_periods=100).min()) / \
        (sig_trade_value.rolling(window=120,min_periods=100).max() - sig_trade_value.rolling(window=120,min_periods=100).min())

    # 将结果赋值回 DataFrame
    data[name] = QUA
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

# fac58
def shortQUA(data, name='shortQUA'):
    data = data.copy()
    data[name] = np.nan

    sig_trade_value = data['volume'].rolling(window=10, min_periods=1).sum()
    QUA = (sig_trade_value.rolling(window=20).quantile(0.1) - sig_trade_value.rolling(window=20).min()) / \
        (sig_trade_value.rolling(window=20).max() - sig_trade_value.rolling(window=20).min())

    # 将结果赋值回 DataFrame
    data[name] = QUA
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

# fac59
def midQUA(data, name='midQUA'):
    data = data.copy()
    data[name] = np.nan

    sig_trade_value = data['volume'].rolling(window=10, min_periods=1).sum()
    QUA = (sig_trade_value.rolling(window=60).quantile(0.1) - sig_trade_value.rolling(window=60).min()) / \
        (sig_trade_value.rolling(window=60).max() - sig_trade_value.rolling(window=60).min())

    # 将结果赋值回 DataFrame
    data[name] = QUA
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

# fac60
def PRICE_PRESSURE(data, name='PRICE_PRESSURE'):
    data = data.copy()
    data[name] = np.nan

    data['last_price'] = data['mid_price'].shift(1)
    PRICE_PRESSURE = (data.AskPrice1 - data.last_price) / (data.AskPrice1 - data.BidPrice1)

    # 将结果赋值回 DataFrame
    data[name] = PRICE_PRESSURE - 0.5
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

# fac61
def ORDER_SLOPE_A(data, name='ORDER_SLOPE_A'):
    data = data.copy()
    data[name] = np.nan

    ORDER_SLOPE_A = (data.AskPrice5 - data.AskPrice1) / (data.AskVolume1 + data.AskVolume2 + data.AskVolume3 + data.AskVolume4 + data.AskVolume5)

    # 将结果赋值回 DataFrame
    data[name] = ORDER_SLOPE_A
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

# fac62
def ORDER_SLOPE_B(data, name='ORDER_SLOPE_B'):
    data = data.copy()
    data[name] = np.nan

    ORDER_SLOPE_B = (data.BidPrice5 - data.BidPrice1) / (data.BidVolume1 + data.BidVolume2 + data.BidVolume3 + data.BidVolume4 + data.BidVolume5)

    # 将结果赋值回 DataFrame
    data[name] = ORDER_SLOPE_B
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

# fac63
def PRICE_STD_B(data, name='PRICE_STD_B'):
    data = data.copy()
    data[name] = np.nan

    PRICE_STD_B = (data[['BidPrice1','BidPrice2', 'BidPrice3', 'BidPrice4', 'BidPrice5']].std(axis=1))

    # 将结果赋值回 DataFrame
    data[name] = PRICE_STD_B
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

# fac64
def PRICE_STD_A(data, name='PRICE_STD_A'):
    data = data.copy()
    data[name] = np.nan

    PRICE_STD_A = (data[['AskPrice1','AskPrice2', 'AskPrice3', 'AskPrice4', 'AskPrice5']].std(axis=1))

    # 将结果赋值回 DataFrame
    data[name] = PRICE_STD_A
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

# fac65
def OI_CHG(data, name='OI_CHG'):
    data = data.copy()
    data[name] = np.nan

    OI_CHG = data['OpenInterest'].pct_change()

    # 将结果赋值回 DataFrame
    data[name] = OI_CHG.clip(-0.01, 0.01)
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

# fac66
def OI_GP_CH(data, name='OI_GP_CH'):
    data = data.copy()
    data[name] = np.nan

    data['M'] = data.mid_price
    OI_GP_CH = data['OpenInterest'].pct_change() * data['M'].pct_change()

    # 将结果赋值回 DataFrame
    data[name] = (OI_GP_CH * 1e6).clip(-1,1)
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

# fac67
def OI_V_DIV(data, name='OI_V_DIV'):
    data = data.copy()
    data[name] = np.nan

    OI_V_DIV = data['TotalTradeVolume'].pct_change() / data['OpenInterest'] * 1e6
    OI_V_DIV = OI_V_DIV.replace([-np.inf, np.inf], np.nan)
    # 将结果赋值回 DataFrame
    data[name] = OI_V_DIV.clip(-1,1)
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

# fac68
# def TRADE_OI_FLOW(data, name='TRADE_OI_FLOW'):
#     data = data.copy()
#     data[name] = np.nan

#     BUY_OI_FLOW = data.TotalBuyVolume / (data.TotalBuyVolume + data.TotalSellVolume) * data['OpenInterest'].diff().values
#     SELL_OI_FLOW = data.TotalSellVolume / (data.TotalBuyVolume + data.TotalSellVolume) * data['OpenInterest'].diff().values

#     # 将结果赋值回 DataFrame
#     data[name] = BUY_OI_FLOW - SELL_OI_FLOW
#     data['time'] = data.datetime
    
#     return data[['time', 'date', name]]

# fac69
def Imbalance_OI(data, name='Imbalance_OI'):
    data = data.copy()
    data[name] = np.nan

    Imbalance_OI = (data.BidVolume1 - data.AskVolume1) / (data.BidVolume1 + data.AskVolume1) * data['OpenInterest'].diff().values

    # 将结果赋值回 DataFrame
    data[name] = Imbalance_OI
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

# fac70
# def Delta_Elasticity(data, name='Delta_Elasticity'):
#     data = data.copy()
#     data[name] = np.nan

#     Delta_Elasticity = (data.CurrDelta - data.PreDelta) / (data.mid_price - data.PreClosePrice) * data.PreClosePrice

#     # 将结果赋值回 DataFrame
#     data[name] = Delta_Elasticity
#     data['time'] = data.datetime
    
#     return data[['time', 'date', name]]

# fac71
def Depth_OI_Pressure(data, name='Depth_OI_Pressure'):
    data = data.copy()
    data[name] = np.nan

    SUMAV = data.AskVolume1 + data.AskVolume2 + data.AskVolume3 + data.AskVolume4 + data.AskVolume5
    SUMBV = data.BidVolume1 + data.BidVolume2 + data.BidVolume3 + data.BidVolume4 + data.BidVolume5

    DOP = (SUMBV - SUMAV) / (SUMAV + SUMBV) * np.log(data.OpenInterest)

    # 将结果赋值回 DataFrame
    data[name] = DOP
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

# # fac72
# def Delta_Volume_Sync(data, name='Delta_Volume_Sync'):
#     data = data.copy()
#     data[name] = np.nan

#     Delta_Volume_Sync = np.sign(data.CurrDelta - data.PreDelta) * data.TotalTradeVolume

#     # 将结果赋值回 DataFrame
#     data[name] = Delta_Volume_Sync
#     data['time'] = data.datetime
    
#     return data[['time', 'date', name]]

# fac73
# def Tick_Strength(data, name='Tick_Strength'):
#     data = data.copy()
#     data[name] = np.nan

#     Tick_Strength  = (data.TotalBuyVolume - data.TotalSellVolume) / (data.TotalBuyVolume + data.TotalSellVolume) * np.sqrt(data.NumTrades)

#     # 将结果赋值回 DataFrame
#     data[name] = Tick_Strength
#     data['time'] = data.datetime
    
#     return data[['time', 'date', name]]

# fac74
def VWAP_Deviation(data, name='VWAP_Deviation'):
    data = data.copy()
    data[name] = np.nan

    VWAP_Deviation = data.mid_price / (data.TotalTradeValue / data.TotalTradeVolume)

    # 将结果赋值回 DataFrame
    data[name] = VWAP_Deviation
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

# fac75
# def Toxic_Ratio(data, name='Toxic_Ratio'):
#     data = data.copy()
#     data[name] = np.nan

#     Toxic_Ratio = (data.TotalBuyVolume * data.AskPrice1 - data.TotalSellVolume * data.BidPrice1) / data.TotalTradeValue

#     # 将结果赋值回 DataFrame
#     data[name] = Toxic_Ratio
#     data['time'] = data.datetime
    
#     return data[['time', 'date', name]]

# fac76
def OI_Price_Accel(data, name='OI_Price_Accel'):
    data = data.copy()
    data[name] = np.nan

    OI_Price_Accel = (data['OpenInterest'].diff() / data['OpenInterest']) * (data['mid_price'].diff() / data['mid_price']) * 1e6

    # 将结果赋值回 DataFrame
    data[name] = OI_Price_Accel.clip(-1,1)
    data['time'] = data.datetime    
    
    return data[['time', 'date', name]]

# fac77
def OFI1(data, name='OFI1'):
    data = data.copy()
    data[name] = np.nan
    i = 1
    dVWA = np.where(data[f'AskPrice{i}'].diff() > 0, -data[f'AskVolume{i}'].shift(),
                    np.where(data[f'AskPrice{i}'].diff() == 0, data[f'AskVolume{i}'].diff(), data[f'AskVolume{i}']))

    dVWB = np.where(data[f'BidPrice{i}'].diff() < 0, -data[f'BidVolume{i}'].shift(),
                    np.where(data[f'BidPrice{i}'].diff() == 0, data[f'BidVolume{i}'].diff(), data[f'BidVolume{i}']))
    OFI1 = dVWA-dVWB 
    
    # 将结果赋值回 DataFrame
    data[name] = OFI1
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

# fac78
def OFI2(data, name='OFI2'):
    data = data.copy()
    data[name] = np.nan
    i = 2
    dVWA = np.where(data[f'AskPrice{i}'].diff() > 0, -data[f'AskVolume{i}'].shift(),
                    np.where(data[f'AskPrice{i}'].diff() == 0, data[f'AskVolume{i}'].diff(), data[f'AskVolume{i}']))

    dVWB = np.where(data[f'BidPrice{i}'].diff() < 0, -data[f'BidVolume{i}'].shift(),
                    np.where(data[f'BidPrice{i}'].diff() == 0, data[f'BidVolume{i}'].diff(), data[f'BidVolume{i}']))
    OFI2 = dVWA-dVWB 
    
    # 将结果赋值回 DataFrame
    data[name] = OFI2
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

# fac79
def OFI3(data, name='OFI3'):
    data = data.copy()
    data[name] = np.nan
    i = 3
    dVWA = np.where(data[f'AskPrice{i}'].diff() > 0, -data[f'AskVolume{i}'].shift(),
                    np.where(data[f'AskPrice{i}'].diff() == 0, data[f'AskVolume{i}'].diff(), data[f'AskVolume{i}']))

    dVWB = np.where(data[f'BidPrice{i}'].diff() < 0, -data[f'BidVolume{i}'].shift(),
                    np.where(data[f'BidPrice{i}'].diff() == 0, data[f'BidVolume{i}'].diff(), data[f'BidVolume{i}']))
    OFI3 = dVWA-dVWB 
    
    # 将结果赋值回 DataFrame
    data[name] = OFI3
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

# fac80
def OFI4(data, name='OFI4'):
    data = data.copy()
    data[name] = np.nan
    i = 4
    dVWA = np.where(data[f'AskPrice{i}'].diff() > 0, -data[f'AskVolume{i}'].shift(),
                    np.where(data[f'AskPrice{i}'].diff() == 0, data[f'AskVolume{i}'].diff(), data[f'AskVolume{i}']))

    dVWB = np.where(data[f'BidPrice{i}'].diff() < 0, -data[f'BidVolume{i}'].shift(),
                    np.where(data[f'BidPrice{i}'].diff() == 0, data[f'BidVolume{i}'].diff(), data[f'BidVolume{i}']))
    OFI4 = dVWA-dVWB 
    
    # 将结果赋值回 DataFrame
    data[name] = OFI4
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

# fac81
def OFI5(data, name='OFI5'):
    data = data.copy()
    data[name] = np.nan
    i = 5
    dVWA = np.where(data[f'AskPrice{i}'].diff() > 0, -data[f'AskVolume{i}'].shift(),
                    np.where(data[f'AskPrice{i}'].diff() == 0, data[f'AskVolume{i}'].diff(), data[f'AskVolume{i}']))

    dVWB = np.where(data[f'BidPrice{i}'].diff() < 0, -data[f'BidVolume{i}'].shift(),
                    np.where(data[f'BidPrice{i}'].diff() == 0, data[f'BidVolume{i}'].diff(), data[f'BidVolume{i}']))
    OFI1 = dVWA-dVWB 
    
    # 将结果赋值回 DataFrame
    data[name] = OFI1
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

# fac82
def ILLIQ(data, name='ILLIQ'):
    data = data.copy()
    data[name] = np.nan

    data['ILLIQ'] = data['mid_price'].diff() / data['turnover'] * 1e8
    ILLIQ = data['ILLIQ'].rolling(window=20, min_periods=10).mean()
    # 将结果赋值回 DataFrame
    data[name] = ILLIQ
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

# fac83
def ADTM(data, name='ADTM'):
    data = data.copy()
    data[name] = np.nan

    # 计算前一笔价格（使用shift(1)而不是原代码的shift(20)）
    data['prev_price'] = data['mid_price'].shift(20)
    
    # 新的DTM/DBM计算方式（基于Tick数据）
    mask_up = data['mid_price'] > data['prev_price']
    mask_down = data['mid_price'] < data['prev_price']
    
    # 计算DTM和DBM（使用成交量加权）
    DTM = np.where(mask_up, 
                  data['volume'] * (data['mid_price'] - data['prev_price']),
                  0)
    DBM = np.where(mask_down,
                  data['volume'] * (data['prev_price'] - data['mid_price']),
                  0)
    
    data['DTM'] = DTM
    data['DBM'] = DBM
    STM = data['DTM'].rolling(window=20).sum()
    SBM = data['DBM'].rolling(window=20).sum()
    
    ADTM = (STM - SBM) / np.maximum(STM, SBM)

    # 将结果赋值回 DataFrame
    data[name] = ADTM
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

# fac84
def Hurst(data, name='Hurst'):
    data = data.copy()
    data[name] = np.nan

    data['lma'] = data.mid_price - data['mid_price'].rolling(window=20).mean().values
    data['Z'] = data['lma'].rolling(window=20).sum()
    data['rn'] = data['Z'].rolling(window=20).max() - data['Z'].rolling(window=20).min()
    Hurst = data['rn'] / data['mid_price'].rolling(window=20).std()

    # 将结果赋值回 DataFrame
    data[name] = Hurst
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

# fac85
def Bolling(data, name='Bolling'):
    data = data.copy()
    data[name] = np.nan

    data['M'] = data.mid_price
    Bolling = (data['M'] - data['M'].rolling(window=20).mean()) / (1+data['M'].rolling(window=20).std())

    # 将结果赋值回 DataFrame
    data[name] = Bolling
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

# fac86
def RSI(data, window=60, name='RSI'):
    data = data.copy()
    delta = data['mid_price'].diff()
    
    # 计算涨跌幅（SMA直接求和）
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # SMA计算：滚动窗口求和后除以窗口长度
    avg_gain = gain.rolling(window).mean()  # 平均涨幅
    avg_loss = loss.rolling(window).mean()  # 平均跌幅
    
    # 处理初始NaN值（前window-1个点）
    rs = avg_gain / (avg_loss + 1e-8)  # 避免除零
    data[name] = 100 - (100 / (1 + rs))
    
    # 保留时间列
    data['time'] = data['datetime']
    return data[['time', 'date', name]]

# fac87
def MACD(data, name='MACD'):
    data = data.copy()
    data[name] = np.nan

    EMA12oi = wma(data['mid_price'].diff(), window=12)
    EMA26am = wma(data['mid_price'].diff(), window=26)
    data['MACD'] = EMA12oi - EMA26am
    MACD = wma(data['MACD'], window=9)

    # 将结果赋值回 DataFrame
    data[name] = MACD
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

# fac88
def Price_Divergence(data, name='Price_Divergence'):
    data = data.copy()
    data[name] = np.nan

    # 1. 计算价格和成交量的变化
    high_diff = data['HighPrice'].diff(30)        # 20日最高价变化
    low_diff = data['LowPrice'].diff(30)         # 20日最低价变化
    volume_diff_20 = data['volume'].rolling(window=30, min_periods=1).sum()
    volume_diff_60 = data['volume'].rolling(window=60, min_periods=1).sum()

    # 2. 定义Mask函数：过滤无效值（diff < 0 或除零）
    def strict_mask(series, min_val=0):
        """将series中<min_val的值设为NaN，并处理inf"""
        masked = series.where(series >= min_val, np.nan)
        masked.replace([np.inf, -np.inf], np.nan, inplace=True)
        return masked

    # 3. 应用Mask（所有diff必须≥0）
    high_diff = (strict_mask(high_diff)>0).astype(int)            # HighPrice上升时才有效
    low_diff = (strict_mask(-low_diff)>0).astype(int)             # LowPrice下降时才有效（取负后判断）
    volume_diff_20 = strict_mask(volume_diff_20) # 成交量短期变化≥0
    volume_diff_60 = strict_mask(volume_diff_60)  # 成交量长期变化≥0

    # 4. 计算成交量比值（核心修改点）
    with np.errstate(divide='ignore', invalid='ignore'):
        volume_ratio = volume_diff_60 / (volume_diff_20 + 1e-8)  # 避免除零
        volume_ratio = strict_mask(volume_ratio)                 # 再次过滤inf/nan

    # 6. 合成因子（NaN保留）
    data[name] = (volume_ratio * (high_diff.astype(float) - low_diff.astype(float))).clip(-10,10)
    data['time'] = data['datetime']
    
    return data[['time', 'date', name]]

# fac89
def Momentum_Decay(data, name='Momentum_Decay'):
    data = data.copy()
    data[name] = np.nan

    data['num'] = range(1, len(data)+1)
    # 计算过去20个数据点的最后价格的线性回归斜率
    Slope = data['mid_price'].rolling(window=20).corr(data['num'])
    Momentum_Decay = np.sign(Slope) * (abs(Slope) - (abs(Slope).rolling(window=20).mean()))

    # 将结果赋值回 DataFrame
    data[name] = Momentum_Decay
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

# fac90
def Depth_Reversal(data, name='Depth_Reversal', window=50):
    data = data.copy()
    data[name] = np.nan

    sbv = data.BidVolume1 + data.BidVolume2 + data.BidVolume3 + data.BidVolume4 + data.BidVolume5
    sav = data.AskVolume1 + data.AskVolume2 + data.AskVolume3 + data.AskVolume4 + data.AskVolume5
    spread = data['AskPrice1'] - data['BidPrice1']
    mask = (spread > spread.rolling(window).mean())
    m = data.mid_price
    Depth_Reversal = np.where(mask, sbv/sav, np.nan) * m.pct_change(window)

    # 将结果赋值回 DataFrame
    data[name] = Depth_Reversal
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

#fac91
# def Flow_Reversal(data, name='Flow_Reversal'):
#     data = data.copy()
#     data[name] = np.nan

#     tbv = data['TotalBuyVolume']
#     tsv = data['TotalSellVolume']

#     Flow_Reversal = (tbv/tsv<0.9) & ((tbv/tsv).rolling(window=10).mean() > 1.2)

#     # 将结果赋值回 DataFrame
#     data[name] = Flow_Reversal
#     data['time'] = data.datetime
    
#     return data[['time', 'date', name]]

# fac92
# def WeightedMomentum(data, name='WeightedMomentum'):
#     data = data.copy()
#     data[name] = np.nan

#     WeightedMomentum = (data.mid_price - data.PreClosePrice) * (1 + 0.5 * data.NumTrades)

#     # 将结果赋值回 DataFrame
#     data[name] = WeightedMomentum
#     data['time'] = data.datetime
    
#     return data[['time', 'date', name]]

# fac93
def PRICE_VOL_CORR_A(data, name='PRICE_VOL_CORR_A'):
    data = data.copy()
    data[name] = np.nan

    PRICE_VOL_CORR_A = data['AskPrice1'].rolling(window=20).corr(data['AskVolume1'])

    # 将结果赋值回 DataFrame
    data[name] = PRICE_VOL_CORR_A
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

# fac94
def PRICE_VOL_CORR_B(data, name='PRICE_VOL_CORR_B'):
    data = data.copy()
    data[name] = np.nan

    PRICE_VOL_CORR_B = data['BidPrice1'].rolling(window=20).corr(data['BidVolume1'])

    # 将结果赋值回 DataFrame
    data[name] = PRICE_VOL_CORR_B
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

# fac95
def KDJ(data, name='KDJ'):
    data = data.copy()
    data[name] = np.nan

    data['M'] = data.mid_price
    # 计算 L_n 和 H_n
    data['L_n'] = data['LowPrice'].rolling(window=20).min()
    data['H_n'] = data['HighPrice'].rolling(window=20).max()
    
    # 计算 RSV
    data['RSV'] = (data['M'] - data['L_n']) / (data['H_n'] - data['L_n']) * 100
    
    data['K'] = wma(data['RSV'], window=20)
    data['D'] = wma(data['K'], window=20)

    KDJ = 3 * data.K - 2 * data.D

    # 将结果赋值回 DataFrame
    data[name] = KDJ
    data['time'] = data.datetime
    
    return data[['time', 'date', name]]

# fac96
def MFI(data, period=14, name='MFI'):
    data = data.copy()
    
    # 计算典型价格
    data['M'] = data.mid_price
    data['TP'] = (data['HighPrice'] + data['LowPrice'] + data['M']) / 3

    # 计算资金流量
    data['MF'] = data['TP'] * data['volume']

    # 使用 np.where 向量化计算正负资金流
    data['Positive_MF'] = np.where(data['TP'] > data['TP'].shift(1), data['MF'], 0)
    data['Negative_MF'] = np.where(data['TP'] <= data['TP'].shift(1), data['MF'], 0)

    # 计算滚动和
    data['Positive_MF'] = data['Positive_MF'].rolling(window=period).sum()
    data['Negative_MF'] = data['Negative_MF'].rolling(window=period).sum()

    # 计算 MFI
    data[name] = 100 - (100 / (1 + data['Positive_MF'] / data['Negative_MF']))

    data['time'] = data['datetime']
    
    return data[['time', 'date', name]]

# fac97
def fibonacci_retracement(data, name='fibonacci_retracement'):
    data = data.copy()
    
    # 计算典型价格
    data['M'] = data.mid_price

    fibonacci_retracement = data['M'] / ((data['M'].rolling(window=20).max() - (data['M'].rolling(window=20).max() - data['M'].rolling(window=20).min())*0.5))

    # 计算 fibonacci_retracement
    data[name] = fibonacci_retracement-1

    data['time'] = data['datetime']
    
    return data[['time', 'date', name]]

# fac98
def LVC(data, name='LVC'):
    data = data.copy()
    
    LVC = (data.BidVolume1 + data.BidVolume2 + data.BidVolume3 + data.BidVolume4 + data.BidVolume5) * data['mid_price'].rolling(window=20).std() / data['mid_price'].rolling(window=100).std() 

    # 计算 LVC
    data[name] = LVC

    data['time'] = data['datetime']
    
    return data[['time', 'date', name]]

# fac99
def tsi(data, long_period=25, short_period=13, name='tsi'):
    data = data.copy()
    
    data['M'] = data.mid_price

    price_diff = data['M'].diff()
    
    # 计算平滑的价格变化
    smoothed_price_diff = wma(price_diff, window=short_period)
    
    # 计算绝对价格变化的平滑
    smoothed_abs_price_diff = wma(abs(price_diff), window=short_period)
    
    # 计算 TSI
    tsi = 100 * wma(smoothed_price_diff, window=long_period) / wma(smoothed_abs_price_diff, window=long_period)
    # 计算 tsi
    data[name] = tsi

    data['time'] = data['datetime']
    
    return data[['time', 'date', name]]

# fac100
def CMF(data, period=20, name='CMF'):
    data = data.copy()
    
    # 1. 计算典型价格
    typical_price = (data['HighPrice'] + data['LowPrice'] + data['mid_price']) / 3
    
    # 2. 计算成交量变化，并mask掉减少的部分（diff < 0）
    volume_diff = data['volume']
    volume_diff = volume_diff.where(volume_diff >= 0, np.nan)  # 成交量减少时设为NaN
    
    # 3. 计算动态资金流（仅使用成交量增加的部分）
    money_flow = typical_price * volume_diff
    
    # 4. 计算CMF（分母为有效成交量的和）
    numerator = money_flow.rolling(window=period, min_periods=15).mean()
    denominator = volume_diff.rolling(window=period, min_periods=15).mean()
    
    cmf = numerator / denominator.where(denominator>0, np.nan)  
        
    # 6. 存储结果
    data[name] = cmf.pct_change(60*2) * 10000
    data['time'] = data['datetime']
    
    return data[['time', 'date', name]]

# fac100
def RUSHSTOPASK(data, name='RUSHSTOPASK'):
    data = data.copy()

    fac = data.mid_price.pct_change(60*2) * data[['AskPrice2', 'AskPrice3', 'AskPrice4', 'AskPrice5']].max(axis=1).diff().rolling(window=4).mean()

    data[name] = fac.clip(-0.01, 0.01)
    data['time'] = data['datetime']
    
    return data[['time', 'date', name]]

# fac100
def RUSHSTOPBID(data, name='RUSHSTOPBID'):
    data = data.copy()

    fac = data.mid_price.pct_change(60*2) * data[['BidPrice2', 'BidPrice3', 'BidPrice4', 'BidPrice5']].max(axis=1).diff().rolling(window=4).mean()

    data[name] = fac.clip(-0.01, 0.01)
    data['time'] = data['datetime']
    
    return data[['time', 'date', name]]
