import numpy as np
import pandas as pd

# tick data min factors
#-------------------------------------------------------------------------------------------------------------------------------------
# fac 125
def buy_trend(data, data1m):
    def calc_buy_tend(group):
        if group.shape[0]<5:
            return np.nan
        else:
            buy_amount = group['BidVolume1'] + group['BidVolume2'] + group['BidVolume3'] + group['BidVolume4'] + group['BidVolume5']
            sell_amount =  group['AskVolume1'] + group['AskVolume2'] + group['AskVolume3'] + group['AskVolume4'] + group['AskVolume5']
            return (buy_amount>sell_amount).mean()
    fac = data.groupby(['ts']).apply(calc_buy_tend)
    fac = fac.reindex(index = data1m.ts)
    return fac

# fac 126
def buy_trend1(data, data1m):
    def calc_fac(group):
        if group.shape[0]<5:
            return np.nan
        else:
            buy_amount = group['BidVolume1'] 
            sell_amount =  group['AskVolume1'] 
            return (buy_amount>sell_amount).mean()
    fac = data.groupby(['ts']).apply(calc_fac)
    fac = fac.reindex(index = data1m.ts)
    return fac

# fac 127
def buy_trend_rol10(data,data1m):
    def calc_fac(group):
        if group.shape[0]<5:
            return np.nan
        else:
            buy_amount = group['BidVolume1'].rolling(window=10).sum()
            sell_amount = group['AskVolume1'].rolling(window=10).sum()

            return (buy_amount>sell_amount).mean()
    fac = data.groupby(['ts']).apply(calc_fac)
    fac = fac.reindex(index = data1m.ts)
    return fac

# fac 128
def lastprice_bias1(data,data1m,multiplier):
    def calc_fac(group):
        if group.shape[0]<5:
            return np.nan
        else:
            group['vwap'] = (group['amount'] / group['volume']) / multiplier
            buy_amount = (group['vwap']-group['AskPrice1']).abs().rolling(window=1).sum()
            sell_amount = (group['vwap']-group['BidPrice1']).abs().rolling(window=1).sum()
            return (buy_amount>sell_amount).mean()
    fac = data.groupby(['ts']).apply(calc_fac)
    fac = fac.reindex(index = data1m.ts)
    return fac

# fac 129
def tickvol10(data,data1m):
    def calc_fac(group):
        if group.shape[0]<5:
            return np.nan
        else:
            realized_vol = group.mid_price.diff().apply(lambda x:x*x).rolling(window=20).sum()
            return realized_vol.mean()
    fac = data.groupby(['ts']).apply(calc_fac)
    fac = fac.reindex(index = data1m.ts)
    fac = fac.sub(fac.rolling(window=60, min_periods=50).mean()).div(fac.rolling(window=60, min_periods=50).std())
    fac = fac.replace([-np.inf, np.inf], np.nan)
    return fac

# fac 130
def tickvol20(data,data1m):
    def calc_fac(group):
        if group.shape[0]<5:
            return np.nan
        else:
            realized_vol = group.mid_price.diff().apply(lambda x:x*x).rolling(window=10).sum()
            return realized_vol.mean()
    fac = data.groupby(['ts']).apply(calc_fac)
    fac = fac.reindex(index = data1m.ts)
    fac = fac.sub(fac.rolling(window=60, min_periods=50).mean()).div(fac.rolling(window=60, min_periods=50).std())
    fac = fac.replace([-np.inf, np.inf], np.nan)
    return fac

# fac 134
def TMB(data,data1m):
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import r2_score
    def calc_fac(group):
        if group.shape[0]<5:
            return np.nan
        else:
            M = group['mid_price']
            X = np.arange(len(M)).reshape(-1, 1)
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)
            
            valid_mask = M.notna()
            if valid_mask.sum() < 30: 
                return np.nan
                
            model = LinearRegression()
            model.fit(X_poly[valid_mask], M[valid_mask])
            
            y_pred = model.predict(X_poly)
            linear_coefficient = model.coef_[1]
            r_squared = r2_score(M[valid_mask], y_pred[valid_mask])        
            return linear_coefficient * r_squared

    fac = data.groupby(['ts']).apply(calc_fac)
    fac = fac.reindex(index = data1m.ts)
    return fac

# fac 141
def inflectionpoint(data,data1m):
    def calc_fac(group):
        if group.shape[0]<5:
            return np.nan
        else:
            m = (group.AskPrice1/2 + group.BidPrice1/2).fillna(method='ffill')
            point = pd.Series(np.sign(m.diff())).replace(0,np.nan).fillna(method='ffill').diff().abs()
            return point.sum()
    fac = data.groupby(['ts']).apply(calc_fac)
    fac = fac.reindex(index = data1m.ts)
    return fac

# fac 142
def inflectionpoint5(data,data1m):
    def calc_fac(group):
        if group.shape[0]<5:
            return np.nan
        else:
            m = (group.AskPrice1/2 + group.BidPrice1/2).fillna(method='ffill')
            point = pd.Series(np.sign(m.diff(5))).replace(0,np.nan).fillna(method='ffill').diff().abs()
            return point.sum()
    fac = data.groupby(['ts']).apply(calc_fac)
    fac = fac.reindex(index = data1m.ts)
    return fac

