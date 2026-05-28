"""
豆一 (A) 策略 - DCE
基于 LightGBM 的预测模型
"""
# 标准库
import os
import shutil
import sqlite3
import threading
import time as time_module
import warnings
from datetime import datetime, time, timedelta
from typing import Any, Dict, List, Optional

# 第三方库
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# 本地模块
from local_save.strategy_PAMY_dce.data_function import *

# 忽略警告
warnings.filterwarnings('ignore')

def generate_factor_dataframe(fac_generator, 
                            valid_index: pd.Index,
                            factor_col) -> pd.DataFrame:
    from typing import Union, Dict

    print(f'{valid_index[0]} - {valid_index[-1]}')
    factor_dict: Dict[str, np.ndarray] = {}
    
    factor_dict['FAC_ADTMMA_biddommean'] = fac_generator.FAC_ADTMMA('biddommean')
    factor_dict['FAC_ASKDEPTH_vwap'] = fac_generator.FAC_ASKDEPTH('vwap')
    factor_dict['FAC_AVGDEPTH_askdommean'] = fac_generator.FAC_AVGDEPTH('askdommean')
    factor_dict['FAC_AVGDEPTH_biddommean'] = fac_generator.FAC_AVGDEPTH('biddommean')
    factor_dict['FAC_BBI_askdommean'] = fac_generator.FAC_BBI('askdommean')
    factor_dict['FAC_BBI_biddommean'] = fac_generator.FAC_BBI('biddommean')
    factor_dict['FAC_BBI_first'] = fac_generator.FAC_BBI('first')
    factor_dict['FAC_BBI_last'] = fac_generator.FAC_BBI('last')
    factor_dict['FAC_BIDDEPTH_Volraiseap'] = fac_generator.FAC_BIDDEPTH('Volraise')
    factor_dict['FAC_BIDDEPTH_askdommean'] = fac_generator.FAC_BIDDEPTH('askdommean')
    factor_dict['FAC_Bolling_biddommean'] = fac_generator.FAC_Bolling('biddommean')
    factor_dict['FAC_Bolling_downmean'] = fac_generator.FAC_Bolling('downmean')
    factor_dict['FAC_Bolling_first'] = fac_generator.FAC_Bolling('first')
    factor_dict['FAC_Bolling_last'] = fac_generator.FAC_Bolling('last')
    factor_dict['FAC_Bolling_min'] = fac_generator.FAC_Bolling('min')
    factor_dict['FAC_CMF_biddommean'] = fac_generator.FAC_CMF('biddommean')
    factor_dict['FAC_CMF_min'] = fac_generator.FAC_CMF('min')
    factor_dict['FAC_CORR_PVOL_RET_MADmean'] = fac_generator.FAC_CORR_PVOL_RET('MADmean')
    factor_dict['FAC_CORR_PVOL_RET_askdommean'] = fac_generator.FAC_CORR_PVOL_RET('askdommean')
    factor_dict['FAC_CORR_PVOL_RET_skewness'] = fac_generator.FAC_CORR_PVOL_RET('skewness')
    factor_dict['FAC_CORR_PVOL_RET_upmean'] = fac_generator.FAC_CORR_PVOL_RET('upmean')
    factor_dict['FAC_DBCD_askdommean'] = fac_generator.FAC_DBCD('askdommean')
    factor_dict['FAC_Depth_OI_Pressure_askdommean'] = fac_generator.FAC_Depth_OI_Pressure('askdommean')
    factor_dict['FAC_Depth_Reversal_kurtosis'] = fac_generator.FAC_Depth_Reversal('kurtosis')
    factor_dict['FAC_Depth_Reversal_min'] = fac_generator.FAC_Depth_Reversal('min')
    factor_dict['FAC_ILLIQ_askdommean'] = fac_generator.FAC_ILLIQ('askdommean')
    factor_dict['FAC_ILLIQ_min'] = fac_generator.FAC_ILLIQ('min')
    factor_dict['FAC_ILLIQ_skewness'] = fac_generator.FAC_ILLIQ('skewness')
    factor_dict['FAC_Imbalance_OI_first'] = fac_generator.FAC_Imbalance_OI('first')
    factor_dict['FAC_KDJ_askdommean'] = fac_generator.FAC_KDJ('askdommean')
    factor_dict['FAC_KDJ_min'] = fac_generator.FAC_KDJ('min')
    factor_dict['FAC_MAX_askdommean'] = fac_generator.FAC_MAX('askdommean')
    factor_dict['FAC_MAX_max'] = fac_generator.FAC_MAX('max')
    factor_dict['FAC_MAX_std'] = fac_generator.FAC_MAX('std')
    factor_dict['FAC_MCIA_Volraiseap'] = fac_generator.FAC_MCIA('Volraise')
    factor_dict['FAC_MCIA_biddommean'] = fac_generator.FAC_MCIA('biddommean')
    factor_dict['FAC_MCIA_min'] = fac_generator.FAC_MCIA('min')
    factor_dict['FAC_MCIB_biddommean'] = fac_generator.FAC_MCIB('biddommean')
    factor_dict['FAC_MCIB_min'] = fac_generator.FAC_MCIB('min')
    factor_dict['FAC_MFI_biddommean'] = fac_generator.FAC_MFI('biddommean')
    factor_dict['FAC_MLQSweight_biddommean'] = fac_generator.FAC_MLQSweight('biddommean')
    factor_dict['FAC_MLQSweight_corrAskwap'] = fac_generator.FAC_MLQSweight('corrAsk')
    factor_dict['FAC_MLQSweight_max'] = fac_generator.FAC_MLQSweight('max')
    factor_dict['FAC_MLQSweight_min'] = fac_generator.FAC_MLQSweight('min')
    factor_dict['FAC_MOFI_kurtosis'] = fac_generator.FAC_MOFI('kurtosis')
    factor_dict['FAC_MOFI_upmean'] = fac_generator.FAC_MOFI('upmean')
    factor_dict['FAC_MPB_Volraiseap'] = fac_generator.FAC_MPB('Volraise')
    factor_dict['FAC_MPB_biddommean'] = fac_generator.FAC_MPB('biddommean')
    factor_dict['FAC_MPB_max'] = fac_generator.FAC_MPB('max')
    factor_dict['FAC_MPB_std'] = fac_generator.FAC_MPB('std')
    factor_dict['FAC_MPB_vwap'] = fac_generator.FAC_MPB('vwap')
    factor_dict['FAC_MPC_max'] = fac_generator.FAC_MPC('max')
    factor_dict['FAC_MPC_min'] = fac_generator.FAC_MPC('min')
    factor_dict['FAC_MPC_std'] = fac_generator.FAC_MPC('std')
    factor_dict['FAC_Momentum_Decay_min'] = fac_generator.FAC_Momentum_Decay('min')
    factor_dict['FAC_NR_askdommean'] = fac_generator.FAC_NR('askdommean')
    factor_dict['FAC_NR_biddommean'] = fac_generator.FAC_NR('biddommean')
    factor_dict['FAC_NR_std'] = fac_generator.FAC_NR('std')
    factor_dict['FAC_OFI1_biddommean'] = fac_generator.FAC_OFI1('biddommean')
    factor_dict['FAC_OFI1_last'] = fac_generator.FAC_OFI1('last')
    factor_dict['FAC_OFI1_min'] = fac_generator.FAC_OFI1('min')
    factor_dict['FAC_OFI1_upmean'] = fac_generator.FAC_OFI1('upmean')
    factor_dict['FAC_OFI2_Volraiseap'] = fac_generator.FAC_OFI2('Volraise')
    factor_dict['FAC_OFI3_Volraiseap'] = fac_generator.FAC_OFI3('Volraise')
    factor_dict['FAC_OFI3_biddommean'] = fac_generator.FAC_OFI3('biddommean')
    factor_dict['FAC_OFI3_vwap'] = fac_generator.FAC_OFI3('vwap')
    factor_dict['FAC_OFI4_biddommean'] = fac_generator.FAC_OFI4('biddommean')
    factor_dict['FAC_OFI5_vwap'] = fac_generator.FAC_OFI5('vwap')
    factor_dict['FAC_OIR_askdommean'] = fac_generator.FAC_OIR('askdommean')
    factor_dict['FAC_OIR_min'] = fac_generator.FAC_OIR('min')
    factor_dict['FAC_OI_CHG_askdommean'] = fac_generator.FAC_OI_CHG('askdommean')
    factor_dict['FAC_OI_CHG_corrBidwap'] = fac_generator.FAC_OI_CHG('corrBid')
    factor_dict['FAC_OI_CHG_downmean'] = fac_generator.FAC_OI_CHG('downmean')
    factor_dict['FAC_OI_CHG_min'] = fac_generator.FAC_OI_CHG('min')
    factor_dict['FAC_OI_GP_CH_upmean'] = fac_generator.FAC_OI_GP_CH('upmean')
    factor_dict['FAC_OI_Price_Accel_downmean'] = fac_generator.FAC_OI_Price_Accel('downmean')
    factor_dict['FAC_OI_V_DIV_Mstdwap'] = fac_generator.FAC_OI_V_DIV('Mstdwap')
    factor_dict['FAC_OI_V_DIV_biddommean'] = fac_generator.FAC_OI_V_DIV('biddommean')
    factor_dict['FAC_OI_V_DIV_upmean'] = fac_generator.FAC_OI_V_DIV('upmean')
    factor_dict['FAC_ORDER_SLOPE_A_biddommean'] = fac_generator.FAC_ORDER_SLOPE_A('biddommean')
    factor_dict['FAC_ORDER_SLOPE_A_std'] = fac_generator.FAC_ORDER_SLOPE_A('std')
    factor_dict['FAC_ORDER_SLOPE_A_vwap'] = fac_generator.FAC_ORDER_SLOPE_A('vwap')
    factor_dict['FAC_ORDER_SLOPE_B_biddommean'] = fac_generator.FAC_ORDER_SLOPE_B('biddommean')
    factor_dict['FAC_ORDER_SLOPE_B_max'] = fac_generator.FAC_ORDER_SLOPE_B('max')
    factor_dict['FAC_PIR_first'] = fac_generator.FAC_PIR('first')
    factor_dict['FAC_PIR_last'] = fac_generator.FAC_PIR('last')
    factor_dict['FAC_PIR_max'] = fac_generator.FAC_PIR('max')
    factor_dict['FAC_PIR_min'] = fac_generator.FAC_PIR('min')
    factor_dict['FAC_PIR_std'] = fac_generator.FAC_PIR('std')
    factor_dict['FAC_PRICE_PRESSURE_askdommean'] = fac_generator.FAC_PRICE_PRESSURE('askdommean')
    factor_dict['FAC_PRICE_STD_A_askdommean'] = fac_generator.FAC_PRICE_STD_A('askdommean')
    factor_dict['FAC_PRICE_STD_A_biddommean'] = fac_generator.FAC_PRICE_STD_A('biddommean')
    factor_dict['FAC_PRICE_STD_B_Volraiseap'] = fac_generator.FAC_PRICE_STD_B('Volraise')
    factor_dict['FAC_PRICE_STD_B_askdommean'] = fac_generator.FAC_PRICE_STD_B('askdommean')
    factor_dict['FAC_PRICE_STD_B_biddommean'] = fac_generator.FAC_PRICE_STD_B('biddommean')
    factor_dict['FAC_PRICE_VOL_CORR_A_biddommean'] = fac_generator.FAC_PRICE_VOL_CORR_A('biddommean')
    factor_dict['FAC_PRICE_VOL_CORR_B_TrendRevmean'] = fac_generator.FAC_PRICE_VOL_CORR_B('trend_rev')
    factor_dict['FAC_PVcorrsub_a2b2_biddommean'] = fac_generator.FAC_PVcorrsub_a2b2('biddommean')
    factor_dict['FAC_PVcorrsub_a2b2_mean'] = fac_generator.FAC_PVcorrsub_a2b2('mean')
    factor_dict['FAC_PVcorrsub_a3b3_Volraiseap'] = fac_generator.FAC_PVcorrsub_a3b3('Volraise')
    factor_dict['FAC_PVcorrsub_a4b4_askdommean'] = fac_generator.FAC_PVcorrsub_a4b4('askdommean')
    factor_dict['FAC_PVcorrsub_a4b4_biddommean'] = fac_generator.FAC_PVcorrsub_a4b4('biddommean')
    factor_dict['FAC_PVcorrsub_a4b4_corrAskwap'] = fac_generator.FAC_PVcorrsub_a4b4('corrAsk')
    factor_dict['FAC_PVcorrsub_a4b4_vwap'] = fac_generator.FAC_PVcorrsub_a4b4('vwap')
    factor_dict['FAC_PVcorrsub_a5b5_TrendRevmean'] = fac_generator.FAC_PVcorrsub_a5b5('trend_rev')
    factor_dict['FAC_PVcorrsub_a5b5_biddommean'] = fac_generator.FAC_PVcorrsub_a5b5('biddommean')
    factor_dict['FAC_PVcorrsub_a5b5_corrAskwap'] = fac_generator.FAC_PVcorrsub_a5b5('corrAsk')
    factor_dict['FAC_PVcorrsub_a5b5_min'] = fac_generator.FAC_PVcorrsub_a5b5('min')
    factor_dict['FAC_RKurt_TrendRevmean'] = fac_generator.FAC_RKurt('trend_rev')
    factor_dict['FAC_RKurt_askdommean'] = fac_generator.FAC_RKurt('askdommean')
    factor_dict['FAC_RKurt_biddommean'] = fac_generator.FAC_RKurt('biddommean')
    factor_dict['FAC_RKurt_min'] = fac_generator.FAC_RKurt('min')
    factor_dict['FAC_RKurt_vwap'] = fac_generator.FAC_RKurt('vwap')
    factor_dict['FAC_RSI_biddommean'] = fac_generator.FAC_RSI('biddommean')
    factor_dict['FAC_RTN_JUMP_MADmean'] = fac_generator.FAC_RTN_JUMP('MADmean')
    factor_dict['FAC_RTN_JUMP_biddommean'] = fac_generator.FAC_RTN_JUMP('biddommean')
    factor_dict['FAC_RTN_JUMP_downmean'] = fac_generator.FAC_RTN_JUMP('downmean')
    factor_dict['FAC_RTN_JUMP_mean'] = fac_generator.FAC_RTN_JUMP('mean')
    factor_dict['FAC_RUSHSTOPBID_TrendRevmean'] = fac_generator.FAC_RUSHSTOPBID('trend_rev')
    factor_dict['FAC_RUSHSTOPBID_min'] = fac_generator.FAC_RUSHSTOPBID('min')
    factor_dict['FAC_RVar_askdommean'] = fac_generator.FAC_RVar('askdommean')
    factor_dict['FAC_RVar_biddommean'] = fac_generator.FAC_RVar('biddommean')
    factor_dict['FAC_RVar_down_rate_biddommean'] = fac_generator.FAC_RVar_down_rate('biddommean')
    factor_dict['FAC_RVar_std'] = fac_generator.FAC_RVar('std')
    factor_dict['FAC_SOIR_askdommean'] = fac_generator.FAC_SOIR('askdommean')
    factor_dict['FAC_SOIR_max'] = fac_generator.FAC_SOIR('max')
    factor_dict['FAC_SPREAD_askdommean'] = fac_generator.FAC_SPREAD('askdommean')
    factor_dict['FAC_SPREAD_biddommean'] = fac_generator.FAC_SPREAD('biddommean')
    factor_dict['FAC_SPREAD_min'] = fac_generator.FAC_SPREAD('min')
    factor_dict['FAC_STREN_Volraiseap'] = fac_generator.FAC_STREN('Volraise')
    factor_dict['FAC_STREN_corrBidwap'] = fac_generator.FAC_STREN('corrBid')
    factor_dict['FAC_STREN_kurtosis'] = fac_generator.FAC_STREN('kurtosis')
    factor_dict['FAC_STREN_upmean'] = fac_generator.FAC_STREN('upmean')
    factor_dict['FAC_STREN_vwap'] = fac_generator.FAC_STREN('vwap')
    factor_dict['FAC_VWAP_Deviation_TrendRevmean'] = fac_generator.FAC_VWAP_Deviation('trend_rev')
    factor_dict['FAC_VWAP_Deviation_biddommean'] = fac_generator.FAC_VWAP_Deviation('biddommean')
    factor_dict['FAC_ask1_vmean_20_biddommean'] = fac_generator.FAC_ask1_vmean_20('biddommean')
    factor_dict['FAC_ask2_vmean_20_MADmean'] = fac_generator.FAC_ask2_vmean_20('MADmean')
    factor_dict['FAC_ask2_vmean_20_biddommean'] = fac_generator.FAC_ask2_vmean_20('biddommean')
    factor_dict['FAC_ask2_vmean_20_max'] = fac_generator.FAC_ask2_vmean_20('max')
    factor_dict['FAC_ask3_vmean_20_min'] = fac_generator.FAC_ask3_vmean_20('min')
    factor_dict['FAC_ask3_vmean_20_upmean'] = fac_generator.FAC_ask3_vmean_20('upmean')
    factor_dict['FAC_ask4_vmean_20_askdommean'] = fac_generator.FAC_ask4_vmean_20('askdommean')
    factor_dict['FAC_ask4_vmean_20_biddommean'] = fac_generator.FAC_ask4_vmean_20('biddommean')
    factor_dict['FAC_ask5_vmean_20_askdommean'] = fac_generator.FAC_ask5_vmean_20('askdommean')
    factor_dict['FAC_ask5_vmean_20_biddommean'] = fac_generator.FAC_ask5_vmean_20('biddommean')
    factor_dict['FAC_ask5_vmean_20_min'] = fac_generator.FAC_ask5_vmean_20('min')
    factor_dict['FAC_bid1_vmean_20_MADmean'] = fac_generator.FAC_bid1_vmean_20('MADmean')
    factor_dict['FAC_bid1_vmean_20_TrendRevmean'] = fac_generator.FAC_bid1_vmean_20('trend_rev')
    factor_dict['FAC_bid2_vmean_20_TrendRevmean'] = fac_generator.FAC_bid2_vmean_20('trend_rev')
    factor_dict['FAC_bid2_vmean_20_askdommean'] = fac_generator.FAC_bid2_vmean_20('askdommean')
    factor_dict['FAC_bid2_vmean_20_biddommean'] = fac_generator.FAC_bid2_vmean_20('biddommean')
    factor_dict['FAC_bid3_vmean_20_askdommean'] = fac_generator.FAC_bid3_vmean_20('askdommean')
    factor_dict['FAC_bid3_vmean_20_min'] = fac_generator.FAC_bid3_vmean_20('min')
    factor_dict['FAC_bid4_vmean_20_TrendRevmean'] = fac_generator.FAC_bid4_vmean_20('trend_rev')
    factor_dict['FAC_bid4_vmean_20_askdommean'] = fac_generator.FAC_bid4_vmean_20('askdommean')
    factor_dict['FAC_bid4_vmean_20_biddommean'] = fac_generator.FAC_bid4_vmean_20('biddommean')
    factor_dict['FAC_bid4_vmean_20_last'] = fac_generator.FAC_bid4_vmean_20('last')
    factor_dict['FAC_bid4_vmean_20_min'] = fac_generator.FAC_bid4_vmean_20('min')
    factor_dict['FAC_bid5_vmean_20_askdommean'] = fac_generator.FAC_bid5_vmean_20('askdommean')
    factor_dict['FAC_bid5_vmean_20_biddommean'] = fac_generator.FAC_bid5_vmean_20('biddommean')
    factor_dict['FAC_bid5_vmean_20_min'] = fac_generator.FAC_bid5_vmean_20('min')
    factor_dict['FAC_bid_amount_sub20_Volraiseap'] = fac_generator.FAC_bid_amount_sub20('Volraise')
    factor_dict['FAC_bid_amount_sub20_askdommean'] = fac_generator.FAC_bid_amount_sub20('askdommean')
    factor_dict['FAC_fibonacci_retracement_askdommean'] = fac_generator.FAC_fibonacci_retracement('askdommean')
    factor_dict['FAC_fibonacci_retracement_first'] = fac_generator.FAC_fibonacci_retracement('first')
    factor_dict['FAC_fibonacci_retracement_min'] = fac_generator.FAC_fibonacci_retracement('min')
    factor_dict['FAC_resiliency_TrendRevmean'] = fac_generator.FAC_resiliency('trend_rev')
    factor_dict['FAC_resiliency_askdommean'] = fac_generator.FAC_resiliency('askdommean')
    factor_dict['FAC_resiliency_biddommean'] = fac_generator.FAC_resiliency('biddommean')
    factor_dict['FAC_shortQUA_biddommean'] = fac_generator.FAC_shortQUA('biddommean')
    factor_dict['FAC_sub_a3b3_vmean_20_askdommean'] = fac_generator.FAC_sub_a3b3_vmean_20('askdommean')
    factor_dict['FAC_sub_a4b4_vmean_20_askdommean'] = fac_generator.FAC_sub_a4b4_vmean_20('askdommean')
    factor_dict['FAC_sub_a5b5_vmean_20_askdommean'] = fac_generator.FAC_sub_a5b5_vmean_20('askdommean')
    factor_dict['FAC_sub_a5b5_vmean_20_biddommean'] = fac_generator.FAC_sub_a5b5_vmean_20('biddommean')
    factor_dict['FAC_sub_a5b5_vmean_20_min'] = fac_generator.FAC_sub_a5b5_vmean_20('min')
    # ========== 跨品种因子 (17个) ==========
    factor_dict['A_M_closepctchg20_sub'] = fac_generator.closepctchg_sub(main_symbol='A', symbol1='A', symbol2='M', window=20)
    factor_dict['A_M_closepctchg5_sub'] = fac_generator.closepctchg_sub(main_symbol='A', symbol1='A', symbol2='M', window=5)
    factor_dict['A_M_cvcorr10_diff'] = fac_generator.cvcorr10_diff(main_symbol='A', symbol1='A', symbol2='M')
    factor_dict['A_M_oi5_diff'] = fac_generator.oi5_diff(main_symbol='A', symbol1='A', symbol2='M')
    factor_dict['A_M_volumediv20_diff5'] = fac_generator.volumediv_diff(main_symbol='A', symbol1='A', symbol2='M', window1=20, window2=5)
    factor_dict['A_M_volumediv5_diff5'] = fac_generator.volumediv_diff(main_symbol='A', symbol1='A', symbol2='M', window1=5, window2=5)
    factor_dict['A_Y_closepctchg20_sub'] = fac_generator.closepctchg_sub(main_symbol='A', symbol1='A', symbol2='Y', window=20)
    factor_dict['A_Y_closepctchg5_sub'] = fac_generator.closepctchg_sub(main_symbol='A', symbol1='A', symbol2='Y', window=5)
    factor_dict['A_Y_oi5_diff'] = fac_generator.oi5_diff(main_symbol='A', symbol1='A', symbol2='Y')
    factor_dict['A_Y_vcorr10'] = fac_generator.vcorr10(main_symbol='A', symbol1='A', symbol2='Y')
    factor_dict['A_Y_volumediv20_diff5'] = fac_generator.volumediv_diff(main_symbol='A', symbol1='A', symbol2='Y', window1=20, window2=5)
    factor_dict['P_A_closepctchg20_sub'] = fac_generator.closepctchg_sub(main_symbol='A', symbol1='P', symbol2='A', window=20)
    factor_dict['P_A_closepctchg5_sub'] = fac_generator.closepctchg_sub(main_symbol='A', symbol1='P', symbol2='A', window=5)
    factor_dict['P_A_cvcorr10_diff'] = fac_generator.cvcorr10_diff(main_symbol='A', symbol1='P', symbol2='A')
    factor_dict['P_A_oi5_diff'] = fac_generator.oi5_diff(main_symbol='A', symbol1='P', symbol2='A')
    factor_dict['P_A_vcorr10'] = fac_generator.vcorr10(main_symbol='A', symbol1='P', symbol2='A')
    factor_dict['P_A_volumediv20_diff5'] = fac_generator.volumediv_diff(main_symbol='A', symbol1='P', symbol2='A', window1=20, window2=5)
    # ========== 普通因子 (25个) ==========
    factor_dict['CCI'] = fac_generator.CCI()
    factor_dict['LR'] = fac_generator.LR()
    factor_dict['MAdiff_Vol_div'] = fac_generator.MAdiff_Vol_div()
    factor_dict['PV_corr_std'] = fac_generator.PV_corr_std()
    factor_dict['QST'] = fac_generator.QST()
    factor_dict['RPP_22D'] = fac_generator.RPP_22D()
    factor_dict['RPP_2H'] = fac_generator.RPP_2H()
    factor_dict['RPP_30M'] = fac_generator.RPP_30M()
    factor_dict['RPP_4H'] = fac_generator.RPP_4H()
    factor_dict['RPP_5D'] = fac_generator.RPP_5D()
    factor_dict['VHF'] = fac_generator.VHF()
    factor_dict['XSMOM1M'] = fac_generator.XSMOM1M()
    factor_dict['bar5_trend_corr'] = fac_generator.bar5_trend_corr()
    factor_dict['before_bot_price_diff'] = fac_generator.before_bot_price_diff()
    factor_dict['day_first10colarrate'] = fac_generator.day_first10colarrate()
    factor_dict['day_first10rev'] = fac_generator.day_first10rev()
    factor_dict['day_first4redcorr'] = fac_generator.day_first4redcorr()
    factor_dict['day_jump'] = fac_generator.day_jump()
    factor_dict['down_shadow_5mean'] = fac_generator.down_shadow_5mean()
    factor_dict['down_shadow_5std'] = fac_generator.down_shadow_5std()
    factor_dict['hour'] = fac_generator.hour()
    factor_dict['term_rtn'] = fac_generator.term_rtn()
    factor_dict['up_shadow_5mean'] = fac_generator.up_shadow_5mean()
    factor_dict['up_shadow_5std'] = fac_generator.up_shadow_5std()
    factor_dict['volatility_rg'] = fac_generator.volatility_rg()
    # 创建DataFrame
    fac_df = pd.DataFrame(factor_dict, index=valid_index)
    fac_df = fac_df.replace([np.inf, -np.inf], np.nan)

    print(fac_df.shape)
    
    return fac_df[factor_col].round(8)

if __name__ == "__main__":
    # 加载配置
    config = load_config()
    
    # 从文件名自动解析策略名称和品种
    strategy_name = get_strategy_from_filename()
    print(f"检测到策略名称: {strategy_name}")
    strategies_config = config.get(strategy_name, {})

    main_symbol, other_symbols = strategies_config.get("main_symbol", None), strategies_config.get("other_symbols", None)
    print(f"从文件名解析品种: 主品种={main_symbol}, 关联品种={other_symbols}")

    config_loaded = load_config_for_symbols(config, main_symbol, other_symbols)
    time_config = config_loaded['time_config']

    # 构建需要加载的合约列表
    # 顺序：主品种、其他品种1、其他品种2、其他品种3、下一合约
    instruments_to_load = []
    instruments_to_load.append((f"{main_symbol}_main", f"{main_symbol}_tick", f"{main_symbol}_min_data_concat"))
    for symbol in other_symbols:
        if f"{symbol}_main" in config["instruments"]:
            instruments_to_load.append((f"{symbol}_main", f"{symbol}_tick", f"{symbol}_min_data_concat"))
    instruments_to_load.append((f"{main_symbol}_next", f"{main_symbol}_tick_next", f"{main_symbol}_min_data_concat_next"))

    model_lst, weight_lst, factor_col = load_model_lst(config_loaded)
    
    # 算旧的因子
    print(f"正在计算前序因子...")
    fac_df_old = load_fac_df_old(
        factor_col, 
        config_loaded['instrument_list'], 
        config_loaded['recent_data_path'],  
        trade_type = config_loaded['trade_hours'],
        dict_keys=[main_symbol] + other_symbols,
        generate_factor_dataframe = generate_factor_dataframe
    )

    pred_df = pd.DataFrame(
        [model_lst[i].predict(fac_df_old) for i in range(len(model_lst))], 
        columns=fac_df_old.index, 
        index=[f'model_{i+1}' for i in range(len(model_lst))]
    ).T
    pred_df['weighted'] = pred_df.mul(weight_lst, axis=1).sum(axis=1) / sum(weight_lst)

    if os.path.exists(config_loaded['save_path']):
        shutil.rmtree(config_loaded['save_path'])  
        print(f"已删除文件夹: {config_loaded['save_path']}")

    os.makedirs(config_loaded['save_path'], exist_ok=True)
    print(f"已创建文件夹: {config_loaded['save_path']}")
    pred_df.to_csv(f"{config_loaded['save_path']}/pred_df.csv")
    fac_df_old.to_csv(f"{config_loaded['save_path']}/fac_old.csv")

    time_before = pd.Timestamp('2026-01-25 20:59:00.000000')
    now_pos = 0
    now_holding = 0
    publisher = ZMQPublisher(port=config_loaded.get("publisher_port"))

    # 标记是否已经保存过15点的数据
    data_saved_at_15 = False

    while True:
        # 只读取最后一行的datetime来判断是否有新分钟数据，大幅减少CPU占用
        datetime_str = read_table(
            config["instruments"][f"{config_loaded['main_symbol']}_main"], 
            db_path=config_loaded['db_path'], 
            word=False,
            only_datetime=True  # 使用优化后的轻量查询
        )

        if datetime_str is None:
            print("Empty dataframe. Waiting...")
            time_module.sleep(10)
            continue

        try:
            time_recently = pd.to_datetime(datetime_str)
        except ValueError as e:
            print(f"Datetime format error: {e}, datetime_str: {datetime_str}. Waiting...")
            time_module.sleep(10)
            continue

        # 检查是否是15点，如果是则保存数据库中所有表的数据为feather格式
        # 使用 >= 15 以确保即使时间跳过了15:00也能触发保存
        if time_recently.hour >= 15 and not data_saved_at_15:
            print("=" * 60)
            print(f"检测到15点收盘时间 ({time_recently})，开始保存数据库中的所有表数据...")
            print("=" * 60)
            
            # 创建保存目录 - 使用config中save_path的父目录（去掉品种名称）
            save_path_parent = os.path.dirname(config_loaded['save_path'])
            data_save_path = os.path.join(save_path_parent, "data")
            os.makedirs(data_save_path, exist_ok=True)
            
            # 获取数据库连接
            db_path = config_loaded['db_path']
            conn = sqlite3.connect(f'{db_path}/tick_data.db')
            
            # 获取所有表名
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            saved_count = 0
            for table in tables:
                table_name = table[0]
                # 只保存 tick_data_ 开头的表
                if table_name.startswith('tick_data_'):
                    try:
                        # 读取表数据
                        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                        
                        if not df.empty:
                            # 提取合约代码（去掉 tick_data_ 前缀）
                            instrument_name = table_name.replace('tick_data_', '')
                            # 保存为feather格式
                            feather_path = f"{data_save_path}/{instrument_name}.feather"
                            df.to_feather(feather_path)
                            print(f"✅ 已保存: {table_name} -> {feather_path} ({len(df)} 行)")
                            saved_count += 1
                        else:
                            print(f"⚠️ 表为空，跳过: {table_name}")
                    except Exception as e:
                        print(f"❌ 保存失败: {table_name}, 错误: {e}")
            
            conn.close()
            print("=" * 60)
            print(f"数据保存完成！共保存 {saved_count} 个表到 {data_save_path}")
            print("=" * 60)
            
            data_saved_at_15 = True
            print("程序将在15点收盘后退出...")
            exit(0)  # 完全退出程序

        trigger = (
            time_recently.minute != time_before.minute) & (
            not (time_before.hour == 20 and time_before.minute == 59)) & (
            not (time_before.hour == 8 and time_before.minute == 59)) & (
            not (time_recently.hour == 8 and time_recently.minute == 59)
        ) 

        if trigger:
            time_module.sleep(0.5)  # 睡半秒保证数据都写入完毕

            if is_in_no_trade_period(time_recently.time(), time_config):
                print(f'当前时间{time_recently.time()}在不交易时间段内，暂时不交易')

            print('-----------------新一分钟数据到达，开始计算因子和预测-----------------')
            print(f'当前时间{time_recently.time()}')
            readable_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            results = Parallel(n_jobs=2)(
                delayed(load_instrument_parallel)(
                    ins_key, tick_var, min_var, config, config_loaded, time_recently
                )
                for ins_key, tick_var, min_var in instruments_to_load
            )

            data_dict = {}
            for tick_var, tick_data, min_var, min_data in results:
                data_dict[tick_var] = tick_data
                data_dict[min_var] = min_data
            
            var_name_main = f'{config_loaded["main_symbol"]}_min_data_concat'
            var_name_next = f'{config_loaded["main_symbol"]}_min_data_concat_next'
            if var_name_main in data_dict and var_name_next in data_dict:
                data_dict[var_name_next] = data_dict[var_name_next].set_index('datetime').reindex(
                    index=data_dict[var_name_main].datetime
                ).reset_index()
            
            if not check_minute_data_consistency(instruments_to_load, data_dict):
                print("⚠️ 数据检查未通过，进入下一循环")
                continue

            main_tick_var = instruments_to_load[0][1]  
            main_min_var = instruments_to_load[0][2]   
            other_min_vars = [item[2] for item in instruments_to_load[1:-1]]  
            next_min_var = instruments_to_load[-1][2]  

            # 创建Factor_generator
            time1 = time_module.time()
            fac_generator = Factor_generator(
                data_dict[main_tick_var],      # 主品种tick
                data_dict[main_min_var],       # 主品种分钟数据
                data_dict[other_min_vars[0]],  # 其他品种1分钟数据
                data_dict[other_min_vars[1]],  # 其他品种2分钟数据
                data_dict[other_min_vars[2]],  # 其他品种3分钟数据
                data_dict[next_min_var]        # 主品种下一合约分钟数据
            )
            
            # ---------------------------------------------------------
            # 在这里插入前面每个品种算出来的数据的代码，其他地方不能动        
            # ---------------------------------------------------------
            # 保存各品种数据到CSV
            # data_dict[main_tick_var].to_csv(f"{config_loaded['save_path']}/{main_symbol}_tick_{time_recently}.csv", index=False)
            # data_dict[main_min_var].to_csv(f"{config_loaded['save_path']}/{main_symbol}_min_{time_recently}.csv", index=False)
            # data_dict[other_min_vars[0]].to_csv(f"{config_loaded['save_path']}/{other_symbols[0]}_min_{time_recently}.csv", index=False)
            # data_dict[other_min_vars[1]].to_csv(f"{config_loaded['save_path']}/{other_symbols[1]}_min_{time_recently}.csv", index=False)
            # data_dict[other_min_vars[2]].to_csv(f"{config_loaded['save_path']}/{other_symbols[2]}_min_{time_recently}.csv", index=False)
            # data_dict[next_min_var].to_csv(f"{config_loaded['save_path']}/{main_symbol}_next_min_{time_recently}.csv", index=False)

            fac_generator.dict_keys = [main_symbol] + other_symbols + [f'{main_symbol}_next']
            fac_generator.load_df_names()

            fac_df = generate_factor_dataframe(fac_generator, fac_generator.valid_index, factor_col)
            pred = pd.DataFrame(
                [model_lst[i].predict(fac_df) for i in range(len(model_lst))], 
                columns=fac_df.index, 
                index=[f'model_{i+1}' for i in range(len(model_lst))]
            ).T
            pred['weighted'] = pred.mul(weight_lst, axis=1).sum(axis=1) / sum(weight_lst)
            time2 = time_module.time()

            pred = pred.iloc[-3:]
            
            pred_df = pd.concat([pred_df, pred])
            pred_df = pred_df[~pred_df.index.duplicated(keep='last')]
            
            now_pos, now_holding, weighted, thresholds_df, df = run_345(
                pred_df, 
                th_open=config_loaded['th1'], 
                th_close=config_loaded['th2'], 
                now_pos=now_pos, 
                now_holding=now_holding, 
                max_holding=config_loaded["holding_period_max"],
                bar_in_day=config_loaded['bar_in_day'],
                time_config=config_loaded['time_config']
            )

            try:
                time_recently_str = pd.to_datetime(time_recently).replace(second=0).strftime("%Y-%m-%d_%H-%M-00")
                pred_filename = f"{config_loaded['save_path']}/predictions_{time_recently_str}.csv"
                fac_filename = f"{config_loaded['save_path']}/factors_{time_recently_str}.csv"
                df.iloc[-10:].to_csv(pred_filename)
                fac_df.iloc[-10:].to_csv(fac_filename)
                
                percent_fac = round((df.iloc[:-1]['weighted_s'].dropna() < df.iloc[-1]['weighted_s']).mean() * 100, 4)
                status_info = {
                    'timestamp': readable_time,
                    'time_recently': str(time_recently),
                    'now_pos': now_pos,
                    'now_holding': now_holding,
                    'fac_now': df.iloc[-1]['weighted_s'],
                    'current_quantile(%)': percent_fac, 
                    'factor_calculation_time': round(time2 - time1, 2)
                }
                
                status_filename = f"{config_loaded['save_path']}/trading_status_{time_recently_str}.json"
                with open(status_filename, 'w', encoding='utf-8') as f:
                    json.dump(status_info, f, indent=2, ensure_ascii=False)

                # pic_save_path = f"{config_loaded['save_path']}/trading_status_{time_recently_str}.png"
                # plt.figure(figsize=(12, 6))
                # plt.plot(thresholds_df.index, thresholds_df['long_open'], label='long_open')
                # plt.plot(thresholds_df.index, thresholds_df['long_close'], label='long_close')
                # plt.plot(thresholds_df.index, thresholds_df['short_open'], label='short_open')
                # plt.plot(thresholds_df.index, thresholds_df['short_close'], label='short_close')
                # plt.plot(thresholds_df.index, thresholds_df['now'], label='current_value', linewidth=2, marker='o')
                # plt.legend()
                # plt.grid(True, alpha=0.3)
                # plt.savefig(pic_save_path)
                # plt.close()

                print(f"当前状态: 仓位={now_pos}, 持仓时间={now_holding}")

            except Exception as e:
                print(f"保存数据时出错: {e}")

        time_before = time_recently
        time_module.sleep(0.5)
