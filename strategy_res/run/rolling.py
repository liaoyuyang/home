import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('/home/future_commodity')

import pandas as pd
import numpy as np
np.seterr(all='ignore')
import lightgbm as lgb
from pathlib import Path
import os
from typing import List, Tuple
import function_future.pre_train as pt
import function_future.train_model as tm
import function_future.FactorFilter as FF
import function_future.backtest_v3 as bv
import function_future.DataLoader as DL, function_future.date_selection as DS
import function_future.trading_visualization as TV
import function_future.single_fac_eval as sfe

from tqdm.auto import tqdm
from joblib import Parallel, delayed

import time
# print('waiting')
# time.sleep(60*60*2)
# print("ready")

train_end_date_lst = [
    # "2024-01-01",
    # "2024-07-01",
    # "2025-01-01",
    # "2025-07-01",
    "2026-01-01",
    ]

# symbol_lst = ['SM']
# symbol_lst =  ["RB", "I", "SM", "SF", "HC", "SS", "J", "JM", "WR"]
symbol_lst = ["P", "A", "M", "Y", "LH", "B", "C", "CS"]

for symbol in symbol_lst:
    config_loader = DL.InstrumentConfig()
    config_loader.get_instrument_config(symbol)
    train_label = 5
    rtn_mul = 1

    for train_end_date in tqdm(train_end_date_lst[::-1], leave=False, desc=symbol):
        folder_name = f'{symbol}_pred{train_label}_{train_end_date}_rolling'
        fac_df = pd.read_feather(f'/mnt/Data/writable/liaoyuyang/factor/{symbol}/all_fac/all_factor.feather').set_index(['datetime']).loc[:train_end_date]
        fac_df = config_loader.df_cut_time(fac_df, config_loader.get_instrument_config(symbol)['trading_hours'], 10)

        exclude_factors = [
                'datetime', 'instrument',
                ]
        factor_col = [x for x in fac_df.columns if x not in exclude_factors]
        main_fac_piv = fac_df[factor_col]
        rtn_df = pd.read_csv(f'/mnt/Data/writable/liaoyuyang/data/1min/active/main_{symbol}.csv', index_col=0, parse_dates=['ts']).set_index('ts').reindex(index=fac_df.index)
        main_fac_piv['pred_ret'] = rtn_df[f'rtn_{train_label}']
        main_fac_piv = main_fac_piv.replace([np.inf, -np.inf], np.nan)
        main_fac_piv['hour'] = main_fac_piv.index.hour

        pretrainer = pt.Pretrainer(symbol, main_fac_piv, train_end_date, train_label=train_label)
        importance = pretrainer.run_full_pretraining(type_lgb = 'reg') 