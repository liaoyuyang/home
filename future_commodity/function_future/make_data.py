import os
from tqdm.auto import tqdm
import sys
sys.path.append(r'/home/future_commodity')
from joblib import Parallel, delayed
import function_future.DataLoader as DL, function_future.date_selection as DS
dl = DL.DataLoader()

def process_contract(symbol, date, contract):
    """
    处理单个合约的worker函数
    """
    loader = DL.FutureDataLoader(symbol)
    output_path = f'/mnt/Data/writable/liaoyuyang/data/level2/{symbol}/{date}_{contract}.feather'
    df = loader.load_contract_data(date, contract)
    df = loader.pre_resample_data(df, date)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_feather(output_path)
    return f"Success: {date}_{contract}"

def process_dates_with_joblib(symbol, dates, n_jobs=10):
    """
    使用joblib多进程处理所有日期和合约
    
    参数:
        symbol: 品种代码
        dates: 日期列表
        n_jobs: 并行任务数，-1表示使用所有CPU核心
    """
    def prepare_single_date(symbol, date):
        """准备单个日期的任务"""
        loader = DL.FutureDataLoader(symbol)
        contracts = loader.get_contracts(symbol, date)
        main_contracts = dl.load_valid_codes(symbol)
        if contracts:
            return [(symbol, date, contract) for contract in contracts if contract in main_contracts]
        return []
    
    tasks_lists = Parallel(n_jobs=n_jobs)(
        delayed(prepare_single_date)(symbol, date)
        for date in tqdm(dates, desc="Preparing tasks")
    )
    
    # 合并所有任务
    tasks = []
    for task_list in tasks_lists:
        if task_list:
            tasks.extend(task_list)

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_contract)(symbol, date, contract) 
        for symbol, date, contract in tqdm(tasks, desc="Processing contracts", mininterval=0.1)
    )
    return results