#!/usr/bin/env python3
"""
批量因子月度稳定性评估

用法:
    python batch_factor_eval.py --symbols A M Y P --output-dir ./results
    
默认评估 mnt 上所有有 all_factor.feather 的品种。
"""

import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('/home/future_commodity')

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import function_future.FactorFilter as FF
import function_future.DataLoader as DL


def load_factor_data(symbol, train_end_date=None):
    """加载单个品种的因子数据"""
    path = f'/mnt/Data/writable/liaoyuyang/factor/{symbol}/all_fac/all_factor.feather'
    df = pd.read_feather(path).set_index('datetime')
    if train_end_date:
        df = df.loc[:train_end_date]
    # 尝试做时间切割（如果 InstrumentConfig 支持该品种）
    try:
        config_loader = DL.InstrumentConfig()
        config = config_loader.get_instrument_config(symbol)
        df = config_loader.df_cut_time(df, config['trading_hours'], 10)
    except Exception:
        pass
    return df


def eval_symbol(symbol, train_end_date=None,
                max_outlier_ratio=0.25,
                mean_r2_thresh=0.4,
                std_r2_thresh=0.7,
                max_consecutive_outliers=2):
    """评估单个品种的所有因子月度稳定性"""
    df = load_factor_data(symbol, train_end_date)
    
    # 只保留数值列（因子列）
    exclude = ['datetime', 'instrument', 'pred_ret', 'hour',
               'rtn_1', 'rtn_5', 'rtn_10', 'rtn_20']
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    numeric_df = numeric_df.drop(columns=[c for c in exclude if c in numeric_df.columns], errors='ignore')
    
    # 用 FactorFilter 的静态方法计算月度稳定性
    stability_df = FF.FactorFilter.calc_monthly_stability(
        None,  # self 不需要
        numeric_df,
        max_outlier_ratio=max_outlier_ratio,
        mean_r2_thresh=mean_r2_thresh,
        std_r2_thresh=std_r2_thresh,
        max_consecutive_outliers=max_consecutive_outliers
    )
    
    stability_df['symbol'] = symbol
    return stability_df


def batch_eval(symbols, output_dir, train_end_date=None, **kwargs):
    """批量评估多个品种"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    for symbol in tqdm(symbols, desc="评估品种"):
        try:
            df = eval_symbol(symbol, train_end_date, **kwargs)
            # 保存单个品种结果
            df.to_csv(output_dir / f"{symbol}_stability.csv")
            all_results.append(df)
        except Exception as e:
            print(f"⚠️ {symbol} 失败: {e}")
    
    if all_results:
        # 合并所有品种
        combined = pd.concat(all_results)
        combined.to_csv(output_dir / "all_symbols_stability.csv")
        
        # 生成汇总透视表：品种 × 因子 → is_stable
        pivot = combined.reset_index().pivot_table(
            index='factor', columns='symbol', values='is_stable', aggfunc='first'
        )
        pivot.to_csv(output_dir / "stability_pivot.csv")
        
        # 统计每个因子在多少个品种上稳定
        pivot['stable_count'] = pivot.sum(axis=1)
        pivot['stable_ratio'] = pivot['stable_count'] / len(symbols)
        pivot.sort_values('stable_ratio', ascending=False, inplace=True)
        pivot.to_csv(output_dir / "stability_summary.csv")
        
        return combined, pivot
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', nargs='+', default=None,
                       help='指定品种列表，默认自动扫描 mnt 下所有品种')
    parser.add_argument('--output-dir', default='/home/single_factor_eval/results')
    parser.add_argument('--train-end-date', default=None)
    parser.add_argument('--mean-r2-thresh', type=float, default=0.4)
    parser.add_argument('--std-r2-thresh', type=float, default=0.7)
    parser.add_argument('--max-outlier-ratio', type=float, default=0.25)
    args = parser.parse_args()
    
    if args.symbols is None:
        # 自动扫描
        base = Path('/mnt/Data/writable/liaoyuyang/factor')
        args.symbols = sorted([p.name for p in base.iterdir() 
                               if (p / 'all_fac/all_factor.feather').exists()])
        print(f"自动发现 {len(args.symbols)} 个品种: {args.symbols[:10]}...")
    
    combined, pivot = batch_eval(
        args.symbols,
        args.output_dir,
        args.train_end_date,
        mean_r2_thresh=args.mean_r2_thresh,
        std_r2_thresh=args.std_r2_thresh,
        max_outlier_ratio=args.max_outlier_ratio
    )
    
    print(f"\n结果已保存到: {args.output_dir}")
    if pivot is not None:
        print(f"\n稳定因子最多的前10个（跨品种）:")
        print(pivot[['stable_count', 'stable_ratio']].head(10))


if __name__ == '__main__':
    main()
