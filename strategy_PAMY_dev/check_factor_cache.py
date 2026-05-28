#!/usr/bin/env python3
"""
简单比对 factor_cache 与研究环境 all_factor.feather 的差异。
直接运行：python3 check_factor_cache.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

TOL_ABS = 0.01
TOL_REL = 0.05


def check_symbol(cache_file, research_root='/mnt/Data/writable/liaoyuyang/factor'):
    # 解析品种和日期，如 A_fac_2026-03-23.parquet -> symbol=A, date=2026-03-23
    stem = cache_file.stem
    parts = stem.split('_')
    symbol = parts[0]
    date_str = parts[2]
    target_date = pd.to_datetime(date_str).date()

    print(f"\n[{symbol}] 读取 factor_cache ...")
    fac_rt = pd.read_parquet(cache_file)
    fac_rt.index = pd.to_datetime(fac_rt.index)

    # 读取研究环境：先只读 meta 定位 instrument
    research_path = Path(research_root) / symbol / 'all_fac' / 'all_factor.feather'
    if not research_path.exists():
        print(f"  研究环境文件不存在: {research_path}")
        return

    print(f"  读取研究环境 meta ...")
    meta = pd.read_feather(research_path, columns=['datetime', 'instrument'])
    meta['datetime'] = pd.to_datetime(meta['datetime'])
    meta_target = meta[meta['datetime'].dt.date == target_date]
    if len(meta_target) == 0:
        print(f"  {date_str} 在研究环境中无数据")
        return
    inst = meta_target['instrument'].iloc[0]

    # 确定共有因子列，减少读取列数
    drop_cols = {'datetime', 'instrument', 'hour'}
    rt_cols = set(fac_rt.columns) - drop_cols
    # 研究环境的列很多，先取 feather 的列名（不读数据）
    # pandas 不支持只读列名，我们用 pyarrow 读 schema
    import pyarrow.feather as feather
    schema = feather.read_table(research_path, columns=None).schema
    rs_cols = set(str(c) for c in schema.names) - drop_cols
    common_factors = sorted(rt_cols & rs_cols)

    print(f"  读取研究环境 {len(common_factors)} 列 + datetime + instrument ...")
    read_cols = ['datetime', 'instrument'] + common_factors
    fac_all = pd.read_feather(research_path, columns=read_cols)
    fac_all['datetime'] = pd.to_datetime(fac_all['datetime'])
    fac_rs = fac_all[(fac_all['instrument'] == inst) & (fac_all['datetime'].dt.date == target_date)].copy()
    del fac_all
    fac_rs = fac_rs.set_index('datetime')

    # 对齐
    common_idx = fac_rt.index.intersection(fac_rs.index)
    if len(common_idx) == 0:
        print(f"  时间对齐失败，实时:{fac_rt.index.min()}~{fac_rt.index.max()} 研究:{fac_rs.index.min()}~{fac_rs.index.max()}")
        return

    rt = fac_rt.loc[common_idx, common_factors]
    rs = fac_rs.loc[common_idx, common_factors]

    print(f"  对齐分钟: {len(common_idx)} | 比对因子: {len(common_factors)}")

    # 差异
    diff = (rs - rt).abs()
    mask_na = rs.isna() | rt.isna()
    diff = diff.where(~mask_na, np.nan)

    with np.errstate(divide='ignore', invalid='ignore'):
        denom = rs.abs().replace(0, np.nan)
        rel = diff / denom
        rel = rel.fillna(0)

    ok = (diff < TOL_ABS) | (rel < TOL_REL)
    ok = ok.where(~mask_na, np.nan)

    fail = (ok == False).sum(axis=0)
    total = ok.notna().sum(axis=0)
    ratio = fail / total

    summary = pd.DataFrame({
        'total': total,
        'fail': fail,
        'ratio': ratio,
        'max_diff': diff.max(axis=0),
    }).sort_values('ratio', ascending=False)

    fully_bad = summary[summary['ratio'] == 1.0]
    perfect = summary[summary['ratio'] == 0.0]

    print(f"  完全匹配: {len(perfect)} | 完全不一致: {len(fully_bad)} | 部分: {len(summary) - len(perfect) - len(fully_bad)}")

    if len(fully_bad) > 0:
        print(f"  🔴 完全不一致 Top10:")
        for fac in fully_bad.index[:10]:
            print(f"     {fac}: max_diff={fully_bad.loc[fac, 'max_diff']:.4f}")
        if len(fully_bad) > 10:
            print(f"     ... 还有 {len(fully_bad)-10} 个")

    bad_ratio = len(fully_bad) / len(common_factors)
    if bad_ratio > 0.3:
        print(f"  ⚠️ 结论: {bad_ratio:.1%} 因子完全不一致，差异很大")
    elif bad_ratio > 0.1:
        print(f"  ⚠️ 结论: {bad_ratio:.1%} 因子完全不一致，有一定差异")
    else:
        print(f"  ✅ 结论: 仅 {bad_ratio:.1%} 因子完全不一致，整体正常")


def main():
    cache_root = Path('/home/strategy_PAMY_dev/factor_cache')
    if not cache_root.exists():
        print("factor_cache 目录不存在")
        return

    files = sorted(cache_root.rglob('*_fac_*.parquet'))
    if not files:
        print("factor_cache 中未找到 *_fac_*.parquet 文件")
        return

    print(f"找到 {len(files)} 个因子缓存文件")
    for f in files:
        check_symbol(f)


if __name__ == '__main__':
    main()
