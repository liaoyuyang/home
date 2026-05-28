"""
因子研究工具模块
提供数据加载、因子计算、可视化等通用功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 路径配置
DATA_DIR = Path('/mnt/Data/writable/liaoyuyang/data/1day/active')
OUTPUT_DIR = Path('/home/day_strategy/factors/output')
OUTPUT_DIR.mkdir(exist_ok=True)

# 商品分类配置
COMMODITY_CLASSIFICATION = {
    "原油": ["FU", "LU", "SC", "BU", "PG"],
    "黑色金属": ["RB", "I", "HC", "WR", "SF", "SM", "SS"],
    "化工": ["RU", "BR", "NR", "L", "TA", "PF", "V", "EG", "MA", "PP", "EB", "UR", "SA", "SH", "PX", "PR"],
    "煤炭": ["JM", "J", "ZC"],
    "轻工": ["FG", "SP", "FB", "BB"],
    "有色金属": ["NI", "SN", "ZN", "PB", "AL", "CU", "AO", "BC", "SI", "LC"],
    "贵金属": ["AU", "AG"],
    "谷物": ["C", "A", "CS", "RR", "RI", "JR", "WH", "PH", "LR"],
    "农副": ["JD", "LH", "AP", "CJ"],
    "软商品": ["SR", "CF", "CY"],
    "油脂油料": ["A", "M", "Y", "OI", "RM", "P", "PK"],
    "航运": ["EC"]
}


def get_symbol_category(symbol: str) -> str:
    """获取品种所属行业分类"""
    for category, symbols in COMMODITY_CLASSIFICATION.items():
        if symbol in symbols:
            return category
    return "其他"


def load_symbols_by_category(category: str) -> Dict[str, pd.DataFrame]:
    """加载指定行业的所有品种数据"""
    symbols = COMMODITY_CLASSIFICATION.get(category, [])
    result = {}
    for symbol in symbols:
        try:
            result[symbol] = load_symbol(symbol)
        except FileNotFoundError:
            pass
    return result


def get_all_categories() -> List[str]:
    """获取所有行业分类名称"""
    return list(COMMODITY_CLASSIFICATION.keys())


def load_all_symbols() -> Dict[str, pd.DataFrame]:
    """加载所有品种的主力合约数据"""
    symbols_data = {}
    for file in DATA_DIR.glob('*_active.feather'):
        symbol = file.stem.replace('_active', '')
        df = pd.read_feather(file)
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        symbols_data[symbol] = df
    return symbols_data


def load_symbol(symbol: str) -> pd.DataFrame:
    """加载单个品种数据"""
    file = DATA_DIR / f'{symbol}_active.feather'
    if not file.exists():
        raise FileNotFoundError(f"找不到品种 {symbol} 的数据")
    df = pd.read_feather(file)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    return df


def normalize_factor(factor: pd.Series, method: str = 'zscore', 
                     lookback: int = 252, min_periods: int = 60) -> pd.Series:
    """
    因子标准化，使分布稳定
    
    Args:
        factor: 原始因子值
        method: 标准化方法 ('zscore', 'rank', 'mad', 'none')
        lookback: 滚动窗口大小
        min_periods: 最小观测数
    """
    if method == 'zscore':
        mean = factor.rolling(window=lookback, min_periods=min_periods).mean()
        std = factor.rolling(window=lookback, min_periods=min_periods).std()
        return (factor - mean) / std
    
    elif method == 'rank':
        return factor.rolling(window=lookback, min_periods=min_periods).apply(
            lambda x: (x.rank().iloc[-1] - 1) / (len(x) - 1) * 2 - 1 if len(x) > 1 else 0
        )
    
    elif method == 'mad':
        median = factor.rolling(window=lookback, min_periods=min_periods).median()
        mad = factor.rolling(window=lookback, min_periods=min_periods).apply(
            lambda x: np.median(np.abs(x - np.median(x)))
        )
        return (factor - median) / (1.4826 * mad)
    
    else:
        return factor


def analyze_factor_distribution(factor: pd.Series, title: str = "Factor Distribution"):
    """分析因子分布特征"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=14)
    
    # 1. 时间序列
    ax = axes[0, 0]
    ax.plot(factor.index, factor.values)
    ax.set_title('Factor Time Series')
    ax.set_xlabel('Time')
    ax.set_ylabel('Factor Value')
    ax.grid(True)
    
    # 2. 分布直方图
    ax = axes[0, 1]
    ax.hist(factor.dropna(), bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(factor.mean(), color='r', linestyle='--', label=f'Mean: {factor.mean():.3f}')
    ax.axvline(factor.median(), color='g', linestyle='--', label=f'Median: {factor.median():.3f}')
    ax.set_title('Distribution')
    ax.set_xlabel('Factor Value')
    ax.set_ylabel('Frequency')
    ax.legend()
    
    # 3. Q-Q图（检验正态性）
    ax = axes[1, 0]
    from scipy import stats
    stats.probplot(factor.dropna(), dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (Normality Test)')
    
    # 4. 自相关
    ax = axes[1, 1]
    from pandas.plotting import autocorrelation_plot
    autocorrelation_plot(factor.dropna(), ax=ax)
    ax.set_title('Autocorrelation')
    
    plt.tight_layout()
    plt.show()
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"因子分布统计: {title}")
    print(f"{'='*60}")
    print(f"样本数: {factor.count()}")
    print(f"均值: {factor.mean():.4f}")
    print(f"标准差: {factor.std():.4f}")
    print(f"偏度: {factor.skew():.4f}")
    print(f"峰度: {factor.kurtosis():.4f}")
    print(f"最小值: {factor.min():.4f}")
    print(f"最大值: {factor.max():.4f}")
    print(f"中位数: {factor.median():.4f}")
    print(f"25%分位数: {factor.quantile(0.25):.4f}")
    print(f"75%分位数: {factor.quantile(0.75):.4f}")
    print(f"{'='*60}\n")


def analyze_factor_ic(factor: pd.Series, returns: pd.Series, 
                      title: str = "Factor IC Analysis"):
    """
    分析因子IC（信息系数）
    
    Args:
        factor: 因子值
        returns: 未来收益率
    """
    # 对齐数据
    aligned = pd.DataFrame({'factor': factor, 'returns': returns}).dropna()
    
    # 计算IC
    ic = aligned['factor'].corr(aligned['returns'])
    rank_ic = aligned['factor'].corr(aligned['returns'], method='spearman')
    
    # 滚动IC
    rolling_ic = aligned['factor'].rolling(window=63).corr(
        aligned['returns']
    )
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=14)
    
    # 1. 因子vs收益散点图
    ax = axes[0, 0]
    ax.scatter(aligned['factor'], aligned['returns'], alpha=0.3)
    ax.set_xlabel('Factor')
    ax.set_ylabel('Future Return')
    ax.set_title(f'Scatter Plot (IC={ic:.3f}, Rank IC={rank_ic:.3f})')
    ax.grid(True)
    
    # 2. 滚动IC
    ax = axes[0, 1]
    ax.plot(rolling_ic.index, rolling_ic.values)
    ax.axhline(0, color='r', linestyle='--')
    ax.axhline(rolling_ic.mean(), color='g', linestyle='--', 
               label=f'Mean IC: {rolling_ic.mean():.3f}')
    ax.set_title('Rolling IC (63 days)')
    ax.set_xlabel('Time')
    ax.set_ylabel('IC')
    ax.legend()
    ax.grid(True)
    
    # 3. IC分布
    ax = axes[1, 0]
    ax.hist(rolling_ic.dropna(), bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(rolling_ic.mean(), color='r', linestyle='--', 
               label=f'Mean: {rolling_ic.mean():.3f}')
    ax.set_title('IC Distribution')
    ax.set_xlabel('IC')
    ax.set_ylabel('Frequency')
    ax.legend()
    
    # 4. 分组收益
    ax = axes[1, 1]
    aligned['group'] = pd.qcut(aligned['factor'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    group_returns = aligned.groupby('group')['returns'].mean()
    group_returns.plot(kind='bar', ax=ax)
    ax.set_title('Returns by Factor Quintile')
    ax.set_xlabel('Quintile')
    ax.set_ylabel('Mean Return')
    ax.tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n{'='*60}")
    print(f"因子IC分析: {title}")
    print(f"{'='*60}")
    print(f"IC (Pearson): {ic:.4f}")
    print(f"Rank IC (Spearman): {rank_ic:.4f}")
    print(f"IC均值: {rolling_ic.mean():.4f}")
    print(f"IC标准差: {rolling_ic.std():.4f}")
    print(f"IR (IC/Std): {rolling_ic.mean() / rolling_ic.std():.4f}")
    print(f"IC>0比例: {(rolling_ic > 0).mean():.2%}")
    print(f"{'='*60}\n")
    
    return ic, rank_ic


def quick_backtest_single(factor: pd.Series, df: pd.DataFrame,
                          entry_threshold: float = 1.0,
                          exit_threshold: float = 0.0,
                          max_holding_days: int = 10) -> pd.DataFrame:
    """
    快速单品种回测
    
    Args:
        factor: 因子值序列
        df: 包含open, close, open_to_open的DataFrame
        entry_threshold: 开仓阈值
        exit_threshold: 平仓阈值
        max_holding_days: 最大持仓天数
    
    Returns:
        交易记录DataFrame
    """
    df = df.copy()
    df['factor'] = factor.values
    df = df.reset_index(drop=True)
    
    trades = []
    position = 0
    entry_idx = None
    entry_price = None
    holding_days = 0
    
    for i in range(len(df) - 2):
        current_factor = df.loc[i, 'factor']
        
        if pd.isna(current_factor):
            continue
        
        # 开仓
        if position == 0:
            if current_factor > entry_threshold:
                position = 1
                entry_idx = i
                entry_price = df.loc[i + 1, 'open']
                holding_days = 0
            elif current_factor < -entry_threshold:
                position = -1
                entry_idx = i
                entry_price = df.loc[i + 1, 'open']
                holding_days = 0
        
        # 持仓
        else:
            holding_days += 1
            exit_flag = False
            
            if abs(current_factor) <= exit_threshold:
                exit_flag = True
            if holding_days >= max_holding_days:
                exit_flag = True
            if i >= len(df) - 3:
                exit_flag = True
            
            if exit_flag:
                exit_price = df.loc[i + 1, 'open']
                
                if position == 1:
                    pnl = (exit_price - entry_price) / entry_price
                else:
                    pnl = (entry_price - exit_price) / entry_price
                
                trades.append({
                    'entry_date': df.loc[entry_idx, 'trade_date'],
                    'exit_date': df.loc[i, 'trade_date'],
                    'direction': 'long' if position == 1 else 'short',
                    'holding_days': holding_days,
                    'pnl': pnl - 0.0002,  # 扣除双边手续费
                    'entry_factor': df.loc[entry_idx, 'factor'],
                    'exit_factor': current_factor
                })
                
                position = 0
                entry_idx = None
                holding_days = 0
    
    return pd.DataFrame(trades)


def analyze_backtest_results(trades: pd.DataFrame, title: str = "Backtest Results"):
    """分析回测结果"""
    if len(trades) == 0:
        print("无交易记录")
        return
    
    # 计算指标
    total_trades = len(trades)
    win_rate = (trades['pnl'] > 0).mean()
    avg_return = trades['pnl'].mean()
    sharpe = trades['pnl'].mean() / trades['pnl'].std() * np.sqrt(252) if trades['pnl'].std() > 0 else 0
    
    # 权益曲线
    equity = (1 + trades['pnl']).cumprod()
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    max_dd = drawdown.min()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=14)
    
    # 1. 权益曲线
    ax = axes[0, 0]
    ax.plot(equity.values)
    ax.set_title('Equity Curve')
    ax.set_xlabel('Trade Number')
    ax.set_ylabel('Cumulative Return')
    ax.grid(True)
    
    # 2. 回撤
    ax = axes[0, 1]
    ax.fill_between(range(len(drawdown)), drawdown.values, 0, alpha=0.5)
    ax.set_title(f'Drawdown (Max: {max_dd:.2%})')
    ax.set_xlabel('Trade Number')
    ax.set_ylabel('Drawdown')
    ax.grid(True)
    
    # 3. 收益分布
    ax = axes[1, 0]
    ax.hist(trades['pnl'], bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(avg_return, color='r', linestyle='--', label=f'Mean: {avg_return:.4f}')
    ax.set_title('Return Distribution')
    ax.set_xlabel('Return')
    ax.set_ylabel('Frequency')
    ax.legend()
    
    # 4. 月度收益
    ax = axes[1, 1]
    trades['month'] = pd.to_datetime(trades['exit_date']).dt.to_period('M')
    monthly = trades.groupby('month')['pnl'].sum()
    monthly.plot(kind='bar', ax=ax)
    ax.set_title('Monthly Returns')
    ax.set_xlabel('Month')
    ax.set_ylabel('Return')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n{'='*60}")
    print(f"回测结果: {title}")
    print(f"{'='*60}")
    print(f"总交易次数: {total_trades}")
    print(f"胜率: {win_rate:.2%}")
    print(f"平均收益: {avg_return:.4f}")
    print(f"夏普比率: {sharpe:.2f}")
    print(f"最大回撤: {max_dd:.2%}")
    print(f"总收益: {equity.iloc[-1] - 1:.2%}")
    print(f"{'='*60}\n")


# 便捷函数：快速查看因子在单个品种上的表现
def quick_factor_analysis(symbol: str, factor_func, **kwargs):
    """
    快速分析因子在单个品种上的表现
    
    Args:
        symbol: 品种代码
        factor_func: 因子计算函数，接收df返回factor
        **kwargs: 传递给factor_func的参数
    """
    print(f"\n{'='*80}")
    print(f"快速因子分析: {symbol}")
    print(f"{'='*80}\n")
    
    # 加载数据
    df = load_symbol(symbol)
    print(f"数据范围: {df['trade_date'].min()} ~ {df['trade_date'].max()}")
    print(f"数据条数: {len(df)}\n")
    
    # 计算因子
    factor = factor_func(df, **kwargs)
    
    # 分析因子分布
    analyze_factor_distribution(factor, title=f"{symbol} Factor")
    
    # 分析因子IC（使用open_to_open作为未来收益）
    future_return = df['open_to_open'].shift(-1)
    ic, rank_ic = analyze_factor_ic(factor, future_return, title=f"{symbol} Factor IC")
    
    # 快速回测
    trades = quick_backtest_single(factor, df)
    analyze_backtest_results(trades, title=f"{symbol} Backtest")
    
    return factor, trades


def run_backtest_by_category(factor_func, factor_kwargs: dict = None,
                              entry_threshold: float = 1.0,
                              exit_threshold: float = 0.0,
                              max_holding_days: int = 10) -> Dict[str, pd.DataFrame]:
    """
    按行业分类运行回测
    
    Args:
        factor_func: 因子计算函数
        factor_kwargs: 传递给因子函数的参数
        entry_threshold: 开仓阈值
        exit_threshold: 平仓阈值
        max_holding_days: 最大持仓天数
    
    Returns:
        各行业交易记录 {行业名: DataFrame}
    """
    if factor_kwargs is None:
        factor_kwargs = {}
    
    category_results = {}
    
    for category in get_all_categories():
        print(f"\n{'='*60}")
        print(f"回测行业: {category}")
        print(f"{'='*60}")
        
        symbols_data = load_symbols_by_category(category)
        if not symbols_data:
            print(f"  {category}: 无数据")
            continue
        
        all_trades = []
        for symbol, df in symbols_data.items():
            try:
                factor = factor_func(df, **factor_kwargs)
                trades = quick_backtest_single(
                    factor, df,
                    entry_threshold=entry_threshold,
                    exit_threshold=exit_threshold,
                    max_holding_days=max_holding_days
                )
                if len(trades) > 0:
                    trades['symbol'] = symbol
                    all_trades.append(trades)
            except Exception as e:
                print(f"  Error in {symbol}: {e}")
        
        if all_trades:
            category_results[category] = pd.concat(all_trades, ignore_index=True)
            print(f"  {category}: {len(category_results[category])} 笔交易")
        else:
            print(f"  {category}: 无交易")
    
    return category_results


def plot_category_equity_curves(category_results: Dict[str, pd.DataFrame],
                                 title: str = "行业回测权益曲线"):
    """
    绘制各行业权益曲线对比
    
    图表解读要点：
    1. 曲线斜率：斜率越大说明收益越高，但要注意是否平稳
    2. 曲线平滑度：波动小的曲线说明策略在该行业更稳定
    3. 曲线相关性：如果多条曲线同涨同跌，说明行业间相关性高
    4. 最大回撤：曲线从高点回落的幅度，反映风险
    5. 时间分布：某些行业可能在特定时期表现更好
    
    Args:
        category_results: {行业名: 交易记录DataFrame}
        title: 图表标题
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # 准备数据：按日期聚合各行业收益
    category_daily_returns = {}
    all_dates = set()
    
    for category, trades in category_results.items():
        if len(trades) == 0:
            continue
        
        # 按退出日期聚合日收益
        trades['exit_date'] = pd.to_datetime(trades['exit_date'])
        daily = trades.groupby('exit_date')['pnl'].sum()
        category_daily_returns[category] = daily
        all_dates.update(daily.index)
    
    if not category_daily_returns:
        print("无数据可绘制")
        return
    
    # 创建统一日期索引
    date_range = pd.date_range(start=min(all_dates), end=max(all_dates), freq='D')
    
    # 图1：各行业权益曲线（绝对收益）
    ax1 = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(category_daily_returns)))
    
    category_stats = []
    for i, (category, daily) in enumerate(category_daily_returns.items()):
        # 补齐缺失日期
        daily_aligned = daily.reindex(date_range, fill_value=0)
        # 计算权益曲线
        equity = (1 + daily_aligned).cumprod()
        
        ax1.plot(equity.index, equity.values, label=category, 
                color=colors[i], linewidth=1.5, alpha=0.8)
        
        # 统计信息
        total_return = equity.iloc[-1] - 1
        sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
        max_dd = (equity / equity.cummax() - 1).min()
        category_stats.append({
            'category': category,
            'total_return': total_return,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'trades': len(category_results[category])
        })
    
    ax1.axhline(1, color='black', linestyle='--', alpha=0.3)
    ax1.set_title('各行业权益曲线（绝对收益）', fontsize=12)
    ax1.set_xlabel('日期')
    ax1.set_ylabel('累计净值')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 图2：各行业相对强弱（归一化到起点）
    ax2 = axes[1]
    
    for i, (category, daily) in enumerate(category_daily_returns.items()):
        daily_aligned = daily.reindex(date_range, fill_value=0)
        # 计算超额收益（相对于等权组合）
        equity = (1 + daily_aligned).cumprod()
        # 归一化：从1开始
        normalized = equity / equity.iloc[0]
        ax2.plot(normalized.index, normalized.values, label=category,
                color=colors[i], linewidth=1.5, alpha=0.8)
    
    ax2.axhline(1, color='black', linestyle='--', alpha=0.3)
    ax2.set_title('各行业归一化权益曲线（便于比较走势）', fontsize=12)
    ax2.set_xlabel('日期')
    ax2.set_ylabel('归一化净值')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 打印行业统计对比
    stats_df = pd.DataFrame(category_stats)
    stats_df = stats_df.sort_values('sharpe', ascending=False)
    
    print(f"\n{'='*80}")
    print("各行业回测绩效对比")
    print(f"{'='*80}")
    print(stats_df.to_string(index=False))
    print(f"{'='*80}\n")
    
    # 绘制行业绩效对比图
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    fig2.suptitle('各行业绩效指标对比', fontsize=12, fontweight='bold')
    
    # 总收益
    axes2[0].barh(stats_df['category'], stats_df['total_return'])
    axes2[0].set_xlabel('总收益')
    axes2[0].set_title('总收益对比')
    axes2[0].axvline(0, color='red', linestyle='--', alpha=0.5)
    
    # 夏普比率
    axes2[1].barh(stats_df['category'], stats_df['sharpe'])
    axes2[1].set_xlabel('夏普比率')
    axes2[1].set_title('夏普比率对比')
    axes2[1].axvline(0, color='red', linestyle='--', alpha=0.5)
    
    # 最大回撤
    axes2[2].barh(stats_df['category'], stats_df['max_dd'])
    axes2[2].set_xlabel('最大回撤')
    axes2[2].set_title('最大回撤对比')
    
    plt.tight_layout()
    plt.show()
    
    return stats_df


def analyze_category_characteristics(category_results: Dict[str, pd.DataFrame]):
    """
    分析各行业特征，帮助理解因子在不同行业的表现差异
    
    解读要点：
    1. 胜率差异：某些行业可能更适合做多/做空
    2. 持仓周期：反映行业的趋势持续性
    3. 收益分布：了解行业的盈亏特征
    4. 长短期表现：某些行业可能在特定周期表现更好
    """
    print(f"\n{'='*80}")
    print("行业特征分析")
    print(f"{'='*80}\n")
    
    for category, trades in category_results.items():
        if len(trades) == 0:
            continue
        
        print(f"\n【{category}】")
        print("-" * 60)
        
        # 基础统计
        print(f"总交易次数: {len(trades)}")
        print(f"做多次数: {(trades['direction'] == 'long').sum()}")
        print(f"做空次数: {(trades['direction'] == 'short').sum()}")
        print(f"平均持仓天数: {trades['holding_days'].mean():.1f}")
        
        # 胜率分析
        long_trades = trades[trades['direction'] == 'long']
        short_trades = trades[trades['direction'] == 'short']
        
        if len(long_trades) > 0:
            print(f"做多胜率: {(long_trades['pnl'] > 0).mean():.2%}")
        if len(short_trades) > 0:
            print(f"做空胜率: {(short_trades['pnl'] > 0).mean():.2%}")
        
        # 收益分位数
        print(f"收益25%分位: {trades['pnl'].quantile(0.25):.4f}")
        print(f"收益中位数: {trades['pnl'].median():.4f}")
        print(f"收益75%分位: {trades['pnl'].quantile(0.75):.4f}")
        
        # 盈亏比
        avg_win = trades[trades['pnl'] > 0]['pnl'].mean() if (trades['pnl'] > 0).any() else 0
        avg_loss = abs(trades[trades['pnl'] < 0]['pnl'].mean()) if (trades['pnl'] < 0).any() else 1
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        print(f"盈亏比: {profit_loss_ratio:.2f}")
