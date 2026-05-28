#!/usr/bin/env python3
"""
单品种时序因子回测框架
- 因子计算接口（用户可自定义因子）
- 时序阈值开仓/平仓逻辑
- 回测：第二天open买，第三天open卖
- 因子分析和绩效报告
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 路径配置
DATA_DIR = Path('/mnt/Data/writable/liaoyuyang/data/1day/active')
OUTPUT_DIR = Path('/home/day_strategy/prepare_data/backtest_results')
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class BacktestConfig:
    """回测配置"""
    entry_threshold: float = 1.0      # 开仓阈值（因子值大于此阈值做多，小于负阈值做空）
    exit_threshold: float = 0.0       # 平仓阈值（因子值回归此范围平仓）
    max_holding_days: int = 20        # 最大持仓天数
    position_size: float = 1.0        # 仓位大小（1表示满仓）
    commission_rate: float = 0.0001   # 手续费率（万分之一）
    

class FactorCalculator:
    """
    因子计算基类
    用户需要继承此类并实现calculate_factor方法
    """
    
    def __init__(self, name: str):
        self.name = name
    
    def calculate_factor(self, df: pd.DataFrame) -> pd.Series:
        """
        计算因子值
        
        Args:
            df: 包含日线数据的DataFrame
            
        Returns:
            因子值的Series，与df等长
        """
        raise NotImplementedError("请子类实现此方法")
    
    def normalize_factor(self, factor: pd.Series, method: str = 'zscore') -> pd.Series:
        """
        因子标准化，使分布稳定
        
        Args:
            factor: 原始因子值
            method: 标准化方法 ('zscore', 'rank', 'mad')
        """
        if method == 'zscore':
            # Z-score标准化
            mean = factor.rolling(window=252, min_periods=60).mean()
            std = factor.rolling(window=252, min_periods=60).std()
            return (factor - mean) / std
        
        elif method == 'rank':
            # 排序标准化到[-1, 1]
            rank = factor.rolling(window=252, min_periods=60).apply(
                lambda x: (x.rank().iloc[-1] - 1) / (len(x) - 1) * 2 - 1 if len(x) > 1 else 0
            )
            return rank
        
        elif method == 'mad':
            # MAD标准化（中位数绝对偏差）
            median = factor.rolling(window=252, min_periods=60).median()
            mad = factor.rolling(window=252, min_periods=60).apply(
                lambda x: np.median(np.abs(x - np.median(x)))
            )
            return (factor - median) / (1.4826 * mad)
        
        else:
            return factor


class BacktestEngine:
    """回测引擎"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.results = {}
    
    def run_backtest(self, symbol: str, df: pd.DataFrame, 
                     factor: pd.Series, factor_name: str) -> Dict:
        """
        运行单品种回测
        
        交易逻辑：
        - 信号产生日（T）：根据因子值判断
        - 开仓日（T+1）：以open价格买入
        - 平仓日（T+2或之后）：以open价格卖出
        
        Args:
            symbol: 品种代码
            df: 日线数据
            factor: 因子值
            factor_name: 因子名称
            
        Returns:
            回测结果字典
        """
        df = df.copy()
        df['factor'] = factor
        df = df.reset_index(drop=True)
        
        # 获取open_to_open收益率（即T+1到T+2的收益率）
        df['next_open_return'] = df['open_to_open'].shift(-1)
        
        trades = []
        position = 0  # 0: 无持仓, 1: 多头, -1: 空头
        entry_idx = None
        entry_price = None
        holding_days = 0
        
        for i in range(len(df) - 2):  # 留出最后两天用于平仓
            current_factor = df.loc[i, 'factor']
            
            # 跳过NaN因子值
            if pd.isna(current_factor):
                continue
            
            # 开仓逻辑
            if position == 0:
                if current_factor > self.config.entry_threshold:
                    # 开多仓
                    position = 1
                    entry_idx = i
                    entry_price = df.loc[i + 1, 'open']  # T+1开盘价买入
                    holding_days = 0
                    
                elif current_factor < -self.config.entry_threshold:
                    # 开空仓
                    position = -1
                    entry_idx = i
                    entry_price = df.loc[i + 1, 'open']  # T+1开盘价卖出（做空）
                    holding_days = 0
            
            # 持仓逻辑
            else:
                holding_days += 1
                exit_flag = False
                
                # 平仓条件1：因子值回归
                if abs(current_factor) <= self.config.exit_threshold:
                    exit_flag = True
                
                # 平仓条件2：超过最大持仓天数
                if holding_days >= self.config.max_holding_days:
                    exit_flag = True
                
                # 平仓条件3：最后数据点
                if i >= len(df) - 3:
                    exit_flag = True
                
                if exit_flag:
                    exit_price = df.loc[i + 1, 'open']  # T+1开盘价平仓
                    
                    # 计算收益
                    if position == 1:
                        pnl = (exit_price - entry_price) / entry_price
                    else:
                        pnl = (entry_price - exit_price) / entry_price
                    
                    # 扣除手续费（双边）
                    pnl -= 2 * self.config.commission_rate
                    
                    trades.append({
                        'entry_date': df.loc[entry_idx, 'trade_date'],
                        'exit_date': df.loc[i, 'trade_date'],
                        'direction': 'long' if position == 1 else 'short',
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'holding_days': holding_days,
                        'pnl': pnl,
                        'entry_factor': df.loc[entry_idx, 'factor'],
                        'exit_factor': current_factor
                    })
                    
                    position = 0
                    entry_idx = None
                    holding_days = 0
        
        # 计算绩效指标
        results = self._calculate_metrics(trades, df, factor_name)
        results['symbol'] = symbol
        results['trades'] = trades
        
        return results
    
    def _calculate_metrics(self, trades: List[Dict], df: pd.DataFrame, 
                          factor_name: str) -> Dict:
        """计算绩效指标"""
        if not trades:
            return {
                'factor_name': factor_name,
                'total_trades': 0,
                'win_rate': 0,
                'avg_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'annual_return': 0,
                'profit_factor': 0
            }
        
        trades_df = pd.DataFrame(trades)
        
        # 基础统计
        total_trades = len(trades_df)
        win_trades = len(trades_df[trades_df['pnl'] > 0])
        win_rate = win_trades / total_trades if total_trades > 0 else 0
        avg_return = trades_df['pnl'].mean()
        
        # 构建权益曲线
        equity_curve = (1 + trades_df['pnl']).cumprod()
        
        # 夏普比率（简化版，假设交易均匀分布）
        returns = trades_df['pnl']
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # 最大回撤
        cummax = equity_curve.cummax()
        drawdown = (equity_curve - cummax) / cummax
        max_drawdown = drawdown.min()
        
        # 年化收益（假设每年252个交易日，平均持仓天数计算）
        avg_holding = trades_df['holding_days'].mean()
        if avg_holding > 0:
            trades_per_year = 252 / avg_holding
            annual_return = (1 + avg_return) ** trades_per_year - 1
        else:
            annual_return = 0
        
        # 盈亏比
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if win_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if win_trades < total_trades else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        return {
            'factor_name': factor_name,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'annual_return': annual_return,
            'profit_factor': profit_factor,
            'equity_curve': equity_curve,
            'trades_df': trades_df
        }


class FactorAnalyzer:
    """因子分析器"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
    
    def analyze_factor(self, all_results: Dict[str, Dict], factor_name: str):
        """分析因子在所有品种上的表现"""
        
        # 汇总所有品种的绩效
        summary = []
        for symbol, results in all_results.items():
            summary.append({
                'symbol': symbol,
                'total_trades': results['total_trades'],
                'win_rate': results['win_rate'],
                'avg_return': results['avg_return'],
                'sharpe_ratio': results['sharpe_ratio'],
                'max_drawdown': results['max_drawdown'],
                'annual_return': results['annual_return'],
                'profit_factor': results['profit_factor']
            })
        
        summary_df = pd.DataFrame(summary)
        
        # 保存汇总结果
        summary_df.to_csv(self.output_dir / f'{factor_name}_summary.csv', index=False)
        
        # 生成分析报告
        self._generate_report(summary_df, all_results, factor_name)
        
        return summary_df
    
    def _generate_report(self, summary_df: pd.DataFrame, all_results: Dict, factor_name: str):
        """生成分析报告和图表"""
        
        # 过滤inf和nan值用于统计
        summary_clean = summary_df.replace([np.inf, -np.inf], np.nan)
        
        # 1. 绩效汇总表
        print(f"\n{'='*80}")
        print(f"因子回测报告: {factor_name}")
        print(f"{'='*80}")
        print(f"\n品种数量: {len(summary_df)}")
        print(f"平均交易次数: {summary_clean['total_trades'].mean():.1f}")
        print(f"平均胜率: {summary_clean['win_rate'].mean():.2%}")
        print(f"平均收益: {summary_clean['avg_return'].mean():.4f}")
        print(f"平均夏普比率: {summary_clean['sharpe_ratio'].mean():.2f}")
        print(f"平均最大回撤: {summary_clean['max_drawdown'].mean():.2%}")
        print(f"平均年化收益: {summary_clean['annual_return'].mean():.2%}")
        
        # 2. 按品种排序
        print(f"\n按夏普比率排序（前10）:")
        top_sharpe = summary_df.nlargest(10, 'sharpe_ratio')[['symbol', 'sharpe_ratio', 'win_rate', 'avg_return']]
        print(top_sharpe.to_string(index=False))
        
        # 3. 绘制图表
        self._plot_results(summary_df, all_results, factor_name)
    
    def _plot_results(self, summary_df: pd.DataFrame, all_results: Dict, factor_name: str):
        """绘制结果图表"""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Factor Backtest Results: {factor_name}', fontsize=14)
        
        # 1. 夏普比率分布
        ax = axes[0, 0]
        ax.hist(summary_df['sharpe_ratio'], bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(summary_df['sharpe_ratio'].mean(), color='r', linestyle='--', 
                   label=f'Mean: {summary_df["sharpe_ratio"].mean():.2f}')
        ax.set_xlabel('Sharpe Ratio')
        ax.set_ylabel('Frequency')
        ax.set_title('Sharpe Ratio Distribution')
        ax.legend()
        
        # 2. 胜率分布
        ax = axes[0, 1]
        ax.hist(summary_df['win_rate'], bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(summary_df['win_rate'].mean(), color='r', linestyle='--',
                   label=f'Mean: {summary_df["win_rate"].mean():.2%}')
        ax.set_xlabel('Win Rate')
        ax.set_ylabel('Frequency')
        ax.set_title('Win Rate Distribution')
        ax.legend()
        
        # 3. 平均收益分布
        ax = axes[0, 2]
        avg_return_clean = summary_df['avg_return'].replace([np.inf, -np.inf], np.nan).dropna()
        ax.hist(avg_return_clean, bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(avg_return_clean.mean(), color='r', linestyle='--',
                   label=f'Mean: {avg_return_clean.mean():.4f}')
        ax.set_xlabel('Average Return')
        ax.set_ylabel('Frequency')
        ax.set_title('Average Return Distribution')
        ax.legend()
        
        # 4. 夏普比率 vs 胜率
        ax = axes[1, 0]
        ax.scatter(summary_df['win_rate'], summary_df['sharpe_ratio'], alpha=0.6)
        ax.set_xlabel('Win Rate')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title('Sharpe Ratio vs Win Rate')
        
        # 5. 综合权益曲线（等权组合）
        ax = axes[1, 1]
        combined_equity = self._calculate_combined_equity(all_results)
        if combined_equity is not None:
            ax.plot(combined_equity.index, combined_equity.values)
            ax.set_xlabel('Trade Number')
            ax.set_ylabel('Cumulative Return')
            ax.set_title('Combined Equity Curve (Equal Weight)')
            ax.grid(True)
        
        # 6. 年化收益 vs 最大回撤
        ax = axes[1, 2]
        ax.scatter(summary_df['max_drawdown'], summary_df['annual_return'], alpha=0.6)
        ax.set_xlabel('Max Drawdown')
        ax.set_ylabel('Annual Return')
        ax.set_title('Annual Return vs Max Drawdown')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{factor_name}_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n图表已保存至: {self.output_dir / f'{factor_name}_analysis.png'}")
    
    def _calculate_combined_equity(self, all_results: Dict) -> Optional[pd.Series]:
        """计算等权组合的权益曲线"""
        
        # 收集所有交易
        all_trades = []
        for symbol, results in all_results.items():
            if 'trades_df' in results and len(results['trades_df']) > 0:
                trades = results['trades_df'].copy()
                trades['symbol'] = symbol
                all_trades.append(trades)
        
        if not all_trades:
            return None
        
        # 合并所有交易并按时间排序
        combined = pd.concat(all_trades, ignore_index=True)
        combined['exit_date'] = pd.to_datetime(combined['exit_date'])
        combined = combined.sort_values('exit_date')
        
        # 计算等权组合的收益率（假设每个交易等权）
        combined['portfolio_return'] = combined.groupby('exit_date')['pnl'].transform('mean')
        
        # 去重并计算累计收益
        daily_returns = combined.drop_duplicates('exit_date')[['exit_date', 'portfolio_return']]
        daily_returns = daily_returns.set_index('exit_date')['portfolio_return']
        
        equity_curve = (1 + daily_returns).cumprod()
        
        return equity_curve


# ==================== 用户自定义因子区域 ====================

class ExampleMomentumFactor(FactorCalculator):
    """
    示例因子：简单动量因子
    用户可以参考此格式实现自己的因子
    """
    
    def __init__(self, lookback: int = 20):
        super().__init__(name=f"Momentum_{lookback}")
        self.lookback = lookback
    
    def calculate_factor(self, df: pd.DataFrame) -> pd.Series:
        """
        计算动量因子：过去N天的收益率
        """
        # 计算动量
        momentum = df['close'].pct_change(self.lookback)
        
        # 标准化（使分布稳定）
        factor = self.normalize_factor(momentum, method='zscore')
        
        return factor


class ExampleMeanReversionFactor(FactorCalculator):
    """
    示例因子：均值回复因子
    """
    
    def __init__(self, lookback: int = 20):
        super().__init__(name=f"MeanReversion_{lookback}")
        self.lookback = lookback
    
    def calculate_factor(self, df: pd.DataFrame) -> pd.Series:
        """
        计算均值回复因子：价格偏离均线的程度
        """
        # 计算均线
        ma = df['close'].rolling(window=self.lookback).mean()
        
        # 计算偏离度
        deviation = (df['close'] - ma) / ma
        
        # 标准化（取负号，因为均值回复）
        factor = -self.normalize_factor(deviation, method='zscore')
        
        return factor


class ExampleVolatilityFactor(FactorCalculator):
    """
    示例因子：波动率因子
    """
    
    def __init__(self, lookback: int = 20):
        super().__init__(name=f"Volatility_{lookback}")
        self.lookback = lookback
    
    def calculate_factor(self, df: pd.DataFrame) -> pd.Series:
        """
        计算波动率因子：过去N天的收益率波动率
        """
        # 计算日收益率
        returns = df['close_to_close']
        
        # 计算波动率
        volatility = returns.rolling(window=self.lookback).std()
        
        # 标准化
        factor = self.normalize_factor(volatility, method='zscore')
        
        return factor


# ==================== 主程序 ====================

def load_all_symbols():
    """加载所有品种数据"""
    symbols_data = {}
    
    for file in DATA_DIR.glob('*_active.feather'):
        symbol = file.stem.replace('_active', '')
        df = pd.read_feather(file)
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        symbols_data[symbol] = df
    
    return symbols_data


def run_factor_backtest(factor_calculator: FactorCalculator, 
                        config: BacktestConfig = None):
    """
    运行因子回测
    
    Args:
        factor_calculator: 因子计算器实例
        config: 回测配置，使用默认配置如果为None
    """
    if config is None:
        config = BacktestConfig()
    
    print(f"\n运行因子回测: {factor_calculator.name}")
    print(f"开仓阈值: {config.entry_threshold}")
    print(f"平仓阈值: {config.exit_threshold}")
    print(f"最大持仓天数: {config.max_holding_days}")
    
    # 加载数据
    symbols_data = load_all_symbols()
    print(f"加载了 {len(symbols_data)} 个品种的数据")
    
    # 初始化回测引擎
    engine = BacktestEngine(config)
    
    # 对每个品种运行回测
    all_results = {}
    for symbol, df in symbols_data.items():
        # 计算因子
        factor = factor_calculator.calculate_factor(df)
        
        # 运行回测
        results = engine.run_backtest(symbol, df, factor, factor_calculator.name)
        all_results[symbol] = results
    
    # 分析因子
    analyzer = FactorAnalyzer(OUTPUT_DIR)
    summary = analyzer.analyze_factor(all_results, factor_calculator.name)
    
    return all_results, summary


def main():
    """主函数 - 示例用法"""
    
    # 配置回测参数
    config = BacktestConfig(
        entry_threshold=1.0,    # 因子值>1做多，<-1做空
        exit_threshold=0.0,     # 因子值回归0附近平仓
        max_holding_days=10,    # 最多持仓10天
        commission_rate=0.0001  # 万分之一手续费
    )
    
    # 示例1：动量因子
    print("\n" + "="*80)
    print("示例1: 动量因子 (20日)")
    print("="*80)
    momentum_factor = ExampleMomentumFactor(lookback=20)
    results1, summary1 = run_factor_backtest(momentum_factor, config)
    
    # 示例2：均值回复因子
    print("\n" + "="*80)
    print("示例2: 均值回复因子 (20日)")
    print("="*80)
    meanrev_factor = ExampleMeanReversionFactor(lookback=20)
    results2, summary2 = run_factor_backtest(meanrev_factor, config)
    
    # 示例3：波动率因子
    print("\n" + "="*80)
    print("示例3: 波动率因子 (20日)")
    print("="*80)
    vol_factor = ExampleVolatilityFactor(lookback=20)
    results3, summary3 = run_factor_backtest(vol_factor, config)
    
    print(f"\n\n所有结果已保存至: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
