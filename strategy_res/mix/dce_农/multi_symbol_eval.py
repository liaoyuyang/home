#!/usr/bin/env python3
"""
multi_symbol_eval.py
多品种多版本回测绩效对比脚本

支持：
- 批量回测多个品种 × 多个 train_end_date × 多个 version
- 统一回测窗口（bt_start ~ bt_end），对比不同训练截止日期在同一时段的表现
- 汇总绩效指标表格（年化收益、夏普、最大回撤、胜率、交易次数等）
- 多品种并排对比图（资金曲线、多空拆分、指标柱状图、等权合成曲线）

用法示例：
    # Python API
    from multi_symbol_eval import MultiSymbolEvaluator
    ev = MultiSymbolEvaluator(
        model_base_dir='./model',
        bt_start='2025-07-01',
        bt_end='2026-01-01',
        bt_params={'th1': 0.9, 'th2': 0.5, 'holding_bars': 10, 'day': 1725, 'fee': 0, 'v': 2}
    )
    ev.run_batch(
        symbols=['A', 'B', 'C', 'CS', 'M', 'Y', 'P', 'LH'],
        train_end_dates=['2025-07-01', '2025-10-01'],
        versions=['v0'],
        train_label=5
    )
    summary = ev.get_summary_df()
    ev.plot_equity_curves()
    ev.plot_metrics_bar()

    # 命令行
    python multi_symbol_eval.py \
        --model-dir ./model \
        --symbols A B C CS M Y P LH \
        --train-end-dates 2025-07-01 2025-10-01 \
        --versions v0 \
        --bt-start 2025-07-01 --bt-end 2026-01-01 \
        --output-dir ./eval_output
"""

import sys
import argparse
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')
sys.path.append('/home/future_commodity')

import function_future.backtest_v3 as bv
import function_future.DataLoader as DL
from function_future.margin_calculator import calculate_margin

# ------------------------------------------------------------------
# 辅助函数
# ------------------------------------------------------------------

def _calc_dd(cum_returns: pd.Series) -> pd.Series:
    """计算回撤序列（百分比）"""
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max * 100
    return drawdown


def _extract_daily(merged_data: pd.DataFrame) -> pd.DataFrame:
    """从 merged_data 提取日级别汇总"""
    df = merged_data.copy()
    if 'datetime' in df.columns:
        df = df.set_index('datetime')
    daily = df.groupby('date').agg({
        'pnl_ret': 'sum',
        'pos': 'mean',
        'long_pos': 'max',
        'short_pos': 'max',
        'cost&slippage_rate': 'sum',
    }).reset_index()
    daily['date'] = pd.to_datetime(daily['date'])
    return daily


# ------------------------------------------------------------------
# 核心类
# ------------------------------------------------------------------

class MultiSymbolEvaluator:
    """
    多品种多版本回测绩效对比器
    """

    def __init__(
        self,
        model_base_dir: str,
        bt_start: str,
        bt_end: str,
        bt_params: Dict,
        output_dir: Optional[str] = None,
        money: Optional[Union[int, float, Dict[str, float], str]] = None,
    ):
        """
        Parameters
        ----------
        model_base_dir : str
            模型根目录，子文件夹名为 {SYMBOL}_pred{LABEL}_{DATE}_{VERSION}
        bt_start, bt_end : str
            统一回测起止日期（YYYY-MM-DD）
        bt_params : dict
            回测参数，如 {'th1': 0.9, 'th2': 0.5, 'holding_bars': 10, 'day': 1725, 'fee': 0, 'v': 2}
        output_dir : str, optional
            图表/表格保存目录，默认不保存只展示
        money : int/float/Dict[str,float]/str, optional
            本金设置。默认 None 使用 backtest_v3 默认值（1000万）。
            - int/float: 统一设置所有品种本金
            - Dict[str, float]: 按品种分别设置本金，如 {'C': 42105, 'LH': 456160}
            - 'auto': 自动按 margin/0.4 计算（10手保证金占本金40%）
        """
        self.model_base_dir = Path(model_base_dir)
        self.bt_start = bt_start
        self.bt_end = bt_end
        self.bt_params = bt_params.copy()
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.money = money
        self._money_cache: Dict[str, float] = {}

        self.results: List[Dict] = []          # 汇总指标列表
        self.merged_data_dict: Dict[Tuple[str, str, str], pd.DataFrame] = {}   # (sym, date, ver) -> merged
        self.daily_dict: Dict[Tuple[str, str, str], pd.DataFrame] = {}         # 日级别数据缓存

    def _resolve_money(self, symbol: str) -> Optional[float]:
        """解析该品种的本金设置"""
        if self.money is None:
            return None
        if isinstance(self.money, (int, float)):
            return float(self.money)
        if isinstance(self.money, dict):
            return self.money.get(symbol)
        if self.money == 'auto':
            if symbol not in self._money_cache:
                margin = calculate_margin(symbol, lots=10)
                self._money_cache[symbol] = margin / 0.4
            return self._money_cache[symbol]
        raise ValueError(f"不支持的 money 类型: {type(self.money)} = {self.money}")

    # ------------------------------------------------------------------
    # 单条回测
    # ------------------------------------------------------------------

    def run_single(
        self,
        symbol: str,
        train_end_date: str,
        version: str,
        train_label: int = 5,
        verbose: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        跑单个品种单个版本的回测，返回 merged_data
        """
        folder_name = f"{symbol}_pred{train_label}_{train_end_date}_{version}"
        model_dir = self.model_base_dir / folder_name

        if not model_dir.exists():
            if verbose:
                print(f"  [跳过] 模型目录不存在: {model_dir}")
            return None

        if verbose:
            print(f"  回测: {symbol} | train_end={train_end_date} | version={version}")

        # 加载回测配置
        config = bv.load_config(r"/mnt/Data/writable/liaoyuyang/backtest/backtest_config.json")
        config['MODEL_DIR'] = model_dir

        # 初始化回测器
        bt = bv.ModelBacktester(train_end_date, config)
        bt.window_end = self.bt_end

        # 加载品种配置（交易时间、保证金率等）
        bt.load_config(symbol)

        # 加载因子和市场数据（限定统一回测窗口）
        bt.load_factor(symbol, start_date=self.bt_start, end_date=self.bt_end)
        bt.load_mktdata(symbol, start_date=self.bt_start, end_date=self.bt_end)

        # 加载模型、生成预测、组合预测
        bt.load_models()
        bt.generate_predictions()
        bt.combine_models('best_iteration_log_weighted', avg=True)

        # 设置本金（支持统一、按品种、auto 动态计算）
        money_val = self._resolve_money(symbol)
        if money_val is not None:
            bt.money = money_val
            if verbose:
                print(f"    [{symbol}] money={bt.money:,.0f}")

        # 设置手续费（通过属性而非参数）
        if 'fee' in self.bt_params:
            bt.fee = self.bt_params['fee']

        # day 参数：支持 holding_days 自动换算，或直接传 day
        if 'day' in self.bt_params:
            day_val = self.bt_params['day']
        elif 'holding_days' in self.bt_params:
            # 从品种配置读取 bars_per_day
            _cfg_loader = DL.InstrumentConfig()
            inst_cfg = _cfg_loader.get_instrument_config(symbol)
            bars_per_day = inst_cfg.get('bars_per_day', 345)
            day_val = self.bt_params['holding_days'] * bars_per_day
            if verbose:
                print(f"    [{symbol}] holding_days={self.bt_params['holding_days']} × bars_per_day={bars_per_day} → day={day_val}")
        else:
            day_val = 1725  # 默认 fallback

        # 回测（并行模式下静默内部 tqdm + print）
        if verbose:
            merged_data = bt.backtest(
                th1=self.bt_params.get('th1', 0.9),
                th2=self.bt_params.get('th2', 0.5),
                holding_bars=self.bt_params.get('holding_bars', 10),
                day=day_val,
                v=self.bt_params.get('v', 2),
                model_name='best_iteration_log_weighted',
            )
        else:
            import io, contextlib

            class _DummyTqdm:
                """哑 tqdm：屏蔽所有进度条输出"""
                def __init__(self, iterable=None, *args, **kwargs):
                    self.iterable = iterable
                def __iter__(self):
                    return iter(self.iterable) if self.iterable is not None else iter([])
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
                def update(self, n=1):
                    pass
                def close(self):
                    pass
                def set_postfix(self, *args, **kwargs):
                    pass

            _orig_tqdm = bv.tqdm
            bv.tqdm = _DummyTqdm
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    merged_data = bt.backtest(
                        th1=self.bt_params.get('th1', 0.9),
                        th2=self.bt_params.get('th2', 0.5),
                        holding_bars=self.bt_params.get('holding_bars', 10),
                        day=day_val,
                        v=self.bt_params.get('v', 2),
                        model_name='best_iteration_log_weighted',
                    )
            finally:
                bv.tqdm = _orig_tqdm

        # 存储结果
        key = (symbol, train_end_date, version)
        self.merged_data_dict[key] = merged_data
        self.daily_dict[key] = _extract_daily(merged_data)

        # 提取指标
        metrics = self._extract_metrics(symbol, train_end_date, version, merged_data)
        self.results.append(metrics)
        if verbose:
            print(f"    → 年化收益: {metrics['annual_ret']:.2%}, 夏普: {metrics['sharpe']:.2f}, 最大回撤: {metrics['max_dd']:.2%}")
        return merged_data

    # ------------------------------------------------------------------
    # 批量回测
    # ------------------------------------------------------------------

    def _run_single_safe(
        self, symbol: str, train_end_date: str, version: str, train_label: int,
        verbose: bool = True,
    ) -> bool:
        """带异常捕获的 run_single 包装，供并行调用"""
        try:
            self.run_single(symbol, train_end_date, version, train_label, verbose=verbose)
            return True
        except Exception as e:
            if verbose:
                print(f"  ✗ 失败 [{symbol} {train_end_date} {version}]: {e}")
                import traceback
                traceback.print_exc()
            return False

    def run_batch(
        self,
        symbols: List[str],
        train_end_dates: List[str],
        versions: List[str],
        train_label: int = 5,
        n_jobs: int = 8,
    ):
        """
        批量运行：品种列表 × train_end_date 列表 × version 列表

        Parameters
        ----------
        n_jobs : int
            并行进程/线程数。默认 1（顺序执行），-1 表示用满所有核心。
            建议用 n_jobs=-1 或 n_jobs=len(symbols)。
        """
        from joblib import Parallel, delayed

        tasks = [
            (symbol, train_end_date, version)
            for symbol in symbols
            for train_end_date in train_end_dates
            for version in versions
        ]
        total = len(tasks)

        if n_jobs == 1:
            # 顺序执行
            for idx, (symbol, train_end_date, version) in enumerate(tasks, 1):
                print(f"\n[{idx}/{total}] {symbol} | {train_end_date} | {version}")
                self._run_single_safe(symbol, train_end_date, version, train_label, verbose=True)
        else:
            # 多线程并行（backend='threading'，共享实例状态无需序列化）
            from tqdm import tqdm
            print(f"并行回测: {total} 组任务, n_jobs={n_jobs}")
            with tqdm(total=total, desc="回测进度") as pbar:
                def _wrapper(args):
                    symbol, train_end_date, version = args
                    result = self._run_single_safe(symbol, train_end_date, version, train_label, verbose=False)
                    pbar.update(1)
                    return result
                Parallel(n_jobs=n_jobs, backend='threading')(
                    delayed(_wrapper)(task) for task in tasks
                )

        if self.results:
            print(f"\n完成: 成功 {len(self.results)}/{total} 组回测")

    # ------------------------------------------------------------------
    # 指标提取
    # ------------------------------------------------------------------

    def _extract_metrics(
        self,
        symbol: str,
        train_end_date: str,
        version: str,
        merged_data: pd.DataFrame,
    ) -> Dict:
        """
        从 merged_data 提取关键绩效指标
        """
        daily = _extract_daily(merged_data)
        daily_ret = daily['pnl_ret']

        # 基本指标
        annual_ret = daily_ret.mean() * 252
        sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else np.nan
        cum = daily_ret.cumsum()
        max_dd = _calc_dd(cum + 1).min()

        # 胜率（日级别）
        win_rate = (daily_ret > 0).mean()

        # 交易次数：按 pos 变化统计
        md = merged_data.copy()
        pos_changes = (md['pos'] != md['pos'].shift()).sum()
        # 过滤掉 0->0 的变化，只保留实际开仓/平仓/换向
        md['pos_shift'] = md['pos'].shift()
        real_trades = md[(md['pos'] != md['pos_shift']) & (~(md['pos'].eq(0) & md['pos_shift'].eq(0)))].shape[0]

        # 平均持仓
        avg_pos = md['pos'].abs().mean()

        # 多头/空头收益贡献
        long_mask = md['pos'] > 0
        short_mask = md['pos'] < 0
        long_ret = md.loc[long_mask, 'pnl_ret'].sum() if long_mask.any() else 0
        short_ret = md.loc[short_mask, 'pnl_ret'].sum() if short_mask.any() else 0

        return {
            'symbol': symbol,
            'train_end_date': train_end_date,
            'version': version,
            'annual_ret': annual_ret,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'win_rate': win_rate,
            'trade_count': real_trades,
            'avg_position': avg_pos,
            'long_contrib': long_ret,
            'short_contrib': short_ret,
            'days': len(daily),
            'start_date': self.bt_start,
            'end_date': self.bt_end,
        }

    def get_summary_df(self) -> pd.DataFrame:
        """返回汇总 DataFrame"""
        if not self.results:
            return pd.DataFrame()
        df = pd.DataFrame(self.results)
        # 排序：品种 -> 年化收益降序
        df = df.sort_values(['symbol', 'annual_ret'], ascending=[True, False])
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 画图
    # ------------------------------------------------------------------

    def plot_equity_curves(
        self,
        ncols: int = 4,
        figsize_per_sub: Tuple[float, float] = (4.5, 3),
    ):
        """
        每个品种一个子图，叠放不同 train_end_date 的资金曲线
        """
        if not self.daily_dict:
            print("无数据，跳过画图")
            return

        symbols = sorted({k[0] for k in self.daily_dict.keys()})
        nrows = (len(symbols) + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per_sub[0] * ncols, figsize_per_sub[1] * nrows))
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1 or ncols == 1:
            axes = axes.reshape(nrows, ncols)
        axes = axes.flatten()

        for idx, symbol in enumerate(symbols):
            ax = axes[idx]
            # 收集该品种的所有 (train_end_date, version)
            items = {k: v for k, v in self.daily_dict.items() if k[0] == symbol}

            for (sym, ted, ver), daily in items.items():
                label = f"{ted}_{ver}"
                cum = daily['pnl_ret'].cumsum()
                ax.plot(daily['date'], cum, label=label, linewidth=1.2)

            ax.set_title(f"{symbol} Equity Curve", fontsize=11, fontweight='bold')
            ax.legend(fontsize=7, loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=30, labelsize=7)

        # 隐藏多余子图
        for idx in range(len(symbols), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        self._save_or_show(fig, "equity_curves.png")
        return fig

    def plot_long_short_split(
        self,
        ncols: int = 4,
        figsize_per_sub: Tuple[float, float] = (4.5, 3),
    ):
        """
        多空拆分对比：每个品种一个子图，叠放 long_cum + short_cum
        """
        if not self.merged_data_dict:
            print("无数据，跳过画图")
            return

        symbols = sorted({k[0] for k in self.merged_data_dict.keys()})
        nrows = (len(symbols) + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per_sub[0] * ncols, figsize_per_sub[1] * nrows))
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1 or ncols == 1:
            axes = axes.reshape(nrows, ncols)
        axes = axes.flatten()

        for idx, symbol in enumerate(symbols):
            ax = axes[idx]
            items = {k: v for k, v in self.merged_data_dict.items() if k[0] == symbol}

            for (sym, ted, ver), md in items.items():
                md = md.copy()
                md['date'] = pd.to_datetime(md['date'])
                long_cum = md[md['pos'] > 0].groupby('date')['pnl_ret'].sum().cumsum()
                short_cum = md[md['pos'] < 0].groupby('date')['pnl_ret'].sum().cumsum()

                if not long_cum.empty:
                    ax.plot(long_cum.index, long_cum.values, linestyle='-', alpha=0.7,
                            label=f"{ted}_{ver} long")
                if not short_cum.empty:
                    ax.plot(short_cum.index, short_cum.values, linestyle='--', alpha=0.7,
                            label=f"{ted}_{ver} short")

            ax.set_title(f"{symbol} Long/Short Split", fontsize=11, fontweight='bold')
            ax.legend(fontsize=6, loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=30, labelsize=7)

        for idx in range(len(symbols), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        self._save_or_show(fig, "long_short_split.png")
        return fig

    def plot_metrics_bar(
        self,
        metrics: Optional[List[str]] = None,
        figsize: Tuple[float, float] = (14, 5),
    ):
        """
        绩效指标柱状图：每个品种一组柱子，不同 train_end_date 并排
        """
        df = self.get_summary_df()
        if df.empty:
            print("无数据，跳过画图")
            return

        if metrics is None:
            metrics = ['annual_ret', 'sharpe', 'max_dd']

        # 构造 x 轴标签
        df['group'] = df['train_end_date'] + '_' + df['version']
        groups = df['group'].unique()
        symbols = df['symbol'].unique()
        x = np.arange(len(symbols))
        width = 0.8 / len(groups)

        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(figsize[0], figsize[1]))
        if n_metrics == 1:
            axes = [axes]

        colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))

        for m_idx, metric in enumerate(metrics):
            ax = axes[m_idx]
            for g_idx, group in enumerate(groups):
                sub = df[df['group'] == group].set_index('symbol').reindex(symbols)
                vals = sub[metric].fillna(0).values
                ax.bar(x + (g_idx - len(groups)/2 + 0.5) * width, vals, width,
                       label=group, color=colors[g_idx], alpha=0.85)

            ax.set_xticks(x)
            ax.set_xticklabels(symbols)
            ax.set_title(metric, fontsize=12, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
            ax.axhline(0, color='black', linewidth=0.5)

        plt.tight_layout()
        self._save_or_show(fig, "metrics_bar.png")
        return fig

    def plot_combined_equity(self, figsize: Tuple[float, float] = (10, 5)):
        """
        多品种等权合成总资金曲线
        对每个 train_end_date × version 组合，把各品种日收益等权合成一条曲线
        """
        if not self.daily_dict:
            print("无数据，跳过画图")
            return

        # 按 (train_end_date, version) 分组
        combo_dict = defaultdict(list)
        for (sym, ted, ver), daily in self.daily_dict.items():
            combo_dict[(ted, ver)].append(daily.set_index('date')['pnl_ret'])

        fig, ax = plt.subplots(figsize=figsize)
        for (ted, ver), ret_series_list in combo_dict.items():
            # 对齐日期后等权平均
            combined = pd.concat(ret_series_list, axis=1).mean(axis=1)
            cum = combined.cumsum()
            ax.plot(cum.index, cum.values, label=f"{ted}_{ver} (n={len(ret_series_list)})", linewidth=1.5)

        ax.set_title("Combined Equal-Weight Equity Curve", fontsize=12, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=30)
        plt.tight_layout()
        self._save_or_show(fig, "combined_equity.png")
        return fig

    # ------------------------------------------------------------------
    # 保存
    # ------------------------------------------------------------------

    def _save_or_show(self, fig, filename: str):
        """如果有 output_dir 就保存，否则只展示"""
        if self.output_dir:
            path = self.output_dir / filename
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"  图已保存: {path}")
        else:
            plt.show()

    def save_summary_csv(self, filename: str = "summary.csv"):
        """保存汇总表格为 CSV"""
        df = self.get_summary_df()
        if self.output_dir:
            path = self.output_dir / filename
            df.to_csv(path, index=False, float_format='%.4f')
            print(f"  表格已保存: {path}")
        return df

    def save_all(self, prefix: str = ""):
        """一键保存表格 + 所有图"""
        self.save_summary_csv(f"{prefix}summary.csv" if prefix else "summary.csv")
        self.plot_equity_curves()
        self.plot_metrics_bar()
        self.plot_combined_equity()


# ------------------------------------------------------------------
# 命令行入口
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="多品种多版本回测绩效对比")
    parser.add_argument("--model-dir", required=True, help="模型根目录")
    parser.add_argument("--symbols", nargs="+", default=['A', 'B', 'C', 'CS', 'M', 'Y', 'P', 'LH'],
                        help="品种列表，默认 8 个 DCE 农产品")
    parser.add_argument("--train-end-dates", nargs="+", required=True,
                        help="训练截止日期列表，如 2025-07-01 2025-10-01")
    parser.add_argument("--versions", nargs="+", default=['v0'],
                        help="版本列表，如 v0 v1 baseline trick5_huber")
    parser.add_argument("--train-label", type=int, default=5, help="预测周期，默认 5min")
    parser.add_argument("--bt-start", required=True, help="回测开始日期 YYYY-MM-DD")
    parser.add_argument("--bt-end", required=True, help="回测结束日期 YYYY-MM-DD")
    parser.add_argument("--th1", type=float, default=0.9, help="开仓阈值")
    parser.add_argument("--th2", type=float, default=0.5, help="减仓阈值")
    parser.add_argument("--holding-bars", type=int, default=10, help="最大持仓 bar 数")
    parser.add_argument("--day", type=int, default=1725, help="日最大持仓限制")
    parser.add_argument("--fee", type=float, default=0, help="手续费率")
    parser.add_argument("--v", type=int, default=2, choices=[2, 3], help="回测版本 2 或 3")
    parser.add_argument("--money", default=None, help="本金设置。支持：数字（统一本金）、JSON字典（按品种）、'auto'（自动按margin/0.4）")
    parser.add_argument("--output-dir", default="./eval_output", help="输出目录")
    parser.add_argument("--no-plot", action="store_true", help="只跑回测不画图")

    args = parser.parse_args()

    bt_params = {
        'th1': args.th1,
        'th2': args.th2,
        'holding_bars': args.holding_bars,
        'day': args.day,
        'fee': args.fee,
        'v': args.v,
    }

    # 解析 money 参数
    money_arg = args.money
    if money_arg is not None:
        if money_arg.lower() == 'auto':
            money_arg = 'auto'
        else:
            try:
                money_arg = float(money_arg)
            except ValueError:
                import json
                money_arg = json.loads(money_arg)

    evaluator = MultiSymbolEvaluator(
        model_base_dir=args.model_dir,
        bt_start=args.bt_start,
        bt_end=args.bt_end,
        bt_params=bt_params,
        output_dir=args.output_dir,
        money=money_arg,
    )

    evaluator.run_batch(
        symbols=args.symbols,
        train_end_dates=args.train_end_dates,
        versions=args.versions,
        train_label=args.train_label,
    )

    summary = evaluator.get_summary_df()
    print("\n=== 汇总表格 ===")
    print(summary.to_string(index=False))

    evaluator.save_summary_csv()

    if not args.no_plot:
        print("\n=== 生成图表 ===")
        evaluator.plot_equity_curves()
        evaluator.plot_metrics_bar()
        evaluator.plot_combined_equity()


if __name__ == '__main__':
    main()
