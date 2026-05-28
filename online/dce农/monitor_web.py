"""
盘中交易监控 + Web 面板
============================
独立进程，不和主策略耦合。
每 5 秒扫描主程序保存的 json status / min 文件，
通过 now_pos 变化推断信号，并按"延迟一分钟"规则计算成交价。

启动:
    python monitor_web.py

访问:
    http://localhost:5000
"""

import os
import sys
import json
import time
import glob
import atexit
import shutil
import threading
import webbrowser
from datetime import timezone, timedelta, datetime
from typing import Dict, List, Optional
import pandas as pd
from flask import Flask, render_template_string, request

BEIJING_TZ = timezone(timedelta(hours=8))

app = Flask(__name__)

# 全局状态（监控线程写入，Flask 读取）
MONITOR_STATE = {
    'positions': {},
    'pending_opens': {},
    'pending_closes': {},
    'trades': {},
    'latest_signals': {},
    'last_update': '-',
}

SYMBOLS: List[str] = []


def _load_kline(symbol: str, save_root: str) -> list:
    """读取当日完整行情（含前一夜盘），剔除异常 bar"""
    data_dir = os.path.join(save_root, symbol, 'data')
    now = datetime.now(BEIJING_TZ)
    date_str = now.strftime('%Y-%m-%d')
    # 前一个交易日：周一回退到上周五，其他回退1天（跳过周末）
    weekday = now.weekday()
    if weekday == 0:  # Monday
        prev_trade_day = (now - timedelta(days=3)).strftime('%Y-%m-%d')
    else:
        prev_trade_day = (now - timedelta(days=1)).strftime('%Y-%m-%d')

    raw_rows = []
    for d in [prev_trade_day, date_str]:
        pattern = os.path.join(data_dir, f'{symbol}_min_{d}_*.csv')
        for fpath in sorted(glob.glob(pattern)):
            basename = os.path.basename(fpath)
            time_part = basename.replace(f'{symbol}_min_{d}_', '').replace('.csv', '')
            try:
                hour = int(time_part.split('-')[0])
            except ValueError:
                continue
            if d == prev_trade_day and hour < 21:
                continue

            try:
                df = pd.read_csv(fpath)
                if df.empty:
                    continue
                row = df.iloc[0]
                o = float(row.get('open', 0))
                h = float(row.get('high', 0))
                l = float(row.get('low', 0))
                c = float(row.get('close', 0))
                if o <= 0 or h <= 0 or l <= 0 or c <= 0:
                    continue
                # 过滤单个 bar 内价格严重不协调的异常（如 low 比 open/close 小一个量级）
                if l < min(o, c) * 0.5 or h > max(o, c) * 2:
                    continue
                dt_str = str(row.get('datetime', ''))
                # 标准化：去掉可能的微秒，确保 time_key 与交易记录分钟字符串一致
                dt_clean = dt_str.split('.')[0]
                raw_rows.append({
                    'time': dt_clean,
                    'time_key': dt_clean.replace(' ', '_').replace(':', '-'),
                    'open': o, 'high': h, 'low': l, 'close': c,
                    'avg_price': float(row.get('avg_price_from_5s', c)),
                })
            except Exception:
                pass

    return raw_rows


def _calc_holding(open_str: str, close_str: str) -> str:
    """计算持仓时间（纯数字分钟）"""
    try:
        dt_open = datetime.strptime(open_str, "%Y-%m-%d_%H-%M-%S")
        dt_close = datetime.strptime(close_str, "%Y-%m-%d_%H-%M-%S")
        minutes = int((dt_close - dt_open).total_seconds() / 60)
        return str(minutes)
    except Exception:
        return "-"


def _fmt_time(ts_str: str) -> str:
    """2026-05-18_21-12-00 -> 21:12:00"""
    try:
        return ts_str.split('_')[1].replace('-', ':')
    except Exception:
        return ts_str


def _build_url(**kwargs) -> str:
    """基于当前 query string 构建新 URL，保留现有参数并覆盖指定参数"""
    args = request.args.copy()
    for k, v in kwargs.items():
        if v is None or v == '':
            args.pop(k, None)
        else:
            args[k] = v
    if not args:
        return '/'
    return '?' + '&'.join(f'{k}={v}' for k, v in args.items())


class TradeMonitor:
    def __init__(self, save_root: str, symbols: list, clear_on_start: bool = True):
        self.save_root = save_root
        self.symbols = symbols
        self.data_dir = os.path.join(save_root, 'data')
        self.monitor_dir = os.path.join(save_root, '盘中分析')
        os.makedirs(self.monitor_dir, exist_ok=True)

        if clear_on_start:
            self._clear_monitor_history()

        self.positions: Dict[str, dict] = {}
        self.pending_opens: Dict[str, dict] = {}
        self.pending_closes: Dict[str, dict] = {}
        self.trades: Dict[str, list] = {}
        self.latest_signals: Dict[str, str] = {}
        self.last_pos: Dict[str, int] = {}
        self.processed: Dict[str, set] = {sym: set() for sym in symbols}
        self.market_time: Optional[str] = None

        for sym in symbols:
            self.trades[sym] = []

    def _clear_monitor_history(self):
        try:
            for fpath in glob.glob(os.path.join(self.monitor_dir, '*_trades_*.json')):
                os.remove(fpath)
                print(f"[Monitor] 清理旧文件: {os.path.basename(fpath)}")
        except Exception as e:
            print(f"[Monitor] 清理历史文件失败: {e}")

    def clear_all(self):
        self._clear_monitor_history()
        self.positions.clear()
        self.pending_opens.clear()
        self.pending_closes.clear()
        self.latest_signals.clear()
        self.last_pos.clear()
        for sym in self.symbols:
            self.trades[sym].clear()
            self.processed[sym].clear()

    def _read_pos(self, symbol: str, minute_str: str) -> Optional[int]:
        json_dir = os.path.join(self.save_root, symbol, 'json')
        fpath = os.path.join(json_dir, f'trading_status_{minute_str}.json')
        if not os.path.exists(fpath):
            return None
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if 'now_pos' in data:
                return int(data['now_pos'])
        except Exception:
            pass
        return None

    def _read_avg_price(self, symbol: str, minute_str: str) -> Optional[float]:
        fpath = os.path.join(self.save_root, symbol, 'data', f'{symbol}_min_{minute_str}.csv')
        if not os.path.exists(fpath):
            return None
        try:
            df = pd.read_csv(fpath)
            if 'avg_price_from_5s' in df.columns and not df.empty:
                return float(df['avg_price_from_5s'].iloc[0])
        except Exception:
            pass
        return None

    def _infer_signal(self, prev_pos: int, curr_pos: int) -> str:
        if prev_pos == 0 and curr_pos == 1:
            return '开多'
        if prev_pos == 0 and curr_pos == -1:
            return '开空'
        if prev_pos == 1 and curr_pos == 0:
            return '平多'
        if prev_pos == -1 and curr_pos == 0:
            return '平空'
        if prev_pos == 1 and curr_pos == -1:
            return '平多开空'
        if prev_pos == -1 and curr_pos == 1:
            return '平空开多'
        if curr_pos == 1:
            return '持多'
        if curr_pos == -1:
            return '持空'
        return '等待行情'

    def _process_symbol(self, symbol: str, minute_str: str):
        # 更新市场时间（回放场景下用文件时间作为市场时间）
        self.market_time = minute_str

        if symbol in self.pending_closes and symbol in self.positions:
            price = self._read_avg_price(symbol, minute_str)
            if price is not None:
                pos = self.positions[symbol]
                trade = (pos['open_time'], minute_str, pos['direction'], pos['open_price'], round(price, 2))
                self.trades[symbol].append(trade)
                print(f"[Monitor][{symbol}] 平仓完成: {trade}")
                del self.positions[symbol]
                del self.pending_closes[symbol]
                self._save_trades(symbol)

        if symbol in self.pending_opens:
            price = self._read_avg_price(symbol, minute_str)
            if price is not None:
                self.positions[symbol] = {
                    'direction': self.pending_opens[symbol]['direction'],
                    'open_time': minute_str,
                    'open_price': round(price, 2)
                }
                print(f"[Monitor][{symbol}] 开仓完成: 时间={minute_str}, 方向={self.pending_opens[symbol]['direction']}, 价格={round(price, 2)}")
                del self.pending_opens[symbol]
                self._save_trades(symbol)

        curr_pos = self._read_pos(symbol, minute_str)
        if curr_pos is None:
            return

        prev_pos = self.last_pos.get(symbol, 0)
        signal = self._infer_signal(prev_pos, curr_pos)
        self.latest_signals[symbol] = signal

        open_direction = None
        if signal in ('开多', '平空开多'):
            open_direction = 1
        elif signal in ('开空', '平多开空'):
            open_direction = -1

        close_signals = ('平多', '平空', '平多开空', '平空开多', '过长时间无合适信号，平仓')

        if signal in close_signals and symbol in self.positions and symbol not in self.pending_closes:
            self.pending_closes[symbol] = {'time': minute_str}
            print(f"[Monitor][{symbol}] 标记待平仓: {minute_str} {signal}")

        if open_direction is not None and symbol not in self.positions and symbol not in self.pending_opens:
            self.pending_opens[symbol] = {'time': minute_str, 'direction': open_direction}
            print(f"[Monitor][{symbol}] 标记待开仓: {minute_str} {signal}")

        self.last_pos[symbol] = curr_pos

    def _save_trades(self, symbol: str):
        date_str = datetime.now(BEIJING_TZ).strftime('%Y%m%d')
        fpath = os.path.join(self.monitor_dir, f'{symbol}_trades_{date_str}.json')
        try:
            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(self.trades[symbol], f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Monitor][{symbol}] 保存失败: {e}")

    def scan_once(self):
        """扫描所有品种，自动检测当日所有未处理的分钟文件并补齐。"""
        try:
            now = pd.Timestamp.now(tz=BEIJING_TZ)
            current_date = now.strftime('%Y-%m-%d')

            for symbol in self.symbols:
                try:
                    # 日期切换时清理 processed
                    if self.processed[symbol]:
                        sample = next(iter(self.processed[symbol]))
                        if not sample.startswith(current_date):
                            self.processed[symbol].clear()
                            self.last_pos.pop(symbol, None)

                    json_dir = os.path.join(self.save_root, symbol, 'json')
                    if not os.path.exists(json_dir):
                        continue

                    # 自动检测所有当日未处理的文件（补漏机制）
                    files = glob.glob(os.path.join(json_dir, f'trading_status_{current_date}*.json'))
                    for fpath in sorted(files):
                        basename = os.path.basename(fpath)
                        minute_str = basename.replace('trading_status_', '').replace('.json', '')
                        if minute_str in self.processed[symbol]:
                            continue
                        self._process_symbol(symbol, minute_str)
                        self.processed[symbol].add(minute_str)
                except Exception as e:
                    print(f"[Monitor][{symbol}] 扫描异常: {e}")

            global MONITOR_STATE
            MONITOR_STATE = {
                'positions': {k: v.copy() for k, v in self.positions.items()},
                'pending_opens': {k: v.copy() for k, v in self.pending_opens.items()},
                'pending_closes': {k: v.copy() for k, v in self.pending_closes.items()},
                'trades': {k: list(v) for k, v in self.trades.items()},
                'latest_signals': self.latest_signals.copy(),
                'last_update': now.strftime('%H:%M:%S'),
                'market_time': self.market_time,
            }
        except Exception as e:
            print(f"[Monitor] scan_once 异常: {e}")

    def run(self, poll_interval: float = 5.0, skip_history: bool = True):
        if skip_history:
            # 清空状态，只把当天已有文件标记为已处理，不做实际复盘
            today = pd.Timestamp.now(tz=BEIJING_TZ).strftime('%Y-%m-%d')
            print(f"[Monitor] 启动（跳过历史复盘）| 品种: {self.symbols}")
            for symbol in self.symbols:
                try:
                    json_dir = os.path.join(self.save_root, symbol, 'json')
                    if not os.path.exists(json_dir):
                        continue
                    files = glob.glob(os.path.join(json_dir, f'trading_status_{today}*.json'))
                    for fpath in files:
                        basename = os.path.basename(fpath)
                        minute_str = basename.replace('trading_status_', '').replace('.json', '')
                        self.processed[symbol].add(minute_str)
                    print(f"[Monitor][{symbol}] 已标记 {len(files)} 个历史文件为已处理（不复盘）")
                except Exception as e:
                    print(f"[Monitor][{symbol}] 标记历史文件异常: {e}")
        else:
            print(f"[Monitor] 启动复盘 | 品种: {self.symbols}")
            for symbol in self.symbols:
                try:
                    json_dir = os.path.join(self.save_root, symbol, 'json')
                    if not os.path.exists(json_dir):
                        continue
                    files = sorted(glob.glob(os.path.join(json_dir, 'trading_status_*.json')))
                    for fpath in files:
                        try:
                            basename = os.path.basename(fpath)
                            minute_str = basename.replace('trading_status_', '').replace('.json', '')
                            self._process_symbol(symbol, minute_str)
                            self.processed[symbol].add(minute_str)
                        except Exception as e:
                            print(f"[Monitor][{symbol}] 复盘单文件异常 ({basename}): {e}")
                    print(f"[Monitor][{symbol}] 复盘完成 | 已处理 {len(files)} 分钟 | 交易 {len(self.trades[symbol])} 笔")
                except Exception as e:
                    print(f"[Monitor][{symbol}] 复盘异常: {e}")

        print(f"[Monitor] 进入实时监控 | 扫描间隔 {poll_interval}s")
        while True:
            try:
                self.scan_once()
            except Exception as e:
                print(f"[Monitor] 扫描循环异常: {e}")
            time.sleep(poll_interval)


# ==============================================================================
# HTML Template
# ==============================================================================
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>盘中实时监控</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
            margin: 0; padding: 15px;
            background: #f0f2f5;
            color: #333;
            height: 100vh;
            overflow: hidden;
        }
        h1 { margin: 0 0 8px 0; font-size: 22px; }
        .subtitle { color: #888; font-size: 12px; margin-bottom: 12px; }
        .main-layout {
            display: flex;
            flex-direction: column;
            gap: 12px;
            height: calc(100vh - 70px);
        }
        .top-row {
            display: flex;
            gap: 12px;
            flex: 0 0 auto;
            height: 390px;
            min-height: 0;
        }
        .top-row > .card {
            flex: 1.25;
            overflow: hidden;
        }
        .top-row > .card:last-child {
            flex: 0.75;
        }
        .bottom-row {
            flex: 1;
            min-height: 0;
            display: flex;
            flex-direction: column;
            position: relative;
        }
        .card {
            background: white; border-radius: 10px; padding: 14px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            display: flex;
            flex-direction: column;
        }
        .card h2 { margin: 0 0 10px 0; font-size: 15px; color: #555; }
        table { width: 100%; border-collapse: collapse; font-size: 12px; }
        th, td { padding: 5px 8px; text-align: left; border-bottom: 1px solid #eee; }
        th { background: #fafafa; font-weight: 600; color: #666; }
        .long { color: #d32f2f; font-weight: bold; }
        .short { color: #388e3c; font-weight: bold; }
        .pending { color: #f57c00; font-weight: bold; }
        .muted { color: #999; }
        .right { text-align: right; }
        .pnl-pos { color: #d32f2f; font-weight: bold; }
        .pnl-neg { color: #388e3c; font-weight: bold; }
        .sym-btns {
            display: flex;
            gap: 6px;
            flex-wrap: wrap;
            margin-bottom: 10px;
        }
        .sym-btns a {
            display: inline-block;
            padding: 4px 12px;
            border: 1px solid #ddd;
            background: white;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
            text-decoration: none;
            color: #333;
            transition: all 0.15s;
        }
        .sym-btns a:hover { background: #f5f5f5; }
        .sym-btns a.active {
            background: #333;
            color: white;
            border-color: #333;
        }
        .scroll-table {
            flex: 1;
            overflow-y: auto;
            min-height: 0;
        }
        .kline-canvas {
            flex: 1;
            min-height: 0;
            width: 100%;
            border: 1px solid #eee;
            border-radius: 4px;
            background: #fafafa;
        }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
</head>
<body>
    <h1>盘中实时监控</h1>
    <div class="subtitle">最后更新: {{ last_update }} | 自动刷新间隔: 10秒 {% if thread_stale %}<span style="color:#d32f2f;font-weight:bold;">| ⚠️ 监控线程已停止更新</span>{% endif %}</div>

    <div class="main-layout">
        <!-- 第一排：品种状态 + 交易记录 -->
        <div class="top-row">
            <div class="card">
                <h2>品种状态</h2>
                <div class="scroll-table">
                    <table>
                        <tr>
                            <th>品种</th>
                            <th>最新信号</th>
                            <th>持仓</th>
                            <th>开仓时间</th>
                            <th class="right">开仓价</th>
                            <th class="right">持仓分钟</th>
                            <th>待处理</th>
                            <th class="right">开仓次数</th>
                            <th class="right">当日盈亏</th>
                        </tr>
                        {% for sym in symbols %}
                        <tr>
                            <td><b>{{ sym }}</b></td>
                            <td>{{ signals.get(sym, '-') }}</td>
                            <td>
                                {% if positions.get(sym) %}
                                    {% if positions[sym].direction == 1 %}
                                        <span class="long">多仓</span>
                                    {% else %}
                                        <span class="short">空仓</span>
                                    {% endif %}
                                {% else %}
                                    <span class="muted">无持仓</span>
                                {% endif %}
                            </td>
                            <td>{{ positions.get(sym, {}).get('open_time', '-') }}</td>
                            <td class="right">{{ positions.get(sym, {}).get('open_price', '-') }}</td>
                            <td class="right">{{ holding_minutes.get(sym, '-') }}</td>
                            <td>
                                {% if pending_opens.get(sym) %}
                                    <span class="pending">待开仓({% if pending_opens[sym].direction == 1 %}多{% else %}空{% endif %})</span>
                                {% elif pending_closes.get(sym) %}
                                    <span class="pending">待平仓</span>
                                {% else %}
                                    <span class="muted">-</span>
                                {% endif %}
                            </td>
                            <td class="right">{{ open_counts.get(sym, 0) }}</td>
                            <td class="right">
                                {% set p = sym_pnl.get(sym, 0) %}
                                {% if p > 0 %}
                                    <span class="pnl-pos">+{{ "%.2f"|format(p) }}</span>
                                {% elif p < 0 %}
                                    <span class="pnl-neg">{{ "%.2f"|format(p) }}</span>
                                {% else %}
                                    <span class="muted">0.00</span>
                                {% endif %}
                            </td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>
            </div>

            <div class="card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                    <h2 style="margin:0;">今日交易记录</h2>
                    <div class="sym-btns" style="margin:0;">
                        <a href="{{ build_url(sym='ALL') }}" class="{% if selected_sym == 'ALL' %}active{% endif %}">全部</a>
                        {% for sym in symbols %}
                        <a href="{{ build_url(sym=sym) }}" class="{% if selected_sym == sym %}active{% endif %}">{{ sym }}</a>
                        {% endfor %}
                    </div>
                </div>
                <div class="scroll-table">
                    <table>
                        <tr>
                            <th>品种</th>
                            <th>方向</th>
                            <th>开仓时间</th>
                            <th class="right">开仓价</th>
                            <th>平仓时间</th>
                            <th class="right">平仓价</th>
                            <th class="right">持仓</th>
                            <th class="right">盈亏</th>
                        </tr>
                        {% for sym in symbols %}
                            {% set sym_trades_list = filtered_trades.get(sym, []) %}
                            {% if sym_trades_list %}
                                {% for trade in sym_trades_list %}
                                <tr>
                                    <td>{{ sym }}</td>
                                    <td>
                                        {% if trade[2] == 1 %}
                                            <span class="long">多</span>
                                        {% else %}
                                            <span class="short">空</span>
                                        {% endif %}
                                    </td>
                                    <td>{{ trade[0] | fmt_time }}</td>
                                    <td class="right">{{ "%.2f"|format(trade[3]) }}</td>
                                    <td>{{ trade[1] | fmt_time }}</td>
                                    <td class="right">{{ "%.2f"|format(trade[4]) }}</td>
                                    <td class="right">{{ holding_times.get(sym, [])[loop.index0] }}</td>
                                    <td class="right">
                                        {% set pnl = trade[4] - trade[3] if trade[2] == 1 else trade[3] - trade[4] %}
                                        {% if pnl > 0 %}
                                            <span class="pnl-pos">+{{ "%.2f"|format(pnl) }}</span>
                                        {% elif pnl < 0 %}
                                            <span class="pnl-neg">{{ "%.2f"|format(pnl) }}</span>
                                        {% else %}
                                            <span class="muted">0.00</span>
                                        {% endif %}
                                    </td>
                                </tr>
                                {% endfor %}
                            {% endif %}
                        {% endfor %}
                        {% set total_trades = namespace(count=0) %}
                        {% for sym in symbols %}{% set total_trades.count = total_trades.count + filtered_trades.get(sym, [])|length %}{% endfor %}
                        {% if total_trades.count == 0 %}
                        <tr><td colspan="8" class="muted" style="text-align:center;">暂无交易</td></tr>
                        {% endif %}
                    </table>
                </div>
            </div>
        </div>

        <!-- 第二排：行情图独占 -->
        <div class="card bottom-row">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h2 style="margin:0;">
                    {% if selected_sym == 'ALL' %}
                        行情图 - 全部（请选择品种）
                    {% else %}
                        行情图 - {{ selected_sym }}
                        {% if selected_sym in positions %}
                            {% set pos = positions[selected_sym] %}
                            <span style="font-size:13px;font-weight:normal;color:#666;margin-left:8px;">
                                | {% if pos.direction == 1 %}<span class="long">多仓</span>{% else %}<span class="short">空仓</span>{% endif %}
                                | 开仓价 {{ pos.open_price }}
                                | 已持仓 {{ holding_minutes.get(selected_sym, '-') }} 分钟
                            </span>
                        {% endif %}
                    {% endif %}
                </h2>
            </div>
            <div id="klineChart" style="flex:1;min-height:0;width:100%;border:1px solid #eee;border-radius:4px;background:#fafafa;"></div>
        </div>
    </div>

    <script>
        // 10 秒自动刷新，保留当前 URL 参数
        setInterval(function() {
            window.location.reload();
        }, 10000);

        (function() {
            const klineData = {{ kline_data | safe }};
            const trades = {{ sym_trades | safe }};
            const chartDom = document.getElementById('klineChart');
            if (!chartDom || klineData.length === 0) {
                if (chartDom) {
                    chartDom.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#999;font-size:16px;">暂无行情数据</div>';
                }
                return;
            }

            const chart = echarts.init(chartDom);

            const times = klineData.map(d => d.time);
            const candleData = klineData.map(d => [d.open, d.close, d.low, d.high]);
            // 保持完整长度，缺失填 null，避免 line 系列错位
            const twapLineData = klineData.map(d => {
                if (typeof d.avg_price === 'number' && !isNaN(d.avg_price)) {
                    return d.avg_price;
                }
                return null;
            });

            // 计算价格范围，用于箭头定位在画布上下两侧
            const allPrices = klineData.flatMap(d => [d.high, d.low]);
            const dataMin = Math.min(...allPrices);
            const dataMax = Math.max(...allPrices);
            const dataRange = dataMax - dataMin || 1;
            const arrowPad = dataRange * 0.12;

            // 持仓区间 markArea
            const markAreas = [];
            trades.forEach(trade => {
                const openKey = trade[0];
                const closeKey = trade[1];
                const dir = trade[2];
                const oIdx = klineData.findIndex(d => d.time_key === openKey);
                const cIdx = closeKey ? klineData.findIndex(d => d.time_key === closeKey) : -1;
                if (oIdx >= 0) {
                    markAreas.push([{
                        xAxis: oIdx,
                        itemStyle: { color: dir === 1 ? 'rgba(211,47,47,0.08)' : 'rgba(56,142,60,0.08)' }
                    }, {
                        xAxis: cIdx >= 0 ? cIdx : klineData.length - 1
                    }]);
                }
            });

            // 买卖点 markPoint
            const markPoints = [];
            trades.forEach(trade => {
                const openKey = trade[0];
                const closeKey = trade[1];
                const dir = trade[2];
                const openPrice = trade[3];
                const closePrice = trade[4];
                const oIdx = klineData.findIndex(d => d.time_key === openKey);
                const cIdx = closeKey ? klineData.findIndex(d => d.time_key === closeKey) : -1;

                // 开仓箭头：固定在画布上下两侧（不遮挡K线）
                if (oIdx >= 0) {
                    const arrowY = dir === 1 ? dataMin - arrowPad : dataMax + arrowPad;
                    markPoints.push({
                        coord: [oIdx, arrowY],
                        value: dir === 1 ? '开多' : '开空',
                        itemStyle: { color: dir === 1 ? '#d32f2f' : '#388e3c' },
                        symbol: 'triangle',
                        symbolRotate: dir === 1 ? 0 : 180,
                        symbolSize: 14,
                        label: { show: false }
                    });
                }

                // 平仓标记：X 留在 K 线附近（平仓价位置）
                let avgPrice = closePrice;
                if (cIdx >= 0 && klineData[cIdx].avg_price) {
                    avgPrice = klineData[cIdx].avg_price;
                }
                if (cIdx >= 0) {
                    markPoints.push({
                        coord: [cIdx, avgPrice],
                        value: '平仓',
                        itemStyle: { color: '#222' },
                        symbol: 'path://M -5 -5 L 5 5 M 5 -5 L -5 5',
                        symbolSize: 10,
                        label: { show: false }
                    });
                }
            });

            const option = {
                tooltip: {
                    trigger: 'axis',
                    axisPointer: { type: 'cross' },
                    formatter: function(params) {
                        const idx = params[0].dataIndex;
                        const d = klineData[idx];
                        const candle = params.find(p => p.seriesType === 'candlestick');
                        const twap = params.find(p => p.seriesName === 'TWAP');
                        let s = '<div style="font-weight:bold;margin-bottom:4px;">' + d.time + '</div>';
                        if (candle) {
                            const cd = candle.data;
                            s += '<div>开: ' + cd[0] + '</div>';
                            s += '<div>收: ' + cd[1] + '</div>';
                            s += '<div>低: ' + cd[2] + '</div>';
                            s += '<div>高: ' + cd[3] + '</div>';
                        }
                        if (twap) {
                            s += '<div style="color:#42a5f5;font-weight:bold;">TWAP(>5s): ' + twap.data.toFixed(2) + '</div>';
                        }
                        return s;
                    }
                },
                grid: { left: 60, right: 60, top: 40, bottom: 38 },
                xAxis: {
                    type: 'category',
                    data: times,
                    axisLabel: {
                        formatter: function(v) {
                            return v ? v.split(' ')[1] || v : '';
                        }
                    },
                    axisLine: { lineStyle: { color: '#ccc' } },
                    splitLine: { show: false }
                },
                yAxis: {
                    type: 'value',
                    min: dataMin - arrowPad * 1.2,
                    max: dataMax + arrowPad * 1.2,
                    splitLine: { lineStyle: { color: '#e0e0e0' } },
                    axisLine: { lineStyle: { color: '#ccc' } }
                },
                series: [
                    {
                        name: 'K线',
                        type: 'candlestick',
                        data: candleData,
                        itemStyle: {
                            color: '#d32f2f',
                            color0: '#388e3c',
                            borderColor: '#d32f2f',
                            borderColor0: '#388e3c'
                        },
                        markArea: { data: markAreas },
                        markPoint: { data: markPoints, symbolOffset: [0, 0] }
                    },
                    {
                        name: 'TWAP',
                        type: 'line',
                        data: twapLineData,
                        symbol: 'none',
                        lineStyle: {
                            color: '#42a5f5',
                            width: 1.5,
                            type: 'dashed'
                        },
                        z: 5
                    }
                ],
                animation: false
            };

            chart.setOption(option);

            window.addEventListener('resize', function() {
                chart.resize();
            });
        })();
    </script>
</body>
</html>
'''


@app.template_filter('fmt_time')
def fmt_time_filter(ts_str):
    return _fmt_time(ts_str)


@app.route('/')
def index():
    state = MONITOR_STATE
    save_root = app.config.get('SAVE_ROOT', '')
    selected_sym = request.args.get('sym', 'ALL')
    if not selected_sym:
        selected_sym = 'ALL'

    # 线程健康检测（超过 30 秒未更新视为 stale）
    thread_stale = False
    last_update = state.get('last_update', '-')
    if last_update != '-':
        try:
            now = datetime.now(BEIJING_TZ)
            lu = datetime.strptime(last_update, '%H:%M:%S').replace(
                year=now.year, month=now.month, day=now.day, tzinfo=BEIJING_TZ
            )
            if (now - lu).total_seconds() > 30:
                thread_stale = True
        except Exception:
            pass

    # 行情图数据：ALL 时留白
    kline_data = []
    sym_trades = []
    if selected_sym != 'ALL' and selected_sym in SYMBOLS and save_root:
        kline_data = _load_kline(selected_sym, save_root)
        sym_trades = list(state.get('trades', {}).get(selected_sym, []))
        # 当前持仓也加入 sym_trades，用于绘制未平仓区间的背景色
        pos = state.get('positions', {}).get(selected_sym)
        if pos:
            sym_trades.append([pos['open_time'], '', pos['direction'], pos['open_price'], 0])

    # 交易记录筛选（与行情图联动）
    filtered_trades = {}
    holding_times = {}
    for sym in SYMBOLS:
        sym_trades_list = state.get('trades', {}).get(sym, [])
        if selected_sym != 'ALL' and sym != selected_sym:
            filtered_trades[sym] = []
            holding_times[sym] = []
        else:
            filtered_trades[sym] = sym_trades_list
            holding_times[sym] = [_calc_holding(t[0], t[1]) for t in sym_trades_list]

    sym_pnl = {}
    open_counts = {}
    for sym in SYMBOLS:
        trades = state.get('trades', {}).get(sym, [])
        total = sum(t[4] - t[3] if t[2] == 1 else t[3] - t[4] for t in trades)
        sym_pnl[sym] = round(total, 2)
        open_counts[sym] = len(trades)

    # 计算当前持仓分钟数（回放场景用 market_time，实盘用系统时间）
    holding_minutes = {}
    market_time_str = state.get('market_time')
    now_ref = None
    if market_time_str:
        try:
            now_ref = datetime.strptime(market_time_str, "%Y-%m-%d_%H-%M-%S").replace(tzinfo=BEIJING_TZ)
        except Exception:
            pass
    if now_ref is None:
        now_ref = datetime.now(BEIJING_TZ)
    for sym, pos in state.get('positions', {}).items():
        try:
            dt_open = datetime.strptime(pos['open_time'], "%Y-%m-%d_%H-%M-%S").replace(tzinfo=BEIJING_TZ)
            minutes = int((now_ref - dt_open).total_seconds() / 60)
            holding_minutes[sym] = minutes
        except Exception:
            holding_minutes[sym] = '-'

    return render_template_string(
        HTML_TEMPLATE,
        symbols=SYMBOLS,
        selected_sym=selected_sym,
        kline_data=json.dumps(kline_data),
        sym_trades=json.dumps(sym_trades),
        last_update=last_update,
        thread_stale=thread_stale,
        signals=state.get('latest_signals', {}),
        positions=state.get('positions', {}),
        pending_opens=state.get('pending_opens', {}),
        pending_closes=state.get('pending_closes', {}),
        filtered_trades=filtered_trades,
        holding_times=holding_times,
        holding_minutes=holding_minutes,
        sym_pnl=sym_pnl,
        open_counts=open_counts,
        build_url=_build_url,
    )


# ==============================================================================
# 启动入口
# ==============================================================================
if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    save_root = config['paths']['save_files_root']
    SYMBOLS = config.get('symbols', [])
    app.config['SAVE_ROOT'] = save_root

    monitor = TradeMonitor(save_root, SYMBOLS, clear_on_start=True)
    atexit.register(monitor.clear_all)

    monitor_thread = threading.Thread(target=monitor.run, kwargs={'poll_interval': 5.0, 'skip_history': False}, daemon=True)
    monitor_thread.start()

    import socket
    host = '0.0.0.0'
    base_port = 5001
    port = base_port
    # 自动探测可用端口（5001~5010）
    for p in range(base_port, base_port + 10):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((host, p)) != 0:
                port = p
                break
    else:
        print("[Monitor] ❌ 端口 5001-5010 全部占用，请手动释放")
        sys.exit(1)

    url = f"http://127.0.0.1:{port}"
    print(f"=" * 60)
    print(f"[Monitor] Web 监控面板: {url}")
    print(f"[Monitor] 按 Ctrl+C 停止")
    print(f"=" * 60)

    def open_browser():
        time.sleep(1.5)
        webbrowser.open_new_tab(url)
        print(f"[Monitor] 已尝试打开浏览器: {url}")

    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host=host, port=port, debug=False, use_reloader=False)
