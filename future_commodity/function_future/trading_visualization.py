
import function_future.date_selection as ds
import function_future.DataLoader as DL
import threading
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display, clear_output
import ipywidgets as widgets
import pandas as pd
import numpy as np

class TradingVisualizationPager:
    def __init__(self, symbol, data, date_format='%H:%M', skip_weekends=True, trading_hours=None):
        """
        交易数据分页可视化工具 - 按交易日分页
        
        参数:
            data: 必须包含以下列的DataFrame:
                  - datetime: 时间戳
                  - date: 交易日日期
                  - factor: 因子值
                  - pos: 仓位
                  - close: 收盘价
                  - equity: 权益曲线
                  - pnl_ret: 收益率
                  - pnl_ret_cum: 累积收益率
            date_format: 时间显示格式
            skip_weekends: 是否跳过周末
            trading_hours: 交易时间列表，例如 ["09:00-11:30", "13:30-15:00", "21:00-1:00"]
                          如果为None，则从品种配置中获取
        """
        # 初始化参数
        self._initialize_parameters(symbol, data, date_format, skip_weekends, trading_hours)

        # 数据预处理
        self._prepare_data()
        self._calculate_total_pages_by_trading_day()
        
        # 创建控件
        self._create_widgets()
        self._create_manual_input()

        if 'datetime' in data.columns:
            data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')
        
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'], errors='coerce').dt.date

    def _initialize_parameters(self, symbol, data, date_format, skip_weekends, trading_hours=None):
        """初始化参数"""
        
        self.current_page = 0
        self.data = data.copy()
        self.symbol = symbol
        
        # 处理交易时间
        if trading_hours is not None:
            # 使用传入的交易时间
            self.trading_hours = self._parse_trading_hours(trading_hours)
        else:
            # 从品种配置获取交易时间
            self.config_future = DL.InstrumentConfig().get_instrument_config(symbol)
            self.trading_hours = self.config_future['trading_hours']
        
        # 生成交易时间序列
        trade_bars = ds.generate_trading_bars(sorted(data.date.dropna().unique()), self.trading_hours)
        self.data = self.data.set_index('datetime').reindex(trade_bars).reset_index(names='datetime')
        
        self.date_format = date_format
        self.skip_weekends = skip_weekends
        self.output_lock = threading.Lock()
    
    def _parse_trading_hours(self, trading_hours):
        """
        解析交易时间字符串列表，统一时间格式（补全小时为两位数）
        
        支持格式:
        - ["09:00-11:30", "13:30-15:00", "21:00-1:00"]  -> ["09:00-11:30", "13:30-15:00", "21:00-01:00"]
        - ["09:00-11:30", "13:30-15:00", "21:00-23:00"] -> ["09:00-11:30", "13:30-15:00", "21:00-23:00"]
        - ["09:00-11:30", "13:30-15:00", "21:00-2:30"]  -> ["09:00-11:30", "13:30-15:00", "21:00-02:30"]
        """
        if not trading_hours:
            return []
        
        result = []
        for th in trading_hours:
            if '-' in th:
                start, end = th.split('-', 1)
                start = self._format_time(start.strip())
                end = self._format_time(end.strip())
                result.append(f"{start}-{end}")
        
        return result
    
    def _format_time(self, time_str):
        """格式化时间字符串，确保小时为两位数"""
        parts = time_str.split(':')
        hour = parts[0].strip()
        minute = parts[1].strip() if len(parts) > 1 else "00"
        # 补零，确保小时是两位数
        hour = hour.zfill(2)
        return f"{hour}:{minute}"
    
    def _prepare_data(self):
        """数据预处理"""
        # 确保时间列是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(self.data['datetime']):
            self.data['datetime'] = pd.to_datetime(self.data['datetime'])
        
        # 确保日期列是日期类型
        if not pd.api.types.is_datetime64_any_dtype(self.data['date']):
            self.data['date'] = pd.to_datetime(self.data['date']).dt.date
        self.data['date'] = self.data['date'].ffill().bfill()
        # 创建时间字符串列（用于显示）
        self.data['ts_str'] = self.data['datetime'].dt.strftime(self.date_format)
        
        # 计算每日累计收益率（从0开始）
        if 'pnl_ret' in self.data.columns:
            # 按日期分组，计算每日累计收益率
            self.data['date_cum_ret'] = self.data.groupby('date')['pnl_ret'].cumsum()
        
        # 计算技术指标（如果不存在）
        if 'close' in self.data.columns and 'tick10avg' not in self.data.columns:
            self.data['tick10avg'] = self.data['close'].rolling(10, min_periods=1).mean()
        
        # 按datetime排序确保时间顺序正确
        self.data = self.data.sort_values('datetime').reset_index(drop=True)
        # self.data = self.data.dropna(subset=['factor'])
    
    def _calculate_total_pages_by_trading_day(self):
        """按交易日计算总页数"""
        # 获取所有唯一交易日并排序
        self.unique_trading_days = sorted(self.data['date'].unique())
        self.total_pages = len(self.unique_trading_days)
        
        if self.total_pages == 0:
            self.total_pages = 1
    
    def _create_widgets(self):
        """创建所有交互控件"""
        # 页码显示
        self.page_label = widgets.Label(value=f"页码: 1/{self.total_pages}")
        
        # 导航按钮
        self.prev_button = widgets.Button(
            description="← 上一交易日",
            layout=widgets.Layout(width='120px')
        )
        self.next_button = widgets.Button(
            description="下一交易日 →",
            layout=widgets.Layout(width='120px')
        )
        self.exit_button = widgets.Button(
            description="退出",
            button_style='danger',
            layout=widgets.Layout(width='80px')
        )
        
        # 绑定事件
        self.prev_button.on_click(self._on_prev)
        self.next_button.on_click(self._on_next)
        self.exit_button.on_click(self._on_exit)
        
        # 控件布局
        self.controls = widgets.HBox([
            self.prev_button,
            self.page_label,
            self.next_button,
            self.exit_button
        ], layout=widgets.Layout(justify_content='center'))
        
        # 输出区域
        self.output = widgets.Output()
    
    def _create_manual_input(self):
        """创建手动页码输入控件"""
        self.page_input = widgets.IntText(
            value=1,
            min=1,
            max=self.total_pages,
            description='跳至页码:',
            layout=widgets.Layout(width='200px')
        )
        self.jump_button = widgets.Button(
            description="跳转",
            layout=widgets.Layout(width='80px')
        )
        self.jump_button.on_click(self._on_jump)
        
        # 添加到控件栏
        self.controls.children = tuple(list(self.controls.children) + [
            self.page_input,
            self.jump_button
        ])
    
    def _on_prev(self, b):
        """上一页事件"""
        if self.current_page > 0:
            self.current_page -= 1
            self._update_display()
    
    def _on_next(self, b):
        """下一页事件"""
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self._update_display()
    
    def _on_jump(self, b):
        """跳转到指定页码"""
        try:
            target_page = self.page_input.value - 1
            if 0 <= target_page < self.total_pages:
                self.current_page = target_page
                self._update_display()
            else:
                with self.output:
                    print(f"⚠️ 请输入1到{self.total_pages}之间的页码")
        except Exception as e:
            with self.output:
                print(f"跳转错误: {str(e)}")
    
    def _on_exit(self, b):
        """退出可视化"""
        with self.output:
            clear_output()
            print("📊 交易数据可视化工具已关闭")
    
    def _get_current_page_data(self):
        """获取当前页数据（基于交易日分页）"""
        if self.total_pages == 0 or self.current_page >= len(self.unique_trading_days):
            return pd.DataFrame()
        
        # 获取当前页对应的交易日
        current_trading_day = self.unique_trading_days[self.current_page]
        
        # 筛选该交易日的所有数据（包括前一天的夜盘和当天的日盘）
        page_data = self.data[self.data['date'] == current_trading_day].copy()
        
        return page_data
    
    def _get_current_trading_day_info(self):
        """获取当前页的交易日信息"""
        if self.total_pages == 0:
            return "无数据", 0, 0, 0
        
        current_trading_day = self.unique_trading_days[self.current_page]
        day_data = self.data[self.data['date'] == current_trading_day]
        
        # 分析时间组成
        datetimes = pd.to_datetime(day_data['datetime'])
        time_components = {
            '夜盘': len(day_data[datetimes.dt.hour >= 21]),  # 21:00-23:00
            '上午盘': len(day_data[(datetimes.dt.hour >= 9) & (datetimes.dt.hour < 12)]),  # 09:00-11:30
            '下午盘': len(day_data[(datetimes.dt.hour >= 13) & (datetimes.dt.hour < 15)])  # 13:30-15:00
        }
        
        return current_trading_day, len(day_data), time_components
    
    def _calculate_symmetric_range(self, data, padding=0.1):
        """计算对称Y轴范围"""
        if data.empty or data.isna().all():
            return [-1, 1]
        
        valid_data = data.dropna()
        if len(valid_data) == 0:
            return [-1, 1]
            
        abs_max = max(abs(valid_data.max()), abs(valid_data.min()))
        if abs_max == 0:
            return [-1, 1]
            
        return [-abs_max * (1 + padding), abs_max * (1 + padding)]
    
    def _create_chart(self):
        """创建交易数据可视化图表"""
        page_data = self._get_current_page_data()
        
        if page_data.empty:
            # 创建空图表
            fig = go.Figure()
            fig.add_annotation(text="暂无数据", xref="paper", yref="paper", x=0.5, y=0.5, 
                             showarrow=False, font=dict(size=20))
            fig.update_layout(title="无数据可显示", height=600)
            return fig
        
        # 获取当前交易日信息
        current_trading_day, data_count, time_components = self._get_current_trading_day_info()
        
        # 计算价格波动幅度
        price_volatility_text = self._calculate_price_volatility(page_data)
        
        # 创建包含三个子图的图表
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=[
                "收益率和仓位", 
                f"价格和因子 | {price_volatility_text}",
                "账户权益和浮动盈亏"
            ],
            specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}]]
        )
        
        # ========== 第一子图：收益率和仓位 ==========
        self._add_returns_and_positions(fig, page_data)
        
        # ========== 第二子图：价格和因子 ==========
        self._add_prices_and_factors(fig, page_data)
        
        # ========== 第三子图：账户权益和浮动盈亏 ==========
        self._add_equity_and_margin(fig, page_data)
        
        # ========== 图表布局配置 ==========
        self._configure_chart_layout(fig, page_data, current_trading_day, data_count, time_components)
        
        return fig
    
    def _add_returns_and_positions(self, fig, page_data):
        """添加收益率和仓位子图"""
        # 收益率线（左侧Y轴）
        fig.add_trace(
            go.Scatter(
                x=page_data['ts_str'] if self.skip_weekends else page_data['datetime'],
                y=page_data['pnl_ret'],
                name='收益率',
                line=dict(color='#3B82F6', width=1.5),
                opacity=0.8,
                hovertemplate='时间: %{x}<br>收益率: %{y:.4f}<extra></extra>'
            ),
            row=1, col=1, secondary_y=False
        )
        
        # 累积收益率线（左侧Y轴）
        fig.add_trace(
            go.Scatter(
                x=page_data['ts_str'] if self.skip_weekends else page_data['datetime'],
                y=page_data['date_cum_ret'],
                name='累积收益率',
                line=dict(color='#F97316', width=1.5),
                opacity=0.8,
                hovertemplate='时间: %{x}<br>累积收益: %{y:.4f}<extra></extra>'
            ),
            row=1, col=1, secondary_y=False
        )
        
        # 仓位线（右侧Y轴）
        fig.add_trace(
            go.Scatter(
                x=page_data['ts_str'] if self.skip_weekends else page_data['datetime'],
                y=page_data['pos'],
                name='仓位',
                line=dict(color='#10B981', width=2),
                opacity=0.8,
                hovertemplate='时间: %{x}<br>仓位: %{y:.2f}<extra></extra>'
            ),
            row=1, col=1, secondary_y=True
        )
    
    def _add_prices_and_factors(self, fig, page_data):
        """添加价格和因子子图"""
        # 收盘价线（左侧Y轴）
        fig.add_trace(
            go.Scatter(
                x=page_data['ts_str'] if self.skip_weekends else page_data['datetime'],
                y=page_data['close'],
                name='收盘价',
                line=dict(color='#8B5CF6', width=2),
                opacity=0.8,
                hovertemplate='时间: %{x}<br>价格: %{y:.2f}<extra></extra>'
            ),
            row=2, col=1, secondary_y=False
        )
        
        # 10周期均价线（左侧Y轴）
        # if 'tick10avg' in page_data.columns:
        #     fig.add_trace(
        #         go.Scatter(
        #             x=page_data['ts_str'] if self.skip_weekends else page_data['datetime'],
        #             y=page_data['tick10avg'],
        #             name='10周期均价',
        #             line=dict(color='#0EA5E9', width=1.5),
        #             opacity=0.7,
        #             hovertemplate='时间: %{x}<br>均价: %{y:.2f}<extra></extra>'
        #         ),
        #         row=2, col=1, secondary_y=False
        #     )
        
        # 因子值线（右侧Y轴）
        fig.add_trace(
            go.Scatter(
                x=page_data['ts_str'] if self.skip_weekends else page_data['datetime'],
                y=page_data['factor'],
                name='因子值',
                line=dict(color='#EC4899', width=2),
                opacity=0.8,
                hovertemplate='时间: %{x}<br>因子: %{y:.4f}<extra></extra>'
            ),
            row=2, col=1, secondary_y=True
        )
    
    def _add_equity_and_margin(self, fig, page_data):
        """添加账户权益和保证金子图"""
        # 账户权益线（左侧Y轴）
        fig.add_trace(
            go.Scatter(
                x=page_data['ts_str'] if self.skip_weekends else page_data['datetime'],
                y=page_data['equity'],
                name='账户权益',
                line=dict(color='#047857', width=2),
                opacity=0.8,
                hovertemplate='时间: %{x}<br>权益: %{y:.2f}<extra></extra>'
            ),
            row=3, col=1, secondary_y=False
        )
        
        # 保证金率线（右侧Y轴）
        if 'Margin_rate' in page_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=page_data['ts_str'] if self.skip_weekends else page_data['datetime'],
                    y=page_data['Margin_rate'],
                    name='保证金率',
                    line=dict(color='#EF4444', width=2),
                    opacity=0.8,
                    hovertemplate='时间: %{x}<br>保证金率: %{y:.4f}<extra></extra>'
                ),
                row=3, col=1, secondary_y=True
            )
        
        # 浮动盈亏线（左侧Y轴）
        if 'Floating P&L' in page_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=page_data['ts_str'] if self.skip_weekends else page_data['datetime'],
                    y=page_data['Floating P&L'],
                    name='浮动盈亏',
                    line=dict(color='#F59E0B', width=1.5),
                    opacity=0.8,
                    hovertemplate='时间: %{x}<br>浮动盈亏: %{y:.2f}<extra></extra>'
                ),
                row=3, col=1, secondary_y=False
            )
    
    def _calculate_price_volatility(self, page_data):
        """计算价格波动率"""
        if page_data.empty or 'close' not in page_data.columns:
            return "价格波动: N/A"
        
        close_prices = page_data['close'].dropna()
        if len(close_prices) < 2:
            return "价格波动: N/A"
        
        high = close_prices.max()
        low = close_prices.min()
        mean = close_prices.mean()
        
        if mean != 0:
            price_volatility = (high - low) / mean
            return f"价格波动: {(price_volatility * 100):.2f}%"
        return "价格波动: N/A"
    
    def _configure_chart_layout(self, fig, page_data, current_trading_day, data_count, time_components):
        """配置图表布局"""
        # 时间范围标题
        time_range = self._get_time_range_text(page_data, time_components)
        
        fig.update_layout(
            title=dict(
                text=f"{self.symbol} 交易分析 | 第 {self.current_page + 1}/{self.total_pages} 页 | 交易日: {current_trading_day}",
                x=0.5,
                xanchor='center'
            ),
            height=600,
            template="plotly_white",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(t=80, b=50, l=60, r=60),
            font=dict(size=10)
        )
        
        # 配置Y轴范围
        self._configure_yaxis_ranges(fig, page_data)
        
        # X轴配置
        fig.update_xaxes(title_text="时间", row=3, col=1)
        
        # 添加网格线
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
    
    def _get_time_range_text(self, page_data, time_components):
        """获取时间范围文本"""
        if page_data.empty or 'datetime' not in page_data.columns:
            return "无数据"
        
        datetimes = pd.to_datetime(page_data['datetime'])
        start_time = datetimes.iloc[0].strftime('%H:%M')
        end_time = datetimes.iloc[-1].strftime('%H:%M')
        
        # 显示各时间段数据量
        time_info = f"夜盘:{time_components['夜盘']}条 上午:{time_components['上午盘']}条 下午:{time_components['下午盘']}条"
        
        return f"时间: {start_time}至{end_time} | 数据点: {len(page_data)} | {time_info}"
    
    def _configure_yaxis_ranges(self, fig, page_data):
        """配置Y轴范围"""
        # 第一子图左侧Y轴（收益率）
        if 'pnl_ret' in page_data.columns:
            fig.update_yaxes(
                title_text="收益率",
                range=self._calculate_symmetric_range(page_data['date_cum_ret']),
                row=1, col=1, secondary_y=False
            )
        
        # 第一子图右侧Y轴（仓位）
        fig.update_yaxes(
            title_text="仓位",
            range=[-1.5, 1.5],
            row=1, col=1, secondary_y=True
        )
        
        # 第二子图左侧Y轴（价格）
        if 'close' in page_data.columns and not page_data['close'].isna().all():
            close_prices = page_data['close'].dropna()
            if len(close_prices) > 0:
                price_min = close_prices.min()
                price_max = close_prices.max()
                price_range = price_max - price_min
                if price_range > 0:
                    price_padding = price_range * 0.1
                    fig.update_yaxes(
                        title_text="价格",
                        range=[price_min - price_padding, price_max + price_padding],
                        row=2, col=1, secondary_y=False
                    )
        
        # 第二子图右侧Y轴（因子值）
        if 'factor' in page_data.columns:
            fig.update_yaxes(
                title_text="因子值",
                range=self._calculate_symmetric_range(page_data['factor']),
                row=2, col=1, secondary_y=True
            )
        
        # 第三子图左侧Y轴（权益）
        if 'equity' in page_data.columns and not page_data['equity'].isna().all():
            equity_values = page_data['equity'].dropna()
            if len(equity_values) > 0:
                equity_min = equity_values.min()
                equity_max = equity_values.max()
                equity_range = equity_max - equity_min
                if equity_range > 0:
                    equity_padding = equity_range * 0.01
                    fig.update_yaxes(
                        title_text="权益/盈亏",
                        range=[equity_min - equity_padding, equity_max + equity_padding],
                        row=3, col=1, secondary_y=False
                    )
        
        # 第三子图右侧Y轴（保证金率）
        fig.update_yaxes(
            title_text="保证金率",
            range=[0, 1.2],
            row=3, col=1, secondary_y=True
        )
    
    def _update_display(self):
        """更新所有显示内容"""
        with self.output_lock:
            # 更新控件状态
            self.page_label.value = f"页码: {self.current_page + 1}/{self.total_pages}"
            self.page_input.value = self.current_page + 1
            self.prev_button.disabled = (self.current_page == 0)
            self.next_button.disabled = (self.current_page >= self.total_pages - 1)
            
            # 更新图表显示
            with self.output:
                clear_output(wait=True)
                try:
                    fig = self._create_chart()
                    display(fig)
                except Exception as e:
                    print(f"图表创建错误: {str(e)}")
                    # 显示错误信息但继续运行
                    import traceback
                    traceback.print_exc()
    
    def run(self):
        """启动交互式可视化"""
        display(self.controls)
        display(self.output)
        self._update_display()
        
        print(f"✅ 交易可视化工具已启动")
        print(f"📈 品种: {self.symbol}")
        print(f"📅 总交易日数: {self.total_pages}")
        print(f"📊 总数据点: {len(self.data)}")
        print(f"🎯 使用导航按钮或输入页码进行浏览")
