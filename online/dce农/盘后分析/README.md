# DCE农产品盘后分析工具

支持品种: **A(豆一), B(豆二), C(玉米), CS(玉米淀粉), M(豆粕), Y(豆油), P(棕榈油), LH(生猪)**

## 📁 文件结构

```
/home/online/dce农/盘后分析/
├── config.json              # 配置文件（品种、合约、路径等）
├── dce农_analysis.py        # 核心分析模块
├── README.md                # 本文档
└── analysis/                # 分析结果输出目录（自动生成）
    ├── {date}/              # 日期文件夹
    │   ├── A/
    │   ├── B/
    │   ├── ...
    │   └── analysis/        # 合并后的分析数据
```

## 🚀 快速开始

### 1. 分析单品种单日

```python
from dce农_analysis import analyze_trading_day, analyze_P

# 完整函数调用
result = analyze_trading_day('P', '20260515', save_data=True, save_plot=True)

# 便捷函数
analyze_P('20260515')  # 棕榈油
```

### 2. 分析所有品种

```python
from dce农_analysis import analyze_all_symbols

results = analyze_all_symbols('20260515', save_data=True, save_plot=True)
```

### 3. 查看可用日期

```python
from dce农_analysis import list_available_dates

# 查看所有有数据的日期
dates = list_available_dates()
print(f"共有 {len(dates)} 个交易日")

# 查看特定品种有数据的日期
dates_p = list_available_dates('P')
```

### 4. 汇总历史绩效

```python
from dce农_analysis import summary_all_days

# 汇总所有日期所有品种
summary = summary_all_days()

# 汇总特定品种
summary_p = summary_all_days('P')

# 汇总日期范围
summary = summary_all_days('M', start_date='20260501', end_date='20260515')
```

## ⚙️ 配置说明 (config.json)

```json
{
    "symbols": ["A", "B", "C", "CS", "M", "Y", "P", "LH"],
    "data_paths": {
        "save_files_root": "/home/online/dce农/save_files",
        "tick_data_root": "/mnt/Data/future/decode_csv_dce",
        "main_data_path": "/mnt/Data/writable/liaoyuyang/data/1min/active"
    },
    "symbol_specs": {
        "A": {"name": "豆一", "multiplier": 10, "commission_rate": 0.0001, "min_price_tick": 1, "night_trading": true},
        ...
    }
}
```

## 📊 核心函数说明

### `analyze_trading_day(symbol, date, save_data=True, save_plot=True)`

完整分析指定品种和日期的交易数据，包括：
- 合并 factors 数据
- 合并 predictions 数据
- 加载交易状态
- 计算交易绩效
- 生成图表

### `merge_factors_data(symbol, date)`

合并所有 `factors_*.csv` 文件，每个文件取最后一行。

### `merge_predictions_data(symbol, date)`

合并所有 `predictions_*.csv` 文件，每个文件取最后一行。

### `load_trading_status(symbol, date)`

加载所有 `trading_status_*.json` 文件，提取交易状态。

### `summary_all_days(symbol=None, start_date=None, end_date=None)`

汇总历史交易绩效，返回 DataFrame。

## 📝 使用示例

### 单日分析

```python
from dce农_analysis import analyze_trading_day

# 分析某日交易
result = analyze_trading_day('P', '20260515')

# 输出:
# ============================================================
# 分析结果摘要:
# ============================================================
# 品种: P (棕榈油)
# 日期: 20260515
# Factors 数据: xxx 条
# Predictions 数据: xxx 条
# 交易状态: xxx 条
# 总盈亏: xxx.xx
# 最大盈利: xxx.xx
# 最大亏损: xxx.xx
# 交易次数: x
# 多头持仓: xx 分钟
# 空头持仓: xx 分钟
```

### 多品种对比

```python
from dce农_analysis import analyze_all_symbols
import pandas as pd

# 分析某日所有品种
results = analyze_all_symbols('20260515')

# 创建对比表格
comparison = pd.DataFrame([
    {
        '品种': r['symbol'],
        '名称': r['name'],
        '总盈亏': r.get('total_pnl', 0),
        '交易次数': r.get('num_trades', 0),
        '多头持仓': r.get('long_bars', 0),
        '空头持仓': r.get('short_bars', 0),
    }
    for r in results
])
print(comparison)
```

### 批量分析

```python
from dce农_analysis import analyze_all_symbols

# 分析多个日期
for date in ['20260512', '20260513', '20260514', '20260515']:
    print(f"\n{'#'*60}")
    print(f"# 日期: {date}")
    print(f"{'#'*60}")
    results = analyze_all_symbols(date, save_data=True, save_plot=True)
```

## 🔑 关键交易逻辑

**盈亏计算**:
- 多头: `(平仓价 - 开仓价) × 乘数`
- 空头: `(开仓价 - 平仓价) × 乘数`
- 手续费: `手数 × 价格 × 乘数 × 手续费率`

**数据来源**:
- 实盘数据: `/home/online/dce农/save_files/{date}/{symbol}/`
- trading_status_*.json: 持仓状态
- predictions_*.csv: 预测信号
- factors_*.csv: 因子数据

## ⚠️ 注意事项

1. **数据路径**: 确保 `config.json` 中的路径配置正确
2. **合约切换**: 换月时需要更新主策略的合约配置
3. **交易时间**: DCE 农产品交易时间
   - 白天: 09:00-11:30, 13:30-15:00
   - 夜盘: 21:00-23:00 (LH 生猪无夜盘)
4. **手续费**: 可在调用时覆盖默认配置

## 📞 问题反馈

如有问题，请检查：
1. 数据文件路径是否正确（`save_files/YYYYMMDD/SYMBOL/`）
2. 文件命名是否符合规范（`trading_status_*.json`, `predictions_*.csv`, `factors_*.csv`）
3. 数据列名是否符合预期（`now_pos`, `weighted`, `datetime` 等）