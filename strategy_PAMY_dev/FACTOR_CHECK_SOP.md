# 研究环境 vs 实时环境 — 因子核对 SOP

> 本文档维护核对因子的标准流程。每次核对前请先通读本文件，确保不遗漏步骤。

---

## 一、核对目标

确保 **实时环境**（test / prod）生成的因子值与 **研究环境**（research）计算的历史因子值在数值上保持一致。

核心原则：**暴露问题，不掩盖问题**。如果两边不一致，先排查数据管道或计算逻辑，而不是用 `ffill` / `reindex` 让两边"看起来一样"。

---

## 二、数据来源

| 环境 | 数据路径 | 格式 |
|------|---------|------|
| 实时（test） | `save_files/{SYMBOL}/factors/factors_*.csv` | CSV，通常 1 行（当前分钟） |
| 实时（test fallback） | `factor_cache/{DATE}/{SYMBOL}_fac_*.parquet` | Parquet，多行 |
| 研究 | `/mnt/Data/writable/liaoyuyang/factor/{SYMBOL}/all_fac/all_factor.feather` | Feather，全量历史 |

> **合约映射**：test 环境用 2605 合约回放，研究环境用对应品种的主力合约代码（如 `c2605`）。

---

## 三、核对工具

### 3.1 主要工具：`check_intraday copy.ipynb`

Jupyter Notebook，按 **时间（HH:MM:SS）** 对齐，并排显示原始数字。

**运行前准备**：
1. 确保实时测试环境已生成 `save_files/{SYMBOL}/factors/factors_*.csv`
2. 确保研究环境 feather 文件存在

**关键 Cell**：

| Cell | 名称 | 用途 |
|------|------|------|
| 7 | **原始值并排显示** | 选一个时间点，逐因子显示研究值/实时值/绝对差/相对差 |
| 8 | **逐分钟差异矩阵** | 统计每个因子在所有时间点的通过/失败比例 |
| 9 | **问题因子详情** | 列出差异比例高、完全不一致的因子，按分类排序 |
| 10 | **单因子走势对比** | 选一个因子画图，看两边时间序列是否贴合 |
| 12 | **交互式查看** | 修改 `VIEW_FACTOR` 和 `VIEW_TIME`，看任意因子前后 5 分钟 |

**日期映射规则**：
- `research_date = None`：自动在研究环境找与实时数据时间范围匹配的最近交易日
- `research_date = "2026-03-24"`：强制指定研究环境比对日期（回放历史数据时用）

### 3.2 快速脚本：`check_factor_cache.py`

直接对比 `factor_cache/*.parquet` 与研究 `all_factor.feather`，输出差异统计。

```bash
cd /home/strategy_PAMY_dev
python3 check_factor_cache.py
```

输出格式：
- 完全匹配因子数
- 完全不一致因子数及 Top10
- 差异比例结论（正常 / 有一定差异 / 差异很大）

---

## 四、核对阈值

```python
TOL_ABS = 0.01      # 绝对差异 < 0.01 视为一致
TOL_REL = 0.05      # 相对差异 < 5% 视为一致
```

**差异判断经验**：
- 完全不一致比例 **< 10%**：正常
- **10% ~ 30%**：有一定差异，需排查主要问题因子
- **> 30%**：差异很大，需检查数据对齐或因子计算逻辑

---

## 五、因子差异分类

发现差异时，先判断该因子属于哪一类，排查方向不同：

### 5.1 日级别因子（如 `day_jump`、`day_*`）
- 依赖日级开盘价、昨收等数据
- 夜盘场景下天然容易对不上（数据起点不同）
- **处理**：结合原始值人工判断，允许一定偏差

### 5.2 跨品种因子（如 `P_A_cvcorr10_diff`、`A_B_*`）
- 涉及两个品种的数据对齐
- 对 `ffill` 填充、交易时间差异敏感
- **处理**：检查两个品种的 `valid_index` 是否一致，是否有过期数据残留

### 5.3 分钟级因子（如 `FAC_bid_amount_sub20_*`）
- 基于 tick 数据降频计算
- **处理**：重点排查 tick 数据截断、rolling 窗口边界、降频方法实现

---

## 六、核对红线（绝对不能做）

1. **不能掩盖缺失**：禁止用 `ffill` / `bfill` / `reindex(...).ffill()` 让实时和研究"看起来一致"。
2. **错位要暴露**：如果 `shift()` 因缺失分钟而错位，diff 变大是对的，说明要修 `data_service` / 数据源，而不是修计算层去容错。
3. **不能用过期数据填充**：实时 tick 数据截断后，rolling 窗口的边界处理必须和研究环境一致，不能用默认值或常数填充。

---

## 七、常见问题排查流程

### 7.1 某个因子完全对不上（diff 100%）

```
Step 1: 确认该因子属于哪一类（日级 / 跨品种 / 分钟级）
Step 2: 用 Notebook Cell 7 看原始值，判断是"量级差异"还是"符号相反"
Step 3: 检查实时 save_files 中该因子的数值分布（是否全是同一个值 / NaN / 异常大）
Step 4: 检查 data_function.py 中该因子的降频方法（agg_method 是否被正确处理）
Step 5: 检查 _tick_indexed 中依赖的列（如 M_std、bvall、corrBidwap）是否存在且计算正确
```

### 7.2 多个因子值完全相同（严重 bug）

**现象**：完全不同的因子（如 `Mstdwap`、`askdommean`、`corrBidwap`）在某些时间点值完全一样。

**排查方向**：
1. 检查 `resample_agg` 返回的数组是否被错误复用（如同一个 `np.array` 被赋给多个列）
2. 检查 `_prepare_data` 中 `pd.concat` 是否因为索引不对齐导致 `extra_cols` 全为 NaN
3. 检查 `tick_data` 截断长度（`data_service.py` 中 `iloc[-1000:]`）是否导致 rolling 窗口边界异常
4. 打印 `weight_sum` 和 `weighted_sum`，确认是否走到了某个 fallback 分支

### 7.3 大量 NaN

**现象**：某个因子 30% 以上时间为 NaN。

**排查方向**：
1. 检查 `resample_agg` 开头的 `len(tick_fac) <= 480` 判断（ tick 数不足时返回全 NaN）
2. 检查降频方法中的权重是否全为 0（如 `bvall < avall` 从未满足）
3. 检查 `_tick_indexed` 中依赖列（如 `M_std`、`corrBidwap`）是否因 `min_periods=100` 而大量 NaN

### 7.4 数值尺度对不上

**现象**：实时值是 -16765，研究值是 -0.03。

**排查方向**：
1. 确认截图/表格中的值是**原始因子值**还是**标准化后**的值（zscore / rank）
2. 如果研究环境经过标准化，实时环境也需要用同样的标准化参数
3. 检查 `money_flow_b.diff()` 的基准点是否一致（实时截断 tick 后，diff 的基准不同）

---

## 八、手动验证脚本（必要时使用）

当 Notebook 和 `check_factor_cache.py` 无法定位问题时，用以下脚本手动复现计算：

```python
import pandas as pd
import numpy as np
import sqlite3

# 1. 读取实时 tick 数据
conn = sqlite3.connect('/home/strategy_PAMY_dev/tick_data.db')
df = pd.read_sql("SELECT * FROM tick_data_c2605 WHERE datetime LIKE '2026-05-21 22:47%' ORDER BY datetime", conn)
conn.close()

df['datetime'] = pd.to_datetime(df['datetime'])
df = df.set_index('datetime')

# 2. 计算依赖列（和数据函数中的逻辑保持一致）
df['bvall'] = df[[f'bid_volume{i}' for i in range(1, 6)]].sum(axis=1)
df['avall'] = df[[f'ask_volume{i}' for i in range(1, 6)]].sum(axis=1)
df['mid_price'] = (df['bid_price1'] + df['ask_price1']) / 2
df['M_std'] = df['mid_price'].rolling(120, min_periods=100).std()
df['corrBidwap'] = df['bvall'].diff().rolling(120, min_periods=100).corr(df['mid_price'])

# 3. 计算因子原始值
money_flow_b = (df['bid_volume1'] * df['bid_price1'] + ...).sum(axis=1)  # 补全五档
df['factor_value'] = money_flow_b.diff()

# 4. 按 resample_agg 中的逻辑计算分钟值
grouper = pd.Grouper(freq='1min', label='right', closed='right')
# ... 按具体降频方法计算
```

---

## 九、已知陷阱（按发现时间倒序）

### 2026-05-21 — `FAC_DBCD_skewness` 部分 bar 不一致

**根因**：实时 `calc_recent_data.py` 与研究环境 `function_future/DataLoader.py` 的 **日期归属逻辑不同**。
- 研究环境 `level2_all` 中，`night` 数据的 `datetime` 被映射到 **前一个交易日**（`date_to_prev_date`），但 `trade_date` 保持原始文件日期。
- 实时 `_tick.csv` 只保存 `date_lst[-tick_cache_days:]` 的数据，`night` 数据因 `datetime` 被映射到前一天，导致 `2026-03-23` 的 `_tick.csv` 只包含 `light`（白天盘），而 `level2_all` 包含 `light + night`。
- `FAC_DBCD` 依赖 `mid_price` 的 rolling window，白天盘开盘初期的 `SMA` 会受到前一日 `night` 数据是否存在于同一张表的影响，从而在某些 bar 产生 skewness 差异。

**结论**：核对时若发现仅白天盘早期 bar（如 09:01、14:13）有差异，需检查 `night` 数据日期映射是否一致。

### 2026-05-21 — `mid_price` 精度丢失

**根因**：`parse_df` 重新计算 `mid_price` 时缺少 `.round(4)`，覆盖 `_tick.csv` 中已 round 的值。
- 研究环境 `save/data_function.py`（第 3287 行）和 `function_future/DataLoader.py`（第 618 行）均有 `.round(4)`。
- 实时 `strategy_PAMY_dev/data_function.py` 之前缺失，已修复。
- 影响因子：`FAC_MFI`（`TP` 阈值判断）、`FAC_ask_amount_sub20_Mstdwap`（`M_std` 权重）等。

### 2026-05-21 — `datetime` 去重粒度不一致

**根因**：研究环境 `DataLoader.py` 将 `datetime` 截断到 **10ms**（`str[:-4]`）后再 `drop_duplicates`；实时环境之前使用微秒精度去重。
- 同一 10ms 窗口内的多个 tick 在研究环境会被压缩为 `keep='last'` 一条，实时环境之前保留全部。
- 影响因子：分布敏感型统计量（`skewness`、`kurtosis`、`std`）。
- 已同步到 `online/dce农/calc_recent_data.py`。

### 2026-05-21 — `last_twap` 逻辑差异

**根因**：`strategy_PAMY_dev` 中实验性将 `last_twap` 改为 `np.where(second > 5, last_price, nan)`，但研究环境始终使用 `last_price`。
- **Prod (online) 保持 `last_price`**，与研究环境一致。
- `strategy_PAMY_dev` 的 `parse_df` 若使用 `np.where(second > 5, ...)` 会覆盖 `_tick.csv` 中的 `last_price`，导致 `last_twap` 列与研究环境不一致。

---

## 十、修改记录

| 日期 | 修改内容 | 修改人 |
|------|---------|--------|
| 2026-05-21 | 创建本文档 | Kimi |
| 2026-05-21 | 补充已知陷阱：日期映射、mid_price 精度、10ms 去重、last_twap | Kimi |

---

## 十一、路径速查

| 用途 | 路径 |
|------|------|
| 核对 Notebook | `/home/strategy_PAMY_dev/check_intraday copy.ipynb` |
| 快速核对脚本 | `/home/strategy_PAMY_dev/check_factor_cache.py` |
| 实时 factors | `/home/strategy_PAMY_dev/save_files/{SYMBOL}/factors/factors_*.csv` |
| 实时 tick DB | `/home/strategy_PAMY_dev/tick_data.db` |
| 研究 feather | `/mnt/Data/writable/liaoyuyang/factor/{SYMBOL}/all_fac/all_factor.feather` |
| 策略逻辑 | `/home/strategy_PAMY_dev/strategies.py`、`data_function.py` |
| DataService | `/home/strategy_PAMY_dev/data_service.py` |
