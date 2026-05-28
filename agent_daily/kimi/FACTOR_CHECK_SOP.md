# 因子核对标准操作流程（SOP）

> 本文档描述"研究环境 ↔ 实盘环境"因子核对的完整流程，包含两边框架的架构、数据流、核对工具使用方法。每次开新对话框时请先通读本文件。

---

## 一、总体架构

整个系统分为三条线：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           研 究 环 境                                    │
│  （历史回测、模型训练、因子研究的"真相源"）                                │
├─────────────────────────────────────────────────────────────────────────┤
│  原始 tick CSV  →  calc_factors_dce农.py  →  m2m/t2m/t2t 因子文件       │
│       ↓                                                                     │
│  make_main_factor_dataframe_dce农.py  →  all_factor.feather（全量历史）  │
└─────────────────────────────────────────────────────────────────────────┘
                                      ↑
                                      │ 核对
                                      ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                           实 盘 环 境                                    │
│  （分为：历史补算部分 + 实时运行部分）                                    │
├─────────────────────────────────────────────────────────────────────────┤
│  【历史补算】calc_recent_data.py                                        │
│    - 从原始 tick CSV（与研究环境同源）读取                                │
│    - 预处理 → 降频成 1min → 输出 _min.csv / _tick.csv                   │
│    - 用于策略启动时加载"历史前序因子"                                    │
├─────────────────────────────────────────────────────────────────────────┤
│  【实时运行】orchestrator.py + strategies.py                            │
│    - DataService 从 SQLite tick_data.db 接收实时/模拟 tick                │
│    - 每新分钟触发：打包 tick(最近1000条) + min(全量) → 策略进程           │
│    - Factor_generator 计算当前分钟因子 → 模型预测 → 交易信号              │
│    - 同时保存 factors_*.csv（供核对用）                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 二、研究环境详细流程

### 2.1 原始数据

- **路径**：`/mnt/Data/future/decode_csv_{exchange}/{year}/{month}/{day}/{light|night}/{instrument}`
- **格式**：CSV，列包含 update_time, last_price, bid/ask price1-5, bid/ask volume1-5, volume, turnover 等
- **说明**：这是从交易所接收的 Level2 原始数据，研究环境和 `calc_recent_data.py` 都从这里读取

### 2.2 中间数据（预处理后）

- **1分钟线**：`/mnt/Data/writable/liaoyuyang/data/1min/{symbol}/{instrument}.feather`
- **tick级数据**：`/mnt/Data/writable/liaoyuyang/data/level2_all/{symbol}/{instrument}.feather`
- **说明**：由独立的数据预处理管道生成，不是本文档重点

### 2.3 因子计算（calc_factors_dce农.py）

研究环境的因子分三类计算，分别对应三个函数：

| 类型 | 函数 | 输入数据 | 输出路径 | 说明 |
|------|------|---------|---------|------|
| **m2m** (分钟-分钟) | `calc_m2m(instrument)` | `1min/{instrument}.feather` | `factor/{symbol}/m2m/` | 只用分钟线计算的因子，如JC、ZCpriceinterval |
| **t2m** (tick-分钟) | `calc_t2m(instrument)` | `level2_all/{instrument}.feather` + 1min | `factor/{symbol}/t2m/` | 用tick数据但输出分钟频率，如lastprice_bias1 |
| **t2t** (tick-tick) | `calc_t2t(instrument)` | `level2_all/{instrument}.feather` | `factor/{symbol}/t2t/{instrument}/` | 先在tick频率计算原始值，再降频到1min |

**t2t 因子的特殊降频逻辑**：
1. `factor_generator.calculate_tick_tick_factor(fac)` 在 tick 频率计算原始因子值
2. 然后将 tick 数据 `resample('1min', label='right', closed='right')` 降频
3. 对每分钟的 tick 做统计：mean/std/min/max/last/first/vwap/upmean/downmean/corrAskwap/corrBidwap/Mstdwap/...等
4. 每个统计量成为一个独立因子（如 `FAC_bid_amount_sub20_Mstdwap`）

### 2.4 合并为 all_factor.feather

**文件**：`make_main_factor_dataframe_dce农.py`

流程：
1. 读取某品种所有合约的 m2m、t2m、t2t 因子
2. 按 (datetime, instrument) 索引 join 合并
3. 加入跨品种因子 `mixfac`（如 `P_A_cvcorr10_diff`）
4. 输出为 `all_factor.feather`

**最终文件**：`/mnt/Data/writable/liaoyuyang/factor/{symbol}/all_fac/all_factor.feather`

---

## 三、实盘环境详细流程

实盘环境分为**历史补算**和**实时运行**两部分，两者使用同一套因子计算逻辑（`data_function.py` 中的 `Factor_generator`），但输入数据来源不同。

### 3.1 历史补算：calc_recent_data.py

**作用**：每天盘前/盘后补算最近 N 天的历史因子，供策略启动时加载"历史前序数据"。

**流程**：

```
读取 config.json
  ↓
确定需要处理的合约列表（instruments）和日期范围（recent_days）
  ↓
对每个合约、每个交易日：
  1. load_data(instrument, date) 
     - 从 /mnt/Data/future/decode_csv_... 读取原始 tick CSV
     - pre_resample_data() 预处理：
       * process_time_column_vectorized_floor()：时间向上取整到0.25秒
       * process_datetime()：处理夜盘日期归属（夜盘归到下一个交易日）
       * df_is_trading_time()：过滤非交易时段
       * 数值清洗（超大值→NA）
       * 计算 volume/turnover 差分（去重前）
       * drop_duplicates(subset=['datetime'], keep='last')：同一微秒保留最后一条
       * 计算 mid_price, spread, high/low/open/close
       * 计算 last_twap（排除前5秒）
       * 计算 bar_count（每分钟的tick序号）
  2. min_data = df.groupby('ts').agg(agg_dict)
     - agg_dict 定义了降频方式：
       * open:first, high:max, low:min, close:last
       * volume:sum, turnover:sum
       * last_twap:mean, mid_price:mean
       * open_interest:last, bar_count:last
  3. 过滤 bar_count>1 的分钟（去掉只有1条tick的分钟）
  4. 保存：{instrument}_min.csv + {instrument}_tick.csv
```

**关键注意点**：
- `calc_recent_data.py` 和研究环境读取**同源的原始 tick CSV**
- 但预处理逻辑有差异：研究环境在 `calc_factors_dce农.py` 中有自己的清洗逻辑，而 `calc_recent_data.py` 是为了复现实盘逻辑而写的
- **核对的第一个层次**：确保 `calc_recent_data.py` 生成的分钟线和研究环境的分钟线一致

### 3.2 实时运行：orchestrator.py

**整体流程**：

```
启动 orchestrator.py
  ↓
DataService 初始化
  - 连接 SQLite tick_data.db（ZMQ 数据由 sql_writer_dce.py 写入）
  - tick_limit=50000：每次从DB读取的最大tick数
  ↓
主循环（每分钟触发一次）：
  1. DataService._refresh_instrument(inst)
     - 从 tick_data.db 读取新到的 tick
     - 更新 tick_cache[inst]（内存中缓存所有历史tick）
     - 重新聚合分钟线 → min_cache[inst]
  2. 检测到新分钟 → _package_data(current_time)
     - tick: tick_cache[inst] 中最近1000条（≤ current_time）
     - min: min_cache[inst] 全量
  3. 8个品种并行执行策略（ProcessPoolExecutor）
     - 每个子进程运行 BaseStrategy.compute(data_package)
  4. 主进程接收结果 → run_logic() → 保存结果
```

**BaseStrategy.compute() 内部流程**：

```
_extract_data(data_package) 
  → 提取 {main}_tick, {main}_min_data_concat, {other}_min_data_concat
  ↓
_align_and_check() 
  → 数据时间对齐检查
  ↓
Factor_generator(tick, main_min, other_mins...)
  → _load_data_tick(tick)
  → _load_data_min(main_min, other_mins)
    → _tick_indexed = tick_data.set_index('datetime')
    → valid_index = 交易时段内的分钟索引
    → 实时模式下（tick<5000条）：valid_index 只保留最后一个点
  ↓
generate_factor(fac_generator, valid_index, factor_col)
  → 调用 generate_factor_dataframe_{SYMBOL}()
  → 逐个调用 fac_generator.FAC_XXX(agg_method)
    → 计算 tick 级原始值 → resample_agg() 降频到 1min
  → 返回 fac_df（当前分钟的因子值）
  ↓
模型预测（5-fold LightGBM）
  → 返回 pred_df
```

**关键注意点**：
- 实时模式下，`data_package['tick']` **只包含最近 1000 条 tick**（`iloc[-1000:]`）
- `valid_index` 在实时模式下**只保留最后一个点**（当前分钟）
- 但 `_tick_indexed` 包含这 1000 条 tick 的全部数据，用于 rolling 窗口计算
- **核对的第二个层次**：实时生成的当前分钟因子值，与 `calc_recent_data.py` 生成的历史因子值是否一致

---

## 四、研究环境 vs 实盘环境的对比方法

### 4.1 对比层级

因子核对分为**两个层级**：

| 层级 | 对比对象 | 工具/方法 | 目的 |
|------|---------|----------|------|
| **L1** | `calc_recent_data.py` 输出 vs 研究环境 `all_factor.feather` | `check_past_data.ipynb` | 确认历史补算逻辑与研究环境一致 |
| **L2** | 实时 `factors_*.csv` vs `calc_recent_data.py` 输出 / 研究环境 | `check_intraday copy.ipynb` | 确认实时计算逻辑与历史补算一致 |

**为什么要分两层？**
- 如果 L1 就不一致，说明 `calc_recent_data.py` 的预处理/降频逻辑与研究环境有差异，需要先修 L1
- 如果 L1 一致但 L2 不一致，说明问题出在实时运行的数据截断（1000条tick）或 Factor_generator 的计算逻辑

### 4.2 核对工具 1：check_intraday copy.ipynb

**路径**：`/home/strategy_PAMY_dev/check_intraday copy.ipynb`

**用途**：将 `save_files/{symbol}/factors_*.csv` 与研究环境 `all_factor.feather` 按**时间（HH:MM:SS）**对齐，并排显示原始数字。

**配置区（Cell 1）**：
```python
symbol = "P"           # 品种: P, A, M, Y, C, CS, B, LH
research_date = None   # None=自动匹配；回放时手动指定如 "2026-03-24"
night_date = "2026-03-23"   # 夜盘在研究环境中的日历日期
day_date = "2026-03-24"     # 日盘在研究环境中的日历日期
```

**关键 Cell**：

| Cell | 名称 | 用途 |
|------|------|------|
| 1 | **配置区** | 修改 symbol、research_date、night_date、day_date |
| 7 | **原始值并排显示** | 选一个时间点，逐因子显示研究值/实时值/绝对差/相对差 |
| 8 | **逐分钟差异矩阵** | 统计每个因子在所有时间点的通过/失败比例 |
| 9 | **问题因子详情** | 列出差异比例高、完全不一致的因子，按分类排序 |
| 10 | **单因子走势对比** | 选一个因子画图，看两边时间序列是否贴合 |
| 12 | **交互式查看** | 修改 VIEW_FACTOR 和 VIEW_TIME，看任意因子任意分钟 |
| 17 | **预测值对比** | 加载5-fold模型，对比研究环境和实盘的预测值差异 |

**日期映射规则**：
- 研究环境的 `all_factor.feather` 中，夜盘数据（21:00-23:00）的 `trade_date` 归属**下一个交易日**
- 例如：2026-03-23 21:00 的夜盘，在 feather 中的 trade_date 是 2026-03-24
- Notebook 中需要分别指定 `night_date` 和 `day_date`

### 4.3 核对工具 2：check_factor_cache.py

**路径**：`/home/strategy_PAMY_dev/check_factor_cache.py`

**用途**：直接对比 `factor_cache/*.parquet` 与研究 `all_factor.feather`，输出差异统计。

```bash
cd /home/strategy_PAMY_dev
python3 check_factor_cache.py
```

**输出格式**：
- 完全匹配因子数
- 完全不一致因子数及 Top10
- 差异比例结论

**注意**：factor_cache 是按交易日拆分的 parquet，由 `_save_fac_cache()` 生成。

### 4.4 核对工具 3：check_past_data.ipynb

**路径**：`/home/strategy_PAMY_dev/check_past_data.ipynb`

**用途**：对比 `calc_recent_data.py` 生成的历史数据与研究环境。

---

## 五、核对阈值与判定标准

```python
TOL_ABS = 0.01      # 绝对差异 < 0.01 视为一致
TOL_REL = 0.05      # 相对差异 < 5% 视为一致
```

**差异判断经验**：
- 完全不一致比例 **< 10%**：正常
- **10% ~ 30%**：有一定差异，需排查主要问题因子
- **> 30%**：差异很大，需检查数据对齐或因子计算逻辑

---

## 六、因子差异分类与排查方向

发现差异时，先判断该因子属于哪一类，排查方向不同：

### 6.1 日级别因子（如 `day_jump`、`day_*`）
- 依赖日级开盘价、昨收等数据
- 夜盘场景下天然容易对不上（数据起点不同）
- **处理**：结合原始值人工判断，允许一定偏差

### 6.2 跨品种因子（如 `P_A_cvcorr10_diff`、`A_B_*`）
- 涉及两个品种的数据对齐
- 对 `ffill` 填充、交易时间差异敏感
- **处理**：检查两个品种的 `valid_index` 是否一致，是否有过期数据残留

### 6.3 分钟级因子（如 `FAC_bid_amount_sub20_*`）
- 基于 tick 数据降频计算
- **处理**：重点排查 tick 数据截断、rolling 窗口边界、降频方法实现

---

## 七、核对红线（绝对不能做）

1. **不能掩盖缺失**：禁止用 `ffill` / `bfill` / `reindex(...).ffill()` 让实时和研究"看起来一致"。
2. **错位要暴露**：如果 `shift()` 因缺失分钟而错位，diff 变大是对的，说明要修 `data_service` / 数据源，而不是修计算层去容错。
3. **不能用过期数据填充**：实时 tick 数据截断后，rolling 窗口的边界处理必须和研究环境一致，不能用默认值或常数填充。

---

## 八、常见问题排查流程

### 8.1 某个因子完全对不上（diff 100%）

```
Step 1: 确认该因子属于哪一类（日级 / 跨品种 / 分钟级）
Step 2: 用 Notebook Cell 7 看原始值，判断是"量级差异"还是"符号相反"
Step 3: 检查实时 save_files 中该因子的数值分布（是否全是同一个值 / NaN / 异常大）
Step 4: 检查 data_function.py 中该因子的降频方法（agg_method 是否被正确处理）
Step 5: 检查 _tick_indexed 中依赖的列（如 M_std、bvall、corrBidwap）是否存在且计算正确
```

### 8.2 多个因子值完全相同（严重 bug）

**现象**：完全不同的因子（如 `Mstdwap`、`askdommean`、`corrBidwap`）在某些时间点值完全一样。

**排查方向**：
1. 检查 `resample_agg` 返回的数组是否被错误复用（如同一个 `np.array` 被赋给多个列）
2. 检查 `_prepare_data` 中 `pd.concat` 是否因为索引不对齐导致 `extra_cols` 全为 NaN
3. 检查 `tick_data` 截断长度（`data_service.py` 中 `iloc[-1000:]`）是否导致 rolling 窗口边界异常
4. 打印 `weight_sum` 和 `weighted_sum`，确认是否走到了某个 fallback 分支

### 8.3 大量 NaN

**现象**：某个因子 30% 以上时间为 NaN。

**排查方向**：
1. 检查 `resample_agg` 开头的 `len(tick_fac) <= 480` 判断（ tick 数不足时返回全 NaN）
2. 检查降频方法中的权重是否全为 0（如 `bvall < avall` 从未满足）
3. 检查 `_tick_indexed` 中依赖列（如 `M_std`、`corrBidwap`）是否因 `min_periods=100` 而大量 NaN

### 8.4 数值尺度对不上

**现象**：实时值是 -16765，研究值是 -0.03。

**排查方向**：
1. 确认截图/表格中的值是**原始因子值**还是**标准化后**的值（zscore / rank）
2. 如果研究环境经过标准化，实时环境也需要用同样的标准化参数
3. 检查 `money_flow_b.diff()` 的基准点是否一致（实时截断 tick 后，diff 的基准不同）

---

## 九、路径速查

| 用途 | 路径 |
|------|------|
| 核对 Notebook（主要） | `/home/strategy_PAMY_dev/check_intraday copy.ipynb` |
| 核对 Notebook（历史） | `/home/strategy_PAMY_dev/check_past_data.ipynb` |
| 快速核对脚本 | `/home/strategy_PAMY_dev/check_factor_cache.py` |
| 实时 factors | `/home/strategy_PAMY_dev/save_files/{SYMBOL}/factors/factors_*.csv` |
| 实时 factor_cache | `/home/strategy_PAMY_dev/factor_cache/{trade_date}/{symbol}_fac_*.parquet` |
| 实时 tick DB | `/home/strategy_PAMY_dev/tick_data.db` |
| 研究 all_factor | `/mnt/Data/writable/liaoyuyang/factor/{SYMBOL}/all_fac/all_factor.feather` |
| 研究 m2m 因子 | `/mnt/Data/writable/liaoyuyang/factor/{SYMBOL}/m2m/` |
| 研究 t2m 因子 | `/mnt/Data/writable/liaoyuyang/factor/{SYMBOL}/t2m/` |
| 研究 t2t 因子 | `/mnt/Data/writable/liaoyuyang/factor/{SYMBOL}/t2t/{instrument}/` |
| calc_recent_data.py | `/home/strategy_PAMY_dev/calc_recent_data.py` |
| 策略逻辑 | `/home/strategy_PAMY_dev/strategies.py`、`data_function.py` |
| DataService | `/home/strategy_PAMY_dev/data_service.py` |
| orchestrator | `/home/strategy_PAMY_dev/orchestrator.py` |

---

## 十、新增调试工具（2026-05-22）

### 10.1 factor_trigger_logger.py — 源数据捕获器

**路径**：`/home/strategy_PAMY_dev/factor_trigger_logger.py`

**用途**：当发现某个因子（如 `RPP_5D`）实时与研究环境差异大时，**捕获每次 trigger 用到的完整源数据**，用于事后复现对比。

**输出结构**（每分钟一个文件夹，用市场时间命名）：
```
logs/factor_debug/P/20260323_225700/
  tick.parquet          ← 主品种 tick_data（Factor_generator.tick_data）
  min.parquet           ← 主品种 min_data（Factor_generator.min_data）
  valid_index.csv       ← valid_index 时间序列
  min_M.parquet         ← 跨品种 M 的 min_data
  min_A.parquet         ← 跨品种 A 的 min_data
  ...（按 dict_keys 顺序）
  result.csv            ← 该因子计算结果
  info.json             ← 仅：trigger_time（市场时间）、symbol、factor、dict_keys
```

**关键规则**：
- 文件夹名 = `valid_index[-1]` 的市场时间（`%Y%m%d_%H%M%S`）
- **不写系统时间**、不写行数统计、不写 result_tail/mean/std
- 只存实际数据表（parquet/csv）和极简 info.json

**启用方式**（在 `strategies.py` 的 `compute()` 中插入）：
```python
if self.main_symbol == 'P' and 'RPP_5D' in fac_df.columns:
    from factor_trigger_logger import make_logger
    _logger = make_logger(symbol='P', target_factor='RPP_5D')
    _logger.log('RPP_5D', fac_generator, fac_df['RPP_5D'].values, data_dict=data_dict)
```

### 10.2 extract_research_data.py — 研究环境同源数据提取

**路径**：`/home/strategy_PAMY_dev/extract_research_data.py`

**用途**：从研究环境提取与实时 logger **同分钟**的源数据，用于并排对比。

**用法**：
```bash
python extract_research_data.py 2026-03-23_22:57:00 P
```

**输出**：`logs/factor_debug/research/20260323_225700/`
- `min.parquet` — 从 `1min/{contract}.feather` 提取的该分钟前后 7 天 min 数据
- `factor_row.parquet` — 从 `all_factor.feather` 提取的该分钟因子值
- `info.json`

### 10.3 排查流程（当某个因子差异大时）

**Step 1**：确认差异模式
- 打开 `check_intraday copy.ipynb` 或 `check_factor_realtime.ipynb`
- 看 diff 表：是**所有分钟**都有差异，还是**前面分钟**有差异、最后一秒没差异？
- 如果是后者 → 高度怀疑 **DataService 打包的数据在历史边界处有问题**（如 tick 截断、min 拼接断层）

**Step 2**：启用 factor_trigger_logger
- 在 `strategies.py` 的 `compute()` 中插入 logger（见 10.1）
- 重启 orchestrator，跑回放

**Step 3**：提取研究环境数据
- 用 `extract_research_data.py` 提取有差异的分钟

**Step 4**：对比
- 用 notebook 读取实时 `tick.parquet` vs 研究 `min.parquet`
- 对比 high/low/close/open 是否一致
- 如果源数据一致但结果不同 → `data_function.py` 中该因子的计算逻辑有问题
- 如果源数据不一致 → DataService / sql_writer / 历史 CSV 有问题

**Step 5**：复现
- 在 notebook 中手动用捕获的数据调用 `RPP_5D(window=1200)`
- 确认是否能复现实时结果

---

## 十一、本次核对关键发现（2026-05-22）

### 发现 1：P 品种 `RPP_5D` 前面分钟有差异，最后一秒（23:00:00）完全一致

**现象**：
- `check_factor_realtime.ipynb` 中，22:57~22:59 的 `RPP_5D` 有差异
- 23:00:00 的 `RPP_5D` diff = 0

**推断**：
- `RPP_5D` 使用 `rolling(window=1200)`（5天），极度依赖历史 min_data
- 最后一分钟没差异，说明**历史数据的尾部是正确的**
- 前面分钟有差异，说明**DataService 在历史边界处的 min_data 拼接或 tick 截断有问题**
- 可能是 `min_cache` 中历史 CSV（到 15:00）和实时 tick 聚合（21:00 开始）之间存在 gap 或重复

**待验证**：
- 用 logger 捕获 22:57 的 min.parquet，检查 `high`/`low` 的 rolling(window=1200) 边界
- 对比研究环境同分钟的 min 数据

### 发现 2：系统时间在项目中无意义

**教训**：
- `sql_writer_dce.py` 中 `parse_time_from_update_time` 使用 `datetime.now()` 作为日期
- 回放时市场时间 ≠ 系统时间
- 所有调试日志、文件命名、对比都必须用 **市场时间（valid_index[-1]）**
- `factor_trigger_logger.py` 已修正：文件夹名用 `valid_index[-1]`，不写系统时间

---

## 十二、修改记录

| 日期 | 修改内容 | 修改人 |
|------|---------|--------|
| 2026-05-21 | 创建本文档，整合策略流程、研究框架、实盘框架、核对工具 | Kimi |
| 2026-05-22 | 新增 factor_trigger_logger.py、extract_research_data.py；修正 logger 用市场时间；记录 P 品种 RPP_5D 排查发现 | Kimi |
