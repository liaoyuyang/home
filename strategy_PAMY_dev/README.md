# strategy_PAMY_dev — DCE 期货实时交易测试环境

> **定位**：P/A/M/Y/C/CS/B/LH 八品种期货量化策略的**完全自包含**实时交易框架。
> 与原项目隔离：独立 DB、独立模型、独立 ZMQ 端口（44225~44232）。

---

## 一、启动方式

```bash
cd /home/strategy_PAMY_dev

# 终端 1：行情写入
python sql_writer_dce.py

# 终端 2：策略编排（-B 禁用缓存，-u 无缓冲输出）
python3 -B -u orchestrator.py
# 可选只跑部分品种：python3 -B -u orchestrator.py --symbols P M
```

- 检测到 `15:00` 自动退出主循环，保存缓存
- `sql_writer_dce.py` 需手动 `Ctrl+C` 停止

---

## 二、架构 4 层

```
交易所 ZMQ 行情
        │
        ▼
┌─────────────────┐    ┌─────────────────┐    ┌──────────────────────────────┐
│ sql_writer_dce  │───→│   tick_data.db  │───→│ DataService (独立线程, 0.5s) │
│  (写入 SQLite)   │    │  (本目录独立)   │    │  读尾部 + 聚合分钟线 + Queue  │
└─────────────────┘    └─────────────────┘    └──────────────┬───────────────┘
                                                              │ 数据包
                                                              ▼
                                            ┌─────────────────────────────────┐
                                            │      Orchestrator (主进程)       │
                                            │  Queue.get() 阻塞 → 数据齐备检查  │
                                            │  → ProcessPoolExecutor 并行调度   │
                                            └────────┬────────────────────────┘
                                                     │
           ┌─────────┬─────────┬─────────┬─────────┼─────────┬─────────┬─────────┐
           ▼         ▼         ▼         ▼         ▼         ▼         ▼         ▼
        StrategyA StrategyB StrategyC StrategyCS StrategyLH StrategyM StrategyP StrategyY
           │         │         │         │         │         │         │         │
           └─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
                                                     │ ZMQ 发信号
                                                     ▼
                                               交易终端/风控
```

| 层 | 文件 | 关键类/函数 |
|---|---|---|
| 行情写入 | `sql_writer_dce.py` | `RealTimeMarketData` → `AsyncTickWriter` → `DatabaseManager` |
| 数据服务 | `data_service.py` | `DataService.run()` / `_refresh_instrument()` / `_package_data()` |
| 策略编排 | `orchestrator.py` | `main()` → `run_strategy()` → `ProcessPoolExecutor(spawn)` |
| 策略执行 | `strategies.py` | `BaseStrategy.on_new_minute()` → `generate_factor_dataframe_*` |
| 工具库 | `data_function.py` | `Factor_generator` / `run_345()` / `ZMQPublisher` |

---

## 三、数据包流转（单分钟）

```
DataService._package_data()
    tick: {inst: df.tail(1000)}    # 只传最近 1000 条 tick
    min:  {inst: full_min_df}      # 完整分钟线（历史 CSV + DB 实时聚合）
    timestamp: 截断到整分钟

→ Queue.put(pkg)

Orchestrator 收到后，每个策略内部 7 阶段：
    extract  → align → fac_init → generate → predict → run_345 → save
   ~0.001s   ~0.0s     ~0.01s    ~1-5s     ~0.1s    ~0.01s  ~0.005s
```

---

## 四、所有已完成的修改（按时间倒序）

### 4.1 信号文本精简（`data_function.py`）

| 旧文本 | 新文本 | 触发条件 |
|--------|--------|---------|
| `非交易时间段 {t}，强制平仓` | `未开仓` | 不在交易时段内 |
| `看多，刷新多头持有时间` | `继续看多` | 多头持仓且 weighted > long_open |
| `看空，刷新空头持有时间` | `继续看空` | 空头持仓且 weighted < short_open |
| `继续持有多头` | `保持多头` | 多头持仓，阈值未触达 |
| `继续持有空头` | `保持空头` | 空头持仓，阈值未触达 |

### 4.2 汇总表格重构（`orchestrator.py`）

- **删除列**：extract / align / fac_init / predict / run345 / save / fac_shape 等细粒度耗时列
- **保留列**：pos / hold / elapsed / gen / pct
- **新增列**：`pred_t2` / `pred_t1` / `pred_t0`（最近 3 分钟的原始 weighted 预测值）
- **表头全英文**，避免中文 pandas 对齐问题
- **标题行**：`[Orchestrator] #{loop_count} | market: {time_recently} | sys: {readable_time}`

当前表格列：`symbol | pos | hold | elapsed | gen | pct | pred_t2 | pred_t1 | pred_t0 | signal`

### 4.3 子进程状态同步修复（`orchestrator.py` + `strategies.py`）

**问题**：ProcessPoolExecutor 的 worker 进程会缓存 `BaseStrategy` 对象，但主进程每次传给子进程的 `now_pos` / `now_holding` 来自主进程的 `ready_strategies` 对象，而这些对象从未被子进程的返回结果更新，导致每轮传入的都是 `0, 0`。子进程的真实持仓状态被覆盖，出现"刚开仓下一秒变空仓"的 bug。

**修复**（两处）：
1. `run_strategy()` 签名增加 `now_pos, now_holding` 参数，在调用 `on_new_minute()` 前显式覆盖子进程缓存状态。
2. `orchestrator.py` 主循环在 `future.result()` 后，用 `result['now_pos']` / `result['now_holding']` 回写主进程中的 `ready_strategies` 对象，确保下一轮传入的是最新状态。

### 4.4 Factor Cache 重构（`strategies.py`）

**旧逻辑**：`factor_cache/{date}/{sym}_pred_{date}.parquet` 存储 `pred_df`（模型预测值）。
**新逻辑**：`factor_cache/{date}/{sym}_fac_{date}.parquet` 存储 `fac_df`（原始因子表）。

原因：预测值可由因子 + 模型重新生成，存因子更安全，`save_files` 可随时重建。

- `_save_fac_cache()`：将 `self.fac_df` 按交易日拆分保存为 parquet
- `_load_fac_cache()`：加载所有 parquet，拼接后用 `model.predict()` 重建 `pred_df`
- `_init_history()`：历史因子计算完成后保存 fac 缓存，不再写 `pred_df.csv` / `fac_old.csv`

**遗留**：旧版 `*_pred_*.parquet` 文件需手动删除。

### 4.5 Save-Files 清理 relocated（`orchestrator.py` + `strategies.py`）

**旧逻辑**：`BaseStrategy.__init__()` 中清理 `save_files/{sym}/` 下的历史文件。问题：`__init__` 在每个 subprocess worker spawn 时都会执行，可能误删正在写入的文件。
**新逻辑**：清理逻辑移到 `orchestrator.py` 的 `main()` 主进程启动阶段，只执行一次。

### 4.6 数据核对工具

| 文件 | 用途 |
|------|------|
| `check_factor.ipynb` | Jupyter notebook，逐分钟对比实时因子 CSV vs 研究环境 `all_factor.feather`，输出 diff 矩阵和 top 问题因子 |
| `check_factor_README.md` | notebook 的配置说明、阈值含义、日期映射规则 |
| `check_factor_cache.py` | 命令行脚本，对比 `factor_cache` parquet vs 研究数据，输出不一致因子比例 |

已知：实时 vs 研究在 `FAC_KDJ_*` / `zigzag` 等因子上有 ~0–2.2% 的差异，用户已确认可接受。

### 4.7 计算优化（`data_function.py`，2026-05-06 完成）

| # | 优化点 | 具体修改 | 效果 |
|---|--------|---------|------|
| 1 | numba 模块级提取 | `@njit` 函数移到模块顶部（`_rolling_corr_3` / `_rolling_corr_5` / `_spearman_rank`） | JIT 6s → 0.0016s |
| 2 | trade_date 截断 | `MIN_FACTOR_TRADE_DATES` + `_truncate_by_trade_date` 装饰器 | `min_data` 30天 → 1-2天 |
| 3 | resample 向量化 | `groupby.apply(lambda)` → `groupby.transform` / `groupby.agg` | 1.9s → 0.009s |
| 4 | wma numba 化 | `rolling.apply(lambda)` → `_wma_numba` | 1.3s → 0.014s |
| 5 | skewness/kurtosis | `groupby.skew()` / `nankurt` | 0.47s→0.007s / 0.47s→0.18s |
| 6 | copy 优化 | `tick_data.copy()` → `copy(deep=False)` | 消除 300 次 deep copy |

### 4.8 架构改造（`orchestrator.py`，2026-05-06 完成）

| # | 改造点 | 具体修改 | 原因 |
|---|--------|---------|------|
| 7 | ThreadPoolExecutor → ProcessPoolExecutor | `ProcessPoolExecutor(max_workers=8, mp_context=spawn)` | 绕过 GIL，真正并行 |
| 8 | spawn 模式 | `multiprocessing.get_context('spawn')` | fork 继承 sqlite 连接导致死锁 |
| 9 | 子进程策略缓存 | `_WORKER_STRATEGIES = {}`，首次创建后复用 | 避免每轮加载 5 个 .lgb 模型 |
| 10 | 子进程 stdout suppress | `sys.stdout = open(os.devnull, 'w')` | 消除每分钟噪音 |
| 11 | 表格按 symbol 排序 | `sorted(results.items())` | 方便纵向对比 |

### 4.9 数据对齐逻辑改造（`orchestrator.py` + `strategies.py`）

| # | 改造点 | 具体修改 | 原因 |
|---|--------|---------|------|
| 12 | 重试次数增加 | `range(3)` → `range(10)`，每次重试刷新所有品种 | 给 DB 写入留缓冲 |
| 13 | 永不剔除策略 | 删除剔除逻辑，改为警告后继续 | 用户明确要求 |
| 14 | reindex 对齐 | 以主合约 `datetime` 为基准，`reindex(method='ffill')` | 避免时间错位 |

---

## 五、当前性能基准

| 指标 | 数值 | 说明 |
|---|---|---|
| 单策略 generate | ~1.0-1.3s | 全量因子计算（worker 缓存生效后） |
| 8 策略并行总耗时 | **稳定态 ~2s** | ProcessPoolExecutor + spawn + 策略缓存 |
| 首轮耗时 | ~6-8s | worker 首次 spawn，加载模型 + numba JIT |
| 第二轮+ | ~4-5s → ~2s | `_WORKER_STRATEGIES` 缓存逐轮命中 |

---

## 六、关键代码位置速查

| 功能 | 文件 | 行号/函数 |
|------|------|----------|
| 主循环 | `orchestrator.py` | `main()` |
| 子进程执行 | `orchestrator.py` | `run_strategy()` |
| 主进程状态回写 | `orchestrator.py` | `future.result()` 后的 `st.now_pos = ...` |
| 数据对齐 | `strategies.py` | `_align_and_check()` |
| 策略核心 | `strategies.py` | `on_new_minute()` |
| 因子缓存加载 | `strategies.py` | `_load_fac_cache()` / `_save_fac_cache()` |
| 因子计算 | `data_function.py` | `Factor_generator` / `generate_factor_dataframe_*` |
| 交易信号 | `data_function.py` | `run_345()` |
| numba 优化 | `data_function.py` | `_rolling_corr_3` / `_rolling_corr_5` / `_wma_numba` |
| trade_date 截断 | `data_function.py` | `_truncate_by_trade_date` 装饰器 |
| 数据服务 | `data_service.py` | `DataService` 类 |

---

## 七、`weighted_s` 生成链路与历史值覆盖问题（已修复）

### 7.1 完整代码链路

```
DataService 触发新分钟
  │
  ├─ _package_data(current_time)                    data_service.py
  │   └─ tick_data = tick_df[tick_df.datetime <= current_time].tail(1000)
  │
  ├─ Factor_generator.__init__ -> _load_data_min    data_function.py
  │   └─ valid_index = tick_data.resample('1min', label='right', closed='right').last().index
  │   └─ 实时模式：valid_index = valid_index[-1:]
  │
  ├─ generate_factor(fac_generator, valid_index)    strategies.py
  │   └─ fac_df = pd.DataFrame(factor_dict, index=valid_index)   # 1 行
  │
  ├─ model.predict(fac_df) -> pred                  strategies.py
  │   └─ pred 只有 1 行，索引 = valid_index[-1]
  │
  ├─ pred_df 更新（已修复）
  │   └─ 新逻辑：new_mask = ~pred.index.isin(self.pred_df.index)，只追加新索引
  │
  ├─ run_345(pred_df)                                data_function.py
  │   └─ df['weighted_s'] = weighted*0.6 + shift()*0.3 + shift(2)*0.1
  │
  └─ _last_weighted_s = df['weighted_s'].dropna().iloc[-5:].tolist()   strategies.py
```

### 7.2 根因与修复

DataService 的触发条件是"所有品种中最新的 tick 时间跨分钟"。如果 B 品种在 21:21:30 有新 tick，而 A 品种在 21:21:00 之后根本没有新 tick，仍然会统一触发 21:22:00。此时 A 品种的 `valid_index` 和上一轮重复，会导致 `pred` 索引重复。

修复：`new_mask = ~pred.index.isin(self.pred_df.index)`，只追加真正的新索引，避免覆盖历史值。

---

## 八、已知问题

| 问题 | 影响 | 状态 |
|------|------|------|
| 首轮耗时 6-8s | 开盘后第一分钟稍慢 | 预期内，spawn + 模型加载不可避免 |
| ZMQ 端口残留 | 重启时可能 `Address already in use` | 待处理（P3） |
| A 策略 tick 提前到达 | DataService 层未统一等待 | 待处理（P4） |
| 旧 `*_pred_*.parquet` 残留 | 占用磁盘，不影响运行 | 需手动删除 |

---

## 九、核心约束（开发时必须遵守）

1. **因子计算方法零改动**：`data_function.py` / `strategies.py` 中的因子计算公式保持原样，避免实盘/回测数据不一致。
2. **DataService 逻辑已稳定**：数据层不动。
3. **对齐逻辑已固化**：`_align_and_check` 采用 `reindex(method='ffill')`，永不剔除策略。
4. **子进程静默执行**：`run_strategy` 内 suppress stdout，新增 print 需考虑是否在子进程中产生噪音。
5. **Factor Cache 格式**：缓存只存 `fac_df`（原始因子），不存 `pred_df`。

---

## 十、调试速查

- 单品种独立测试：`python3 benchmark_pool.py`
- 查看子进程错误：`result['err']` 会打印在主进程输出中
- 检查对齐结果：在 `_align_and_check` 中加 `print(df_aligned.head())`
- 检查 factor cache：`check_factor_cache.py`
- 对比研究数据：`check_factor.ipynb`
