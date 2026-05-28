# 策略系统架构与修改记录

## 一、原策略流程（main_strategy_*.py）

```
每个策略独立进程运行
  ↓
while True 循环，每 0.5s 读 DB 最新时间
  ↓
检测到新分钟 → load_tick_min() 加载 tick + min 数据
  ↓
Factor_generator → generate_factor() 计算因子
  ↓
5 个 LightGBM predict() → pred_df 追加
  ↓
run_345() 分位数阈值 + 开平仓信号
  ↓
保存 CSV/JSON，ZMQ 发送
```

**问题**：4 个策略各读一次 DB + 历史 CSV，共 16 次/分钟，IO 冗余。

---

## 二、新架构流程

```
sql_writer_dce.py（收行情写 DB，独立进程）
        ↓
DataService（后台线程，统一读 DB，每 0.5s 轮询）
        ↓ 检测到新分钟 → 数据包 {tick, min, timestamp}
Orchestrator（主线程，Queue.get() 阻塞等待）
        ↓ ThreadPoolExecutor(max_workers=4)
    ┌───┴───┬───────┬───────┐
  StrategyP StrategyA StrategyM StrategyY
```

### 核心文件

| 文件 | 职责 |
|------|------|
| `orchestrator.py` | 统一入口：启动 DataService → 初始化 4 策略 → 主循环并行执行 |
| `data_service.py` | 后台线程：initialize() 加载历史 → run() 轮询 DB → _package_data() 打包 |
| `strategies.py` | 4 个策略类 + 嵌入的 generate_factor_dataframe_* 函数 |
| `data_function.py` | 工具库：DB 读写、Factor_generator、run_345、parse_df 等 |
| `sql_writer_dce.py` | 行情接收 → SQLite（独立进程，未改动） |

---

## 三、已做修改

### 3.1 架构层
1. **4 策略合一**：从 `main_strategy_P/A/M/Y.py` 提取到 `strategies.py`，移除 `importlib` 动态加载
2. **DataService 统一读数**：DB 读取从 16 次/分钟 → 1 次/分钟
3. **ThreadPoolExecutor 并行**：4 策略共享数据包，同时执行
4. **ZMQ LINGER=0**：进程退出立即释放端口，解决残留问题
5. **Config 路径隔离**：test 目录自包含，所有路径指向自身

### 3.2 缓存层（按交易日拆分）
- 启动：`factor_cache/{sym}_pred_{trade_date}.parquet` → 加载所有历史
- 运行：`pred_df` 追加当前分钟预测值
- 退出：只覆盖保存**当天最后一个交易日**的缓存文件
- trade_date 规则：`hour >= 20` → `chinesecalendar` 下一交易日；否则当日

### 3.3 性能优化
1. **因子并行计算**：`parallel_factor_compute` 装饰器，4 线程并行跑 196~231 个因子
2. **耗时拆解**：`on_new_minute` 拆成 7 段（extract / align / fac_init / generate / predict / run345 / save）

### 3.4 正确性修复
1. **非交易时段不跳过**：删除 `is_in_no_trade_period` 过滤，由 `run_345` 内部补 0
2. **长度不一致对齐**：`_align_and_check` 切到最短长度，不跳过该分钟
3. **时间截断到整分钟**：`DataService.run()` 中 `time_recently.replace(second=0, microsecond=0)`，避免秒级数据被 resample 归入下一分钟

### 3.5 日志精简
- 删除 `print(df)`、`print("==========df=========")`
- 超时 60s 不再刷屏
- 4 策略汇总成一张表格（pos / holding / 耗时拆解 / fac_shape / weighted_s(最近5分) / 分位数 / 信号）

---

## 四、当前状态与待优化

| 项目 | 状态 |
|------|------|
| 架构稳定性 | ✅ 基本稳定 |
| 单策略耗时 | ⚠️ ~10-15s（目标 <5s）|
| 4 策略总耗时 | ⚠️ ~15s（看最慢的那个）|
| 数据正确性 | ✅ 非交易时段补 0、长度对齐、时间截断 |
| 缓存机制 | ✅ 按交易日拆分，启动加载 / 退出仅保存当天 |

### 已知瓶颈（待进一步定位）
- `generate_factor` 已 4 线程并行，但仍占大头
- `Factor_generator.__init__` 中的 `resample` 可能较慢
- `run_345` 中 `df.iloc[:-1].quantile()` 对 1725 行 x 7 列计算
- `_save_results` 每次写 2 个 CSV（本地磁盘，影响较小）

---

## 五、关键代码位置

| 功能 | 文件 | 函数/类 |
|------|------|---------|
| 统一调度入口 | `orchestrator.py` | `main()` |
| 数据读取线程 | `data_service.py` | `DataService.run()` |
| 策略基类 | `strategies.py` | `BaseStrategy.on_new_minute()` |
| 因子并行装饰器 | `strategies.py` | `parallel_factor_compute` |
| 因子计算 | `strategies.py` | `generate_factor_dataframe_*` |
| 交易信号 | `data_function.py` | `run_345()` |
| DB 读写 | `data_function.py` | `read_table()` / `parse_df()` |
| 行情写入 | `sql_writer_dce.py` | 独立进程，未改动 |
