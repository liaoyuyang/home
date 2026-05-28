# 性能优化开发日志

> 记录时间：2026-04-30  
> 项目：`strategy_PAMY_dev` — DCE 期货实时交易系统  
> 硬件：64核 Intel Xeon Gold 6246R @ 3.40GHz

---

## 一、核心发现（今日定位）

### 1.1 瓶颈不在 tick 降频，在 min 跨品种因子

通过 `benchmark_factor.py` 分类测试（B 策略，300 因子，1000 tick，valid=1）：

| 类别 | 因子数 | 总耗时 | 平均耗时 |
|------|--------|--------|----------|
| tick 降频因子 | 219 | **1.096 s** | 0.005 s |
| min 跨品种因子 | 81 | **4.126 s** | 0.051 s |

**结论：min 因子虽少，但总耗时是 tick 因子的 4 倍。**

### 1.2 两大元凶

| 因子名 | 单策略耗时 | 占比 |
|--------|-----------|------|
| `bar3_trend_corr` | **3.004 s** | min 因子的 73% |
| `bar5_trend_corr` | **0.720 s** | min 因子的 17% |

**根因：** 这两个因子用 numba 对 **完整 min_data（约 10,000+ 行）** 逐行做 `argsort` 循环，最后 `reindex(valid_index[-1:])` 只取最后一个值。运行时 `valid_index` 已截断到仅 1 个点，但内部仍然遍历了全部历史分钟线，大量计算被浪费。

### 1.3 8 策略并行 vs 单策略

- **单策略 generate**：~5.2 s（tick 1.1s + min 4.1s）
- **8 策略并行总时间**：**28–32 s**（目标 < 5 s）
- **问题**：pandas/numpy 内存带宽竞争 + cache thrashing，单核性能被严重稀释。

---

## 二、今日已完成优化

| 优化项 | 文件 | 说明 |
|--------|------|------|
| 对齐修复 | `strategies.py` | `_align_and_check` 改为对齐 `main_symbol` 长度，长的截断、短的仅非交易时段允许 |
| tick 减量 | `data_service.py` | `_package_data` 传 `tick_df.iloc[-1000:]`（原为 2000） |
| 预缓存索引 | `data_function.py` | `Factor_generator.__init__` 预存 `set_index('datetime')`，避免重复执行 |
| valid_index 截断 | `data_function.py` | 实时模式（`len(tick) < 5000`）`valid_index = valid_index[-1:]`，减少 reindex 开销 |
| 因子计算去线程化 | `strategies.py` | 移除 `parallel_factor_compute` 内部 `ThreadPoolExecutor`，纯 `map()` 循环（线程开销 > 收益） |
| 数据就绪重试 | `orchestrator.py` | `_check_data_ready()` + 最多 3 次 × 0.5 s 重试，应对 A 策略 tick 提前到达 |
| 缓存目录重构 | `strategies.py` | `_save/_load_pred_cache` 改为 `factor_cache/{trade_date}/` 子目录 |
| 日志降噪 | 多处 | 移除 `run_345`、`parallel_factor_compute`、`_align_and_check` 的 verbose print |
| `load_fac_df_old` 保留 | `strategies.py` | 启动时计算完整历史一次，不参与实时路径 |

---

## 三、待解决问题（按优先级）

### 🔴 P0 — min 因子全历史重复计算（最大瓶颈）

- **现象**：每分钟运行时，81 个 min 因子各自遍历 10,000+ 行历史分钟线，但最终 `reindex` 只保留 1 个点。
- **影响**：`bar3_trend_corr` 单策略 3 s → 8 策略并行被放大到可能 10 s+。
- **建议方案**：
  1. **截断传入数据**：实时运行时，`Factor_generator` 的 `min_data` 只保留最近 **N 行**（如 500 行，覆盖最大窗口即可）。`zigzag` 窗口为 1200，需保留至少 1500–2000 行。
  2. **因子级增量计算**：启动后，每分钟各因子只计算最新窗口的值，而非全序列滚动。

### 🟠 P1 — 8 策略并行资源竞争

- **现象**：单策略 ~5 s，8 策略并行 → 28–32 s，远未达到线性加速。
- **根因**：Python GIL + pandas/numpy 内存带宽争抢 + CPU cache 抖动。
- **建议方案**：将 8 策略拆分为 **独立进程**，DataService 通过 Queue/SharedMemory 分发数据包。64 核机器上可真正并行。

### 🟡 P2 — ZMQ 端口冲突

- **现象**：Orchestrator 重启时若旧进程未杀干净，`Address already in use`。
- **建议方案**：设置 `zmq.LINGER=0` + `SO_REUSEADDR`，或改用动态端口/Unix Domain Socket。

### 🟡 P3 — A 策略 tick 提前到达

- **现象**：A 策略 tick 比 CS/M/Y/P 早 1 个 tick 到达，触发 `_check_data_ready` 重试（"数据不齐" + 2 次重试 = 1 s 延迟）。
- **建议方案**：在 DataService 层统一等待所有活跃合约都有新分钟数据后再打包，而非在 Orchestrator 层重试。

### 🟢 P4 — LH 夜盘缺失数据

- **现象**：LH 无夜盘（21:00–23:00 无 tick），C/CS/B 策略的 other_symbol 包含 LH 时夜间会缺失数据。
- **现状**：`_align_and_check` 已允许非交易时段缺失，当前处理正确，无需改动。

---

## 四、关键决策记录（2026-04-30）

> **决定：因子计算逻辑完全不动**，保持历史数据核对的一致性。单策略 5 s 可接受，不可接受的是 8 策略并行时的 GIL/内存带宽竞争导致总时间暴涨到 30 s。

**选定方案 — B：统一 Launcher 启动 8 个独立子进程**

- **不动因子**：`data_function.py`、`strategies.py` 中的因子方法保持原样，避免数据核对风险。
- **不动 DataService**：数据服务层逻辑不变。
- **改动范围**：仅 `orchestrator.py` 层，把 ThreadPoolExecutor 多线程改为 multiprocessing.Process 多进程。

### 方案 B 设计草案

```
launcher.py（主控）
    ├── 启动 DataService 进程
    └── 启动 8 个 strategy_worker.py 子进程（A/B/C/CS/M/Y/P/LH）
            ├── 每个进程独立 import strategies.py
            ├── 各自初始化 Factor_generator + LightGBM 模型
            ├── 通过 multiprocessing.Queue / ZMQ 接收数据包
            └── 独立计算 → 预测 → ZMQ 发信号
```

**优势**：
- 彻底绕过 GIL，64 核机器上 8 进程真正并行。
- 总时间 ≈ 最慢单策略（~5 s），而非 8 策略叠加（30 s）。
- 因子代码零改动，历史回测/实盘结果完全一致。

**待考虑细节**：
- 数据包序列化：pandas DataFrame 通过 Queue 传递可能较重，需测试开销；或改用 ZMQ + SharedMemory。
- ZMQ 端口：8 个策略各自发信号，需分配不同端口或统一汇总。
- 进程监控：launcher 需捕获子进程崩溃并自动重启。

---

## 五、下一步行动建议

1. **本周实施**：基于方案 B，重写 `orchestrator.py` 为 `launcher.py` + `strategy_worker.py`。
2. **验证**：8 进程并行总时间是否降至 ~5 s 级别。

---

## 五、测试脚本备忘

- `benchmark_factor.py`：单策略分类 benchmark（tick vs min），可用于验证优化效果。
- 测试命令：`cd /home/strategy_PAMY_dev && python benchmark_factor.py`
