# DCE 农产品期货策略 —— Agent 项目指南

> 本文档供 minimax（或其他 Agent）快速理解项目全貌，涵盖架构、数据流、核心逻辑、环境差异与已知陷阱。
>
> 最后更新：2026-05-25

---

## 一、项目概述

这是一个 **DCE（大连商品交易所）农产品期货的分钟级量化交易策略系统**，覆盖 8 个品种：

| 品种代码 | 合约示例 | 名称 |
|---------|---------|------|
| A | a2605 / a2607 | 豆一 |
| B | b2605 / b2607 | 豆二 |
| C | c2605 / c2607 | 玉米 |
| CS | cs2605 / cs2607 | 玉米淀粉 |
| M | m2605 / m2609 | 豆粕 |
| Y | y2605 / y2609 | 豆油 |
| P | p2605 / p2609 | 棕榈油 |
| LH | lh2605 / lh2607 | 生猪 |

每个品种有独立的 LightGBM 模型（5-fold），分钟级生成因子 → 模型预测 → 产生开平信号。

---

## 二、双环境架构

### 2.1 环境对比

| 维度 | Dev (Test) | Prod (Online) |
|------|-----------|---------------|
| **路径** | `/home/strategy_PAMY_dev/` | `/home/online/dce农/` |
| **目的** | 回放核对因子正确性 | 实盘交易 |
| **合约** | 2605（已过期，用于回放） | 近月/主力（如 2607/2609） |
| **数据源** | SQLite 历史 tick 数据 | 生产 ZMQ 行情（`192.168.2.238:77718`） |
| **启动** | 手动 | 每日 20:00 自动启动，15:00 自动退出 |
| **监控** | `http://localhost:5001`（模拟盘） | `http://localhost:5000`（实盘） |
| **归档** | 无自动归档 | 每日 15:00 自动归档到 `history_save_files/YYYYMMDD/` |

### 2.2 核心原则：Dev 与 Prod 的策略逻辑必须一致

- `weighted_s` 计算**必须使用 `shift()`**，绝对禁止 `reindex + ffill`
- 任何用 `ffill` / `bfill` 掩盖数据缺失的做法都是禁止的
- 差异必须暴露，而不是掩盖

---

## 三、系统架构与数据流

```
┌─────────────────────────────────────────────────────────────────┐
│                          数据源层                                │
│  ┌──────────────┐    ┌─────────────────────────────────────┐   │
│  │ 模拟数据源    │    │ 生产行情 ZMQ (192.168.2.238:77718)   │   │
│  │ 172.17.0.6   │    │ (dev 测试时可用)                      │   │
│  └──────┬───────┘    └─────────────────────────────────────┘   │
└─────────┼───────────────────────────────────────────────────────┘
          │ ZMQ 推送 tick
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                       sql_writer_dce.py                         │
│  - 接收 ZMQ tick 数据                                            │
│  - 写入 SQLite: tick_data.db                                    │
│  - dev: WAL 模式，全局缓存连接                                   │
└─────────┬───────────────────────────────────────────────────────┘
          │ SQLite DB
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                       data_service.py                           │
│  - 启动时加载历史分钟 CSV (`files/{inst}_min.csv`)               │
│  - 全量/增量读取 DB tick → 聚合为分钟线                          │
│  - 检测新分钟 → 打包数据包 → 放入 Queue                          │
│  - `_package_data(time_recently)`：截断到整分钟，含秒时间传给策略 │
└─────────┬───────────────────────────────────────────────────────┘
          │ 数据包 (tick + min + timestamp)
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                       orchestrator.py                           │
│  - 主循环：从 Queue 取数据包                                      │
│  - 数据齐备性检查（4个策略并行，要求 < 5秒）                      │
│  - ProcessPoolExecutor 并行执行各品种策略                         │
│  - 15:00 收盘退出（dev 回放时可能注释掉）                         │
└─────────┬───────────────────────────────────────────────────────┘
          │ 返回 pred / fac_df / data_dict
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                       strategies.py                             │
│  - BaseStrategy: 每个品种一个实例                                │
│  - `compute(data_package, time_recently)`                        │
│    ├─ 生成因子 (Factor_generator)                                │
│    ├─ 模型预测 (LightGBM 5-fold)                                 │
│    ├─ `run_345()` 信号逻辑（开多/开空/平仓/持多/持空）            │
│    └─ 保存结果                                                   │
│       ├─ predictions_{timestamp}.csv                             │
│       ├─ factors/{timestamp}.csv                                 │
│       ├─ json/trading_status_{timestamp}.json (now_pos)          │
│       └─ data/{sym}_min_{timestamp}.csv (含 avg_price_from_5s)   │
└─────────┬───────────────────────────────────────────────────────┘
          │ CSV / JSON 文件
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                       monitor_web.py                            │
│  - 独立 Flask 进程，端口 5000/5001                               │
│  - 每 5 秒扫描 save_files/                                      │
│  - 读取 json 中的 now_pos，通过差分推断信号                       │
│  - 延迟 1 分钟成交：T 分钟信号 → T+1 分钟 avg_price_from_5s 成交  │
│  - K 线图（echarts）+ 持仓背景色 + 交易记录                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 四、核心逻辑详解

### 4.1 信号生成流程

```
data_package (tick + min) 
    → BaseStrategy.compute()
        → Factor_generator 生成因子 (~400+ 个)
        → LightGBM 5-fold 预测 (pred_df['weighted'])
        → run_345(now_pos, pred_df, thresholds)
            → 输出: now_pos, signal, thresholds_df
```

`run_345` 是核心信号逻辑：
- 输入：当前持仓 `now_pos`、预测值 `weighted`、阈值 `th1/th2`
- 输出：`now_pos`（更新后）、`signal`（字符串，如"开多""平多"）
- 规则：基于 `weighted` 与阈值的交叉，支持多仓/空仓切换

### 4.2 持仓与成交

- **信号产生**：分钟 T 产生（如 09:15:00）
- **实际成交**：分钟 T+1 的 `avg_price_from_5s`（第 5 秒后的 last_price 均值）
- `monitor_web.py` 负责延迟成交的记账，不是策略主循环

### 4.3 分钟线定义（右端点）

- `ts=21:19:00` 代表 `21:18:00.001 ~ 21:19:00.000` 的数据
- `_save_min_data` 取的是**上一分钟** tick 来计算 `avg_price_from_5s`
- 排除前 5 秒：`second > 5` 的 tick 才计入均价

### 4.4 交易日规则

```python
def datetime_to_trade_date(dt):
    if dt.hour >= 20:  # 夜盘 21:00-23:00
        return 下一交易日
    return dt.strftime('%Y-%m-%d')  # 日盘
```

- 夜盘数据归属下一个交易日
- 文件名中的日期 = 交易日期（不是 tick 发生的日历日期）

---

## 五、关键文件与路径

### 5.1 Dev 环境

| 用途 | 路径 |
|------|------|
| 策略入口 | `/home/strategy_PAMY_dev/orchestrator.py` |
| 配置 | `/home/strategy_PAMY_dev/config.json` |
| 策略逻辑 | `/home/strategy_PAMY_dev/strategies.py` |
| 因子/数据函数 | `/home/strategy_PAMY_dev/data_function.py` |
| 数据服务 | `/home/strategy_PAMY_dev/data_service.py` |
| ZMQ 写入 | `/home/strategy_PAMY_dev/sql_writer_dce.py` |
| 监控面板 | `/home/strategy_PAMY_dev/monitor_web.py` |
| SQLite DB | `/home/strategy_PAMY_dev/tick_data.db` |
| 模型 | `/home/strategy_PAMY_dev/models/{SYMBOL}/kfold_fold{1-5}_0.lgb` |
| 实时结果 | `/home/strategy_PAMY_dev/save_files/{SYMBOL}/predictions/` |
| 持仓状态 | `/home/strategy_PAMY_dev/save_files/{SYMBOL}/json/` |
| 分钟附加数据 | `/home/strategy_PAMY_dev/save_files/{SYMBOL}/data/` |
| 因子缓存 | `/home/strategy_PAMY_dev/factor_cache/` |
| 监控输出 | `/home/strategy_PAMY_dev/save_files/盘中分析/` |
| 核对 Notebook | `/home/strategy_PAMY_dev/check_intraday copy.ipynb` |
| 研究因子 | `/mnt/Data/writable/liaoyuyang/factor/{SYMBOL}/all_fac/all_factor.feather` |

### 5.2 Prod 环境

| 用途 | 路径 |
|------|------|
| 策略入口 | `/home/online/dce农/orchestrator.py` |
| 配置 | `/home/online/dce农/config.json` |
| 监控面板 | `/home/online/dce农/monitor_web.py` |
| 历史归档 | `/home/online/dce农/history_save_files/YYYYMMDD/` |
| 盘后补算 | `/home/strategy_online/strategy_PAMY_dce/calc_recent_data.py` |

---

## 六、运行方式

### 6.1 Dev 回放

```bash
cd /home/strategy_PAMY_dev
# 1. 确保 tick_data.db 有历史数据
# 2. 启动策略
python orchestrator.py
# 3. 另开窗口启动监控
python monitor_web.py
# 访问 http://localhost:5001
```

### 6.2 Prod 实盘

Prod 每日 **20:00 自动启动**，**15:00 自动退出**。

```bash
# 监控面板（可独立重启，不影响策略主循环）
cd /home/online/dce农
python monitor_web.py
# 访问 http://localhost:5000
```

**绝对不能重启 orchestrator**（实盘在跑），除非用户明确同意。

---

## 七、config.json 关键配置

```json
{
    "symbols": ["A", "B", "C", "CS", "M", "Y", "P", "LH"],
    "active_symbols": ["A", "B", "C", "CS", "M", "Y", "P", "LH"],
    "paths": {
        "db_path": "/home/strategy_PAMY_dev",
        "load_recent_data_path": "/home/strategy_PAMY_dev/files",
        "models_root": "/home/strategy_PAMY_dev/models",
        "save_files_root": "/home/strategy_PAMY_dev/save_files"
    },
    "trading_params": {
        "th1": 0.9,
        "th2": 0.5,
        "holding_period_max": 10
    },
    "calculation_params": {
        "cache_keep_days": 5
    },
    "replay_mode": {
        "enabled": false,
        "day_date": "2026-03-23",
        "night_date": "2026-03-22"
    }
}
```

### replay_mode（dev 专用）

当 `enabled: true` 时，`monitor_web.py` 不使用系统日期，而是：
- K 线加载：`day_date` + `night_date`
- 文件扫描：扫这两个日期的 `trading_status_*.json`
- 交易记录保存：用 `day_date` 生成文件名

方便回放固定日期时，monitor_web 能正确显示历史数据。

---

## 八、已知问题与陷阱（按优先级）

### 🔴 绝对不能做的事

1. **绝对不能重启 Prod orchestrator**（`/home/online/dce农/orchestrator.py`）
   - 实盘在跑，重启会断线
   - 需要重启的改动只能改文件，等下次自然调度（20:00）生效

2. **禁止用 `ffill` / `bfill` / `reindex(...).ffill()` 掩盖数据缺失**
   - `weighted_s` 已回退到 `shift()`
   - 如果缺失分钟导致 diff 变大，说明数据管道有问题，要修 data_service / 数据源

3. **不能混淆数据的三个层级**
   - 层级 A：SQLite 原始 tick
   - 层级 B：DataService 的 `tick_cache` / `min_cache`
   - 层级 C：策略层接收的 `data_package` / `fac_generator`
   - B 和 C 之间可能因为 `_package_data` 的截断/覆盖导致差异

### 🟡 常见陷阱

4. **`_package_data` 时间截断**
   - 5/22 修复：必须传 `time_recently`（已 `replace(second=0)`），不能传 `raw_time`（含秒）
   - 否则 tick 会被错误归入下一分钟桶

5. **`aggregate_ticks` + `update_concat` 是覆盖操作**
   - `update_concat` 用 `drop_duplicates(keep='last')` 覆盖同一时间戳的旧值
   - 如果 `read_table` 读出的 tick 不完整，会覆盖掉正确的历史 min

6. **WAL 模式 + 连接缓存 = 数据不同步风险**
   - sql_writer 写 WAL，data_service 缓存连接可能看不到最新数据

7. **夜盘日期归属**
   - 夜盘 21:00-23:00 的文件名日期 = 下一个交易日
   - `_load_kline` 周一需要回退到上周五（不是周日）

8. **`avg_price_from_5s` 取上一分钟 tick**
   - 分钟线是右端点，`ts=21:19:00` 对应 `21:18:00.001~21:19:00.000`
   - `_save_min_data` 中 `tick_mask` 是 `current_minute - 1min`

9. **回放时 orchestrator 15:00 退出**
   - dev 中已注释掉，避免回放中途退出

10. **`monitor_web.py` 的 `trades` 不包含当前持仓**
    - 未平仓持仓需要从 `positions` 单独注入前端才能渲染背景色

---

## 九、核对与调试工具

### 9.1 因子核对

- **Notebook**: `/home/strategy_PAMY_dev/check_intraday copy.ipynb`
  - Cell 7: 原始值并排显示
  - Cell 8: 逐分钟差异矩阵
  - Cell 9: 问题因子详情
  - Cell 10: 单因子走势对比

- **阈值**: `TOL_ABS = 0.01`, `TOL_REL = 0.05`

### 9.2 快速脚本

```bash
# 对比 factor_cache vs 研究环境
python3 check_factor_cache.py

# 手动读取 tick 验证
sqlite3 tick_data.db "SELECT instrument, COUNT(*) FROM tick_data_c2605;"
```

### 9.3 21:15 快照调试

`data_service.py` 和 `strategies.py` 中插入了 21:15 快照逻辑，保存到：
```
/home/strategy_PAMY_dev/logs/factor_debug/snapshot_2115/
├── ds_tick_cache_{inst}.parquet
├── ds_min_cache_{inst}.parquet
├── ds_pkg_tick_{inst}.parquet
├── ds_pkg_min_{inst}.parquet
```

---

## 十、修改记录索引

| 日期 | 关键修改 | 位置 |
|------|---------|------|
| 0518 | weighted_s 回退 shift()、gap filling、monitor_web 独立、auto-archive | dev + prod |
| 0519 | monitor_web 13 项修复（时区、信号推断、UI 增强） | prod |
| 0520 | avg_price_from_5s / last_twap 修复、per-symbol data 目录 | dev + prod |
| 0521 | mid_price .round(4)、10ms 去重、night 数据日期归属差异定位 | dev + prod |
| 0522 | `_package_data(raw_time)` 根因修复、RPP_5D 对齐 | dev |
| 0525 | 回放模式 replay_mode（config 控制日期） | dev |
| 0525 | 15:00 退出逻辑注释（dev 回放用） | dev |
| 0525 | K 线 Monday 回退逻辑（prev_trade_day） | dev + prod |

---

## 十一、联系上下文

- **日报/周报**: `/home/agent_daily/kimi/`
- **SOP**: `/home/strategy_PAMY_dev/FACTOR_CHECK_SOP.md`
- **AGENTS 约定**: `/home/strategy_PAMY_dev/AGENTS.md`
- **诊断日志**: `/home/strategy_PAMY_dev/DIAGNOSTIC_LOG.md`
