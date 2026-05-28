---
name: factor-recon-workflow
description: >
  DCE 农产品期货因子核对工作的标准流程与行为准则。
  当用户说"核对因子"、"继续核对"、"查一下差异"、"因子对不上"时触发。
  核心原则：先问后做，不确定时不擅自改代码。
---

# Factor Reconciliation Workflow

## 零、行为红线（先问原则）—— 最高优先级

> ⚠️ **用户对业务逻辑的口头说明 > agent 的一切推断。**
> 遇到任何不确定、意义不明确、或可能产生副作用的操作，**必须先问用户，得到明确指令后再动手**。

### 必须先问用户的场景（包括但不限于）

| 场景 | 示例 |
|------|------|
| **业务逻辑推断** | 用户说"trade_date 应该全天统一"，agent 不要自己推断"hour≥18 才统一"还是"所有情况都统一"，要问清楚 |
| **意义不明确的操作** | 用户提到"前十分钟"，agent 不要自己猜是"夜盘前十分钟"还是"日盘前十分钟"，要先确认 |
| **数据清理/删除** | 删除 DB 记录、清理缓存、重置状态前必须问 |
| **修改策略核心参数** | 阈值、持仓时间、过滤条件等 |
| **跨文件联动修改** | 改 A 文件可能影响 B 文件的行为时 |
| **用户说"以前是对的"** | 说明最近某次改动引入了 bug，agent 不要自己猜是哪次改动，要汇报现象等用户确认方向 |

### 绝对禁止
- ❌ 看到报错/异常，不汇报就直接改代码
- ❌ 用户只给了一半信息，agent 用推断补全另一半并执行
- ❌ "我觉得应该这样" → 先问 "您看这样对吗？"

### 允许直接做（无需多问）
- ✅ 纯信息查询（读代码、读日志、读 DB）
- ✅ 用户明确说过的固定修复（如"把 shift() 改回原来的"）
- ✅ 格式化、拼写错误、明显 typo

---

2. **长期问题写进日志**
   - 把已知陷阱、环境约束、用户口头确认的信息记录到 `notes/factor/` 和 `FACTOR_CHECK_SOP.md`
   - 每天结束写日报到 `agent_daily/kimi/YYYYMMDD.md`

## 一、工作环境约束（用户口头确认，非代码推导）

| 约束 | 说明 |
|------|------|
| `all_factor.feather` | **旧代码生成**，不是最新版；10ms 去重它不一定有 |
| 10ms 去重 | 用户认为**意义不大**，因为他的 tick 精度到不了 10ms |
| LH 夜盘回放 | **不放 LH 数据**；factor_cache 里的 LH 是历史白天盘缓存 |
| 核对基准 | 以 `factor_cache/{date}/{sym}_fac_{date}.parquet` 为准，不是 `save_files/` 下的实时 CSV |

## 二、核对分层（L1 → L4）

| 层级 | 内容 | 通过标准 |
|------|------|----------|
| L1 | tick drill-down | boundary tick volume/turnover/price 一致 |
| L2 | 1min K 线对齐 | open/high/low/close/volume/turnover/OI max_diff = 0 |
| L3 | factor 逐 bar 对齐 | fail ratio = 0（或分布指标在阈值内） |
| L4 | prediction 对齐 | weighted / weighted_s 一致 |

当前主要在做 **L3**。

## 三、核对工具

| 工具 | 路径 | 用途 |
|------|------|------|
| 快速脚本 | `strategy_PAMY_dev/check_factor_cache.py` | 批量比对 factor_cache vs all_factor.feather |
| Notebook | `strategy_PAMY_dev/check_intraday copy.ipynb` | 逐分钟并排显示、差异矩阵、单因子画图 |
| 手动验证 | `strategy_PAMY_dev/tick_data.db` | 读原始 tick 复现计算 |

## 四、核对阈值

```python
TOL_ABS = 0.01      # 绝对差异 < 0.01 视为一致
TOL_REL = 0.05      # 相对差异 < 5% 视为一致
```

## 五、数据路径速查

| 用途 | 路径 |
|------|------|
| 实时 factor_cache | `/home/strategy_PAMY_dev/factor_cache/{DATE}/{SYM}_fac_{DATE}.parquet` |
| 实时 tick DB | `/home/strategy_PAMY_dev/tick_data.db` |
| 研究 feather | `/mnt/Data/writable/liaoyuyang/factor/{SYM}/all_fac/all_factor.feather` |
| 历史 CSV (tick) | `/home/strategy_PAMY_dev/files/{contract}_tick.csv` |
| 历史 CSV (min) | `/home/strategy_PAMY_dev/files/{contract}_min.csv` |
| 策略逻辑 | `/home/strategy_PAMY_dev/strategies.py`、`data_function.py` |
| DataService | `/home/strategy_PAMY_dev/data_service.py` |

## 六、常见陷阱（已知）

1. **night 数据日期归属不一致**：研究环境 `level2_all` 把夜盘 datetime 映射到前一交易日，实时 `_tick.csv` 只保存 `date_lst[-tick_cache_days:]`，导致白天盘开盘初期 rolling window 差异
2. **LH 夜盘无数据**：回放时不放 LH，factor_cache 里的 LH 是旧缓存，核对 LH 前必须先确认数据来源日期
3. **all_factor.feather 是旧代码生成**：不要假设它包含了最新的 mid_price.round(4) 或 10ms 去重
4. **`parse_df(..., local=True)` 与 `local=False`**：`local=True` 时不做 volume/turnover 差分，走 `_tick.csv` 路径时传入的是 `local=True`
5. **skewness/kurtosis 对数据分布敏感**：同一 10ms 窗口内多 tick 的压缩/保留会影响结果
