# Agent 工作约定 — strategy_PAMY_dev / online dce农

> 以下约定由用户直接确认，不需要每次重复询问。

---

## 一、Test (dev) 与 Prod (online) 的同步原则

### 1.1 策略核心逻辑必须一致
- `weighted_s` 计算**必须使用 `shift()`**，绝对禁止 `reindex + ffill`。
  - 原因：`ffill` 会静默掩盖数据缺失分钟，导致实时和研究表面一致但实际数据质量下降。
  - 取舍：如果数据缺失，`shift()` 会错位，diff 会变大，从而暴露管道问题。
  - Prod 的 `data_function.py` 已按此约定修改为 `shift()` 版本。

### 1.2 Test 环境专属（不往 Prod 同步）
- `calc_recent_data.py` 中的固定测试日期（如 `end_date = "2026-03-23"`）是回放需要，Prod 保持动态日期。
- Test 的合约是 2605（过期合约，用于回放核对），Prod 是近月/主力合约，两者不需要一致。

### 1.3 Prod 改进往 Test 同步的边界
- **需要同步**：`strategies.py` 的数据保存格式（signal 列、TWAP 精细计算）、`data_service.py` 的 timestamp 截断和 raw_time 保留、`monitor_web.py` 的前端和健壮性改进。
- **不需要同步**：
  - `orchestrator.py` 的 `save_files` 自动归档逻辑（Test 不需要留档）。
  - `orchestrator.py` / `sql_writer_dce.py` 的 19:00/20:00 等待启动逻辑（Test 手动启动）。
  - `config.json` 中的合约、ZMQ 端点、路径（环境专属）。

---

## 二、Test 环境的定位

Test (`/home/strategy_PAMY_dev/`) 的核心目的是：
> **和研究环境核对数据正确性**，而不是模拟实盘留档或归档。

因此：
- Test 不需要每天自动归档 `save_files`。
- Test 允许使用过期合约（2605）和固定回放日期。
- Test 的 `monitor_web.py` 可以复用 Prod 版本（路径从 config 读取，自动适配）。

---

## 三、数据核对红线

1. **不能掩盖缺失**：任何用 `ffill` / `bfill` / `reindex(...).ffill()` 来让实时和研究"看起来一致"的做法都是禁止的。
2. **错位要暴露**：如果 `shift()` 因为缺失分钟而错位，diff 变大是对的，说明要修 data_service / 数据源，而不是修计算层去容错。
3. **Tick 过滤精度**：`data_service.py` 打包 tick 数据时，timestamp 必须截断到整分钟，但 tick 过滤要保留原始含秒时间（`raw_time`），避免秒级边界 tick 被错误截断。

---

## 四、Prod 运行约束（绝对禁止）

- **绝对不能重启 Prod orchestrator**（`/home/online/dce农/orchestrator.py`），除非用户明确同意。
- Prod 在跑实盘，任何需要重启 orchestrator 的改动只能改文件，等下次自然调度（每日 20:00 启动 / 15:00 退出）生效。
- `monitor_web.py` 是独立进程，可以单独重启，不影响策略主循环。

---

## 五、已知问题与待办

### 5.1 upmean / downmean 降频逻辑差异（2026-05-19 发现）

**现象**：核对时发现大量 `*_upmean` / `*_downmean` 因子 diff 巨大（实时有值、研究为 NaN）。

**根因**：研究环境 `factor_generator.py` 中 `compute_up_mean` / `compute_down_mean` 使用了 `np.where(...).mean()`，**numpy ndarray 的 `.mean()` 不会跳过 NaN**。只要组内有一个 NaN tick，整个分钟结果就是 NaN。实时环境原先用的是 pandas `.mean()`（skipna=True），所以能算出值。

**模型影响**：upmean/downmean 在模型中权重较高（如 `FAC_OI_CHG_upmean` gain=1445、`FAC_CORR_PVOL_RET_upmean` gain=1420 等），不能轻易让实时也变成全 NaN。

**修复与增量重算（2026-05-19 晚间）**：
- 研究环境 `/home/future_commodity/function_future/factor_generator.py` 已修复：`compute_up_mean` / `compute_down_mean` 改用 `pd.Series(np.where(...)).mean()`，正确跳过 NaN。
- 增量脚本 `/home/future_commodity/prepare_data/prepare_factor/recalc_updown_dce农.py` 已就绪：只重算 8 品种（A/B/C/CS/M/Y/P/LH）tick-tick 因子的 upmean/downmean，不降频其他统计量（vwap/kurtosis/...），直接 `groupby(grouper).mean()` 覆盖写。今晚运行。

**实时环境状态**：
- dev / online 的 `data_function.py` 仍保持临时对齐（`.agg(np.mean)`）。
- **明天**：研究环境增量重算完成后，恢复实时正确逻辑（改回 pandas `.mean()`），重新核对并训练模型。

---

## 六、时间显示约定

- **所有时间统一显示北京时间（UTC+8）**，不显示 UTC 或带 +00:00 后缀。
- 系统命令（如 `ps`、`ls -lt`、`date`）返回的是 UTC，解读时必须 +8h 后再呈现给用户。
- 日志中的时间戳如果是 UTC，也要换算成北京时间后再引用。

## 七、修改记录索引

- 0518 修改详情见 `/home/agent_daily/kimi/20260518.md`
- 0519 修改详情见 `/home/agent_daily/kimi/20260519.md`
