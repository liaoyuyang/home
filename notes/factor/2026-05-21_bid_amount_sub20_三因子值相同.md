# FAC_bid_amount_sub20 三因子实时值完全相同

## 现象

核对 c2605 时，发现实时环境中 `FAC_bid_amount_sub20` 的三个降频因子在某些时间点值**完全一样**：

| 时间 | Mstdwap | askdommean | corrBidwap |
|------|---------|------------|------------|
| 22:45 | -16769.0 | -16769.0 | -16769.0 |
| 22:48 | -16765.0 | -16765.0 | -16765.0 |

此外：
- `askdommean` 大量 NaN（最近 20 个时间点中 10 个为 NaN）
- 实时值与数据库全量 tick 手动计算结果**完全不符**（手动算 22:48:00 得到 Mstdwap=NaN, askdommean=-2128.94, corrBidwap=NaN）

## 根因分析（进行中）

**已确认**：
1. `data_service.py` 中 `pkg['tick'][inst] = tick_df[mask].iloc[-1000:].copy()`，策略只拿到最近 1000 条 tick，不是全量。
2. 手动用数据库 22:47 的 198 条 tick 复现计算，结果和实时保存值完全不同。
3. 三个因子走完全不同的降频路径（`resample_Mstd` / `resample_askdommean` / `resample_corrBid`），数学上不可能总是相等。

**待确认**：
- 实时 `data_package['tick']` 的 1000 条 tick 具体分布？
- `_prepare_data` 中的 `pd.concat` 是否因索引不对齐导致 `extra_cols`（M_std、bvall、corrBidwap）异常？
- `resample_agg` 的 `ensure_1d_array` 是否在某些边界条件下返回了相同的 fallback 值？

## 结论 / Action

- [ ] **高**：在 `resample_Mstd`、`resample_askdommean`、`resample_corrBid` 中加临时日志，输出当前分钟的 `weight_sum`、`weighted_sum`、以及 `_prepare_data` 返回的 `merged` 中依赖列的统计信息，定位为什么三个值会相同。
- [ ] **高**：对比 `data_package['tick']` 的 1000 条 tick 和数据库对应时间段的 tick，确认截断逻辑是否引入了系统性偏差。
- [ ] **中**：评估 `iloc[-1000:]` 截断对 rolling window（120 tick, min_periods=100）因子的影响，是否需要延长截断长度或改为时间截断。
