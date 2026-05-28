# Volraiseap 等基于阈值采样的 modifier 存在结构性漂移

## 现象

月度稳定性筛选发现 `FAC_CORR_PVOL_RET_Volraiseap` 均值漂移 R²=0.916，月度均值从 2021 年的 ~4.0 持续攀升到 2025 年的 ~11.5（涨了近 3 倍）。同类被淘汰因子还有 `CORR_PVOL_RET_vwap/max`、`MOFI_kurtosis`、`STREN_kurtosis` 等。

## 根因

### 1. Modifier `Volraiseap` 的采样阈值是绝对倍数

```python
# factor_generator.py:150
def compute_Volraiseap(group):
    weights = (group['TotalTradeVolume'].diff() > 1.5 * group.volume_avg).astype(int)
    weighted_value = (group['factor_value'] * weights).sum() / weights.sum()
    return weighted_value
```

`Volraiseap` 只在**成交量 > 1.5 × 日内平均成交量**的 bar 上采样。随着豆一市场整体活跃度从 2021 到 2025 持续上升：

- 2021 年平均成交量 800 手/分钟 → "激增"门槛 1200 手
- 2025 年平均成交量 2500 手/分钟 → "激增"门槛 3750 手

**同一个"1.5倍"阈值，选出来的 bar 完全不是一回事。** 2025 年被采样的都是大单涌入的极端时刻，而 2021 年只是普通放量。因子分布随市场活跃度系统性右移。

### 2. 基础因子 `CORR_PVOL_RET` 本身也有量纲隐患

```python
# tick_factors.py:679
CORR_PVOL_RET = (data['volume'] - data['volume'].rolling(window=120).mean()) \
    / (RET.rolling(window=120).std() * data['volume'].rolling(window=120).std())
```

注意：虽然叫 `CORR_PVOL_RET`，但**实际不是相关系数**，而是标准化后的成交量偏差。分母中的 `volume.std` 随市场整体成交量膨胀而变大，导致比值中枢长期不稳。

## 结论 / Action

- **短期（已落地）**：用月度稳定性筛选（`mean_r2_thresh=0.4`）把这类漂移因子淘汰掉。新版 notebook 已跑通，A 品种从 1865 → 134 个因子。
- **中期（待验证）**：跑新版 vs 原版的训练和回测对比，确认淘汰这些因子后模型效果是否提升或至少不下降。
- **长期（大工程）**：改造 modifier 构造逻辑，把绝对阈值（`1.5×均值`）改为**日内相对排名**（如前 10% 成交量 bar）或**横截面 z-score**，从根本上消除市场活跃度这个隐藏变量。
- **扩展排查**：同类 modifier 可能还有 `biddommean/askdommean`（基于买卖盘深度绝对比较）、`Mstdwap`（基于波动率加权）、`TrendRevmean`（基于趋势反转期）等，需要逐一检查是否存在类似问题。
