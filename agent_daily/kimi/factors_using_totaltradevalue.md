# 依赖 TotalTradeValue / TotalTradeVolume 的因子清单

> 目标：把 `diff()` 逻辑改为直接使用单笔列 `turnover` / `volume`，消除跨日/跨批次负跳变。

## tick_factors.py 中用到 TotalTradeValue / TotalTradeVolume 的因子

### 需要修改的（使用了 .diff()，会产生负跳变）

| 函数名 | 当前代码（关键行） | 建议改为 | 说明 |
|--------|-------------------|----------|------|
| **ILLIQ** | `data['mid_price'].diff() / data['TotalTradeValue'].diff() * 1e8` | `data['mid_price'].diff() / data['turnover'] * 1e8` | `turnover` 是单笔成交额 |
| **MPB** | `data['TotalTradeValue'].diff() / (data['TotalTradeVolume'].diff() * multiplier)` | `data['turnover'] / (data['volume'] * multiplier)` | `turnover`/`volume` 已是单笔 |
| **CORR_PVOL_RET** | `data['TotalTradeVolume'].diff()` | `data['volume']` | 单笔成交量 |
| **STREN** | `data['TotalTradeVolume'].diff(10)` | `data['volume'].rolling(window=10, min_periods=1).sum()` | 近10笔成交量之和 |
| **VOL_FLU** | `data['TotalTradeVolume'].diff().rolling(window=240).std()` | `data['volume'].rolling(window=240, min_periods=1).std()` | 近240笔成交量std |
| **QUA** | `data['TotalTradeVolume'].diff(10)` | `data['volume'].rolling(window=10, min_periods=1).sum()` | 近10笔成交量之和 |
| **shortQUA** | `data['TotalTradeVolume'].diff(10)` | `data['volume'].rolling(window=10, min_periods=1).sum()` | 近10笔成交量之和 |
| **midQUA** | `data['TotalTradeVolume'].diff(10)` | `data['volume'].rolling(window=10, min_periods=1).sum()` | 近10笔成交量之和 |
| **Price_Divergence** | `data['TotalTradeVolume'].diff(30)` / `diff(60)` | `data['volume'].rolling(window=30, min_periods=1).sum()` / `rolling(60).sum()` | 近30/60笔成交量之和 |
| **MFI** | `data['TP'] * data['TotalTradeVolume'].diff()` | `data['TP'] * data['volume']` | 单笔成交量 |
| **CMF** | `data['TotalTradeVolume'].diff()` | `data['volume']` | 单笔成交量 |

### 不需要修改的（使用原始值本身，不涉及 diff）

这些因子用 `TotalTradeValue` / `TotalTradeVolume` 的**绝对值**（当日累计），在 `level2_all` 数据修复后逻辑正确：

| 函数名 | 当前代码（关键行） | 说明 |
|--------|-------------------|------|
| **resiliency** | `(HighPrice - LowPrice) / (TotalTradeVolume / OpenInterest)` | 用当日累计成交量，不涉及 diff |
| **VWAP_Deviation** | `mid_price / (TotalTradeValue / TotalTradeVolume)` | 用当日累计值，不涉及 diff |
| **OI_V_DIV** | `TotalTradeVolume.pct_change() / OpenInterest` | 用累计值算增速，数据修复后正确 |

> **注意**：`OI_V_DIV` 虽然用了 `pct_change()`（内含 diff），但它的语义是累计增速，且只涉及 `TotalTradeVolume` 不涉及 `TotalTradeValue`。在数据修复后（无混合），跨日 `pct_change()` 会得到一个很大的正值（因为新一天从 0 开始），这不是负跳变，但可能也需要留意。

---

## 修改后需要重跑的因子文件

上述 11 个需要修改的因子，每个都有约 19 个变体（`downmean`/`upmean`/`MADmean`/`Mstdwap`/`Volraiseap`/`biddommean`/`askdommean`/`corrAskwap`/`corrBidwap`/`vwap`/`mean`/`std`/`min`/`max`/`last`/`first`/`kurtosis`/`skewness`/`TrendRevmean`）。

**删除路径**：`/mnt/Data/writable/liaoyuyang/factor/{SYMBOL}/t2t/{contract}/`

需要删的文件名前缀：
```
ILLIQ_*
MPB_*
CORR_PVOL_RET_*
STREN_*
VOL_FLU_*
QUA_*
shortQUA_*
midQUA_*
Price_Divergence_*
MFI_*
CMF_*
```

改完 `tick_factors.py` 后，运行 `calc_factors_dce农.py` 的 `calc_t2t` 重新生成即可。
