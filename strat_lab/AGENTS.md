# Strat Lab — 代码风格与工程约定

> 本文档由 Agent 分析用户现有代码库后总结，后续所有新增代码必须遵循。

---

## 一、命名约定

### 1.1 变量命名

| 类型 | 风格 | 示例 |
|------|------|------|
| 普通变量 | snake_case，短而小写 | `df`, `data`, `symbol`, `instrument`, `min_data` |
| DataFrame | 小写缩写或描述性名词 | `df`, `df_symbol`, `df_min`, `data1m`, `rtn_df` |
| Series | 与 DataFrame 一致或单数 | `pred`, `position_`, `daily_max_contract` |
| 布尔标志 | 无前缀，直接用形容词或动词过去分词 | `open_drop`, `close_drop`, `updated_any`, `trigger` |
| 全局/类常量 | 全大写下划线 | `_TRADING_DAYS`, `TOL_ABS`, `MAX_FILL` |
| 私有/内部变量 | 单下划线前缀 | `_running`, `_thread`, `_config`, `_cache` |
| 临时/中间变量 | 直接命名，不 t_ / tmp_ | `fac`, `res`, `lst`, `results` |

**反例**：不要用 `is_open_drop`（实际代码用 `open_drop`），不要用 `dataFrame`（用 `df`）。

### 1.2 函数/方法命名

- **全部 snake_case**
- **动词开头**：`load_`, `calc_`, `process_`, `sync_`, `compute_`, `run_`, `build_`, `get_`, `drop_`
- **数据处理函数**：`time_scale_df`, `round_datetime_to_minute`, `process_contract`
- **因子函数**：与因子名完全一致，首字母大写（如 `ASI`, `BOLLING`, `drawback`）
- **私有方法**：单下划线前缀 `_load_research_factors`, `_is_in_trade_session`

### 1.3 类命名

- **CamelCase**，首字母大写
- **后缀习惯**：
  - 加载类：`DataLoader`, `FutureDataLoader`
  - 生成器：`FactorGenerator`
  - 管理器：`ConfigManager`, `PathResolver`
  - 流水线：`TrainingPipeline`, `BacktestPipeline`
  - 处理器：`Pretrainer`, `ModelBacktester`
  - 过滤器：`FactorFilter`

### 1.4 文件名命名

- **脚本**：snake_case，描述功能
  - `calc_factors_dce农.py`, `make_main_factor_dataframe_dce农.py`
  - `rebuild_level2_all_dce农.py`, `sync_mkt_data.py`
- **数据文件**：
  - feather: `{name}.feather`
  - csv: `{symbol}_{timestamp}.csv`, `main_{symbol}.csv`
  - 模型: `kfold_fold{i}_{group}.lgb`, `{symbol}_pred{label}_{date}_v0/`

---

## 二、代码结构习惯

### 2.1 函数长度

- **偏好中等长度函数**（30-100 行），一个函数做一件事
- 复杂逻辑会拆成多个小函数，但不会过度拆分
- 因子计算函数通常很短（10-30 行），直接返回 Series

### 2.2 类 vs 函数

- **数据处理/流水线**：用类封装（`DataService`, `FactorGenerator`, `Pretrainer`）
- **纯计算/工具**：用函数（`compute_vwap`, `compute_up_mean`, `time_scale_df`）
- **脚本入口**：用 `if __name__ == "__main__": main()`

### 2.3 注释风格

- **中文注释为主**，技术术语保留英文
- **docstring 用简洁描述**，非 Google/reST 严格格式：
  ```python
  def compute_up_mean(factor_series, mid_price):
      """
      修复：用 pd.Series.mean() 跳过 NaN（numpy mean 不跳过 NaN 会导致大量 NaN 输出）
      """
  ```
- **关键逻辑必有注释**，尤其是修复/陷阱：
  ```python
  # 给DB写入留缓冲
  time.sleep(1.0)
  
  # 边界置NaN（避免未来信息泄露）
  X_valid[:60*4] = np.nan
  ```
- **行内注释用 `#`**，与代码隔两个空格

---

## 三、数据处理习惯

### 3.1 Pandas 使用

- **倾向 copy()**：`data = data.copy()` 在因子计算中非常常见
- **链式操作适中**：不会一行写 5 个方法链，通常 2-3 个
- **inplace=False**：几乎不用 `inplace=True`，总是赋值给新变量
- **索引处理**：
  - 时间序列优先用 `datetime` 索引
  - `set_index('datetime')` / `reset_index()` 频繁切换
  - 对齐时用 `.reindex()` 或 `.loc[common_index]`

### 3.2 缺失值处理

- **NaN 不静默填充**：暴露问题，不掩盖
  - 禁止 `ffill` / `bfill` 让实时和研究"看起来一致"
  - `shift()` 错位时 diff 变大是对的，要修数据源而不是修计算层
- **数值替换**：`df.replace([np.inf, -np.inf], np.nan)` 很常见

### 3.3 时间处理

- **datetime 列名**：`datetime`, `ts`, `trade_date`
- **格式转换**：`pd.to_datetime()` 统一处理
- **时间过滤**：用 `.dt.time` 或 `time_scale_df()` 函数
- **北京时间**：所有时间统一显示 UTC+8，不显示 +00:00

---

## 四、错误处理

### 4.1 风格

- **常用 try/except**，但通常是粗粒度包裹整块逻辑
- **文件不存在时抛 FileNotFoundError**，不返回 None  silently
- **数据校验用 assert**：
  ```python
  assert self.train_data.datetime.max() <= self.train_end_date, "训练集包含超出训练截止日期的数据！"
  ```

### 4.2 警告信息

- **print 为主**，不用 logging 模块（除了 orchestrator 的日志）
- **格式**：`[模块/类] 信息`，如 `[DataService] 初始化完成`
- **错误前缀**：`❌`, `⚠️`, `WARNING:`

---

## 五、Import 组织

```python
# 1. 标准库
import os
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime, time

# 2. 第三方库
import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm.auto import tqdm
from joblib import Parallel, delayed

# 3. 项目内部模块
import function_future.DataLoader as DL
from src.config_manager import ConfigManager

# 4. 全局设置
warnings.filterwarnings("ignore")
pd.set_option("future.no_silent_downcasting", True)
```

---

## 六、工程组织习惯

### 6.1 数据文件组织

- **中间数据 feather 化**：大数据量用 `.feather`，小数据/人可读用 `.csv`
- **按品种/合约分目录**：`factor/{symbol}/m2m/`, `data/1min/{symbol}/`
- **分组产物隔离**：`factor/MIX_INDEX/main_mix_dce农.feather`

### 6.2 配置与硬编码边界

- **环境相关配置放 yaml/json**：路径、端点、合约列表
- **算法参数放 yaml**：LightGBM 参数、阈值、过滤条件
- **交易规则硬编码**：如 `11:31-13:30` 午休判断，直接写在代码里

### 6.3 Notebook 习惯

- **Cell 分块清晰**：加载 → 配置 → 数据 → 模型 → 回测 → 画图
- **Markdown cell 做标题说明**
- **配置变量集中在一个 cell**：`symbol = 'A'`, `train_end_date = '2025-01-01'`

---

## 七、关键红线（绝对不能违反）

1. **`weighted_s` 必须用 `shift()`，绝对禁止 `reindex + ffill`**
2. **不能掩盖缺失**：禁止用 `ffill` / `bfill` / `reindex(...).ffill()` 让实时和研究"看起来一致"
3. **错位要暴露**：如果 `shift()` 因缺失分钟而错位，diff 变大是对的，说明要修 data_service / 数据源
4. **所有时间统一北京时间（UTC+8）**
5. **遇到不确定的业务逻辑先问用户，不擅自推断**

---

## 八、与本项目相关的新约定

在 `strat_lab` 中，除遵循上述习惯外，新增以下约定：

1. **路径解析必须通过 `src/path_resolver.py`**，不直接写字符串路径
2. **分组名用中文**（如 `油脂油粕`），与现有习惯保持一致
3. **本地数据目录与 mnt 目录结构尽量镜像**，降低心智负担
4. **新增脚本用 argparse 提供命令行参数**，支持 `--group`、`--symbol`、`--all-symbols`
5. **pipeline 脚本编号**：`01_`, `02_` ... 表示执行顺序

## 九、拼表逻辑与 CPU 控制约定

### 9.1 all_factor 拼表原则

**禁止直接读取 mnt 的 `all_factor.feather`**。mnt 侧的 `all_factor` 已包含全部 8 品种的跨品种因子，会导致分组隔离失效。

**正确做法**：
1. 单品种因子：从底层 `m2m/t2m/t2t` 重新拼（`FactorAssembler`）
2. 跨品种因子：由 `GroupBuilder.compute_cross_factors()` 按分组计算
3. 拼接：`single_fac.join(cross_fac[related_cols], how='left')`

### 9.2 CPU 并行度控制

**和同事共用机器，CPU 使用必须显式可控**。

- **默认 `n_jobs=1`**（单线程），pipeline 脚本和类构造函数的默认值必须为 1
- 提高并行度必须显式传入：`--n-jobs 4` 或 `n_jobs=4`
- `FactorAssembler` 中 `Parallel` 仅在 `n_jobs > 1` 时启用
- 禁止写死 `Parallel(n_jobs=30)` 等大规模并行

### 9.3 因子名生成

`t2t` 因子通过 `base_functions × modifiers` 笛卡尔积生成，与 `DataLoader.load_tf_lst()` 逻辑一致。

- `m2m` / `t2m`：无 modifiers，直接用 `fac_func_lst`
- `t2t`：有 modifiers，需笛卡尔积
- 配置来源：`config/factor_config.yaml`

### 9.4 合约过滤（不可省略）

拼表时必须按 `contract_info` 过滤，**只保留合约在主力期间的数据**：
```python
df.merge(
    contract_info[["instrument", "trade_date"]].dropna(),
    on=["instrument", "trade_date"],
    how="inner",
)
```
省略此步会导致合约切换时数据重叠。
