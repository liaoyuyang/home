# FactorAssembler 设计文档

## 1. 定位与目的

`FactorAssembler` 负责**从底层单品种因子 feather 重新拼表**，替代原先直接读取 mnt `all_factor.feather` 的做法。

核心约束：
- **只读 mnt**，不写 mnt
- **分组隔离**：不同分组各自拼自己的 `all_factor`，互不污染
- **CPU 可控**：并行度 `n_jobs` 默认 1，显式传入才可提高

## 2. 与参考代码的对照

参考实现：`future_commodity/prepare_data/prepare_factor/.../make_main_factor_dataframe_dce农.py`

| 步骤 | 参考代码 | 本实现 (`FactorAssembler`) | 备注 |
|------|---------|---------------------------|------|
| 读取主力合约表 | `df_symbol = dl.load_contract_info(symbol)` | `self.contract_info = self.dl.load_contract_info(symbol, from_local=True)` | 统一从本地 `data/raw/{symbol}/contract_info.csv` 读取 |
| 读取因子列表 | `dl.load_tf_lst('tick_tick')` / `dl.load_f_lst('min_min')` | 从 `config/factor_config.yaml` 读取，t2t 做笛卡尔积 | 见第 5 节「因子名生成」 |
| 按合约读因子 | `Parallel(n_jobs=30)` 读单个因子，再 `pd.concat(axis=1)` | `n_jobs` 可控，`Parallel` 仅在 `n_jobs>1` 时启用 | 默认单线程，安全 |
| 合约过滤 | `merge(df_symbol[['instrument','trade_date']])` | 完全一致，只保留主力合约期间数据 | 关键步骤，不可省略 |
| 三类因子拼接 | `t2tfac.join(t2mfac).join(m2mfac)` | 完全一致 | 左连接，t2t 为基准 |
| 最终索引 | `set_index('datetime')` | `index.name = 'ts'` | 统一为 `ts` |

## 3. 类结构

```python
class FactorAssembler:
    def __init__(self, symbol: str, n_jobs: int = 1)
    def assemble(self) -> pd.DataFrame           # 主入口
    def _assemble_mode(self, mode, fac_lst, instruments) -> pd.DataFrame
    def _read_factors_for_instrument(self, instrument, mode, fac_lst) -> pd.DataFrame
    def _read_single_factor(self, factor_name, instrument, mode) -> pd.DataFrame
    def _get_instruments(self) -> list[str]
```

### 3.1 `__init__(symbol, n_jobs=1)`

**输入**
- `symbol`: 品种代码，如 `"A"`
- `n_jobs`: 拼单合约内多因子时的并行度。默认 1（和同事共用机器时的安全值）。

**内部状态**
- `self.contract_info`: 主力合约切换表，含 `instrument`, `trade_date`, `start_date`, `end_date` 等列
- `self.instruments_lst`: 该品种历史上所有出现过的**主力合约**列表（从 `contract_info["instrument"]` 提取）
- `self.mf_fac_lst`: m2m 因子名列表（~59 个）
- `self.tmf_fac_lst`: t2m 因子名列表（~9 个）
- `self.tf_fac_lst`: t2t 因子名列表（~92 个 base × 20 modifiers = ~1840 个）

**注意**：t2t 因子名通过 `base_functions × modifiers` 笛卡尔积生成，与参考代码 `load_tf_lst()` 逻辑一致。若 `config/factor_config.yaml` 中的 `modifiers` 列表与实际 mnt 存储不一致，会导致大量因子找不到（静默跳过）。

### 3.2 `assemble() -> pd.DataFrame`

**执行流程**
1. 调用 `_assemble_mode("t2t", ...)` 拼 t2t 因子
2. 取 t2t 成功处理的合约列表作为 `valid_instruments`
3. 用同一批合约拼 t2m、m2m
4. `t2tfac.join(t2mfac, how='left').join(m2mfac, how='left')`
5. `reset_index().set_index("datetime")`，索引名改为 `"ts"`

**输出**
- `DataFrame`，索引为 `ts`（datetime），列为该品种全部单品种因子（**不含跨品种因子**）
- 列数约为 59 + 9 + ~1840 = ~1900 列

**异常**
- 若 t2t 为空，抛出 `ValueError`（说明底层数据缺失或因子名配置错误）

### 3.3 `_assemble_mode(mode, fac_lst, instruments=None) -> pd.DataFrame`

按模式（`m2m`/`t2m`/`t2t`）读取并拼接因子。

**流程**
1. 遍历 `instruments`（默认为全部合约）
2. 对每个合约调用 `_read_factors_for_instrument()`
3. 纵向 `pd.concat(dfs)`
4. `reset_index(names="datetime").set_index(["datetime", "instrument"])`

**输出**
- MultiIndex DataFrame，索引为 `(datetime, instrument)`，列为该模式下的所有因子

### 3.4 `_read_factors_for_instrument(instrument, mode, fac_lst) -> pd.DataFrame`

读取**单个合约**的所有指定因子，横向拼接。

**并行策略**
- `n_jobs == 1`：顺序读取，带 `tqdm` 进度条
- `n_jobs > 1`：`joblib.Parallel(n_jobs=self.n_jobs)` 并行读取

**注意**：进度条在 leave=False 模式下可能不会保留输出。若需要保留日志，可改用 `tqdm(..., file=sys.stdout)` 或关闭进度条。

### 3.5 `_read_single_factor(factor_name, instrument, mode) -> pd.DataFrame`

读取**单个因子**的 feather 文件，并过滤到主力合约期间。

**路径规则**
| 模式 | 路径 |
|------|------|
| t2t | `mnt/factor/{symbol}/t2t/{instrument}/{factor_name}@{instrument}.feather` |
| t2m | `mnt/factor/{symbol}/t2m/{factor_name}@{instrument}.feather` |
| m2m | `mnt/factor/{symbol}/m2m/{factor_name}@{instrument}.feather` |

**过滤逻辑（关键）**
```python
# 1. 读取 feather
df = pd.read_feather(path)
df["trade_date"] = df["datetime"].dt.strftime("%Y-%m-%d")

# 2. 只保留该合约在主力合约表中的日期段
df = df.merge(
    self.contract_info[["instrument", "trade_date"]].dropna(),
    on=["instrument", "trade_date"],
    how="inner",
)
```

**列名处理**
- t2t：`FAC_{factor_name}`（加 `FAC_` 前缀）
- t2m/m2m：`factor_name`（无前缀）

这与参考代码一致。前缀差异来自底层因子计算时的命名约定。

**异常处理**
- 文件不存在 → 返回空 DataFrame（静默跳过）
- 任何其他异常 → 返回空 DataFrame（静默跳过）

> **设计决策**：静默跳过而非报错，是因为不同合约的因子覆盖不完全一致（如远月合约可能缺少某些 tick 因子）。若改为报错，需确保因子列表 100% 准确。

### 3.6 `_get_instruments() -> list[str]`

从 `contract_info["instrument"]` 列提取所有唯一值并排序。

> **注意**：仅取 `instrument` 列（当前主力合约），**不包含** `instrument_next`。参考代码中 `instruments_lst` 也只包含实际当过主力的合约。

## 4. 已知问题与待调整项

### 4.1 因子名配置可能不匹配

`config/factor_config.yaml` 是从 `future_config/basic_config/` 复制过来的。若 mnt 上的实际因子是用**不同版本**的配置生成的，则：
- 某些因子文件不存在 → 被静默跳过 → 列数少于预期
- 某些新因子未在配置中 → 丢失

**建议**：增加一个「扫描模式」——不从配置文件读取，而是遍历 mnt 目录动态发现因子名。这样更健壮，但会丧失对因子列表的显式控制。

### 4.2 进度条输出可能被清除

`tqdm(..., leave=False)` 在部分终端环境下会清除已完成的进度条。若需要审计日志，建议：
- 改为 `leave=True`
- 或关闭 tqdm，改用简单的 `print(f"[{symbol}] {instrument} {mode} done")`

### 4.3 内存占用

t2t 因子约 1840 列，单个合约约 1.5 万行，单个合约的 t2t DataFrame 约 `15000 × 1840 ≈ 2700 万元素`。按 float64 计算约 216 MB。26 个合约全部载入后约 5.6 GB。

实际流程中：
- `_read_factors_for_instrument` 只保留单个合约的数据（216 MB）
- `_assemble_mode` 纵向拼接后，26 个合约约 40 万行 × 1840 列 ≈ 5.6 GB
- 三类因子 join 后总内存可能达 **~6-8 GB**

**优化方向**：
- 使用 `float32` 而非默认 `float64`（可减半内存）
- 分批次处理合约，最后 `pd.concat`

### 4.4 `contract_info` 的 `trade_date` 格式

`load_contract_info` 中 `trade_date` 被格式化为 `"YYYY-MM-DD"` 字符串。 feather 中的 `datetime` 是带时间的 datetime64。`dt.strftime` 后 merge 是安全的，但字符串比较比 datetime 比较慢。若数据量大，可考虑统一为 datetime 类型后按日期部分 merge。

### 4.5 索引名不一致

最终输出索引名为 `"ts"`，但 `_assemble_mode` 返回的索引名为 `"datetime"`。`group_builder.py` 在拼接跨品种因子时会统一处理，需确保 `cross_factors.feather` 的索引名也是 `"ts"`。

## 5. 因子名生成详解

### 5.1 t2t（tick-tick）

```yaml
# config/factor_config.yaml 节选
tick_tick_fac:
  modifiers: [MADmean, Mstdwap, TrendRevmean, ...]  # 20 个
  fac_func_lst: [atr, RVar, RSkew, ADTMMA, ...]      # 92 个
```

生成规则：
```python
[f"{func}_{mod}" for func in fac_func_lst for mod in modifiers]
# 例：ADTMMA_MADmean, ADTMMA_Mstdwap, atr_MADmean, ...
```

### 5.2 t2m（tick-min）

无 `modifiers`，直接使用：
```python
["buy_trend", "buy_trend1", ..., "TMB"]
```

### 5.3 m2m（min-min）

无 `modifiers`，直接使用：
```python
["ASI", "BOLLING", "MAdiff_Vol_div", ...]
```

## 6. 调用示例

```python
from src.factor_assembler import FactorAssembler

# 默认单线程
asm = FactorAssembler("A", n_jobs=1)
df = asm.assemble()
print(df.shape)  # (N_minutes, ~1900)

# 若机器空闲，可适度提高并行度（但需谨慎）
asm = FactorAssembler("A", n_jobs=4)
df = asm.assemble()
```
