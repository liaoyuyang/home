# GroupBuilder 设计文档

## 1. 定位与目的

`GroupBuilder` 是**分组级构建器**，负责：
1. 计算**跨品种因子**（两两品种之间的价差、量比、相关性等）
2. 将**单品种因子** + **跨品种因子**拼接为 `all_factor.feather`

核心约束：
- **分组隔离**：不同分组的 `all_factor` 互不相干，存储在 `data/groups/{group_name}/all_factor/` 下
- **CPU 可控**：`build_all_factor()` 的 `n_jobs` 参数透传给 `FactorAssembler`

## 2. 类结构

```python
class GroupBuilder:
    def __init__(self, group_name: str)
    def build_all_factor(self, single_factor_source, cross_factor_source, n_jobs=1) -> dict[str, Path]
    def _load_single_factors(self, symbol, source, n_jobs=1) -> pd.DataFrame
    def compute_cross_factors(self, method="default") -> pd.DataFrame
```

## 3. 方法详解

### 3.1 `__init__(group_name)`

**输入**
- `group_name`: 分组名，如 `"油脂油粕"`

**内部状态**
- `self.symbols`: 从 `config/groups.yaml` 读取的分组内品种列表，如 `["A", "M", "Y", "P"]`
- `self.group_root`: `data/groups/{group_name}/`

**目录结构**
```
data/groups/油脂油粕/
├── cross_factors.feather          # 跨品种因子（本分组独有）
└── all_factor/
    ├── A_all_fac.feather          # A 的完整因子表
    ├── M_all_fac.feather
    ├── Y_all_fac.feather
    └── P_all_fac.feather
```

### 3.2 `build_all_factor(single_factor_source, cross_factor_source, n_jobs=1)`

**主入口**，为分组内**每个品种**构建 `all_factor.feather`。

**参数**
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `single_factor_source` | str | `"mnt"` | 单品种因子来源：`"mnt"` 从底层重新拼；`"local"` 从本地 `data/processed/` 读取已拼好的表 |
| `cross_factor_source` | str | `"local"` | 跨品种因子来源：`"local"` 读取本地已计算的 `cross_factors.feather` |
| `n_jobs` | int | `1` | 拼单品种因子时的并行度，透传给 `FactorAssembler` |

**执行流程**
1. **加载跨品种因子**
   - 若 `cross_factor_source == "local"` 且 `cross_factors.feather` 存在，则读取
   - 否则 `cross_fac = None`
   - 统一索引名为 `"ts"`

2. **遍历分组内品种**
   ```python
   for symbol in self.symbols:
       # a) 加载单品种因子
       single_fac = _load_single_factors(symbol, source=single_factor_source, n_jobs=n_jobs)
       
       # b) 拼接跨品种因子
       if cross_fac is not None:
           related_cols = [c for c in cross_fac.columns
                           if c.startswith(f"{symbol}_") or f"_{symbol}_" in c]
           single_fac = single_fac.join(cross_fac[related_cols], how="left")
       
       # c) 清理并保存
       single_fac = single_fac.replace([inf, -inf], nan)
       single_fac.to_feather(save_path)
   ```

**输出**
- `dict[str, Path]`: `{symbol: save_path}`

**跨品种因子列筛选规则**
```python
related_cols = [
    c for c in cross_fac.columns
    if c.startswith(f"{symbol}_") or f"_{symbol}_" in c
]
```

例如 A 品种的跨品种因子列：
- `A_M_closepctchg5_sub`
- `M_A_closepctchg5_sub`（若存在，也会被 `f"_{symbol}_"` 匹配到）
- `A_Y_volumediv5_diff5`
- ...

### 3.3 `_load_single_factors(symbol, source, n_jobs=1)`

加载某品种的单品种因子。

**source="mnt"（方案 B）**
- 实例化 `FactorAssembler(symbol, n_jobs=n_jobs)`
- 调用 `assembler.assemble()`
- **不再读取 mnt 的 `all_factor.feather`**，而是底层重新拼

**source="local"（回退方案）**
- 从 `data/processed/{symbol}/all_factor.feather` 读取
- 用于本地已预先拼好单品种因子的场景（如后续增量更新时）

**异常**
- 若 `source="local"` 且本地文件不存在，抛出 `FileNotFoundError`

### 3.4 `compute_cross_factors(method="default")`

计算**跨品种因子**，两两组合遍历分组内品种。

**默认计算内容**
| 因子名 | 计算方式 |
|--------|---------|
| `{vi}_{vj}_closepctchg5_sub` | `close.pct_change(5)` 差 |
| `{vi}_{vj}_closepctchg20_sub` | `close.pct_change(20)` 差 |
| `{vi}_{vj}_volumediv5_diff5` | `volume.rolling(5).sum()` 比值，再 `diff(5)` |
| `{vi}_{vj}_volumediv20_diff5` | `volume.rolling(20).sum()` 比值，再 `diff(5)` |
| `{vi}_{vj}_vcorr10` | `volume.rolling(10).corr()` |
| `{vi}_{vj}_cvcorr10_diff` | `close.rolling(10).corr(volume)` 差 |
| `{vi}_{vj}_oi5_diff` | `open_interest.pct_change(5)` 差 |

**输入数据**
- 从 `data/raw/{symbol}/1min_active.feather` 读取的 1 分钟数据
- 索引为 `ts`，列含 `open`, `high`, `low`, `close`, `volume`, `open_interest`

**输出**
- `DataFrame`，索引为 `ts`，列为跨品种因子
- 保存到 `data/groups/{group_name}/cross_factors.feather`

**注意**：两两组合是**有向的**（i < j），所以 `A_M` 和 `M_A` 不会同时出现。若后续模型需要双向因子，需修改列名生成逻辑。

## 4. 与旧方案（直接读 mnt all_factor）的对比

### 旧方案（已废弃）
```python
# _load_single_factors 旧实现
path = mnt/factor/{symbol}/all_fac/all_factor.feather
df = pd.read_feather(path)
# 问题：mnt 的 all_factor 已包含全部 8 品种的跨品种因子
#       A 在"油脂油粕"分组里会拿到 A_B、A_C 等不属于本分组的因子
```

### 新方案（当前实现）
```python
# _load_single_factors 新实现
assembler = FactorAssembler(symbol, n_jobs=n_jobs)
return assembler.assemble()
# 优点：单品种因子纯净，跨品种因子由本地分组计算后 join
```

## 5. 已知问题与待调整项

### 5.1 `compute_cross_factors` 的数据对齐

当前实现假设所有品种的 1 分钟数据**时间索引完全对齐**。若某些品种有缺失分钟（如停牌），`di["close"].pct_change(5).sub(dj["close"].pct_change(5))` 会按索引对齐，缺失值处为 NaN。这在 pandas 中是安全的，但需确认训练时是否能正确处理 NaN。

### 5.2 `compute_cross_factors` 的重复计算

每次调用 `compute_cross_factors` 都会重新计算所有跨品种因子。对于 4 品种分组：
- 组合数 = C(4,2) = 6
- 每对 7 个因子
- 总列数 = 42

计算量很小，但若有几十个分组，重复计算效率低。建议后续增加缓存机制（如按日期检查是否需要更新）。

### 5.3 `build_all_factor` 的增量更新

当前实现是**全量重建**。若每天只需 append 最新数据：
1. 单品种因子：可缓存已拼好的 `single_fac`，只 append 最新合约/日期
2. 跨品种因子：类似

增量更新逻辑较复杂，需处理合约切换日。当前先保持全量重建，后续按需优化。

### 5.4 列名冲突

若 `FactorAssembler` 生成的单品种因子列名，与 `compute_cross_factors` 生成的跨品种因子列名**恰好相同**（理论上概率极低，因为跨品种因子含 `_` 连接的两个品种代码），`join(how='left')` 会报错。

**防御措施**：可在 join 前检查列名交集，若存在则抛出异常或重命名。

### 5.5 内存峰值

`build_all_factor` 循环中，同时持有：
- `cross_fac`：整个分组的跨品种因子（~40 万行 × 42 列，~135 MB）
- `single_fac`：单个品种的完整因子（~40 万行 × ~1900 列，~6 GB）

总内存峰值约 **6-7 GB**。若机器内存紧张，可考虑：
- 逐品种释放 `single_fac`（Python GC 不保证立即释放，可用 `del single_fac; gc.collect()`）
- 使用 `float32`

## 6. 调用示例

```python
from src.group_builder import GroupBuilder

builder = GroupBuilder("油脂油粕")

# 步骤 1：计算跨品种因子（若尚未计算）
cross = builder.compute_cross_factors()

# 步骤 2：拼 all_factor（默认单线程）
saved = builder.build_all_factor(single_factor_source="mnt", n_jobs=1)
print(saved)
# {'A': PosixPath('data/groups/油脂油粕/all_factor/A_all_fac.feather'), ...}
```
