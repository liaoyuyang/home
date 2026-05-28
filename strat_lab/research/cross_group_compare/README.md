# Cross Group Compare — 跨分组对比框架

## 预留功能

此目录用于对比同一品种在不同分组中的因子/模型表现差异。

### 研究内容

- 同一品种（如 A）在「油脂油粕」和「农产品」分组中的因子差异
- 跨品种因子对该品种模型贡献度的对比
- 不同分组下同一品种的信号一致性分析

### 接口约定

```python
from src.data_loader import DataLoader

dl = DataLoader()

# 加载同一品种在不同分组下的因子表
df_group1 = dl.load_all_factor(group_name="油脂油粕", symbol="A")
df_group2 = dl.load_all_factor(group_name="农产品", symbol="A")

# 对比差异
diff_cols = set(df_group1.columns) ^ set(df_group2.columns)
```
