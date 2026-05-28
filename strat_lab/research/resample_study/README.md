# Resample Study — 降频研究框架

## 预留功能

此目录用于存放将分钟数据降频到更高时间粒度后的研究脚本。

### 计划支持的降频粒度

1. **5 分钟**
2. **10 分钟**
3. **日频**

### 研究内容

- 降频后因子计算逻辑的调整
- 降频后因子有效性的对比（分钟 vs 5min vs 日频）
- 不同频率下的模型表现差异

### 接口约定

```python
from src.data_loader import DataLoader

dl = DataLoader()
df_1min = dl.load_main_1min("A", from_local=True)

# 降频到 5min
df_5min = df_1min.resample('5min', label='right', closed='right').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum',
    'turnover': 'sum',
})
```
