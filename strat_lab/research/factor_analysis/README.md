# Factor Analysis — 因子分析框架

## 预留功能

此目录用于存放因子有效性分析的相关脚本和 notebook。

### 计划支持的分析类型

1. **IC 分析**
   - 输入：`all_factor.feather` + `rtn_5`
   - 输出：各因子的 Rank IC、IC 均值、IC 标准差、IC_IR

2. **IC 衰减分析**
   - 输入：`all_factor.feather` + 多周期 rtn（rtn_1, rtn_5, rtn_10）
   - 输出：因子预测能力随时间衰减曲线

3. **因子拥挤度**
   - 输入：`all_factor.feather`
   - 输出：因子截面相关性、因子波动率、异常值比例

4. **因子相关性时变分析**
   - 输入：`all_factor.feather`
   - 输出：滚动窗口内因子相关性矩阵的变化

### 接口约定

```python
from src.data_loader import DataLoader
from src.config_manager import ConfigManager

dl = DataLoader()
cm = ConfigManager()

# 加载某分组下某品种的因子表
df = dl.load_all_factor(group_name="油脂油粕", symbol="A")
```
