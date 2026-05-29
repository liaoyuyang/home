# experiment_tricks 8 方案批量对比结论

## 现象

昨晚跑完 8 品种 × 8 方案（5-fold）批量实验，结果与预期偏差较大：

1. **标签平滑全面翻车**：所有品种、所有基线方案加上平滑后 test_corr 全部下降
2. **方向敏感损失效果平庸**：平均 corr 低于 Huber 和 baseline
3. **分位数样本权重部分品种报 NaN**：A/B/CS/LH 的 trick8 所有 fold 的 test_corr 为 NaN
4. **Huber vs MSE 的品种分化规律得到完全验证**

## 实验设计

- **品种**：A / B / C / CS / M / Y / P / LH
- **训练截止**：2025-07-01
- **预测周期**：5min
- **方案**：
  1. baseline — MSE
  2. baseline_smooth — MSE + 标签平滑 [0.6, 0.3, 0.1]
  3. trick5_huber — Huber (alpha=0.9)
  4. trick5_huber_smooth — Huber + 标签平滑
  5. trick6_direction — 方向敏感损失（方向错了 penalty=3）
  6. trick6_direction_smooth — 方向敏感 + 标签平滑
  7. trick8_quantile_mse — 分位数压缩样本权重 + MSE
  8. trick8_quantile_huber — 分位数压缩样本权重 + Huber

## 核心数据

### 各品种最佳方案

| 品种 | 最佳方案 | test_corr | 最差方案 | test_corr | 差距 |
|------|---------|-----------|---------|-----------|------|
| A | trick5_huber | +0.0151 | baseline_smooth | -0.0131 | 0.0282 |
| B | trick5_huber | +0.0333 | baseline_smooth | -0.0146 | 0.0479 |
| C | baseline | +0.0544 | trick6_direction_smooth | +0.0117 | 0.0427 |
| CS | trick5_huber | +0.0329 | baseline_smooth | -0.0150 | 0.0479 |
| LH | baseline | +0.0621 | baseline_smooth | +0.0306 | 0.0315 |
| M | trick8_quantile_huber | +0.0255 | trick6_direction_smooth | -0.0131 | 0.0387 |
| P | trick5_huber | +0.0082 | baseline_smooth | -0.0028 | 0.0110 |
| Y | trick5_huber | +0.0194 | trick8_quantile_mse | -0.0103 | 0.0297 |

### Huber vs MSE 品种分化验证

| 品种 | MSE | Huber | winner | delta |
|------|-----|-------|--------|-------|
| LH | 0.0621 | 0.0472 | **MSE** | -0.0149 |
| C  | 0.0544 | 0.0483 | **MSE** | -0.0061 |
| B  | 0.0294 | 0.0333 | Huber | +0.0039 |
| CS | 0.0265 | 0.0329 | Huber | +0.0064 |
| M  | 0.0191 | 0.0238 | Huber | +0.0047 |
| A  | 0.0057 | 0.0151 | Huber | +0.0094 |
| P  | -0.0011 | 0.0082 | Huber | +0.0093 |
| Y  | -0.0011 | 0.0194 | Huber | **+0.0205** |

排序与 5/28 峰度分析完全一致：峰度高的 C/LH 用 MSE 更好，峰度低的 Y/M/A/P 用 Huber 提升显著。

### 标签平滑效果（smooth - 原方案）

| 基线方案 | 平均变化 | 说明 |
|---------|---------|------|
| baseline | **-0.0262** | 8/8 品种下降 |
| trick5_huber | **-0.0262** | 8/8 品种下降 |
| trick6_direction | **-0.0201** | 8/8 品种下降 |

### 分位数样本权重效果

- A/B/CS/LH 的 trick8 所有 fold 报 NaN，说明实现有 bug
- 有值的品种（C/M/P/Y）中，相比不加权重提升微弱甚至为负

### 方向敏感损失效果

- 平均 test_corr = 0.0172，低于 Huber（0.0285）和 baseline（0.0244）
- 加上标签平滑后更差（0.0172 → -0.0029）

## 根因分析

### 标签平滑为何翻车

推理端已做 0.6/0.3/0.1 三期加权平滑，训练端再做同样的平滑等于**二次平滑**：
- 训练目标被过度平滑，高频信号被抹除
- 模型学到的"趋势"与推理端的真实目标失配

### trick8 NaN 问题

待排查。可能原因：
- 分位数权重计算后某些 fold 的 train_set 全为 0 权重
- 或样本权重与 LightGBM 的 `huber` objective 组合时触发数值问题

### high_conf_dir_acc 指标废了

`|pred| > 0.9` 的阈值在标准化标签（std=1）且 test_corr 仅 0.01~0.06 的场景下，pred 的标准差 ≈ 0.05，0.9 是 **18 个标准差**，几乎永远达不到。

## 结论 / Action

1. **按品种配置损失函数直接落地**
   ```python
   LOSS_CONFIG = {
       'C':  {'objective': 'regression', 'metric': 'rmse'},
       'LH': {'objective': 'regression', 'metric': 'rmse'},
       'Y':  {'objective': 'huber', 'metric': 'l1', 'alpha': 0.9},
       'M':  {'objective': 'huber', 'metric': 'l1', 'alpha': 0.9},
       'A':  {'objective': 'huber', 'metric': 'l1', 'alpha': 0.9},
       'P':  {'objective': 'huber', 'metric': 'l1', 'alpha': 0.9},
       'CS': {'objective': 'huber', 'metric': 'l1', 'alpha': 0.9},
       'B':  {'objective': 'huber', 'metric': 'l1', 'alpha': 0.9},
   }
   ```

2. **标签平滑放弃**：推理端已做平滑，训练端不再重复做

3. **方向敏感损失不优先集成**：效果一般，且自定义 objective 维护成本高

4. **分位数样本权重放弃**：有 bug + 效果微弱

5. **high_conf_dir_acc 阈值修复**：改为 0.3 或按分位数取 top 20%

## 相关文件

- 实验脚本：`/home/strategy_res/single/dce_农/20250701/experiment_tricks.py`
- 实验结果：`/home/experiment_results_20250701.csv`
- 标签分析：`/home/strategy_res/single/dce_农/20250701/analyze_label_distribution.py`
