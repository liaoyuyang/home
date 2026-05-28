# 策略因子更新流程

## 概述
每次更换新模型后，需要更新四个品种（A、M、P、Y）主程序文件中的因子代码。

## 重要原则
**必须**通过调用 `/home/strategy_to_load/PAMY_dce/analyze_model_factors.py` 来生成因子代码，
直接手动编辑容易出错（如方法名拼写错误：`up_shadow_5()` vs `up_shadow_5std()`）。

## 自动化更新流程

### 方法：使用自动化脚本（推荐）

```bash
cd /home/strategy_to_load/PAMY_dce
python3 auto_update_factors.py
```

脚本会自动：
1. 遍历 A、M、P、Y 四个品种
2. 调用 `analyze_model_factors.py` 分析每个品种的模型
3. 提取生成的因子定义代码
4. 精确定位并替换主程序中 `generate_factor_dataframe` 函数的因子部分
5. 保留所有其他代码不变

### 手动验证步骤

如需手动验证某个品种的因子代码：

```bash
cd /home/strategy_to_load/PAMY_dce
python3 analyze_model_factors.py
```

查看输出中的因子定义，确认与主程序文件中一致。

## 文件位置

- **分析脚本**: `/home/strategy_to_load/PAMY_dce/analyze_model_factors.py`
- **自动化脚本**: `/home/strategy_to_load/PAMY_dce/auto_update_factors.py`
- **主程序文件**:
  - `/home/strategy_to_load/PAMY_dce/main_strategy_A.py`
  - `/home/strategy_to_load/PAMY_dce/main_strategy_M.py`
  - `/home/strategy_to_load/PAMY_dce/main_strategy_P.py`
  - `/home/strategy_to_load/PAMY_dce/main_strategy_Y.py`

## 注意事项

1. **不要手动编辑** `generate_factor_dataframe` 函数中的因子代码
2. **总是使用** `analyze_model_factors.py` 生成的代码
3. 运行自动化脚本前确保模型文件已放入 `models/` 目录
4. 脚本会保留主程序的其他部分（导入、配置、主循环等）不变

## 故障排查

如果更新后出现问题：
1. 检查 `analyze_model_factors.py` 是否能正常运行
2. 检查模型文件是否在正确的位置
3. 对比自动化脚本更新前后的代码差异
