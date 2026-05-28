# Pipeline 04：拼因子表

## 1. 定位

`pipeline/04_make_factor_df.py` 是拼因子表的**CLI 入口脚本**，调用 `GroupBuilder.build_all_factor()` 完成实际工作。

## 2. 用法

```bash
# 默认单线程，从 mnt 底层重新拼单品种因子
python pipeline/04_make_factor_df.py --group 油脂油粕

# 显式指定并行度（和同事共用机器时请谨慎）
python pipeline/04_make_factor_df.py --group 油脂油粕 --n-jobs 4

# 从本地已拼好的单品种因子读取（跳过 FactorAssembler）
python pipeline/04_make_factor_df.py --group 油脂油粕 --single-source local
```

## 3. CLI 参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--group` | str | 是 | - | 分组名称，如 `油脂油粕` |
| `--single-source` | str | 否 | `mnt` | 单品种因子来源：`mnt` 从底层拼；`local` 从本地读 |
| `--n-jobs` | int | 否 | `1` | 拼单品种因子时的并行度，**默认 1** |

## 4. 前置依赖

执行前需确保：
1. `config/groups.yaml` 中定义了该分组及品种列表
2. `data/raw/{symbol}/contract_info.csv` 已同步（来自 `01_sync_mkt_data.py`）
3. `data/raw/{symbol}/1min_active.feather` 已同步（用于跨品种因子计算）
4. `config/factor_config.yaml` 已配置且与 mnt 实际因子匹配

**可选前置**：若 `cross_factors.feather` 不存在，会在首次 `build_all_factor` 前自动计算（当前实现中 `build_all_factor` 只读取已存在的 cross_factors，不会自动计算。需手动先运行 `compute_cross_factors()` 或确保 03 步已执行）。

> **注意**：当前脚本不会自动调用 `compute_cross_factors()`。若跨品种因子未生成，`build_all_factor` 只会拼接单品种因子（`cross_fac = None`）。

## 5. 输出

```
data/groups/{group_name}/
├── cross_factors.feather          # 若已存在则直接读取
└── all_factor/
    ├── {symbol1}_all_fac.feather
    ├── {symbol2}_all_fac.feather
    └── ...
```

控制台会打印每个品种的保存路径和 shape：
```
All factor tables built for group [油脂油粕]:
  A: data/groups/油脂油粕/all_factor/A_all_fac.feather
  M: data/groups/油脂油粕/all_factor/M_all_fac.feather
  ...
```

## 6. 与 pipeline 其他步骤的关系

```
01_sync_mkt_data.py    --> 同步 raw 数据（contract_info, 1min_active）
02_calc_single_factors.py   （待实现：本地计算 m2m/t2m/t2t 因子）
03_calc_cross_factors.py    --> 计算 cross_factors.feather
04_make_factor_df.py   --> 拼 all_factor.feather（本脚本）
05_train.py            --> 读取 all_factor 训练模型
```

当前 `02_calc_single_factors.py` 尚未实现，因为 mnt 上已有现成的底层因子。若后续需要在本地重新计算因子（不依赖 mnt），则需补充此步骤。

## 7. 待调整项

### 7.1 自动计算跨品种因子

当前脚本不会自动调用 `compute_cross_factors()`。可考虑增加逻辑：
```python
if not cross_fac_path.exists():
    print("Cross factors not found, computing...")
    builder.compute_cross_factors()
```

### 7.2 分批保存与恢复

拼表过程耗时较长（4 品种约 7 分钟），若中途失败需重新来过。建议增加：
- 逐品种保存，失败时跳过已完成的品种
- 或增加 `--resume` 参数

### 7.3 日志输出到文件

当前仅打印到 stdout。建议增加 `--log-file` 参数，将进度和 shape 信息写入日志，方便审计。
