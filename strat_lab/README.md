# Strat Lab — 独立量化研究框架

## 项目定位

与 `future_commodity` / `strategy_PAMY_dev` 解耦的独立研究项目，支持：
- **多套品种流程**：同一品种可属于不同分组，单品种因子复用，跨品种因子按分组隔离
- **数据分层**：核心大数据仍在 `/mnt`，主力合约拼接后的小数据、因子表、模型、回测结果放在项目本地
- **可迁移**：换机器时只需改 `config/paths.yaml` 中的 `mnt_root`，重新同步即可

## 快速开始

```bash
cd /home/strat_lab
# 1. 校验环境并创建必要目录
bash scripts/setup_env.sh

# 2. 同步主力合约数据到本地
python pipeline/01_sync_mkt_data.py --symbols A M Y P

# 3. 同步单品种因子到本地（或跳过，直接在 mnt 计算）
python pipeline/02_sync_single_factors.py --symbols A M Y P

# 4. 计算跨品种因子并按分组拼表
python pipeline/03_calc_cross_factors.py --group 油脂油粕
python pipeline/04_make_factor_df.py --group 油脂油粕

# 5. 训练 + 回测（参见 notebooks/template_train_backtest.ipynb）
```

## 目录说明

| 目录 | 说明 |
|------|------|
| `config/` | 所有配置文件（路径、分组、品种参数、流程参数） |
| `data/raw/` | 从 mnt 同步的主力合约 1min 数据 + 合约切换表 |
| `data/processed/` | 单品种因子（m2m/t2m/t2t），可本地重算或从 mnt 同步 |
| `data/groups/{group}/` | 按分组隔离的产物：all_factor、models、predictions、backtest |
| `pipeline/` | 数据同步、因子计算、拼表、训练、回测的流程脚本 |
| `src/` | 公共工具库（路径解析、配置管理、数据加载、因子工具、分组构建） |
| `research/` | 新增研究功能框架（因子分析、降频研究、跨分组对比） |
| `notebooks/` | 研究 Notebook 模板 |
| `tests/` | 数据完整性校验、因子对齐测试 |
| `scripts/` | 环境初始化等辅助脚本 |

## 核心设计

### 路径管理

所有路径通过 `config/paths.yaml` 集中管理，`src/path_resolver.py` 提供统一解析接口：

```python
from src.path_resolver import PathResolver
pr = PathResolver()

# mnt 侧路径
pr.resolve('mnt', 'factor', 'A', 'all_fac')
# → /mnt/Data/writable/liaoyuyang/factor/A/all_fac/all_factor.feather

# 本地侧路径
pr.resolve('local', 'group', '油脂油粕', 'models')
# → ./data/groups/油脂油粕/models
```

支持环境变量覆盖：`export STRAT_LAB_MNT_ROOT=/new/path`

### 分组体系

`config/groups.yaml` 定义分组，核心逻辑：
- 单品种因子（m2m/t2m/t2t）在品种级别计算，所有分组**复用**
- 跨品种因子（cross index）在分组级别计算，不同分组**隔离**
- `04_make_factor_df.py` 按分组读取单品种因子 + 该分组的跨品种因子 → 拼接为 `all_factor.feather`

## 复用现有逻辑

单品种因子计算逻辑复杂且已验证，通过 `sys.path.append('/home/future_commodity')` 直接复用：
- `function_future.DataLoader`
- `function_future.factor_generator`
- `function_future.min_factors`
- `function_future.tick_factors`
- `function_future.tick2min_factors`

训练回测逻辑（`pre_train.py`、`train_model.py`、`backtest_v3.py`）已复制到 `src/` 并适配新路径体系。
