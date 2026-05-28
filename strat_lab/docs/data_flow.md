# 数据流总览

## 1. 整体 Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              mnt 数据（只读）                                │
│  /mnt/Data/writable/liaoyuyang/                                              │
│    ├── level2_all/{symbol}/          tick 级别原始数据                        │
│    ├── 1min/{symbol}/                1 分钟聚合数据                           │
│    ├── factor/{symbol}/m2m/          分钟-分钟因子                            │
│    ├── factor/{symbol}/t2m/          tick-分钟因子                            │
│    ├── factor/{symbol}/t2t/{inst}/   tick-tick 因子（按合约分目录）            │
│    ├── factor/{symbol}/all_fac/      ❌ 旧 all_factor（含全部跨品种因子）      │
│    └── future_info/{symbol}/         主力合约切换表                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼ (只读同步)
┌─────────────────────────────────────────────────────────────────────────────┐
│                              本地数据（读写）                                │
│  /home/strat_lab/data/                                                       │
│    ├── raw/{symbol}/                 同步后的原始数据                         │
│    │    ├── contract_info.csv        主力合约切换表                          │
│    │    └── 1min_active.feather      1 分钟主力合约数据                       │
│    ├── processed/{symbol}/           单品种处理结果（预留）                   │
│    └── groups/{group_name}/          分组隔离数据                             │
│         ├── cross_factors.feather    跨品种因子（分组独有）                   │
│         └── all_factor/              各品种完整因子表                         │
│              ├── {symbol}_all_fac.feather                                   │
│              └── ...                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. 分步数据流

### Step 01：同步市场数据
**脚本**：`pipeline/01_sync_mkt_data.py`

**输入**
- mnt: `future_info/{symbol}/main_instrument.csv`
- mnt: `1min/{symbol}/1min_active.feather`

**输出**
- local: `data/raw/{symbol}/contract_info.csv`
- local: `data/raw/{symbol}/1min_active.feather`

**说明**
- `contract_info.csv` 用于后续拼表时的合约过滤（只保留主力合约期间数据）
- `1min_active.feather` 用于计算跨品种因子

---

### Step 02：计算单品种因子（预留）
**脚本**：`pipeline/02_calc_single_factors.py`（待实现）

**当前状态**
- 跳过此步，直接从 mnt `factor/{symbol}/` 读取已计算好的 m2m/t2m/t2t 因子

**未来扩展**
- 若需在本地重新计算因子（不依赖 mnt），可在此步实现
- 输入：mnt `level2_all/` 或本地 `raw/`
- 输出：local `processed/{symbol}/m2m/`, `t2m/`, `t2t/`

---

### Step 03：计算跨品种因子
**脚本**：`pipeline/03_calc_cross_factors.py`

**输入**
- local: `data/raw/{symbol}/1min_active.feather`（分组内所有品种）

**输出**
- local: `data/groups/{group_name}/cross_factors.feather`

**计算内容**
- 两两品种的 close 收益率差、成交量比差、滚动相关性、量价相关性差、持仓量变动差
- 共 C(n,2)×7 列，n 为分组内品种数

---

### Step 04：拼因子表
**脚本**：`pipeline/04_make_factor_df.py`

**输入**
- mnt: `factor/{symbol}/m2m/`, `t2m/`, `t2t/`（底层单品种因子，只读）
- local: `data/raw/{symbol}/contract_info.csv`（主力合约切换表）
- local: `data/groups/{group_name}/cross_factors.feather`（跨品种因子）

**处理流程**
```
for symbol in group:
    # 1. 拼单品种因子（从底层 m2m/t2m/t2t）
    single_fac = FactorAssembler(symbol).assemble()
    #    -> 索引 ts, 列 ~1900 个

    # 2. 拼接跨品种因子（分组隔离）
    related_cols = [c for c in cross_fac.columns
                    if c 涉及该 symbol]
    all_fac = single_fac.join(cross_fac[related_cols], how='left')
    #    -> 索引 ts, 列 ~1900 + ~10 个

    # 3. 保存
    all_fac.to_feather(f"data/groups/{group}/all_factor/{symbol}_all_fac.feather")
```

**输出**
- local: `data/groups/{group_name}/all_factor/{symbol}_all_fac.feather`

**关键设计**
- 单品种因子从底层重新拼，不直接读 mnt 的 `all_factor.feather`
- 跨品种因子只拼接与本分组相关的列
- 不同分组的 `all_factor` 完全隔离

---

### Step 05：训练模型
**脚本**：`pipeline/05_train.py`

**输入**
- local: `data/groups/{group_name}/all_factor/{symbol}_all_fac.feather`
- local: `data/raw/{symbol}/contract_info.csv`（用于打标签）

**处理**
- 读取 all_factor，按主力合约切换表打标签（如 future return）
- 特征筛选（当前简化版取重要性 top 300）
- LightGBM 训练

**输出**
- local: `models/{group_name}/{symbol}_model.pkl`
- local: `models/{group_name}/{symbol}_feature_importance.csv`

## 3. 时间列约定

| 数据级别 | 时间列名 | 含义 | 示例 |
|---------|---------|------|------|
| tick 级 | `datetime` | 带微秒的真实时间 | `2024-01-02 09:01:00.123456` |
| 分钟级 | `ts` | 分钟 bar 右界 | `2024-01-02 09:01:00` |

**统一规则**
- 分钟级 DataFrame 的索引/时间列统一命名为 `ts`
- `datetime` 仅用于 tick 级数据
- 读入 mnt 数据时若列名为 `datetime`，需 `rename(columns={"datetime": "ts"})`

## 4. 分组隔离原则

```
分组 A（油脂油粕）: A, M, Y, P
    all_factor/A_all_fac.feather 只含 A 的单品种因子 + A_M/A_Y/A_P 跨品种因子
    all_factor/M_all_fac.feather 只含 M 的单品种因子 + M_A/M_Y/M_P 跨品种因子
    ...

分组 B（谷物）: C, CS, WH, RI
    all_factor/C_all_fac.feather 只含 C 的单品种因子 + C_CS/C_WH/C_RI 跨品种因子
    ...
```

**为什么隔离？**
- mnt 的 `all_factor.feather` 包含全部 8 品种的跨品种因子（如 A_B, A_C, A_LH）
- 若 A 在"油脂油粕"分组里直接读 mnt all_factor，会拿到 A_B、A_C 等不属于该分组的因子
- 这些"外来"因子在训练时可能引入泄漏或噪音
- 隔离后，每个分组的模型只学习本分组内的品种关系
