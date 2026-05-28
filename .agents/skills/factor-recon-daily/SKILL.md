---
name: factor-recon-daily
description: >
  每日因子核对工作结束后，写一份 Agent 状态简报 markdown，告诉下一个 agent 今天做了什么、接下来该干什么。
  当用户说"总结一下"、"写日报"、"记录到 YYYYMMDD.md"、"给下一个 agent 交代一下"时触发。
  输出到 strategy_PAMY_dce_test/YYYYMMDD.md。
  不写具体因子数值，只写代码变更摘要、已知陷阱、待办清单、路径速查。
---

# Factor Reconciliation Daily Summary

## 目标受众

下一个打开这个项目的 agent。读完就知道该干什么，不用从头问用户。

## 触发条件

- "总结一下今天的工作"
- "写个日报"
- "记录到 20250509.md"
- "给下一个 agent 交代一下进度"
- "今天核对完了，记一下"

## 工作流程

### Step 0: 先问原则（最高优先级）

写日报前先自检：今天有没有在用户未明确同意的情况下做了推断性操作？
- 如果有，在日报里**诚实记录**并标注"此处应先问用户确认"
- 这是为了防止下一个 agent 重复踩坑

### Step 1: 收集当日状态

1. 检查 git diff / 询问用户今日改了哪些文件
2. 读取关键文件：
   - `calc_recent_data.py`
   - `data_function.py`
   - `check_history_factor copy.ipynb`
3. 提取：
   - 当前验证到第几个 symbol（8 个：A/B/C/CS/M/Y/P/LH）
   - Market diff 是否清零
   - Factor 还有多少个 fail
   - 有无未重跑的 calc_recent_data

### Step 2: 按模板写 Markdown

输出路径：`/home/agent_daily/kimi/YYYYMMDD.md`

**模板：**

```markdown
# Agent 状态简报 — YYYY-MM-DD

> 这是给下一个 agent 看的。读完你就知道该干什么，不用重头问用户。

## 当前进度
- 验证范围：8 个 DCE 农产品期货
- 已验证：X（contract），Y 日窗口
- Market：对齐 / 未对齐
- Factor：大部分对齐 / 还有 N 个因子有 diff

## 今天做了什么（代码变更摘要）
### 修改 N：文件名
- **问题**：一句话现象
- **根因**：技术原因
- **修复**：改了什么、怎么改的
- **文件位置**：绝对路径

## 关键待办（按优先级）
- [ ] ...

## 如果你要继续做，建议顺序
1. ...
2. ...

## 已知陷阱（别踩）
- ...

## 路径速查
| 用途 | 路径 |
|------|------|
| ... | ... |

## 参数
- TOL_ABS = 0.01, TOL_REL = 0.05, TAIL_N = 8
```

### Step 3: 约束

- **不写具体因子名称和数值**：只写问题类别
- **必须写"建议顺序"**：让下一个 agent 知道第一步该干嘛
- **必须写"已知陷阱"**：避免重复踩坑
- **必须写待办优先级**：用 `[ ]` 清单，高优先级放前面
- **如果文件已存在**：追加当日章节，或询问用户是否覆盖

## 参考信息

### 数据流水线
- **Research**: decode_csv_dce → prepare_mkt_data.py → level2_all/ → calc_factors_dce农.py → all_factor.feather
- **Realtime**: decode_csv_dce → calc_recent_data.py → files/{contract}_tick.csv + _min.csv → runtime factor

### 核对分层
| 层级 | 内容 | 通过标准 |
|------|------|----------|
| L1 | tick drill-down | boundary tick volume/turnover/price 一致 |
| L2 | 1min K 线对齐 | open/high/low/close/volume/turnover/OI max_diff = 0 |
| L3 | factor 逐 bar 对齐 | fail ratio = 0（或分布指标在阈值内） |
| L4 | prediction 对齐 | weighted / weighted_s 一致 |

### 关键路径
- notebook: `/home/strategy_online/strategy_PAMY_dce_test/check_history_factor copy.ipynb`
- realtime preprocessor: `/home/strategy_online/strategy_PAMY_dce_test/calc_recent_data.py`
- factor generator: `/home/strategy_online/strategy_PAMY_dce_test/data_function.py`
- research factor: `/mnt/Data/writable/liaoyuyang/factor/{SYMBOL}/all_fac/all_factor.feather`
- research 1min: `/mnt/Data/writable/liaoyuyang/data/1min/{SYMBOL}/{contract}.feather`
- research tick: `/mnt/Data/writable/liaoyuyang/data/level2_all/{SYMBOL}/{contract}.feather`
- realtime files: `/home/strategy_online/strategy_PAMY_dce_test/files/{contract}_tick.csv`
- summary CSV: `/home/strategy_online/strategy_PAMY_dce_test/factor_diff_summary_{SYMBOL}.csv`

### 常用参数
- TOL_ABS = 0.01
- TOL_REL = 0.05
- TAIL_N = 8
- Session 过滤：09:10-10:05, 10:40-11:20, 13:40-14:50, 21:10-22:50

### Trade-date 规则
夜盘 21:00-23:00 归属 **下一个** trade_date。

### 高频陷阱
1. **Dedup 顺序**：volume/turnover 必须先 diff 再 `drop_duplicates(keep='last')`
2. **Boundary tick**：tick-level factor（如 LVC）需要 08:59、09:00、13:30 等边界 tick，不能提前过滤 `is_trading=False`
3. **Zigzag 类因子**：`rolling(240*5)` 需要约 5 日历史，`_truncate_by_trade_date` 不能截太短
4. **Session 过滤**：统计 fail ratio 前先 `filter_active_trading` 剔除前后 10min
