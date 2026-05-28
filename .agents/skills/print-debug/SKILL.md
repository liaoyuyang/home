---
name: print-debug
description: >
  当排查代码 bug、行为异常、值对不上时，强制要求通过 print / logging 输出实际变量值来定位问题，
  绝对禁止仅靠阅读代码在脑海中推断执行路径和变量状态。
  当用户说"print一下"、"跑一下看"、"不要猜"、"debug一下"时触发。
---

# Print Debug —— 变量优先，脑补禁止

## 核心原则（最高优先级）

> **看实际跑出来的值，而不是想象代码应该跑成什么样。**

- ❌ **禁止**：读了几行代码就开始推断"这里应该返回 False"、"这个变量应该是 0"
- ✅ **必须**：在可疑位置加 print，重新运行，把实际输出贴出来再分析

## 为什么必须这样

1. **Python 的隐式转换、索引类型、NaN 行为、时区处理** 经常和直觉相反
2. **DataFrame 的布尔索引、iloc/loc、条件组合** 只有看到实际值才能确认
3. **代码是动态的**：历史数据长度、缓存状态、配置覆盖都可能改变行为
4. **脑补的代价**：猜 10 分钟不如 print 1 分钟跑一遍

## 执行标准

### Step 1：找到可疑函数
- 不要从头到尾读整个文件
- 根据现象定位到最内层的判断/计算函数（如 `run_345`、`aggregate_ticks`）

### Step 2：在分支和计算前插入 print
至少输出以下信息：
- **输入参数**：传入的关键值（如 `now_pos`, `current_time`）
- **中间变量**：条件判断依赖的值（如 `condition`, `threshold`）
- **索引/时间**：`df.index[-3:]`, `last_time`（时间相关 bug 高频）
- **输出结果**：最终返回前的关键值

```python
# 示例模板
print(f"[DEBUG] now_pos={now_pos} | last_time={df.index[-1].time()} | last_ws={df.iloc[-1]['weighted_s']}")
print(f"[DEBUG] condition_last3={condition[-3:].tolist()}")
print(f"[DEBUG] long_open={weighted['long_open']} | short_open={weighted['short_open']} | now={weighted['now']}")
```

### Step 3：重新运行，贴输出给用户
- 只跑最小复现场景（如第一分钟、第一个品种）
- 把输出直接贴给用户，**先不做结论**
- 让用户确认现象，或一起根据实际值推断根因

### Step 4：确认后清理
- bug 定位后，**立即删除或注释掉 debug print**
- 如果用户希望保留，改成 `logger.debug` 级别

## 绝对禁止的行为

| 行为 | 后果 |
|------|------|
| 读了代码就说"这里应该是 xxx" | 浪费用户时间，可能完全猜错 |
| 在不确定的情况下直接改代码 | 引入新 bug，掩盖真问题 |
| 让用户"你先跑一下我看看"但自己不先加 print | 把排查成本转嫁给用户 |

## 记忆锚点

- 用户说 **"这个错误不应该这么难改"** → 说明我在脑补，立刻加 print
- 用户说 **"你 print 一下"** / **"跑一下看"** / **"不要猜"** → 触发本 skill
- 连续读了超过 20 行代码还没跑 → 停下来，加 print
