---
name: ipynb-format-guard
description: >
  修改 .ipynb notebook 文件时的格式保护规范。
  防止 cell source 被意外压成单行、丢失换行结构，导致代码不可读甚至运行失败。
  当需要修改 notebook cell 内容时触发。
---

# Jupyter Notebook 格式保护规范

## 问题现象

用 Python 脚本修改 `.ipynb` 文件后，notebook cell 里的代码被压缩成一行，所有注释、缩进、空行全部丢失。例如：

```
# 修改前（正常）
# ============================================================
# Cell 4: 因子筛选
# ============================================================
import pandas as pd

# 修改后（损坏）
# ============================================================# Cell 4: 因子筛选# ============================================================import pandas as pd
```

## 根因

`.ipynb` 的 `cells[*].source` 字段是 **list of strings**，格式如下：

```json
{
  "source": [
    "# 标题\\n",
    "import pandas as pd\\n",
    "x = 1\\n"
  ]
}
```

- 每行是一个独立的字符串元素
- 除最后一行外，每行末尾带有 `\n`
- **绝对不能**用 `''.join(source)` 合并成单个字符串做替换，再用 `split('\n')` 拆回 list——这会导致格式丢失、空行消失、缩进混乱

## 行为准则

### ❌ 禁止

```python
# 禁止：合并 → 替换 → 拆分
src = ''.join(cell['source'])
src = src.replace('old', 'new')
cell['source'] = src.split('\n')
```

### ✅ 推荐

```python
# 推荐 1：直接构造新的 list of strings
cell['source'] = [
    '# 标题\n',
    'import pandas as pd\n',
    'x = 1\n',
]

# 推荐 2：只修改需要改的那几行，不动其他行
for i, line in enumerate(cell['source']):
    if 'old_param' in line:
        cell['source'][i] = line.replace('old_param', 'new_param')
```

## 修改后必须验证

修改 `.ipynb` 后，立即运行以下检查：

```python
import json
with open('file.ipynb') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = cell['source']
        assert isinstance(src, list), f"Cell {i} source 不是 list"
        assert len(src) > 0, f"Cell {i} source 为空"
        assert len(src[0]) < 500, f"Cell {i} 第 0 行过长({len(src[0])})，可能被压成单行"
```

## 历史事故

- **2026-05-27**：修改 `A_2025-07-01_v0_monthly_stability.ipynb` Cell 4 时，使用 `''.join()` + `split('\n')` 方式，导致整个 cell（1692 字符）被压成单行，筛选流程代码不可读，用户多次报错后修复。
