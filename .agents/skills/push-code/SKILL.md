---
name: push-code
description: 自动将本地 git 仓库代码推送到 GitHub。当用户说"推送一下代码"、"push 代码"、"提交代码"、"git push"、"同步到 GitHub"、"备份代码"时触发。支持 strategy_PAMY_dev（dev 环境）和 online/dce农（实盘环境）两个仓库。
---

# Push Code

用户的两个核心仓库：

| 目录 | 用途 | GitHub remote |
|---|---|---|
| `/home/strategy_PAMY_dev/` | dev/test 环境、因子核对开发 | `git@github.com:liaoyuyang/home.git` |
| `/home/online/dce农/` | 实盘运行环境 | 需用户创建第二个仓库（如 `dce-strategy-online`） |

> 注意：`user.email` 已设为 `948142104@qq.com`，`user.name` 为 `liaoyuyang`。SSH key 已配置。

## 工作流程

### 1. 确定目标仓库

根据用户话语判断：
- 提到 "online"、"prod"、"实盘"、"dce农" → `/home/online/dce农/`
- 提到 "dev"、"test"、"strategy_PAMY_dev"、"开发环境" → `/home/strategy_PAMY_dev/`
- **未指定 → 先执行 `git status` 查看两个仓库是否有改动，把有改动的列出来让用户确认，或两个都推**

### 2. 检查并推送

进入目录后执行：

```bash
cd <仓库目录>
git status --short
```

- **有未提交的改动**（`git status --short` 有输出）：
  1. 自动生成提交信息。优先从用户最近对话中提取改动摘要；如果提取不到，用简短的默认信息如 `update: 同步代码修改`
  2. 执行：
     ```bash
     git add .
     git commit -m "<提交信息>"
     git push
     ```

- **无未提交改动，但 ahead of remote**：
  - 直接执行 `git push`

- **无任何改动且已同步**：
  - 告诉用户"当前没有需要推送的改动"

### 3. online 仓库未配置 remote 的处理

如果 `/home/online/dce农/` 还没有 remote（首次推送），提示用户：
> "online 代码还没有 GitHub 仓库。请去 github.com 新建一个仓库（建议取名 `dce-strategy-online`），不要勾选 README。建好后告诉我，我帮你推送。"

用户告知仓库名后，执行：
```bash
cd /home/online/dce农
git remote add origin git@github.com:liaoyuyang/<仓库名>.git
git branch -M main
git push -u origin main
```

## 提交信息规范

- 修复 bug：`fix: 修复 xxx`
- 新增功能/模块：`feat: 新增 xxx`
- 配置调整：`config: 调整 xxx`
- 文档/日报更新：`docs: 更新 xxx`
- 不清楚具体改动了什么：`update: 同步代码修改`
