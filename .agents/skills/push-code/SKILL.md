---
name: push-code
description: 管理 /home 目录的 Git 代码仓库。当用户说"推送一下代码"、"push 代码"、"提交代码"、"git push"、"同步到 GitHub"、"备份代码"时触发推送；当用户说"版本管理"、"查看历史"、"回退代码"、"diff"、"对比版本"、"tag"、"打标签"时触发版本管理操作。仓库路径为 /home，remote 为 git@github.com:liaoyuyang/home.git。
---

# Push Code & Version Control

用户的工作目录 `/home` 已配置为 git 仓库：
- **Local path**: `/home`
- **Remote**: `git@github.com:liaoyuyang/home.git`
- **Branch**: `main`
- **Git config**: `user.name=liaoyuyang`, `user.email=948142104@qq.com`

`.gitignore` 已排除数据文件、日志、缓存和大目录，不会被推送。

---

## 第一部分：推送代码

当用户要求推送时，执行：

```bash
cd /home
git status --short
```

### 有未提交的改动

1. 自动生成提交信息（默认 `update: 同步代码修改`，用户明确给了就用用户的）
2. 执行：
   ```bash
   git add .
   git commit -m "<提交信息>"
   git push
   ```

### 无改动但 ahead of remote

直接 `git push`。

### 已同步

告诉用户"当前没有需要推送的改动"。

---

## 第二部分：版本管理

当用户询问版本管理相关操作时，在 `/home` 下执行对应命令。

### 查看历史

```bash
git log --oneline -20          # 最近20次提交
```

### 查看某次提交改了什么

```bash
git show <commit编号>
```

### 对比版本（diff）

```bash
git diff HEAD~1                # 对比现在和上一次提交
git diff HEAD~3                # 对比现在和往前第3次
git diff <commit编号>          # 对比现在和某次特定提交
```

### 回退/撤销

**仅查看旧版本（不修改）：**
```bash
git checkout <commit编号>
git checkout main              # 看完回到最新
```

**彻底回退到上一次提交（丢弃当前未提交的改动）：**
```bash
git reset --hard HEAD~1
```

**撤销某次旧提交（保留后续提交，生成反向提交）：**
```bash
git revert <commit编号>
git push
```

### 打标签（标记重要节点）

```bash
git tag -a <标签名> -m "<说明>"
git push origin <标签名>
```

例如训练好新模型后：
```bash
git tag -a v20260528 -m "模型更新：全品种重新训练"
git push origin v20260528
```

---

## 提交信息规范

- 默认：`update: 同步代码修改`
- 代码改动：`update: 摘要` / `fix: 修复 xxx` / `feat: 新增 xxx`
- 配置改动：`config: 调整 xxx`
- 文档改动：`docs: 更新 xxx`
