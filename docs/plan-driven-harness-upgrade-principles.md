# forgepilot 二次开发原理说明：你到底实现了什么

## 先回答最重要的问题

你这次不是在“让 AI 帮你随便改代码”。你做的是一个 agent harness 的二次开发。

forgepilot 原来已经有这些能力：

- 可以和模型对话。
- 可以让模型调用工具读写文件、运行命令。
- 有 memory、context manager、run report、checkpoint、benchmark。

但它少了两个 Claude Code 类工具里很关键的机制：

1. 显式计划：模型现在到底在做第几步？哪些完成了？哪里阻塞了？
2. 按需技能：遇到 review、安全、性能等特定任务时，模型能不能加载一张专业提示词卡片？

你这次的二开就是补这两个机制。

## 为什么一会儿用 Codex，一会儿用 Claude Code，一会儿用 WSL 终端

这三个角色不一样。

### Codex：主控和验收

Codex 在这次流程里主要做：

- 读任务书。
- 判断任务顺序。
- 给 Claude Code 拆小任务。
- 检查 diff 有没有跑偏。
- 解释失败原因。
- 决定什么时候可以提交。

也就是说，Codex 更像技术负责人或 reviewer，不是每一步都亲自敲代码。

### Claude Code：执行工

Claude Code 被用来做具体小任务，例如：

- 写 Task 2 的 runtime 集成测试。
- 修改 `runtime.py` 和 `context_manager.py`。
- 写 Task 3 的 `update_plan` 工具。
- 写阶段二 A 的 `skills.py`。

为什么要给它很窄的指令？

因为 agent 很容易自由发挥。你给它的任务越宽，它越可能顺手改 benchmark、改无关文件、跑 Windows 环境下不稳定的命令。所以我们一直限制它：

```text
只允许修改这些文件
不要做下一阶段
不要提交 commit
不要运行 git
```

这不是麻烦，而是工程控制。

### WSL 终端：真实测试和 Git 提交

你的 forgepilot 项目实际测试环境在 WSL 里更稳定：

```text
/mnt/d/AIProjects/forgepilot-main/forgepilot-main
```

而 Claude Code 启动后经常落在 Windows PowerShell 环境：

```text
D:\AIProjects\forgepilot-main\forgepilot-main
```

这会导致：

- `git` 找不到。
- `python3` 找不到。
- Windows Python 版本和 WSL `.venv` 不一致。
- 某些 symlink、shell、临时目录测试失败。

所以最终我们采用了分工：

- Claude Code 负责改代码。
- WSL 终端负责跑 pytest 和 git commit。
- Codex 负责判断测试失败是不是代码问题，还是环境问题。

这就是为什么你一会儿复制指令给 CC，一会儿又回 WSL 终端执行命令。

## Task 1：planner.py 是干什么的

`forgepilot/planner.py` 是计划状态核心模块。

它把“我要做几步”变成结构化数据，而不是普通文本。

一个计划大概长这样：

```json
{
  "items": [
    {"id": "step-1", "text": "Read code", "status": "completed"},
    {"id": "step-2", "text": "Run tests", "status": "in_progress"}
  ],
  "active_id": "step-2"
}
```

它做了几件事：

- 清理用户或模型传来的计划文本。
- 生成稳定 id，比如 `step-1`、`step-2`。
- 限制最多 12 个计划项。
- 限制每个计划项文本最多 180 字符。
- 保证最多只有一个 `in_progress`。
- 可以把计划渲染成：

```text
Plan:
- [completed] step-1: Read code
- [in_progress] step-2: Run tests
```

为什么这重要？

因为 agent 的计划不能只存在于模型嘴里。它必须进入程序状态，才能被保存、恢复、展示、评测。

## Task 2：为什么要接入 runtime / prompt / report

`runtime.py` 是 forgepilot 的主循环核心。

模型每次工作时，大概经过这个流程：

```text
用户请求
-> ContextManager 拼 prompt
-> 模型返回 tool 或 final
-> forgepilot 执行工具
-> 记录 trace/report/session
```

Task 2 把 plan 接进这个流程：

### 1. 进入 session

在 session 中新增：

```python
self.session["plan"]
```

这样计划可以随会话保存和恢复。

### 2. 进入 prompt

`context_manager.py` 会把计划文本放进 prompt。

意义是：模型每一轮都能看到当前计划，不会忘记自己做到哪一步。

### 3. 进入 report.json

一次 run 结束后，report 中会出现：

```json
"plan": {
  "total": 2,
  "pending": 0,
  "in_progress": 1,
  "completed": 1,
  "blocked": 0
}
```

意义是：计划状态可观测、可审计、可 benchmark。

## Task 3：update_plan 工具是什么

`update_plan` 是模型可以调用的工具。

没有它之前，模型只能在自然语言里说：

```text
我先读代码，再跑测试。
```

有了它以后，模型可以写入结构化计划：

```xml
<tool>{"name":"update_plan","args":{"items":[{"text":"Read code","status":"completed"},{"text":"Run tests","status":"in_progress"}]}}</tool>
```

forgepilot 收到这个 tool call 后，会执行：

```python
agent.set_plan_items(items)
```

然后计划就被写进 session。

为什么这像 Claude Code？

Claude Code 也有类似 TodoWrite 的机制。核心思想是：

> 让 agent 把任务进度写成可检查的结构化状态，而不是只靠聊天记录。

## /plan 命令是什么

`/plan` 是 forgepilot REPL 里的查看命令。

你在交互式 forgepilot 里输入：

```text
/plan
```

就能看到当前计划。

它的价值是演示和调试：你可以直观看到 agent 现在认为自己做到哪一步。

## Task 4：为什么要改 benchmark

如果只写功能，不写 benchmark，简历上会弱很多。

Task 4 给固定 benchmark 新增一条 planning 任务：

```text
plan_update_visible
```

这条任务要求模型：

1. 创建两步计划。
2. 把 reading 标记为 completed。
3. 把 testing 标记为 in_progress。
4. 结束任务。

然后 verifier 检查 `.forgepilot/runs/.../report.json`：

```python
assert report['plan']['total'] == 2
assert report['plan']['completed'] == 1
assert report['plan']['in_progress'] == 1
```

这说明你的计划层不是“看起来能用”，而是进入了 forgepilot 的评测闭环。

## 阶段二 A：skills 按需加载是什么

你新增了：

```text
forgepilot/skills.py
forgepilot/builtin_skills/code-review-expert.md
tests/test_skills.py
```

这个机制很轻量：

1. 在 `builtin_skills/*.md` 里存技能卡片。
2. 每张卡片有 Triggers，比如：
   - review
   - code review
   - 审查
   - 安全
   - performance
3. 用户请求来了以后，`select_relevant_skills()` 看哪些 trigger 出现在请求里。
4. 命中的 skill 被 `render_relevant_skills()` 渲染进 prompt。

例如用户说：

```text
帮我做一次代码审查，重点看安全和边界条件
```

forgepilot prompt 中会出现：

```text
Relevant skills:
## code-review-expert
...
Default to review-only...
```

这不是让 Claude Code review 你的代码，而是让 forgepilot 自己学会“按任务加载专业提示词”。

## 为什么要先写测试

你每一步都基本遵循了：

```text
写失败测试
-> 看到失败
-> 实现功能
-> 看到测试通过
```

这样做的好处是：

- 确认测试真的能捕捉缺失功能。
- 避免先写代码再补测试时自欺欺人。
- 每个功能都有可回归验证。

比如 Task 3 一开始失败是因为 `update_plan` 不存在；后来又失败是因为 dict item 被当成字符串。测试把真实 bug 暴露出来了。

## 为什么有些失败不是代码问题

你遇到过几类环境问题：

### Windows PowerShell 里没有 git

所以 CC 里运行：

```text
git status --short
```

会失败。

### Windows 里没有 python3

benchmark verifier 用的是：

```text
python3 -c ...
```

在 WSL 下可以，在 Windows PowerShell 下可能失败。

### Windows/WSL 临时目录和 symlink 差异

有些安全测试涉及 symlink、shell、临时目录，Windows 环境下容易出现无关失败。

所以最终判断标准是：

> 用 WSL 项目的 `.venv/bin/python` 跑测试。

也就是：

```bash
.venv/bin/python -m pytest ...
```

## 现在是否需要上传 GitHub

如果这是简历项目，建议上传 GitHub，但要先整理干净。

计算机系项目不一定必须上传 GitHub，但简历项目最好有 GitHub 链接。原因很现实：

- 面试官可以看到 commit 历史。
- 能看到测试、文档、代码组织。
- 你可以在简历上放项目链接。
- 比只写一句“做过 agent 二开”可信很多。

## 上传前必须检查什么

上传前不要提交这些：

```text
.claude/
experiments/smoke_pico_minimal.log
```

还要确认 `forgepilot/models.py` 是不是你想保留的改动。如果不是本项目主线，先不要提交。

建议加到 `.gitignore`：

```gitignore
.claude/
experiments/*.log
```

如果 `.env`、API key、token 出现过，也必须确认没有被提交。

## GitHub 上传流程

### 方案 A：新建你自己的仓库

在 GitHub 网页上新建一个 repo，例如：

```text
forgepilot-plan-driven-agent
```

不要勾选初始化 README，因为本地已有仓库。

然后在 WSL 里执行：

```bash
cd /mnt/d/AIProjects/forgepilot-main/forgepilot-main
git remote add origin https://github.com/你的用户名/forgepilot-plan-driven-agent.git
git push -u origin codex/forgepilot-plan-driven-harness
```

如果你想把当前分支作为 GitHub 默认主分支，可以之后在 GitHub 上设置 default branch，或本地改名：

```bash
git branch -M main
git push -u origin main
```

但如果原项目不是你自己的，建议保留分支名，并写清楚这是二次开发分支。

### 方案 B：fork 原仓库后推分支

如果 forgepilot 原项目在 GitHub 上，你可以先 fork，然后：

```bash
git remote add origin https://github.com/你的用户名/forgepilot-main.git
git push -u origin codex/forgepilot-plan-driven-harness
```

这个方式更适合说明“我基于开源项目做二次开发”。

## 上传前推荐补一个文档

建议在仓库里放：

```text
docs/plan-driven-harness-upgrade-summary.md
docs/plan-driven-harness-upgrade-principles.md
```

第一份给面试官快速看成果。

第二份给你自己复盘原理，防止面试时说不清。

## 面试时怎么讲

你可以按这个顺序讲：

1. forgepilot 是一个本地 coding agent harness。
2. 原项目已有工具调用、记忆、上下文管理、checkpoint、report、benchmark。
3. 我参考 Claude Code 的 Todo/Skill 思路，补了计划层和按需技能层。
4. 计划层包括 planner 状态模型、update_plan 工具、prompt/report 接入、/plan 命令。
5. 技能层包括 markdown skill 卡片、trigger 召回、prompt 注入。
6. 我用 pytest 和 benchmark 验证功能，主回归 98 passed。

## 简历 bullet

可以写：

> 基于 Python 二次开发本地 coding agent harness forgepilot，参考 Claude Code 的 Todo/Skill 机制，实现结构化任务计划、模型可调用 `update_plan` 工具、REPL `/plan` 命令、run report 可观测性、benchmark planning 覆盖和关键词触发的按需 skill 加载；补充 pytest 回归测试，核心测试 98 passed。

或者更短：

> Extended a local coding-agent harness with persistent task planning, model-callable plan updates, prompt/report observability, benchmark validation, and on-demand skill loading inspired by Claude Code.
