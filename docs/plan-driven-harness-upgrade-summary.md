# forgepilot 计划驱动工程代理增强包：项目总结

## 一句话概括

本次二次开发把 forgepilot 从一个轻量本地 coding agent harness，升级成了一个具备“显式计划、模型可更新计划、运行报告可观测、benchmark 可评测、技能按需加载”的工程代理实验台。

它不是简单加几个命令，而是在 forgepilot 原有 runtime、tools、memory、context_manager、run_store、evaluator 分层上，补上了 Claude Code 类 agent harness 中很关键的两类机制：

- 计划层：回答“接下来做什么、做到哪一步了”。
- 技能层：回答“遇到特定任务时，该加载哪类专业指导”。

## 已完成的功能

### 1. 结构化计划状态

新增 `forgepilot/planner.py`，定义计划项的四种状态：

- `pending`
- `in_progress`
- `completed`
- `blocked`

计划状态保存在 session 里，支持规范化、渲染、统计，并保证同一时间最多只有一个 `in_progress` 项。

对应测试：

- `tests/test_planner.py`

### 2. Runtime / Prompt / Report 接入计划

在 `forgepilot/runtime.py` 中增加：

- `plan_text()`：把计划渲染成人类可读文本。
- `set_plan_items()`：设置当前计划。
- `update_plan_item_status()`：更新单个计划项状态。
- `build_report()`：把计划 summary 和 plan items 写入 run report。

在 `forgepilot/context_manager.py` 中把计划文本加入 prompt，使模型在每轮推理时能看到当前计划。

对应测试：

- `tests/test_planner_runtime.py`

### 3. 模型可调用的 update_plan 工具

在 `forgepilot/tools.py` 中新增 safe tool：

```text
update_plan(items: list[dict|str])
```

模型可以通过工具调用更新当前任务计划，例如：

```xml
<tool>{"name":"update_plan","args":{"items":[{"text":"Read code","status":"completed"},{"text":"Run tests","status":"in_progress"}]}}</tool>
```

这个机制类似 Claude Code 的 TodoWrite：模型不只是口头说“我会做什么”，而是把计划写进可持久化、可观测的 agent state。

### 4. REPL /plan 命令

在 `forgepilot/cli.py` 中新增：

```text
/plan
```

用于在 forgepilot REPL 中查看当前任务计划。

### 5. Benchmark 覆盖计划层

在 `benchmarks/coding_tasks.json` 中新增 benchmark 任务：

```text
plan_update_visible
```

在 `forgepilot/evaluator.py` 中为该任务增加 scripted model output：先调用 `update_plan`，再返回 final。

在 `tests/test_evaluator.py` 中验证：

- benchmark 任务数从 12 增加到 13。
- 新增 `planning` 类别。
- run report 中记录了计划状态。

### 6. Skills 按需加载

新增轻量 skill 系统：

- `forgepilot/skills.py`
- `forgepilot/builtin_skills/code-review-expert.md`
- `tests/test_skills.py`

当用户请求中出现 review、安全、审查、性能、SOLID 等关键词时，forgepilot 会把 `code-review-expert` 的指导内容渲染进 prompt。

这一步的意义是：forgepilot 不需要把所有专业规则常驻 prompt，而是按任务需要加载相关技能卡片，降低 prompt 噪声，也更接近 Claude Code 的 skill 机制。

## 提交记录

当前核心提交包括：

```text
c1c6e0e feat: add on-demand skill loading
c772f61 test: cover visible plan state in benchmark
dc49d3a feat: add update plan tool and command
276ce37 feat: add plan state to runtime
```

## 验证结果

阶段一主回归：

```text
88 passed in 21.48s
```

阶段二 A 加主回归：

```text
98 passed in 27.08s
```

说明新增的计划层、工具、benchmark 和 skill 按需加载没有破坏原有核心测试。

## 当前未提交文件说明

当前仓库还可能有这些未提交项：

```text
 M forgepilot/models.py
?? .claude/
?? experiments/smoke_pico_minimal.log
```

它们不属于本次 plan / skills 改造主线：

- `forgepilot/models.py` 是此前已有改动，需要单独判断是否保留。
- `.claude/` 是 Claude Code 的本地配置，不建议提交。
- `experiments/smoke_pico_minimal.log` 是实验日志，不建议和功能代码一起提交。

## 可以怎么演示

### 演示 1：计划状态渲染

运行 forgepilot 后让模型做多步骤任务，观察 prompt/report 中出现计划状态。

### 演示 2：update_plan 工具

让模型输出：

```xml
<tool>{"name":"update_plan","args":{"items":[{"text":"Inspect code","status":"completed"},{"text":"Run tests","status":"in_progress"}]}}</tool>
```

然后用 `/plan` 查看当前计划。

### 演示 3：report.json 可观测性

一次 run 结束后查看 `.forgepilot/runs/.../report.json`，确认其中包含：

```json
{
  "plan": {
    "total": 2,
    "completed": 1,
    "in_progress": 1
  },
  "plan_items": []
}
```

### 演示 4：benchmark planning row

运行 evaluator 测试，说明新增 planning benchmark 能自动验证计划状态是否进入 report。

### 演示 5：skills 按需加载

输入类似：

```text
please review this diff for security issues
```

或：

```text
帮我做一次代码审查，重点看安全和边界条件
```

forgepilot prompt 中会出现 `Relevant skills: code-review-expert`。

## 简历写法

可以写成：

> 二次开发 forgepilot 本地 coding agent harness，参考 Claude Code 的 Todo/Skill 机制，新增结构化计划状态、模型可调用 `update_plan` 工具、REPL `/plan` 命令、run report 计划可观测性、benchmark planning 覆盖，并实现关键词触发的 `code-review-expert` 按需技能加载；补充 pytest 回归测试，核心测试 98 passed。

更工程化一点：

> Designed and implemented a plan-driven extension for a local coding-agent harness, adding persistent task-plan state, model-callable planning tools, prompt/report integration, benchmark coverage, and on-demand skill loading with regression tests.
