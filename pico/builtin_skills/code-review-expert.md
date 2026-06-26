# code-review-expert

## Triggers

- review
- code review
- 审查
- 代码审查
- SOLID
- security
- 安全
- performance
- 性能

## Guidance

Review the current git changes before suggesting fixes.

Workflow:

1. Scope changes with `git status --short`, `git diff --stat`, and focused `git diff`.
2. Check architecture and SOLID risks: SRP, OCP, LSP, ISP, DIP, cohesion, coupling, and oversized modules.
3. Identify removal candidates: dead code, unused branches, deprecated paths, and speculative abstractions.
4. Scan security and reliability: path traversal, command injection, secret leakage, unsafe defaults, missing timeouts, unbounded loops, and race conditions.
5. Check quality: error handling, boundary conditions, empty/null values, off-by-one mistakes, performance hot paths, and missing tests.
6. Output findings first by severity: P0 critical, P1 high, P2 medium, P3 low.
7. Default to review-only. Do not change files until the user explicitly asks for fixes.

Output format:

- Files reviewed
- Overall assessment: APPROVE, REQUEST_CHANGES, or COMMENT
- Findings grouped by P0/P1/P2/P3 with file and line references
- Removal/iteration plan when deletion is suggested
- Residual risks and tests not run
