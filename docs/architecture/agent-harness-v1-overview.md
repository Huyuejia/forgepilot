# Agent Harness v1 Overview

This document summarizes the forgepilot agent harness architecture.

## Scope

forgepilot is a local coding agent harness. The main workflow is:

1. User request enters the CLI.
2. The runtime builds agent context.
3. Tools read, search, edit, or execute commands in the workspace.
4. Memory and context management compress useful information.
5. Run traces and reports are saved for later review.

## Main Modules

- `forgepilot/cli.py`: command-line entry and agent construction
- `forgepilot/runtime.py`: main agent loop and tool execution
- `forgepilot/context_manager.py`: prompt construction and context budget control
- `forgepilot/memory.py`: working, episodic, file, and durable memory
- `forgepilot/run_store.py`: trace, report, and checkpoint persistence

## Task state

The task state records the current task lifecycle, progress, completion status, and intermediate decisions used by the agent harness during a run.
