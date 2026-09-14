# Review Pack

This folder contains reviewer-facing materials for forgepilot, including benchmark notes, reports, screenshots, and reproducible review instructions.

## Contents

- Baseline experiment logs
- Benchmark results
- Demo materials
- Review notes

## Project pitch

forgepilot is a lightweight local coding agent harness for experimenting with tool use, workspace operations, memory, context management, checkpoints, traces, and evaluation.

This review pack summarizes the project motivation, architecture, benchmark setup, and reproducible review steps.

## Architecture map

forgepilot's high-level architecture is:

User request
-> CLI
-> Runtime loop
-> Context manager
-> Tools
-> Memory
-> Run store
-> Trace and report

Main module responsibilities:

- `forgepilot/cli.py`: command-line entry and agent construction.
- `forgepilot/runtime.py`: main agent loop and tool execution.
- `forgepilot/context_manager.py`: prompt construction and context budget control.
- `forgepilot/memory.py`: working memory, episodic notes, file summaries, and durable memory.
- `forgepilot/tools.py`: file, search, edit, and shell tools.
- `forgepilot/run_store.py`: traces, reports, sessions, and checkpoints.

## Benchmark evidence

Benchmark evidence should include pytest results, baseline experiment logs, prompt length records, tool call counts, repeated file read counts, and task success notes.

## Sample run artifact list

Sample run artifacts include:

- `.forgepilot/runs/` trace files
- `.forgepilot/runs/` reports
- baseline pytest output
- benchmark result tables
- demo screenshots or GIFs
