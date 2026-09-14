#!/usr/bin/env python3
"""Run the Long-horizon Drift Evaluation MVP."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forgepilot.drift_evaluation import (
    DEFAULT_DRIFT_ARTIFACT_PATH,
    DEFAULT_DRIFT_BENCHMARK_PATH,
    DEFAULT_DRIFT_HOLDOUT_PATH,
    DriftBenchmarkEvaluator,
    load_drift_benchmark,
    load_drift_holdout,
)


def main():
    parser = argparse.ArgumentParser(description="Run ForgePilot's Long-horizon Drift Evaluation MVP.")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_DRIFT_BENCHMARK_PATH)
    parser.add_argument("--holdout", type=Path, default=DEFAULT_DRIFT_HOLDOUT_PATH)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_DRIFT_ARTIFACT_PATH)
    parser.add_argument("--workspace-root", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--exclude-holdout", action="store_true")
    parser.add_argument("--task-id", action="append", dest="task_ids")
    args = parser.parse_args()

    task_ids = args.task_ids
    if args.exclude_holdout:
        benchmark = load_drift_benchmark(args.benchmark)
        holdout = load_drift_holdout(args.holdout, benchmark=benchmark)
        holdout_ids = set(holdout["task_ids"])
        if task_ids is None:
            task_ids = [task["id"] for task in benchmark["tasks"] if task["id"] not in holdout_ids]
        elif holdout_ids.intersection(task_ids):
            parser.error("--exclude-holdout cannot be combined with a holdout --task-id")

    result = DriftBenchmarkEvaluator(
        benchmark_path=args.benchmark,
        holdout_path=args.holdout,
        artifact_path=args.artifact,
        workspace_root=args.workspace_root,
        repeats=args.repeats,
        task_ids=task_ids,
    ).run()
    print(json.dumps({"artifact": str(args.artifact), "summary": result["summary"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
