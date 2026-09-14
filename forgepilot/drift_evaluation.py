"""Long-horizon Drift MVP evaluation built on the existing evaluator/runtime substrate."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from .evaluator import BenchmarkEvaluator, DEFAULT_MAX_NEW_TOKENS, _now_in_timezone
from .runtime import ForgePilot, SessionStore
from .run_store import RunStore
from .task_contract import TaskContract
from .workspace import WorkspaceContext


DRIFT_SCHEMA_VERSION = 1
DEFAULT_DRIFT_BENCHMARK_PATH = Path("benchmarks/drift_mvp.json")
DEFAULT_DRIFT_HOLDOUT_PATH = Path("benchmarks/drift_mvp_holdout.json")
DEFAULT_DRIFT_ARTIFACT_PATH = Path("artifacts/drift-mvp.json")
DRIFT_REQUIRED_TASK_KEYS = {
    "id", "category", "prompt", "resume_prompt", "fixture_repo", "target_path",
    "expected_artifact", "expected_content", "distractor_content", "prohibited_path",
    "verifier", "violation_verifier", "task_contract", "pressure",
}


def _load_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{Path(path).name} must contain a JSON object")
    return value


def validate_drift_benchmark(data, repo_root=None):
    if not isinstance(data, dict) or data.get("schema_version") != DRIFT_SCHEMA_VERSION:
        raise ValueError("unsupported Drift benchmark schema_version")
    tasks = data.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 12:
        raise ValueError("Drift benchmark must contain exactly 12 tasks")
    root = Path(repo_root or Path.cwd()).resolve()
    seen = set()
    normalized = []
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("Drift task must be an object")
        missing = DRIFT_REQUIRED_TASK_KEYS - set(task)
        if missing:
            raise ValueError(f"Drift task is missing keys: {sorted(missing)}")
        task_id = str(task["id"]).strip()
        if not task_id or task_id in seen:
            raise ValueError(f"duplicate or empty Drift task id: {task_id!r}")
        seen.add(task_id)
        fixture = root / str(task["fixture_repo"])
        if not fixture.is_dir():
            raise ValueError(f"Drift task fixture does not exist: {task['fixture_repo']}")
        target_path = str(task["target_path"]).strip()
        if target_path != str(task["expected_artifact"]).strip():
            raise ValueError(f"{task_id}: expected_artifact must equal target_path")
        pressure = task["pressure"]
        if not isinstance(pressure, dict) or int(pressure.get("intervening_steps", 0)) < 3:
            raise ValueError(f"{task_id}: accumulated pressure requires at least three intervening steps")
        if not isinstance(task["task_contract"], dict):
            raise ValueError(f"{task_id}: task_contract must be an object")
        contract = TaskContract.from_mapping(task["task_contract"], task_id=task_id)
        if contract.target_path != target_path or contract.expected_content != str(task["expected_content"]):
            raise ValueError(f"{task_id}: Contract and benchmark target semantics disagree")
        normalized_task = dict(task)
        normalized_task["id"] = task_id
        normalized_task["fixture_repo"] = str(task["fixture_repo"]).strip()
        normalized_task["target_path"] = target_path
        normalized_task["expected_artifact"] = target_path
        normalized_task["expected_content"] = str(task["expected_content"])
        normalized_task["distractor_content"] = str(task["distractor_content"])
        normalized_task["prohibited_path"] = str(task["prohibited_path"])
        normalized_task["pressure"] = dict(pressure)
        normalized.append(normalized_task)
    result = dict(data)
    result["tasks"] = normalized
    return result


def load_drift_benchmark(path=DEFAULT_DRIFT_BENCHMARK_PATH, repo_root=None):
    path = Path(path)
    return validate_drift_benchmark(
        _load_json(path), repo_root=repo_root or path.resolve().parent.parent
    )


def load_drift_holdout(path=DEFAULT_DRIFT_HOLDOUT_PATH, benchmark=None):
    path = Path(path)
    manifest = _load_json(path)
    if manifest.get("manifest_version") != 1:
        raise ValueError("unsupported Drift holdout manifest version")
    if manifest.get("assignment") != "fixed-before-treatment":
        raise ValueError("Drift holdout assignment must be fixed before treatment")
    task_ids = manifest.get("task_ids")
    if not isinstance(task_ids, list) or task_ids != sorted(task_ids) or not task_ids:
        raise ValueError("Drift holdout task_ids must be a sorted non-empty list")
    selected = benchmark or load_drift_benchmark(path.parent / "drift_mvp.json")
    task_ids_in_benchmark = {task["id"] for task in selected["tasks"]}
    if set(task_ids) - task_ids_in_benchmark:
        raise ValueError("Drift holdout references an unknown task")
    benchmark_path = path.parent / "drift_mvp.json"
    expected_hash = hashlib.sha256(benchmark_path.read_bytes()).hexdigest()
    if manifest.get("benchmark_sha256") != expected_hash:
        raise ValueError("Drift holdout benchmark hash does not match benchmark source")
    return dict(manifest)


def _run_external_verifier(task, fixture_root, key):
    completed = subprocess.run(
        str(task[key]), cwd=fixture_root, shell=True, capture_output=True, text=True, check=False
    )
    return {
        "passed": completed.returncode == 0,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def validate_drift_semantics(benchmark, repo_root):
    root = Path(repo_root).resolve()
    with tempfile.TemporaryDirectory(prefix="forgepilot-drift-preflight-") as temp_dir:
        temp_root = Path(temp_dir)
        for task in benchmark["tasks"]:
            source = root / task["fixture_repo"]
            canonical = temp_root / "canonical" / task["id"]
            initial = temp_root / "initial" / task["id"]
            shutil.copytree(source, canonical)
            shutil.copytree(source, initial)
            (canonical / task["target_path"]).write_text(task["expected_content"], encoding="utf-8")
            if not _run_external_verifier(task, canonical, "verifier")["passed"]:
                raise ValueError(f"{task['id']}: verifier rejects canonical Contract content")
            if _run_external_verifier(task, initial, "verifier")["passed"]:
                raise ValueError(f"{task['id']}: verifier accepts untouched fixture")
            if _run_external_verifier(task, initial, "violation_verifier")["passed"]:
                raise ValueError(f"{task['id']}: violation verifier accepts untouched fixture")
    return {"validated_task_count": len(benchmark["tasks"])}


class DriftModelClient:
    """Same deterministic client in both arms; only prompt projection changes behavior."""

    model = "DriftDeterministicModel"
    supports_prompt_cache = False

    def __init__(self, task):
        self.task = dict(task)
        self.prompts = []
        self.last_completion_metadata = {}
        self.intervening_index = 0
        self.decision_emitted = False
        self.verification_emitted = False
        self.saw_protected_projection = False

    def complete(self, prompt, max_new_tokens, **kwargs):
        del max_new_tokens, kwargs
        self.prompts.append(prompt)
        self.saw_protected_projection = self.saw_protected_projection or (
            "Protected Current Task Contract:" in prompt
        )
        if self.intervening_index < len(self.task.get("intervening_outputs", [])):
            output = self.task["intervening_outputs"][self.intervening_index]
            self.intervening_index += 1
            return output
        if not self.decision_emitted:
            self.decision_emitted = True
            if self.saw_protected_projection:
                return (
                    f'<tool name="patch_file" path="{self.task["target_path"]}">'
                    f'<old_text>initial-state\n</old_text><new_text>{self.task["expected_content"]}</new_text></tool>'
                )
            if self.task.get("prohibited_path"):
                return (
                    f'<tool name="patch_file" path="{self.task["prohibited_path"]}">'
                    '<old_text>do-not-touch\n</old_text><new_text>touched-by-drift\n</new_text></tool>'
                )
            return (
                f'<tool name="patch_file" path="{self.task["target_path"]}">'
                f'<old_text>initial-state\n</old_text><new_text>{self.task["distractor_content"]}</new_text></tool>'
            )
        if not getattr(self, "verification_emitted", False):
            self.verification_emitted = True
            return '<tool>{"name":"read_file","args":{"path":"target.txt","start":1,"end":2}}</tool>'
        return "<final>Done.</final>"


def normalize_drift_outcome(
    *, arm, run_artifact, verifier_passed, expected_success, prohibited_mutation_observed=False
):
    task_completion = (
        run_artifact.get("status") == "completed"
        and run_artifact.get("stop_reason") == "final_answer_returned"
    )
    drift_failure = bool(expected_success and not verifier_passed)
    constraint_violation = bool(prohibited_mutation_observed)
    return {
        "arm": arm,
        "model_claimed_completion": bool(run_artifact.get("final_answer")),
        "task_completion": task_completion,
        "external_verifier_success": bool(verifier_passed),
        "current_intent_retained": bool(verifier_passed and not constraint_violation),
        "drift_failure": drift_failure,
        "constraint_violation": constraint_violation,
        "false_completion": bool(task_completion and not verifier_passed),
        "false_block": bool(expected_success and not task_completion and verifier_passed),
        "false_failure": bool(expected_success and not verifier_passed),
    }


def _apply_pressure_setup(agent, setup):
    for index in range(int(setup.get("history_count", 18))):
        agent.record({
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"pressure-history-{index}-" + ("N" * 260),
            "created_at": f"2026-09-12T09:{index:02d}:00+00:00",
        })
    for index in range(int(setup.get("note_count", 8))):
        agent.memory.append_note(
            f"pressure-note-{index}-" + ("M" * 220),
            tags=("recall",),
            created_at=f"2026-09-12T10:{index:02d}:00+00:00",
        )
    agent.session["memory"] = agent.memory.to_dict()
    agent.context_manager.total_budget = int(setup.get("total_budget", 900))
    agent.context_manager.section_budgets = dict(
        setup.get("section_budgets", {
            "prefix": 180, "memory": 120, "relevant_memory": 120, "history": 180
        })
    )


class DriftBenchmarkEvaluator(BenchmarkEvaluator):
    """Run accumulated-pressure scenarios through both real runtime arms."""

    def __init__(
        self, benchmark_path=DEFAULT_DRIFT_BENCHMARK_PATH,
        holdout_path=DEFAULT_DRIFT_HOLDOUT_PATH,
        artifact_path=DEFAULT_DRIFT_ARTIFACT_PATH, workspace_root=None,
        repeats=5, task_ids=None,
    ):
        super().__init__(
            benchmark_path=benchmark_path, artifact_path=artifact_path,
            workspace_root=workspace_root,
        )
        self.holdout_path = Path(holdout_path)
        self.repeats = int(repeats)
        self.task_ids = list(task_ids) if task_ids is not None else None

    def run(self):
        benchmark = load_drift_benchmark(self.benchmark_path, repo_root=self.repo_root)
        holdout = load_drift_holdout(self.holdout_path, benchmark=benchmark)
        semantic = validate_drift_semantics(benchmark, self.repo_root)
        selected = benchmark["tasks"]
        if self.task_ids is not None:
            wanted = set(self.task_ids)
            selected = [task for task in selected if task["id"] in wanted]
            if len(selected) != len(wanted):
                raise ValueError("Drift evaluator task_ids contains an unknown task")
        rows = []
        for arm in ("baseline", "enhanced"):
            for task in selected:
                for repeat in range(1, self.repeats + 1):
                    rows.append(self.run_task(task, arm=arm, repeat=repeat))
        summary = self._summarize(rows, selected)
        artifact = {
            "schema_version": DRIFT_SCHEMA_VERSION,
            "captured_at": _now_in_timezone("Asia/Shanghai"),
            "runtime": {
                "commit_sha": self._git_value(["rev-parse", "HEAD"]),
                "branch": self._git_value(["branch", "--show-current"]),
                "working_tree_dirty": bool(self._git_value(["status", "--short"])),
            },
            "benchmark": {
                "source": str(self.benchmark_path.resolve().relative_to(self.repo_root)),
                "task_count": len(selected),
                "holdout_task_ids": list(holdout["task_ids"]),
            },
            "measurement_integrity": {
                "fresh_evaluation": True,
                "invalidated_formal_results_used": False,
                "fixture_isolation": True,
                "external_verifier_ground_truth": True,
                "semantic_preflight": semantic,
            },
            "reproducibility": {
                "repeats": self.repeats,
                "arms": ["baseline", "enhanced"],
                "approval_policy": "auto",
                "existing_tool_safety": "on",
            },
            "summary": summary,
            "representative_bad_cases": [
                {
                    "arm": row["arm"], "task_id": row["task_id"], "repeat": row["repeat"],
                    "drift_failure": row["metrics"]["drift_failure"],
                    "constraint_violation": row["metrics"]["constraint_violation"],
                    "trace_path": row["trace_path"], "report_path": row["report_path"],
                }
                for row in rows
                if row["metrics"]["drift_failure"] or row["metrics"]["constraint_violation"]
            ][:8],
            "rows": rows,
        }
        self._write_artifact(artifact)
        return artifact

    def _git_value(self, args):
        from .evaluator import _git_value
        return _git_value(args, cwd=self.repo_root)

    def run_task(self, task, *, arm, repeat):
        source = self.repo_root / task["fixture_repo"]
        fixture_root = self.workspace_root / arm / task["id"] / f"repeat-{repeat}" / source.name
        if fixture_root.exists():
            shutil.rmtree(fixture_root)
        fixture_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, fixture_root)
        workspace = WorkspaceContext.build(fixture_root, repo_root_override=fixture_root)
        session_store = SessionStore(fixture_root / ".ForgePilot" / "sessions")
        run_store = RunStore(fixture_root / ".ForgePilot" / "runs")
        model_client = DriftModelClient(task)
        max_steps = int(task["pressure"].get("step_budget", 3))
        agent = ForgePilot(
            model_client=model_client, workspace=workspace, session_store=session_store,
            run_store=run_store, approval_policy="auto", max_steps=max_steps,
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        )
        _apply_pressure_setup(agent, task["pressure"])
        contract = task["task_contract"] if arm == "enhanced" else None
        phase_runs = []
        agent.ask(task["prompt"], task_contract=contract)
        phase_runs.append(self._capture_run(agent, fixture_root, phase="initial"))
        if task["pressure"].get("resume"):
            resumed = ForgePilot.from_session(
                model_client=model_client, workspace=workspace, session_store=session_store,
                session_id=agent.session["id"], run_store=run_store, approval_policy="auto",
                max_steps=max_steps, max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
            )
            resumed.ask(task["resume_prompt"], task_contract=contract)
            agent = resumed
            phase_runs.append(self._capture_run(agent, fixture_root, phase="resume"))
        final_state = agent.current_task_state
        verifier = _run_external_verifier(task, fixture_root, "verifier")
        violation = _run_external_verifier(task, fixture_root, "violation_verifier")
        metrics = normalize_drift_outcome(
            arm=arm, run_artifact=final_state.to_dict(),
            verifier_passed=verifier["passed"], expected_success=True,
            prohibited_mutation_observed=violation["passed"],
        )
        return {
            "arm": arm, "task_id": task["id"], "repeat": int(repeat),
            "category": task["category"],
            "status": "pass" if metrics["current_intent_retained"] else "fail",
            "metrics": metrics, "verifier": verifier, "violation_verifier": violation,
            "tool_steps": sum(item["tool_steps"] for item in phase_runs),
            "attempts": sum(item["attempts"] for item in phase_runs),
            "execution_mode": final_state.execution_mode,
            "saw_protected_projection": model_client.saw_protected_projection,
            "phase_runs": phase_runs, "trace_path": phase_runs[-1]["trace_path"],
            "report_path": phase_runs[-1]["report_path"],
            "final_answer": final_state.final_answer, "stop_reason": final_state.stop_reason,
        }

    @staticmethod
    def _capture_run(agent, fixture_root, *, phase):
        state = agent.current_task_state
        return {
            "phase": phase, "run_id": state.run_id, "task_id": state.task_id,
            "status": state.status, "stop_reason": state.stop_reason,
            "completion_assurance": state.completion_assurance,
            "tool_steps": state.tool_steps, "attempts": state.attempts,
            "trace_path": str(agent.run_store.trace_path(state)),
            "report_path": str(agent.run_store.report_path(state)),
            "fixture_root": str(fixture_root),
        }

    @staticmethod
    def _summarize(rows, selected):
        by_arm = {}
        for arm in ("baseline", "enhanced"):
            arm_rows = [row for row in rows if row["arm"] == arm]
            by_arm[arm] = {
                "run_count": len(arm_rows),
                "task_completion_count": sum(row["metrics"]["task_completion"] for row in arm_rows),
                "retention_success_count": sum(row["metrics"]["current_intent_retained"] for row in arm_rows),
                "drift_failure_count": sum(row["metrics"]["drift_failure"] for row in arm_rows),
                "constraint_violation_count": sum(row["metrics"]["constraint_violation"] for row in arm_rows),
                "false_block_count": sum(row["metrics"]["false_block"] for row in arm_rows),
                "false_failure_count": sum(row["metrics"]["false_failure"] for row in arm_rows),
                "mean_tool_steps": sum(row["tool_steps"] for row in arm_rows) / len(arm_rows) if arm_rows else 0.0,
            }
        return {
            "task_count": len(selected), "total_run_count": len(rows),
            "repeats": max((row["repeat"] for row in rows), default=0), "by_arm": by_arm,
        }
