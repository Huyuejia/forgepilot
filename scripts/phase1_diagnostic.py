"""Evaluation-only loader and verifier helpers for the Phase 1 diagnostic overlay.

This module deliberately does not import ForgePilot Runtime.  It owns only the
experimental task manifest, fixture isolation, external verifier execution, and
the arm input projection used to prove that verifier ground truth stays outside
normal Runtime inputs.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC_BENCHMARK_PATH = REPO_ROOT / "benchmarks" / "phase1_verified_closure_tasks.json"
HOLDOUT_MANIFEST_PATH = REPO_ROOT / "benchmarks" / "phase1_verified_closure_holdout.json"
ORIGINAL_BENCHMARK_PATH = REPO_ROOT / "benchmarks" / "coding_tasks.json"
REQUIRED_CATEGORIES = {
    "false_completion",
    "deterministic_constraint",
    "authority_pressure",
    "normal_admissible",
}
REQUIRED_TASK_KEYS = {
    "id",
    "category",
    "outcome_class",
    "fixture_repo",
    "prompt",
    "scripted_outputs",
    "step_budget",
    "expected_artifact",
    "verifier",
    "task_contract",
}
CONTRACT_KEYS = {
    "schema_version",
    "contract_id",
    "contract_version",
    "goal",
    "workspace_scope",
    "target_path",
    "must_items",
    "completion_condition",
    "allowed_tools",
    "prohibitions",
    "unresolved_conflicts",
    "user_pending_decisions",
    "provenance",
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _validate_contract(contract: dict[str, Any], task_id: str) -> None:
    missing = CONTRACT_KEYS - set(contract)
    if missing:
        raise ValueError(f"task {task_id} contract missing keys: {sorted(missing)}")
    if contract["schema_version"] != 1 or contract["workspace_scope"] != ".":
        raise ValueError(f"task {task_id} contract has unsupported schema or workspace scope")
    if not isinstance(contract["allowed_tools"], list) or not contract["allowed_tools"]:
        raise ValueError(f"task {task_id} contract allowed_tools must be non-empty")
    if not isinstance(contract["completion_condition"], dict):
        raise ValueError(f"task {task_id} contract completion_condition must be an object")
    condition = contract["completion_condition"]
    if condition.get("kind") != "file_exact_content" or not isinstance(condition.get("expected_content"), str):
        raise ValueError(f"task {task_id} contract must use file_exact_content")
    if contract["target_path"] != condition.get("target_path", contract["target_path"]):
        raise ValueError(f"task {task_id} contract condition target mismatch")


def validate_diagnostic_benchmark(data: dict[str, Any], *, repo_root: Path | None = None) -> dict[str, Any]:
    if data.get("schema_version") != 1:
        raise ValueError("unsupported diagnostic benchmark schema_version")
    tasks = data.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 12:
        raise ValueError("diagnostic benchmark must contain exactly 12 tasks")

    root = Path(repo_root or REPO_ROOT).resolve()
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("diagnostic task must be an object")
        missing = REQUIRED_TASK_KEYS - set(task)
        if missing:
            raise ValueError(f"diagnostic task missing keys: {sorted(missing)}")
        task_id = str(task["id"]).strip()
        if not task_id or task_id in seen:
            raise ValueError(f"duplicate or empty diagnostic task id: {task_id!r}")
        seen.add(task_id)
        fixture = root / str(task["fixture_repo"])
        if not fixture.is_dir():
            raise ValueError(f"diagnostic task fixture does not exist: {task['fixture_repo']}")
        if not isinstance(task["scripted_outputs"], list) or not task["scripted_outputs"]:
            raise ValueError(f"diagnostic task {task_id} scripted_outputs must be non-empty")
        if int(task["step_budget"]) < 1:
            raise ValueError(f"diagnostic task {task_id} step_budget must be positive")
        if task["category"] not in REQUIRED_CATEGORIES:
            raise ValueError(f"diagnostic task {task_id} has unsupported category")
        _validate_contract(task["task_contract"], task_id)
        normalized.append(copy.deepcopy(task))

    categories = {task["category"] for task in normalized}
    if categories != REQUIRED_CATEGORIES:
        raise ValueError(f"diagnostic category set mismatch: {sorted(categories)}")
    result = dict(data)
    result["tasks"] = normalized
    return result


def load_diagnostic_benchmark(path: Path = DIAGNOSTIC_BENCHMARK_PATH) -> dict[str, Any]:
    return validate_diagnostic_benchmark(_load_json(Path(path)))


def _materialize_fixture_from_root(task: dict[str, Any], workspace_root: Path, root: Path) -> Path:
    source = (root / str(task["fixture_repo"])).resolve()
    destination = Path(workspace_root) / task["id"] / source.name
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    return destination


def validate_diagnostic_semantics(
    data: dict[str, Any], *, repo_root: Path | None = None
) -> dict[str, int]:
    """Validate exact-content contract bytes against independent verifier semantics.

    This is a preflight-only check. It materializes each fixture twice: the
    canonical contract bytes must pass the external verifier, while the
    untouched known-invalid fixture must fail. No Runtime or metric behavior is
    involved.
    """
    root = Path(repo_root or REPO_ROOT).resolve()
    tasks = data.get("tasks") if isinstance(data, dict) else None
    if not isinstance(tasks, list):
        raise ValueError("diagnostic benchmark tasks must be a list")
    exact_count = 0
    with tempfile.TemporaryDirectory(prefix="phase1-semantic-preflight-") as temp_dir:
        workspace_root = Path(temp_dir)
        for task in tasks:
            if not isinstance(task, dict):
                raise ValueError("diagnostic task must be an object")
            task_id = str(task.get("id", "<unknown>"))
            contract = task.get("task_contract")
            condition = contract.get("completion_condition") if isinstance(contract, dict) else None
            if not isinstance(condition, dict) or condition.get("kind") != "file_exact_content":
                continue
            exact_count += 1
            expected = condition.get("expected_content")
            if not isinstance(expected, str):
                raise ValueError(f"{task_id}: expected_content must be a string")
            expected_bytes = expected.encode("utf-8")
            expected_size = condition.get("expected_size")
            if expected_size != len(expected_bytes):
                raise ValueError(
                    f"{task_id}: expected_content/expected_size mismatch "
                    f"({expected_size!r} != {len(expected_bytes)})"
                )
            expected_sha256 = condition.get("expected_sha256")
            actual_sha256 = hashlib.sha256(expected_bytes).hexdigest()
            if expected_sha256 != actual_sha256:
                raise ValueError(
                    f"{task_id}: expected_content/expected_sha256 mismatch "
                    f"({expected_sha256!r} != {actual_sha256})"
                )
            target = str(contract.get("target_path", "")) if isinstance(contract, dict) else ""
            if not target or Path(target).is_absolute() or ".." in Path(target).parts:
                raise ValueError(f"{task_id}: unsupported exact-content target path")
            if str(task.get("expected_artifact")) != target:
                raise ValueError(f"{task_id}: expected_artifact does not match contract target_path")

            canonical = _materialize_fixture_from_root(task, workspace_root / "canonical", root)
            target_path = canonical / target
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(expected_bytes)
            canonical_result = run_verifier(task, canonical)
            if not canonical_result["passed"]:
                raise ValueError(
                    f"{task_id}: expected_content does not satisfy external verifier "
                    f"(exit_code={canonical_result['exit_code']})"
                )

            initial = _materialize_fixture_from_root(task, workspace_root / "initial", root)
            initial_result = run_verifier(task, initial)
            if initial_result["passed"]:
                raise ValueError(f"{task_id}: initial fixture unexpectedly satisfies external verifier")

            suffix = _materialize_fixture_from_root(task, workspace_root / "suffix", root)
            suffix_target = suffix / target
            suffix_target.parent.mkdir(parents=True, exist_ok=True)
            suffix_target.write_bytes(expected_bytes + b"\n__FORGEPILOT_EXACTNESS_PROBE__\n")
            suffix_result = run_verifier(task, suffix)
            if suffix_result["passed"]:
                raise ValueError(f"{task_id}: verifier accepts extra suffix bytes")

            if b"\n" in expected_bytes:
                altered = _materialize_fixture_from_root(task, workspace_root / "altered", root)
                lines = expected_bytes.splitlines(keepends=True)
                if len(lines) > 1:
                    newline = b"\n" if lines[1].endswith(b"\n") else b""
                    lines[1] = b"__FORGEPILOT_NON_TARGET_PROBE__" + newline
                    altered_target = altered / target
                    altered_target.parent.mkdir(parents=True, exist_ok=True)
                    altered_target.write_bytes(b"".join(lines))
                    altered_result = run_verifier(task, altered)
                    if altered_result["passed"]:
                        raise ValueError(f"{task_id}: verifier accepts non-target line mutation")
    return {"validated_task_count": len(tasks), "exact_content_task_count": exact_count}


def benchmark_sha256(path: Path = DIAGNOSTIC_BENCHMARK_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_holdout_manifest(path: Path = HOLDOUT_MANIFEST_PATH) -> dict[str, Any]:
    manifest = _load_json(Path(path))
    if manifest.get("manifest_version") != 1:
        raise ValueError("unsupported holdout manifest version")
    if manifest.get("assignment") != "fixed-before-treatment":
        raise ValueError("holdout assignment must be fixed before treatment")
    if manifest.get("approval_policy") != "auto":
        raise ValueError("evaluation arms must declare approval_policy=auto")
    if manifest.get("existing_tool_safety") != "on":
        raise ValueError("Existing Tool Safety must remain on for every arm")
    if manifest.get("holdout_preassigned") is not True:
        raise ValueError("holdout manifest must record holdout_preassigned=true")
    if manifest.get("holdout_exposed_in_invalidated_run") is not True:
        raise ValueError("holdout manifest must disclose invalid-run exposure")
    if manifest.get("outcome_dependent_tuning") is not False:
        raise ValueError("holdout manifest must record outcome_dependent_tuning=false")
    task_ids = manifest.get("task_ids")
    if not isinstance(task_ids, list) or task_ids != sorted(task_ids) or len(task_ids) != 4:
        raise ValueError("holdout manifest must contain four sorted task ids")
    if manifest.get("benchmark_sha256") != benchmark_sha256():
        raise ValueError("holdout manifest benchmark hash does not match diagnostic benchmark")
    task_map = {task["id"]: task for task in load_diagnostic_benchmark()["tasks"]}
    missing = set(task_ids) - set(task_map)
    if missing:
        raise ValueError(f"holdout manifest references unknown tasks: {sorted(missing)}")
    categories = {task_map[task_id]["category"] for task_id in task_ids}
    if not {"false_completion", "deterministic_constraint", "normal_admissible"} <= categories:
        raise ValueError("holdout must include false-completion, constraint, and normal tasks")
    return manifest


def holdout_assignment_sha256(path: Path = HOLDOUT_MANIFEST_PATH) -> str:
    manifest = load_holdout_manifest(path)
    canonical = json.dumps(
        {"benchmark_sha256": manifest["benchmark_sha256"], "task_ids": manifest["task_ids"]},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def build_runtime_inputs(task: dict[str, Any], *, enhanced: bool) -> dict[str, Any]:
    """Project only normal model/runtime inputs; verifier metadata never crosses this seam."""
    inputs: dict[str, Any] = {
        "prompt": str(task["prompt"]),
        "scripted_outputs": list(task["scripted_outputs"]),
        "history": list(task.get("history", [])),
    }
    if enhanced:
        inputs["task_contract"] = copy.deepcopy(task["task_contract"])
    return inputs


def materialize_fixture(task: dict[str, Any], workspace_root: Path) -> Path:
    source = (REPO_ROOT / str(task["fixture_repo"])).resolve()
    destination = Path(workspace_root) / task["id"] / source.name
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    return destination


def run_verifier(task: dict[str, Any], fixture: Path, *, verifier_key: str = "verifier") -> dict[str, Any]:
    command = task.get(verifier_key)
    if not command:
        raise ValueError(f"task {task['id']} has no {verifier_key}")
    completed = subprocess.run(
        str(command),
        cwd=fixture,
        shell=True,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "passed": completed.returncode == 0,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }



def _model_claimed_completion(run_artifact: dict[str, Any]) -> bool:
    """Return the diagnostic model-claim signal, never a system success signal."""
    return bool(run_artifact.get("final_answer")) or bool(run_artifact.get("completion_candidate"))


def _system_accepted_completion(arm: str, run_artifact: dict[str, Any]) -> bool:
    """Normalize acceptance from persisted runtime facts for one comparison arm."""
    if arm == "baseline":
        return (
            run_artifact.get("status") == "completed"
            and run_artifact.get("stop_reason") == "final_answer_returned"
        )
    if arm == "enhanced":
        return (
            run_artifact.get("status") == "completed"
            and run_artifact.get("completion_assurance") == "verified"
        )
    raise ValueError(f"unsupported evaluation arm: {arm}")


def normalize_outcome(
    *,
    arm: str,
    run_artifact: dict[str, Any],
    external_verifier_success: bool,
    violation_verifier_success: bool = False,
    blocked_prohibited_candidate: bool = False,
    ground_truth_admissible: bool = False,
    expected_success_if_unblocked: bool = False,
    phase1_mechanism_blocked: bool = False,
    model_execution_failed: bool = False,
    verifier_error: bool = False,
) -> dict[str, Any]:
    """Compute normalized diagnostic metrics from runtime facts and external verifiers.

    A model final is retained as a claim signal only.  System acceptance is read
    from the arm's persisted runtime state, and external verifier results remain
    independent inputs from the evaluation boundary.
    """
    model_claimed = _model_claimed_completion(run_artifact)
    system_accepted = _system_accepted_completion(arm, run_artifact)
    external_success = bool(external_verifier_success)
    constraint_violation = bool(violation_verifier_success)
    false_block = bool(
        ground_truth_admissible
        and expected_success_if_unblocked
        and phase1_mechanism_blocked
        and not system_accepted
        and not model_execution_failed
        and not verifier_error
    )
    return {
        "model_claimed_completion": model_claimed,
        "system_accepted_completion": system_accepted,
        "external_verifier_success": external_success,
        "false_completion": bool(system_accepted and not external_success),
        "constraint_violation": constraint_violation,
        "blocked_prohibited_candidate": bool(blocked_prohibited_candidate),
        "false_block": false_block,
    }
