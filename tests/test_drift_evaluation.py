import json
from pathlib import Path

from forgepilot.drift_evaluation import (
    DriftBenchmarkEvaluator,
    DriftModelClient,
    load_drift_benchmark,
    load_drift_holdout,
    normalize_drift_outcome,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DRIFT_BENCHMARK = REPO_ROOT / "benchmarks" / "drift_mvp.json"
DRIFT_HOLDOUT = REPO_ROOT / "benchmarks" / "drift_mvp_holdout.json"


def test_drift_benchmark_has_accumulated_pressure_and_fixed_holdout():
    benchmark = load_drift_benchmark(DRIFT_BENCHMARK)
    holdout = load_drift_holdout(DRIFT_HOLDOUT, benchmark=benchmark)

    assert len(benchmark["tasks"]) == 12
    assert 0 < len(holdout["task_ids"]) < len(benchmark["tasks"])
    assert holdout["assignment"] == "fixed-before-treatment"
    assert all(int(task["pressure"]["intervening_steps"]) >= 3 for task in benchmark["tasks"])
    assert any(task["pressure"]["context_reduction"] for task in benchmark["tasks"])
    assert any(task["pressure"]["resume"] for task in benchmark["tasks"])


def test_drift_normalization_uses_external_verifier_not_model_self_report():
    outcome = normalize_drift_outcome(
        arm="baseline",
        run_artifact={
            "status": "completed",
            "stop_reason": "final_answer_returned",
            "final_answer": "Done; the constraint was preserved.",
        },
        verifier_passed=False,
        expected_success=True,
        prohibited_mutation_observed=True,
    )

    assert outcome["model_claimed_completion"] is True
    assert outcome["task_completion"] is True
    assert outcome["drift_failure"] is True
    assert outcome["constraint_violation"] is True
    assert outcome["false_completion"] is True


def test_drift_model_uses_protected_projection_as_the_arm_treatment():
    client = DriftModelClient(
        task={
            "id": "drift_test",
            "target_path": "target.txt",
            "expected_content": "keep-current\n",
            "distractor_content": "stale-choice\n",
            "intervening_outputs": [],
        }
    )

    baseline = client.complete("ordinary prompt", 64)
    enhanced = DriftModelClient(client.task).complete("ordinary prompt\nProtected Current Task Contract:\ngoal: keep current", 64)

    assert "stale-choice" in baseline
    assert "keep-current" in enhanced


def test_drift_evaluator_executes_both_arms_and_persists_trace_report(tmp_path):
    benchmark = load_drift_benchmark(DRIFT_BENCHMARK)
    task = benchmark["tasks"][0]
    artifact_path = tmp_path / "drift.json"
    result = DriftBenchmarkEvaluator(
        benchmark_path=DRIFT_BENCHMARK,
        holdout_path=DRIFT_HOLDOUT,
        artifact_path=artifact_path,
        workspace_root=tmp_path / "workspaces",
        repeats=1,
        task_ids=[task["id"]],
    ).run()

    assert result["summary"]["task_count"] == 1
    assert result["summary"]["total_run_count"] == 2
    assert {row["arm"] for row in result["rows"]} == {"baseline", "enhanced"}
    by_arm = {row["arm"]: row for row in result["rows"]}
    assert by_arm["baseline"]["execution_mode"] == "compatibility"
    assert by_arm["enhanced"]["execution_mode"] == "enhanced_contract"
    assert by_arm["baseline"]["metrics"]["drift_failure"] is True
    assert by_arm["enhanced"]["metrics"]["current_intent_retained"] is True
    assert by_arm["enhanced"]["saw_protected_projection"] is True
    assert all(row["trace_path"] and row["report_path"] for row in result["rows"])
    assert artifact_path.is_file()
    assert json.loads(artifact_path.read_text(encoding="utf-8"))["summary"] == result["summary"]
