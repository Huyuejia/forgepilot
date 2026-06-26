import json

from pico import FakeModelClient, MiniAgent, SessionStore, WorkspaceContext


def build_workspace(tmp_path):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    return WorkspaceContext.build(tmp_path)


def build_agent(tmp_path, outputs):
    workspace = build_workspace(tmp_path)
    store = SessionStore(tmp_path / ".pico" / "sessions")
    return MiniAgent(
        model_client=FakeModelClient(outputs),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
    )


def test_runtime_initializes_empty_plan_state(tmp_path):
    agent = build_agent(tmp_path, [])

    assert agent.session["plan"] == {"items": [], "active_id": ""}
    assert agent.plan_text() == "Plan:\n- none"


def test_plan_text_is_included_in_prompt(tmp_path):
    agent = build_agent(tmp_path, [])
    agent.set_plan_items(["Read code", "Run tests"])

    prompt = agent.prompt("Continue")

    assert "Plan:\n- [in_progress] step-1: Read code" in prompt
    assert "- [pending] step-2: Run tests" in prompt


def test_report_contains_plan_summary(tmp_path):
    agent = build_agent(tmp_path, ["<final>Done.</final>"])
    agent.set_plan_items(["Read code", "Run tests"])

    assert agent.ask("Do the work") == "Done."

    report = json.loads(agent.run_store.report_path(agent.current_task_state).read_text(encoding="utf-8"))
    assert report["plan"]["total"] == 2
    assert report["plan"]["in_progress"] == 1
    assert report["plan"]["pending"] == 1
