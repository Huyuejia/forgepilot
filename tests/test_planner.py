from pico.planner import (
    PLAN_STATUSES,
    PLAN_STATUS_BLOCKED,
    PLAN_STATUS_COMPLETED,
    PLAN_STATUS_IN_PROGRESS,
    PLAN_STATUS_PENDING,
    default_plan_state,
    normalize_plan_state,
    plan_summary,
    render_plan_text,
    set_plan_items,
    update_plan_item_status,
)


def test_plan_status_constants_match_task_spec():
    assert PLAN_STATUS_PENDING == "pending"
    assert PLAN_STATUS_IN_PROGRESS == "in_progress"
    assert PLAN_STATUS_COMPLETED == "completed"
    assert PLAN_STATUS_BLOCKED == "blocked"
    assert PLAN_STATUSES == {
        PLAN_STATUS_PENDING,
        PLAN_STATUS_IN_PROGRESS,
        PLAN_STATUS_COMPLETED,
        PLAN_STATUS_BLOCKED,
    }


def test_normalize_plan_state_keeps_only_first_active_item():
    state = normalize_plan_state(
        {
            "items": [
                {"id": "read", "text": "Read code", "status": PLAN_STATUS_PENDING},
                {"id": "edit", "text": "Edit code", "status": PLAN_STATUS_IN_PROGRESS},
                {"id": "test", "text": "Run tests", "status": PLAN_STATUS_IN_PROGRESS},
            ],
            "active_id": "test",
        }
    )

    assert [item["status"] for item in state["items"]] == [
        PLAN_STATUS_PENDING,
        PLAN_STATUS_IN_PROGRESS,
        PLAN_STATUS_PENDING,
    ]
    assert state["active_id"] == "edit"


def test_set_plan_items_creates_stable_ids_and_summary_counts():
    state = set_plan_items(default_plan_state(), ["Read code", "Edit code"])

    assert state == {
        "items": [
            {"id": "step-1", "text": "Read code", "status": PLAN_STATUS_IN_PROGRESS},
            {"id": "step-2", "text": "Edit code", "status": PLAN_STATUS_PENDING},
        ],
        "active_id": "step-1",
    }
    assert plan_summary(state) == {
        "total": 2,
        "pending": 1,
        "in_progress": 1,
        "completed": 0,
        "blocked": 0,
    }


def test_update_plan_item_status_completes_and_moves_active_item():
    state = set_plan_items(default_plan_state(), ["Read code", "Edit code"])

    state = update_plan_item_status(state, "step-1", PLAN_STATUS_COMPLETED)
    state = update_plan_item_status(state, "step-2", PLAN_STATUS_IN_PROGRESS)

    assert state == {
        "items": [
            {"id": "step-1", "text": "Read code", "status": PLAN_STATUS_COMPLETED},
            {"id": "step-2", "text": "Edit code", "status": PLAN_STATUS_IN_PROGRESS},
        ],
        "active_id": "step-2",
    }


def test_render_plan_text_outputs_status_lines():
    state = {
        "items": [
            {"id": "step-1", "text": "Read code", "status": PLAN_STATUS_COMPLETED},
            {"id": "step-2", "text": "Edit code", "status": PLAN_STATUS_BLOCKED},
        ],
        "active_id": "",
    }

    assert render_plan_text(state) == (
        "Plan:\n"
        "- [completed] step-1: Read code\n"
        "- [blocked] step-2: Edit code"
    )
