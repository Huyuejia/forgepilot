"""Structured task plan state for Pico.

The plan layer is intentionally small: it tracks what the agent is doing now,
what remains, and what is blocked. It complements memory instead of replacing it.
"""

from __future__ import annotations

from copy import deepcopy


PLAN_STATUS_PENDING = "pending"
PLAN_STATUS_IN_PROGRESS = "in_progress"
PLAN_STATUS_COMPLETED = "completed"
PLAN_STATUS_BLOCKED = "blocked"
PLAN_STATUSES = {
    PLAN_STATUS_PENDING,
    PLAN_STATUS_IN_PROGRESS,
    PLAN_STATUS_COMPLETED,
    PLAN_STATUS_BLOCKED,
}
MAX_PLAN_ITEMS = 12
MAX_PLAN_TEXT_CHARS = 180


def _clean_text(value):
    text = " ".join(str(value or "").strip().split())
    return text[:MAX_PLAN_TEXT_CHARS]


def default_plan_state():
    return {
        "items": [],
        "active_id": "",
    }


def _normalize_item(raw, index, active_seen):
    if isinstance(raw, str):
        item_id = f"step-{index + 1}"
        text = _clean_text(raw)
        status = PLAN_STATUS_PENDING
    elif isinstance(raw, dict):
        item_id = _clean_text(raw.get("id")) or f"step-{index + 1}"
        text = _clean_text(raw.get("text") or raw.get("step") or raw.get("title"))
        status = str(raw.get("status", PLAN_STATUS_PENDING)).strip()
    else:
        item_id = f"step-{index + 1}"
        text = _clean_text(raw)
        status = PLAN_STATUS_PENDING

    if status not in PLAN_STATUSES:
        status = PLAN_STATUS_PENDING
    if status == PLAN_STATUS_IN_PROGRESS:
        if active_seen:
            status = PLAN_STATUS_PENDING
        active_seen = True
    return {"id": item_id, "text": text, "status": status}, active_seen


def normalize_plan_state(state):
    if not isinstance(state, dict):
        state = default_plan_state()

    items = []
    active_seen = False
    for raw in list(state.get("items") or [])[:MAX_PLAN_ITEMS]:
        item, active_seen = _normalize_item(raw, len(items), active_seen)
        if item["text"]:
            items.append(item)

    active_id = ""
    for item in items:
        if item["status"] == PLAN_STATUS_IN_PROGRESS:
            active_id = item["id"]
            break

    return {
        "items": items,
        "active_id": active_id,
    }


def set_plan_items(state, items):
    normalized_items = []
    for raw in list(items or [])[:MAX_PLAN_ITEMS]:
        if isinstance(raw, dict):
            text = _clean_text(raw.get("text") or raw.get("step") or raw.get("title"))
            status = str(raw.get("status", PLAN_STATUS_PENDING)).strip()
            if status not in PLAN_STATUSES:
                status = PLAN_STATUS_PENDING
        else:
            text = _clean_text(raw)
            status = PLAN_STATUS_IN_PROGRESS if not normalized_items else PLAN_STATUS_PENDING
        if not text:
            continue
        normalized_items.append(
            {
                "id": f"step-{len(normalized_items) + 1}",
                "text": text,
                "status": status,
            }
        )
    return normalize_plan_state({"items": normalized_items})


def update_plan_item_status(state, item_id, status):
    state = normalize_plan_state(deepcopy(state))
    item_id = str(item_id or "").strip()
    status = str(status or "").strip()
    if status not in PLAN_STATUSES:
        raise ValueError(f"invalid plan status: {status}")

    found = False
    for item in state["items"]:
        if item["id"] != item_id:
            if status == PLAN_STATUS_IN_PROGRESS and item["status"] == PLAN_STATUS_IN_PROGRESS:
                item["status"] = PLAN_STATUS_PENDING
            continue
        item["status"] = status
        found = True

    if not found:
        raise ValueError(f"unknown plan item: {item_id}")

    return normalize_plan_state(state)


def render_plan_text(state):
    state = normalize_plan_state(state)
    lines = ["Plan:"]
    if not state["items"]:
        lines.append("- none")
        return "\n".join(lines)
    for item in state["items"]:
        lines.append(f"- [{item['status']}] {item['id']}: {item['text']}")
    return "\n".join(lines)


def plan_summary(state):
    state = normalize_plan_state(state)
    summary = {status: 0 for status in sorted(PLAN_STATUSES)}
    for item in state["items"]:
        summary[item["status"]] += 1
    return {
        "total": len(state["items"]),
        PLAN_STATUS_PENDING: summary[PLAN_STATUS_PENDING],
        PLAN_STATUS_IN_PROGRESS: summary[PLAN_STATUS_IN_PROGRESS],
        PLAN_STATUS_COMPLETED: summary[PLAN_STATUS_COMPLETED],
        PLAN_STATUS_BLOCKED: summary[PLAN_STATUS_BLOCKED],
    }
