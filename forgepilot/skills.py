"""Small on-demand skill loader for ForgePilot prompts."""

from pathlib import Path


MAX_SKILL_CHARS = 1800


def _skill_root():
    return Path(__file__).parent / "builtin_skills"


def _parse_skill(path):
    text = path.read_text(encoding="utf-8")
    triggers = []
    in_triggers = False
    for raw in text.splitlines():
        line = raw.strip()
        if line == "## Triggers":
            in_triggers = True
            continue
        if line.startswith("## ") and in_triggers:
            break
        if in_triggers and line.startswith("- "):
            triggers.append(line[2:].strip().lower())
    return {
        "name": path.stem,
        "path": path.name,
        "triggers": [item for item in triggers if item],
        "text": text.strip(),
    }


def load_builtin_skills(root=None):
    root = Path(root) if root is not None else _skill_root()
    if not root.exists():
        return []
    return [_parse_skill(path) for path in sorted(root.glob("*.md"))]


def select_relevant_skills(user_message, skills=None, limit=2):
    query = str(user_message or "").lower()
    skills = list(load_builtin_skills() if skills is None else skills)
    ranked = []
    for skill in skills:
        score = sum(1 for trigger in skill["triggers"] if trigger and trigger in query)
        if score:
            ranked.append((score, skill["name"], skill))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [skill for _, _, skill in ranked[:limit]]


def render_relevant_skills(user_message, skills=None, limit=2):
    selected = select_relevant_skills(user_message, skills=skills, limit=limit)
    lines = ["Relevant skills:"]
    if not selected:
        lines.append("- none")
        return "\n".join(lines)
    for skill in selected:
        body = skill["text"]
        if len(body) > MAX_SKILL_CHARS:
            body = body[: MAX_SKILL_CHARS - 3] + "..."
        lines.append(f"## {skill['name']}")
        lines.append(body)
    return "\n".join(lines)
