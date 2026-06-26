from pico.skills import load_builtin_skills, render_relevant_skills, select_relevant_skills


def test_builtin_code_review_skill_is_selected_for_review_requests():
    skills = load_builtin_skills()

    selected = select_relevant_skills("please review this diff for security issues", skills=skills)

    assert [skill["name"] for skill in selected] == ["code-review-expert"]


def test_builtin_code_review_skill_is_selected_for_chinese_review_requests():
    skills = load_builtin_skills()

    selected = select_relevant_skills("帮我做一次代码审查，重点看安全和边界条件", skills=skills)

    assert [skill["name"] for skill in selected] == ["code-review-expert"]


def test_render_relevant_skills_is_compact():
    skills = load_builtin_skills()

    text = render_relevant_skills("code review", skills=skills)

    assert text.startswith("Relevant skills:")
    assert "code-review-expert" in text
    assert "Default to review-only" in text
