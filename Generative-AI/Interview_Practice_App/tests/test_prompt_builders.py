"""Unit tests for the pure prompt builders and registries in categories.py."""

import pytest

from categories import (
    CATEGORIES,
    QUESTIONS_PER_SET,
    ROLE_CONTEXT,
    ROLES,
    TECHNIQUES,
    build_evaluation_prompt,
    build_generation_prompt,
    get_category,
)

# ---------------------------------------------------------------------------
# get_category
# ---------------------------------------------------------------------------


def test_get_category_known():
    cat = get_category("Python Coding")
    assert cat["focus"]


def test_get_category_unknown_raises_helpful_keyerror():
    with pytest.raises(KeyError) as excinfo:
        get_category("Basket Weaving")
    assert "Available" in str(excinfo.value)


# ---------------------------------------------------------------------------
# build_generation_prompt
# ---------------------------------------------------------------------------


def test_generation_prompt_contains_parameters():
    prompt = build_generation_prompt("Python Coding", "Medium", "Mid")
    assert f"exactly {QUESTIONS_PER_SET}" in prompt
    assert "Medium-level" in prompt
    assert "Mid candidate" in prompt
    assert '"questions"' in prompt  # matches the structured-output schema


def test_generation_prompt_includes_examples_and_rules():
    prompt = build_generation_prompt("Behavioural", "Easy", "Junior")
    cat = CATEGORIES["Behavioural"]
    assert cat["example_questions"][0] in prompt
    assert "STAR" in prompt


def test_generation_prompt_without_cv_has_no_cv_block():
    prompt = build_generation_prompt("Data Science", "Hard", "Senior")
    assert "Candidate background" not in prompt


def test_generation_prompt_with_cv_embeds_summary_in_user_prompt():
    prompt = build_generation_prompt(
        "Data Science", "Hard", "Senior", cv_summary="10 years of SQL at Acme."
    )
    assert "Candidate background" in prompt
    assert "10 years of SQL at Acme." in prompt


def test_generation_prompt_custom_n():
    prompt = build_generation_prompt("Machine Learning", "Easy", "Junior", n=5)
    assert "exactly 5" in prompt


def test_generation_prompt_without_job_description_has_no_jd_block():
    prompt = build_generation_prompt("Data Science", "Hard", "Senior")
    assert "Target job description" not in prompt


def test_generation_prompt_with_job_description_embeds_it():
    prompt = build_generation_prompt(
        "Data Science",
        "Hard",
        "Senior",
        job_description="Senior analyst role focused on A/B testing at Acme.",
    )
    assert "Target job description" in prompt
    assert "A/B testing at Acme." in prompt


# ---------------------------------------------------------------------------
# build_evaluation_prompt
# ---------------------------------------------------------------------------


def test_evaluation_prompt_contains_question_answer_and_rubric():
    prompt = build_evaluation_prompt(
        "AI Engineering", "What is RAG?", "RAG retrieves documents first."
    )
    assert "What is RAG?" in prompt
    assert "RAG retrieves documents first." in prompt
    assert CATEGORIES["AI Engineering"]["evaluation_focus"] in prompt


def test_evaluation_prompt_unknown_category_raises():
    with pytest.raises(KeyError):
        build_evaluation_prompt("Nope", "q", "a")


# ---------------------------------------------------------------------------
# Registry invariants (state metadata the UI relies on)
# ---------------------------------------------------------------------------


def test_all_categories_have_required_fields():
    required = {
        "description",
        "focus",
        "topics",
        "extra_rules",
        "example_questions",
        "evaluation_focus",
    }
    for name, cat in CATEGORIES.items():
        missing = required - set(cat)
        assert not missing, f"{name} missing fields: {missing}"
        assert len(cat["example_questions"]) >= 3


def test_all_techniques_have_system_and_examples():
    for name, spec in TECHNIQUES.items():
        assert spec["system"].strip(), f"{name} has an empty system prompt"
        assert isinstance(spec["examples"], list)
        for example in spec["examples"]:
            assert set(example) == {"user", "assistant"}


def test_few_shot_technique_has_message_pair_examples():
    assert len(TECHNIQUES["Few-shot"]["examples"]) >= 2


def test_roles_match_role_context_keys():
    # The sidebar options and the role-context keys can never drift apart.
    assert ROLES == list(ROLE_CONTEXT)
    for role, levels in ROLE_CONTEXT.items():
        assert set(levels) == {"Junior", "Mid", "Senior"}, role
