"""Offline tests for the AI & LLM Lab exercises (:mod:`src.lab.llm_exercises`).

Both exercises take their network-touching collaborators as injected arguments — a ``generate``
callable and a :class:`~src.eval.judge.Judge` — so the whole module runs deterministically here
with a fake generator and a stub judge, exactly as :mod:`src.eval` tests do. No API key, no
network, no optional data dependencies.
"""

from __future__ import annotations

from src.eval.schemas import ClaimVerdict, ContextVerdict, RelevancyScore
from src.lab.llm_exercises import (
    LLM_EXERCISES,
    compare_prompt_techniques,
    evaluate_answer,
)


class _StubJudge:
    """A deterministic :class:`~src.eval.judge.Judge`: every claim/passage is supported."""

    def extract_claims(self, text: str) -> list[str]:
        return [text.strip()] if text.strip() else []

    def verify_claims(self, claims: list[str], contexts: list[str]) -> list[ClaimVerdict]:
        return [ClaimVerdict(claim=c, supported=True, reason="stub") for c in claims]

    def score_relevancy(self, question: str, answer: str) -> RelevancyScore:
        return RelevancyScore(score=0.9, reason="stub relevancy")

    def rate_contexts(self, question: str, contexts: list[str]) -> list[ContextVerdict]:
        return [
            ContextVerdict(index=i, relevant=True, reason="stub")
            for i, _ in enumerate(contexts, start=1)
        ]


def _fake_generate(system: str, human: str) -> str:
    # Echo the question so the answer is non-empty and deterministic; the system prompt (which
    # carries the technique instruction) is ignored — relevancy comes from the stub judge.
    return f"A grounded answer to: {human}"


# ------------------------------------------------------------------------------- registry
def test_registry_exposes_the_two_exercises():
    assert set(LLM_EXERCISES) == {"prompt_compare", "llm_eval"}
    assert all(ex.key == key for key, ex in LLM_EXERCISES.items())


# ------------------------------------------------------------------------- prompt_compare
def test_compare_prompt_techniques_scores_each_technique():
    techniques = ["Standard", "Chain-of-Thought", "Analogy-first"]
    result = compare_prompt_techniques(
        "Why does dropout reduce overfitting?",
        techniques,
        level="Beginner",
        model_id="openai/gpt-4o-mini",
        generate=_fake_generate,
        judge=_StubJudge(),
    )
    assert result.error is None
    assert [r["Technique"] for r in result.rows] == techniques
    for row in result.rows:
        assert 0.0 <= row["Relevancy"] <= 1.0
        assert isinstance(row["Est. cost"], float) and row["Est. cost"] >= 0.0
        assert row["Answer"].startswith("A grounded answer")
    assert "technique" in result.summary.lower()


def test_compare_prompt_techniques_rejects_empty_inputs():
    stub = _StubJudge()
    assert compare_prompt_techniques(
        "",
        ["Standard"],
        level="Beginner",
        model_id="m",
        generate=_fake_generate,
        judge=stub,
    ).error
    assert compare_prompt_techniques(
        "A real question?",
        [],
        level="Beginner",
        model_id="m",
        generate=_fake_generate,
        judge=stub,
    ).error


def test_compare_prompt_techniques_does_not_generate_when_question_is_empty():
    calls: list[tuple[str, str]] = []

    def spy(system: str, human: str) -> str:
        calls.append((system, human))
        return "x"

    compare_prompt_techniques(
        "   ", ["Standard"], level="Beginner", model_id="m", generate=spy, judge=_StubJudge()
    )
    assert calls == []  # guarded before any (costly) model call


# ------------------------------------------------------------------------------- llm_eval
def test_evaluate_answer_with_context_and_reference_scores_all_metrics():
    result = evaluate_answer(
        "What is the bias-variance tradeoff?",
        "It is the balance between underfitting and overfitting.",
        contexts=["Bias is error from wrong assumptions; variance is sensitivity to data."],
        reference="The tradeoff balances bias and variance to minimise total error.",
        judge=_StubJudge(),
    )
    assert result.error is None
    names = {r["Metric"] for r in result.rows}
    assert names == {"faithfulness", "answer relevancy", "context precision", "context recall"}
    # Every metric is defined (stub supports everything), so none is the "—" placeholder.
    assert all(r["Score"] != "—" for r in result.rows)


def test_evaluate_answer_without_context_or_reference_skips_recall():
    result = evaluate_answer(
        "What is gradient descent?",
        "An optimisation method that follows the negative gradient.",
        contexts=[],
        reference=None,
        judge=_StubJudge(),
    )
    assert result.error is None
    names = [r["Metric"] for r in result.rows]
    assert "context recall" not in names  # needs a reference answer
    assert "answer relevancy" in names


def test_evaluate_answer_rejects_empty_answer():
    assert evaluate_answer("Q?", "   ", contexts=[], reference=None, judge=_StubJudge()).error


# ----------------------------------------------------------------------------- curriculum
def test_curriculum_llm_practices_reference_real_exercises():
    from src.lab.curriculum import TRACKS

    llm_practices = [
        lesson.practice
        for track in TRACKS
        for lesson in track.lessons
        if lesson.practice is not None and lesson.practice.topic == "AI & LLM"
    ]
    assert llm_practices, "expected at least one AI & LLM practice lesson"
    for practice in llm_practices:
        assert practice.exercise_key in LLM_EXERCISES
        assert not practice.recipe_key and not practice.dataset_key
