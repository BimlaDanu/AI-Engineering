"""Typed contracts for quiz generation (structured-output results).

These Pydantic models are the *schemas* handed to the LLM via OpenRouter structured outputs
(``response_format`` + ``json_schema`` through LangChain's ``with_structured_output``). The
model is forced to emit JSON matching the schema, which we validate into typed objects — so
:mod:`src.quiz.quiz` builds the quiz from real fields (a bounded ``correct_index``, a list of
options) and never parses free-form prose. Mirrors the discipline used by the router
(:mod:`src.core.schemas`) and the evaluation judge (:mod:`src.eval.schemas`).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class QuizItem(BaseModel):
    """One multiple-choice question grounded in the provided knowledge-base passages."""

    question: str = Field(
        description=(
            "A clear, self-contained multiple-choice question testing understanding of a "
            "concept stated in the provided passages. Do not reference 'the passage' or "
            "'the text' — the learner does not see the passages."
        )
    )
    options: list[str] = Field(
        description=(
            "Exactly four answer options. Exactly one is correct; the other three are "
            "plausible but wrong distractors. Do not prefix them with letters or numbers."
        )
    )
    correct_index: int = Field(
        ge=0,
        le=3,
        description="0-based index into ``options`` of the single correct answer.",
    )
    explanation: str = Field(
        description=(
            "One or two sentences explaining why the correct option is right, grounded in "
            "the passages — the feedback shown after the learner answers."
        )
    )
    source_index: int = Field(
        ge=1,
        description=(
            "1-based index of the context passage this question is based on, so the answer "
            "can cite its source document."
        ),
    )


class QuizItems(BaseModel):
    """A batch of generated quiz questions (the structured-output envelope)."""

    items: list[QuizItem] = Field(
        description="The generated questions, one per requested item where possible."
    )
