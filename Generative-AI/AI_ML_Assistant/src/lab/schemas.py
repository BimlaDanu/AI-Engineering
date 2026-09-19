"""Typed contract for the AI/ML Lab's LLM code-seeding (structured output).

Mirrors the discipline used across the app (router, judge): the model is forced to emit JSON
matching this schema via ``with_structured_output(..., method="json_schema")``, and we consume
a validated object — never parsed prose.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SeededSnippet(BaseModel):
    """A runnable starter snippet the model writes for the current lesson + dataset."""

    code: str = Field(
        description=(
            "A short, self-contained Python snippet (<= ~25 lines) using only numpy, pandas, "
            "matplotlib.pyplot, and scikit-learn. It must use the pre-bound data variables "
            "exactly as described in the prompt's 'Pre-bound variables' section and the "
            "handles `np`, `pd`, `plt`. `y` is already the target — never rebuild it from "
            "`df`. It must NOT read files, access the network, or import os/sys. Print results "
            "and/or draw one matplotlib figure."
        )
    )
    explanation: str = Field(
        description="One or two plain sentences explaining what the snippet demonstrates."
    )


class UnderstandingQuestion(BaseModel):
    """One short, open-ended check-your-understanding question for a conceptual lesson."""

    question: str = Field(
        description=(
            "A single open-ended question that checks whether the learner understood the "
            "lesson's core idea, pitched at the stated learner level. One or two sentences, "
            "answerable in a few sentences — not multiple-choice, not a yes/no question."
        )
    )
    reference_answer: str = Field(
        description=(
            "A concise, correct model answer to the question (2–4 sentences) used only to grade "
            "the learner's response — never shown before they answer."
        )
    )


class UnderstandingGrade(BaseModel):
    """The grade of a learner's free-text answer against the reference answer."""

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="How correct and complete the answer is, 0.0–1.0, judged at the learner level.",
    )
    verdict: str = Field(
        description="A three-word-max headline, e.g. 'Correct', 'Partially correct', 'Off track'."
    )
    feedback: str = Field(
        description="Two or three encouraging, specific sentences: what was right, what to fix."
    )
    missed: list[str] = Field(
        default_factory=list,
        description="Key points the reference answer covers that the learner's answer missed.",
    )
