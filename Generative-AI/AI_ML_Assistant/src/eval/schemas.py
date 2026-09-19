"""Typed contracts for the evaluation judge (structured-output results).

These Pydantic models are the *schemas* handed to the judge LLM via OpenRouter structured
outputs (``response_format`` + ``json_schema`` through LangChain's
``with_structured_output``). The model is forced to emit JSON matching the schema, which we
validate into one of these objects — so the metric maths in :mod:`src.eval.metrics` runs on
typed booleans and bounded floats, never on parsed prose. Mirrors the same discipline used
by the router/injection classifier in :mod:`src.core.schemas`.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClaimList(BaseModel):
    """The atomic factual claims decomposed from a passage of text.

    Used to break an answer (for faithfulness) or a reference answer (for context recall)
    into individually checkable statements before judging support.
    """

    claims: list[str] = Field(
        description=(
            "The distinct, self-contained factual claims made by the text, each phrased so "
            "it can be verified on its own. Omit greetings, questions, and filler. Return "
            "an empty list if the text makes no verifiable factual claim."
        )
    )


class ClaimVerdict(BaseModel):
    """Whether one claim is supported by (attributable to) the provided context."""

    claim: str = Field(description="The claim being judged, echoed back verbatim.")
    supported: bool = Field(
        description=(
            "True if the claim can be directly inferred from the provided context "
            "passages; False if the context does not support it (or contradicts it)."
        )
    )
    reason: str = Field(description="One short sentence justifying the verdict.")


class ClaimVerdicts(BaseModel):
    """A verdict for every claim checked against the context, in the input order."""

    verdicts: list[ClaimVerdict] = Field(
        description="One verdict per input claim, in the same order as the claims given."
    )


class RelevancyScore(BaseModel):
    """How well an answer actually addresses the question that was asked."""

    score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Answer relevancy from 0 to 1: 1.0 = the answer directly and completely "
            "addresses the question with no off-topic or evasive content; 0.0 = it ignores "
            "the question. Penalise incompleteness, hedging, and padding."
        ),
    )
    reason: str = Field(description="One short sentence explaining the score.")


class ContextVerdict(BaseModel):
    """Whether one retrieved context passage is relevant to the question."""

    index: int = Field(description="The 1-based position of the passage in the input list.")
    relevant: bool = Field(
        description=(
            "True if this passage contains information useful for answering the question; "
            "False if it is off-topic or unhelpful padding."
        )
    )
    reason: str = Field(description="One short sentence justifying the verdict.")


class ContextVerdicts(BaseModel):
    """A relevance verdict for every retrieved passage, in retrieval order."""

    verdicts: list[ContextVerdict] = Field(
        description="One verdict per passage, in the same order as the passages given."
    )
