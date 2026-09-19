"""The evaluation judge: an LLM-as-judge over OpenRouter structured outputs.

The metric functions in :mod:`src.eval.metrics` depend on the :class:`Judge` *protocol*,
not a concrete class. In production that protocol is satisfied by :class:`LLMJudge`, which
forces the model to emit JSON matching a Pydantic schema (``with_structured_output`` with
``method="json_schema"``) and returns the validated object — the metrics never see prose.
In tests it is satisfied by a tiny deterministic stub, so the whole harness runs offline.

Every judge call is a *single* structured-output round-trip; the metrics compose these
primitives (extract claims, verify claims, score relevancy, rate contexts) into the four
RAGAs metrics.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from src.eval.schemas import (
    ClaimList,
    ClaimVerdict,
    ClaimVerdicts,
    ContextVerdict,
    ContextVerdicts,
    RelevancyScore,
)
from src.utils import call_with_retry


@runtime_checkable
class Judge(Protocol):
    """The judgement primitives the metrics need — the seam tests stub out.

    Any object providing these four methods can drive the metrics. Keeping this a
    ``Protocol`` (structural typing) means the offline test stub needs no base class and no
    import cycle, while the metrics stay fully type-annotated against a real interface.
    """

    def extract_claims(self, text: str) -> list[str]:
        """Decompose ``text`` into atomic, individually verifiable factual claims."""
        ...

    def verify_claims(self, claims: list[str], contexts: list[str]) -> list[ClaimVerdict]:
        """Judge, for each claim, whether the context passages support it."""
        ...

    def score_relevancy(self, question: str, answer: str) -> RelevancyScore:
        """Score how well ``answer`` addresses ``question`` (0..1)."""
        ...

    def rate_contexts(self, question: str, contexts: list[str]) -> list[ContextVerdict]:
        """Judge, for each retrieved passage, whether it is relevant to ``question``."""
        ...


_CLAIM_SYSTEM = (
    "You decompose text into atomic factual claims for evaluation. Return each distinct, "
    "self-contained factual statement the text asserts, phrased so it can be checked on its "
    "own. Ignore questions, greetings, hedging, and 'Try next:' style follow-ups. If the "
    "text asserts no verifiable fact, return an empty list."
)
_VERIFY_SYSTEM = (
    "You judge whether each claim is supported by the given context passages. A claim is "
    "supported only if it can be directly inferred from the passages — not from your own "
    "outside knowledge. Return one verdict per claim, in the same order as the claims."
)
_RELEVANCY_SYSTEM = (
    "You score how well an answer addresses the user's question, independent of whether the "
    "answer is factually correct. 1.0 = fully on-point and complete; lower for incomplete, "
    "evasive, padded, or off-topic answers. Judge relevance to the question only."
)
_CONTEXT_SYSTEM = (
    "You judge whether each retrieved passage is relevant to answering the question. A "
    "passage is relevant if it contains information useful for the answer; it is irrelevant "
    "if it is off-topic or unhelpful. Return one verdict per passage, in input order."
)


# Substrings that mark a *parse* failure of a structured-output completion — the model's
# JSON was truncated or malformed (e.g. "Invalid JSON: EOF while parsing" from a response cut
# off mid-string). These are stochastic, not deterministic bugs, so a fresh completion usually
# parses cleanly — hence retry-worthy. Matched case-insensitively against ``str(exc)``.
_PARSE_FAILURE_MARKERS: tuple[str, ...] = (
    "invalid json",
    "json_invalid",
    "eof while parsing",
    "validation error",
    "expecting value",
    "unterminated string",
    "failed to parse",
)


def _is_parse_failure(exc: Exception) -> bool:
    """True if ``exc`` is a truncated/malformed structured-output response worth re-rolling."""
    message = str(exc).lower()
    return any(marker in message for marker in _PARSE_FAILURE_MARKERS)


def _format_contexts(contexts: list[str], limit: int = 1500) -> str:
    """Render context passages as a compact numbered block for a judge prompt."""
    if not contexts:
        return "(no context passages)"
    return "\n\n".join(f"[{i}] {c[:limit]}" for i, c in enumerate(contexts, start=1))


def _format_claims(claims: list[str]) -> str:
    """Render a claim list as a compact numbered block for a judge prompt."""
    return "\n".join(f"{i}. {c}" for i, c in enumerate(claims, start=1))


class LLMJudge:
    """A :class:`Judge` backed by an OpenRouter chat model and structured outputs.

    Each method performs exactly one ``with_structured_output(..., method="json_schema")``
    call and returns the validated Pydantic result. Construct it with a low-temperature
    model for stable, reproducible judgements.
    """

    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm

    def _structured(self, schema: type[BaseModel], system: str, human: str) -> BaseModel:
        """Force one structured-output completion and return the validated object.

        Each attempt is wrapped in :func:`~src.utils.call_with_retry` so a transient
        rate-limit or overload (likely when many judge calls fan out concurrently) backs off
        and retries. ``retry_on=_is_parse_failure`` additionally re-rolls a truncated/malformed
        JSON response — a stochastic failure a fresh completion usually fixes.

        For cross-provider robustness the call first asks for a ``json_schema`` response
        format (best for OpenAI-family models); if that keeps failing to parse, it falls back
        to **tool-calling** structured output, which providers that don't honour the
        json_schema response format — notably Anthropic models via OpenRouter — do support.
        Only then does a failure propagate (where the metric layer degrades it to ``None``),
        so picking a non-OpenAI judge model no longer silently yields all-empty scores.
        """
        messages = [("system", system), ("human", human)]

        def invoke(method: str) -> BaseModel:
            model = self._llm.with_structured_output(schema, method=method)
            return call_with_retry(lambda: model.invoke(messages), retry_on=_is_parse_failure)

        try:
            return invoke("json_schema")
        except Exception:
            return invoke("function_calling")

    def extract_claims(self, text: str) -> list[str]:
        """Decompose ``text`` into atomic claims (empty list if it asserts no facts)."""
        if not text.strip():
            return []
        result = self._structured(ClaimList, _CLAIM_SYSTEM, f"Text:\n{text}")
        return list(result.claims) if isinstance(result, ClaimList) else []

    def verify_claims(self, claims: list[str], contexts: list[str]) -> list[ClaimVerdict]:
        """Judge support of each claim against the context passages, in order."""
        if not claims:
            return []
        human = (
            f"Context passages:\n{_format_contexts(contexts)}\n\n"
            f"Claims to check:\n{_format_claims(claims)}"
        )
        result = self._structured(ClaimVerdicts, _VERIFY_SYSTEM, human)
        return list(result.verdicts) if isinstance(result, ClaimVerdicts) else []

    def score_relevancy(self, question: str, answer: str) -> RelevancyScore:
        """Score how directly ``answer`` addresses ``question`` (0..1)."""
        human = f"Question:\n{question}\n\nAnswer:\n{answer}"
        result = self._structured(RelevancyScore, _RELEVANCY_SYSTEM, human)
        if isinstance(result, RelevancyScore):
            return result
        return RelevancyScore(score=0.0, reason="Judge returned no score.")

    def rate_contexts(self, question: str, contexts: list[str]) -> list[ContextVerdict]:
        """Judge relevance of each retrieved passage to ``question``, in order."""
        if not contexts:
            return []
        human = f"Question:\n{question}\n\nRetrieved passages:\n{_format_contexts(contexts)}"
        result = self._structured(ContextVerdicts, _CONTEXT_SYSTEM, human)
        return list(result.verdicts) if isinstance(result, ContextVerdicts) else []
