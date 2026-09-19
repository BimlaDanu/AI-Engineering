"""The four RAGAs metrics, implemented over the :class:`~src.eval.judge.Judge` protocol.

Each metric is a pure function of a judge plus the relevant text: it calls one or two judge
primitives and turns the typed verdicts into a bounded score. No metric talks to an LLM
directly, so every one is unit-testable with a deterministic stub judge.

Score semantics (all in ``[0, 1]``, higher is better), following RAGAs:

* **faithfulness** — fraction of the *answer's* claims that the retrieved context supports
  (measures hallucination: 1.0 = every claim grounded).
* **answer relevancy** — how directly the answer addresses the question (judged 0..1).
* **context precision** — rank-weighted average precision of the retrieved passages
  (relevant passages ranked first score higher).
* **context recall** — fraction of the *reference answer's* claims that the retrieved
  context supports (did retrieval bring back what was needed).

A metric returns ``score=None`` when it is genuinely undefined for the input (e.g.
faithfulness of an answer that asserts no checkable claim); aggregation skips those.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from src.eval.judge import Judge
from src.eval.schemas import ContextVerdict


@dataclass
class MetricResult:
    """One metric's outcome for one sample: a score in ``[0, 1]`` (or None) plus detail."""

    name: str
    score: float | None
    detail: str


def _support_ratio(judge: Judge, claims: list[str], contexts: list[str]) -> tuple[int, int]:
    """Return ``(supported, total)`` for ``claims`` judged against ``contexts``.

    The denominator is the number of *claims asked about*, **not** the number of verdicts
    returned. A judge that returns fewer verdicts than claims (a truncated or malformed
    structured response) would otherwise inflate the ratio by silently dropping the unjudged
    claims from the denominator — so a hallucinated claim the judge failed to return a verdict
    for would raise faithfulness instead of lowering it. Counting missing verdicts as
    unsupported keeps faithfulness/recall conservative; any surplus verdicts beyond the claim
    count are ignored.
    """
    verdicts = judge.verify_claims(claims, contexts)
    supported = sum(1 for v in verdicts[: len(claims)] if v.supported)
    return supported, len(claims)


def faithfulness(judge: Judge, answer: str, contexts: list[str]) -> MetricResult:
    """Fraction of the answer's atomic claims that are grounded in the context."""
    claims = judge.extract_claims(answer)
    if not claims:
        return MetricResult("faithfulness", None, "Answer makes no verifiable claim.")
    if not contexts:
        return MetricResult("faithfulness", 0.0, f"No context to ground {len(claims)} claim(s).")
    supported, total = _support_ratio(judge, claims, contexts)
    score = supported / total if total else None
    return MetricResult("faithfulness", score, f"{supported}/{total} claims grounded in context.")


def answer_relevancy(judge: Judge, question: str, answer: str) -> MetricResult:
    """How directly the answer addresses the question (judged 0..1)."""
    if not answer.strip():
        return MetricResult("answer_relevancy", 0.0, "Empty answer.")
    verdict = judge.score_relevancy(question, answer)
    return MetricResult("answer_relevancy", verdict.score, verdict.reason)


def _relevance_flags(verdicts: list[ContextVerdict], n: int) -> list[bool]:
    """Project relevance verdicts onto the ``n`` retrieved passages, by retrieval rank.

    Each verdict is placed at its declared 1-based ``index`` so the rank-weighting below
    reflects the true *retrieval* order even if the judge returns verdicts out of order or
    with gaps; a passage with no in-range verdict counts as not relevant (missing ≠ relevant).
    If no verdict carries a usable index (a judge that ignores the field), fall back to
    positional order so the metric still scores.
    """
    flags = [False] * n
    placed = False
    for v in verdicts:
        i = v.index - 1
        if 0 <= i < n:
            flags[i] = v.relevant
            placed = True
    if placed:
        return flags
    for i, v in enumerate(verdicts[:n]):
        flags[i] = v.relevant
    return flags


def context_precision(judge: Judge, question: str, contexts: list[str]) -> MetricResult:
    """Rank-weighted average precision of the retrieved passages.

    Uses the standard average-precision formulation: passages judged relevant that appear
    earlier in the retrieval order contribute more, rewarding good ranking rather than only
    the relevant/total ratio.
    """
    if not contexts:
        return MetricResult("context_precision", None, "No passages retrieved.")
    verdicts = judge.rate_contexts(question, contexts)
    flags = _relevance_flags(verdicts, len(contexts))
    total_relevant = sum(flags)
    if total_relevant == 0:
        return MetricResult("context_precision", 0.0, "No retrieved passage was relevant.")
    hits = 0
    precision_sum = 0.0
    for rank, relevant in enumerate(flags, start=1):
        if relevant:
            hits += 1
            precision_sum += hits / rank
    score = precision_sum / total_relevant
    return MetricResult(
        "context_precision",
        score,
        f"{total_relevant}/{len(flags)} passages relevant (rank-weighted).",
    )


def context_recall(judge: Judge, reference: str, contexts: list[str]) -> MetricResult:
    """Fraction of the reference answer's claims that the context supports."""
    if not reference.strip():
        return MetricResult("context_recall", None, "No reference answer provided.")
    claims = judge.extract_claims(reference)
    if not claims:
        return MetricResult("context_recall", None, "Reference makes no verifiable claim.")
    if not contexts:
        return MetricResult(
            "context_recall", 0.0, f"No context to cover {len(claims)} reference claim(s)."
        )
    supported, total = _support_ratio(judge, claims, contexts)
    score = supported / total if total else None
    return MetricResult("context_recall", score, f"{supported}/{total} reference claims covered.")


# Ordered registry of every metric and how it is called. ``needs_reference`` flags the
# metrics that require a golden reference answer (only context_recall does). The runner
# iterates this so adding a metric is a one-line change here.
METRICS: list[dict] = [
    {"name": "faithfulness", "needs_reference": False},
    {"name": "answer_relevancy", "needs_reference": False},
    {"name": "context_precision", "needs_reference": False},
    {"name": "context_recall", "needs_reference": True},
]


def evaluate_sample(
    judge: Judge,
    question: str,
    answer: str,
    contexts: list[str],
    reference: str | None,
    *,
    parallel: bool = False,
) -> list[MetricResult]:
    """Run all four metrics for one (question, answer, contexts, reference) sample.

    The metrics are mutually independent — each one calls the judge on its own — so with
    ``parallel=True`` they run concurrently in a small thread pool, cutting the per-sample
    judge critical path from ~6 serial round-trips to ~2 (the two two-call metrics run
    side by side). The returned list is always in :data:`METRICS` order regardless of
    completion order, so the result is identical to the sequential path. ``parallel=False``
    (the default) preserves the original strictly-sequential behaviour.

    Each metric is isolated: if its judge call ultimately fails (e.g. a structured-output
    response that stays malformed after retries), that one metric degrades to an undefined
    (``None``) score with the error in its detail, while the other metrics for the sample
    still score. One bad judge response can never sink the whole evaluation run.
    """
    specs: list[tuple[str, Callable[[], MetricResult]]] = [
        ("faithfulness", lambda: faithfulness(judge, answer, contexts)),
        ("answer_relevancy", lambda: answer_relevancy(judge, question, answer)),
        ("context_precision", lambda: context_precision(judge, question, contexts)),
    ]
    if reference is not None:
        specs.append(("context_recall", lambda: context_recall(judge, reference, contexts)))

    def run(spec: tuple[str, Callable[[], MetricResult]]) -> MetricResult:
        name, fn = spec
        try:
            return fn()
        except Exception as exc:  # one metric's judge failure must not sink the sample
            return MetricResult(name, None, f"Judge error: {exc}")

    if not parallel or len(specs) == 1:
        return [run(spec) for spec in specs]

    with ThreadPoolExecutor(max_workers=len(specs)) as pool:
        futures = [pool.submit(run, spec) for spec in specs]
        # Iterating the futures in submission order keeps results in METRICS order.
        return [future.result() for future in futures]
