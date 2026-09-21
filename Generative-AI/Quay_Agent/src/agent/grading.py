"""Model-backed judgement on retrieved passages, spent only where it pays.

The retrieval layer takes a grader and a rewriter as arguments and runs
deterministic ones when neither is supplied. This module supplies the
model-backed versions, and the interesting part is when they decline to run.

Both start by computing the deterministic answer. If the cheap answer is
confident they return ``None`` and no call is made, so the question is answered
at the speed of a vector search. A model is reached for in one situation: the
round kept nothing, which is where the alternative is telling a person the corpus
does not cover their question.

Grading every round with a model would also catch passages the heuristic wrongly
kept, and it is the wrong trade -- a network round trip in front of every
successful answer, to improve the precision of answers that were already
grounded.

A retrieved passage is text this project did not write, about to be shown to a
model. It is neutralised and fenced before it enters a prompt, and the grader is
asked for identifiers rather than prose, so the worst a hostile passage can
achieve is to nominate itself -- which is what an irrelevant passage does anyway,
and what the identifier check catches.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from pydantic import BaseModel, Field

from src.agent.llm import as_data
from src.agent.model_selection import ModelPool
from src.agent.prompts import (
    PASSAGE_GRADING,
    PASSAGE_ORDERING,
    QUERY_EXPANSION,
    QUERY_REWRITE,
)
from src.logging_setup import get_logger
from src.rag.retrieve import Grade, Passage, heuristic_grade

_logger = get_logger("agent.grading")

SNIPPET_CHARACTERS = 600
"""How much of a passage the grader is shown.

Enough to judge relevance and not enough to pay for the whole passage four times
over. Relevance is decided in the first paragraph -- if it is not, the chunking
put two ideas in one chunk and that is a corpus problem rather than a grading
one.
"""

MAX_GRADED = 8
"""How many candidates one grading call will consider.

A bound on the prompt rather than on the search. Beyond this the call costs more
than the answer is worth, and a round offering more than eight candidates has
already failed to rank them.
"""

MIN_TO_REORDER = 2
"""Below this many passages, reranking is skipped without a call.

One passage has one ordering and two have two, one of which the fused ranking
already chose on the evidence of both halves of the search agreeing. Paying a
network round trip to consider swapping two items is the kind of call that makes an
application feel slow for no measurable gain.
"""


class Relevance(BaseModel):
    """Which of the offered passages actually address the question.

    Identifiers rather than text, so that the grader selects from what it was
    given instead of producing something new. A passage it invents is dropped by
    the caller, which makes the worst case a wasted call rather than a fabricated
    citation.
    """

    keep: list[str] = Field(
        default_factory=list,
        description=(
            "Identifiers of the passages that help answer the question, best "
            "first. Empty if none of them do."
        ),
    )
    reason: str = Field(description="One short sentence on why those were kept, or why none were.")
    better_query: str = Field(
        default="",
        description=(
            "A search query more likely to find an answer, borrowing the wording "
            "the passages themselves use. Empty if there is nothing better to try."
        ),
    )


class Reformulation(BaseModel):
    """A second attempt at a query that found nothing useful."""

    query: str = Field(
        description=(
            "A better search query, in the vocabulary the source material uses. "
            "Empty if the subject is genuinely absent and searching again is futile."
        )
    )


def _offer(candidates: Sequence[Passage]) -> str:
    """Render candidates for a grading prompt, fenced as untrusted material.

    Args:
        candidates: The round's candidates.

    Returns:
        One block per passage, each labelled with the identifier the grader must
        answer with.
    """
    return "\n\n".join(
        f"[{passage.identifier}] {passage.title}\n{as_data(passage.text[:SNIPPET_CHARACTERS])}"
        for passage in candidates[:MAX_GRADED]
    )


def model_grader(pool: ModelPool) -> Callable[[str, tuple[Passage, ...]], Grade | None]:
    """Build a grader that escalates to a model only when the cheap rules fail.

    Args:
        pool: The campaign's models. The call is made at the fast tier, because
            it sits between a person and their answer.

    Returns:
        A grader. It returns ``None`` -- meaning "use the deterministic rules" --
        whenever those rules kept something, which is the ordinary case and costs
        nothing.
    """

    def grade(question: str, candidates: tuple[Passage, ...]) -> Grade | None:
        cheap = heuristic_grade(question, candidates)
        if cheap.keep or not candidates:
            return None
        answer = pool.invoke(
            "passage_grading",
            Relevance,
            PASSAGE_GRADING,
            f"Question: {question}\n\nPassages:\n{_offer(candidates)}",
        )
        if answer is None:
            return None
        offered = {passage.identifier for passage in candidates}
        kept = tuple(name for name in answer.keep if name in offered)
        _logger.info(
            "grading_escalated",
            extra={"offered": len(candidates), "kept": len(kept), "rescued": bool(kept)},
        )
        return Grade(
            keep=kept,
            reason=answer.reason,
            query=answer.better_query,
            graded_by="model",
        )

    return grade


def model_rewriter(pool: ModelPool) -> Callable[[str, str, tuple[Passage, ...]], str]:
    """Build a rewriter that asks a model for a better query.

    Called by the retrieval layer only when a round kept nothing *and* the grader
    proposed no reformulation of its own, so it never duplicates a judgement that
    has already been paid for.

    Args:
        pool: The campaign's models.

    Returns:
        A rewriter, which answers with the deterministic reformulation whenever a
        model is unavailable or declines to improve on it.
    """

    def reformulate(question: str, failed: str, candidates: tuple[Passage, ...]) -> str:
        seen = "\n".join(f"- {passage.title}" for passage in candidates[:MAX_GRADED])
        answer = pool.invoke(
            "query_rewrite",
            Reformulation,
            QUERY_REWRITE,
            (
                f"Question: {question}\n"
                f"Query that failed: {failed}\n"
                f"Titles it returned:\n{seen or '- nothing'}"
            ),
        )
        proposed = (answer.query if answer is not None else "").strip()
        if not proposed or proposed.lower() == failed.lower():
            return ""
        _logger.info("query_rewritten", extra={"detail": "model proposed a reformulation"})
        return proposed

    return reformulate


class Phrasings(BaseModel):
    """Other ways of searching for what a question is asking about.

    Queries rather than questions, and the field description says so, because the
    thing being bridged is a vocabulary gap: the corpus is written in the language
    of papers and the question is written in the language of whoever is asking.
    """

    queries: list[str] = Field(
        default_factory=list,
        description=(
            "Short search queries -- keywords and technical noun phrases -- that "
            "would find the same answer worded differently. No questions, no full "
            "sentences. Empty if the question is already in the source material's "
            "own vocabulary and there is nothing to bridge."
        ),
    )


class Ordering(BaseModel):
    """Which of the relevant passages the answer should be written from first.

    A permutation, not a selection. Every passage offered here has already been
    graded relevant, and this call is not allowed to change that -- the caller
    appends anything the model failed to mention rather than dropping it.
    """

    order: list[str] = Field(
        default_factory=list,
        description=(
            "Identifiers of the passages, exactly as given, most directly useful "
            "first. Every identifier offered should appear exactly once."
        ),
    )


def model_expander(pool: ModelPool) -> Callable[[str, Sequence[str]], tuple[str, ...]]:
    """Build an expander that proposes other vocabularies for one question.

    Unlike :func:`model_grader` and :func:`model_rewriter` this one does not
    escalate. It runs before the first search, on every question that reaches
    retrieval, because a vocabulary gap cannot be detected without searching and by
    the time a search has come back empty the round it would have saved is spent.

    It stays affordable through its tier and its shape: a fast-tier call for three
    short strings, made once per question, feeding searches that are local.

    Args:
        pool: The campaign's models.

    Returns:
        An expander. Answers with an empty tuple when no model is available, when the
        model declines, or when everything it proposed merely repeats the question,
        which leaves :func:`src.rag.retrieve.retrieve` searching the question alone.
    """

    def expand(question: str, topics: Sequence[str]) -> tuple[str, ...]:
        hint = ", ".join(topic.replace("-", " ") for topic in topics if topic)
        answer = pool.invoke(
            "query_expansion",
            Phrasings,
            QUERY_EXPANSION,
            f"Question: {question}" + (f"\nTopics it was routed to: {hint}" if hint else ""),
        )
        if answer is None:
            return ()
        asked = question.strip().lower()
        proposed = tuple(
            dict.fromkeys(
                query.strip()
                for query in answer.queries
                # A phrasing identical to the question is not a phrasing. It would
                # be searched anyway, and fusing a list with itself doubles every
                # score in it -- which changes no ordering and costs a search.
                if query.strip() and query.strip().lower() != asked
            )
        )
        _logger.info(
            "query_expansion",
            extra={"proposed": len(proposed), "detail": "phrasings to fuse with the original"},
        )
        return proposed

    return expand


def model_reranker(pool: ModelPool) -> Callable[[str, tuple[Passage, ...]], tuple[str, ...]]:
    """Build a reranker that decides which relevant passage is read first.

    Called once, on everything that survived grading across every round, before the
    list is cut to length. That ordering is deliberate and it is the only reason
    this is worth a call: reranking after the cut can only shuffle the passages
    that were already going to be used, while reranking before it can promote the
    fourth-ranked passage that actually answers the question.

    It escalates on size rather than on failure. One or two passages have no
    interesting orderings, so the call is skipped; beyond that a fused ranking is
    ordered by *agreement between two searches*, which is a good prior about
    relevance and says nothing about which passage states the answer outright.

    Args:
        pool: The campaign's models. Fast tier: this is a permutation of at most
            eight short identifiers, with a person waiting for it.

    Returns:
        A reranker. Answers with an empty tuple whenever a model is unavailable or
        declines, which leaves the fused order -- a real ranking, not an absence
        of one. The caller drops invented identifiers and appends anything the
        model omitted, so the worst this can do is reorder.
    """

    def reorder(question: str, passages: tuple[Passage, ...]) -> tuple[str, ...]:
        if len(passages) <= MIN_TO_REORDER:
            return ()
        answer = pool.invoke(
            "passage_ordering",
            Ordering,
            PASSAGE_ORDERING,
            f"Question: {question}\n\nPassages:\n{_offer(passages)}",
        )
        if answer is None:
            return ()
        offered = {passage.identifier for passage in passages}
        named = tuple(name for name in answer.order if name in offered)
        _logger.info(
            "passage_ordering",
            extra={"offered": len(passages), "named": len(named)},
        )
        return named

    return reorder
