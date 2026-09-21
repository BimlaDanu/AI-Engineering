"""The model-backed query rewriter for retrieval.

The other half of agentic RAG. :mod:`src.agent.grading` decides whether what came
back is worth reading; this decides what to search for when it was not. It lives
here for the same reason: it calls a model, and :mod:`src.rag` is not allowed to.
:data:`~src.rag.retrieve.Rewriter` is the seam and this module fills it, so the
retriever stays testable with no credential.

Why a second call rather than leaning on the grader's ``next_query``: the grader
proposes a reformulation only when it has one, and the *heuristic* grader --
which is what decides every round when no model is configured, and the fallback
whenever a grading call fails -- cannot propose one at all. Word overlap has not
read the corpus, so it cannot know that a question about "how quickly the
computer has to run" is answered by notes that say *adiabatic theorem* and
*minimum gap*. That gap between the user's vocabulary and the corpus's is the
whole problem a rewrite exists to solve.

The rewriter sees the passages that *failed*, which is more information than the
question alone carries. Those passages are the corpus telling the caller what
vocabulary it actually uses: the nearest neighbours of a query are wrong, but
they are wrong in the corpus's own words. A rewrite composed without them is a
guess, and :func:`src.rag.retrieve.rewrite` is already the better guess.

What the model may not do is answer. It returns a query, that query is used for
nothing but a similarity search, and it is neutralised on the way in like every
other model output that reaches a trace the user reads.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from src import security
from src.agent.grading import MAX_QUERY_CHARACTERS, as_material
from src.agent.llm import ask_structured, chat_model_or_none
from src.logging_setup import get_logger
from src.rag.retrieve import Passage, Rewriter

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from langchain_core.language_models import BaseChatModel

    from src.settings import Settings

LOG = get_logger("agent.rewriting")

REWRITE_SYSTEM = """You write search queries for a library of notes about the \
one-dimensional transverse-field Ising model.

A search has just failed. You are given the question, the query that was used, \
and the passages that came back and were judged not to answer it. Write one \
better query.

The failed passages are your evidence. They are the wrong notes, but they are \
written in the vocabulary this library uses, so read them for the technical \
terms the question is missing. A question asking how long a quantum computer \
must run to stay in its ground state should become a query about the adiabatic \
theorem and the minimum gap, because that is what the notes call it.

Rules. Write keywords, not a sentence -- the query is embedded and compared to \
passages, and a polite question wastes most of its length on words every passage \
contains. Do not repeat the failed query. Do not answer the question, do not \
add facts, and do not name a specific chain length, coupling or field: numbers \
are computed elsewhere and no note contains them.

Return an empty query if you can see nothing better to try. That is a useful \
answer: searching again for the same thing costs a round and reaches the same \
conclusion, and an honest refusal is a valid outcome."""


class Reformulation(BaseModel):
    """A rewritten query, in the shape the model must return it.

    The field descriptions are what the model reads for each field, which is why
    they restate the system prompt. Constraints are absent on purpose -- see
    :data:`src.agent.grading.MAX_QUERY_CHARACTERS`.

    Attributes:
        query: The keyword query to try next, or empty for "nothing better".
        reason: One sentence on what was missing, for the trace.
    """

    model_config = ConfigDict(frozen=True)

    query: str = Field(description="A short keyword query to try instead, or an empty string.")
    reason: str = Field(description="One plain sentence on what the first query was missing.")


def rewrite_query(
    question: str,
    failed_query: str,
    candidates: tuple[Passage, ...],
    *,
    model: BaseChatModel | None = None,
    settings: Settings | None = None,
) -> str:
    """Ask a model what to search for after ``failed_query`` found nothing useful.

    Args:
        question: The question as asked, already screened by
            :func:`src.agent.router.guard`.
        failed_query: The query that was searched.
        candidates: The passages it returned, which the grader rejected.
        model: An explicit chat model, normally supplied only by tests.
        settings: Configuration to build a model from.

    Returns:
        A query, or ``""`` when there is no model, the call failed, the model saw
        nothing better, or it returned the failed query again. Empty means the
        caller keeps :func:`src.rag.retrieve.rewrite`, so an unavailable model
        costs a better query rather than the round.
    """
    resolved = chat_model_or_none(model, settings)
    if resolved is None:
        return ""

    material = f"FAILED QUERY: {failed_query}\n\n{as_material(question, candidates)}"
    answer = ask_structured(resolved, Reformulation, REWRITE_SYSTEM, material, purpose="rewrite")
    if answer is None:
        return ""

    # Whitespace is collapsed, not just trimmed: a query is one line by
    # definition, and a model that returned a bulleted list has written notes.
    query = " ".join(security.neutralise(answer.query).split())[:MAX_QUERY_CHARACTERS]
    # A query differing from the failed one only in spacing or case embeds to the
    # same place. Treating that as "nothing better" spends the round on the
    # deterministic rewrite, which at least searches different words.
    if query.casefold().split() == failed_query.casefold().split():
        LOG.info("rewrite_unchanged", extra={"detail": "falling back to the deterministic rewrite"})
        return ""
    LOG.info(
        "rewritten_by_model",
        extra={
            "candidates": len(candidates),
            "gave_up": not query,
            "reason": security.neutralise(answer.reason).strip()[:MAX_QUERY_CHARACTERS],
        },
    )
    return query


def build_rewriter(
    *,
    model: BaseChatModel | None = None,
    settings: Settings | None = None,
) -> Rewriter:
    """Bind a model to :func:`rewrite_query` so it fits the seam retrieval expects.

    Args:
        model: An explicit chat model, normally supplied only by tests.
        settings: Configuration to build a model from.

    Returns:
        A callable matching :data:`~src.rag.retrieve.Rewriter`, safe to pass to
        :func:`src.rag.retrieve.retrieve` whether or not a model exists.
    """

    def rewriter(question: str, failed_query: str, candidates: tuple[Passage, ...]) -> str:
        return rewrite_query(question, failed_query, candidates, model=model, settings=settings)

    return rewriter
