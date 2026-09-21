"""The model-backed relevance grader for retrieval.

This is the half of agentic RAG that decides whether what came back is worth
reading. It lives here rather than in :mod:`src.rag` because it calls a model and
``rag`` is not allowed to: :mod:`src.rag.retrieve` declares the seam
(:data:`~src.rag.retrieve.Grader`) and this module fills it, so the dependency
points one way and the retriever stays testable with no credential.

One structured call per round, and it answers two questions at once: which of
these passages to keep, and -- having just read the weak ones -- what to search
for instead. The second half is what makes the loop worth running twice; a
rewrite composed without seeing the failed results is a guess.

The model is authoritative on relevance, which is the opposite of the
arrangement in :func:`src.agent.router.guard`, where the classifier may only add
a block. The asymmetry is deliberate. Injection screening is a security boundary,
so the strict layer wins and the model is advisory. Relevance is a judgement
call, and the model is better at it than word overlap; the cost of it being
wrong is a thin answer or a needless refusal, not a leak. What the model cannot
do is invent evidence: only ids that were actually retrieved survive
:func:`_kept_ids`, so a hallucinated citation drops out instead of appearing in
an answer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from src import security
from src.agent.llm import ask_structured, chat_model_or_none
from src.logging_setup import get_logger
from src.rag.retrieve import Grade, Grader, Passage

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from langchain_core.language_models import BaseChatModel

    from src.settings import Settings

LOG = get_logger("agent.grading")

MAX_QUERY_CHARACTERS = 200
"""Longest rewritten query accepted from the model.

The schema carries no length constraint -- providers behind OpenRouter support
those unevenly -- so the bound is applied after the fact. A model that answers
the "what should I search for instead" question with a paragraph has written an
explanation, not a query, and embedding a paragraph retrieves nothing in
particular.
"""

GRADING_SYSTEM = """You judge whether retrieved passages can answer a physics \
question about the one-dimensional transverse-field Ising model.

You are given the question and numbered passages, each with an id. For each \
passage ask one thing: does this passage contain material that helps answer \
this question? Keep it if it does. Discard it if it is merely about the same \
general subject -- a passage about the phase diagram does not answer a question \
about correlation functions.

Keeping nothing is a correct and expected answer. The passages come from a \
similarity search, which always returns its nearest neighbours whether or not \
any of them are relevant, so a question the corpus does not cover still \
produces confident-looking results. Downstream, an empty answer becomes an \
honest refusal; a passage kept out of politeness becomes a citation that does \
not support the claim beside it.

Return the ids you keep, one sentence saying why you kept or rejected them, and \
-- if you kept nothing -- a short keyword query to try instead, phrased in the \
vocabulary the passages actually use. Return an empty query if you kept \
something, or if you can see nothing better to search for."""


class Relevance(BaseModel):
    """A grading decision, in the shape the model must return it.

    The field descriptions are the instructions the model reads for each field,
    which is why they repeat what the system prompt says. Constraints are absent
    on purpose: see :data:`MAX_QUERY_CHARACTERS`.

    Attributes:
        keep: Ids of the passages that help, most useful first.
        reason: One sentence, shown to the user when retrieval comes back empty.
        next_query: A better query to try, or empty for "nothing better to try".
    """

    model_config = ConfigDict(frozen=True)

    keep: list[str] = Field(description="Ids of passages that help answer the question.")
    reason: str = Field(description="One plain sentence explaining what was kept and why.")
    next_query: str = Field(description="A short keyword query to try instead, or an empty string.")


def as_material(question: str, candidates: tuple[Passage, ...]) -> str:
    """Lay out the question and the candidates for the model to read.

    Args:
        question: The question as asked.
        candidates: The passages to judge.

    Returns:
        A block naming each passage's id, so the reply can refer to them.
        Passage text is already neutralised by
        :meth:`~src.rag.retrieve.Passage.of`; the whole block is neutralised
        again by :func:`~src.agent.llm.as_data` on the way out.
    """
    lines = [f"QUESTION: {question}", "", "PASSAGES:"]
    for passage in candidates:
        lines.append(f"id={passage.identifier} ({passage.citation})")
        lines.append(passage.text)
        lines.append("")
    return "\n".join(lines)


def _kept_ids(keep: list[str], candidates: tuple[Passage, ...]) -> tuple[str, ...]:
    """Keep only ids that were actually retrieved, in the model's order.

    Args:
        keep: Ids as the model returned them.
        candidates: The passages it was shown.

    Returns:
        The valid ids, deduplicated. An id the model invented is dropped rather
        than reported, because there is no passage behind it -- the alternative
        is a citation to text nobody retrieved.
    """
    known = {passage.identifier for passage in candidates}
    return tuple(dict.fromkeys(identifier for identifier in keep if identifier in known))


def grade(
    question: str,
    candidates: tuple[Passage, ...],
    *,
    model: BaseChatModel | None = None,
    settings: Settings | None = None,
) -> Grade | None:
    """Ask a model which of ``candidates`` answer ``question``.

    Args:
        question: The question as asked, already screened by
            :func:`src.agent.router.guard`.
        candidates: One round's passages.
        model: An explicit chat model, normally supplied only by tests.
        settings: Configuration to build a model from.

    Returns:
        The judgement, or ``None`` for "no opinion" -- when there is nothing to
        grade, no model to ask, or the call failed. ``None`` sends the caller to
        :func:`src.rag.retrieve.heuristic_grade`, so an unavailable model costs
        accuracy rather than availability.
    """
    if not candidates:
        return None
    resolved = chat_model_or_none(model, settings)
    if resolved is None:
        return None

    material = as_material(question, candidates)
    answer = ask_structured(resolved, Relevance, GRADING_SYSTEM, material, purpose="grade")
    if answer is None:
        return None

    keep = _kept_ids(answer.keep, candidates)
    LOG.info(
        "graded_by_model",
        extra={
            "candidates": len(candidates),
            "kept": len(keep),
            "invented": len(answer.keep) - len(keep),
        },
    )
    return Grade(
        keep=keep,
        reason=security.neutralise(answer.reason).strip(),
        query=security.neutralise(answer.next_query).strip()[:MAX_QUERY_CHARACTERS],
        graded_by="model",
    )


def build_grader(
    *,
    model: BaseChatModel | None = None,
    settings: Settings | None = None,
) -> Grader:
    """Bind a model to :func:`grade` so it fits the seam retrieval expects.

    Args:
        model: An explicit chat model, normally supplied only by tests.
        settings: Configuration to build a model from.

    Returns:
        A callable matching :data:`~src.rag.retrieve.Grader`, safe to pass to
        :func:`src.rag.retrieve.retrieve` whether or not a model exists.
    """

    def grader(question: str, candidates: tuple[Passage, ...]) -> Grade | None:
        return grade(question, candidates, model=model, settings=settings)

    return grader
