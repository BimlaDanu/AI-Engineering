"""The four model-backed retrieval stages, and what each does when there is no model.

`src/agent/grading.py` supplies the language-model versions of four judgements the
retrieval layer can make without one: which passages are relevant, how to reword a
query that found nothing, what else the question could be called, and which relevant
passage the answer should be written from first.

**The property every one of these tests holds is the same.** A stage that is
unavailable must leave the deterministic route running, not break the search. That is
what lets `make run` answer questions on a checkout with no credential, and it is why
these tests need no network: an offline `ModelPool` returns ``None`` from every call,
which is exactly the case that has to work.

The second property is narrower and matters more. Three of the four stages ask a
model for **identifiers it was given**, never for prose, so the worst a wrong answer
can do is nominate a passage that was already on offer. A model that invents an
identifier, returns nothing, or raises must cost the ordering and never the answer.
"""

from __future__ import annotations

import pytest

from src.agent.grading import (
    MIN_TO_REORDER,
    Ordering,
    Phrasings,
    model_expander,
    model_grader,
    model_reranker,
    model_rewriter,
)
from src.agent.model_selection import ModelPool
from src.rag.retrieve import Passage


def passage(identifier: str, text: str = "the gap closes at the critical point") -> Passage:
    """One retrieved passage, with only the fields these stages read."""
    return Passage(
        identifier=identifier,
        text=text,
        document=identifier.split("#")[0],
        title="Exact solution of the transverse-field Ising chain",
        source="Pfeuty, Annals of Physics 57, 79 (1970)",
        arxiv="",
        section="The gap",
        topics=("exact-solution",),
        score=0.8,
    )


CANDIDATES = (passage("a#1"), passage("b#2"), passage("c#3"))


# --------------------------------------------------------------------------
# With no model at all
# --------------------------------------------------------------------------


def test_no_model_means_the_deterministic_rules_decide() -> None:
    # `None` from a grader is not a failure signal -- it is the retrieval layer's
    # instruction to use its own rules, which is what an offline campaign runs on.
    pool = ModelPool(offline=True)
    assert model_grader(pool)("Why does the gap close?", ()) is None


def test_no_model_means_no_extra_phrasings_rather_than_a_broken_search() -> None:
    pool = ModelPool(offline=True)
    assert model_expander(pool)("Why does the gap close?", ("exact-solution",)) == ()


def test_no_model_leaves_the_fused_order_alone() -> None:
    pool = ModelPool(offline=True)
    assert model_reranker(pool)("Why does the gap close?", CANDIDATES) == ()


def test_no_model_falls_back_to_the_deterministic_reformulation() -> None:
    # An empty string here means "nothing better than the floor", and the floor is
    # `retrieve.rewrite`, which needs no model.
    pool = ModelPool(offline=True)
    assert model_rewriter(pool)("Why does the gap close?", "gap close", CANDIDATES) == ""


# --------------------------------------------------------------------------
# Reranking: a permutation, and never anything else
# --------------------------------------------------------------------------


@pytest.mark.parametrize("count", range(MIN_TO_REORDER + 1))
def test_a_set_too_small_to_have_an_interesting_order_is_not_a_call(count: int) -> None:
    # One passage has one ordering; two have two, and the fused ranking already chose
    # between them on the evidence of both halves of the search agreeing. A network
    # round trip to consider swapping two items makes an application feel slow for
    # nothing measurable.
    pool = ModelPool(pinned=_Refusing())  # type: ignore[arg-type]
    # `_Refusing` raises if it is reached, so reaching the assertion is the proof.
    assert model_reranker(pool)("Why does the gap close?", CANDIDATES[:count]) == ()


def test_an_invented_identifier_is_dropped_rather_than_trusted() -> None:
    # The reason these stages answer with identifiers: a model that returns something
    # it was not given can be caught by set membership, which no amount of prompt
    # wording can guarantee.
    pool = ModelPool(pinned=_Answering(Ordering(order=["c#3", "not-a-real-id", "a#1"])))  # type: ignore[arg-type]
    named = model_reranker(pool)("Why does the gap close?", CANDIDATES)
    assert named == ("c#3", "a#1")


# --------------------------------------------------------------------------
# Expansion: additive, bounded, and never a copy of the question
# --------------------------------------------------------------------------


def test_a_phrasing_identical_to_the_question_is_not_a_phrasing() -> None:
    # The original is searched regardless, so returning it again fuses a list with
    # itself. That doubles every score in it, changes no ordering, and costs a search.
    asked = "Why does the gap close?"
    pool = ModelPool(
        pinned=_Answering(  # type: ignore[arg-type]
            Phrasings(queries=[asked, "  WHY DOES THE GAP CLOSE?  ", "gap closing"])
        )
    )
    assert model_expander(pool)(asked, ()) == ("gap closing",)


def test_a_repeated_phrasing_is_searched_once() -> None:
    pool = ModelPool(
        pinned=_Answering(  # type: ignore[arg-type]
            Phrasings(queries=["gap closing", "gap closing", "correlation length"])
        )
    )
    assert model_expander(pool)("Why?", ()) == ("gap closing", "correlation length")


def test_an_empty_proposal_is_not_searched() -> None:
    pool = ModelPool(pinned=_Answering(Phrasings(queries=["", "   ", "criticality"])))  # type: ignore[arg-type]
    assert model_expander(pool)("Why?", ()) == ("criticality",)


def test_the_routed_topics_reach_the_expansion_prompt() -> None:
    # The router has already decided what this question is about. An expander that
    # ignored that would propose phrasings for a subject the question was not routed
    # to, and the searches it bought would be spent on the wrong shelf's vocabulary.
    model = _Answering(Phrasings(queries=["criticality"]))
    pool = ModelPool(pinned=model)  # type: ignore[arg-type]
    model_expander(pool)("Why does it get hard?", ("critical-point", "energy-gap"))
    sent = " ".join(model.seen)
    assert "critical point" in sent
    assert "energy gap" in sent


# --------------------------------------------------------------------------
# Doubles
# --------------------------------------------------------------------------


class _Structured:
    """What ``with_structured_output`` returns: something with an ``invoke``."""

    def __init__(self, answer: object, seen: list[str]) -> None:
        self._answer = answer
        self._seen = seen

    def invoke(self, messages: object) -> object:
        """Record what was asked and answer with the canned value."""
        self._seen.append(str(messages))
        return self._answer


class _Answering:
    """A chat model that returns one canned structured answer.

    Attributes:
        seen: Everything it was sent, so a test can assert what reached the prompt
            without asserting the prompt's wording -- which is a text this project
            expects to keep editing.
    """

    model_name = "stub"

    def __init__(self, answer: object) -> None:
        """Hold the answer every call will return."""
        self._answer = answer
        self.seen: list[str] = []

    def with_structured_output(self, schema: object, include_raw: bool = False) -> _Structured:
        """Hand back something that answers with the canned value."""
        return _Structured(self._answer, self.seen)


class _Refusing:
    """A chat model that fails if it is ever called.

    Standing in for the model a stage decided not to reach for. A test asserting
    "no call was made" against a model that would happily answer is asserting the
    ledger's bookkeeping; this asserts the decision.
    """

    model_name = "stub"

    def with_structured_output(self, schema: object, include_raw: bool = False) -> object:
        """Fail: nothing should reach this."""
        raise AssertionError("a stage called a model it should have skipped")
