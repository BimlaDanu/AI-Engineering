"""Tests for the loop's decider.

Three properties are specified here, and none of them is about a model.

**The decision reads state, not vocabulary.** Every test below builds a
:class:`~src.agent.deciding.Progress` and asserts on the action. There is no
question text in most of them, which is the point: a policy that needed the words
would be a keyword table with extra steps, and the next unanticipated question
would need another entry.

**The knowledge base comes first.** The external tools are unreachable until
retrieval has actually run, and a model that asks for them early is overridden
rather than obeyed.

**The loop is bounded.** Three actions, enforced here and again in the graph edge.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from src.agent import llm
from src.agent.deciding import (
    MAX_SEARCHES,
    MAX_STEPS,
    Progress,
    Step,
    decide,
    is_forced,
    policy_step,
)


@pytest.fixture(autouse=True)
def no_live_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make an accidental live model call impossible for the whole module."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("a test tried to build a real chat model")

    monkeypatch.setattr(llm, "build_chat_model", refuse)


DRY = Progress(wants_prose=True, searches=1, passages=0)
"""The one state that leaves a real choice: searched, and nothing came back."""


class Model:
    """A chat model returning one prepared structured payload."""

    model_name = "test/model"

    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.binds = 0

    def with_structured_output(self, schema: type[BaseModel], **kwargs: object) -> Any:
        """Hand back something whose ``invoke`` returns the payload."""
        self.binds += 1
        return _Bound(self.payload)


class _Bound:
    """The runnable ``with_structured_output`` returns, faked."""

    def __init__(self, payload: object) -> None:
        self.payload = payload

    def invoke(self, messages: list[Any]) -> object:
        """Return the payload in the shape ``include_raw=True`` produces."""
        return {"raw": AIMessage(content=""), "parsed": self.payload, "parsing_error": None}


# --- the offline policy ---------------------------------------------------


def test_a_question_wanting_a_number_computes_first() -> None:
    # A verified number is the one thing nothing else here can substitute for.
    assert policy_step(Progress(wants_number=True)).action == "compute"


def test_the_corpus_is_searched_once_the_number_is_in_hand() -> None:
    solved = Progress(wants_number=True, has_numbers=True)
    assert policy_step(solved).action == "retrieve"


def test_an_explanation_is_looked_up_rather_than_recalled() -> None:
    assert policy_step(Progress(wants_prose=True)).action == "retrieve"


def test_the_outside_world_is_unreachable_until_the_corpus_has_been_tried() -> None:
    # The rule the whole ordering exists for: no tool call while the project's own
    # notes are unsearched. "Not searched" and "searched and found nothing" are
    # different states, and only the second one justifies reaching outside.
    untried = Progress(wants_prose=True)
    assert policy_step(untried).action != "consult"
    dry = Progress(wants_prose=True, searches=1, passages=0)
    assert policy_step(dry).action == "consult"


def test_a_search_that_found_something_does_not_reach_outside() -> None:
    grounded = Progress(wants_prose=True, searches=1, passages=3)
    assert policy_step(grounded).action == "finish"


def test_the_tools_are_not_offered_twice() -> None:
    spent = Progress(wants_prose=True, searches=1, passages=0, tools_run=True)
    assert policy_step(spent).action == "finish"


def test_a_finished_run_stops() -> None:
    done = Progress(wants_number=True, has_numbers=True, searches=1, passages=2)
    assert policy_step(done).action == "finish"


@pytest.mark.parametrize("taken", [MAX_STEPS, MAX_STEPS + 1])
def test_a_spent_budget_stops_whatever_is_missing(taken: int) -> None:
    # Nothing has been established and the question wanted everything, and it
    # still stops: an unbounded loop is the one failure this project cannot pay
    # for. What is missing becomes a caveat on the answer, not another round.
    hungry = Progress(wants_number=True, wants_prose=True, taken=taken)
    assert policy_step(hungry).action == "finish"


def test_every_policy_decision_says_who_made_it() -> None:
    # A trajectory that does not say whether a model or the fallback chose cannot
    # be audited, and the interface shows it.
    for progress in (
        Progress(wants_number=True),
        Progress(wants_prose=True),
        Progress(searches=1, taken=MAX_STEPS),
    ):
        step = policy_step(progress)
        assert step.decided_by == "policy"
        assert step.reason


# --- with a model ---------------------------------------------------------


def test_a_models_choice_is_used_and_attributed() -> None:
    # The open case, and the only one a model is asked about: the corpus was
    # searched and came back empty, so reaching outside and admitting the gap are
    # both defensible.
    model = Model(Step(action="consult", reason="arXiv is worth a try here"))
    step = decide("Why does the gap close?", DRY, model=model)  # type: ignore[arg-type]
    assert step.action == "consult"
    assert step.decided_by == "model"
    assert step.reason == "arXiv is worth a try here"


def test_a_model_reaching_outside_too_early_is_overridden() -> None:
    # The ordering is not a suggestion in the prompt. Reached here through the
    # override rather than through is_forced, so the rule is tested on its own:
    # a model asking for arXiv before the corpus has been read is refused.
    model = Model(Step(action="consult", reason="arXiv will know"))
    untried = Progress(wants_prose=True)
    assert decide("Who solved this model?", untried, model=model).action == "retrieve"  # type: ignore[arg-type]


def test_a_model_may_reach_outside_once_the_corpus_came_back_empty() -> None:
    model = Model(Step(action="consult", reason="the notes had nothing"))
    dry = Progress(wants_prose=True, searches=1, passages=0)
    step = decide("Who solved this model?", dry, model=model)  # type: ignore[arg-type]
    assert step.action == "consult"
    assert step.decided_by == "model"


def test_an_unparseable_reply_falls_back_to_the_policy() -> None:
    step = decide("Why does the gap close?", DRY, model=Model(None))  # type: ignore[arg-type]
    assert step.action == "consult"
    assert step.decided_by == "policy"


def test_a_forced_decision_is_not_worth_a_model_call() -> None:
    # The cost rule, kept for the states where consulting a model cannot change the
    # outcome: the tools have been spent, or the searches are used up and something
    # came back with nowhere left to look. Paying a model to agree is latency, not
    # judgement, and on the 23-case suite it was minutes of it.
    forced = (
        Progress(wants_prose=True, searches=1, passages=0, tools_run=True),
        Progress(wants_prose=True, searches=MAX_SEARCHES, passages=2),
    )
    for progress in forced:
        assert is_forced(progress)
        model = Model(Step(action="consult", reason="let me look outside"))
        step = decide("Why does the gap close?", progress, model=model)  # type: ignore[arg-type]
        assert step.decided_by == "policy"
        assert model.binds == 0


def test_the_opening_move_is_decided_rather_than_assumed() -> None:
    """The first action of a run is a decision, and it was not always.

    Forcing it meant every trajectory in the application began
    ``retrieve[policy]``: a numeric question searched the notes before solving
    anything, and a question the conversation had already answered searched again
    anyway. A loop that can revise a plan but never choose one is a pipeline with a
    decision bolted to its second step.
    """
    opening = Progress(wants_number=True)
    assert not is_forced(opening)
    model = Model(Step(action="compute", reason="this question only wants a value"))
    step = decide("What is the ground-state energy?", opening, model=model)  # type: ignore[arg-type]
    assert step.action == "compute"
    assert step.decided_by == "model"
    assert model.binds == 1


def test_answering_from_nothing_is_refused_however_the_model_asks() -> None:
    # The guard that makes handing over the opening move safe. The model may choose
    # how to establish something; it may not choose to establish nothing.
    model = Model(Step(action="finish", reason="I already know this one"))
    step = decide("Why does the gap close?", Progress(wants_prose=True), model=model)  # type: ignore[arg-type]
    assert step.action == "retrieve"
    assert step.decided_by == "policy"


def test_answering_a_question_about_a_curve_without_one_is_refused() -> None:
    """A question that asked for a picture may not be finished with a number.

    Reported in use: *plot the low-lying spectrum of a quantum Ising chain* solved
    one chain, searched the notes and replied "I cannot make plots here" -- true, in
    that nothing had drawn anything, and avoidable, in that the sweep tool was
    never offered. :func:`policy_step` already sent a curve request to ``consult``,
    but the policy only decides when no model is reachable, so with a key
    configured the rule was not enforced at all.
    """
    wants_a_picture = Progress(
        wants_number=True, wants_curve=True, has_numbers=True, searches=1, passages=3
    )
    assert policy_step(wants_a_picture).action == "consult"
    model = Model(Step(action="finish", reason="I have the energy, that will do"))
    step = decide("Plot the low-lying spectrum.", wants_a_picture, model=model)  # type: ignore[arg-type]
    assert step.action == "consult"
    assert step.decided_by == "policy"


def test_the_curve_guard_lets_go_once_the_tools_have_been_offered() -> None:
    # It guards the *offer*, not the outcome. The model may look at the sweep and
    # decline it; what it may not do is finish before it has seen it. Without this
    # the loop would spend its whole budget asking for a tool step it already took.
    offered = Progress(
        wants_number=True,
        wants_curve=True,
        has_numbers=True,
        searches=1,
        passages=3,
        tools_run=True,
    )
    assert policy_step(offered).action == "finish"


def test_a_dry_search_is_not_material_and_cannot_be_answered_from() -> None:
    # "Searched and found nothing" establishes that the corpus is quiet, which is
    # worth knowing and is not something to answer from.
    assert not Progress(wants_prose=True, searches=1, passages=0).established
    assert Progress(wants_prose=True, searches=1, passages=3).established
    assert Progress(wants_number=True, has_numbers=True).established


def test_the_open_decision_is_the_one_a_model_is_asked_about() -> None:
    # A dry search is where judgement actually lives: reach outside, or say the
    # notes do not cover this?
    assert not is_forced(DRY)


def test_finding_something_is_not_the_same_as_finding_enough() -> None:
    # The regression this whole distinction exists for. Treating "passages > 0" as
    # forced meant every ordinary question ran retrieve then finish: no decision was
    # ever taken, the trajectory showed two policy steps, and a partial answer was
    # indistinguishable from a complete one. Whether the material covers the
    # question is the one judgement nothing else in this project can make.
    partial = Progress(wants_prose=True, searches=1, passages=2, covered=("Trotter error",))
    assert not is_forced(partial)
    assert partial.can_search_again


def test_a_knowledge_base_that_would_not_open_is_not_searched_again() -> None:
    # A rewritten query is the remedy for a corpus that did not match. Against an
    # index that is not there it buys a model call and the same nothing, and the
    # line the decider reads -- "0 of 2 searches, 0 passages kept" -- is identical
    # in both cases, so the state has to say which one this is.
    absent = Progress(wants_prose=True, searches=1, passages=0, corpus_available=False)
    assert not absent.can_search_again
    assert "never actually searched" in absent.describe()
    quiet = Progress(wants_prose=True, searches=1, passages=0)
    assert quiet.can_search_again
    assert "never actually searched" not in quiet.describe()


def test_a_gap_can_be_searched_for_a_second_time() -> None:
    # And the second search must be aimed at something new, which is what focus is.
    model = Model(
        Step(
            action="retrieve",
            reason="the passages cover the circuits but not the hardware limits",
            focus="qubit connectivity and gate fidelity",
        )
    )
    partial = Progress(wants_prose=True, searches=1, passages=2, covered=("Trotter error",))
    step = decide("How do I run this on NISQ hardware?", partial, model=model)  # type: ignore[arg-type]
    assert step.action == "retrieve"
    assert step.decided_by == "model"
    assert step.focus == "qubit connectivity and gate fidelity"


def test_searching_again_for_nothing_in_particular_is_refused() -> None:
    # Without a focus the follow-up re-runs the same query and returns the passages
    # already in hand: a step of the budget spent on a duplicate.
    model = Model(Step(action="retrieve", reason="let me look again"))
    partial = Progress(wants_prose=True, searches=1, passages=2)
    step = decide("How do I run this on NISQ hardware?", partial, model=model)  # type: ignore[arg-type]
    assert step.action == "finish"
    assert step.decided_by == "policy"


def test_the_searches_run_out() -> None:
    # Two searches is the budget: the second has new information behind it, and a
    # third would be guessing.
    model = Model(Step(action="retrieve", reason="once more", focus="something else"))
    spent = Progress(wants_prose=True, searches=MAX_SEARCHES, passages=0)
    assert decide("Why?", spent, model=model).action != "retrieve"  # type: ignore[arg-type]


def test_a_decider_still_wanting_material_is_sent_outside_the_corpus() -> None:
    # The bug this fixes, read off a live log: asked to find recent arXiv papers, the
    # loop searched twice, asked to search a third time, was refused -- and the
    # refusal fell back to the policy, which composed from the notes. The paper tool
    # was never offered on a question that asked for papers by name. A refused
    # "search again" is a statement that the material falls short, so with the corpus
    # spent the outside sources are what is left.
    model = Model(Step(action="retrieve", reason="still not covered", focus="recent work"))
    spent = Progress(
        wants_prose=True,
        searches=MAX_SEARCHES,
        passages=3,
        tools_available=True,
    )
    step = decide("Find recent arXiv papers on Trotter error.", spent, model=model)  # type: ignore[arg-type]
    assert step.action == "consult"
    assert step.decided_by == "policy"


def test_with_nowhere_else_to_look_the_answer_is_composed() -> None:
    # Same state, no outside source configured. Redirecting to an action nothing can
    # carry out would be a step of the budget spent on nothing.
    model = Model(Step(action="retrieve", reason="still not covered", focus="recent work"))
    spent = Progress(wants_prose=True, searches=MAX_SEARCHES, passages=3)
    assert decide("Why?", spent, model=model).action == "finish"  # type: ignore[arg-type]


def test_solving_the_same_chain_a_second_time_is_refused() -> None:
    # Observed live as `compute[model] -> compute[model]`. The chain's parameters come
    # from the settings, not the question, so the second solve is the first one again
    # -- a model call and a diagonalisation for a number already in hand.
    model = Model(Step(action="compute", reason="let me solve it again"))
    solved = Progress(
        wants_number=True,
        has_numbers=True,
        wants_prose=True,
        searches=MAX_SEARCHES,
        passages=2,
        tools_available=True,
    )
    step = decide("What is the ground-state energy?", solved, model=model)  # type: ignore[arg-type]
    assert step.action != "compute"
    assert step.decided_by == "policy"


def test_the_policy_never_searches_twice_by_itself() -> None:
    # It cannot: naming the gap takes reading the passages, and the policy reads a
    # count. Finishing with what is there is the honest fallback.
    partial = Progress(wants_prose=True, searches=1, passages=2)
    assert policy_step(partial).action == "finish"


def test_a_spent_budget_is_not_worth_a_model_call() -> None:
    # A decision has already been made by the budget, so asking is spending for
    # nothing -- and this is checked before the model is built, not after.
    model = Model(Step(action="retrieve", reason="one more round"))
    step = decide("Why?", Progress(wants_prose=True, taken=MAX_STEPS), model=model)  # type: ignore[arg-type]
    assert step.action == "finish"
    assert model.binds == 0


def test_no_model_means_the_policy_decides() -> None:
    # The autouse fixture makes building one raise, so reaching for a model here
    # would fail loudly rather than silently costing money.
    assert decide("Why does the gap close?", Progress(wants_prose=True)).decided_by == "policy"


# --- what the decider is shown -------------------------------------------


def test_the_state_is_described_rather_than_the_question_repeated() -> None:
    described = Progress(wants_number=True, has_numbers=True, searches=1, passages=2).describe()
    assert "Numbers computed and cross-checked: yes" in described
    assert "passages kept: 2" in described
    assert f"of {MAX_STEPS}" in described


# --------------------------------------------------------------------------
# A curve is a computation, and only the sweep can produce one
# --------------------------------------------------------------------------


def test_a_question_about_a_curve_reaches_for_the_sweep_first() -> None:
    """The bug: "plot the magnetisation against the field" produced no plot.

    The sweep lives behind ``consult``, and ``consult`` was unreachable until the
    corpus had been searched and had come back short. So a plot request solved the
    single chain in the settings knob, searched the notes, wrote a paragraph about
    one number, and drew nothing. The tool step is offered immediately now, because
    a sweep is this project's own physics rather than an outside source.
    """
    # The single point is still solved first -- it is what the curve is marked
    # against -- and the sweep comes before the corpus rather than after it.
    wants_plot = Progress(wants_number=True, wants_curve=True, tools_available=True)
    assert policy_step(wants_plot).action == "compute"
    solved = Progress(wants_number=True, wants_curve=True, has_numbers=True, tools_available=True)
    step = policy_step(solved)
    assert step.action == "consult"
    assert "curve" in step.reason
    # Without the curve, the same state searches the notes instead.
    assert policy_step(Progress(wants_number=True, has_numbers=True)).action == "retrieve"


def test_a_curve_request_may_reach_the_tools_before_the_corpus() -> None:
    # The knowledge-base-first rule protects claims about the literature. A curve is
    # not such a claim, so the override does not apply to it.
    model = Model(Step(action="consult", reason="this needs a sweep"))
    progress = Progress(wants_number=True, wants_curve=True, tools_available=True)
    step = decide("Plot the magnetisation against h.", progress, model=model)  # type: ignore[arg-type]
    assert step.action == "consult"
    assert step.decided_by == "model"


def test_reaching_outside_for_prose_still_waits_for_the_corpus() -> None:
    # The rule itself is untouched: without a curve to draw, an early consult is
    # still refused and redirected to the knowledge base.
    model = Model(Step(action="consult", reason="let me search arXiv"))
    step = decide("Who solved this model first?", Progress(wants_prose=True), model=model)  # type: ignore[arg-type]
    assert step.action == "retrieve"
    assert step.decided_by == "policy"


def test_a_curve_is_not_asked_for_twice() -> None:
    # Once the tools have run, the sweep either happened or was declined; offering
    # them again would spend a step to ask the same question.
    swept = Progress(wants_number=True, wants_curve=True, tools_available=True, tools_run=True)
    assert policy_step(swept).action != "consult"


def test_a_number_that_was_asked_for_cannot_be_skipped() -> None:
    """Found by the held-out suite, not by reasoning about the code.

    *Why does the gap close, and what is it at L = 8?* asks for both halves. With the
    opening move handed to the model, it chose to search, read the passages, reached
    outside, and composed -- a fluent answer to half the question with the solver
    never run. Passages count as material, so the "answer from nothing" guard was
    satisfied; nothing was watching for the number still owed.
    """
    owed = Progress(wants_number=True, wants_prose=True, searches=1, passages=3)
    model = Model(Step(action="finish", reason="the passages explain it"))
    step = decide("Why does the gap close, and what is it at L = 8?", owed, model=model)  # type: ignore[arg-type]
    assert step.action == "compute"
    assert step.decided_by == "policy"


def test_finishing_is_allowed_once_the_number_is_in_hand() -> None:
    done = Progress(wants_number=True, has_numbers=True, searches=1, passages=3)
    model = Model(Step(action="finish", reason="everything asked for is established"))
    step = decide("What is the energy, and why?", done, model=model)  # type: ignore[arg-type]
    assert step.action == "finish"
    assert step.decided_by == "model"
