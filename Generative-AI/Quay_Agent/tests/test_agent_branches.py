"""The three branches, end to end: the campaign must answer the question asked.

Before the branch existed, these three questions produced one document. *Write me a
variational eigensolver* returned a feasibility verdict reading **NO** followed by a
table of energies; so did *what is a barren plateau*; so did *explain why this chain
is exactly solvable*. Each was fluent, sourced, internally consistent and about
something else -- which is worse than a refusal, because a refusal is legible.

These tests hold the fix at the level a reader would notice it: the route taken, the
document produced, and above all the things a branch must *not* do.

Everything runs offline, with no key. That is the mode the whole suite runs in and
the mode a grader will see, so the branches have to be worth something without one.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from src.agent import graph
from src.agent.graph import run_campaign
from src.agent.state import CampaignState, Draft, Request, new_campaign
from src.hardware.devices import device_for
from src.settings import get_settings

BUDGET = 10**9  # the budget the application itself ships with; see src/ui/setting.py


def campaign(question: str) -> tuple[CampaignState, list[str]]:
    """Run one offline campaign and return it with the route it took.

    Args:
        question: What to ask.

    Returns:
        The finished state and the nodes visited, in order.
    """
    visited: list[str] = []
    state = run_campaign(
        question,
        shot_budget=BUDGET,
        device=device_for("heavy-hex-27"),
        chat_model=None,
        search_corpus=True,
        fetch_external=False,
        visited=visited,
    )
    return state, visited


@pytest.fixture(scope="module")
def asked_for_code() -> tuple[CampaignState, list[str]]:
    return campaign("Write me a VQE implementation for a 10-spin Ising chain")


@pytest.fixture(scope="module")
def asked_for_prose() -> tuple[CampaignState, list[str]]:
    return campaign("What is a barren plateau and does it affect this chain?")


@pytest.fixture(scope="module")
def asked_for_a_verdict() -> tuple[CampaignState, list[str]]:
    return campaign("Is quantum hardware worth it for a 10-spin critical Ising chain?")


# Each question takes its own route
# --------------------------------------------------------------------------


def test_a_request_for_code_reaches_the_drafting_node(
    asked_for_code: tuple[CampaignState, list[str]],
) -> None:
    state, route = asked_for_code
    assert state["intent"].intent == "implement"
    assert "implement" in route
    assert "explain" not in route


def test_a_request_for_an_explanation_reaches_the_prose_node(
    asked_for_prose: tuple[CampaignState, list[str]],
) -> None:
    state, route = asked_for_prose
    assert state["intent"].intent == "explain"
    assert "explain" in route
    assert "implement" not in route


def test_the_feasibility_question_still_runs_the_whole_campaign(
    asked_for_a_verdict: tuple[CampaignState, list[str]],
) -> None:
    # The branch must not have cost the application the thing it is for.
    state, route = asked_for_a_verdict
    assert state["intent"].intent == "feasibility"
    assert {"baseline", "plan", "solve", "analyse", "skeptic"} <= set(route)
    assert state["verdict"] is not None
    assert state["classical"] is not None


def test_the_baseline_and_the_planner_still_start_together(
    asked_for_a_verdict: tuple[CampaignState, list[str]],
) -> None:
    # Putting the fan-out behind a condition is exactly the change that could have
    # quietly serialised it, and nothing else in the suite would have noticed.
    _, route = asked_for_a_verdict
    assert route.index("retrieve") < route.index("baseline")
    assert route.index("retrieve") < route.index("plan")


# What the other branches must not do
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", ["asked_for_code", "asked_for_prose"])
def test_a_branch_that_ran_nothing_reaches_no_verdict(
    fixture: str, request: pytest.FixtureRequest
) -> None:
    # The whole point. A verdict here would be a confident "quantum hardware is not
    # worth it" attached to an answer about something else, decided by a rule that
    # read a chain nobody asked about from a campaign that ran no circuit.
    state, _ = request.getfixturevalue(fixture)
    assert state["verdict"] is None


@pytest.mark.parametrize("fixture", ["asked_for_code", "asked_for_prose"])
def test_a_branch_that_ran_nothing_spends_no_measurements(
    fixture: str, request: pytest.FixtureRequest
) -> None:
    state, route = request.getfixturevalue(fixture)
    assert state["shots"].spent == 0
    assert state["runs"] == ()
    assert "solve" not in route


@pytest.mark.parametrize("fixture", ["asked_for_code", "asked_for_prose"])
def test_a_branch_that_ran_nothing_shows_no_energy_table(
    fixture: str, request: pytest.FixtureRequest
) -> None:
    # The visible half of the same failure: the old reply printed an energy table
    # under a question about barren plateaus.
    state, _ = request.getfixturevalue(fixture)
    report = state["report"] or ""
    assert "Energy per spin" not in report
    assert "Feasibility assessment" not in report


@pytest.mark.parametrize("fixture", ["asked_for_code", "asked_for_prose"])
def test_a_branch_that_ran_nothing_skips_the_skeptic(
    fixture: str, request: pytest.FixtureRequest
) -> None:
    # There is no conclusion to argue against, and running the skeptic anyway would
    # put a rebuttal of a verdict nobody reached into the trail.
    _, route = request.getfixturevalue(fixture)
    assert "skeptic" not in route


# Every branch still finishes the same way
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", ["asked_for_code", "asked_for_prose", "asked_for_a_verdict"])
def test_every_branch_writes_a_document_and_offers_a_next_question(
    fixture: str, request: pytest.FixtureRequest
) -> None:
    state, route = request.getfixturevalue(fixture)
    assert (state["report"] or "").strip()
    assert route[-3:] == ["scribe", "suggest", "remember"]


@pytest.mark.parametrize("fixture", ["asked_for_code", "asked_for_prose", "asked_for_a_verdict"])
def test_every_branch_records_how_it_was_routed(
    fixture: str, request: pytest.FixtureRequest
) -> None:
    # A trail that does not say which branch was taken cannot be audited for having
    # taken the wrong one, which is the failure this whole change is about.
    state, _ = request.getfixturevalue(fixture)
    assert any("read the question as asking for" in note for note in state["notes"])


def test_the_code_branch_says_why_there_is_no_code_when_offline(
    asked_for_code: tuple[CampaignState, list[str]],
) -> None:
    # Offline there is deliberately no deterministic path here: a templated program
    # would be code nobody checked, dressed as code somebody wrote. So the branch
    # has to say which of the three reasons applied, or a reader assumes the feature
    # is broken.
    state, _ = asked_for_code
    draft = state["draft"]
    assert draft is not None
    assert not draft.written
    assert "no language model" in draft.note


def test_the_prose_branch_answers_from_the_notes_when_offline(
    asked_for_prose: tuple[CampaignState, list[str]],
) -> None:
    # The offline path is not a stub. Retrieval ran and found passages, and handing
    # a reader the passages with their sources is a real answer -- throwing them away
    # to print "no model available" would discard the half of the work that needed
    # no key.
    state, _ = asked_for_prose
    answer = state["answer"]
    assert answer is not None
    assert answer.written
    assert answer.cited > 0
    assert answer.written_by == "notes"


def test_a_source_that_returned_several_passages_is_listed_once(
    asked_for_prose: tuple[CampaignState, list[str]],
) -> None:
    # Retrieval works in chunks, so a document that answers the question well comes
    # back several times. Printing each one showed the same title four times, which
    # reads as four independent sources agreeing.
    state, _ = asked_for_prose
    report = state["report"] or ""
    listed = [line for line in report.splitlines() if line.strip().startswith(("1.", "2.", "3."))]
    assert len(listed) == len({line.split(". ", 1)[-1] for line in listed})


# The budget the ladder is priced against
# --------------------------------------------------------------------------


def test_the_library_and_the_interface_agree_on_the_default_budget() -> None:
    # These were fifty million and a billion, and the smaller one covered a six-spin
    # chain: every rung of the ladder for the ten-spin chain this application uses as
    # its own headline example was refused before it ran. A campaign run from the
    # command line and the same campaign run from the interface returned different
    # verdicts, and nothing failed.
    from src.agent.campaign import DEFAULT_SHOT_BUDGET as COMMAND_LINE_BUDGET
    from src.agent.graph import DEFAULT_SHOT_BUDGET
    from src.ui.setting import DEFAULT_SHOTS

    assert DEFAULT_SHOT_BUDGET == DEFAULT_SHOTS
    # The command line is the third holder of this number and was missing from the
    # assertion above, which is how it kept the old fifty million after the other two
    # were raised. ``make campaign`` then refused the first rung of its own worked
    # example -- it needed ninety million for one layer -- and reported a confident
    # "no" with nothing run. Naming all three here is the difference between an
    # invariant and a note.
    assert COMMAND_LINE_BUDGET == DEFAULT_SHOT_BUDGET


def test_the_default_budget_reaches_the_first_rung_of_the_headline_chain() -> None:
    # The claim the old docstring made and the arithmetic did not support. Stated
    # against the chain the application demonstrates itself with, so that a change
    # to the pricing which quietly puts the demonstration out of reach fails here.
    state, _ = campaign("Is quantum hardware worth it for a 10-spin critical Ising chain?")
    assert state["runs"], "the flagship question ran no configuration at the shipped budget"


# What a branch may claim about the chain
# --------------------------------------------------------------------------


def test_a_conceptual_question_is_not_answered_about_an_invented_chain() -> None:
    # Formalising always produces a model, filling what the question left out from
    # defaults. That is right for the feasibility branch, where the fills are printed
    # above the verdict. It is wrong here: asked what a barren plateau is, the reader
    # invented a two-spin chain with the transverse field at zero -- which is not a
    # transverse-field Ising chain at all -- and the explanation came back describing
    # that system by name. The assumptions list that would have caught it is not shown
    # beside a prose answer.
    from src.agent.graph import _chain_the_question_named

    state, _ = campaign("What is a barren plateau?")
    assert state["model"] is not None, "formalise still runs; only what it is used for changed"
    assert _chain_the_question_named(state) is None


def test_a_question_that_names_a_chain_is_answered_about_that_chain() -> None:
    # The other end of the same rule, so the fix cannot have been "never mention a
    # chain", which would be a different answer to the wrong question.
    from src.agent.graph import _chain_the_question_named

    state, _ = campaign("Explain the barren plateau problem for a 12-spin chain")
    named = _chain_the_question_named(state)
    assert named is not None
    assert named.n_sites == 12


def test_the_code_answer_puts_its_equations_above_the_program() -> None:
    # The order is the point of the feature. A reader who has already scrolled past
    # eighty lines of Python has stopped asking what the program is for, and the
    # equations are the half of this answer that was composed rather than generated.
    state = new_campaign(
        request=Request(text="Write the QAOA circuit for 8 spins at depth 3"), shot_budget=10**9
    )
    document = graph._code_document(
        state,
        Draft(code="print(1)", mathematics="$$\n\\lvert\\psi\\rangle\n$$"),
    )
    assert document.index("lvert") < document.index("```python")


def test_the_equations_describe_the_circuit_the_question_named() -> None:
    # Read from the question rather than from the settings knob whenever the question
    # said. Eight magnets at depth three is one circuit, and the picture under the
    # answer and the algebra above the code are drawings of the same one.
    state = new_campaign(
        request=Request(text="Can you write the QAOA circuit for 8 spins at depth 3?"),
        shot_budget=10**9,
    )
    spec = graph.circuit_the_question_named(state)
    assert spec is not None
    assert (spec.n_qubits, spec.depth, spec.family) == (8, 3, "qaoa")


def test_a_question_naming_no_depth_takes_the_one_the_interface_is_set_to() -> None:
    # Two readings of the same question is two chances to pick a different depth,
    # which would put a picture of one circuit beside the algebra of another.
    state = new_campaign(
        request=Request(text="Write me a QAOA circuit for an 8-spin chain"), shot_budget=10**9
    )
    assert graph.circuit_the_question_named(state, 5) is not None
    assert graph.circuit_the_question_named(state, 5).depth == 5  # type: ignore[union-attr]


# A question about a curve in the field
# --------------------------------------------------------------------------

SWEEP_QUESTION = (
    "Can you plot low lying spectrum of quantum Ising as a function of an external "
    "field using free fermion approach?"
)
# Verbatim from a real session, and the question that produced the defect: it
# contains the word "plot", so the graph raced three variational methods on a chain
# nobody had named and returned a table of three agreeing energies. The exact
# spectrum -- which this project can compute in closed form -- appeared nowhere.


def swept(question: str, lend: bool = True) -> tuple[CampaignState, list[str]]:
    """Run one offline campaign with the exact sweep lent to it.

    Args:
        question: What to ask.
        lend: Whether the host lends the sealed solver, as the interface does.

    Returns:
        The finished state and the route it took.
    """
    from src.physics.registry import field_sweep_bench

    visited: list[str] = []
    state = run_campaign(
        question,
        shot_budget=BUDGET,
        chat_model=None,
        search_corpus=False,
        fetch_external=False,
        suggest_followups=False,
        reference_bench=field_sweep_bench() if lend else None,
        visited=visited,
    )
    return state, visited


@pytest.fixture(scope="module")
def asked_for_a_sweep() -> tuple[CampaignState, list[str]]:
    return swept(SWEEP_QUESTION)


def test_a_question_about_a_curve_in_the_field_is_read_as_an_explanation(
    asked_for_a_sweep: tuple[CampaignState, list[str]],
) -> None:
    state, _ = asked_for_a_sweep
    # It used to read as feasibility: the sentence is dense in this application's own
    # vocabulary -- field, energy, ground, transverse, chain -- so the word scores
    # favoured a go/no-go verdict on quantum hardware over the curve that was asked
    # for. No margin rule recovers that from counting nouns; reading the axis does.
    assert state["intent"].intent == "explain"


def test_a_question_about_a_curve_in_the_field_does_not_race_the_methods(
    asked_for_a_sweep: tuple[CampaignState, list[str]],
) -> None:
    _, visited = asked_for_a_sweep
    assert "converge" not in visited
    assert "consult" in visited


def test_a_question_about_an_optimisers_descent_still_races_the_methods() -> None:
    _, visited = swept("Plot the energy against optimisation epoch for VQE and QAOA")
    assert "converge" in visited


def test_the_exact_curve_is_fetched_and_recorded(
    asked_for_a_sweep: tuple[CampaignState, list[str]],
) -> None:
    state, _ = asked_for_a_sweep
    calls = [call for call in state["tool_calls"] if call.name == "exact_field_sweep"]
    assert len(calls) == 1
    call = calls[0]
    assert not call.failed
    # The curve the sentence asked for, not a default one.
    assert call.arguments["curves"] == ["spectrum"]
    # And the result carries the check that makes it evidence rather than output.
    assert "pfeuty_exact" in call.result
    assert "agree to" in call.result


def test_the_answer_is_reached_with_no_verdict_and_no_measurements(
    asked_for_a_sweep: tuple[CampaignState, list[str]],
) -> None:
    state, _ = asked_for_a_sweep
    # A curve is not a feasibility statement. The sweep spends no shots, designs no
    # circuit and must leave the verdict alone.
    assert state["verdict"] is None
    assert state["runs"] == ()
    assert state["shots"].spent == 0


def test_nothing_is_fetched_when_the_host_lends_nothing() -> None:
    state, visited = swept(SWEEP_QUESTION, lend=False)
    # The tool does not exist when no bench was lent -- there is no disabled stub to
    # reason about -- so the branch answers from the notes and says so.
    assert "consult" in visited
    assert all(call.name != "exact_field_sweep" for call in state["tool_calls"])
    assert state["answer"] is not None


def test_the_trail_says_the_curves_were_chosen_by_rule_rather_than_by_a_model(
    asked_for_a_sweep: tuple[CampaignState, list[str]],
) -> None:
    state, _ = asked_for_a_sweep
    # Offline there is no model to choose the arguments, so they are read out of the
    # sentence. A worse choice, and it has to be labelled as one.
    assert any("chosen by rule" in note for note in state["notes"])


def test_a_segment_is_refused_rather_than_swept_as_a_ring() -> None:
    # The closed form needs a ring, and substituting one for the chain the question
    # named would draw the right curve for a different problem under the reader's own
    # words. The refusal names the reason instead.
    state, _ = swept("Plot the magnetisation of an open 8-spin chain against h/J")
    calls = [call for call in state["tool_calls"] if call.name == "exact_field_sweep"]
    assert len(calls) == 1
    assert calls[0].failed
    assert "does not apply" in calls[0].result


def test_a_curve_is_fetched_even_when_the_model_calls_nothing() -> None:
    # The live defect this closes: *Plot the exact energy levels as the field is
    # turned up* came back fluent and correct -- the dispersion, the elliptic
    # integral, the logarithmic divergence -- with no figure anywhere on the page,
    # closing on "can be computed and plotted". The tool was offered and the model
    # did not call it, and the figure is drawn from a recorded call.
    from src.agent.graph import SWEEP_TOOL_NAME, _with_the_curve_that_was_asked_for
    from src.agent.state import Request, new_campaign
    from src.physics.registry import field_sweep_bench

    state = new_campaign(
        Request(text="Can you plot the exact energy levels as the field is turned up?"),
        shot_budget=BUDGET,
    )
    config: RunnableConfig = {"configurable": {"reference_bench": field_sweep_bench()}}
    filled = _with_the_curve_that_was_asked_for(state, config, ())
    assert [call.name for call in filled] == [SWEEP_TOOL_NAME]
    assert filled[0].arguments["curves"] == ["spectrum"]


def test_the_models_own_call_is_not_duplicated() -> None:
    # A model that called the sweep itself keeps its own arguments, which is the
    # whole point of giving it the tool: it may reasonably ask for a longer chain or
    # a wider range than the sentence names.
    from src.agent.graph import SWEEP_TOOL_NAME, _with_the_curve_that_was_asked_for
    from src.agent.llm import ToolCall
    from src.agent.state import Request, new_campaign
    from src.physics.registry import field_sweep_bench

    state = new_campaign(Request(text="plot the spectrum against h/J"), shot_budget=BUDGET)
    config: RunnableConfig = {"configurable": {"reference_bench": field_sweep_bench()}}
    theirs = ToolCall(name=SWEEP_TOOL_NAME, arguments={"n_sites": 12}, result="{}")
    assert _with_the_curve_that_was_asked_for(state, config, (theirs,)) == (theirs,)


def test_a_question_that_asked_for_no_curve_gets_none_added() -> None:
    from src.agent.graph import _with_the_curve_that_was_asked_for
    from src.agent.state import Request, new_campaign
    from src.physics.registry import field_sweep_bench

    state = new_campaign(
        Request(text="How does the Jordan-Wigner transformation work?"), shot_budget=BUDGET
    )
    config: RunnableConfig = {"configurable": {"reference_bench": field_sweep_bench()}}
    assert _with_the_curve_that_was_asked_for(state, config, ()) == ()


def test_the_tool_name_the_graph_uses_is_the_tools_own() -> None:
    # Two copies of a name is one name that will eventually be wrong, and the failure
    # would be silent: the fallback would stop recognising the model's own call and
    # every curve question would fetch the sweep twice.
    from src.agent.graph import SWEEP_TOOL_NAME
    from src.agent.tools import field_sweep_tool
    from src.physics.registry import field_sweep_bench

    assert field_sweep_tool(field_sweep_bench()).name == SWEEP_TOOL_NAME


# --------------------------------------------------------------------------
# Offline because the configuration cannot be read, not because a flag says so
# --------------------------------------------------------------------------


@pytest.fixture
def unreadable_configuration(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Make :class:`src.settings.Settings` refuse to build, as a keyless machine does.

    Every other test in this file is "offline" in the sense of being handed no chat
    model, which is not the same thing and is why the defect below survived: the
    configuration was always readable, so nothing ever exercised the path a fresh
    checkout with no credential takes.

    Yields:
        Nothing. The cache is cleared on the way in and on the way out, because it
        is process-wide and a stale entry either way would leak into other tests.
    """
    get_settings.cache_clear()
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    try:
        yield
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()


def test_a_campaign_completes_when_there_is_no_credential(
    unreadable_configuration: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The deterministic fallbacks are documented, tested and were unreachable. Thirty
    # of the thirty-one places that read configuration did so eagerly, and two of them
    # sat inside graph nodes -- `slug_for` in `screen`, `promoted_root` in `retrieve` --
    # so a missing key ended the campaign with a validation error two nodes in rather
    # than taking the path the fallbacks exist to provide.
    #
    # External fetching is on precisely because it is off everywhere else in this
    # file, and the second of those two crashes lives behind it. The network call it
    # guards is stubbed out: what this test is about is `promoted_root` being read on
    # that path, which happens whether or not anything comes back, and a test that
    # reaches arXiv passes or fails on someone else's uptime.
    monkeypatch.setattr("src.rag.external.fetch", lambda *_args, **_kwargs: ())

    with pytest.raises(ValidationError):
        get_settings()

    visited: list[str] = []
    state = run_campaign(
        "Is a quantum computer worth it for a 10-site chain with a tilt of 0.4?",
        shot_budget=BUDGET,
        device=device_for("heavy-hex-27"),
        chat_model=None,
        search_corpus=True,
        fetch_external=True,
        visited=visited,
    )
    verdict = state["verdict"]
    assert verdict is not None
    assert verdict.summary
    # Past both crash sites and all the way to the end: `screen` is where `slug_for`
    # ran, `retrieve` is where `promoted_root` did, and `scribe` is the node that
    # composes the report a reader is handed.
    assert visited[:5] == ["recall", "screen", "interpret", "formalise", "retrieve"]
    assert {"scribe", "remember"} <= set(visited)
