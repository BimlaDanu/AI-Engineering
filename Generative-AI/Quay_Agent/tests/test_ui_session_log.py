"""The session ledger: what it derives, and the two things it refuses to conflate.

Every function here is a pure reduction over a tuple of records, which is why none
of these tests drives an interface. That separation is the point of the module: a
chart built from a counter kept beside the conversation is a chart that can drift
out of step with it, and the drift is invisible until somebody checks.

Two distinctions are load-bearing and are each defended by a test. **A branch is
read off the nodes that ran, not off what the router decided**, because the two
come apart on exactly the runs worth looking at. And **a node revisited inside one
question is not a second question**, because mixing the two would let a single
question with a three-attempt planning loop outweigh three questions.
"""

from __future__ import annotations

from src.agent.state import Answer, Reading, Request, Verdict, new_campaign
from src.agent.usage import Call, Usage
from src.ui.session_log import (
    Asked,
    branch_counts,
    branch_of,
    calls_by_model,
    node_counts,
    record,
    retry_counts,
    timing_by_task,
    totals,
)


def usage(*purposes: tuple[str, str, float]) -> Usage:
    """A metered run, assembled from (purpose, model, seconds) triples.

    Args:
        *purposes: One triple per call.

    Returns:
        The usage.
    """
    return Usage(
        calls=tuple(
            Call(
                model=model,
                purpose=purpose,
                prompt_tokens=100,
                completion_tokens=20,
                reported=True,
                cost_usd=0.001,
                latency_ms=seconds * 1000,
            )
            for purpose, model, seconds in purposes
        )
    )


def asked(**overrides: object) -> Asked:
    """One session row, with sensible defaults.

    Args:
        **overrides: Fields to replace.

    Returns:
        The row.
    """
    base: dict[str, object] = {"question": "a question", "branch": "explain"}
    base.update(overrides)
    return Asked(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Which arm ran
# --------------------------------------------------------------------------


def test_a_run_that_priced_circuits_is_a_feasibility_run() -> None:
    assert branch_of(["screen", "interpret", "formalise", "baseline", "plan", "solve"]) == (
        "feasibility"
    )


def test_a_run_that_only_wrote_prose_is_an_explanation() -> None:
    assert branch_of(["screen", "interpret", "retrieve", "explain"]) == "explain"


def test_a_run_that_reached_no_arm_says_so_rather_than_guessing() -> None:
    # A question screened out before anything ran. Reporting it as an explanation
    # would credit the pipeline with work it did not do.
    assert branch_of(["screen"]) == "none"


def test_a_feasibility_run_that_also_wrote_prose_is_still_a_feasibility_run() -> None:
    # The graph permits both. "It explained something" is the weaker description of
    # a run that also priced circuits, so the more specific arm wins.
    assert branch_of(["screen", "formalise", "baseline", "solve", "explain"]) == "feasibility"


def test_the_branch_comes_from_the_path_and_not_from_the_router() -> None:
    # The router said "feasibility". Nothing that prices a circuit ran, because the
    # formalisation failed. The chart must show what happened.
    state = new_campaign(request=Request(text="q"), shot_budget=1)
    state["intent"] = Reading(intent="feasibility", decided_by="model", reason="asked for a call")
    row = record("q", state, ["screen", "interpret", "formalise"], Usage(), 1.0)
    assert row.decided_by == "model"
    assert row.branch == "none"


# --------------------------------------------------------------------------
# What one row records
# --------------------------------------------------------------------------


def test_a_row_carries_the_verdict_and_who_wrote_the_prose() -> None:
    state = new_campaign(request=Request(text="q"), shot_budget=1)
    state["verdict"] = Verdict(call="no", summary="s", crossover_condition="c")
    state["answer"] = Answer(text="words", written_by="model", cited=4)
    row = record("q", state, ["screen", "explain"], Usage(), 2.0)
    assert row.verdict == "no"
    assert row.written_by == "model"
    assert row.cited == 4
    assert not row.refused


def test_a_refusal_is_recorded_as_one() -> None:
    # A refusal drawn as an ordinary answer would undo the honesty it exists for,
    # and the analytics must be able to count them.
    state = new_campaign(request=Request(text="q"), shot_budget=1)
    state["answer"] = Answer(text="the notes hold nothing", written_by="refused")
    row = record("q", state, ["screen", "explain"], Usage(), 1.0)
    assert row.refused


def test_a_row_with_no_calls_is_an_offline_answer_rather_than_a_missing_one() -> None:
    assert asked().offline
    assert not asked(usage=usage(("explanation", "vendor/m", 1.0))).offline


# --------------------------------------------------------------------------
# The reductions
# --------------------------------------------------------------------------


def test_the_session_total_is_derived_and_not_accumulated() -> None:
    rows = [
        asked(usage=usage(("explanation", "a", 1.0))),
        asked(usage=usage(("report_writing", "b", 2.0), ("shelf_choice", "a", 0.1))),
    ]
    combined = totals(rows)
    assert combined.calls_made == 3
    assert combined.total_tokens == 3 * 120


def test_an_empty_session_totals_to_nothing_rather_than_failing() -> None:
    assert totals([]).calls_made == 0


def test_branches_are_counted_commonest_first() -> None:
    rows = [asked(branch="explain"), asked(branch="feasibility"), asked(branch="explain")]
    assert list(branch_counts(rows)) == ["explain", "feasibility"]


def test_nodes_are_reported_in_the_graphs_own_order() -> None:
    # A hand-kept order goes stale silently, and the symptom is a chart whose bars
    # are in the wrong sequence rather than a failure.
    rows = [asked(path=("solve", "screen"))]
    assert list(node_counts(rows, ("screen", "plan", "solve"))) == ["screen", "solve"]


def test_a_node_nobody_reached_is_dropped_rather_than_charted_as_zero() -> None:
    rows = [asked(path=("screen",))]
    assert "plan" not in node_counts(rows, ("screen", "plan", "solve"))


def test_a_node_revisited_inside_one_question_counts_once_for_that_question() -> None:
    # Otherwise a single question with a three-attempt planning loop outweighs
    # three questions, and the chart stops being about how questions were served.
    rows = [asked(path=("plan", "solve", "plan", "solve", "plan"))]
    assert node_counts(rows, ("plan", "solve"))["plan"] == 1


def test_revisits_are_counted_separately_as_the_repair_loop() -> None:
    rows = [asked(path=("plan", "solve", "plan", "solve", "plan"))]
    assert retry_counts(rows) == {"plan": 2, "solve": 1}


def test_a_session_that_never_looped_reports_no_retries() -> None:
    assert retry_counts([asked(path=("screen", "explain"))]) == {}


def test_timing_is_broken_down_by_task_with_its_tier_beside_it() -> None:
    # The tier design's own measurement: if the fast tier is not fast, the tiering
    # bought nothing, and this is where that would show.
    rows = [asked(usage=usage(("shelf_choice", "a", 0.2), ("report_writing", "b", 4.0)))]
    timings = timing_by_task(rows)
    assert [timing.task for timing in timings] == ["report_writing", "shelf_choice"]
    assert timings[0].tier == "strong"
    assert timings[1].tier == "fast"
    assert timings[0].mean_seconds == 4.0


def test_a_task_this_module_has_never_heard_of_gets_a_row_and_no_tier() -> None:
    # A call site can name a purpose before this mapping hears about it, and a row
    # with no tier is better than a lookup that raises inside a chart.
    rows = [asked(usage=usage(("something_new", "a", 0.5)))]
    assert timing_by_task(rows)[0].tier == "—"


def test_a_call_with_no_recorded_latency_counts_but_does_not_drag_the_mean_down() -> None:
    untimed = Usage(
        calls=(
            Call(
                model="a",
                purpose="explanation",
                prompt_tokens=1,
                completion_tokens=1,
                reported=True,
                cost_usd=0.0,
                latency_ms=None,
            ),
        )
    )
    timings = timing_by_task([asked(usage=untimed)])
    assert timings[0].calls == 1
    assert timings[0].seconds == 0.0


def test_calls_are_attributed_to_the_model_that_served_them() -> None:
    # The check on the tier configuration: one slug serving every call means a
    # deployment pointed all three tiers at one model, meant or not.
    rows = [asked(usage=usage(("shelf_choice", "cheap", 0.1), ("report_writing", "cheap", 1.0)))]
    assert calls_by_model(rows) == {"cheap": 2}
