"""Tests for the evaluation suite.

Two different things are under test here and they are worth keeping apart.

The **grader** is tested against constructed answers, so that "this case fails when
the run refuses" is verified without running an agent at all. A grader nobody checks
is the one component whose bugs make a project look better than it is.

The **suite** is then run for real, offline, and asserted to pass. That makes the
scorecard a regression test rather than a report: a change that breaks the routing
or drops a caveat turns this file red before anybody reads a Markdown file.

An autouse fixture makes building a chat model raise, so no test here can reach the
gateway with the key in ``.env``. That is also what the offline scorecard measures,
which is the point: the deterministic path is a supported way to run this
application, not a degraded one.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr

from src.agent.graph import Answer, ask
from src.agent.router import Guard, RouteChoice, Routing
from src.evals import tracking
from src.evals.cases import CASES, Case, chain
from src.evals.run import run_suite, validate, write_report
from src.evals.scoring import Scorecard, grade
from src.security import screen
from src.settings import Settings, get_settings


@pytest.fixture(autouse=True)
def no_live_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make an accidental live model call impossible for the whole module."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("a test tried to build a real chat model")

    monkeypatch.setattr("src.agent.llm.build_chat_model", refuse)


def answer(status: str = "answered", **overrides: object) -> Answer:
    """An answer the agent might have produced, without running it."""
    values: dict[str, object] = {
        "question": "What is the ground-state energy?",
        "status": status,
        "text": "It is -7.727406610.",
        "caveats": (),
        "spec": None,
        "guard": Guard(screening=screen("safe"), verdict=None, second_opinion="unavailable"),
        "routing": None,
        "plan": None,
        "verification": None,
        "retrieval": None,
    }
    values.update(overrides)
    return Answer(**values)  # type: ignore[arg-type]


# --- the suite is well formed ----------------------------------------------


def test_the_shipped_suite_is_valid() -> None:
    assert validate() == ()


def test_a_case_that_asserts_nothing_is_rejected() -> None:
    # It would pass every run and report success, which is worse than no case.
    empty = Case(name="vacuous", family="routing", question="anything?", why="nothing")
    assert "asserts nothing" in validate((empty,))[0]


def test_a_case_with_no_stated_purpose_is_rejected() -> None:
    quiet = Case(
        name="unexplained",
        family="routing",
        question="anything?",
        why="   ",
        expect_status="answered",
    )
    assert "does not say what it defends" in validate((quiet,))[0]


def test_duplicate_case_names_are_rejected() -> None:
    # Two rows with one label: the scorecard would show one of them and hide the
    # other, which is how a failing case disappears.
    one = Case(
        name="same",
        family="routing",
        question="q",
        why="w",
        expect_status="answered",
    )
    assert "duplicate case name" in validate((one, one))[0]


def test_every_case_names_a_family_the_scorecard_groups() -> None:
    families = {case.family for case in CASES}
    assert families == {
        "routing",
        "refusal",
        "verification",
        "exact limits",
        "memory",
        "knowledge",
    }


# --- the grader -------------------------------------------------------------


def test_a_wrong_status_fails_and_says_what_happened() -> None:
    case = Case(name="c", family="routing", question="q", why="w", expect_status="answered")
    result = grade(case, answer("refused"))
    assert not result.passed
    assert result.failures[0].detail == "ended refused"


def test_a_forbidden_node_fails_the_case() -> None:
    # "A numeric question must not search the corpus" is a claim about the path, and
    # the path is read off the answer's own fields.
    case = Case(
        name="c",
        family="routing",
        question="q",
        why="w",
        forbid_nodes=("remember",),
    )
    assert not grade(case, answer()).passed


def test_a_shelf_the_case_requires_must_be_among_those_picked() -> None:
    case = Case(
        name="c",
        family="knowledge",
        question="q",
        why="w",
        require_shelves=("applications",),
    )
    chose = Routing(
        choice=RouteChoice(
            route="retrieve",
            reason="because",
            topics=[],
            shelves=["physics-notes"],
            confidence=0.9,
        ),
        decided_by="heuristic",
    )
    assert not grade(case, answer(routing=chose)).passed


def test_a_second_relevant_shelf_does_not_fail_a_required_shelf_case() -> None:
    # The point of `require_shelves`: the case names the shelf that must answer, and a
    # router that also read a related one behaved better than the case asked. Demanding
    # equality there is how an expectation goes stale.
    case = Case(
        name="c",
        family="knowledge",
        question="q",
        why="w",
        require_shelves=("applications",),
    )
    chose = Routing(
        choice=RouteChoice(
            route="retrieve",
            reason="because",
            topics=[],
            shelves=["quantum-computing", "applications"],
            confidence=0.9,
        ),
        decided_by="heuristic",
    )
    assert grade(case, answer(routing=chose)).passed


def test_a_shelf_the_case_forbids_fails_the_case_if_it_was_picked() -> None:
    # "Must not be narrowed onto the applications shelf" is the claim; the case says
    # nothing about which of the other two answered, because both are defensible.
    case = Case(
        name="c",
        family="knowledge",
        question="q",
        why="w",
        forbid_shelves=("applications",),
    )
    picked = ["quantum-computing", "applications"]
    chose = Routing(
        choice=RouteChoice(
            route="retrieve",
            reason="because",
            topics=[],
            shelves=picked,
            confidence=0.9,
        ),
        decided_by="heuristic",
    )
    assert not grade(case, answer(routing=chose)).passed
    allowed = Routing(
        choice=RouteChoice(
            route="retrieve",
            reason="because",
            topics=[],
            shelves=["quantum-computing"],
            confidence=0.9,
        ),
        decided_by="heuristic",
    )
    assert grade(case, answer(routing=allowed)).passed


def test_an_uncorroborated_number_fails_a_verification_case() -> None:
    case = Case(
        name="c",
        family="verification",
        question="q",
        why="w",
        expect_verified=True,
    )
    assert not grade(case, answer()).passed


def test_an_unverified_number_must_admit_it() -> None:
    # The half a fluent agent fails: the number may be unverified, and it may not be
    # unverified in silence.
    case = Case(
        name="c",
        family="verification",
        question="q",
        why="w",
        expect_unverified_notice=True,
    )
    assert not grade(case, answer()).passed
    spoken = answer(caveats=("Only one method applies, so the number is unverified.",))
    assert grade(case, spoken).passed


def test_a_missing_follow_up_is_a_failure_not_an_exception() -> None:
    # A runner that could not finish a case should produce a red row, not a stack
    # trace that hides the rows after it.
    case = Case(
        name="c",
        family="memory",
        question="q",
        why="w",
        follow_up="and?",
        follow_up_expect_status="answered",
    )
    result = grade(case, answer(), None)
    assert not result.passed
    assert "the follow-up was asked" in result.failures[0].claim


def test_a_number_is_checked_against_arithmetic() -> None:
    case = Case(
        name="c",
        family="exact limits",
        question="q",
        why="w",
        expect_energy=-6.0,
    )
    assert not grade(case, answer()).passed  # no energy was produced at all


# --- the scorecard ----------------------------------------------------------


def test_a_scorecard_with_no_calls_says_no_model_answered() -> None:
    # Measured, not inferred from configuration: a configured key whose gateway is
    # unreachable produces a complete run on the deterministic path.
    card = Scorecard(results=(), model="anthropic/claude-haiku-4.5", model_calls=0)
    assert "No model was called" in card.provenance()


def test_a_scorecard_that_called_a_model_names_it() -> None:
    card = Scorecard(results=(), model="openai/gpt-4o-mini", model_calls=7)
    assert "gpt-4o-mini" in card.provenance()
    assert "7 calls" in card.provenance()


def test_adding_a_note_cannot_change_what_the_card_says_about_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This shipped: `publish` rebuilt the scorecard field by field, dropped
    # `model_calls` to its default of zero, and a suite that had just made about a
    # hundred live calls reported "No model was called: every answer here came from
    # the deterministic path". A note is an annotation, never a rewrite.
    def refuse(card: Scorecard, settings: Settings) -> str:
        raise RuntimeError("no network in a test")

    monkeypatch.setattr(tracking, "_upload", refuse)
    configured = Settings(  # type: ignore[call-arg]
        _env_file=None,
        # Both credentials are passed explicitly, like every other constructed
        # Settings in this suite. `_env_file=None` shuts out `.env` but not the
        # process environment, so leaving the required key to be filled from
        # `OPENROUTER_API_KEY` made this the one test in the suite that passed
        # only on a machine that happened to export one -- red on a fresh
        # checkout, and a contradiction of the claim that `make check` needs no
        # credential. Nothing here reaches a gateway; the value is a placeholder.
        openrouter_api_key=SecretStr("not-a-real-key"),
        langsmith_api_key=SecretStr("not-a-real-key"),
    )
    card = Scorecard(results=(), model="openai/gpt-4o-mini", model_calls=97)
    filed = tracking.publish(card, settings=configured)
    assert "was not filed" in filed.notes[-1]
    assert filed.model_calls == 97
    assert "97 calls" in filed.provenance()


def test_failures_are_spelled_out_in_the_report() -> None:
    # A red row whose meaning has to be reverse-engineered gets deleted rather than
    # fixed, so the report carries the case's own statement of what it defends.
    case = Case(
        name="c",
        family="routing",
        question="q",
        why="the reason it exists",
        expect_status="answered",
    )
    text = Scorecard(results=(grade(case, answer("refused")),)).markdown()
    assert "## Failures" in text
    assert "the reason it exists" in text
    assert "ended refused" in text


def test_a_clean_report_says_so_rather_than_leaving_a_blank() -> None:
    case = Case(
        name="c",
        family="routing",
        question="q",
        why="w",
        expect_status="answered",
    )
    assert "Every case passed." in Scorecard(results=(grade(case, answer()),)).markdown()


# --- the whole suite, offline ----------------------------------------------


_OFFLINE_CARD: Scorecard | None = None


@pytest.fixture
def offline_card() -> Scorecard:
    """The whole suite, run once and shared by every test that reads it.

    Running it is the most expensive thing in this file -- every case through the
    real graph -- and it is deterministic on the offline path, so three tests
    asking for it separately paid three times for one answer. Memoised here
    rather than declared module-scoped on purpose: a higher-scoped fixture would
    be built *before* the autouse guard that makes a live model call impossible,
    and the suite would quietly bill the gateway.
    """
    global _OFFLINE_CARD
    if _OFFLINE_CARD is None:
        _OFFLINE_CARD = run_suite()
    return _OFFLINE_CARD


@pytest.mark.slow
def test_the_agent_passes_its_own_suite(offline_card: Scorecard) -> None:
    # The regression test that matters: routing, refusals, verification, the two
    # exact limits and memory, all on the deterministic path.
    #
    # The one part of the suite that a credential buys. Every model call here is
    # stubbed, but retrieval still **embeds its queries through the gateway**, and
    # with no key the store is unreachable: the two knowledge cases find nothing,
    # refuse honestly, and the suite scores 24 of 26. That is the agent behaving
    # correctly, so asserting 26 would be asserting the network. Skipped rather than
    # relaxed, because a suite that quietly accepts 24 stops being a regression test.
    try:
        get_settings()
    except Exception:  # pragma: no cover - a checkout with no credential
        pytest.skip("no credential: retrieval cannot embed, so the knowledge cases refuse")
    assert offline_card.notes == ()
    assert offline_card.passed == offline_card.total, offline_card.markdown()


@pytest.mark.slow
def test_the_offline_suite_calls_no_model(offline_card: Scorecard) -> None:
    assert offline_card.model_calls == 0


@pytest.mark.slow
def test_the_report_is_written_in_both_forms(offline_card: Scorecard, tmp_path: Path) -> None:
    # The Markdown is for a person and the JSON is what the Evaluation page reads;
    # the page must never have to parse prose that exists to be rewritten.
    report, summary = tmp_path / "card.md", tmp_path / "card.json"
    write_report(offline_card, report, summary)
    assert "# Evaluation scorecard" in report.read_text(encoding="utf-8")
    assert '"families"' in summary.read_text(encoding="utf-8")


@pytest.mark.slow
def test_running_the_cases_at_once_changes_nothing_but_the_wall_clock(
    offline_card: Scorecard,
) -> None:
    # `eval_workers` exists to cut the wall-clock of a live run, where a case is
    # seven sequential network calls. It must buy time and nothing else: a suite
    # whose verdict depends on how many cases were in flight is not a measurement.
    #
    # `offline_card` is the *default* configuration, so this compares the shipped
    # default against one case at a time. Both sides are pinned rather than one
    # being left implicit -- when the default changed from one worker to four,
    # a test written against "the default" would have compared four with four and
    # gone on passing while checking nothing.
    try:
        base = get_settings()
    except Exception:  # pragma: no cover - an environment with no configuration
        pytest.skip("no configuration to derive a worker count from")
    assert base.eval_workers > 1, "the default is meant to run cases concurrently"
    sequential = run_suite(settings=base.model_copy(update={"eval_workers": 1}))
    assert sequential.passed == offline_card.passed
    assert sequential.total == offline_card.total
    # The usage meter is a ContextVar installed inside `ask`, so each worker
    # thread counts its own calls and cannot see a sibling's. If that ever stopped
    # being true, the totals would drift apart here before anything else noticed.
    assert sequential.model_calls == offline_card.model_calls
    assert [result.case.name for result in sequential.results] == [
        result.case.name for result in offline_card.results
    ]
    assert [result.status for result in sequential.results] == [
        result.status for result in offline_card.results
    ]


@pytest.mark.slow
def test_a_case_runs_through_the_real_graph() -> None:
    # Not a mock in sight: the case's setting reaches the solver and the answer comes
    # back verified, which is what makes the suite's numbers worth reading.
    result = ask("What is the ground-state energy?", setting=chain(n_sites=6))
    assert result.status == "answered"
    assert result.is_verified
