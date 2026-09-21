"""Tests for the retrieval score: the labels, the metrics, and the pipeline it runs.

These run against the **real corpus on disk**, deliberately, and they are the only
tests in the project that do. Everything else in ``tests/`` builds a fake store with
two or three documents, which is right for testing a ranking stage in isolation and
useless for the question this suite exists to answer: does the search find the right
note out of a hundred and twenty-six? A fixture with three documents cannot fail that
question, so a test built on one would be scoring nothing.

The cost of that decision is that these tests are skipped on a checkout with no
ingested corpus rather than failing, and that the pass rate they assert is a floor
rather than an equality. A floor because the corpus is refetched from arXiv by
``make corpus``, so the exact ranking is not a constant of this repository -- and a
test asserting the exact number would fail on the next refetch for a reason that is
not a regression. What is asserted instead is that the score is *computed correctly*
and that it has not collapsed.
"""

from __future__ import annotations

import pytest

from src.evals import scorecard
from src.evals.retrieval_cases import (
    CASES,
    DEFAULT_TOP_K,
    OFF_CORPUS,
    VECTOR_SHARE,
    Case,
    Judged,
)
from src.evals.retrieval_suite import Report, judge, run_retrieval_suite
from src.rag.ingest import load_library, shelf_names
from src.rag.retrieve import DEFAULT_VECTOR_SHARE, Passage

MINIMUM_PASSES = 9
"""How many of the twelve cases must pass for the suite to be considered working.

Nine rather than twelve, and it is a floor on the *measurement* rather than a target
for the product. The corpus is refetched from arXiv, so a specific ranking is not a
property of this repository; what is a property is that the search is not broken. At
nine, a keyword index that had stopped being built or a store that had stopped
opening fails this immediately, and a modest ranking change does not.
"""


def _corpus_available() -> bool:
    """Whether there is an ingested corpus to search.

    Returns:
        True when the library on disk holds notes. A checkout that has never run
        ``make corpus`` has none, and these tests skip rather than fail -- the
        absence of data is not a defect in the code being tested.
    """
    try:
        return len(load_library()) > 0
    except Exception:
        return False


needs_corpus = pytest.mark.skipif(
    not _corpus_available(),
    reason="no ingested corpus on disk; run `make corpus`",
)


def _passage(topics: tuple[str, ...], shelf: str = "physics-notes") -> Passage:
    """Build a passage carrying the given topics.

    Args:
        topics: What the source note declares.
        shelf: Which knowledge base it came from.

    Returns:
        A passage with everything else left empty, because judging reads the
        topics and the shelf and nothing else.
    """
    return Passage(
        identifier="chunk",
        text="text",
        document="doc",
        title="A note",
        source="source",
        arxiv="",
        section="Abstract",
        topics=topics,
        score=0.5,
        shelf=shelf,
    )


# --- the labels themselves ------------------------------------------------


def test_every_label_names_topics_the_corpus_actually_carries() -> None:
    # The failure this catches is the one that makes a suite worthless without
    # looking broken: a label naming a topic no note declares can never be
    # satisfied, so the case fails for ever and reads as a retrieval defect.
    library = load_library()
    if not library:
        pytest.skip("no ingested corpus on disk; run `make corpus`")
    declared = {topic for note in library for topic in note.topics}
    for case in CASES:
        missing = sorted(set(case.topics) - declared)
        assert not missing, (
            f"case {case.name!r} is labelled with topics no note in the corpus "
            f"declares: {missing}. Either the label is wrong or the note it was "
            f"written for is gone -- both make this case unpassable."
        )


def test_every_expected_shelf_is_a_shelf_that_exists() -> None:
    existing = set(shelf_names())
    for case in CASES:
        if case.shelf:
            assert case.shelf in existing, (
                f"case {case.name!r} expects shelf {case.shelf!r}, which is not one "
                f"of {sorted(existing)}"
            )


def test_exactly_one_case_is_unanswerable() -> None:
    # A retrieval score with no unanswerable case rewards confident irrelevance:
    # a search that always returns its four best guesses would score full marks.
    unanswerable = [case.name for case in CASES if not case.answerable]
    assert unanswerable == [OFF_CORPUS], (
        "the suite needs exactly one question the corpus cannot answer, and it "
        f"needs to be the one named by OFF_CORPUS; found {unanswerable}"
    )


def test_every_case_says_why_it_is_in_the_suite() -> None:
    for case in CASES:
        assert len(case.why.split()) >= 10, (
            f"case {case.name!r} has no real justification. A hand-labelled suite is "
            "auditable only if each label can be defended in a sentence."
        )


def test_the_suite_measures_the_mix_the_interface_ships() -> None:
    # The knob's default and the eval's default have to be the same number, or the
    # score is of a configuration nobody runs. Checked here rather than trusted,
    # because the retrieval layer's own default is a third value (0.5) and the
    # three have drifted apart before.
    from src.ui.setting import DEFAULT_VECTOR_SHARE as SHIPPED

    assert VECTOR_SHARE == SHIPPED, (
        "the retrieval suite scores a vector share the interface does not use: "
        f"suite {VECTOR_SHARE}, interface {SHIPPED}"
    )
    assert DEFAULT_VECTOR_SHARE != SHIPPED, (
        "src.rag.retrieve.DEFAULT_VECTOR_SHARE now agrees with the interface's "
        "default. That is an improvement, but this test documents a known "
        "divergence -- delete it and the comment in retrieval_cases.VECTOR_SHARE."
    )


def test_the_suite_keeps_as_many_passages_as_the_interface_shows() -> None:
    from src.ui.setting import DEFAULT_PASSAGES

    assert DEFAULT_TOP_K == DEFAULT_PASSAGES, (
        "precision is measured at a depth the product does not use: "
        f"suite {DEFAULT_TOP_K}, interface {DEFAULT_PASSAGES}"
    )


# --- the metrics ----------------------------------------------------------


def test_relevance_is_a_topic_intersection_not_a_text_match() -> None:
    case = Case(
        name="x",
        question="a question",
        topics=("criticality",),
        shelf="physics-notes",
        why="x" * 60,
    )
    result = judge(
        case,
        (
            _passage(("hardware",)),
            _passage(("criticality", "entanglement")),
        ),
        "grounded",
        0.1,
    )
    assert result.relevant == 1
    assert result.first_relevant_rank == 2
    assert result.precision == pytest.approx(0.5)
    assert result.reciprocal_rank == pytest.approx(0.5)


def test_the_first_hit_is_what_reciprocal_rank_reports() -> None:
    # Precision cannot tell "the right note was first" from "the right note was
    # fourth", and the first citation is the one a reader actually reads.
    case = CASES[0]
    early = judge(case, (_passage(case.topics[:1]), _passage(("hardware",))), "grounded", 0.1)
    late = judge(case, (_passage(("hardware",)), _passage(case.topics[:1])), "grounded", 0.1)
    assert early.precision == late.precision
    assert early.reciprocal_rank > late.reciprocal_rank


def test_returning_nothing_passes_only_the_unanswerable_case() -> None:
    answerable = next(case for case in CASES if case.answerable)
    unanswerable = next(case for case in CASES if not case.answerable)
    assert not judge(answerable, (), "nothing_relevant", 0.1).passed
    assert judge(unanswerable, (), "nothing_relevant", 0.1).passed
    # And the reverse: confidently answering the unanswerable one is a failure.
    assert not judge(unanswerable, (_passage(("hardware",)),), "grounded", 0.1).passed


def test_a_router_that_names_no_shelf_is_not_scored_as_wrong() -> None:
    # An empty shelf list is the router declining to guess, and the search that
    # follows is unrestricted -- slower, never wrong. Counting it as an error would
    # push the design towards guessing, which is what the fallback exists to avoid.
    case = next(case for case in CASES if case.shelf)
    left_open = judge(case, (_passage(case.topics[:1]),), "grounded", 0.1, routed_to=())
    assert left_open.routed_nowhere
    assert not left_open.routed_correctly
    wrong = judge(case, (_passage(case.topics[:1]),), "grounded", 0.1, routed_to=("applications",))
    assert not wrong.routed_nowhere
    assert not wrong.routed_correctly


def test_the_unanswerable_case_is_left_out_of_the_precision_average() -> None:
    # Averaging in a precision of zero for a case whose correct behaviour is to
    # return nothing would make the metric fall when the suite does the right thing.
    answerable = next(case for case in CASES if case.answerable)
    unanswerable = next(case for case in CASES if not case.answerable)
    report = Report(
        judged=(
            judge(answerable, (_passage(answerable.topics[:1]),), "grounded", 0.1),
            judge(unanswerable, (), "nothing_relevant", 0.1),
        ),
        searched_with="keyword only",
        top_k=DEFAULT_TOP_K,
        seconds=0.2,
    )
    assert report.mean_precision == pytest.approx(1.0)
    assert report.pass_rate == pytest.approx(1.0)


def test_a_suite_that_did_not_run_scores_a_dash_rather_than_zero() -> None:
    # The one error a scorecard must never make is reporting worse machinery as
    # better, or better as worse, on the strength of what was skipped.
    score = scorecard.retrieval_score(None)
    assert not score.ran
    assert score.percent == "--"
    assert score.tally == "--"


def test_a_failed_search_is_a_row_rather_than_a_missing_case() -> None:
    case = CASES[0]
    result = Judged(
        case=case,
        returned=0,
        relevant=0,
        first_relevant_rank=None,
        outcome="failed",
        failure="RuntimeError",
    )
    report = Report(
        judged=(result,), searched_with="keyword only", top_k=DEFAULT_TOP_K, seconds=0.1
    )
    rendered = scorecard.render((), (), report)
    assert case.name in rendered
    assert "| NO |" in rendered


# --- the pipeline, against the real corpus --------------------------------


@needs_corpus
def test_the_keyword_half_alone_finds_most_of_the_labelled_notes() -> None:
    # Keyword only, so this needs no credential and no network and is the version
    # that runs in CI. It is also the honest floor: whatever the vector half adds,
    # the search should not depend on it to work at all.
    report = run_retrieval_suite(keyword_only=True)
    assert report.searched_with == "keyword only"
    assert report.passed >= MINIMUM_PASSES, (
        f"only {report.passed} of {len(report.judged)} labelled questions found a "
        "relevant note with keyword search. Either the index is not being built or "
        "the ranking has regressed: "
        + ", ".join(result.case.name for result in report.judged if not result.passed)
    )
    assert 0.0 < report.mean_precision <= 1.0
    assert 0.0 < report.mean_reciprocal_rank <= 1.0


@needs_corpus
def test_the_question_the_corpus_cannot_answer_is_refused_before_a_search() -> None:
    # In the product this is the router's job, and the suite runs the router. A
    # version of this suite that handed retrieval the shelf from the label skipped
    # the router entirely and so tested a path the application never takes.
    report = run_retrieval_suite(keyword_only=True)
    off = next(result for result in report.judged if result.case.name == OFF_CORPUS)
    assert off.returned == 0
    assert off.passed


@needs_corpus
def test_the_question_the_whole_verdict_turns_on_is_not_refused_as_off_topic() -> None:
    # The regression this pins. "Can an ordinary computer already do this better?"
    # was routed out of scope, because the scope gate held the word "classical" and
    # not the phrase the application itself prints on screen. The question the
    # product exists to answer was refused in the product's own vocabulary.
    from src.agent.router import in_domain

    for asked in (
        "Can an ordinary computer already do this better?",
        "Would a normal computer be faster?",
        "Is a conventional computer enough for this?",
    ):
        assert in_domain(asked), f"the scope gate refuses {asked!r}"


@needs_corpus
def test_the_score_is_reported_with_the_search_mode_that_produced_it() -> None:
    # Two different measurements, and a score whose configuration is invisible
    # cannot be compared with the one before it.
    report = run_retrieval_suite(keyword_only=True)
    rendered = scorecard.render((), (), report)
    assert "keyword only" in rendered
    assert f"{report.passed}/{len(report.judged)}" in rendered
