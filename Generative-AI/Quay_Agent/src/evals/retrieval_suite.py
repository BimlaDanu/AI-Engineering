"""Run the labelled search questions and score what came back.

Separate from :mod:`src.evals.harness` because it grades a different thing in a
different way. The harness runs whole campaigns and grades their numbers against a
solver the agent cannot reach; this runs the retrieval layer alone and grades its
ranking against labels a person wrote. Nothing here calls the agent, and nothing
here needs an exact answer.

**It runs with no credential, and says which half ran.** The keyword index is built
from the corpus on disk and needs nothing else, so a checkout with no ``.env``
scores the lexical half honestly. With a credential the vector half joins in and the
score is the score of the hybrid the application actually ships. Both are legitimate
and they are not the same measurement, so :attr:`Report.searched_with` records which
one produced the number -- a score whose configuration is invisible is a score that
cannot be compared with the one before it.

**Grading is offline and deterministic in both modes.** The relevance judgement is a
set intersection against curated topic tags; no model is asked to grade anything, so
the score does not move because a model had a different day. That is the same
discipline the accuracy suite follows and the reason both are worth putting in a
report.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from src.agent.router import route
from src.evals.retrieval_cases import CASES, DEFAULT_TOP_K, VECTOR_SHARE, Case, Judged
from src.logging_setup import get_logger
from src.rag.lexical import default_index
from src.rag.retrieve import Expander, Passage, retrieve
from src.settings import Settings, get_settings

_log = get_logger("evals.retrieval")


@dataclass(frozen=True, slots=True)
class Report:
    """Everything one run of the retrieval suite established.

    Attributes:
        judged: One result per case, in case order.
        searched_with: ``"hybrid"`` when both halves ran, ``"keyword only"`` when
            there was no credential for the vector half. Recorded because the two
            are different measurements.
        expanded: Whether each question was also searched under other phrasings.
            Recorded for the same reason as ``searched_with``: it changes what is
            being measured, and it is the difference between scoring the retrieval
            layer alone and scoring the one that answers questions.
        top_k: How many passages each case was allowed to keep.
        seconds: Wall-clock for the whole suite.
    """

    judged: tuple[Judged, ...]
    searched_with: str
    top_k: int
    seconds: float
    expanded: bool = False

    @property
    def answerable(self) -> tuple[Judged, ...]:
        """The cases the corpus is supposed to be able to answer."""
        return tuple(result for result in self.judged if result.case.answerable)

    @property
    def passed(self) -> int:
        """How many cases passed, the off-corpus case included."""
        return sum(1 for result in self.judged if result.passed)

    @property
    def pass_rate(self) -> float:
        """Share of cases that passed, in ``[0, 1]``.

        Returns:
            Passes over cases, or ``0.0`` for an empty suite. This is the headline
            number the scorecard quotes.
        """
        return 0.0 if not self.judged else self.passed / len(self.judged)

    @property
    def mean_precision(self) -> float:
        """Average share of returned passages that were relevant.

        Averaged over the answerable cases only. Including the off-corpus case
        would average in a precision of zero for a case whose correct behaviour is
        to return nothing, which would make the metric go *down* when the suite
        does the right thing.

        Returns:
            The mean, or ``0.0`` when there is nothing to average.
        """
        cases = self.answerable
        return 0.0 if not cases else sum(result.precision for result in cases) / len(cases)

    @property
    def mean_reciprocal_rank(self) -> float:
        """Average reciprocal rank of the first relevant passage.

        Returns:
            The mean over answerable cases, or ``0.0``. One means the best passage
            was always relevant; a half means it was typically second.
        """
        cases = self.answerable
        return 0.0 if not cases else sum(result.reciprocal_rank for result in cases) / len(cases)

    @property
    def routed_correctly(self) -> int:
        """How many answerable cases the router sent to the expected shelf."""
        return sum(1 for result in self.answerable if result.routed_correctly)

    @property
    def routed_nowhere(self) -> int:
        """How many answerable cases the router left unrestricted."""
        return sum(1 for result in self.answerable if result.routed_nowhere)

    @property
    def routed_elsewhere(self) -> int:
        """How many answerable cases went to a shelf the label did not expect.

        The only one of the three routing counts that is a mistake, and even then
        a mild one: retrieval widens after the first round, so a wrong shelf costs
        a round rather than an answer.
        """
        return len(self.answerable) - self.routed_correctly - self.routed_nowhere

    @property
    def routing_rate(self) -> float:
        """Share of answerable cases sent to the expected shelf.

        Read alongside :attr:`routed_nowhere`. The denominator is every answerable
        case, so a router that names no shelf scores zero here -- which is the
        honest reading of "did it pick the right one" and the reason the other two
        counts are reported beside it rather than folded in.

        Returns:
            The rate, or ``0.0`` when there is nothing to score.
        """
        cases = self.answerable
        return 0.0 if not cases else self.routed_correctly / len(cases)


def judge(
    case: Case,
    passages: tuple[Passage, ...],
    outcome: str,
    seconds: float,
    routed_to: tuple[str, ...] = (),
) -> Judged:
    """Grade one case's passages against its labels.

    The relevance test is a set intersection: a passage is relevant when the note
    it came from declares at least one of the topics the case names. No model is
    consulted and no text is compared, so this returns the same verdict on every
    run and on every machine.

    Args:
        case: The labelled question.
        passages: What retrieval returned, best first.
        outcome: What the retrieval layer reported doing.
        seconds: How long the search took.
        routed_to: Shelves the router chose, best guess first.

    Returns:
        The graded result.
    """
    wanted = frozenset(case.topics)
    hits = [bool(wanted & frozenset(passage.topics)) for passage in passages]
    first = next((index + 1 for index, hit in enumerate(hits) if hit), None)
    return Judged(
        case=case,
        returned=len(passages),
        relevant=sum(hits),
        first_relevant_rank=first,
        shelves=tuple(passage.shelf for passage in passages),
        titles=tuple(passage.title for passage in passages),
        outcome=outcome,
        routed_to=routed_to,
        seconds=round(seconds, 3),
    )


def run_case(
    case: Case,
    *,
    top_k: int = DEFAULT_TOP_K,
    vector_share: float,
    settings: Settings | None = None,
    expander: Expander | None = None,
) -> Judged:
    """Search for one case and grade the result.

    Args:
        case: The labelled question.
        top_k: How many passages to keep.
        vector_share: How much of the search is vector similarity. Zero runs the
            keyword half alone, which is what a checkout with no credential gets.
        settings: Configuration to search under.
        expander: Asked for other phrasings of the question before the search, as
            :func:`src.agent.graph.run_campaign` does. ``None`` searches the
            question as written, which scores a narrower system than the one that
            answers questions.

    Returns:
        The graded result. A search that raises is recorded as a failure with its
        exception type rather than stopping the suite, because one broken case
        should not cost the report the other eleven.
    """
    started = perf_counter()
    try:
        # The router first, exactly as the graph does it, and with `allow_model`
        # off so the decision is deterministic. This is the difference between
        # scoring the pipeline and scoring one stage of it: an earlier version of
        # this suite handed retrieval the shelf from the case's own label, which
        # made the routing score 100% by construction and measured nothing. The
        # router is also what refuses a question the corpus does not cover, so
        # without it the off-corpus case would be testing a path the product
        # never takes.
        routing = route(case.question, allow_model=False)
        if not routing.needs_retrieval:
            return Judged(
                case=case,
                returned=0,
                relevant=0,
                first_relevant_rank=None,
                outcome="not_needed",
                routed_to=(),
                seconds=round(perf_counter() - started, 3),
            )
        found = retrieve(
            case.question,
            topics=routing.topics,
            shelves=routing.shelves,
            lexical=default_index(),
            limit=top_k,
            vector_share=vector_share,
            settings=settings,
            expander=expander,
        )
    except Exception as error:
        _log.warning(
            "retrieval_case_failed",
            extra={"case": case.name, "detail": type(error).__name__},
        )
        return Judged(
            case=case,
            returned=0,
            relevant=0,
            first_relevant_rank=None,
            outcome="failed",
            seconds=round(perf_counter() - started, 3),
            failure=type(error).__name__,
        )
    return judge(
        case,
        found.passages,
        found.outcome,
        perf_counter() - started,
        routed_to=routing.shelves,
    )


def run_retrieval_suite(
    cases: tuple[Case, ...] = CASES,
    *,
    top_k: int = DEFAULT_TOP_K,
    settings: Settings | None = None,
    keyword_only: bool | None = None,
    expander: Expander | None = None,
) -> Report:
    """Run every labelled question and return the scored report.

    Args:
        cases: What to run. Defaults to the whole labelled set.
        top_k: How many passages each case may keep.
        settings: Configuration to search under.
        keyword_only: Force the keyword half alone. ``None`` decides from whether
            a credential is configured, which is what makes this runnable in the
            test suite and on a fresh checkout without a special flag.
        expander: Other phrasings of each question, as the campaign asks for them.
            ``None`` searches each question as written.

    Returns:
        The report, including which halves of the search produced it.
    """
    resolved = get_settings() if settings is None else settings
    credentialled = bool(resolved.openrouter_api_key.get_secret_value())
    lexical_only = (not credentialled) if keyword_only is None else keyword_only
    share = 0.0 if lexical_only else VECTOR_SHARE
    started = perf_counter()
    judged = tuple(
        run_case(case, top_k=top_k, vector_share=share, settings=resolved, expander=expander)
        for case in cases
    )
    report = Report(
        judged=judged,
        searched_with="keyword only" if lexical_only else "hybrid",
        top_k=top_k,
        seconds=round(perf_counter() - started, 2),
        expanded=expander is not None,
    )
    _log.info(
        "retrieval_suite_scored",
        extra={
            "cases": len(judged),
            "passed": report.passed,
            "searched_with": report.searched_with,
            "expanded": report.expanded,
            "mean_precision": round(report.mean_precision, 3),
            "mean_reciprocal_rank": round(report.mean_reciprocal_rank, 3),
        },
    )
    return report


def summarise(report: Report) -> dict[str, Any]:
    """Reduce a report to primitives an interface can draw without parsing prose.

    Args:
        report: What the suite produced.

    Returns:
        A JSON-serialisable mapping, shaped like the other suites' summaries so the
        scorecard can treat all three the same way.
    """
    return {
        "cases": len(report.judged),
        "passed": report.passed,
        "pass_rate": round(report.pass_rate, 4),
        "mean_precision": round(report.mean_precision, 4),
        "mean_reciprocal_rank": round(report.mean_reciprocal_rank, 4),
        "routing_rate": round(report.routing_rate, 4),
        "routed_correctly": report.routed_correctly,
        "routed_elsewhere": report.routed_elsewhere,
        "routed_nowhere": report.routed_nowhere,
        "answerable": len(report.answerable),
        "searched_with": report.searched_with,
        "expanded": report.expanded,
        "top_k": report.top_k,
        "seconds": report.seconds,
        "cases_detail": [
            {
                "name": result.case.name,
                "question": result.case.question,
                "shelf": result.case.shelf,
                "topics": list(result.case.topics),
                "why": result.case.why,
                "returned": result.returned,
                "relevant": result.relevant,
                "precision": round(result.precision, 4),
                "first_relevant_rank": result.first_relevant_rank,
                "reciprocal_rank": round(result.reciprocal_rank, 4),
                "routed_correctly": result.routed_correctly,
                "routed_to": list(result.routed_to),
                "passed": result.passed,
                "outcome": result.outcome,
                "titles": list(result.titles),
                "seconds": result.seconds,
                "failure": result.failure,
            }
            for result in report.judged
        ],
    }
