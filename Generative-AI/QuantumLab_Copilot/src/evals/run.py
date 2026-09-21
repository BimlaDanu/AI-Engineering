"""Running the suite: locally always, and in LangSmith when it is credentialed.

``make evals`` lands here. It runs every case in :data:`src.evals.cases.CASES`,
grades each one with :func:`src.evals.scoring.grade`, writes
:data:`REPORT_PATH` and -- if LangSmith is configured -- files the same run as an
experiment there so the results are comparable between commits.

**The local run is the authority.** LangSmith is an observability tool, not a
dependency: without a key the suite still runs, still scores identically, and still
writes the scorecard, with a note saying the experiment was not filed. An
evaluation that only works when a third-party service is reachable is an evaluation
that cannot be run in CI or on a plane.

**Two grades, one grader.** The pass/fail a reader sees on the Evaluation page and
the score LangSmith records come from the same
:func:`src.evals.scoring.grade` call. Two graders would eventually disagree, and
then the question "did this commit regress?" would have two answers.

Each case is run with a memory of its own, in a temporary directory, for the same
reason the tests are: a suite whose second run behaves differently from its first is
not a measurement. The memory cases still exercise recall -- they simply do it
inside a store that is thrown away afterwards.
"""

from __future__ import annotations

import json
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from src.agent.graph import Answer, ask
from src.agent.memory import FileMemory
from src.evals.cases import CASES, Case
from src.evals.scoring import CaseResult, Scorecard, grade
from src.logging_setup import get_logger
from src.rag.lexical import default_index
from src.rag.retrieve import warm_corpus
from src.settings import Settings, get_settings

LOG = get_logger("evals")

REPORT_PATH = Path("reports/scorecard.md")
SUMMARY_PATH = Path("reports/scorecard.json")
"""Where the scorecard is written.

Two files with one content: the Markdown is for a person, and the JSON is what the
Evaluation page reads to draw its metrics and its chart. The page could parse the
Markdown, but then a change to the report's wording would break the page, and the
report exists to be rewritten.

Committed rather than ignored. The scorecard is the evidence for the project's
central claim, and evidence nobody can see without running the suite themselves is
not evidence.
"""

THREAD_PREFIX = "eval"
"""Thread each case is filed under, suffixed by the case index.

One conversation per case, so no case can recall another's turns -- which would
make the suite's result depend on the order it happened to run in.
"""


@dataclass(frozen=True, slots=True)
class Run:
    """One case's two runs, before grading.

    Attributes:
        case: What was asked.
        answer: The first run.
        follow_up: The second run, when the case asked a follow-up.
    """

    case: Case
    answer: Answer
    follow_up: Answer | None = None


def validate(cases: tuple[Case, ...] = CASES) -> tuple[str, ...]:
    """Check the suite itself before trusting what it says.

    A case that asserts nothing passes every time, which is worse than a missing
    case: it reports success. A duplicate name silently overwrites a row in the
    scorecard. Both are mistakes made while *editing* a suite, so they are checked
    on every run rather than left to review.

    Args:
        cases: The suite to check.

    Returns:
        One message per problem, empty when the suite is well formed.
    """
    problems: list[str] = []
    seen: set[str] = set()
    for case in cases:
        if case.name in seen:
            problems.append(f"duplicate case name: {case.name!r}")
        seen.add(case.name)
        if not case.why.strip():
            problems.append(f"{case.name!r} does not say what it defends")
        asserts_something = any(
            (
                case.expect_status,
                case.expect_nodes,
                case.forbid_nodes,
                case.expect_verified,
                case.expect_unverified_notice,
                case.expect_energy is not None,
                case.expect_route,
                case.expect_shelves,
                case.require_shelves,
                case.forbid_shelves,
                case.expect_followups,
                case.forbid_followups,
                case.follow_up_expect_status,
                case.forbid_follow_up_route,
            )
        )
        if not asserts_something:
            problems.append(f"{case.name!r} asserts nothing and would always pass")
    return tuple(problems)


def run_case(case: Case, index: int, root: Path, settings: Settings | None = None) -> Run:
    """Ask one case's question, and its follow-up if it has one.

    Args:
        case: The case to run.
        index: Its position in the suite, used to keep threads apart.
        root: Directory to put this case's memory in.
        settings: Configuration. Defaults to the process settings, which is what
            decides whether a model narrates the answers at all.

    Returns:
        The runs, ungraded.
    """
    thread = f"{THREAD_PREFIX}-{index}/1"
    memory = FileMemory(root / f"{THREAD_PREFIX}-{index}.jsonl")
    answer = ask(
        case.question, setting=case.setting, thread=thread, settings=settings, memory=memory
    )
    follow_up = None
    if case.follow_up:
        follow_up = ask(
            case.follow_up,
            setting=case.setting,
            thread=thread,
            settings=settings,
            memory=memory,
        )
    return Run(case=case, answer=answer, follow_up=follow_up)


def _resolved_settings(settings: Settings | None) -> Settings | None:
    """Return the configuration this run should use, or ``None`` if there is none.

    Args:
        settings: What the caller passed. ``None`` means "read the environment".

    Returns:
        The settings, or ``None`` when the environment holds no usable
        configuration -- no credential, most often. That is a supported way to
        run the suite, not a failure: every model call then falls back to the
        deterministic path.
    """
    if settings is not None:
        return settings
    try:
        return get_settings()
    except Exception as error:
        LOG.info(
            "evals_using_defaults",
            extra={"error_type": type(error).__name__, "detail": "no configuration available"},
        )
        return None


def _run_and_log(case: Case, index: int, total: int, root: Path, settings: Settings | None) -> Run:
    """Run one case and record that it started, finished and how long it took.

    Both ends are logged because the suite is slow and mostly silent: live, a
    case is several rate-limited model calls, so a start line is what shows the
    run progressing, and a finish line carrying ``elapsed_ms`` is what shows
    *which* case is the slow one when the total looks wrong.

    Args:
        case: The case to run.
        index: Its position in the suite, from zero.
        total: How many cases there are, so a log line can say "3 of 26".
        root: Directory to put this case's throwaway memory in.
        settings: Configuration to answer with.

    Returns:
        The run, ungraded.
    """
    LOG.info("eval_case_start", extra={"case": case.name, "index": index + 1, "total": total})
    started = time.perf_counter()
    run = run_case(case, index, root, settings)
    LOG.info(
        "eval_case_finished",
        extra={
            "case": case.name,
            "index": index + 1,
            "total": total,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "model_calls": run.answer.usage.calls_made
            + (run.follow_up.usage.calls_made if run.follow_up else 0),
        },
    )
    return run


def _warm_shared_caches(settings: Settings | None) -> None:
    """Open the corpus once before any case asks for it.

    Retrieval rests on two things a process builds lazily: the memoised keyword
    index, and the vector collection, which is opened per search. Sequentially
    that is invisible -- the first case pays for both and the rest find them
    ready. Run several cases at once and it stops being invisible: the cases that
    start together all reach a cold collection, opening it races, and the losers
    log ``retrieval_store_unavailable`` and answer without the corpus.

    The symptom was a literature question refused for want of grounding on the
    first run of a process and answered on the second -- the suite measuring its
    own cache state rather than the agent. Warming here rather than locking the
    store keeps the fix in the caller that creates the concurrency, and leaves
    single-threaded behaviour exactly as it was.

    Called on the sequential path too, so both paths start warm and the scorecard
    cannot depend on which one produced it.

    Args:
        settings: Configuration the collection should be opened with.

    Failures stay where they are reported. Both helpers already log and return
    ``None`` when the corpus cannot be read, and a run without it is a supported
    run, not a crash.
    """
    default_index()
    warm_corpus(settings)


def score(runs: list[Run]) -> tuple[CaseResult, ...]:
    """Grade every run.

    Args:
        runs: What the suite produced.

    Returns:
        One result per run, in the order they were run.
    """
    return tuple(grade(run.case, run.answer, run.follow_up) for run in runs)


def run_suite(
    cases: tuple[Case, ...] = CASES,
    *,
    settings: Settings | None = None,
    notes: tuple[str, ...] = (),
) -> Scorecard:
    """Run and grade the whole suite.

    Args:
        cases: The suite to run.
        settings: Configuration. ``None`` means the process settings, and an
            environment with no credential runs the whole suite on the
            deterministic path -- which is a supported result, not a skipped one.
        notes: Anything that qualifies the run, carried onto the scorecard.

    Returns:
        The scorecard. Its contents do not depend on
        :attr:`~src.settings.Settings.eval_workers`: running several cases at
        once changes how long the suite takes and nothing else, because the
        results are collected back into case order before they are graded.
    """
    problems = validate(cases)
    resolved = _resolved_settings(settings)
    model = resolved.chat_model if resolved is not None else ""
    workers = min(resolved.eval_workers if resolved is not None else 1, max(1, len(cases)))

    with tempfile.TemporaryDirectory(prefix="quantumlab-evals-") as scratch:
        root = Path(scratch)
        LOG.info("evals_started", extra={"total": len(cases), "workers": workers})
        _warm_shared_caches(resolved)
        numbered = list(enumerate(cases))
        if workers == 1:
            runs = [
                _run_and_log(case, index, len(cases), root, settings) for index, case in numbered
            ]
        else:
            # Cases share nothing: each has its own conversation thread and its own
            # memory file under `root`, so the only coupling is the rate limiter,
            # which is process-wide and thread-safe by design. Results are put back
            # in case order below, so the scorecard cannot depend on the worker
            # count -- a suite whose report changes with its concurrency is not a
            # measurement.
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="eval") as pool:
                futures = {
                    pool.submit(_run_and_log, case, index, len(cases), root, settings): index
                    for index, case in numbered
                }
                done: dict[int, Run] = {}
                for future in as_completed(futures):
                    done[futures[future]] = future.result()
            runs = [done[index] for index, _case in numbered]

    calls = sum(
        run.answer.usage.calls_made + (run.follow_up.usage.calls_made if run.follow_up else 0)
        for run in runs
    )
    card = Scorecard(results=score(runs), model=model, model_calls=calls, notes=(*notes, *problems))
    LOG.info("evals_finished", extra={"passed": card.passed, "total": card.total})
    return card


def write_report(card: Scorecard, report: Path = REPORT_PATH, summary: Path = SUMMARY_PATH) -> None:
    """Write the scorecard to disk, in both forms.

    Args:
        card: What to write.
        report: Where the Markdown goes.
        summary: Where the machine-readable form goes.
    """
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(card.markdown(), encoding="utf-8")
    summary.write_text(
        json.dumps(
            {
                "passed": card.passed,
                "total": card.total,
                "rate": card.rate,
                "model": card.model,
                "model_calls": card.model_calls,
                "provenance": card.provenance(),
                "notes": list(card.notes),
                "families": [
                    {"family": score.family, "passed": score.passed, "total": score.total}
                    for score in card.by_family()
                ],
                "cases": [
                    {
                        "name": result.case.name,
                        "family": result.case.family,
                        "passed": result.passed,
                        "status": result.status,
                        "nodes": list(result.nodes),
                        "failures": [
                            f"expected {check.claim}; {check.detail}" for check in result.failures
                        ],
                    }
                    for result in card.results
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _start_logging() -> None:
    """Attach a log handler, so a long run is visibly a long run.

    Without this the suite emitted nothing at all for its entire duration: the
    loggers exist, but nothing had called :func:`src.logging_setup.configure_logging`,
    so every record was dropped by the root logger's default. A process that runs
    for minutes while printing nothing is indistinguishable from a hung one, and
    was reported as exactly that.

    Never raises. An unconfigured environment is a supported way to run the suite
    (every model call falls back to the deterministic path), and it must not become
    an unsupported way to *see* the suite run.
    """
    from src.logging_setup import configure_logging

    try:
        key = get_settings().openrouter_api_key.get_secret_value()
    except Exception:
        configure_logging()
        return
    configure_logging(secrets=[key])


def main() -> int:
    """Run the suite, file it with LangSmith if possible, and write the report.

    Returns:
        A process exit status: ``0`` when every case passed and ``1`` otherwise, so
        that ``make evals`` can be a gate rather than a report nobody reads.
    """
    from src.evals import tracking  # imported here so a missing extra cannot break an import

    _start_logging()
    LOG.info("evals_starting", extra={"cases": len(CASES)})
    notes = tracking.begin()
    card = run_suite(notes=notes)
    card = tracking.publish(card)
    write_report(card)
    LOG.info("evals_written", extra={"path": str(REPORT_PATH), "summary": card.summary()})
    return 0 if card.passed == card.total else 1


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
