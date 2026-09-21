"""Running the suite, and grading what comes back.

The harness is the only place in the project that holds both halves at once: it
calls the agent, which cannot see an exact answer, and then calls an exact solver,
which the agent cannot reach. That is why it lives here and not under
``src/agent`` -- the import wall that makes the experiment mean anything runs
between those two lines.

Two properties are worth stating because they are what make a scorecard
comparable to the one before it.

**Cases are independent.** Each gets its own campaign, its own conversation and no
memory. Nothing a case learns can reach the next one, so the order they run in
cannot change the result -- which is what makes running them concurrently safe.

**Concurrency is wall-clock only.** Answering four cases at once does not send
requests any faster: the throttle is process-wide and shared by every model this
application builds, so the outbound rate is unchanged and what overlaps is the
time each worker spends waiting. Results are collected back into case order before
grading, so the scorecard does not depend on how many workers ran.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from src.agent.graph import run_campaign
from src.agent.memory import Memory
from src.agent.model_selection import ModelPool
from src.agent.state import CampaignState
from src.evals.cases import Case
from src.evals.metrics import Accuracy, Honesty, score_accuracy, score_honesty
from src.logging_setup import get_logger
from src.physics.model import TFIMSpec
from src.physics.registry import exact_survey
from src.settings import Settings, get_settings

_log = get_logger("evals.harness")


@dataclass(frozen=True, slots=True)
class Outcome:
    """One case, run and graded.

    Attributes:
        case: What was asked.
        accuracy: How close the campaign got to the true answer.
        honesty: What its report said and how firmly.
        elapsed_s: Wall-clock seconds the campaign took.
        read_as: What the agent decided the question described, as a label. Kept
            beside the case so a wrong answer can be attributed: a campaign that
            solved a different chain perfectly failed at reading, not at physics,
            and those are separate bugs with separate fixes.
        failure: Why the case did not run at all, if it did not. Empty on success.
    """

    case: Case
    accuracy: Accuracy | None
    honesty: Honesty | None
    elapsed_s: float
    read_as: str = ""
    failure: str = ""

    @property
    def ok(self) -> bool:
        """Whether the case produced a gradeable campaign."""
        return not self.failure


def exact_energy_per_site(spec: TFIMSpec) -> float:
    """The true ground-state energy per spin.

    This is the grader's privilege and the reason the harness may import what the
    agent may not. The first applicable exact route is used; where two apply they
    agree to within numerical precision, which the physics suite checks and this
    one relies on rather than re-proving.

    Args:
        spec: The chain.

    Returns:
        The exact energy divided by the number of spins.

    Raises:
        RuntimeError: If no exact route applies. Every case in the suite is
            constructed to be solvable exactly, so this means the suite is wrong
            rather than the physics -- and a case that cannot be graded must fail
            loudly rather than quietly score zero.
    """
    survey = exact_survey(spec)
    if not survey.applicable:
        raise RuntimeError(
            f"no exact solver applies to {spec}; a case that cannot be graded "
            "does not belong in the suite"
        )
    return survey.applicable[0].solve(spec) / spec.n_sites


def run_one(
    case: Case,
    *,
    models: ModelPool | None = None,
    offline: bool = False,
    search_corpus: bool = True,
) -> Outcome:
    """Run one case and grade it.

    Args:
        case: What to ask.
        models: The models to answer with. ``None`` takes them from configuration.
        offline: Run with no model at all, on the deterministic paths only. This is
            what makes the suite runnable without a credential -- the physics and
            the verdict rules are still exercised, and only the language model's
            contribution is absent.
        search_corpus: Whether the campaign may retrieve background. Off by
            default in an offline run, since retrieval needs embeddings.

    Returns:
        The graded outcome. A campaign that raises is caught and returned as a
        failure rather than allowed to end the suite: one broken case should cost
        one row of the scorecard, not the other nine.
    """
    started = perf_counter()
    try:
        state = run_campaign(
            case.question,
            framing=case.framing,
            shot_budget=case.shot_budget,
            chat_model=None if offline else "auto",
            models=models,
            search_corpus=search_corpus,
            fetch_external=False,
            # Every case starts from nothing. A case that could recall the previous
            # one would make the suite order-dependent, and an order-dependent
            # scorecard cannot be compared with the one before it.
            memory=Memory.disabled(),
        )
    except Exception as error:  # one bad case must cost one row, not the suite
        _log.warning(
            "eval_case_failed",
            extra={"case": case.name, "error_type": type(error).__name__},
        )
        return Outcome(
            case=case,
            accuracy=None,
            honesty=None,
            elapsed_s=perf_counter() - started,
            failure=f"{type(error).__name__}: {error}",
        )
    elapsed = perf_counter() - started
    outcome = _grade(case, state, elapsed)
    _log.info(
        "eval_case",
        extra={
            "case": case.name,
            "verdict": outcome.honesty.verdict if outcome.honesty else "none",
            "error_per_site": outcome.accuracy.error_per_site if outcome.accuracy else None,
            "elapsed_s": round(elapsed, 2),
        },
    )
    return outcome


def _grade(case: Case, state: CampaignState, elapsed: float) -> Outcome:
    """Score one finished campaign.

    Args:
        case: What was asked.
        state: What came back.
        elapsed: How long it took.

    Returns:
        The graded outcome.
    """
    model = state.get("model")
    return Outcome(
        case=case,
        accuracy=score_accuracy(state, exact_energy_per_site(case.spec)),
        honesty=score_honesty(state),
        elapsed_s=elapsed,
        read_as="" if model is None else model.label(),
    )


def run_suite(
    cases: tuple[Case, ...],
    *,
    models: ModelPool | None = None,
    offline: bool = False,
    search_corpus: bool = True,
    settings: Settings | None = None,
) -> tuple[Outcome, ...]:
    """Run every case and return the results in case order.

    Args:
        cases: What to run.
        models: The models to answer with.
        offline: Run on the deterministic paths only.
        search_corpus: Whether campaigns may retrieve background.
        settings: Configuration to read the worker count from.

    Returns:
        One outcome per case, in the order the cases were given, whatever order
        they finished in.
    """
    if not cases:
        return ()
    resolved = get_settings() if settings is None else settings
    workers = min(resolved.eval_workers, len(cases))
    if workers <= 1:
        return tuple(
            run_one(case, models=models, offline=offline, search_corpus=search_corpus)
            for case in cases
        )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # ``map`` rather than ``as_completed``: it yields in submission order, so
        # the reordering the scorecard depends on is a property of the call rather
        # than something this function has to remember to do.
        return tuple(
            pool.map(
                lambda case: run_one(
                    case, models=models, offline=offline, search_corpus=search_corpus
                ),
                cases,
            )
        )


def summarise(outcomes: tuple[Outcome, ...]) -> dict[str, Any]:
    """Reduce a set of outcomes to the numbers a scorecard leads with.

    Args:
        outcomes: What the suite produced.

    Returns:
        Counts and averages, in primitives. Averages are taken over the cases that
        produced a number rather than over all of them, and the count they were
        taken over is reported alongside -- an average over three of ten cases is
        a different claim from an average over ten.
    """
    graded = [outcome for outcome in outcomes if outcome.ok and outcome.accuracy is not None]
    errors = [
        outcome.accuracy.error_per_site
        for outcome in graded
        if outcome.accuracy is not None and outcome.accuracy.error_per_site is not None
    ]
    return {
        "cases": len(outcomes),
        "ran": len(graded),
        "failed": sum(not outcome.ok for outcome in outcomes),
        "solved": sum(
            outcome.accuracy.solved for outcome in graded if outcome.accuracy is not None
        ),
        "measured": len(errors),
        "mean_error_per_site": sum(errors) / len(errors) if errors else None,
        "worst_error_per_site": max(errors) if errors else None,
        "seconds": round(sum(outcome.elapsed_s for outcome in outcomes), 1),
    }
