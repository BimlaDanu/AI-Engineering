"""Orchestrate an evaluation run: answer each sample, score it, aggregate the report.

The runner depends only on two injected seams — an ``answer_fn`` that turns a
:class:`~src.eval.dataset.GoldenSample` into ``(answer, contexts)`` and a
:class:`~src.eval.judge.Judge` — so it is fully testable offline with fakes for both. In the
app, ``answer_fn`` wraps :meth:`src.core.service.AssistantService.answer` (reading
``bundle.text`` and ``bundle.contexts``) and the judge is an
:class:`~src.eval.judge.LLMJudge`.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from statistics import mean

from src.eval.dataset import GoldenSample
from src.eval.judge import Judge
from src.eval.metrics import METRICS, MetricResult, evaluate_sample

# Produce a (answer_text, retrieved_contexts) pair for one golden sample.
AnswerFn = Callable[[GoldenSample], tuple[str, list[str]]]

# Called after each unit of work completes as ``(done, total)`` for UI progress bars.
ProgressFn = Callable[[int, int], None]


@dataclass
class SampleEvaluation:
    """One golden sample's generated answer plus its per-metric scores."""

    sample: GoldenSample
    answer: str
    contexts: list[str]
    metrics: list[MetricResult]

    @property
    def scores(self) -> dict[str, float | None]:
        """Metric-name -> score (None where the metric was undefined for this sample)."""
        return {m.name: m.score for m in self.metrics}


@dataclass
class EvalReport:
    """A full evaluation run: per-sample results plus mean scores per metric."""

    results: list[SampleEvaluation]
    aggregates: dict[str, float] = field(default_factory=dict)
    answer_model: str = ""
    judge_model: str = ""

    def as_rows(self) -> list[dict[str, object]]:
        """Flatten to one row per sample (for a dataframe or CSV export)."""
        rows: list[dict[str, object]] = []
        for r in self.results:
            row: dict[str, object] = {
                "question": r.sample.question,
                "subject": r.sample.subject,
                "level": r.sample.level,
            }
            for m in METRICS:
                score = r.scores.get(m["name"])
                row[m["name"]] = None if score is None else round(score, 3)
            rows.append(row)
        return rows


def _aggregate(results: list[SampleEvaluation]) -> dict[str, float]:
    """Mean of each metric across the samples where it was defined (non-None)."""
    aggregates: dict[str, float] = {}
    for metric in METRICS:
        name = metric["name"]
        values = [r.scores[name] for r in results if r.scores.get(name) is not None]
        if values:
            aggregates[name] = mean(values)  # type: ignore[arg-type]
    return aggregates


def _evaluate_one(
    answer_fn: AnswerFn,
    judge: Judge,
    sample: GoldenSample,
    parallel_metrics: bool,
) -> SampleEvaluation:
    """Answer and score a single golden sample (the unit of work the pools schedule).

    A sample whose ``answer_fn`` raises is recorded with an empty answer/context and scored
    as usual, so a single pipeline failure never sinks the whole run.
    """
    try:
        answer, contexts = answer_fn(sample)
    except Exception as exc:  # a failed answer becomes a zero-context sample, not a crash
        answer, contexts = f"(answer failed: {exc})", []
    metrics = evaluate_sample(
        judge,
        sample.question,
        answer,
        contexts,
        sample.reference,
        parallel=parallel_metrics,
    )
    return SampleEvaluation(sample, answer, contexts, metrics)


def run_evaluations(
    jobs: list[tuple[str, AnswerFn]],
    judge: Judge,
    samples: list[GoldenSample],
    progress: ProgressFn | None = None,
    *,
    max_workers: int = 1,
    parallel_metrics: bool = False,
) -> dict[str, EvalReport]:
    """Evaluate several named answer functions over one golden set and shared judge.

    Every ``(job, sample)`` pair is an independent unit of work, so they are all scheduled
    into a **single** thread pool — this is what lets an A/B run overlap both variants (and
    every sample within them) instead of running one variant fully before the other.

    Args:
        jobs: ``(key, answer_fn)`` pairs; keys must be unique and label the returned reports.
        judge: The shared judgement backend (identical across jobs for a fair comparison).
        samples: The golden set every job is evaluated over.
        progress: Optional ``(done, total)`` callback fired as each unit completes, where
            ``total`` is ``len(jobs) * len(samples)``. Always invoked on the calling thread.
        max_workers: Concurrent units. ``1`` (default) runs everything strictly sequentially
            — byte-identical to the original behaviour. ``>1`` fans out over a thread pool;
            because the work is network-bound (waiting on OpenRouter), this cuts wall-clock
            roughly linearly without loading the local CPU.
        parallel_metrics: When True, each sample's four metrics are also judged concurrently.

    Returns:
        ``{key: EvalReport}`` — one aggregated report per job, each with its samples in the
        original golden-set order regardless of completion order.
    """
    total = len(jobs) * len(samples)
    # Per job, a slot list aligned to ``samples`` order so results stay deterministic.
    slots: dict[str, list[SampleEvaluation | None]] = {
        key: [None] * len(samples) for key, _ in jobs
    }
    units = [
        (key, answer_fn, index, sample)
        for key, answer_fn in jobs
        for index, sample in enumerate(samples)
    ]

    done = 0
    if max_workers <= 1:
        for key, answer_fn, index, sample in units:
            slots[key][index] = _evaluate_one(answer_fn, judge, sample, parallel_metrics)
            done += 1
            if progress is not None:
                progress(done, total)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_slot = {
                pool.submit(_evaluate_one, answer_fn, judge, sample, parallel_metrics): (key, index)
                for key, answer_fn, index, sample in units
            }
            for future in as_completed(future_to_slot):
                key, index = future_to_slot[future]
                slots[key][index] = future.result()  # re-raises a hard per-unit failure
                done += 1
                if progress is not None:
                    progress(done, total)

    reports: dict[str, EvalReport] = {}
    for key, _ in jobs:
        ordered = [r for r in slots[key] if r is not None]
        reports[key] = EvalReport(results=ordered, aggregates=_aggregate(ordered))
    return reports


def run_evaluation(
    answer_fn: AnswerFn,
    judge: Judge,
    samples: list[GoldenSample],
    progress: ProgressFn | None = None,
    *,
    max_workers: int = 1,
    parallel_metrics: bool = False,
) -> EvalReport:
    """Evaluate every sample and return an aggregated :class:`EvalReport`.

    A thin single-job wrapper over :func:`run_evaluations`. With the default
    ``max_workers=1`` this is strictly sequential and identical to the original runner;
    raise ``max_workers`` (and optionally ``parallel_metrics``) to fan the samples out over
    a thread pool.

    Args:
        answer_fn: Produces ``(answer, contexts)`` for a sample (wraps the real pipeline).
        judge: The judgement backend (an :class:`~src.eval.judge.LLMJudge`, or a stub).
        samples: The golden set to evaluate.
        progress: Optional ``(done, total)`` callback fired as each sample completes.
        max_workers: Concurrent samples (``1`` = sequential; see :func:`run_evaluations`).
        parallel_metrics: When True, judge each sample's four metrics concurrently.
    """
    reports = run_evaluations(
        [("", answer_fn)],
        judge,
        samples,
        progress,
        max_workers=max_workers,
        parallel_metrics=parallel_metrics,
    )
    return reports[""]
