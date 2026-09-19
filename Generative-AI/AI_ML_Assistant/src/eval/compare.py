"""Compare two evaluation runs — the A/B testing of RAG strategies (hard optional #2).

This is a *pure* comparison layer over :class:`~src.eval.runner.EvalReport`: given two runs
produced by the **same** golden subset and the **same** judge but **different**
configurations (variant *A* vs *B*), it diffs their aggregate metric scores and picks a
per-metric and an overall winner. No LLM, no Streamlit, no I/O — so the whole comparison is
unit-testable with hand-built reports.

It reuses the existing harness wholesale: the UI runs both variants through
:func:`~src.eval.runner.run_evaluations` (a single shared thread pool that overlaps the two
variants rather than running one fully before the other), then hands the two resulting
reports here. All four RAGAs metrics are higher-is-better, so "winner" is simply the higher
aggregate (within a tie margin).

Fairness is the *caller's* contract: an A/B result is only meaningful when both reports
scored the same questions with the same judge, differing solely in the variant configuration
under test. The UI enforces this by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.eval.metrics import METRICS
from src.eval.runner import EvalReport

# Outcome labels, used as return values and rendered verbatim in the UI.
WINNER_A = "A"
WINNER_B = "B"
TIE = "tie"


@dataclass(frozen=True)
class Variant:
    """One side of an A/B test: a short label plus a human-readable config summary."""

    label: str
    config: str


@dataclass(frozen=True)
class MetricComparison:
    """One metric's A-vs-B scores, their signed difference, and the higher scorer."""

    name: str
    a: float | None
    b: float | None
    delta: float | None  # b - a: positive => B higher. None if either side is undefined.
    winner: str | None  # WINNER_A / WINNER_B / TIE, or None when a side is undefined.


@dataclass(frozen=True)
class ABComparison:
    """The full head-to-head: per-metric rows, win tallies, and an overall verdict."""

    variant_a: Variant
    variant_b: Variant
    metrics: list[MetricComparison]
    wins_a: int
    wins_b: int
    overall_winner: str  # WINNER_A / WINNER_B / TIE

    def as_rows(self) -> list[dict[str, object]]:
        """Flatten to one row per metric (for a dataframe or CSV export)."""
        return [
            {
                "metric": m.name,
                self.variant_a.label: None if m.a is None else round(m.a, 3),
                self.variant_b.label: None if m.b is None else round(m.b, 3),
                "delta": None if m.delta is None else round(m.delta, 3),
                "winner": m.winner or "—",
            }
            for m in self.metrics
        ]


def _pick_winner(a: float | None, b: float | None, tie_eps: float) -> str | None:
    """Higher score wins; ``None`` if either side is undefined; ``TIE`` within ``tie_eps``."""
    if a is None or b is None:
        return None
    if abs(b - a) <= tie_eps:
        return TIE
    return WINNER_B if b > a else WINNER_A


def compare_reports(
    report_a: EvalReport,
    report_b: EvalReport,
    variant_a: Variant,
    variant_b: Variant,
    *,
    tie_eps: float = 0.01,
) -> ABComparison:
    """Diff two aggregated reports metric-by-metric and decide an overall winner.

    Args:
        report_a: The evaluation report for variant A.
        report_b: The evaluation report for variant B (same golden set + judge as A).
        variant_a: Label/config metadata describing variant A.
        variant_b: Label/config metadata describing variant B.
        tie_eps: Absolute score gap within which a metric is called a tie (default 0.01).

    Returns:
        An :class:`ABComparison`. The overall winner is whichever variant wins more metrics;
        an equal number of wins (including zero decided metrics) is an overall tie.
    """
    metrics: list[MetricComparison] = []
    wins_a = wins_b = 0
    for metric in METRICS:
        name = metric["name"]
        a = report_a.aggregates.get(name)
        b = report_b.aggregates.get(name)
        delta = None if (a is None or b is None) else b - a
        winner = _pick_winner(a, b, tie_eps)
        if winner == WINNER_A:
            wins_a += 1
        elif winner == WINNER_B:
            wins_b += 1
        metrics.append(MetricComparison(name=name, a=a, b=b, delta=delta, winner=winner))

    if wins_a > wins_b:
        overall = WINNER_A
    elif wins_b > wins_a:
        overall = WINNER_B
    else:
        overall = TIE
    return ABComparison(
        variant_a=variant_a,
        variant_b=variant_b,
        metrics=metrics,
        wins_a=wins_a,
        wins_b=wins_b,
        overall_winner=overall,
    )
