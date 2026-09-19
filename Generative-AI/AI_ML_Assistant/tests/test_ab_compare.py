"""Offline tests for the A/B comparison layer (:mod:`src.eval.compare`).

The comparator is pure — it diffs two already-computed :class:`~src.eval.runner.EvalReport`
aggregates — so these build reports by hand (no LLM, no pipeline) and pin the behaviours that
matter: per-metric winner selection, the tie margin, undefined (None) metrics, the overall
verdict from win tallies, and the row flattening used by the UI/export.
"""

from __future__ import annotations

from src.eval.compare import (
    TIE,
    WINNER_A,
    WINNER_B,
    Variant,
    compare_reports,
)
from src.eval.runner import EvalReport


def _report(aggregates: dict[str, float]) -> EvalReport:
    """A minimal report carrying only the aggregate scores the comparator reads."""
    return EvalReport(results=[], aggregates=aggregates)


VA = Variant("A", "model-a · k=4")
VB = Variant("B", "model-b · k=6")


def test_b_wins_every_metric() -> None:
    a = _report(
        {
            "faithfulness": 0.5,
            "answer_relevancy": 0.5,
            "context_precision": 0.5,
            "context_recall": 0.5,
        }
    )
    b = _report(
        {
            "faithfulness": 0.9,
            "answer_relevancy": 0.8,
            "context_precision": 0.7,
            "context_recall": 0.6,
        }
    )
    result = compare_reports(a, b, VA, VB)
    assert result.overall_winner == WINNER_B
    assert result.wins_b == 4 and result.wins_a == 0
    assert all(m.winner == WINNER_B for m in result.metrics)


def test_a_wins_when_higher() -> None:
    a = _report({"faithfulness": 0.9})
    b = _report({"faithfulness": 0.4})
    result = compare_reports(a, b, VA, VB)
    assert result.overall_winner == WINNER_A
    faith = next(m for m in result.metrics if m.name == "faithfulness")
    assert faith.winner == WINNER_A
    assert faith.delta == -0.5  # b - a


def test_scores_within_tie_margin_are_a_tie() -> None:
    a = _report({"faithfulness": 0.80})
    b = _report({"faithfulness": 0.805})  # 0.005 gap, inside the default 0.01 eps
    result = compare_reports(a, b, VA, VB)
    faith = next(m for m in result.metrics if m.name == "faithfulness")
    assert faith.winner == TIE
    assert result.overall_winner == TIE  # no decided metric


def test_split_decision_counts_wins() -> None:
    a = _report({"faithfulness": 0.9, "answer_relevancy": 0.2})
    b = _report({"faithfulness": 0.2, "answer_relevancy": 0.9})
    result = compare_reports(a, b, VA, VB)
    assert result.wins_a == 1 and result.wins_b == 1
    assert result.overall_winner == TIE


def test_undefined_metric_has_no_winner_and_no_delta() -> None:
    # context_recall is absent from B (e.g. no reference claims), so it can't be decided.
    a = _report({"faithfulness": 0.6, "context_recall": 0.7})
    b = _report({"faithfulness": 0.8})
    result = compare_reports(a, b, VA, VB)
    recall = next(m for m in result.metrics if m.name == "context_recall")
    assert recall.winner is None
    assert recall.delta is None
    assert recall.b is None
    # Only the decided metric (faithfulness, B higher) counts toward the verdict.
    assert result.overall_winner == WINNER_B and result.wins_b == 1


def test_as_rows_uses_variant_labels_and_rounds() -> None:
    a = _report({"faithfulness": 0.123456})
    b = _report({"faithfulness": 0.987654})
    rows = compare_reports(a, b, VA, VB).as_rows()
    faith_row = next(r for r in rows if r["metric"] == "faithfulness")
    assert faith_row["A"] == 0.123  # rounded, keyed by the variant label
    assert faith_row["B"] == 0.988
    assert faith_row["winner"] == WINNER_B


def test_empty_aggregates_is_an_overall_tie() -> None:
    result = compare_reports(_report({}), _report({}), VA, VB)
    assert result.overall_winner == TIE
    assert result.wins_a == 0 and result.wins_b == 0
    assert all(m.winner is None for m in result.metrics)
