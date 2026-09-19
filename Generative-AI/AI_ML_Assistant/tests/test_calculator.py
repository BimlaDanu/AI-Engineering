"""Offline tests for the SymPy calculator tool and its abuse guards.

The correctness core (:func:`_compute`) is tested inline (fast, no process). The input guards
(length cap, blocklist) are tested through the tool and return before any process is spawned.
The last two exercise the killable-worker path and the thing that makes it correct: the
evaluation budget covers the evaluation, and getting the worker up is charged separately.
"""

from __future__ import annotations

import pytest

from src.tools import calculator
from src.tools.calculator import (
    _MAX_EXPR_CHARS,
    _MAX_RESULT_CHARS,
    _compute,
    _compute_with_timeout,
    math_calculator,
)


def test_compute_evaluates_and_solves() -> None:
    assert "4" in _compute("2 + 2", "evaluate")
    # solve treats the expression as = 0; roots of x**2 - 5x + 6 are 2 and 3.
    solved = _compute("x**2 - 5*x + 6", "solve")
    assert "2" in solved and "3" in solved


def test_compute_reports_errors_gracefully() -> None:
    # A malformed expression is caught and returned as a message, never raised.
    assert _compute("2 +", "evaluate").startswith("Could not compute that:")


def test_result_is_size_capped() -> None:
    # A long symbolic result (301-term polynomial) exceeds the cap and is truncated.
    out = _compute("expand((x + 1)**300)", "evaluate")
    assert out.endswith("(result truncated)")
    assert len(out) < _MAX_RESULT_CHARS + 50


def test_huge_integer_result_does_not_crash() -> None:
    # A pathological bignum (str() would trip Python's 4300-digit limit) must return a
    # bounded message, never raise — this is the resource-exhaustion guard.
    out = _compute("2 ** 999999", "evaluate")
    assert out.startswith("Result:")
    assert len(out) < _MAX_RESULT_CHARS + 50


def test_tool_rejects_overlong_expression() -> None:
    result = math_calculator.invoke({"expression": "1+" * (_MAX_EXPR_CHARS)})
    assert "too long" in result.lower()


def test_tool_blocks_disallowed_syntax() -> None:
    for payload in ("__import__('os')", "import os", "eval('1')", "open('x')", "a; b"):
        assert "disallowed" in math_calculator.invoke({"expression": payload}).lower()


def test_tool_allows_evalf_numeric_evaluation() -> None:
    # Regression: the old substring blocklist rejected any expression containing "eval",
    # which killed SymPy's legitimate ``.evalf()`` numeric evaluation. It must now pass the
    # guard and return a numeric result (50-digit pi starts 3.1415926535…).
    result = math_calculator.invoke({"expression": "pi.evalf(50)"})
    assert "disallowed" not in result.lower()
    assert "3.1415926535" in result


def test_tool_evaluates_simple_expression_end_to_end() -> None:
    # Exercises the spawn/timeout path on a trivial expression.
    assert "4" in _compute_with_timeout("2 + 2", "evaluate")
    assert "4" in math_calculator.invoke({"expression": "2 + 2"})


def test_worker_startup_is_not_charged_to_the_evaluation_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The regression. A ``spawn`` worker is a fresh interpreter that re-imports the parent's
    # ``__main__`` — pytest here, the Streamlit CLI in the app — and that cost used to come out
    # of the evaluation budget, so ``integrate(2*x, (x, 0, 3))`` could be reported as "too
    # expensive to evaluate". Squeezing the budget to a second proves the split: far less than
    # the spawn costs, comfortably more than the integral does.
    monkeypatch.setattr(calculator, "_TIMEOUT_SECONDS", 1.0)
    assert "9" in _compute_with_timeout("integrate(2*x, (x, 0, 3))", "evaluate")


def test_an_overrunning_evaluation_is_reported_rather_than_left_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The guard the budget exists for, still intact: past it the worker is killed and the
    # caller gets an actionable message instead of a hang.
    monkeypatch.setattr(calculator, "_TIMEOUT_SECONDS", 0.001)
    out = _compute_with_timeout("integrate(exp(-x**2) * sin(x**3), (x, -oo, oo))", "evaluate")
    assert "timed out" in out.lower()


def test_the_two_budgets_are_not_interchangeable() -> None:
    # If these ever converge, the split above has been undone and the bug is back.
    assert calculator._STARTUP_SECONDS > calculator._TIMEOUT_SECONDS
