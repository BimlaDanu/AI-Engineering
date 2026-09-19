"""Offline tests for the Analytics dashboard's pure row-extraction helper.

``_answer_rows`` turns raw chat history into per-answer token/cost rows for charting. It is
the only non-Streamlit logic on the page, so it is unit-tested directly; importing the page
module is guarded by ``importorskip`` in case Streamlit is absent from a minimal environment.
"""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")

from src.ui.pages.analytics import _answer_rows


def _assistant(model: str, tin: int, tout: int, cost: float) -> dict:
    return {
        "role": "assistant",
        "content": "answer",
        "meta": {
            "model": model,
            "level": "Beginner",
            "tokens_in": tin,
            "tokens_out": tout,
            "cost": cost,
        },
    }


def test_empty_history_yields_no_rows() -> None:
    assert _answer_rows([]) == []


def test_skips_user_turns_and_assistant_turns_without_meta() -> None:
    history = [
        {"role": "user", "content": "q"},
        _assistant("m", 10, 5, 0.001),
        {"role": "assistant", "content": "refusal — no meta"},  # early-exit, excluded
    ]
    rows = _answer_rows(history)
    assert len(rows) == 1
    assert rows[0]["Answer"] == "#1"
    assert rows[0]["Tokens in"] == 10 and rows[0]["Tokens out"] == 5
    assert rows[0]["Model"] == "m"


def test_rows_are_numbered_sequentially() -> None:
    rows = _answer_rows([_assistant("m", 1, 1, 0.0), _assistant("m", 2, 2, 0.0)])
    assert [r["Answer"] for r in rows] == ["#1", "#2"]
    assert rows[1]["Tokens in"] == 2


def test_cost_is_rounded_for_display() -> None:
    rows = _answer_rows([_assistant("m", 1, 1, 0.123456789)])
    assert rows[0]["Cost ($)"] == round(0.123456789, 6)
