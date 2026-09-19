"""Offline tests for conversation export (JSON / CSV / PDF).

The JSON, CSV, and record-flattening paths are pure and always tested. The PDF path needs
the optional ``fpdf2`` dependency, so its test is skipped (``importorskip``) until the
environment is synced — mirroring how the app degrades to a disabled PDF button.
"""

from __future__ import annotations

import csv
import io
import json

import pytest

from src.export import (
    conversation_to_csv,
    conversation_to_json,
    conversation_to_pdf,
    to_records,
)

# A representative history: a user turn, a rich assistant turn, and an early-exit assistant
# turn with no meta/sources (e.g. a refusal) — the exporters must handle all three.
HISTORY = [
    {"role": "user", "content": "What is an embedding?"},
    {
        "role": "assistant",
        "content": "An embedding is a dense vector [1].",
        "question": "What is an embedding?",
        "sources": [
            {
                "ref": 1,
                "score": 0.82,
                "title": "Embeddings",
                "topic": "nlp",
                "difficulty": "beginner",
                "subject": "overlap",
            },
        ],
        "tools": [],
        "no_rag": None,
        "meta": {
            "model": "openai/gpt-4o-mini",
            "level": "Beginner",
            "tokens_in": 120,
            "tokens_out": 45,
            "cost": 0.00012,
        },
    },
    {"role": "assistant", "content": "I'll pass on that one."},
]


def test_to_records_flattens_meta_and_sources() -> None:
    records = to_records(HISTORY)
    assert [r.role for r in records] == ["user", "assistant", "assistant"]
    rich = records[1]
    assert rich.model == "openai/gpt-4o-mini"
    assert rich.tokens_in == 120
    assert rich.sources and rich.sources[0].startswith("[1] Embeddings")
    # The early-exit assistant turn has no metadata.
    assert records[2].model is None
    assert records[2].sources == []


def test_conversation_to_json_roundtrips() -> None:
    restored = json.loads(conversation_to_json(HISTORY))
    assert restored == HISTORY


def test_conversation_to_csv_has_header_and_row_per_turn() -> None:
    rows = list(csv.reader(io.StringIO(conversation_to_csv(HISTORY))))
    assert rows[0] == [
        "turn",
        "role",
        "content",
        "model",
        "level",
        "tokens_in",
        "tokens_out",
        "cost_usd",
        "sources",
    ]
    assert len(rows) == 1 + len(HISTORY)  # header + one row per message
    # The rich assistant row carries flattened metadata and a source line.
    assistant_row = rows[2]
    assert assistant_row[3] == "openai/gpt-4o-mini"
    assert "Embeddings" in assistant_row[8]


def test_conversation_to_csv_handles_empty_history() -> None:
    assert conversation_to_csv([]).splitlines()[0].startswith("turn,role")


def test_csv_safe_neutralises_formula_injection() -> None:
    from src.export import _csv_safe

    # Leading formula/DDE trigger characters get a single-quote prefix so a spreadsheet
    # treats the cell as literal text instead of executing it.
    assert _csv_safe("=cmd|'/c calc'!A1") == "'=cmd|'/c calc'!A1"
    for payload in ("+1+1", "-2+3", "@SUM(A1)", "\tTAB", "\rCR"):
        assert _csv_safe(payload).startswith("'")
    # Benign content is untouched.
    assert _csv_safe("An embedding is a vector.") == "An embedding is a vector."
    assert _csv_safe("") == ""


def test_conversation_to_csv_escapes_dangerous_content() -> None:
    history = [{"role": "assistant", "content": '=HYPERLINK("http://evil")'}]
    rows = list(csv.reader(io.StringIO(conversation_to_csv(history))))
    # The content cell (index 2) must be quoted so Excel/Sheets won't evaluate it.
    assert rows[1][2].startswith("'=HYPERLINK")


def test_conversation_to_pdf_produces_a_pdf() -> None:
    pytest.importorskip("fpdf")  # skip until fpdf2 is synced into the environment
    data = conversation_to_pdf(HISTORY, title="Synapse")
    assert isinstance(data, bytes)
    assert data[:5] == b"%PDF-"  # PDF magic number
    assert len(data) > 500


def test_pdf_safe_maps_unicode_glyphs() -> None:
    from src.export import _pdf_safe

    assert _pdf_safe("a — b ≈ c") == "a - b ~ c"
    # An unrepresentable glyph (emoji) is dropped, not left as a crash or mojibake.
    assert "🚀" not in _pdf_safe("launch 🚀 now")


def test_json_and_records_handle_empty_history() -> None:
    assert conversation_to_json([]) == "[]"
    assert to_records([]) == []


def test_conversation_to_pdf_handles_empty_history() -> None:
    pytest.importorskip("fpdf")  # skip until fpdf2 is synced into the environment
    data = conversation_to_pdf([], title="Synapse")
    assert isinstance(data, bytes) and data[:5] == b"%PDF-"  # header renders with no turns


def test_conversation_to_pdf_survives_unicode_content() -> None:
    pytest.importorskip("fpdf")
    unicode_history = [
        {"role": "user", "content": "Explain ∇f, softmax σ, and 注意力 (attention) 🚀"},
        {
            "role": "assistant",
            "content": "The gradient ∇ points uphill; naïve façade — β ≥ α. CJK: 变压器. 🎯",
            "sources": [{"ref": 1, "title": "Attention — 注意力", "score": 0.9}],
            "meta": {
                "model": "m",
                "level": "Researcher",
                "tokens_in": 5,
                "tokens_out": 7,
                "cost": 0.0001,
            },
        },
    ]
    data = conversation_to_pdf(unicode_history, title="Synapse — 测试")
    assert isinstance(data, bytes) and data[:5] == b"%PDF-"  # no crash on non-Latin-1 glyphs
