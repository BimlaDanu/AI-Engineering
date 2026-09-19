"""Offline tests for the knowledge-base availability notice (:mod:`src.ui.state`).

A missing/empty index and a *load failure* both leave the KB unavailable, but they need
different advice: ``make ingest`` fixes an empty index, whereas a load error (corrupt store,
unreadable ``chroma_db/``, embedding/dimension mismatch) must be surfaced so the user is not
misled into re-ingesting something that will not help. These pin that the notice picks the
right branch, without a real Chroma store or a Streamlit run context.
"""

from __future__ import annotations

from src.ui import state
from src.ui.state import KBStatus


def test_failed_flag_is_set_only_on_load_error() -> None:
    assert KBStatus(kb=None, error="boom").failed is True
    assert KBStatus(kb=None).failed is False


def _capture(monkeypatch) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    infos: list[str] = []
    monkeypatch.setattr(state.st, "error", errors.append)
    monkeypatch.setattr(state.st, "info", infos.append)
    return errors, infos


def test_notice_surfaces_the_cause_on_load_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        state, "load_kb_status", lambda: KBStatus(kb=None, error="embedding dim 384 != 1536")
    )
    errors, infos = _capture(monkeypatch)
    state.kb_status_notice("empty message")
    assert not infos
    assert errors and "embedding dim 384 != 1536" in errors[0]


def test_notice_shows_empty_message_when_index_is_empty(monkeypatch) -> None:
    monkeypatch.setattr(state, "load_kb_status", lambda: KBStatus(kb=None))
    errors, infos = _capture(monkeypatch)
    state.kb_status_notice("empty message")
    assert infos == ["empty message"]
    assert not errors


def test_notice_is_silent_when_kb_present(monkeypatch) -> None:
    monkeypatch.setattr(state, "load_kb_status", lambda: KBStatus(kb=object()))
    errors, infos = _capture(monkeypatch)
    state.kb_status_notice("empty message")
    assert not errors and not infos
