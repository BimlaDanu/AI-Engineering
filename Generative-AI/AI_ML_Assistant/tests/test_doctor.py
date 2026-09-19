"""Offline tests for the ``make doctor`` readiness report (:mod:`src.doctor`).

Two properties are worth pinning, and they are the two that would matter if the report were
wrong:

* **It never prints a credential.** The report exists to be pasted into an issue or a README,
  and a report that leaks a key is worse than no report.
* **Drift is reported in both directions.** A file indexed but deleted from ``data/`` is the
  dangerous one — its chunks stay retrievable and citable until the next ``make ingest``, so
  an answer cites a note that no longer exists.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src import doctor


def _fake_index(root: Path, sources: dict[str, int]) -> None:
    """Write the one table :func:`doctor.indexed_sources` reads, with nothing else in it."""
    root.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(root / "chroma.sqlite3") as conn:
        conn.execute("create table embedding_metadata (id integer, key text, string_value text)")
        for name, chunks in sources.items():
            conn.executemany(
                "insert into embedding_metadata values (?, 'source', ?)",
                [(i, name) for i in range(chunks)],
            )


@pytest.fixture
def index_at(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the report's index and data directories at a temporary pair."""

    def _build(indexed: dict[str, int], on_disk: list[str]) -> None:
        chroma, data = tmp_path / "chroma_db", tmp_path / "data"
        if indexed:
            _fake_index(chroma, indexed)
        (data / "ml").mkdir(parents=True, exist_ok=True)
        for name in on_disk:
            (data / "ml" / name).write_text("# note", encoding="utf-8")
        monkeypatch.setattr(doctor, "CHROMA_DIR", chroma)
        monkeypatch.setattr(doctor, "DATA_DIR", data)

    return _build


def _find(label: str) -> list[doctor.Check]:
    return [c for c in doctor.collect() if c.label == label]


# --- Reading the index ------------------------------------------------------------------------


def test_chunk_counts_come_back_per_source(index_at):
    index_at({"attention.md": 3, "cnn.md": 2}, ["attention.md", "cnn.md"])
    assert doctor.indexed_sources() == {"attention.md": 3, "cnn.md": 2}


def test_a_missing_index_reads_as_empty_rather_than_raising(index_at):
    index_at({}, ["attention.md"])
    assert doctor.indexed_sources() == {}


def test_an_unreadable_index_reads_as_empty_rather_than_raising(tmp_path, monkeypatch):
    # doctor is what you run when something is already broken; it must survive a corrupt store.
    chroma = tmp_path / "chroma_db"
    chroma.mkdir()
    (chroma / "chroma.sqlite3").write_text("not a database", encoding="utf-8")
    monkeypatch.setattr(doctor, "CHROMA_DIR", chroma)
    assert doctor.indexed_sources() == {}


# --- Drift ------------------------------------------------------------------------------------


def test_no_index_is_reported_as_the_blocking_problem(index_at):
    index_at({}, ["attention.md"])
    check = _find("Knowledge base")[0]
    assert check.status == "warn"
    assert "make ingest" in check.fix


def test_a_matching_index_reports_no_drift(index_at):
    index_at({"attention.md": 3}, ["attention.md"])
    assert [c.status for c in _find("Index drift")] == ["ok"]


def test_a_deleted_note_still_in_the_index_is_flagged(index_at):
    # The dangerous direction: until the next ingest those chunks are still citable.
    index_at({"attention.md": 3, "deleted.md": 2}, ["attention.md"])
    drift = _find("Index drift")[0]
    assert drift.status == "warn"
    assert "deleted.md" in drift.detail
    assert "citable" in drift.fix


def test_a_new_note_not_yet_indexed_is_flagged(index_at):
    index_at({"attention.md": 3}, ["attention.md", "brand_new.md"])
    drift = _find("Index drift")[0]
    assert drift.status == "warn"
    assert "brand_new.md" in drift.detail


def test_the_eval_golden_set_is_not_mistaken_for_a_knowledge_note(tmp_path, monkeypatch):
    data = tmp_path / "data" / "eval"
    data.mkdir(parents=True)
    (data / "golden.txt").write_text("not a note", encoding="utf-8")
    monkeypatch.setattr(doctor, "DATA_DIR", tmp_path / "data")
    assert doctor.kb_files() == set()


# --- Secrets --------------------------------------------------------------------------------


def test_the_report_names_credentials_but_never_prints_one(index_at, monkeypatch):
    index_at({"attention.md": 1}, ["attention.md"])
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-supersecret-value")
    text = doctor.report()
    assert "OPENROUTER_API_KEY" in text
    assert "supersecret" not in text


def test_absent_credentials_are_a_warning_with_a_fix(index_at, monkeypatch):
    index_at({"attention.md": 1}, ["attention.md"])
    for name in ("OPENROUTER_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    check = _find("Credentials")[0]
    assert check.status == "warn"
    assert ".env" in check.fix


def test_a_configured_auth_section_is_named_without_its_values(index_at, tmp_path, monkeypatch):
    secrets = tmp_path / ".streamlit"
    secrets.mkdir()
    (secrets / "secrets.toml").write_text(
        '[auth]\nclient_secret = "must-not-appear"\n', encoding="utf-8"
    )
    index_at({"attention.md": 1}, ["attention.md"])
    monkeypatch.setattr(doctor, "PROJECT_ROOT", tmp_path)
    text = doctor.report()
    assert "native OIDC" in text
    assert "must-not-appear" not in text
