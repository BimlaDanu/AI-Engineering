"""Offline tests for the structured JSONL run-log.

All writes target a pytest ``tmp_path`` so nothing touches the real ``logs/`` directory. The
tests pin the append-one-line-per-record contract, parent-directory creation, the enabled
gate, and — most importantly — that a write failure is swallowed and never raised (logging
must never break a request).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.runlog import RunLogger, log_run, now_iso


def test_appends_one_json_object_per_line(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    logger = RunLogger(path=path)
    assert logger.log({"a": 1})
    assert logger.log({"b": 2})
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(line) for line in lines] == [{"a": 1}, {"b": 2}]


def test_creates_missing_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "deep" / "runs.jsonl"
    assert RunLogger(path=path).log({"x": 1})
    assert path.exists()


def test_disabled_logger_is_a_noop(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    assert RunLogger(path=path, enabled=False).log({"a": 1}) is False
    assert not path.exists()


def test_log_run_respects_the_enabled_flag(tmp_path: Path) -> None:
    logger = RunLogger(path=tmp_path / "runs.jsonl")
    assert log_run({"a": 1}, enabled=False, logger=logger) is False
    assert log_run({"a": 1}, enabled=True, logger=logger) is True


def test_write_failure_is_swallowed(tmp_path: Path) -> None:
    # Point the log at an existing directory: opening it for append fails, but log() must
    # return False rather than raise — observability can never sink the request.
    assert RunLogger(path=tmp_path).log({"a": 1}) is False


def test_now_iso_is_iso_parseable() -> None:
    datetime.fromisoformat(now_iso())  # raises if the stamp is malformed
