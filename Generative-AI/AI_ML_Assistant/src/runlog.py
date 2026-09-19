"""Simple JSONL file logger for tracking application runs.

The pipeline creates a trace for each request. This module saves a small,
structured summary of each request as one JSON object per line. These logs can
later be viewed, searched, or analysed without needing a database or external
monitoring tool.

Design rules:

* **Best-effort, never fatal.** If logging fails (for example, due to a
  read-only disk or permission issues), the error is ignored and the function
  returns ``False``. Logging problems must never stop the user from getting an
  answer.

* **No secrets.** Logs only store routing details, request results, and usage
  information, along with a *shortened* version of the question. It never stores
  API keys, request headers, or environment variables.

* **Framework-agnostic.** This module uses only Python standard libraries and
  does not import Streamlit. It works the same way in the app (`Streamlit`),
  standalone scripts, and tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT

# Local, git-ignored by convention; created on first write. One JSON object per line.
DEFAULT_RUN_LOG_PATH = PROJECT_ROOT / "logs" / "runs.jsonl"


def now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string, for stamping records."""
    return datetime.now(UTC).isoformat()


@dataclass
class RunLogger:
    """Appends run records to a JSONL file, creating the parent directory on demand."""

    path: Path = DEFAULT_RUN_LOG_PATH
    enabled: bool = True

    def log(self, record: dict[str, Any]) -> bool:
        """Append one record as a JSON line. Returns True on success, False on any failure."""
        if not self.enabled:
            return False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(record, default=str, ensure_ascii=False)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception:
            return False  # observability must never break the request
        return True


# Module-level default the service logs through; tests can pass their own RunLogger instead.
_DEFAULT_LOGGER = RunLogger()


def log_run(
    record: dict[str, Any], *, enabled: bool = True, logger: RunLogger | None = None
) -> bool:
    """Log ``record`` through ``logger`` (or the module default) when ``enabled``.

    ``enabled`` mirrors ``settings.enable_run_log`` so the caller need not branch; a disabled
    call is a cheap no-op returning ``False``.
    """
    if not enabled:
        return False
    return (logger or _DEFAULT_LOGGER).log(record)
