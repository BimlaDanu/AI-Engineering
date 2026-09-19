"""``make doctor`` — a read-only readiness report for this checkout.

Answers the questions that otherwise cost a wrong-looking app run: is there an index, does
it still match ``data/``, which credentials are present, where do saved conversations go.

Two rules shape the whole module:

* **It never spends anything.** No model is called and no embedding is computed. The index is
  inspected by reading ``chroma_db/chroma.sqlite3`` directly rather than by constructing a
  ``KnowledgeBase``, which would load an embedding backend and, on the API backend, need a key.
* **It never prints a secret.** Credentials are reported as present or absent by variable
  name. A report you cannot paste into an issue is a report nobody runs twice.

The drift check is the reason this exists. ``make ingest`` is a full wipe-and-rebuild, so
until it runs after a change under ``data/``, a deleted note's chunks stay live in the index
and remain retrievable and citable. That is invisible from the UI: the answer looks fine and
cites a file that is no longer there.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass

from src.config import CHROMA_DIR, DATA_DIR, EMBEDDING_BACKEND, PROJECT_ROOT
from src.core.chats import chat_dir

_KEY_NAMES = ("OPENROUTER_API_KEY", "GOOGLE_API_KEY")
_KB_SUFFIXES = (".md", ".txt", ".pdf")

_OK = "ok"
_WARN = "warn"
_INFO = "info"

_MARKS = {_OK: "✓", _WARN: "!", _INFO: "·"}


@dataclass(frozen=True)
class Check:
    """One line of the report: a heading, a verdict, and what to do about it."""

    label: str
    status: str
    detail: str
    fix: str = ""

    def render(self) -> str:
        line = f"  {_MARKS[self.status]} {self.label:<22} {self.detail}"
        return f"{line}\n{'':<27}→ {self.fix}" if self.fix else line


def indexed_sources() -> dict[str, int]:
    """Chunk count per source filename, straight from the Chroma sqlite file.

    Empty when there is no index yet, or when the schema is not the one this reads — an
    unreadable index is reported as empty rather than raised, because ``doctor`` is the
    command you run precisely when something is already wrong.
    """
    db = CHROMA_DIR / "chroma.sqlite3"
    if not db.exists():
        return {}
    try:
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
            rows = conn.execute(
                "select string_value, count(*) from embedding_metadata "
                "where key = 'source' group by 1"
            ).fetchall()
    except sqlite3.Error:
        return {}
    return {name: count for name, count in rows if name}


def kb_files() -> set[str]:
    """Every ingestable filename under ``data/``, excluding the eval golden set."""
    return {
        path.name
        for path in DATA_DIR.rglob("*")
        if path.suffix.lower() in _KB_SUFFIXES and "eval" not in path.parts
    }


def _index_check() -> list[Check]:
    indexed = indexed_sources()
    on_disk = kb_files()
    if not indexed:
        return [
            Check(
                "Knowledge base",
                _WARN,
                "no index found — every answer will degrade to 'no sources'",
                "make ingest",
            )
        ]

    checks = [
        Check(
            "Knowledge base",
            _OK,
            f"{sum(indexed.values())} chunks from {len(indexed)} files",
        )
    ]
    stale = sorted(set(indexed) - on_disk)
    missing = sorted(on_disk - set(indexed))
    if stale:
        checks.append(
            Check(
                "Index drift",
                _WARN,
                f"indexed but gone from data/: {', '.join(stale)}",
                "make ingest (their chunks are still retrievable and citable)",
            )
        )
    if missing:
        checks.append(
            Check(
                "Index drift",
                _WARN,
                f"in data/ but not indexed: {', '.join(missing)}",
                "make ingest",
            )
        )
    if not stale and not missing:
        checks.append(Check("Index drift", _OK, "index matches data/"))
    return checks


def _credential_checks() -> list[Check]:
    """Report which credentials are reachable. Never report their values."""
    present = [name for name in _KEY_NAMES if os.getenv(name)]
    if present:
        detail = f"{', '.join(present)} set"
        status = _OK
    else:
        detail = "no model credentials in the environment"
        status = _WARN
    return [
        Check(
            "Credentials",
            status,
            detail,
            "" if present else "put OPENROUTER_API_KEY in .env, or paste a key via 'Use a key'",
        ),
        Check("Embeddings", _INFO, f"EMBEDDING_BACKEND={EMBEDDING_BACKEND}"),
    ]


def _accounts_check() -> Check:
    """Which sign-in door is configured, by section name only."""
    secrets = PROJECT_ROOT / ".streamlit" / "secrets.toml"
    if not secrets.exists():
        return Check(
            "Accounts",
            _INFO,
            "no secrets.toml — the app runs open, login buttons render disabled",
            "optional: copy .streamlit/secrets.toml.example",
        )
    text = secrets.read_text(encoding="utf-8", errors="replace")
    sections = [
        name
        for name, marker in (("native OIDC", "[auth]"), ("password gate", "[password_auth]"))
        if any(line.strip().startswith(marker) for line in text.splitlines())
    ]
    if not sections:
        return Check("Accounts", _INFO, "secrets.toml present, no auth section in it")
    return Check("Accounts", _OK, f"secrets.toml configures: {', '.join(sections)}")


def collect() -> list[Check]:
    """Every check, in the order the report prints them."""
    saved = chat_dir()
    threads = len(list(saved.rglob("*.json"))) if saved.exists() else 0
    return [
        *_index_check(),
        *_credential_checks(),
        _accounts_check(),
        Check(
            "Saved chats",
            _INFO,
            f"{threads} stored under {saved}",
        ),
    ]


def report() -> str:
    """The full report as one string, so a test can read it without capturing stdout."""
    lines = ["", f"Synapse doctor — {PROJECT_ROOT.name}", ""]
    lines += [check.render() for check in collect()]
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    print(report())


if __name__ == "__main__":
    main()
