"""Load the curated golden evaluation set (question + reference answer pairs).

The golden set lives at ``data/eval/golden.jsonl`` — one JSON object per line, so it is easy
to hand-edit and diff. Each line provides a ``question`` and a ``reference`` answer, plus an
optional ``subject`` (a :data:`src.config.SUBJECTS` label, defaulting to "All subjects") and
``level`` (a :data:`src.config.LEVELS` value). Blank lines and ``#`` comment lines are
ignored so the file can carry section headers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.config import DATA_DIR, LEVELS, SUBJECTS

GOLDEN_SET_PATH = DATA_DIR / "eval" / "golden.jsonl"


@dataclass(frozen=True)
class GoldenSample:
    """One evaluation case: a question, its reference answer, and retrieval scoping."""

    question: str
    reference: str
    subject: str = "All subjects"
    level: str = "Practitioner"


def _coerce(record: dict, line_no: int) -> GoldenSample:
    """Validate one parsed JSON record into a :class:`GoldenSample`."""
    try:
        question = str(record["question"]).strip()
        reference = str(record["reference"]).strip()
    except KeyError as exc:
        raise ValueError(f"golden.jsonl line {line_no}: missing key {exc}") from exc
    if not question or not reference:
        raise ValueError(f"golden.jsonl line {line_no}: empty question or reference")
    subject = record.get("subject", "All subjects")
    if subject not in SUBJECTS:
        raise ValueError(f"golden.jsonl line {line_no}: unknown subject {subject!r}")
    level = record.get("level", "Practitioner")
    if level not in LEVELS:
        raise ValueError(f"golden.jsonl line {line_no}: unknown level {level!r}")
    return GoldenSample(question=question, reference=reference, subject=subject, level=level)


def load_golden_set(path: Path | None = None) -> list[GoldenSample]:
    """Read and validate the golden set from ``path`` (defaults to the packaged file).

    Raises:
        FileNotFoundError: if the golden-set file does not exist.
        ValueError: if any line is malformed (bad JSON, missing/empty fields, unknown
            subject or level).
    """
    path = path or GOLDEN_SET_PATH
    if not path.exists():
        raise FileNotFoundError(f"Golden set not found at {path}")
    samples: list[GoldenSample] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"golden.jsonl line {line_no}: invalid JSON ({exc})") from exc
        samples.append(_coerce(record, line_no))
    return samples
