"""Read the raw WikiNews file into one tidy table.

The raw JSONL needs four fixes before anything else can use it:

1. ``text`` is a list of paragraphs, not a string.
2. ``categories`` mixes the 13 topic labels with place names and date strings.
3. ``date`` is ISO format only on English pages; other pages carry a
   local-format string that no time analysis can use.
4. Some articles have no text at all.

Fix 3 works because articles sharing a ``pageid`` describe the same event, so
the English page's date applies to its translations.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd

from src import config
from src.log import get_logger

logger = get_logger(__name__)

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def read_raw_records(path: Path | None = None) -> Iterator[dict[str, Any]]:
    """Yield one dictionary per line of the raw JSONL corpus.

    Args:
        path: Corpus file. Defaults to the path in ``config``.

    Yields:
        The raw record, unmodified.

    Raises:
        FileNotFoundError: If the corpus has not been fetched yet.
    """
    corpus_path = path or config.RAW_CORPUS
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found at {corpus_path}. Run `make data` first.")

    with corpus_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def extract_topics(categories: list[str]) -> list[str]:
    """Keep only the entries of ``categories`` that are topic labels.

    Args:
        categories: The raw ``categories`` list from a record.

    Returns:
        The topic labels present, in the corpus's own order.
    """
    return [c for c in categories if c in config.TOPIC_LABELS]


def join_paragraphs(paragraphs: list[str]) -> str:
    """Join an article's paragraph list into a single string.

    Args:
        paragraphs: The raw ``text`` list.

    Returns:
        The paragraphs separated by blank lines, stripped.
    """
    return "\n\n".join(p.strip() for p in paragraphs if p and p.strip()).strip()


def _iso_date_by_pageid(records: list[dict[str, Any]]) -> dict[int, str]:
    """Map each ``pageid`` to the ISO date of its English article.

    Args:
        records: Every raw record.

    Returns:
        ``pageid`` to ``YYYY-MM-DD``, for the pages that have a usable date.
    """
    dates: dict[int, str] = {}
    for record in records:
        date = record.get("date") or ""
        if record.get("lang") == "en" and _ISO_DATE.match(date):
            dates[record["pageid"]] = date
    return dates


def load_articles(path: Path | None = None) -> pd.DataFrame:
    """Load the whole corpus into a normalised table.

    Applies the four fixes described in the module docstring. Filtering by
    category or language happens in the corpus stage, not here.

    Args:
        path: Corpus file. Defaults to the path in ``config``.

    Returns:
        One row per article, with text joined, topics separated from the other
        categories, and an ISO date backfilled from the English page.
    """
    records = list(read_raw_records(path))
    logger.info("read %d raw records", len(records))

    iso_dates = _iso_date_by_pageid(records)
    logger.info(
        "ISO dates available for %d of %d events",
        len(iso_dates),
        len({r["pageid"] for r in records}),
    )

    rows: list[dict[str, Any]] = []
    for record in records:
        text = join_paragraphs(record.get("text") or [])
        if not text:
            continue
        date = iso_dates.get(record["pageid"])
        rows.append(
            {
                "pageid": record["pageid"],
                "lang": record["lang"],
                "title": record["title"],
                "text": text,
                "topics": extract_topics(record.get("categories") or []),
                "url": record.get("url", ""),
                "date": pd.Timestamp(date) if date else pd.NaT,
                "n_paragraphs": len(record.get("text") or []),
                "n_chars": len(text),
            }
        )

    frame = pd.DataFrame(rows)
    frame["year"] = frame["date"].dt.year.astype("Int64")
    logger.info(
        "kept %d articles with text (%d dropped as empty)", len(frame), len(records) - len(frame)
    )
    return frame
