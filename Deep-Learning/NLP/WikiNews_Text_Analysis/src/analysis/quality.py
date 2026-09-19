"""Readability and style checks for the generated summaries.

Two kinds of check, neither needing a Java runtime:

* **Readability** -- Flesch reading ease and grade level, computed with the
  weights for the summary's own language.
* **Style faults** -- deterministic detection of a summary cut off
  mid-sentence, a repeated sentence, an opening pronoun with no antecedent, an
  instruction-tuned preamble, and an overlong sentence.

Results are reported per summarizer, because the two methods fail differently.
"""

from __future__ import annotations

import re

import pandas as pd
import textstat

from src.log import get_logger

logger = get_logger(__name__)

# Flesch weights differ by language; textstat ships coefficients for these.
_SUPPORTED_READABILITY = frozenset({"en", "es", "fr", "de"})

# A summary opening with one of these refers to something the reader cannot
# see: the sentence that introduced it was left out.
_OPENING_PRONOUNS = frozenset(
    {
        "he",
        "she",
        "it",
        "they",
        "this",
        "that",
        "these",
        "those",
        "él",
        "ella",
        "ellos",
        "esto",
        "esta",
        "il",
        "elle",
        "ils",
        "cela",
        "celui",
        "er",
        "sie",
        "es",
        "dies",
        "diese",
        "dieser",
    }
)

# Preambles instruction-tuned models add despite the instruction not to.
_PREAMBLE = re.compile(
    r"^\s*(here (is|are)|this (article|summary)|the article|summary:"
    r"|in summary|resumen:|zusammenfassung:)",
    re.IGNORECASE,
)

_TERMINAL_PUNCTUATION = (".", "!", "?", '"', "»", "”", "。")

# Beyond this a sentence is hard to read regardless of language.
_LONG_SENTENCE_WORDS = 40


def readability(text: str, lang: str) -> dict[str, float]:
    """Compute readability statistics in the text's own language.

    Args:
        text: The summary.
        lang: Two-letter language code.

    Returns:
        Flesch reading ease, Flesch-Kincaid grade and mean words per sentence.
        Values are ``float('nan')`` when the language has no Flesch weights.
    """
    words = text.split()
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    words_per_sentence = len(words) / len(sentences) if sentences else 0.0

    if lang not in _SUPPORTED_READABILITY or not text.strip():
        return {
            "flesch": float("nan"),
            "grade": float("nan"),
            "words_per_sentence": words_per_sentence,
        }

    try:
        textstat.set_lang(lang)
        return {
            "flesch": float(textstat.flesch_reading_ease(text)),
            "grade": float(textstat.flesch_kincaid_grade(text)),
            "words_per_sentence": words_per_sentence,
        }
    finally:
        textstat.set_lang("en")


def style_faults(text: str) -> dict[str, bool]:
    """Flag the specific style defects a summarizer can produce.

    Args:
        text: The summary.

    Returns:
        One boolean per fault: ``truncated``, ``repeated_sentence``,
        ``opens_with_pronoun``, ``has_preamble`` and ``overlong_sentence``.
    """
    stripped = text.strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", stripped) if s.strip()]
    first_word = re.sub(r"[^\w]", "", stripped.split()[0]).casefold() if stripped.split() else ""

    return {
        "truncated": bool(stripped) and not stripped.endswith(_TERMINAL_PUNCTUATION),
        "repeated_sentence": len(sentences) != len({s.casefold() for s in sentences}),
        "opens_with_pronoun": first_word in _OPENING_PRONOUNS,
        "has_preamble": bool(_PREAMBLE.match(stripped)),
        "overlong_sentence": any(len(s.split()) > _LONG_SENTENCE_WORDS for s in sentences),
    }


def assess(summaries: pd.DataFrame) -> pd.DataFrame:
    """Score every summary for readability and style.

    Args:
        summaries: The summary table, with ``summary``, ``lang``, ``method``
            and ``category`` columns.

    Returns:
        The identifying columns plus one column per readability statistic and
        per style fault.
    """
    rows: list[dict[str, object]] = []
    for row in summaries.itertuples():
        text = str(row.summary)
        rows.append(
            {
                "pageid": row.pageid,
                "lang": row.lang,
                "category": row.category,
                "method": row.method,
                **readability(text, str(row.lang)),
                **style_faults(text),
            }
        )
    return pd.DataFrame(rows)


def fault_rates(assessed: pd.DataFrame) -> pd.DataFrame:
    """Summarise how often each style fault occurs, per summarizer.

    Args:
        assessed: Output of ``assess``.

    Returns:
        Percentage of summaries showing each fault, methods as rows.
    """
    faults = [
        "truncated",
        "repeated_sentence",
        "opens_with_pronoun",
        "has_preamble",
        "overlong_sentence",
    ]
    return (assessed.groupby("method", observed=True)[faults].mean() * 100).round(1)
