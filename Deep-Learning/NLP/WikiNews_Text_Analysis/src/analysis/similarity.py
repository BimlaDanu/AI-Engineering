"""Score summaries against the articles they came from.

Two scores per summary:

* **semantic similarity** -- cosine between the multilingual embedding of the
  summary and of the full source article;
* **lexical overlap** -- the share of the summary's words that appear in the
  source.

Both are needed to read the result. An extractive summary is made of the
source's own sentences, so its overlap is near 1.0 by construction and its
cosine is high for that reason rather than because it captured the article.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.analysis.embeddings import cosine_pairs, encode_documents
from src.log import get_logger

logger = get_logger(__name__)

_WORD = re.compile(r"\w+", re.UNICODE)


def lexical_overlap(summary: str, source: str) -> float:
    """Share of the summary's distinct words that also appear in the source.

    Args:
        summary: The summary text.
        source: The article it was made from.

    Returns:
        A value in ``[0, 1]``. 1.0 means every word was lifted from the source.
    """
    summary_words = set(_WORD.findall(summary.casefold()))
    if not summary_words:
        return 0.0
    source_words = set(_WORD.findall(source.casefold()))
    return len(summary_words & source_words) / len(summary_words)


def score(summaries: pd.DataFrame) -> pd.DataFrame:
    """Add the semantic and lexical scores to every summary.

    Args:
        summaries: The summary table, with ``summary`` and ``text`` columns.

    Returns:
        The table with ``similarity`` and ``lexical_overlap`` added.
    """
    scored = summaries.copy()

    logger.info("embedding %d summaries and their sources", len(scored))
    summary_vectors = encode_documents(scored["summary"].astype(str).tolist())
    source_vectors = encode_documents(scored["text"].astype(str).tolist())

    scored["similarity"] = cosine_pairs(summary_vectors, source_vectors).round(4)
    scored["lexical_overlap"] = [
        round(lexical_overlap(str(s), str(t)), 4)
        for s, t in zip(scored["summary"], scored["text"], strict=True)
    ]
    return scored


def distribution(scored: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """Summarise the similarity distribution per method and category.

    Args:
        scored: Output of ``score``.
        threshold: The level above which a summary is treated as keeping the
            main information.

    Returns:
        Count, median, spread and the share at or above the threshold.
    """
    grouped = scored.groupby(["method", "category"], observed=True)["similarity"]
    summary = grouped.agg(
        n="size", median="median", p10=lambda s: s.quantile(0.1), max="max"
    ).round(3)
    summary[f"pct_above_{threshold}"] = (
        grouped.apply(lambda s: (s >= threshold).mean()) * 100
    ).round(1)
    return summary.reset_index()


def extremes(scored: pd.DataFrame, n: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Take the best and worst scoring summaries, with the features that explain them.

    Args:
        scored: Output of ``score``.
        n: How many of each to return per method.

    Returns:
        The highest and lowest scoring summaries, with ``compression`` and
        ``lexical_overlap`` alongside the score.
    """
    columns = [
        "method",
        "lang",
        "category",
        "title",
        "similarity",
        "lexical_overlap",
        "compression",
    ]
    ordered = scored.sort_values("similarity")
    lowest = ordered.groupby("method", observed=True).head(n)[columns]
    highest = ordered.groupby("method", observed=True).tail(n)[columns]
    return highest.reset_index(drop=True), lowest.reset_index(drop=True)


_DRIVER_FEATURES = ("compression", "lexical_overlap", "source_chars", "summary_chars")


def drivers(scored: pd.DataFrame) -> pd.DataFrame:
    """Correlate the similarity score with the features that might explain it.

    Computed within each method rather than over the pooled table. The two
    methods occupy different regions of the feature space -- extractive overlap
    is 1.0 by construction -- so a pooled coefficient measures the gap between
    the methods rather than the relationship inside either one. For lexical
    overlap the pooled sign is the opposite of the within-method sign.

    Args:
        scored: Output of ``score``.

    Returns:
        Spearman correlation of ``similarity`` with each feature, methods as
        rows. ``NaN`` where a feature takes one value throughout a method and
        the coefficient is undefined.
    """
    features = [f for f in _DRIVER_FEATURES if f in scored.columns]
    rows = {
        str(method): {
            feature: _spearman(group["similarity"], group[feature]) for feature in features
        }
        for method, group in scored.groupby("method", observed=True)
    }
    return pd.DataFrame(rows).T.round(3)


def _spearman(scores: pd.Series, feature: pd.Series) -> float:
    """Spearman correlation, returning ``NaN`` for a constant feature.

    Args:
        scores: The similarity scores.
        feature: The candidate explanatory feature.

    Returns:
        The coefficient, or ``NaN`` when the feature does not vary and the
        rank correlation is undefined.
    """
    if feature.nunique(dropna=True) < 2 or scores.nunique(dropna=True) < 2:
        return float("nan")
    return float(scores.corr(feature, method="spearman"))


def threshold_check(scored: pd.DataFrame, threshold: float = 0.8) -> pd.Series:
    """Report the share of summaries clearing the similarity threshold.

    Args:
        scored: Output of ``score``.
        threshold: The level to test against.

    Returns:
        Percentage at or above the threshold, per method, in a series named
        for the threshold.
    """
    passing = scored.groupby("method", observed=True)["similarity"].apply(
        lambda values: float(np.mean(values >= threshold) * 100)
    )
    # Named for what the value is. Left as "similarity", the groupby carries the
    # scored column's name into the report table's header.
    return passing.round(1).rename(f"pct_at_or_above_{threshold}")
