"""Stage 1: build the working corpus.

Narrows the raw articles down to the four categories and four languages the
project analyses, then caps the size per language.

Sampling is random with a fixed seed rather than stratified, so the corpus's
own category and year mix survives.

Run with: make corpus
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.data.loader import load_articles
from src.log import get_logger

logger = get_logger(__name__)


def categories_in_scope(topics: list[str]) -> list[str]:
    """Return the analysed categories an article carries, in a fixed order.

    Args:
        topics: The article's topic labels.

    Returns:
        The subset that is in ``config.CATEGORIES``, ordered as that tuple is.
    """
    return [c for c in config.CATEGORIES if c in topics]


def assign_primary_category(topics: list[str]) -> str:
    """Pick one category to group an article under.

    Articles can carry several topics, so the first match in
    ``config.CATEGORIES`` order wins. Deterministic, and that order puts the
    broadest category first.

    Args:
        topics: The article's topic labels.

    Returns:
        The chosen category, or an empty string if none is in scope.
    """
    in_scope = categories_in_scope(topics)
    return in_scope[0] if in_scope else ""


def select_corpus(articles: pd.DataFrame) -> pd.DataFrame:
    """Filter the full corpus to the analysed languages and categories.

    Args:
        articles: Output of ``load_articles``.

    Returns:
        Articles in scope, with ``primary_category``, ``scope_categories`` and
        ``is_single_topic`` added. ``is_single_topic`` marks articles carrying
        exactly one topic label, which is what the topic classifier trains on.
    """
    selected = articles[articles["lang"].isin(config.LANGUAGES)].copy()
    selected["scope_categories"] = selected["topics"].apply(categories_in_scope)
    selected = selected[selected["scope_categories"].str.len() > 0].copy()

    selected["primary_category"] = selected["topics"].apply(assign_primary_category)
    selected["is_single_topic"] = selected["topics"].str.len() == 1
    return selected


def cap_per_language(corpus: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    """Keep at most ``limit`` articles per language, sampled at random.

    Args:
        corpus: Articles already filtered to the analysed scope.
        limit: Maximum articles to keep for each language.
        seed: Seed for the sample, so re-runs select the same articles.

    Returns:
        The capped corpus, sorted by language and date.
    """
    parts = [
        group.sample(n=min(limit, len(group)), random_state=seed)
        for _, group in corpus.groupby("lang", sort=True)
    ]
    return pd.concat(parts).sort_values(["lang", "date"]).reset_index(drop=True)


def summarise(corpus: pd.DataFrame) -> str:
    """Render a language-by-category count table for the run log.

    Args:
        corpus: The selected corpus.

    Returns:
        The crosstab as a printable string.
    """
    table = pd.crosstab(
        corpus["lang"], corpus["primary_category"], margins=True, margins_name="all"
    )
    return table.to_string()


def main() -> None:
    """Build the working corpus and write it to ``config.CORPUS_FILE``."""
    config.ensure_directories()

    articles = load_articles()
    selected = select_corpus(articles)
    logger.info("in scope before capping: %d articles", len(selected))

    corpus = cap_per_language(selected, config.MAX_ARTICLES_PER_LANGUAGE, config.RANDOM_SEED)
    logger.info(
        "after capping at %d per language: %d articles",
        config.MAX_ARTICLES_PER_LANGUAGE,
        len(corpus),
    )
    logger.info("articles per language and category:\n%s", summarise(corpus))
    logger.info("year range: %s-%s", corpus["year"].min(), corpus["year"].max())

    corpus.to_parquet(config.CORPUS_FILE, index=False)
    logger.info("wrote %s", config.CORPUS_FILE)


if __name__ == "__main__":
    main()
