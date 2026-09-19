"""Stage 6: summarize selected articles (requirement 4).

Selects ``config.ARTICLES_PER_CATEGORY`` articles for each category in each
language and summarizes every one twice:

* **extractive** (TextRank), which reuses the article's own sentences;
* **abstractive** (an LLM through OpenRouter), which rewrites them.

Both are kept because stage 8 compares them. Sentences come from the documents
cached by the preprocessing stage, so no article is parsed twice.

Run with: make summarize [SUMMARIZER=extractive|abstractive|both]
"""

from __future__ import annotations

import argparse

import pandas as pd

from src import config
from src.log import get_logger
from src.nlp.pipelines import load_cached_docs
from src.summarize.abstractive import SummaryRequest, summarize_many
from src.summarize.extractive import summarize as summarize_extractive
from src.utils import as_int

logger = get_logger(__name__)


def sentences_by_article(lang: str) -> dict[int, list[str]]:
    """Read the sentence split of every article in one language.

    Args:
        lang: Two-letter language code.

    Returns:
        Sentences keyed by ``pageid``, in document order, with blank ones
        dropped.
    """
    docs, index = load_cached_docs(lang)
    return {
        as_int(pageid): [sentence.text.strip() for sentence in doc.sents if sentence.text.strip()]
        for doc, pageid in zip(docs, index["pageid"], strict=True)
    }


def select_articles(
    corpus: pd.DataFrame, sentences: dict[int, list[str]], lang: str
) -> pd.DataFrame:
    """Choose the articles to summarize for one language.

    Only articles with enough sentences are eligible: summarizing a
    three-sentence article to three sentences is not summarization.

    Args:
        corpus: The working corpus.
        sentences: Sentence split keyed by ``pageid``.
        lang: Two-letter language code.

    Returns:
        ``config.ARTICLES_PER_CATEGORY`` articles per category, sampled with
        the project seed so re-runs pick the same articles.
    """
    eligible = corpus[corpus["lang"] == lang].copy()
    eligible["n_sentences"] = eligible["pageid"].map(
        lambda pid: len(sentences.get(as_int(pid), []))
    )
    eligible = eligible[eligible["n_sentences"] >= config.MIN_SENTENCES_TO_SUMMARIZE]

    parts = [
        group.sample(
            n=min(config.ARTICLES_PER_CATEGORY, len(group)), random_state=config.RANDOM_SEED
        )
        for _, group in eligible.groupby("primary_category", observed=True)
    ]
    selected = pd.concat(parts).reset_index(drop=True)
    logger.info(
        "[%s] selected %d articles across %d categories",
        lang,
        len(selected),
        selected["primary_category"].nunique(),
    )
    return selected


def build_extractive(
    selected: pd.DataFrame, sentences: dict[int, list[str]]
) -> list[dict[str, object]]:
    """Produce a TextRank summary for every selected article.

    Args:
        selected: Articles chosen for summarization.
        sentences: Sentence split keyed by ``pageid``.

    Returns:
        One summary row per article.
    """
    rows: list[dict[str, object]] = []
    for article in selected.itertuples():
        article_sentences = sentences[as_int(article.pageid)]
        rows.append(
            {
                "pageid": as_int(article.pageid),
                "lang": article.lang,
                "category": article.primary_category,
                "method": "extractive",
                "summary": summarize_extractive(article_sentences, config.SUMMARY_SENTENCES),
            }
        )
    return rows


def build_abstractive(selected: pd.DataFrame) -> list[dict[str, object]]:
    """Produce an LLM summary for every selected article.

    Args:
        selected: Articles chosen for summarization.

    Returns:
        One summary row per article. Articles whose call failed are dropped
        rather than stored as empty summaries.
    """
    requests = [
        SummaryRequest(pageid=as_int(row.pageid), lang=str(row.lang), text=str(row.text))
        for row in selected.itertuples()
    ]
    summaries = summarize_many(requests, config.SUMMARY_SENTENCES)

    return [
        {
            "pageid": as_int(article.pageid),
            "lang": article.lang,
            "category": article.primary_category,
            "method": "abstractive",
            "summary": summaries[as_int(article.pageid)],
        }
        for article in selected.itertuples()
        if summaries.get(as_int(article.pageid))
    ]


def main(summarizer: str = "both") -> None:
    """Summarize the selected articles and write the summary table.

    Args:
        summarizer: ``extractive``, ``abstractive`` or ``both``.
    """
    config.ensure_directories()
    corpus = pd.read_parquet(config.CORPUS_FILE)

    rows: list[dict[str, object]] = []
    chosen: list[pd.DataFrame] = []

    for lang in config.LANGUAGES:
        sentences = sentences_by_article(lang)
        selected = select_articles(corpus, sentences, lang)
        chosen.append(selected)

        if summarizer in ("extractive", "both"):
            rows.extend(build_extractive(selected, sentences))
        if summarizer in ("abstractive", "both"):
            rows.extend(build_abstractive(selected))

    summaries = pd.DataFrame(rows)

    # Carry the source text through so the similarity stage need not re-join
    # it, and record the compression achieved.
    sources = pd.concat(chosen)[["pageid", "lang", "title", "text"]]
    summaries = summaries.merge(sources, on=["pageid", "lang"], how="left")
    summaries["source_chars"] = summaries["text"].str.len()
    summaries["summary_chars"] = summaries["summary"].str.len()
    summaries["compression"] = (summaries["summary_chars"] / summaries["source_chars"]).round(3)

    logger.info("produced %d summaries", len(summaries))
    logger.info(
        "median compression by method and language:\n%s",
        summaries.pivot_table(
            index="method", columns="lang", values="compression", aggfunc="median"
        )
        .round(3)
        .to_string(),
    )

    summaries.to_parquet(config.SUMMARIES_FILE, index=False)
    logger.info("wrote %s", config.SUMMARIES_FILE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summarizer",
        choices=("extractive", "abstractive", "both"),
        default="both",
        help="Which summarizer to run. 'abstractive' and 'both' call the API.",
    )
    main(parser.parse_args().summarizer)
