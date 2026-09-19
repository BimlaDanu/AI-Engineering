"""Stage 2: text pre-processing (requirement 1).

Breaks each article into sentences and tokens and records the grammatical role
of every token: part of speech, fine-grained tag, dependency relation and
lemma. This stage runs the spaCy models, so it caches the parsed documents for
the NER and summarization stages.

Run with: make preprocess
"""

from __future__ import annotations

import pandas as pd
from spacy.tokens import Doc

from src import config
from src.log import get_logger
from src.nlp.pipelines import annotate_language
from src.pipeline.corpus import main as build_corpus

logger = get_logger(__name__)

# A few dozen distinct values over ~1M rows, so stored as categoricals.
_CATEGORICAL_COLUMNS = ("lang", "pos", "tag", "dep")


def tokens_from_doc(doc: Doc, pageid: int, lang: str) -> list[dict[str, object]]:
    """Turn one parsed document into token rows.

    Whitespace-only tokens are dropped.

    Args:
        doc: The parsed document.
        pageid: Identifier of the article the document came from.
        lang: Two-letter language code.

    Returns:
        One dictionary per token, carrying its sentence index and its
        grammatical annotation.
    """
    rows: list[dict[str, object]] = []
    for sent_index, sentence in enumerate(doc.sents):
        for token in sentence:
            if token.is_space:
                continue
            rows.append(
                {
                    "pageid": pageid,
                    "lang": lang,
                    "sent_id": sent_index,
                    "token": token.text,
                    "lemma": token.lemma_,
                    "pos": token.pos_,
                    "tag": token.tag_,
                    "dep": token.dep_,
                    "is_stop": token.is_stop,
                    "is_punct": token.is_punct,
                    "is_alpha": token.is_alpha,
                }
            )
    return rows


def build_token_table(corpus: pd.DataFrame, lang: str) -> pd.DataFrame:
    """Parse one language's articles and return their token table.

    Args:
        corpus: The working corpus.
        lang: Two-letter language code.

    Returns:
        One row per token for every article in that language.
    """
    rows = corpus[corpus["lang"] == lang].reset_index(drop=True)
    docs = annotate_language(corpus, lang)

    tokens: list[dict[str, object]] = []
    for doc, pageid in zip(docs, rows["pageid"], strict=True):
        tokens.extend(tokens_from_doc(doc, int(pageid), lang))

    frame = pd.DataFrame(tokens)
    logger.info("[%s] %d tokens in %d articles", lang, len(frame), len(rows))
    return frame


def pos_profile(tokens: pd.DataFrame) -> pd.DataFrame:
    """Share of each part of speech per language.

    A sanity check on the tagging: the four pipelines should roughly agree on
    how much of a news article is nouns, verbs and proper nouns.

    Args:
        tokens: The token table.

    Returns:
        Percentages, parts of speech as rows and languages as columns.
    """
    counts = pd.crosstab(tokens["pos"], tokens["lang"], normalize="columns")
    return (counts * 100).round(1)


def main() -> None:
    """Pre-process the working corpus and write the token table."""
    config.ensure_directories()

    if not config.CORPUS_FILE.exists():
        logger.info("no corpus file yet, building it first")
        build_corpus()
    corpus = pd.read_parquet(config.CORPUS_FILE)

    frames = [build_token_table(corpus, lang) for lang in config.LANGUAGES]
    tokens = pd.concat(frames, ignore_index=True)
    for column in _CATEGORICAL_COLUMNS:
        tokens[column] = tokens[column].astype("category")

    sentences = tokens.groupby(["lang", "pageid"], observed=True)["sent_id"].nunique()
    logger.info("total tokens: %d", len(tokens))
    logger.info("sentences per article: median %.0f", sentences.median())
    logger.info("part-of-speech share (%%) by language:\n%s", pos_profile(tokens).to_string())

    tokens.to_parquet(config.TOKENS_FILE, index=False)
    logger.info("wrote %s", config.TOKENS_FILE)


if __name__ == "__main__":
    main()
