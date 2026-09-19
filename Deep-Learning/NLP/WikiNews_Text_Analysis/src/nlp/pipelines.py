"""Load the spaCy pipelines and run them over article text.

The tagging pass is the slowest part of the project, so it happens once:
``annotate_language`` writes the parsed documents to a ``DocBin`` that the NER
and summarization stages read back.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
import spacy
from spacy.language import Language
from spacy.tokens import Doc, DocBin

from src import config
from src.log import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=len(config.SPACY_MODELS))
def load_pipeline(lang: str) -> Language:
    """Load the spaCy pipeline for a language, caching it for the process.

    Args:
        lang: Two-letter language code present in ``config.SPACY_MODELS``.

    Returns:
        The loaded pipeline, with tagger, parser, lemmatizer and NER enabled.

    Raises:
        KeyError: If the language is not one the project analyses.
        OSError: If the model is not installed. Run ``make sync``.
    """
    model = config.SPACY_MODELS[lang]
    logger.info("loading spaCy pipeline %s", model)
    return spacy.load(model)


def truncate(text: str, limit: int = config.MAX_ARTICLE_CHARS) -> str:
    """Cut an article down to a length the parser handles quickly.

    The cut is made at a paragraph break where possible, so the last sentence
    is not left broken.

    Args:
        text: The article text.
        limit: Maximum characters to keep.

    Returns:
        The text, shortened if it exceeded the limit.
    """
    if len(text) <= limit:
        return text
    head = text[:limit]
    split_at = head.rfind("\n\n")
    return head[:split_at] if split_at > limit // 2 else head


def docbin_path(lang: str) -> Path:
    """Return the cache path for a language's parsed documents.

    Args:
        lang: Two-letter language code.

    Returns:
        Path under ``data/interim``.
    """
    return config.INTERIM_DIR / f"docs_{lang}.spacy"


def index_path(lang: str) -> Path:
    """Return the path of the row index that accompanies a ``DocBin``.

    A ``DocBin`` stores documents in order but carries no article identifiers,
    so the ``pageid`` order is saved beside it.

    Args:
        lang: Two-letter language code.

    Returns:
        Path under ``data/interim``.
    """
    return config.INTERIM_DIR / f"docs_{lang}_index.parquet"


def annotate_language(corpus: pd.DataFrame, lang: str) -> list[Doc]:
    """Parse every article of one language and cache the result.

    Args:
        corpus: Working corpus; only rows with this ``lang`` are used.
        lang: Two-letter language code.

    Returns:
        The parsed documents, in the same order as the language's rows.
    """
    rows = corpus[corpus["lang"] == lang].reset_index(drop=True)
    nlp = load_pipeline(lang)
    texts = [truncate(t) for t in rows["text"]]

    logger.info("[%s] parsing %d articles", lang, len(texts))
    docs = list(nlp.pipe(texts, batch_size=config.SPACY_BATCH_SIZE))

    DocBin(docs=docs, store_user_data=False).to_disk(docbin_path(lang))
    rows[["pageid", "lang", "title", "primary_category", "date", "year"]].to_parquet(
        index_path(lang), index=False
    )
    logger.info("[%s] cached %d documents", lang, len(docs))
    return docs


def load_cached_docs(lang: str) -> tuple[list[Doc], pd.DataFrame]:
    """Read back the documents cached by ``annotate_language``.

    Args:
        lang: Two-letter language code.

    Returns:
        The documents and the matching article metadata, in the same order.

    Raises:
        FileNotFoundError: If the cache is missing. Run ``make preprocess``.
    """
    binary, index = docbin_path(lang), index_path(lang)
    if not binary.exists() or not index.exists():
        raise FileNotFoundError(f"No cached documents for '{lang}'. Run `make preprocess` first.")

    nlp = load_pipeline(lang)
    docs = list(DocBin().from_disk(binary).get_docs(nlp.vocab))
    return docs, pd.read_parquet(index)
