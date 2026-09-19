"""Stage 3: named entity recognition (requirement 2).

Pulls entities out of the documents cached by the preprocessing stage and
attaches each mention to its article's metadata, so an entity traces back to
the story, category, language and date it appeared in.

One row is one mention, not one entity. Aggregation happens in stage 4, which
needs mention-level detail to count articles as well as occurrences.

Run with: make ner
"""

from __future__ import annotations

import pandas as pd
from spacy.tokens import Doc

from src import config
from src.log import get_logger
from src.nlp.labels import clean_surface, entity_key, is_usable, to_coarse
from src.nlp.pipelines import load_cached_docs

logger = get_logger(__name__)


def entities_from_doc(doc: Doc, article: pd.Series) -> list[dict[str, object]]:
    """Extract the usable entity mentions from one document.

    Args:
        doc: The parsed document.
        article: The article's metadata row, supplying identifiers and dates.

    Returns:
        One dictionary per kept mention. Mentions whose label has no
        cross-language equivalent, or whose text is debris, are skipped.
    """
    rows: list[dict[str, object]] = []
    for entity in doc.ents:
        coarse = to_coarse(entity.label_)
        if coarse is None:
            continue

        surface = clean_surface(entity.text)
        key = entity_key(surface, str(article["lang"]))
        if not is_usable(surface, key):
            continue

        rows.append(
            {
                "pageid": int(article["pageid"]),
                "lang": str(article["lang"]),
                "title": str(article["title"]),
                "category": str(article["primary_category"]),
                "date": article["date"],
                "year": article["year"],
                "entity": surface,
                "entity_key": key,
                "label_raw": entity.label_,
                "label": coarse,
                "start_char": entity.start_char,
                "end_char": entity.end_char,
                "sentence": clean_surface(entity.sent.text)[:400],
            }
        )
    return rows


def extract_language(lang: str) -> pd.DataFrame:
    """Extract every entity mention for one language.

    Args:
        lang: Two-letter language code.

    Returns:
        The mention table for that language.
    """
    docs, index = load_cached_docs(lang)
    rows: list[dict[str, object]] = []
    for doc, (_, article) in zip(docs, index.iterrows(), strict=True):
        rows.extend(entities_from_doc(doc, article))

    frame = pd.DataFrame(rows)
    logger.info(
        "[%s] %d mentions, %d distinct entities, from %d articles",
        lang,
        len(frame),
        frame["entity_key"].nunique(),
        len(index),
    )
    return frame


def label_profile(entities: pd.DataFrame) -> pd.DataFrame:
    """Share of each coarse label per language.

    A language finding far fewer ``PERSON`` mentions than the others on
    comparable articles is the first sign of a recall problem.

    Args:
        entities: The mention table.

    Returns:
        Percentages, labels as rows and languages as columns.
    """
    return (pd.crosstab(entities["label"], entities["lang"], normalize="columns") * 100).round(1)


def main() -> None:
    """Extract entities for every language and write the mention table."""
    config.ensure_directories()

    frames = [extract_language(lang) for lang in config.LANGUAGES]
    entities = pd.concat(frames, ignore_index=True)
    for column in ("lang", "label", "label_raw", "category"):
        entities[column] = entities[column].astype("category")

    logger.info("total mentions: %d", len(entities))
    articles = len(entities.drop_duplicates(["lang", "pageid"]))
    logger.info("mentions per article: %.1f", len(entities) / articles)
    logger.info("coarse label share (%%) by language:\n%s", label_profile(entities).to_string())

    entities.to_parquet(config.ENTITIES_FILE, index=False)
    logger.info("wrote %s", config.ENTITIES_FILE)


if __name__ == "__main__":
    main()
