"""Stage 5: investigate wrongly predicted entities (requirement 3).

Runs the three checks in ``src.analysis.ner_errors``, measures how much weaker
NER is outside English, and writes the flagged mentions for inspection.

Run with: make ner-errors
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.analysis.ner_errors import (
    flag_surface_errors,
    label_disagreements,
    location_naming_note,
    missed_entities,
    recall_by_language,
)
from src.log import get_logger

logger = get_logger(__name__)

# Flagged mentions written out for a human to read through.
_AUDIT_SAMPLE = 200


def main() -> None:
    """Run the error checks and write the flagged mentions."""
    config.ensure_directories()

    if not config.ENTITIES_FILE.exists():
        raise FileNotFoundError(f"{config.ENTITIES_FILE} missing. Run `make ner` first.")
    entities = pd.read_parquet(config.ENTITIES_FILE)
    corpus = pd.read_parquet(config.CORPUS_FILE)

    # Check 1: people named in English that the other pipelines failed to tag.
    non_english = [lang for lang in config.LANGUAGES if lang != "en"]
    missed = pd.concat(
        [missed_entities(entities, corpus, lang) for lang in non_english], ignore_index=True
    )
    recall = recall_by_language(missed)
    logger.info(
        "missed-entity check -- person names present in the text but not tagged:\n%s",
        recall.to_string(index=False),
    )

    worst = (
        missed[~missed["found"]]
        .groupby(["lang", "entity"])
        .size()
        .reset_index(name="times_missed")
        .sort_values("times_missed", ascending=False)
        .head(12)
    )
    logger.info("most often missed names:\n%s", worst.to_string(index=False))

    # Check 2: the same name labelled differently by different pipelines.
    disagreements = label_disagreements(entities)
    logger.info(
        "label disagreements across languages: %d entities; the most reported:\n%s",
        len(disagreements),
        disagreements.head(10).to_string(index=False),
    )

    # Check 3: boundary and spurious spans visible from the span itself.
    flagged = flag_surface_errors(entities)
    surface_rates = (
        pd.crosstab(flagged["lang"], flagged["error_type"], normalize="index") * 100
    ).round(2)
    logger.info("surface issue rate (%%) by language:\n%s", surface_rates.to_string())

    # Context: locations are excluded from check 1 because they get translated.
    logger.info(
        "location counts (not errors, context):\n%s",
        location_naming_note(entities).to_string(index=False),
    )

    audit = (
        flagged[flagged["error_type"] != "ok"]
        .sample(
            n=min(_AUDIT_SAMPLE, int((flagged["error_type"] != "ok").sum())),
            random_state=config.RANDOM_SEED,
        )
        .loc[:, ["lang", "entity", "label", "label_raw", "error_type", "sentence"]]
    )
    recall.to_parquet(config.NER_RECALL_FILE, index=False)
    audit.to_parquet(config.NER_ERRORS_FILE, index=False)
    logger.info("wrote %d flagged mentions to %s", len(audit), config.NER_ERRORS_FILE)


if __name__ == "__main__":
    main()
