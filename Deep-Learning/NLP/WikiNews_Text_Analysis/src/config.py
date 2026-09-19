"""Paths, analysis scope and runtime limits.

Everything another module would otherwise hard-code lives here, so the scope of
the analysis can be read and changed in one place.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

# ─────────────────── paths ───────────────────

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_CORPUS: Final[Path] = DATA_DIR / "raw" / "wikinews" / "multilingual_wikinews.jsonl"
INTERIM_DIR: Final[Path] = DATA_DIR / "interim"
PROCESSED_DIR: Final[Path] = DATA_DIR / "processed"

REPORTS_DIR: Final[Path] = PROJECT_ROOT / "reports"
FIGURES_DIR: Final[Path] = REPORTS_DIR / "figures"

# Stage outputs. Each stage reads the previous file and writes its own, so a
# single stage can be re-run without repeating the slow spaCy pass.
CORPUS_FILE: Final[Path] = PROCESSED_DIR / "corpus.parquet"
TOKENS_FILE: Final[Path] = PROCESSED_DIR / "tokens.parquet"
ENTITIES_FILE: Final[Path] = PROCESSED_DIR / "entities.parquet"
SUMMARIES_FILE: Final[Path] = PROCESSED_DIR / "summaries.parquet"
SIMILARITY_FILE: Final[Path] = PROCESSED_DIR / "similarity.parquet"
ENTITY_AGGREGATE_FILE: Final[Path] = PROCESSED_DIR / "entity_aggregate.parquet"
ENTITY_TIMELINE_FILE: Final[Path] = PROCESSED_DIR / "entity_timeline.parquet"
ENTITY_CLUSTERS_FILE: Final[Path] = PROCESSED_DIR / "entity_clusters.parquet"
GEOGRAPHY_FILE: Final[Path] = PROCESSED_DIR / "geographic_focus.parquet"
NER_ERRORS_FILE: Final[Path] = PROCESSED_DIR / "ner_errors.parquet"
NER_RECALL_FILE: Final[Path] = PROCESSED_DIR / "ner_recall.parquet"
GRAMMAR_FILE: Final[Path] = PROCESSED_DIR / "grammar.parquet"
TOPICS_FILE: Final[Path] = PROCESSED_DIR / "topic_predictions.parquet"
TOPIC_METRICS_FILE: Final[Path] = PROCESSED_DIR / "topic_metrics.json"

# ─────────────────── analysis scope ───────────────────

# The 13 topic labels WikiNews uses. The `categories` field also holds place
# names and date strings, so this set is what separates a label from noise.
TOPIC_LABELS: Final[frozenset[str]] = frozenset(
    {
        "Crime and law",
        "Culture and entertainment",
        "Disasters and accidents",
        "Economy and business",
        "Education",
        "Environment",
        "Health",
        "Obituaries",
        "Politics and conflicts",
        "Science and technology",
        "Sports",
        "Wackynews",
        "Weather",
    }
)

# The categories analysed: the four largest in the corpus.
CATEGORIES: Final[tuple[str, ...]] = (
    "Politics and conflicts",
    "Crime and law",
    "Disasters and accidents",
    "Science and technology",
)

# The languages analysed: widest coverage inside those categories, and a
# comparable spaCy pipeline each.
LANGUAGES: Final[tuple[str, ...]] = ("en", "es", "fr", "de")

# One spaCy pipeline per language, all the same size so the cross-language
# comparison holds model capacity constant. Changing the size also means
# swapping the URLs in the `models` dependency group and re-locking.
SPACY_SIZE: Final[str] = os.environ.get("NLP_SPACY_SIZE", "sm")
SPACY_MODELS: Final[dict[str, str]] = {
    "en": f"en_core_web_{SPACY_SIZE}",
    "es": f"es_core_news_{SPACY_SIZE}",
    "fr": f"fr_core_news_{SPACY_SIZE}",
    "de": f"de_core_news_{SPACY_SIZE}",
}

# ─────────────────── runtime limits ───────────────────
#
# Every stage is CPU-only. These caps keep a full run in the order of minutes;
# raise them with the matching environment variable.

# Articles kept per language for tagging and NER. Sampled at random with a
# fixed seed, which preserves the natural category and year mix.
MAX_ARTICLES_PER_LANGUAGE: Final[int] = int(os.environ.get("NLP_MAX_ARTICLES", "800"))

# Characters of an article fed to spaCy. A few articles run past 50,000 and
# would dominate the runtime on their own.
MAX_ARTICLE_CHARS: Final[int] = 20_000

# spaCy batch size. Small, because large batches spike memory on two threads.
SPACY_BATCH_SIZE: Final[int] = 32

# Articles summarized per category, per language.
ARTICLES_PER_CATEGORY: Final[int] = int(os.environ.get("NLP_ARTICLES_PER_CATEGORY", "10"))

# Sentences an extractive summary keeps, and the floor for an article to be
# worth summarizing at all.
SUMMARY_SENTENCES: Final[int] = 3
MIN_SENTENCES_TO_SUMMARIZE: Final[int] = 5

# Multilingual sentence encoder for the similarity scores. ~470 MB, CPU.
EMBEDDING_MODEL: Final[str] = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_BATCH_SIZE: Final[int] = 16

# OpenRouter settings for the abstractive summaries.
LLM_MODEL: Final[str] = os.environ.get("NLP_LLM_MODEL", "openai/gpt-5-mini")
LLM_BASE_URL: Final[str] = "https://openrouter.ai/api/v1"
LLM_TEMPERATURE: Final[float] = 0.2
LLM_MAX_WORKERS: Final[int] = 4

# One seed for every sampling and model-fitting step in the project.
RANDOM_SEED: Final[int] = 42


def ensure_directories() -> None:
    """Create the output directories the pipeline writes into."""
    for directory in (INTERIM_DIR, PROCESSED_DIR, REPORTS_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
