"""Stage 9: predict the news category for held-out articles.

Run with: make topics
"""

from __future__ import annotations

import json

import pandas as pd

from src import config
from src.analysis.topics import TopicResult, evaluate_language, report_frame
from src.data.loader import load_articles
from src.log import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Train and evaluate the topic classifier for every language."""
    config.ensure_directories()

    # The full article table, not the capped corpus: TF-IDF is cheap to fit
    # and the classifier is better for having all the data.
    articles = load_articles()

    results: dict[str, TopicResult] = {}
    predictions = []
    metrics: dict[str, dict[str, float]] = {}

    for lang in config.LANGUAGES:
        result = evaluate_language(articles, lang)
        results[lang] = result
        predictions.append(result.predictions)
        metrics[lang] = {
            "n_train": result.n_train,
            "n_test": result.n_test,
            "accuracy": round(result.accuracy, 3),
            "macro_f1": round(result.macro_f1, 3),
            "baseline_macro_f1": round(result.baseline_macro_f1, 3),
        }
        logger.info(
            "[%s] %d train / %d test -- accuracy %.3f, macro F1 %.3f (baseline %.3f)",
            lang,
            result.n_train,
            result.n_test,
            result.accuracy,
            result.macro_f1,
            result.baseline_macro_f1,
        )

    # Keyed by language, not by position: the per-category detail below is
    # labelled English and has to be English whatever order LANGUAGES is in.
    english = results["en"]
    logger.info("[en] per-category performance:\n%s", report_frame(english).to_string())
    logger.info(
        "[en] confusion matrix (rows true, columns predicted):\n%s", english.confusion.to_string()
    )

    # Drop the article text before writing: the predictions table is for
    # inspecting labels, and the text is already in the corpus file.
    combined = pd.concat(predictions, ignore_index=True).drop(columns=["text"])
    combined.to_parquet(config.TOPICS_FILE, index=False)
    config.TOPIC_METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("wrote %s and %s", config.TOPICS_FILE.name, config.TOPIC_METRICS_FILE.name)


if __name__ == "__main__":
    main()
