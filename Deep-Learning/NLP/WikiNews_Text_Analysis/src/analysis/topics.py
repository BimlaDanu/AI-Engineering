"""Predict a news category from an article's text.

Trained only on articles carrying exactly one topic label: about a quarter of
the corpus carries two or more, and an article that is genuinely both political
and criminal cannot be scored fairly against a single-label prediction.

Scored with macro F1 against a most-frequent-class baseline, since one category
holds more than half the articles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src import config
from src.log import get_logger

logger = get_logger(__name__)

_TEST_SIZE = 0.25


@dataclass
class TopicResult:
    """Outcome of training and evaluating the classifier for one language.

    Attributes:
        lang: Two-letter language code.
        n_train: Articles used for training.
        n_test: Articles held out.
        macro_f1: Macro-averaged F1 on the held-out set.
        baseline_macro_f1: The same metric for a most-frequent-class baseline.
        accuracy: Plain accuracy, reported alongside macro F1.
        report: Per-class precision, recall and F1.
        confusion: Confusion matrix with labelled rows and columns.
        predictions: The held-out articles with their true and predicted label.
    """

    lang: str
    n_train: int
    n_test: int
    macro_f1: float
    baseline_macro_f1: float
    accuracy: float
    report: dict[str, Any] = field(repr=False)
    confusion: pd.DataFrame = field(repr=False)
    predictions: pd.DataFrame = field(repr=False)


def build_classifier() -> Pipeline:
    """Assemble the TF-IDF and logistic regression pipeline.

    Word unigrams and bigrams, since category cues in news are often two words
    ("space station", "supreme court"), with balanced class weights.

    Returns:
        An unfitted scikit-learn pipeline.
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=3,
                    max_features=50_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=config.RANDOM_SEED,
                ),
            ),
        ]
    )


def single_topic_articles(articles: pd.DataFrame, lang: str) -> pd.DataFrame:
    """Select the unambiguously labelled articles for one language.

    Args:
        articles: Any table with ``lang``, ``topics`` and ``text`` columns.
        lang: Two-letter language code.

    Returns:
        Articles carrying exactly one topic, and that topic one we analyse,
        with the label in a ``category`` column.
    """
    subset = articles[articles["lang"] == lang].copy()
    subset["n_topics"] = subset["topics"].str.len()
    subset = subset[subset["n_topics"] == 1].copy()

    subset["category"] = subset["topics"].str[0]
    return subset[subset["category"].isin(config.CATEGORIES)].reset_index(drop=True)


def evaluate_language(articles: pd.DataFrame, lang: str) -> TopicResult:
    """Train and score the classifier for one language.

    Args:
        articles: The full article table.
        lang: Two-letter language code.

    Returns:
        The evaluation result on a stratified held-out split.

    Raises:
        ValueError: If the language has too few single-topic articles to split.
    """
    data = single_topic_articles(articles, lang)
    if len(data) < 50:
        raise ValueError(
            f"Only {len(data)} single-topic articles for '{lang}'; too few to evaluate"
        )

    train_x, test_x, train_y, test_y = train_test_split(
        data["text"],
        data["category"],
        test_size=_TEST_SIZE,
        stratify=data["category"],
        random_state=config.RANDOM_SEED,
    )

    model = build_classifier().fit(train_x, train_y)
    predicted = model.predict(test_x)

    baseline = DummyClassifier(strategy="most_frequent").fit(train_x, train_y)
    baseline_predicted = baseline.predict(test_x)

    labels = sorted(data["category"].unique())
    return TopicResult(
        lang=lang,
        n_train=len(train_x),
        n_test=len(test_x),
        macro_f1=float(f1_score(test_y, predicted, average="macro")),
        baseline_macro_f1=float(
            f1_score(test_y, baseline_predicted, average="macro", zero_division=0)
        ),
        accuracy=float((predicted == test_y).mean()),
        report=classification_report(test_y, predicted, output_dict=True, zero_division=0),
        confusion=pd.DataFrame(
            confusion_matrix(test_y, predicted, labels=labels), index=labels, columns=labels
        ),
        predictions=pd.DataFrame(
            {"lang": lang, "text": test_x.values, "true": test_y.values, "predicted": predicted}
        ),
    )


def report_frame(result: TopicResult) -> pd.DataFrame:
    """Turn the per-class report into a readable table.

    Args:
        result: Output of ``evaluate_language``.

    Returns:
        Precision, recall, F1 and support per category.
    """
    rows = {
        label: metrics for label, metrics in result.report.items() if label in config.CATEGORIES
    }
    return pd.DataFrame(rows).T[["precision", "recall", "f1-score", "support"]].round(3)
