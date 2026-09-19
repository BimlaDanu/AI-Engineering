import pandas as pd
import pytest

from src.analysis.topics import build_classifier, evaluate_language, single_topic_articles


def _articles(n: int = 60) -> pd.DataFrame:
    rows = []
    for index in range(n):
        politics = index % 2 == 0
        rows.append(
            {
                "lang": "en",
                "topics": ["Politics and conflicts"] if politics else ["Science and technology"],
                "text": (
                    "parliament election minister vote government debate"
                    if politics
                    else "spacecraft telescope researchers laboratory experiment orbit"
                ),
            }
        )
    rows.append(
        {"lang": "en", "topics": ["Politics and conflicts", "Crime and law"], "text": "both"}
    )
    rows.append({"lang": "en", "topics": ["Sports"], "text": "match"})
    return pd.DataFrame(rows)


def test_single_topic_articles_excludes_multi_label_articles():
    result = single_topic_articles(_articles(), "en")
    assert "both" not in set(result["text"])


def test_single_topic_articles_excludes_categories_we_do_not_analyse():
    result = single_topic_articles(_articles(), "en")
    assert "Sports" not in set(result["category"])


def test_single_topic_articles_puts_the_label_in_a_category_column():
    result = single_topic_articles(_articles(), "en")
    assert set(result["category"]) == {"Politics and conflicts", "Science and technology"}


def test_build_classifier_returns_a_fittable_pipeline():
    # min_df=3 means the vectorizer needs a few documents per term before it
    # keeps one, so the fixture has to be larger than a toy pair.
    model = build_classifier()
    texts = ["parliament vote minister"] * 4 + ["telescope orbit spacecraft"] * 4
    labels = ["Politics and conflicts"] * 4 + ["Science and technology"] * 4
    model.fit(texts, labels)
    assert model.predict(["parliament vote minister"])[0] == "Politics and conflicts"


def test_evaluate_language_separates_two_clearly_different_categories():
    result = evaluate_language(_articles(200), "en")
    assert result.macro_f1 > 0.9
    assert result.macro_f1 > result.baseline_macro_f1


def test_evaluate_language_refuses_a_sample_too_small_to_split():
    with pytest.raises(ValueError, match="too few"):
        evaluate_language(_articles(10), "en")
