import pandas as pd

from src import config
from src.pipeline.corpus import (
    assign_primary_category,
    cap_per_language,
    categories_in_scope,
    select_corpus,
)


def test_categories_in_scope_uses_the_configured_order_not_the_article_order():
    topics = ["Science and technology", "Politics and conflicts"]
    assert categories_in_scope(topics) == ["Politics and conflicts", "Science and technology"]


def test_categories_in_scope_ignores_categories_we_do_not_analyse():
    assert categories_in_scope(["Sports", "Weather"]) == []


def test_assign_primary_category_is_deterministic_for_multi_topic_articles():
    topics = ["Crime and law", "Politics and conflicts"]
    assert assign_primary_category(topics) == assign_primary_category(list(reversed(topics)))


def test_assign_primary_category_returns_empty_when_out_of_scope():
    assert assign_primary_category(["Sports"]) == ""


def test_select_corpus_keeps_only_analysed_languages_and_categories():
    articles = pd.DataFrame(
        [
            {"lang": "en", "topics": ["Politics and conflicts"], "text": "a"},
            {"lang": "ta", "topics": ["Politics and conflicts"], "text": "b"},
            {"lang": "en", "topics": ["Sports"], "text": "c"},
        ]
    )
    selected = select_corpus(articles)
    assert len(selected) == 1
    assert selected.iloc[0]["primary_category"] == "Politics and conflicts"
    assert bool(selected.iloc[0]["is_single_topic"])


def test_cap_per_language_limits_each_language_independently():
    articles = pd.DataFrame(
        {
            "lang": ["en"] * 10 + ["de"] * 3,
            "date": pd.to_datetime(["2010-01-01"] * 13),
        }
    )
    capped = cap_per_language(articles, limit=5, seed=config.RANDOM_SEED)
    counts = capped["lang"].value_counts()
    assert counts["en"] == 5
    assert counts["de"] == 3, "a language with fewer articles than the cap keeps all of them"


def test_cap_per_language_is_reproducible():
    articles = pd.DataFrame(
        {"lang": ["en"] * 20, "date": pd.to_datetime(["2010-01-01"] * 20), "id": range(20)}
    )
    first = cap_per_language(articles, limit=5, seed=7)
    second = cap_per_language(articles, limit=5, seed=7)
    assert list(first["id"]) == list(second["id"])
