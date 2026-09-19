"""Checks on the real stage outputs.

These are marked ``slow`` because they read the tables a full pipeline run
produces. They are skipped when those tables are absent, so a fresh clone and
CI stay green without the 57 MB corpus. What they verify is the contract
between stages: the columns each stage promises the next one, and the
invariants that would otherwise only break far downstream.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import config

pytestmark = pytest.mark.slow


def _load(path) -> pd.DataFrame:
    if not path.exists():
        pytest.skip(f"{path.name} not built; run `make all` to exercise this test")
    return pd.read_parquet(path)


def test_corpus_holds_only_the_analysed_languages_and_categories():
    corpus = _load(config.CORPUS_FILE)
    assert set(corpus["lang"]) <= set(config.LANGUAGES)
    assert set(corpus["primary_category"]) <= set(config.CATEGORIES)


def test_corpus_respects_the_per_language_cap():
    corpus = _load(config.CORPUS_FILE)
    assert corpus["lang"].value_counts().max() <= config.MAX_ARTICLES_PER_LANGUAGE


def test_corpus_articles_all_have_text():
    corpus = _load(config.CORPUS_FILE)
    assert corpus["text"].str.len().gt(0).all()


def test_corpus_dates_were_backfilled_onto_translations():
    corpus = _load(config.CORPUS_FILE)
    non_english = corpus[corpus["lang"] != "en"]
    assert non_english["date"].notna().mean() > 0.95


def test_tokens_carry_the_grammatical_annotation_requirement_one_asks_for():
    tokens = _load(config.TOKENS_FILE)
    assert {"token", "lemma", "pos", "tag", "dep", "sent_id"} <= set(tokens.columns)


def test_tokens_cover_every_analysed_language():
    tokens = _load(config.TOKENS_FILE)
    assert set(tokens["lang"].unique()) == set(config.LANGUAGES)


def test_entities_carry_the_article_metadata_requirement_two_asks_for():
    entities = _load(config.ENTITIES_FILE)
    required = {
        "pageid",
        "title",
        "lang",
        "category",
        "date",
        "year",
        "entity",
        "label",
        "sentence",
    }
    assert required <= set(entities.columns)


def test_entity_labels_are_the_shared_coarse_scheme():
    from src.nlp.labels import COARSE_LABELS

    entities = _load(config.ENTITIES_FILE)
    assert set(entities["label"].unique()) <= set(COARSE_LABELS)


def test_entity_offsets_are_well_formed():
    entities = _load(config.ENTITIES_FILE)
    assert (entities["end_char"] > entities["start_char"]).all()


def test_every_entity_belongs_to_an_article_in_the_corpus():
    corpus = _load(config.CORPUS_FILE)
    entities = _load(config.ENTITIES_FILE)
    assert set(entities["pageid"]) <= set(corpus["pageid"])


def test_summaries_cover_at_least_ten_articles_per_category():
    summaries = _load(config.SUMMARIES_FILE)
    per_group = summaries.groupby(["lang", "category", "method"], observed=True)["pageid"].nunique()
    assert per_group.min() >= 10, "10-20 articles per category are required"


def test_summaries_are_shorter_than_their_sources():
    summaries = _load(config.SUMMARIES_FILE)
    assert (summaries["compression"] < 1.0).all()


def test_no_summary_is_empty():
    summaries = _load(config.SUMMARIES_FILE)
    assert summaries["summary"].str.strip().str.len().gt(0).all()


def test_similarity_scores_are_in_range():
    scored = _load(config.SIMILARITY_FILE)
    assert scored["similarity"].between(-1.0, 1.0).all()
    assert scored["lexical_overlap"].between(0.0, 1.0).all()


def test_extractive_summaries_reuse_more_wording_than_abstractive_ones():
    # The premise of the whole section 6 argument. If this ever stopped
    # holding, the inflation caveat in the report would be wrong.
    scored = _load(config.SIMILARITY_FILE)
    by_method = scored.groupby("method", observed=True)["lexical_overlap"].median()
    if {"extractive", "abstractive"} <= set(by_method.index):
        assert by_method["extractive"] > by_method["abstractive"]
