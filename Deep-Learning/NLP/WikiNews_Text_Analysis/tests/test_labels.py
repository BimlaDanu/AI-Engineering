import pytest

from src.nlp.labels import (
    LOCATION,
    ORGANISATION,
    OTHER,
    PERSON,
    clean_surface,
    entity_key,
    is_usable,
    to_coarse,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("PERSON", PERSON),
        ("PER", PERSON),
        ("ORG", ORGANISATION),
        ("GPE", LOCATION),
        ("LOC", LOCATION),
        ("MISC", OTHER),
        ("NORP", OTHER),
    ],
)
def test_to_coarse_maps_both_tag_sets_onto_one_scheme(raw, expected):
    assert to_coarse(raw) == expected


@pytest.mark.parametrize("raw", ["DATE", "MONEY", "CARDINAL", "PERCENT"])
def test_to_coarse_drops_numeric_labels_the_other_pipelines_cannot_produce(raw):
    assert to_coarse(raw) is None


def test_clean_surface_strips_quotes_and_collapses_whitespace():
    assert clean_surface('  "United   Nations"  ') == "United Nations"
    assert clean_surface("Angela\nMerkel") == "Angela Merkel"
    assert clean_surface("(Reuters),") == "Reuters"


def test_entity_key_folds_case_article_and_possessive():
    assert entity_key("The United States", "en") == "united states"
    assert entity_key("Barack Obama's", "en") == "barack obama"


def test_entity_key_maps_abbreviations_onto_the_expanded_name():
    assert entity_key("U.S.", "en") == entity_key("US", "en") == entity_key("the US", "en")


def test_entity_key_keeps_a_one_word_name_that_looks_like_an_article():
    # "Die" is a German article but also a standalone token; a single word is
    # never stripped, only a leading article in a longer name.
    assert entity_key("Die", "de") == "die"


@pytest.mark.parametrize(
    ("text", "lang", "expected"),
    [
        ("Los Angeles", "de", "los angeles"),
        ("Los Angeles", "en", "los angeles"),
        ("La Paz", "de", "la paz"),
        ("los Estados Unidos", "es", "estados unidos"),
        ("la France", "fr", "france"),
    ],
)
def test_entity_key_strips_only_the_articles_of_its_own_language(text, lang, expected):
    # "Los" and "La" are Spanish and French articles but part of the name in
    # German and English. Sharing one article list across languages turned
    # "Los Angeles" into "angeles".
    assert entity_key(text, lang) == expected


def test_entity_key_strips_the_english_article_in_every_language():
    # English names are quoted verbatim in the other three languages, so "The
    # Guardian" has to reach the same key wherever it appears.
    assert entity_key("The Guardian", "es") == entity_key("The Guardian", "en")


def test_is_usable_rejects_debris():
    assert not is_usable("", "")
    assert not is_usable("7", "7")
    assert not is_usable("-", "-")
    assert is_usable("NASA", "national aeronautics and space administration")
