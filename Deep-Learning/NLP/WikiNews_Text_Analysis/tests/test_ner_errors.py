import pandas as pd

from src.analysis.ner_errors import (
    _mentions_name,
    flag_surface_errors,
    label_disagreements,
    missed_entities,
    recall_by_language,
)


def test_mentions_name_matches_the_full_name():
    assert _mentions_name("Angela Merkel besuchte Berlin.", "Angela Merkel")


def test_mentions_name_matches_the_surname_alone():
    assert _mentions_name("Merkel besuchte Berlin.", "Angela Merkel")


def test_mentions_name_does_not_match_inside_a_longer_word():
    assert not _mentions_name("The crossing was closed.", "Jim Ross")


def test_mentions_name_ignores_surnames_too_short_to_be_distinctive():
    assert not _mentions_name("Wir gehen.", "Wei Ng")


def test_missed_entities_only_checks_names_present_in_the_target_text(entities):
    corpus = pd.DataFrame([{"pageid": 1, "lang": "de", "text": "Barack Obama besuchte Berlin."}])
    result = missed_entities(entities, corpus, "de")
    assert set(result["entity"]) == {"Barack Obama"}, "a name absent from the text is not a miss"


def test_missed_entities_marks_a_name_the_target_pipeline_failed_to_tag(entities):
    corpus = pd.DataFrame([{"pageid": 1, "lang": "de", "text": "Barack Obama besuchte Berlin."}])
    result = missed_entities(entities, corpus, "de")
    assert not bool(result.iloc[0]["found"]), "German tagged no Obama entity, so this is a miss"


def test_recall_by_language_computes_the_proxy():
    missed = pd.DataFrame({"lang": ["de", "de", "de", "fr"], "found": [True, True, False, True]})
    summary = recall_by_language(missed).set_index("lang")
    assert summary.loc["de", "recall_proxy"] == round(2 / 3, 3)
    assert summary.loc["fr", "recall_proxy"] == 1.0


def test_label_disagreements_finds_names_labelled_differently_across_languages():
    entities = pd.DataFrame(
        {
            "entity_key": ["twitter", "twitter", "berlin", "berlin"],
            "lang": ["en", "de", "en", "de"],
            "label": ["PERSON", "OTHER", "LOCATION", "LOCATION"],
            "pageid": [1, 1, 2, 2],
        }
    )
    result = label_disagreements(entities)
    assert set(result["entity_key"]) == {"twitter"}, "only the disagreeing name is reported"
    assert result.iloc[0]["en"] == "PERSON"
    assert result.iloc[0]["de"] == "OTHER"


def test_flag_surface_errors_separates_convention_from_error():
    frame = pd.DataFrame(
        {
            "entity": ["the United States", "spokesman", "Berlin, Germany", "NASA"],
            "lang": ["en", "en", "en", "en"],
        }
    )
    flagged = flag_surface_errors(frame)
    assert list(flagged["error_type"]) == [
        "determiner_included",
        "lowercase_span",
        "stray_punctuation",
        "ok",
    ]


def test_flag_surface_errors_uses_each_language_own_determiners():
    # "Los" is a Spanish article and part of a German-article-free name; "UN"
    # is the French indefinite article and the United Nations in English.
    frame = pd.DataFrame({"entity": ["Los Angeles", "UN"], "lang": ["de", "en"]})
    assert list(flag_surface_errors(frame)["error_type"]) == ["ok", "ok"]


def test_flag_surface_errors_keeps_names_that_merely_start_lowercase():
    # A span is only a lowercase span when it carries no capital at all.
    frame = pd.DataFrame({"entity": ["al-Qaeda", "eBay", "deutsche"], "lang": ["en", "en", "de"]})
    assert list(flag_surface_errors(frame)["error_type"]) == ["ok", "ok", "lowercase_span"]


def test_flag_surface_errors_never_flags_a_single_word_as_determiner_included():
    frame = pd.DataFrame({"entity": ["Die", "Der"], "lang": ["de", "de"]})
    assert set(flag_surface_errors(frame)["error_type"]) == {"ok"}
