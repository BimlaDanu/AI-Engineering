from src import config
from src.summarize.abstractive import SummaryRequest, _cache_path, build_prompt


def test_build_prompt_names_the_articles_own_language():
    assert "Spanish" in build_prompt("texto", "es", 3)
    assert "German" in build_prompt("Text", "de", 3)


def test_build_prompt_states_the_requested_length():
    assert "at most 4 sentences" in build_prompt("text", "en", 4)


def test_build_prompt_also_sets_a_word_target():
    # A sentence count alone lets the model write three very long sentences
    # and compress nothing, which is what the first version of this prompt did.
    assert "80 words" in build_prompt("text", "en", 4)


def test_build_prompt_includes_the_article():
    assert "The council met." in build_prompt("The council met.", "en", 3)


def test_cache_key_differs_for_different_articles():
    first = SummaryRequest(pageid=1, lang="en", text="one")
    second = SummaryRequest(pageid=2, lang="en", text="two")
    assert _cache_path(first, 3) != _cache_path(second, 3)


def test_cache_key_differs_when_the_requested_length_changes():
    request = SummaryRequest(pageid=1, lang="en", text="one")
    assert _cache_path(request, 3) != _cache_path(request, 5)


def test_cache_key_is_stable_for_the_same_request():
    request = SummaryRequest(pageid=1, lang="en", text="one")
    assert _cache_path(request, 3) == _cache_path(request, 3)


def test_cache_lives_under_the_interim_directory():
    request = SummaryRequest(pageid=1, lang="en", text="one")
    assert config.INTERIM_DIR in _cache_path(request, 3).parents
