from src import config
from src.nlp.pipelines import docbin_path, index_path, truncate


def test_truncate_leaves_a_short_article_untouched():
    text = "A short article."
    assert truncate(text, limit=100) == text


def test_truncate_cuts_a_long_article_to_the_limit():
    text = "word " * 1000
    assert len(truncate(text, limit=200)) <= 200


def test_truncate_prefers_a_paragraph_boundary_past_the_halfway_point():
    text = "a" * 70 + "\n\n" + "x" * 400
    cut = truncate(text, limit=100)
    assert cut == "a" * 70, "the cut should land on the paragraph break"


def test_truncate_ignores_a_paragraph_boundary_too_early_to_be_useful():
    # A break in the first half would throw away most of the article, so a
    # hard cut keeps more text than honouring it would.
    text = "short.\n\n" + "x" * 400
    assert len(truncate(text, limit=100)) == 100


def test_truncate_falls_back_to_a_hard_cut_when_no_break_is_near_the_end():
    text = "x" * 500
    assert len(truncate(text, limit=100)) == 100


def test_cache_paths_are_distinct_per_language():
    assert docbin_path("en") != docbin_path("de")
    assert index_path("en") != index_path("de")


def test_cache_paths_live_under_the_interim_directory():
    assert config.INTERIM_DIR in docbin_path("en").parents
