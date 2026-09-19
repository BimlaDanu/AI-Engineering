import pytest

from src.summarize.extractive import rank_sentences, summarize

SENTENCES = [
    "The council approved the budget on Monday.",
    "The budget covers transport and housing.",
    "Rain is expected on Thursday.",
    "Councillors said the transport budget would rise by ten percent.",
    "Housing spending stays flat, the council confirmed.",
]


def test_summarize_returns_the_requested_number_of_sentences():
    result = summarize(SENTENCES, n_sentences=2)
    assert sum(1 for s in SENTENCES if s in result) == 2


def test_summarize_keeps_document_order():
    result = summarize(SENTENCES, n_sentences=3)
    positions = [result.index(s) for s in SENTENCES if s in result]
    assert positions == sorted(positions), "a summary must not reorder the article"


def test_summarize_prefers_sentences_on_the_articles_main_topic():
    result = summarize(SENTENCES, n_sentences=2)
    assert "Rain is expected on Thursday." not in result


def test_summarize_returns_everything_when_the_article_is_already_short():
    short = SENTENCES[:2]
    assert summarize(short, n_sentences=3) == " ".join(short)


def test_summarize_handles_a_single_sentence():
    assert summarize(["Only one."], n_sentences=3) == "Only one."


def test_rank_sentences_scores_every_sentence():
    assert len(rank_sentences(SENTENCES)) == len(SENTENCES)


def test_rank_sentences_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        rank_sentences([])


def test_summarize_is_deterministic():
    assert summarize(SENTENCES, 2) == summarize(SENTENCES, 2)


def test_summarize_survives_sentences_with_no_countable_words():
    # TF-IDF builds an empty vocabulary here and raises; the summarizer must
    # fall back to document order rather than take the pipeline down.
    degenerate = ["...", "!!!", "???", "----"]
    result = summarize(degenerate, n_sentences=3)
    assert result == "... !!! ???"


def test_rank_sentences_survives_an_empty_vocabulary():
    scores = rank_sentences(["...", "!!!", "???"])
    assert len(scores) == 3
