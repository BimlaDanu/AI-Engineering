import numpy as np

from src.analysis.embeddings import _split_into_chunks, cosine_pairs, encode


def test_a_short_document_becomes_one_chunk():
    assert _split_into_chunks("One sentence only.", 900) == ["One sentence only."]


def test_a_long_document_is_split_into_several_chunks():
    text = " ".join(["This is a sentence about the budget."] * 200)
    chunks = _split_into_chunks(text, 500)
    assert len(chunks) > 1
    assert all(len(chunk) < 700 for chunk in chunks), "chunks should stay near the target size"


def test_splitting_loses_no_words():
    text = "First sentence here. Second sentence here. Third sentence here."
    assert " ".join(_split_into_chunks(text, 25)).split() == text.split()


def test_an_empty_document_still_yields_one_chunk():
    # Every document must own at least one vector, or the owner mapping in
    # encode_documents would silently shift every later document's embedding.
    assert len(_split_into_chunks("", 900)) == 1


def test_encode_returns_an_empty_array_for_no_input():
    assert encode([]).shape == (0, 0)


def test_cosine_pairs_clips_float_error_above_one():
    vector = np.array([[0.6, 0.8]], dtype=np.float32)
    assert cosine_pairs(vector, vector)[0] <= 1.0
