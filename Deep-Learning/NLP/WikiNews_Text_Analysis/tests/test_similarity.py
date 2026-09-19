import warnings

import numpy as np
import pandas as pd
import pytest

from src.analysis.embeddings import cosine_pairs
from src.analysis.similarity import distribution, drivers, extremes, lexical_overlap


def test_lexical_overlap_is_one_when_every_word_came_from_the_source():
    source = "The council approved the budget on Monday."
    assert lexical_overlap("The council approved the budget.", source) == 1.0


def test_lexical_overlap_is_zero_for_a_completely_different_text():
    assert lexical_overlap("penguins swim", "the council met") == 0.0


def test_lexical_overlap_is_between_zero_and_one_for_a_rewrite():
    score = lexical_overlap(
        "Councillors backed the budget", "The council approved the budget on Monday."
    )
    assert 0.0 < score < 1.0


def test_lexical_overlap_handles_an_empty_summary():
    assert lexical_overlap("", "some source text") == 0.0


def test_cosine_pairs_scores_identical_vectors_as_one():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    assert np.allclose(cosine_pairs(vectors, vectors), [1.0, 1.0])


def test_cosine_pairs_scores_orthogonal_vectors_as_zero():
    left = np.array([[1.0, 0.0]], dtype=np.float32)
    right = np.array([[0.0, 1.0]], dtype=np.float32)
    assert np.allclose(cosine_pairs(left, right), [0.0])


def test_cosine_pairs_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="Shape mismatch"):
        cosine_pairs(np.zeros((2, 3), dtype=np.float32), np.zeros((3, 3), dtype=np.float32))


def _scored() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "method": ["extractive"] * 3 + ["abstractive"] * 3,
            "category": ["Crime and law"] * 6,
            "lang": ["en"] * 6,
            "title": [f"t{i}" for i in range(6)],
            "similarity": [0.95, 0.85, 0.60, 0.90, 0.75, 0.55],
            "lexical_overlap": [1.0, 0.98, 0.97, 0.5, 0.45, 0.4],
            "compression": [0.4, 0.3, 0.1, 0.35, 0.25, 0.08],
            "source_chars": [1000, 1200, 4000, 1000, 1200, 4000],
            "summary_chars": [400, 360, 400, 350, 300, 320],
        }
    )


def test_distribution_reports_the_share_above_the_threshold():
    summary = distribution(_scored(), threshold=0.8).set_index("method")
    assert summary.loc["extractive", "pct_above_0.8"] == pytest.approx(66.7, abs=0.1)


def test_extremes_returns_the_best_and_worst_per_method():
    highest, lowest = extremes(_scored(), n=1)
    assert set(highest["method"]) == {"extractive", "abstractive"}
    assert lowest["similarity"].max() < highest["similarity"].min()


def test_drivers_correlates_similarity_with_its_candidate_explanations():
    result = drivers(_scored())

    assert set(result.index) == {"extractive", "abstractive"}
    assert "compression" in result.columns
    assert result["compression"].notna().all()


def test_drivers_leaves_a_constant_feature_undefined_rather_than_warning():
    # Extractive overlap is 1.0 by construction, and a rank correlation against
    # a constant is undefined. NaN, not a coincidental number and no warning.
    scored = _scored()
    scored.loc[scored["method"] == "extractive", "lexical_overlap"] = 1.0

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = drivers(scored)

    assert pd.isna(result.loc["extractive", "lexical_overlap"])
    assert result.loc["abstractive", "lexical_overlap"] == pytest.approx(1.0)


def test_drivers_does_not_pool_methods_that_occupy_different_feature_ranges():
    # Within each method overlap rises with similarity; the abstractive block
    # sits at lower overlap and higher similarity, so pooling inverts the sign.
    scored = pd.DataFrame(
        {
            "method": ["extractive"] * 3 + ["abstractive"] * 3,
            "similarity": [0.60, 0.65, 0.70, 0.85, 0.90, 0.95],
            "lexical_overlap": [0.90, 0.95, 1.00, 0.30, 0.35, 0.40],
            "compression": [0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
        }
    )
    pooled = scored["similarity"].corr(scored["lexical_overlap"], method="spearman")
    within = drivers(scored)["lexical_overlap"]

    assert pooled < 0
    assert (within > 0).all(), "the pooled sign is the opposite of every method's own"
