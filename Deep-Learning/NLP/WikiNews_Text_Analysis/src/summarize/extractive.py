"""Extractive summarization with TextRank.

Sentences are nodes in a graph, edges are TF-IDF cosine similarity, and
PageRank scores them. The top sentences are returned in document order, not in
score order.

The vectorizer is fitted per article rather than per corpus, which keeps the
method language-independent: no stop-word list or stemmer is needed.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Below this, two sentences share so little vocabulary that the edge is noise
# and PageRank flattens towards uniform.
_MIN_EDGE_WEIGHT = 0.05


def sentence_similarity_matrix(sentences: list[str]) -> np.ndarray:
    """Build the pairwise TF-IDF cosine similarity matrix for one article.

    Args:
        sentences: The article's sentences.

    Returns:
        A square matrix with a zeroed diagonal and weak edges removed. An
        article whose sentences hold no countable word comes back edgeless,
        which leaves PageRank nothing to rank on.
    """
    vectorizer = TfidfVectorizer(lowercase=True, sublinear_tf=True)
    try:
        vectors = vectorizer.fit_transform(sentences)
    except ValueError:
        # Every sentence is punctuation or single characters, so the vectorizer
        # builds an empty vocabulary and raises. There is no vocabulary to
        # compare on; an edgeless graph sends the caller to document order.
        return np.zeros((len(sentences), len(sentences)), dtype=float)

    similarity = cosine_similarity(vectors)
    np.fill_diagonal(similarity, 0.0)
    similarity[similarity < _MIN_EDGE_WEIGHT] = 0.0
    return similarity


def rank_sentences(sentences: list[str]) -> list[float]:
    """Score every sentence by PageRank over the similarity graph.

    Args:
        sentences: The article's sentences.

    Returns:
        One score per sentence, in the input order.

    Raises:
        ValueError: If no sentences were given.
    """
    if not sentences:
        raise ValueError("Cannot rank an empty list of sentences")
    if len(sentences) == 1:
        return [1.0]

    graph = nx.from_numpy_array(sentence_similarity_matrix(sentences))
    try:
        scores = nx.pagerank(graph, weight="weight")
    except nx.PowerIterationFailedConvergence:
        # Almost edgeless graph: there is nothing to rank on, so fall back to
        # document order.
        return [1.0 / (index + 1) for index in range(len(sentences))]
    return [scores.get(index, 0.0) for index in range(len(sentences))]


def summarize(sentences: list[str], n_sentences: int = 3) -> str:
    """Select the most central sentences of an article.

    Args:
        sentences: The article's sentences, in document order.
        n_sentences: How many sentences to keep.

    Returns:
        The selected sentences joined by spaces, in document order. Returns the
        whole text when the article is already at or under the target length.
    """
    if len(sentences) <= n_sentences:
        return " ".join(sentences)

    scores = rank_sentences(sentences)
    best = sorted(range(len(sentences)), key=lambda index: scores[index], reverse=True)
    chosen = sorted(best[:n_sentences])
    return " ".join(sentences[index] for index in chosen)
