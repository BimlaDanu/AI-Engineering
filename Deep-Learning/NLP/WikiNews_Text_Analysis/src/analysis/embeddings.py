"""Sentence embeddings, shared by the clustering and similarity stages.

The encoder is multilingual, so a Spanish summary is scored against its
Spanish source on meaning rather than on vocabulary overlap. Everything runs
on the CPU.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from src import config
from src.log import get_logger

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def load_encoder() -> SentenceTransformer:
    """Load the multilingual sentence encoder once per process.

    Returns:
        The encoder, pinned to the CPU.
    """
    from sentence_transformers import SentenceTransformer

    logger.info("loading encoder %s (CPU)", config.EMBEDDING_MODEL)
    return SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")


def encode(texts: list[str], show_progress: bool = False) -> np.ndarray:
    """Embed a list of texts as unit-length vectors.

    Normalising here makes a later cosine similarity a plain dot product.

    Args:
        texts: The texts to embed.
        show_progress: Whether to print a progress bar.

    Returns:
        An array of shape ``(len(texts), dim)``. Empty input gives an empty
        array rather than raising, so callers can pass a filtered list safely.
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    vectors = load_encoder().encode(
        texts,
        batch_size=config.EMBEDDING_BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
    )
    return np.asarray(vectors, dtype=np.float32)


def encode_documents(texts: list[str], chunk_chars: int = 900) -> np.ndarray:
    """Embed documents longer than the encoder's input window.

    The encoder truncates its input, so a long article embedded directly would
    be represented by its opening paragraph alone. Long texts are split into
    chunks and the chunk vectors averaged into one document vector.

    Args:
        texts: The documents to embed.
        chunk_chars: Target chunk size. Split happens at a sentence boundary
            near this length so chunks do not start mid-sentence.

    Returns:
        One unit-length vector per document, shape ``(len(texts), dim)``.
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    chunks: list[str] = []
    owners: list[int] = []
    for index, text in enumerate(texts):
        for chunk in _split_into_chunks(text, chunk_chars):
            chunks.append(chunk)
            owners.append(index)

    vectors = encode(chunks)
    dimension = vectors.shape[1]

    document_vectors = np.zeros((len(texts), dimension), dtype=np.float32)
    for vector, owner in zip(vectors, owners, strict=True):
        document_vectors[owner] += vector

    norms = np.linalg.norm(document_vectors, axis=1, keepdims=True)
    return document_vectors / np.maximum(norms, 1e-12)


def _split_into_chunks(text: str, chunk_chars: int) -> list[str]:
    """Cut a document into chunks at sentence boundaries.

    Args:
        text: The document.
        chunk_chars: Target chunk length in characters.

    Returns:
        The chunks. A document shorter than the target comes back as one chunk;
        an empty document comes back as a single empty string so that every
        document still owns at least one vector.
    """
    stripped = text.strip()
    if len(stripped) <= chunk_chars:
        return [stripped or " "]

    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for sentence in re.split(r"(?<=[.!?])\s+", stripped):
        if length + len(sentence) > chunk_chars and current:
            chunks.append(" ".join(current))
            current, length = [], 0
        current.append(sentence)
        length += len(sentence)

    if current:
        chunks.append(" ".join(current))
    return chunks


def cosine_pairs(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Cosine similarity between matching rows of two embedding matrices.

    Args:
        left: Embeddings of shape ``(n, dim)``, unit length.
        right: Embeddings of the same shape, unit length.

    Returns:
        One similarity per row, clipped to ``[-1, 1]`` to absorb the float
        error that pushes identical vectors just past 1.0.

    Raises:
        ValueError: If the two matrices do not have the same shape.
    """
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: {left.shape} vs {right.shape}")
    return np.clip(np.sum(left * right, axis=1), -1.0, 1.0)
