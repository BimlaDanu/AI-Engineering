"""Pluggable embedding backends used for document indexing and search.

This module supports two interchangeable embedding options through one factory:

- **local** — uses ``sentence-transformers`` on the local machine. It is free,
  works offline, and does not use API tokens during ingestion (the original
  project rule).

- **api** — uses an OpenAI-compatible embedding API (Text Embedding 3, Qwen3)
  through OpenRouter. It can provide stronger retrieval quality but uses tokens
  and may add cost during re-indexing.

Each backend creates vectors with a different size, so changing the embedding
backend requires running a full ``make ingest`` again. Always use the same
backend for both the index and the runtime retriever.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from src.config import (
    API_EMBEDDING_MODEL,
    EMBEDDING_BACKEND,
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL_NAME,
    openrouter_api_key,
)


class LocalEmbeddings(Embeddings):
    """sentence-transformers wrapper — local and free, so ingestion costs no API tokens."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        from sentence_transformers import SentenceTransformer  # heavy import, deferred

        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents into unit-normalised vectors."""
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string into a unit-normalised vector."""
        return self._model.encode([text], normalize_embeddings=True)[0].tolist()


def make_embeddings(backend: str | None = None, model: str | None = None) -> Embeddings:
    """Return the configured embedding backend; ``backend`` overrides the config default.

    ``backend`` is ``"local"`` or ``"api"``. The API backend uses the OpenAI-compatible
    embeddings endpoint via OpenRouter and requires ``OPENROUTER_API_KEY``.
    """
    backend = (backend or EMBEDDING_BACKEND).lower()
    if backend == "api":
        from langchain_openai import OpenAIEmbeddings

        key = openrouter_api_key()
        if not key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set, but the API embedding backend needs it. "
                "Add the key to your .env, or run fully offline by setting "
                "EMBEDDING_BACKEND=local (free sentence-transformers) — then re-run "
                "`make ingest` so the index matches the chosen backend."
            )
        return OpenAIEmbeddings(
            model=model or API_EMBEDDING_MODEL,
            api_key=key,
            base_url=EMBEDDING_BASE_URL,
            # The model id is an OpenRouter slug, not a tiktoken-known name, so disable
            # the client-side context-length tokenisation that would otherwise error.
            check_embedding_ctx_length=False,
        )
    return LocalEmbeddings(model or EMBEDDING_MODEL_NAME)
