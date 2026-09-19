"""Build the Chroma vector store from documents stored in ``data/``.

Supports:
- Markdown files with a simple ``key: value`` front-matter section.
- Plain text files (``.txt``).
- PDF files (``.pdf``).

The document subject (``ml`` / ``dl`` / ``ai`` / ``overlap``) is taken from the
parent folder name by default and can be changed using front matter.

Run ingestion with:
``make ingest``
or:
``python -m src.rag.ingest``

The embedding backend is selected by
:func:`src.rag.embeddings.make_embeddings` using the  config ``EMBEDDING_BACKEND``
configuration. The runtime retriever must use the same embedding backend that
was used to build the index.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from chromadb.api.shared_system_client import SharedSystemClient
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHROMA_DIR, COLLECTION_NAME, DATA_DIR
from src.rag.embeddings import make_embeddings

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
VALID_SUBJECTS = {"ml", "dl", "ai", "overlap"}
# `source` is always the filename (canonical id for stats and citations); a front-matter
# `source:` line is stored as `origin` (human-readable provenance) instead.
_META_KEYS = {"title", "subject", "topic", "difficulty", "year", "origin"}


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Parse a ``key: value`` front-matter block delimited by ``---`` lines."""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    meta: dict[str, str] = {}
    for line in parts[1].strip().splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip().lower()] = value.strip()
    return meta, parts[2].strip()


def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    """Load every .md/.txt/.pdf under data/, attaching subject/topic/difficulty metadata."""
    docs: list[Document] = []
    if not data_dir.exists():
        return docs
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt", ".pdf"}:
            continue
        folder = path.parent.name
        base_meta = {
            "source": path.name,
            "title": path.stem.replace("_", " ").title(),
            "subject": folder if folder in VALID_SUBJECTS else "overlap",
            "topic": "general",
            "difficulty": "beginner",
        }
        try:
            if path.suffix.lower() == ".pdf":
                from pypdf import PdfReader

                text = "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
                meta = base_meta
            else:
                fm, text = parse_front_matter(path.read_text(encoding="utf-8"))
                if "source" in fm:  # keep `source` = filename; provenance goes to `origin`
                    fm["origin"] = fm.pop("source")
                meta = base_meta | {k: v for k, v in fm.items() if k in _META_KEYS}
        except Exception as exc:  # one unreadable file must not sink the whole rebuild
            print(f"  ! skipped {path.name} (unreadable): {exc}")
            continue
        if meta.get("subject") not in VALID_SUBJECTS:
            meta["subject"] = base_meta["subject"]
        if text.strip():
            docs.append(Document(page_content=text, metadata=meta))
        else:
            print(f"  ! skipped {path.name}: no extractable text")
    return docs


# Where a rebuild is assembled before it replaces the live index. A sibling of CHROMA_DIR so
# the swap is a rename within one filesystem; removed on every exit path, so it exists only
# for the length of a rebuild.
STAGING_DIR = CHROMA_DIR.with_name(CHROMA_DIR.name + ".staging")


def build_vector_store(docs: list[Document]) -> int:
    """Chunk, embed, and persist the documents; returns the number of chunks.

    Builds into :data:`STAGING_DIR` and swaps it into place only once every chunk has been
    embedded successfully. The order matters more than it looks: this function used to delete
    CHROMA_DIR first and embed afterwards, so any failure of the embedding call — an expired
    key, exhausted credit, a rate limit, a dropped connection — destroyed the knowledge base
    and left nothing to fall back to. Embedding is a network call over hundreds of chunks, so
    that is not a remote possibility; it is the failure this operation is most likely to have.

    Worse, the loss was quiet. The caller clears the ``load_kb_status`` cache only after a
    successful return, so a failed rebuild left the app answering from the KnowledgeBase still
    held in memory while the index behind it was gone — healthy right up until the next
    restart. Nothing is deleted now until a complete replacement exists on disk.

    Raises:
        ValueError: If the documents produce no chunks. Wiping a good index in order to
            replace it with an empty one is never what the caller meant.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise ValueError(
            "No text to index — add .md/.txt/.pdf files under data/ before re-indexing. "
            "The existing index has been left untouched."
        )
    # Drop any chromadb client this process already opened on CHROMA_DIR (e.g. the app's
    # cached KnowledgeBase). chromadb caches systems process-wide and runs schema migrations
    # only once per system; without this, a rebuild reuses the stale system and the freshly
    # wiped SQLite file never gets its `tenants` table ("no such table: tenants").
    SharedSystemClient.clear_system_cache()
    shutil.rmtree(STAGING_DIR, ignore_errors=True)  # a previous run killed mid-rebuild
    # The Rust-based chromadb client does not create a missing persist directory, so SQLite
    # fails with "unable to open database file"; create it before handing the path over.
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    try:
        Chroma.from_documents(
            chunks,
            make_embeddings(),
            collection_name=COLLECTION_NAME,
            persist_directory=str(STAGING_DIR),
            collection_metadata={"hnsw:space": "cosine"},
        )
        # Release the client holding the staging database before the directory moves out from
        # under it, for the same reason the cache is cleared above.
        SharedSystemClient.clear_system_cache()
        if CHROMA_DIR.exists():
            shutil.rmtree(CHROMA_DIR)  # full rebuild keeps chunk ids and BM25 corpus in sync
        STAGING_DIR.replace(CHROMA_DIR)
    except BaseException:
        # Including KeyboardInterrupt: a rebuild cancelled at the terminal should also leave
        # the live index alone rather than half a directory beside it.
        shutil.rmtree(STAGING_DIR, ignore_errors=True)
        raise
    SharedSystemClient.clear_system_cache()  # the next open must see the new path, not the old
    return len(chunks)


def main() -> None:
    """CLI entry point: load documents from data/ and (re)build the vector store."""
    docs = load_documents()
    if not docs:
        print(f"No documents found in {DATA_DIR} — add .md/.txt/.pdf files first.")
        return
    n_chunks = build_vector_store(docs)
    print(f"Ingested {len(docs)} documents into {n_chunks} chunks at {CHROMA_DIR}.")


if __name__ == "__main__":
    main()
