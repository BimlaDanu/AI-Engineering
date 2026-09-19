---
title: Retrieval-Augmented Generation (RAG)
subject: ai
topic: rag
difficulty: intermediate
year: 2026
source: In-house study note (after Lewis et al. 2020)
---

# Retrieval-Augmented Generation

**RAG** (Lewis et al., 2020) grounds an LLM's answers in external documents: instead of
relying on whatever the model memorised during training, the system first *retrieves*
relevant passages from a knowledge base and injects them into the prompt as context.

## The pipeline

1. **Ingestion (offline):** split documents into chunks, embed each chunk into a vector,
   and store vectors + text + metadata in a vector database.
2. **Retrieval (per query):** embed the user's question with the same model and find the
   most similar chunks (cosine similarity / approximate nearest-neighbour search).
3. **Generation:** build a prompt containing the retrieved passages and the question, and
   instruct the model to answer *from the passages* and cite them.

## Why RAG instead of fine-tuning?

| | RAG | Fine-tuning |
|---|---|---|
| Knowledge updates | Re-index documents — minutes | Retrain — hours/days |
| Attribution | Can cite retrieved sources | No provenance |
| Hallucination | Reduced (grounded context) | Still possible |
| Best for | Facts, documents, freshness | Style, format, skills, domain tone |
| Cost | Per-query retrieval overhead | Upfront training cost |

They compose: fine-tune for behaviour, RAG for knowledge. Fine-tuning is *not* a reliable
way to add facts; it mostly shapes behaviour and style.

## Chunking matters

Chunks must be small enough to be precise (retrieval finds the right passage, and you can
fit several in the context window) but large enough to be self-contained. Typical: 300–1000
tokens with 10–20% overlap so sentences aren't cut mid-thought. **Parent-child chunking**
retrieves with small chunks but hands the LLM the enclosing larger section, getting precision
and context simultaneously.

## Advanced retrieval techniques

- **Query rewriting:** an LLM reformulates a vague or conversational question ("what about
  the second one?") into a self-contained search query before embedding it.
- **Multi-query retrieval:** generate several phrasings of the question, retrieve for each,
  and merge (e.g. reciprocal rank fusion) — recovers from embedding blind spots.
- **Hybrid search:** blend vector similarity with keyword scoring (BM25). Vectors capture
  paraphrase ("car" ≈ "automobile"); BM25 nails exact terms, acronyms, and code identifiers.
- **Metadata filtering:** restrict retrieval by attributes (topic, date, difficulty, source)
  stored alongside each chunk.
- **Re-ranking:** a cross-encoder re-scores the top-k candidates jointly with the query —
  slower but markedly more accurate than bi-encoder similarity alone.

## Failure modes and evaluation

RAG fails when retrieval misses the relevant chunk (garbage in → hallucination out), when
chunks contradict each other, or when the model ignores the context. Evaluate the two stages
separately: retrieval with recall@k / MRR against a labelled set, and generation with
**faithfulness** (is every claim supported by the retrieved context?) and **answer
relevance** — the RAGAS framework automates these with LLM judges.
