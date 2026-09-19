---
title: Vector Databases and Indexing
subject: ai
topic: vector search
difficulty: intermediate
year: 2026
source: In-house study note
---

# Vector Databases and Indexing

A **vector database** stores items as high-dimensional **embeddings** and answers *similarity*
queries: "given this query vector, find the nearest stored vectors." It is the retrieval
engine underneath RAG, semantic search, recommendation, and deduplication. The core operation
is **nearest-neighbour search** in embedding space, where distance encodes semantic similarity.

## Exact vs. approximate search

Exact nearest-neighbour search compares the query to *every* stored vector — accurate but
$O(n)$ per query, too slow for millions of items. **Approximate Nearest Neighbour (ANN)**
indexes trade a little recall for orders-of-magnitude speed:

- **HNSW** (Hierarchical Navigable Small World) builds a multi-layer proximity graph and
  greedily walks it toward the query. It is the default in most modern vector stores —
  excellent recall/latency, higher memory use. Tuning knobs: `M` (graph connectivity) and
  `efSearch` (search breadth vs. speed).
- **IVF** (Inverted File) clusters vectors and searches only the nearest few clusters.
- **PQ** (Product Quantisation) compresses vectors so huge indexes fit in memory, at some
  accuracy cost. Often combined as **IVF-PQ**.

## Distance metrics

Similarity is measured by a metric that must **match how the embeddings were trained**:

$$\text{cosine}(a,b) = \frac{a \cdot b}{\lVert a \rVert \, \lVert b \rVert}$$

**Cosine similarity** (angle, ignoring magnitude) is most common for text embeddings;
**dot product** and **Euclidean (L2)** distance are also used. Normalising vectors makes
cosine and dot product equivalent.

## Chunking and metadata

Documents are split into **chunks** before embedding — small enough to be specific, large
enough to be self-contained (a few hundred tokens, often with overlap so ideas are not cut
mid-sentence). Each chunk carries **metadata** (source, title, subject, date) so retrieval
can *filter* ("only ML docs from 2024") as well as rank, and so answers can cite provenance.
Chunking strategy often matters more to RAG quality than the choice of index.

## Hybrid search

Dense embeddings capture meaning but can miss exact keywords, rare terms, and identifiers.
**Hybrid search** blends a dense vector score with a sparse **BM25** keyword score:

$$\text{score} = \alpha \cdot \text{vector} + (1 - \alpha) \cdot \text{bm25}$$

Tuning $\alpha$ trades semantic recall against keyword precision. A **re-ranker** (a
cross-encoder that scores each candidate against the query) can then reorder the top results
for a final precision boost.
