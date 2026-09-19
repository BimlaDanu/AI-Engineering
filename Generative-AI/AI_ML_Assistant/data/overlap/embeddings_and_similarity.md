---
title: Embeddings and Similarity
subject: overlap
topic: embeddings
difficulty: beginner
year: 2026
source: In-house study note
---

# Embeddings and Similarity Measures

An **embedding** is a learned mapping from a discrete object — a word, sentence, document,
image, user — to a dense vector of real numbers, arranged so that *semantic similarity
becomes geometric closeness*. "Car" and "automobile" land near each other; "car" and
"banana" land far apart. Embeddings are the shared foundation of classical ML (feature
learning, recommender systems) and modern AI engineering (vector search, RAG, semantic
caching).

## Where embeddings come from

Word2vec (2013) trained shallow networks to predict a word from its neighbours — the famous
result $\vec{king} - \vec{man} + \vec{woman} \approx \vec{queen}$ showed directions in the
space encode relations. Modern **sentence embeddings** come from transformer encoders
(e.g. sentence-transformers models) trained contrastively: similar pairs pulled together,
dissimilar pushed apart. LLM providers expose embedding endpoints; local models like
all-MiniLM-L6-v2 produce 384-dimensional vectors free of charge.

## Cosine similarity

The standard similarity measure between embeddings $u$ and $v$ is the cosine of the angle
between them:

$$\cos(\theta) = \frac{u \cdot v}{\lVert u \rVert \, \lVert v \rVert}$$

It ranges from $-1$ (opposite) through $0$ (orthogonal — unrelated) to $1$ (same direction —
same meaning). Cosine ignores vector *length*, which mostly reflects frequency and other
nuisances, and compares *direction*, which carries the meaning. If vectors are normalised to
unit length first, cosine similarity equals the dot product, and Euclidean distance becomes a
monotone function of it: $\lVert u - v \rVert^2 = 2 - 2\cos(\theta)$ — so nearest-neighbour
rankings agree.

Worked example: $u = (1, 2)$, $v = (2, 4)$ point the same way, cosine $= 1$ even though the
magnitudes differ. $u = (1, 0)$, $v = (0, 1)$: cosine $= 0$, unrelated.

## Vector databases and approximate search

Finding the most similar vectors among millions by brute force is $O(n \cdot d)$ per query.
**Vector databases** (Chroma, FAISS, Pinecone, Weaviate, pgvector, …) index embeddings with
**approximate nearest-neighbour (ANN)** structures — HNSW graphs, IVF cells, product
quantization — trading a sliver of recall for orders-of-magnitude speed. They also store the
original text and metadata per vector, enabling filtered semantic search ("only chunks with
topic = transformers").

## The curse of dimensionality — and why embeddings escape it

In high dimensions random points concentrate at similar distances, making "nearest" less
meaningful. Embeddings work anyway because real data lies near a much lower-dimensional
manifold inside the embedding space; the training objective shapes the geometry so distance
stays informative where the data actually lives.

## Practical notes

- Always embed queries and documents with the **same model**; spaces from different models
  are incompatible.
- Embeddings capture the training distribution's notion of similarity — a code-trained model
  ranks code snippets better than a general-prose model.
- Dimensionality (384–3072 today) trades accuracy against storage and search cost.
