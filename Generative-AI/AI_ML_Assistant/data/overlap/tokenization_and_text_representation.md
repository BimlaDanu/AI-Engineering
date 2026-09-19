---
title: Tokenization and Text Representation
subject: overlap
topic: text representation
difficulty: beginner
year: 2026
source: In-house study note
---

# Tokenization and Text Representation

Models compute on numbers, not letters, so every NLP system begins by turning text into a
numerical form. This happens in two stages: **tokenization** (splitting text into discrete
units) and **representation** (mapping those units to vectors). The choices here shape
vocabulary size, sequence length, cost, and how well a model handles rare or unseen words.

## Tokenization

Early systems split on whitespace into **words**, but word vocabularies explode in size and
break on typos, new words, and morphology (`run`, `running`, `ran` become unrelated).
Character-level tokenization has a tiny vocabulary but very long sequences. Modern LLMs use
**subword** tokenization, a middle ground:

- **Byte-Pair Encoding (BPE)** starts from characters and greedily merges the most frequent
  adjacent pairs into new tokens, learning common word-pieces (`token`, `##ization`).
- **WordPiece** (BERT) and **SentencePiece/Unigram** are related schemes.

Subword tokenization keeps the vocabulary bounded (tens of thousands of tokens) while
representing *any* string — unknown words are simply split into smaller known pieces. This is
why LLM cost and context limits are measured in **tokens**, not words (a rough rule of thumb:
~4 characters or ~0.75 words per token in English).

## Sparse (count-based) representations

Before neural embeddings, text was represented by counts:

- **Bag-of-words** — a vector of word counts, ignoring order.
- **TF-IDF** — weights each term by its frequency in the document (**TF**) against how rare it
  is across the corpus (**IDF**), so distinctive words dominate:

$$\text{tfidf}(t, d) = \text{tf}(t, d) \cdot \log\frac{N}{\text{df}(t)}$$

These vectors are high-dimensional, sparse, and *lexical* — they match exact words but see no
similarity between synonyms. **BM25**, a refined TF-IDF, is still the backbone of keyword
search and the sparse half of hybrid retrieval.

## Dense representations (embeddings)

**Embeddings** map tokens or whole texts to dense, low-dimensional vectors where *semantic*
similarity becomes geometric closeness — synonyms land near each other even without shared
words. Early word embeddings (**word2vec**, **GloVe**) gave one vector per word;
**contextual** embeddings from transformers give a different vector depending on surrounding
words (so "bank" of a river ≠ "bank" that holds money). These dense vectors power semantic
search, clustering, and RAG retrieval — see the embeddings and vector-search notes.

## Choosing a representation

Sparse and dense representations are complementary: sparse excels at exact terms, rare tokens,
and identifiers; dense excels at meaning and paraphrase. Production search systems increasingly
**combine both**, which is exactly the motivation for hybrid retrieval.
