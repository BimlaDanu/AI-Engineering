---
title: Evaluating LLMs and RAG Systems
subject: ai
topic: evaluation
difficulty: intermediate
year: 2026
source: In-house study note
---

# Evaluating LLMs and RAG Systems

Evaluating generative systems is hard because there is rarely one correct output. A good
evaluation strategy layers **automatic metrics**, **LLM-as-judge** scoring, and **human
review**, and always ties numbers to the behaviour you actually care about.

## Classic reference-based metrics

When a reference answer exists, string-overlap metrics give cheap signal: **BLEU** and
**ROUGE** (n-gram overlap, common in translation/summarisation) and **exact match / F1**
(for extractive QA). They are fast and reproducible but blind to meaning — a correct
paraphrase scores poorly, and a fluent wrong answer can score well. **BERTScore** improves on
this by comparing embeddings rather than surface tokens.

## LLM-as-judge

For open-ended outputs, a strong LLM can *grade* another model's answer against a rubric or
compare two answers head-to-head. It correlates well with human judgement, scales cheaply,
and can explain its scores. Cautions: judges show **position bias** (favouring the first
answer), **verbosity bias** (favouring longer answers), and **self-preference**. Mitigate by
randomising order, fixing temperature to 0, using a rubric, and validating against human
labels on a sample. Returning **structured** scores (a JSON schema, not prose) makes judgements
machine-readable and stable.

## RAG-specific metrics

RAG has two failure modes — bad *retrieval* and bad *generation* — so it needs metrics that
separate them. The **RAGAs** framework popularised four:

- **Faithfulness** — are the answer's claims supported by the retrieved context? (Detects
  hallucination.)
- **Answer relevancy** — does the answer actually address the question?
- **Context precision** — are the retrieved passages relevant, and ranked well?
- **Context recall** — did retrieval fetch everything needed to answer?

Faithfulness and context precision isolate the *generator*; context recall isolates the
*retriever*. Together they tell you *where* to fix a pipeline rather than just that it scored
low.

## Building a golden set and running offline

Assemble a **golden set** — representative questions with reference answers and (for RAG) the
passages that should be retrieved. Run it **offline and deterministically** (fixed model,
temperature 0, versioned dataset) so scores are comparable across changes. This turns
evaluation into a regression test: an **A/B comparison** runs the same golden set through two
configurations under the same judge, so any metric difference is attributable to the change,
not to noise.

## Beyond accuracy

Production evaluation also tracks **latency**, **cost per query**, **safety/refusal** rates,
and **robustness** to prompt injection and adversarial inputs. A system that is accurate but
slow, expensive, or exploitable is not actually good.
