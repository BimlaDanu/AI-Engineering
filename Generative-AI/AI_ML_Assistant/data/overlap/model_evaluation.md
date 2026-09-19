---
title: Model Evaluation
subject: overlap
topic: evaluation
difficulty: beginner
year: 2026
source: In-house study note
---

# Evaluating Models — from Accuracy to LLM Judges

You cannot improve what you cannot measure, and you cannot trust what you measured wrongly.
Evaluation spans classical ML and modern AI with the same core discipline: hold out data the
model never saw, and pick metrics that reflect the real cost of mistakes.

## Classification metrics

From the confusion matrix (true/false positives/negatives):

- **Accuracy** — fraction correct. Misleading under class imbalance: a 99%-negative dataset
  gives a do-nothing classifier 99% accuracy.
- **Precision** $= \frac{TP}{TP + FP}$ — of the items I flagged, how many were right?
  Matters when false alarms are costly (spam filters).
- **Recall** $= \frac{TP}{TP + FN}$ — of the items I should have flagged, how many did I
  catch? Matters when misses are costly (medical screening).
- **F1** — harmonic mean of precision and recall; **ROC-AUC** — ranking quality across all
  thresholds.

Precision and recall trade off against each other via the decision threshold; choose the
operating point from the application's cost matrix, not by habit.

## Sound methodology

Split data into train / validation / test. Tune everything (hyperparameters, prompts,
thresholds) on validation; touch the test set once, at the end. **Cross-validation** gives
more reliable estimates on small data. Beware **leakage** — any information from the test
set (or from the future, in time-series) reaching training inflates results; it is the most
common cause of "too good to be true" numbers.

## Evaluating LLMs and RAG systems

Free-text output breaks fixed-answer metrics, so the field uses:

- **Benchmarks** (MMLU, GSM8K, HumanEval, …) — standardised task suites; useful but
  saturating, and contamination (test data leaking into training corpora) is a real risk.
- **Human evaluation** — pairwise preference votes, e.g. Chatbot Arena's Elo-style
  leaderboard; the gold standard, but slow and expensive.
- **LLM-as-judge** — a strong model grades outputs against a rubric. Scalable and
  surprisingly consistent, but carries biases: position bias (order of candidates matters),
  verbosity bias (longer looks better), and self-preference (models favour their own style).
  Mitigate by swapping orders, fixing rubrics, and spot-checking against humans.

For **RAG specifically**, evaluate the stages separately: retrieval quality via context
precision/recall (did the right chunks arrive?), and generation quality via faithfulness
(is every claim supported by the retrieved context?) and answer relevance. The **RAGAS**
framework automates these four metrics with LLM judges, giving a regression suite you can
run whenever you change chunking, embeddings, or prompts.

## The habit that matters

Before shipping any change — a new prompt, a new chunk size, a new model — run the same
fixed evaluation set and compare. Vibes are not a metric.
