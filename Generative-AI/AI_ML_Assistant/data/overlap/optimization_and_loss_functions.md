---
title: Loss Functions and Optimization
subject: overlap
topic: optimization
difficulty: intermediate
year: 2026
source: In-house study note
---

# Loss Functions and the Optimization View of Learning

Almost every learning algorithm — from linear regression to GPT — is the same recipe: choose
a **loss function** that scores how wrong the model is, then adjust parameters to minimise
it. The loss is where the task's meaning enters the mathematics; this is the deepest overlap
between classical ML and modern AI.

## Regression losses

**Mean squared error** $L = \frac{1}{n}\sum_i (y_i - \hat{y}_i)^2$ corresponds to maximum
likelihood under Gaussian noise; it punishes large errors quadratically, so outliers dominate.
**Mean absolute error** is robust to outliers (Laplacian noise assumption); **Huber loss**
interpolates: quadratic near zero, linear in the tails.

## Classification: cross-entropy

For a predicted class distribution $\hat{p}$ and true class $c$, **cross-entropy loss** is
$L = -\log \hat{p}_c$. It is maximum likelihood for categorical outcomes, and its gradient
through a softmax is beautifully simple ($\hat{p} - y$), which keeps training stable.
Confidently wrong predictions are punished hard: predicting $\hat{p}_c = 0.01$ for the true
class costs $-\log 0.01 \approx 4.6$, versus $0.05$ for a comfortable $\hat{p}_c = 0.95$.

## Language models minimise cross-entropy too

Next-token prediction is just classification over the vocabulary at every position. The
pre-training loss of an LLM is the average cross-entropy of the true next token —
equivalently, models minimise **perplexity** $= e^{L}$. The entire modern AI stack rests on
the same loss classical ML uses for logistic regression, applied ~trillions of times.

## Contrastive losses — how embedding models are trained

Contrastive objectives (InfoNCE, triplet loss) don't score a single prediction; they shape a
*geometry*: embeddings of positive pairs (a sentence and its paraphrase, an image and its
caption) are pulled together while negatives are pushed apart:

$$L = -\log \frac{e^{\mathrm{sim}(u, v^+)/\tau}}{\sum_{v} e^{\mathrm{sim}(u, v)/\tau}}$$

CLIP, sentence-transformers, and most retrieval embedders are trained this way — which is
why cosine similarity in the resulting space means semantic similarity.

## Regularised objectives

What is actually minimised is usually loss + penalty: $J(\theta) = L(\theta) + \lambda
R(\theta)$, with $R$ an L2/L1 norm (see bias–variance note), or a KL-divergence term as in
RLHF, where the policy is kept close to its reference model. Reading a paper's objective
function line by line — what is rewarded, what is penalised, what is constrained — is the
fastest way to understand what a model is really being asked to do.

## The optimization loop, once more

Whatever the loss: compute it on a batch, backpropagate gradients, update with SGD/Adam,
repeat. Learning-rate schedules, gradient clipping (essential for transformer stability),
and mixed-precision arithmetic are engineering refinements of that single loop.
