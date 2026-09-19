---
title: Bias, Variance and Regularization
subject: ml
topic: generalization
difficulty: intermediate
year: 2026
source: In-house study note
---

# Bias–Variance Trade-off and Regularization

A model's expected prediction error on unseen data decomposes into three parts:

$$\mathbb{E}[(y - \hat{f}(x))^2] = \underbrace{\text{Bias}^2}_{\text{too simple}} +
\underbrace{\text{Variance}}_{\text{too sensitive}} + \underbrace{\sigma^2}_{\text{noise}}$$

**Bias** is error from wrong assumptions: a linear model fitted to a curved relationship
underfits no matter how much data you give it. **Variance** is error from sensitivity to the
particular training sample: a deep decision tree can memorise noise, so retraining on a new
sample gives a very different model — it overfits. Irreducible noise $\sigma^2$ is the floor
no model can beat.

Classic symptoms:

- High bias (underfitting): training error and validation error are both high and close.
- High variance (overfitting): training error is low, validation error is much higher.

## Regularization: trading a little bias for a lot less variance

**Ridge (L2)** adds $\lambda \sum_j \theta_j^2$ to the loss, shrinking all weights smoothly
toward zero. It handles correlated features gracefully and rarely hurts.

**Lasso (L1)** adds $\lambda \sum_j |\theta_j|$, which drives some weights exactly to zero —
built-in feature selection. Elastic Net mixes the two.

In deep learning the same principle appears as **weight decay**, **dropout** (randomly
zeroing activations during training so the network cannot rely on any single unit),
**early stopping** (halt when validation loss stops improving — the training trajectory
itself acts as a regulariser), and **data augmentation** (enlarging the effective dataset).

## Choosing $\lambda$: cross-validation

Split the training data into $k$ folds; train on $k-1$, validate on the held-out fold,
rotate, and average. Pick the $\lambda$ (or any hyperparameter) with the best mean
validation score, then retrain on all data. Never tune hyperparameters on the test set —
that leaks information and inflates your reported performance.

## Modern nuance: double descent

Very large neural networks violate the classical U-shaped validation curve: past the
interpolation threshold (enough parameters to fit training data perfectly), test error can
*decrease again* as models grow. This "double descent" phenomenon is one reason enormous
LLMs generalise despite having far more parameters than training constraints suggest is
sensible — the classical bias–variance picture is a guide, not a law.
