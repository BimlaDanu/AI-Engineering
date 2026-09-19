---
title: Classical ML Models Overview
subject: ml
topic: models
difficulty: beginner
year: 2026
source: In-house study note
---

# Classical Machine Learning Models

Before deep learning, and still today for tabular data, a small set of model families covers
most practical problems.

## Linear and logistic regression

**Linear regression** fits $\hat{y} = \theta^\top x$ by minimising squared error — the
simplest, most interpretable baseline; always try it first. **Logistic regression** pushes
the linear score through a sigmoid $\sigma(z) = 1/(1+e^{-z})$ to output a probability for
binary classification, trained with cross-entropy loss. Despite the name, it is a
classification model. Coefficients are directly interpretable as log-odds contributions.

## Decision trees and ensembles

A **decision tree** recursively splits the feature space ("is age > 30?") to maximise purity
(information gain / Gini decrease). Single trees are interpretable but high-variance.

- **Random forests** average many trees, each trained on a bootstrap sample with random
  feature subsets — variance drops dramatically, and they are hard to beat with almost no
  tuning.
- **Gradient boosting** (XGBoost, LightGBM, CatBoost) builds trees sequentially, each one
  fitting the residual errors of the ensemble so far. Boosted trees remain the strongest
  models for most tabular datasets, frequently outperforming neural networks there.

## Support vector machines

An SVM finds the separating hyperplane with the **maximum margin** to the nearest training
points (the support vectors). The **kernel trick** replaces dot products with a kernel
$K(x, x')$, implicitly mapping data into a high-dimensional space where a linear separator
exists — RBF kernels handle smoothly nonlinear boundaries. SVMs shine on small-to-medium,
high-dimensional datasets but scale poorly beyond ~10⁵ samples.

## k-nearest neighbours and k-means

**k-NN** classifies a point by voting among its $k$ closest training examples — no training
phase, but predictions are slow and it suffers in high dimensions (the curse of
dimensionality: distances concentrate and lose meaning). **k-means** is its unsupervised
cousin for clustering: alternate between assigning points to the nearest centroid and moving
centroids to their cluster mean.

## Naive Bayes

Applies Bayes' theorem with the "naive" assumption that features are conditionally
independent given the class. Wrong in theory, surprisingly effective in practice for text
classification (spam filtering), and extremely fast.

## When to use what

Start with logistic/linear regression for a baseline and interpretability; move to gradient
boosting for maximum tabular accuracy; use SVMs for small high-dimensional data; reserve
neural networks for images, audio, text, and other perceptual data where feature learning
matters more than feature engineering.
