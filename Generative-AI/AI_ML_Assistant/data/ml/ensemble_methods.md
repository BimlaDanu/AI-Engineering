---
title: Ensemble Methods
subject: ml
topic: ensembles
difficulty: intermediate
year: 2026
source: In-house study note
---

# Ensemble Methods

An **ensemble** combines many models so the group predicts better than any single member.
The intuition is the *wisdom of crowds*: if individual models make different, roughly
independent errors, averaging them cancels noise while preserving signal. Formally, the
expected error of a model decomposes into **bias**, **variance**, and irreducible noise
(see the bias–variance note); the two dominant ensemble families each attack one term.

## Bagging (reduces variance)

**Bagging** (bootstrap aggregating) trains many high-variance models in parallel on
different *bootstrap samples* — datasets of the same size drawn with replacement — then
averages their outputs (or takes a majority vote for classification). Because each model
sees a slightly different dataset, their errors decorrelate and the averaged prediction has
lower variance without raising bias.

The **random forest** is bagging over decision trees with one extra twist: at each split
only a random subset of features is considered, which further decorrelates the trees.
Random forests are a strong, low-tuning baseline and give a free **out-of-bag** error
estimate (each tree is validated on the ~37% of rows it did not see).

## Boosting (reduces bias)

**Boosting** trains models *sequentially*, each one focusing on the examples the previous
models got wrong. **AdaBoost** reweights misclassified points; **gradient boosting** fits
each new model to the *negative gradient of the loss* (the residual errors), so adding it
takes a step downhill in function space:

$$F_{m}(x) = F_{m-1}(x) + \eta \, h_m(x)$$

where $h_m$ is the new weak learner (usually a shallow tree) and $\eta$ is a learning rate.
Modern libraries — **XGBoost**, **LightGBM**, **CatBoost** — are regularised, heavily
optimised gradient boosting and dominate tabular-data competitions.

## Stacking

**Stacking** trains a *meta-model* to combine the predictions of several diverse base models
(e.g. a random forest, a boosted tree, a linear model). The base models' out-of-fold
predictions become features for the meta-model, letting it learn *which* model to trust in
*which* region of the input space.

## Practical notes

- **Bagging** helps unstable, low-bias learners (deep trees); **boosting** helps stable,
  high-bias learners (stumps) but can overfit if run too long — use early stopping.
- Ensembles trade interpretability and inference cost for accuracy. Feature-importance and
  SHAP values partly recover interpretability.
- Diversity is the fuel: identical models ensemble to nothing. Vary the data, features,
  algorithm, or random seed.
