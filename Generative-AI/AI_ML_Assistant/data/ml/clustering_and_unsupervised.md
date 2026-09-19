---
title: Clustering and Unsupervised Learning
subject: ml
topic: unsupervised learning
difficulty: intermediate
year: 2026
source: In-house study note
---

# Clustering and Unsupervised Learning

**Unsupervised learning** finds structure in data that has *no labels*. Instead of mapping
inputs to known targets, it answers "how is this data organised?" — grouping similar points
(**clustering**), compressing them into fewer dimensions (**dimensionality reduction**), or
estimating the density they were drawn from. It is the go-to when labelling is expensive,
for exploratory analysis, and for building features that feed supervised models.

## k-means clustering

**k-means** partitions $n$ points into $k$ clusters by minimising the within-cluster sum of
squared distances to each cluster's centroid:

$$J = \sum_{i=1}^{k} \sum_{x \in C_i} \lVert x - \mu_i \rVert^2$$

It alternates two steps until convergence: **assign** each point to its nearest centroid,
then **update** each centroid to the mean of its points. It is fast and simple but assumes
roughly spherical, equally-sized clusters, is sensitive to initialisation (use **k-means++**
seeding), and needs $k$ chosen up front — the **elbow method** or **silhouette score** help.

## Hierarchical and density-based clustering

- **Hierarchical (agglomerative)** clustering repeatedly merges the two closest clusters,
  producing a *dendrogram* you can cut at any level — no need to pre-commit to $k$.
- **DBSCAN** groups points in dense regions and labels sparse points as noise. It finds
  arbitrarily-shaped clusters and needs no $k$, but is sensitive to its radius $\varepsilon$
  and struggles with clusters of very different densities.

## Dimensionality reduction

High-dimensional data is sparse, slow, and hard to visualise (the *curse of dimensionality*).

- **PCA** (Principal Component Analysis) finds orthogonal directions of maximum variance and
  projects data onto the top few, preserving as much variance as possible. It is linear,
  fast, and great for denoising and compression.
- **t-SNE** and **UMAP** are non-linear methods that preserve *local* neighbourhood structure,
  excellent for 2-D visualisation of clusters — but their distances and cluster sizes should
  not be over-interpreted.

## Evaluating without labels

Because there is no ground truth, use *internal* metrics: the **silhouette score** (how well
each point fits its cluster vs. the next-nearest), the **Davies–Bouldin index**, or
downstream task performance. If a few labels exist, *external* metrics like **adjusted Rand
index** compare the clustering to them. Always pair metrics with a domain sanity-check —
statistically tidy clusters are only useful if they mean something.
