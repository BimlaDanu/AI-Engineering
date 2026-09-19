# Mathematical and theoretical analysis

Every method the pipeline uses, written out formally, with what the mathematics says the results
should look like. Figures quoted here come from `reports/findings.md`.

Three sections changed how the pipeline reports its numbers and carry the most detail: section 5
on why a faithful summary can score badly, section 10 on why correlations are computed inside each
method, and section 11 on what the cross-lingual recall figures do and do not measure.

## Contents

1. [Notation](#1-notation)
2. [Term weighting](#2-term-weighting)
3. [TextRank](#3-textrank)
4. [Sentence embeddings and cosine similarity](#4-sentence-embeddings-and-cosine-similarity)
5. [Why a correct summary can score low](#5-why-a-correct-summary-can-score-low)
6. [Lexical overlap](#6-lexical-overlap)
7. [The 0.8 threshold as a decision rule](#7-the-08-threshold-as-a-decision-rule)
8. [K-means and the silhouette](#8-k-means-and-the-silhouette)
9. [Topic classification](#9-topic-classification)
10. [Rank correlation and the pooling trap](#10-rank-correlation-and-the-pooling-trap)
11. [The cross-lingual recall proxy](#11-the-cross-lingual-recall-proxy)
12. [Entity counting](#12-entity-counting)
13. [Readability](#13-readability)

---

## 1. Notation

| Symbol | Meaning |
|---|---|
| $D$ | a document; $\mathcal{D}$ a collection of them |
| $S = (s_1, \dots, s_n)$ | the sentences of $D$ in document order |
| $V$ | the vocabulary |
| $\mathbf{x} \in \mathbb{R}^{\lvert V \rvert}$ | a TF-IDF vector |
| $\mathbf{u} \in \mathbb{R}^{d}$ | a dense sentence embedding, $\lVert \mathbf{u} \rVert = 1$, $d = 384$ |
| $\langle \cdot, \cdot \rangle$ | the Euclidean inner product |
| $\rho$ | mean pairwise cosine between the chunks of one document |

Both similarity measures in this project are cosines, and on unit vectors a cosine collapses to a
plain inner product:

$$
\cos(\mathbf{a}, \mathbf{b}) = \frac{\langle \mathbf{a}, \mathbf{b} \rangle}{\lVert \mathbf{a} \rVert \, \lVert \mathbf{b} \rVert}, \qquad \lVert \mathbf{a} \rVert = \lVert \mathbf{b} \rVert = 1 \;\Longrightarrow\; \cos(\mathbf{a}, \mathbf{b}) = \langle \mathbf{a}, \mathbf{b} \rangle .
$$

That is why `encode` normalises at the point of encoding. Nothing downstream then has to divide by
norms again, and `cosine_pairs` is a row-wise product and sum.

---

## 2. Term weighting

TF-IDF is used in two places: over the sentences of one article inside TextRank
(`summarize/extractive.py`), and over articles inside one language for the topic classifier
(`analysis/topics.py`). Both use sublinear term frequency.

$$
\mathrm{tf}(t, D) = 1 + \log f_{t,D} \quad (f_{t,D} > 0), \qquad
\mathrm{idf}(t) = \log \frac{1 + \lvert \mathcal{D} \rvert}{1 + n_t} + 1,
$$

where $f_{t,D}$ is the raw count of term $t$ in $D$ and $n_t$ the number of documents containing
it. The weight is the product of the two, and the vector is then $L^2$-normalised:

$$
x_t = \frac{\mathrm{tf}(t, D) \cdot \mathrm{idf}(t)}{\sqrt{\sum_{t' \in D} \left( \mathrm{tf}(t', D) \cdot \mathrm{idf}(t') \right)^2}} .
$$

Two properties of this choice show up in the results.

The logarithm bounds what repetition can buy. Raw counts would make a word used eleven times worth
eleven times a word used once; the log makes it worth $1 + \log 11 \approx 3.4$ times as much. News
writing names the subject of the story in nearly every sentence, so without the log the sentence
graph in section 3 would mostly rank whichever sentence repeats that name most often.

Fitting the vectorizer per article is what keeps the extractive summarizer language-independent.
Inside one article a word appearing in every sentence gets $\mathrm{idf} \approx 1$, and a word
appearing once gets the maximum, so function words are pushed down by the article's own statistics.
No stop-word list is needed for any of the four languages, and the same code runs over all of them
unchanged.

---

## 3. TextRank

The extractive summarizer builds a weighted undirected graph $G = (S, W)$ over the sentences of a
single article. Edge weights are TF-IDF cosines, with the diagonal zeroed and weak edges pruned:

$$
W_{ij} =
\begin{cases}
\cos(\mathbf{x}_i, \mathbf{x}_j) & i \neq j \text{ and } \cos(\mathbf{x}_i, \mathbf{x}_j) \geq \tau, \\
0 & \text{otherwise,}
\end{cases}
\qquad \tau = 0.05 .
$$

Pruning at $\tau$ removes edges between sentences that happen to share a couple of common words.
Left in, those edges carry enough total weight to flatten the ranking towards uniform.

### 3.1 The stationary distribution

Sentence importance is the PageRank score: the stationary distribution of a random walk that
follows an edge with probability $\alpha$ and jumps to a uniformly chosen sentence otherwise.
Writing $\mathbf{r}$ for the score vector and $d_j = \sum_k W_{jk}$ for the weighted degree,

$$
r_i = \frac{1 - \alpha}{n} + \alpha \sum_{j \,:\, W_{ij} > 0} \frac{W_{ij}}{d_j} \, r_j ,
\qquad \alpha = 0.85 .
$$

In matrix form, with $P_{ij} = W_{ij} / d_j$ the column-stochastic transition matrix and
$\mathbf{1}$ the all-ones vector,

$$
\mathbf{r} = \left( \alpha P + \frac{1 - \alpha}{n} \mathbf{1} \mathbf{1}^{\top} \right) \mathbf{r} = M \mathbf{r} .
$$

For any $\alpha < 1$ the matrix $M$ is column-stochastic with strictly positive entries.
Perron-Frobenius then gives it a unique eigenvalue of largest modulus, $\lambda_1 = 1$, with a
positive eigenvector. So $\mathbf{r}$ exists, is unique, and is reached from any starting vector.
The teleport term is doing that work: without it a graph with a disconnected component would admit
more than one stationary distribution.

### 3.2 Convergence and the fallback

Power iteration converges at the rate of the second eigenvalue, which the teleport term also
bounds:

$$
\lVert \mathbf{r}^{(k)} - \mathbf{r} \rVert \leq C \lvert \lambda_2 \rvert^{k}, \qquad \lvert \lambda_2 \rvert \leq \alpha = 0.85 .
$$

The error therefore shrinks by a factor of at least 0.85 per iteration, so three decimal places
take about 70 iterations, inside NetworkX's default cap of 100.

That bound assumes there are edges to walk along. Pruning at $\tau$ can remove all of them: an
article whose sentences share almost no vocabulary yields $W = 0$, every node becomes dangling,
and the iteration has nothing to converge to. `rank_sentences` catches the failure and falls back
to document order, which for news is the lead baseline and a reasonable ordering in its own right.

### 3.3 What the ranking optimises, and what it does not

The summary is the top $k$ sentences by $r_i$, returned in document order:

$$
\hat{S} = \left\{ s_i : i \in \operatorname{arg\,top}_k \, r \right\}, \qquad k = 3 .
$$

This is a centrality criterion. A sentence scores highly when it resembles other sentences that
themselves resemble many sentences, which is the recursion in section 3.1. Nothing in the objective
asks the three selected sentences to differ from each other, and nothing asks them to cover the
article.

That gap is where the behaviour in section 5 comes from. Since the score is computed on the graph,
the three top sentences tend to come from the article's densest region, and in a multi-topic
article the densest region is its largest single topic. The usual correction is maximal marginal
relevance, which subtracts a redundancy term from the score. It is not applied here.

---

## 4. Sentence embeddings and cosine similarity

The similarity score uses `paraphrase-multilingual-MiniLM-L12-v2`, a 12-layer transformer with
mean pooling over token states:

$$
\mathbf{u}(D) = \frac{\sum_{t=1}^{L} m_t \, \mathbf{h}_t}{\sum_{t=1}^{L} m_t},
\qquad \mathbf{u} \leftarrow \frac{\mathbf{u}}{\lVert \mathbf{u} \rVert} ,
$$

with $\mathbf{h}_t$ the final hidden state of token $t$ and $m_t$ the attention mask. The model was
distilled so that translations land near each other, which is what lets a Spanish summary be scored
against its Spanish source on the same scale as an English pair.

### 4.1 Chunking

The encoder truncates at $L = 128$ word pieces, roughly 400 characters. A news article is far
longer, so embedding one directly would represent it by its opening paragraph and nothing else.
`encode_documents` splits at sentence boundaries into chunks $C_1, \dots, C_m$ of about 900
characters and averages the chunk vectors:

$$
\mathbf{d} = \frac{\sum_{i=1}^{m} \mathbf{u}(C_i)}{\left\lVert \sum_{i=1}^{m} \mathbf{u}(C_i) \right\rVert} .
$$

The normalised mean is the point on the unit sphere with the largest summed cosine to the chunks,

$$
\mathbf{d} = \arg\max_{\lVert \mathbf{v} \rVert = 1} \sum_{i=1}^{m} \langle \mathbf{v}, \mathbf{u}(C_i) \rangle ,
$$

which represents a document well as long as its chunks are about one thing. Section 5 works out
what happens when they are not.

---

## 5. Why a correct summary can score low

A summary can be accurate and still score badly against its source. The reason is geometric rather
than linguistic, and it follows from the averaging in section 4.1. The same derivation predicts the
sign of every correlation in the drivers table.

### 5.1 The coverage identity

Take a document with $m$ unit chunk vectors whose pairwise cosine is $\rho$ throughout, and a
summary that faithfully covers $k \leq m$ of them, so its vector is the normalised mean over that
subset $A$ with $\lvert A \rvert = k$. The inner product of the two unnormalised sums is

$$
\left\langle \sum_{i \in A} \mathbf{u}_i, \; \sum_{j=1}^{m} \mathbf{u}_j \right\rangle = k \bigl( 1 + (m-1)\rho \bigr),
$$

since each of the $k$ vectors contributes 1 against itself and $\rho$ against the other $m - 1$.
The two norms are

$$
\left\lVert \sum_{i \in A} \mathbf{u}_i \right\rVert^2 = k \bigl( 1 + (k-1)\rho \bigr),
\qquad
\left\lVert \sum_{j=1}^{m} \mathbf{u}_j \right\rVert^2 = m \bigl( 1 + (m-1)\rho \bigr),
$$

and dividing gives

$$
\boxed{\;\cos(\mathbf{s}, \mathbf{d}) = \sqrt{\frac{k}{m}} \cdot \sqrt{\frac{1 + (m-1)\rho}{1 + (k-1)\rho}}\;}
$$

Two checks: at $k = m$ the expression is 1, and at $\rho = 1$ it is 1 for any $k$, both as expected.

### 5.2 Reading the identity

The two limits in $\rho$ bracket everything the similarity stage reports:

$$
\rho \to 1 \;\Longrightarrow\; \cos(\mathbf{s}, \mathbf{d}) \to 1,
\qquad
\rho \to 0 \;\Longrightarrow\; \cos(\mathbf{s}, \mathbf{d}) \to \sqrt{\frac{k}{m}} .
$$

A single-topic article has chunks that all say roughly the same thing, so $\rho$ sits near 1 and
any faithful summary scores near 1 no matter how much it dropped. A multi-topic article has
near-orthogonal chunks, $\rho$ near 0, and a perfect summary of $k$ of its $m$ topics is capped at
$\sqrt{k/m}$, which is 0.61 for three topics out of eight.

What the score measures, then, is coverage moderated by coherence. It is not a measure of whether
the summary is right. Three testable predictions follow, and the pipeline's own numbers confirm all
three:

| Prediction | Measured |
|---|---|
| Similarity rises with $k/m$, which `compression` proxies | $+0.566$ abstractive, $+0.426$ extractive |
| Similarity falls with $m$, which `source_chars` proxies | $-0.569$ abstractive, $-0.238$ extractive |
| The lowest scores are long multi-topic sources compressed hard | the shape of findings section 6.1 |

### 5.3 Empirical check

Computing $\rho$ and $m$ straight from the chunk embeddings of the 126 multi-chunk source articles,
and taking $k$ from the summary's own chunk count:

| Quantity | Value |
|---|---|
| Mean intra-document chunk cosine $\rho$ | 0.558 |
| Predicted mean cosine | 0.855 |
| Observed mean cosine | 0.818 |
| Mean absolute error | 0.066 |
| Correlation of predicted with observed | 0.47 |

The identity sits about 0.04 above the observed mean and tracks individual summaries only loosely.
Both gaps are what you would expect. A real summary is not exactly the mean of the chunks it
covers, and real chunk cosines are not all equal, so the constant-$\rho$ form is an idealisation.
Treat it as a model of the mechanism and not as a predictor of single scores. At $\rho = 0.558$ and
a median of three chunks it puts the whole distribution in the 0.8 band, which is where the
measured distribution is.

### 5.4 Consequence for the comparison

Nothing in the identity refers to how the $k$ chunks were produced. An extractive summary gets no
credit for having copied its sentences, and an abstractive one is not penalised for rewriting them,
because the encoder never compares strings. What separates the two methods is which $k$ they end up
covering. Section 3.3 shows TextRank concentrating on the densest region of the graph, while a
model compressing across the whole article spreads its $k$ over more of $m$. That is the mechanism
behind abstractive summaries clearing the threshold 76.2% of the time against 66.9%, while reusing
far less of the wording.

---

## 6. Lexical overlap

The second score is set containment of the summary's word types in the source's:

$$
\mathrm{overlap}(\hat{D}, D) = \frac{\lvert \mathcal{W}(\hat{D}) \cap \mathcal{W}(D) \rvert}{\lvert \mathcal{W}(\hat{D}) \rvert}, \qquad \mathcal{W}(\cdot) = \text{the set of casefolded word types},
$$

with the value defined as 0 for an empty summary. The measure is asymmetric: the denominator is the
summary alone, not the union, so this is precision against the source and not a Jaccard index. For
an extractive summary $\mathcal{W}(\hat{D}) \subseteq \mathcal{W}(D)$ holds by construction, and the
value is identically 1.

Both scores are reported because neither one identifies a good summary on its own:

| | high overlap | low overlap |
|---|---|---|
| **high cosine** | copied (extractive, median 1.00) | rewritten and faithful (abstractive, median 0.84) |
| **low cosine** | copied the wrong part | unrelated or hallucinated |

The top-left cell is the case that motivates the pair. A summary reproducing its source verbatim
scores 1.0 on overlap and high on cosine while having compressed no meaning at all. On cosine alone
it would rank as the best summary in the set.

---

## 7. The 0.8 threshold as a decision rule

Treating $\cos \geq 0.8$ as "keeps the main information" turns the score into a binary classifier.
Its two error modes come from section 5 and have nothing to do with summary quality.

False negatives are faithful summaries of long, low-$\rho$ articles, which section 5.1 caps below
the threshold whatever they contain. These are systematic and will not average out with more data.

False positives are near-verbatim copies, which pass on coverage alone.

Because the two modes separate cleanly on lexical overlap, the usable rule is the conjunction:

$$
\mathrm{accept}(\hat{D}) = \bigl[ \cos(\mathbf{s}, \mathbf{d}) \geq 0.8 \bigr] \wedge \bigl[ \mathrm{overlap}(\hat{D}, D) < 1 \bigr] .
$$

On abstractive output, where overlap sits well below 1, the first term carries the decision and
76.2% of summaries pass. On extractive output the second term is never satisfied. That is the right
answer: the method can demonstrate selection, but it cannot demonstrate comprehension.

---

## 8. K-means and the silhouette

Entity names are embedded with the same encoder and partitioned into $K = 6$ clusters minimising
the within-cluster sum of squares:

$$
\min_{C_1, \dots, C_K} \sum_{k=1}^{K} \sum_{\mathbf{u} \in C_k} \lVert \mathbf{u} - \boldsymbol{\mu}_k \rVert^2,
\qquad \boldsymbol{\mu}_k = \frac{1}{\lvert C_k \rvert} \sum_{\mathbf{u} \in C_k} \mathbf{u} .
$$

Lloyd's algorithm only reaches a local minimum, so the fit is repeated ten times from different
initialisations (`n_init=10`) under a fixed seed and the best objective kept.

On unit vectors this objective is equivalent to maximising within-cluster cosine, because

$$
\lVert \mathbf{u} - \mathbf{v} \rVert^2 = 2 - 2 \langle \mathbf{u}, \mathbf{v} \rangle .
$$

Euclidean K-means on normalised embeddings is therefore a spherical clustering, and the silhouette
is evaluated with the cosine metric to stay consistent with it:

$$
s(\mathbf{u}) = \frac{b(\mathbf{u}) - a(\mathbf{u})}{\max \{ a(\mathbf{u}), b(\mathbf{u}) \}} \in [-1, 1],
$$

where $a$ is the mean distance to the point's own cluster and $b$ the mean distance to the nearest
other cluster. The reported score is the mean over all points.

The measured 0.196 is about the magnitude to expect here, and not a sign that something went wrong.
An entity name is one to three tokens with no sentence around it, so the encoder places names in a
region of the sphere where neighbouring topics are not cleanly separated. The clusters overlap at
their boundaries and $b$ barely exceeds $a$. A silhouette near 0.2 says the partition is indicative
and should not be relied on as a decomposition, which is how findings section 3.3 presents it. The
substantive result is that six groups still resolve into world regions with no labels supplied, and
the silhouette is the caveat that goes with it.

---

## 9. Topic classification

### 9.1 The model

Multinomial logistic regression over TF-IDF unigrams and bigrams. For $C$ classes,

$$
P(y = c \mid \mathbf{x}) = \frac{\exp(\mathbf{w}_c^{\top} \mathbf{x} + b_c)}{\sum_{c'=1}^{C} \exp(\mathbf{w}_{c'}^{\top} \mathbf{x} + b_{c'})},
$$

fitted by minimising $L^2$-penalised cross-entropy with per-class weights:

$$
\mathcal{L}(W) = - \sum_{i=1}^{N} \omega_{y_i} \log P(y_i \mid \mathbf{x}_i) + \frac{1}{2C_{\text{reg}}} \lVert W \rVert_F^2 .
$$

The objective is convex in $W$, so the optimum is global. Reproducibility here rests on that, not
only on the fixed seed.

The class weights implement `class_weight="balanced"`:

$$
\omega_c = \frac{N}{C \cdot N_c},
$$

with $N_c$ the training count of class $c$. Politics and conflicts holds more than half the
articles, so an unweighted fit can lower its loss simply by predicting that class too often. The
weights give every class an equal share of the total loss and remove the incentive.

### 9.2 Why macro F1, and against what

Per class, read off the confusion matrix,

$$
F_{1,c} = \frac{2 P_c R_c}{P_c + R_c}, \qquad P_c = \frac{\mathrm{TP}_c}{\mathrm{TP}_c + \mathrm{FP}_c}, \quad R_c = \frac{\mathrm{TP}_c}{\mathrm{TP}_c + \mathrm{FN}_c},
$$

and the reported figure is the unweighted mean $F_1^{\text{macro}} = \frac{1}{C} \sum_c F_{1,c}$.

Leaving the mean unweighted is the whole point. A classifier that always predicts the majority
class reaches accuracy $\max_c N_c / N > 0.5$, while its macro F1 is

$$
F_1^{\text{macro}} = \frac{1}{C} \cdot \frac{2 \cdot p \cdot 1}{p + 1}, \qquad p = \frac{N_{\text{maj}}}{N},
$$

which for $C = 4$ and $p \approx 0.55$ works out at roughly 0.18, matching the measured baseline of
0.174 to 0.185. Accuracy alone would credit that classifier with having learned something. Macro F1
against the `DummyClassifier` is what shows the 0.87 to 0.90 result comes from the smaller
categories being learned and not from the class prior.

### 9.3 The single-label restriction

About a quarter of the corpus carries two or more topic labels. Training and scoring are restricted
to $\lvert \text{topics}(D) \rvert = 1$, because a single-label metric on a genuinely multi-label
article measures the label set instead of the model: an article that really is both political and
criminal is marked wrong for whichever of the two it predicts. The Crime and law against Politics
and conflicts confusion left in the matrix is the same overlap surviving inside the single-label
subset.

---

## 10. Rank correlation and the pooling trap

The drivers table reports Spearman's $\rho_s$, which is the Pearson correlation of the ranks:

$$
\rho_s = 1 - \frac{6 \sum_{i=1}^{n} d_i^2}{n(n^2 - 1)}, \qquad d_i = \operatorname{rank}(x_i) - \operatorname{rank}(y_i)
$$

for untied data, and the Pearson coefficient on ranks in general. Ranks are the right choice
because section 5.1 predicts a monotone but curved relationship, with $\cos$ going as
$\sqrt{k/m}$, and Pearson would understate it.

The correlations are computed inside each method and never pooled across both. Pooling two groups
whose feature distributions barely overlap gives a coefficient describing the gap between the
groups instead of anything within either one. This is Simpson's paradox, and in this data it is not
hypothetical:

| | pooled | extractive | abstractive |
|---|---|---|---|
| $\rho_s$(similarity, lexical overlap) | $\approx 0$ | undefined | $+0.162$ |

Extractive overlap is identically 1 by section 6, so inside that method the rank correlation is
undefined, a constant having no ranks. `drivers` returns `NaN` there rather than whatever number
scipy would produce from degenerate input. Pooled, the extractive block sits at overlap 1 with
lower similarity and the abstractive block at overlap 0.84 with higher similarity, so the
coefficient is dragged towards zero and the within-group sign flips. Reading "lexical reuse is
unrelated to similarity" off the pooled number would be reading an artefact of the pooling.

---

## 11. The cross-lingual recall proxy

The corpus ships no gold NER annotation, so recall is estimated from agreement between languages.
For a target language $\ell$, define

$$
\mathcal{N}_{\ell} = \bigl\{ (p, e) : e \in \mathrm{PERSON}_{\text{en}}(p) \;\wedge\; e \sqsubseteq \mathrm{text}_{\ell}(p) \bigr\},
$$

the pairs of an event $p$ and a person name $e$ that the English pipeline tagged and that also
occurs verbatim in the $\ell$-language article. The estimate is the share of those the $\ell$
pipeline tagged as well:

$$
\widehat{R}_{\ell} = \frac{1}{\lvert \mathcal{N}_{\ell} \rvert} \sum_{(p, e) \in \mathcal{N}_{\ell}} \mathbb{1}\bigl[ e \in \mathrm{ENT}_{\ell}(p) \bigr] .
$$

This gives $\widehat{R}_{\mathrm{es}} = 0.973$, $\widehat{R}_{\mathrm{fr}} = 0.972$ and
$\widehat{R}_{\mathrm{de}} = 0.911$.

### 11.1 Why the three restrictions are there

The verbatim condition $e \sqsubseteq \mathrm{text}_{\ell}(p)$ separates a model miss from a
difference in editorial coverage. Drop it, and a name the $\ell$ article never mentions in the
first place counts against the $\ell$ model.

Matching is at token level rather than substring. A substring test accepts "Ross" inside
"Crossing", which pushes every language above 0.96 and erases the effect being measured. Names
under four characters are dropped for the same reason.

Locations are excluded because place names get translated (London, Londres, Londra), so a verbatim
test would report every translated toponym as a miss. Person names stay largely invariant across
these four languages, and the whole proxy rests on that property.

### 11.2 What it does not measure

$\widehat{R}_{\ell}$ is not recall. Three biases are known, and all of them inflate the figure.

1. The reference is itself a model. $\mathrm{PERSON}_{\text{en}}$ is the English pipeline's output
   rather than ground truth, so English false positives enter $\mathcal{N}_{\ell}$ as items the
   other language is then penalised for missing.
2. The estimate conditions on English recall. A person both models miss never enters
   $\mathcal{N}_{\ell}$ at all.
3. The surname fallback accepts a match on the last token alone, which is more permissive than
   exact-mention recall.

All three apply the same way to all three target languages, since $\mathcal{N}_{\ell}$ is built
from the same English side each time. That is what makes the ordering de $<$ fr $\approx$ es
defensible even though the levels are not calibrated. The ordering also has independent support in
the proper-noun share from findings section 2: German tags 7.2% of its tokens PROPN against
English's 11.6%, on articles covering the same events.

---

## 12. Entity counting

### 12.1 Articles, not mentions

An entity's weight is the number of distinct articles mentioning it,

$$
\mathrm{articles}(e, \ell, c) = \bigl\lvert \{ p : (p, \ell, c, e) \in \mathcal{M} \} \bigr\rvert,
$$

over the mention table $\mathcal{M}$, rather than $\lvert \{ (p, \text{offset}) \} \rvert$. Mention
counts are dominated by article length: a long piece naming its subject eleven times contributes
eleven, which measures the writing and not the news. Counting distinct articles answers the
question actually being asked, which is how much of the coverage involved this entity.

The unit of counting is the pair $(\ell, p)$ and never $p$ on its own. A `pageid` identifies an
event, so the English page and its translations share one, and counting over $p$ alone quietly
merges up to four articles into one. The `articles` figure in the aggregate table is computed
inside a single language, where the problem cannot arise. Any table spanning languages has to
deduplicate on the pair first.

### 12.2 The label is derived, not a key

The coarse label is the modal label across the entity's mentions:

$$
\mathrm{label}(e, \ell, c) = \arg\max_{L} \; \bigl\lvert \{ m \in \mathcal{M} : \mathrm{entity}(m) = e, \; \mathrm{label}(m) = L \} \bigr\rvert .
$$

Grouping by $(\ell, c, e, L)$ instead would make the label part of the key, so one article in which
the pipeline tagged a name two ways would split into two rows and count as two articles. Deriving
the label removes that double count and still keeps the disagreement visible, since
`label_disagreements` reports it separately as a finding in its own right.

---

## 13. Readability

Flesch reading ease, applied with each language's own coefficients:

$$
\mathrm{FRE} = a - b \cdot \frac{\text{words}}{\text{sentences}} - c \cdot \frac{\text{syllables}}{\text{words}} ,
$$

with $(a, b, c) = (206.835,\, 1.015,\, 84.6)$ for English and language-specific values elsewhere,
plus the grade-level transform

$$
\mathrm{FKGL} = 0.39 \cdot \frac{\text{words}}{\text{sentences}} + 11.8 \cdot \frac{\text{syllables}}{\text{words}} - 15.59 .
$$

Both are linear in two surface statistics, so they measure sentence length and word length and
nothing else. A fluent summary and an ungrammatical string of the same lengths get the same score.
That is why the readability figures sit next to the deterministic style checks for truncation,
repeated sentences, unresolved opening pronouns, preambles and overlong sentences. The formulae
quantify reading effort, the checks catch defects, and neither one substitutes for the other.

Together they support the comparison the report makes. Extractive summaries inherit their source's
sentence lengths, reaching 27.2 words per sentence with 40% containing a sentence over 40 words.
Abstractive summaries sit at 18.9 words per sentence with none, because the prompt sets an explicit
word target. A sentence count alone does not achieve this, since a model asked for three sentences
returns three long ones, and that is why `build_prompt` carries a word target of $20 n$ as well.
