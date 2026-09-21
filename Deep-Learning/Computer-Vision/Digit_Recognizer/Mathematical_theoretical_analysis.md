# Mathematical and theoretical analysis

A companion to the README. It sets out the mathematics behind the design
decisions in `src/`, and says where each one stops being reliable.

Where PyTorch writes a formula differently from the usual textbook version, the
PyTorch form is the one given here, since that is what the code calls. Numbers
quoted as results come from `README.md` or from files in `reports/`; everything
else is derivation. Section 14 lists which function implements each result.

---

## Contents

1. [Problem and notation](#1-problem-and-notation)
2. [The model class](#2-the-model-class)
3. [The four objectives](#3-the-four-objectives)
4. [Gradients, and what the logged norms mean](#4-gradients-and-what-the-logged-norms-mean)
5. [Initialisation](#5-initialisation)
6. [Normalisation and regularisation](#6-normalisation-and-regularisation)
7. [Optimizers](#7-optimizers)
8. [Learning-rate schedules](#8-learning-rate-schedules)
9. [The data pipeline](#9-the-data-pipeline)
10. [The experimental design](#10-the-experimental-design)
11. [Metrics](#11-metrics)
12. [Ensembling and test-time augmentation](#12-ensembling-and-test-time-augmentation)
13. [What a convolution adds](#13-what-a-convolution-adds)
14. [Where each result lives in the code](#14-where-each-result-lives-in-the-code)

---

## 1. Problem and notation

A labelled image is a pair $(x, y)$, where $x \in \mathbb{R}^{784}$ holds the
pixel values and $y \in \{0, \dots, 9\}$ is the digit. Write $K = 10$ for the
number of classes and $B$ for the batch size.

The pairs come from a distribution $\mathcal{D}$ that nobody has access to. The
quantity worth minimising is the risk,

$$
R(\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}} \big[ \ell(f_\theta(x), y) \big],
$$

which cannot be computed. Training minimises the empirical risk on the training
split instead,

$$
\hat{R}_{\mathrm{train}}(\theta) = \frac{1}{n} \sum_{i=1}^{n} \ell(f_\theta(x_i), y_i).
$$

The three-way split exists to keep track of the difference between those two
quantities. Validation supplies an estimate of $R$ used for choosing between
models, and every choice made against it spends a little of its independence.
The test split is scored once, at the end, so that its estimate is still clean
when it is quoted.

Throughout, $f_\theta(x) \in \mathbb{R}^{K}$ is a vector of **logits**. The
letter $p$ always means probabilities obtained by applying a softmax to logits.

---

## 2. The model class

### 2.1 The network as a composition

With hidden widths $(d_1, \dots, d_L)$, $d_0 = 784$ and $d_{L+1} = K$, the
network computes

$$
h_0 = x, \qquad
h_l = \mathrm{drop}\big( \varphi ( \mathrm{BN} ( W_l h_{l-1} + b_l ) ) \big)
\quad (l = 1, \dots, L), \qquad
f_\theta(x) = W_{L+1} h_L + b_{L+1}.
$$

Batch normalisation and dropout appear only when the configuration asks for
them, and $\varphi$ is whichever activation the run was given. No activation
follows the last layer, so its output ranges over all of $\mathbb{R}^{K}$.

Setting $L = 0$ leaves $f_\theta(x) = W_1 x + b_1$, which is multinomial
logistic regression. The sweep reads its results against that floor, and it is
reachable from the command line by passing `--hidden-sizes` with no values, so
it needs no separate model.

### 2.2 Parameter count

Each linear layer holds $d_{l-1} d_l$ weights and $d_l$ biases, and each
batch-norm layer $2 d_l$ scale and shift parameters:

$$
|\theta| = \sum_{l=1}^{L+1} \big( d_{l-1} d_l + d_l \big) \; + \; 2 \sum_{l=1}^{L} d_l .
$$

For the reported model, hidden widths $(512, 512, 512, 512)$ with batch
normalisation, the terms are

$$
|\theta| = 401{,}920 + 262{,}656 + 262{,}656 + 262{,}656 + 5{,}130 + 4{,}096
= 1{,}199{,}114,
$$

which is the count in the README results table. Where those parameters sit is
the interesting part. The first layer alone accounts for 34% of them, against
22% for each of the three $512 \to 512$ layers behind it, because the input
arrives as 784 separate numbers with no structure attached to them.

### 2.3 What flattening costs

Take any permutation $\pi$ of the 784 pixel positions, with permutation matrix
$P_\pi$. Given a trained network $f_\theta$, define $f_{\theta'}$ by
$W_1' = W_1 P_\pi^{\top}$ and leave every other parameter alone. Then

$$
f_{\theta'}(P_\pi x) = f_\theta(x) \quad \text{for every } x .
$$

The model class is closed under permutations of the pixels: shuffle the image,
relabel the first layer, and the same function comes back. Nothing in the
architecture knows which pixels are neighbours. Adjacency and translation have
to be learned from the data if they are to be used at all.

Sections 12 and 13 both follow from this, and so does the presence of
augmentation in the sweep.

---

## 3. The four objectives

All four receive raw logits $z = f_\theta(x)$ and are built in `build_loss_fn`.

### 3.1 Softmax and cross-entropy

$$
p_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}, \qquad
\ell_{\mathrm{CE}}(z, y) = -\log p_y .
$$

Softmax ignores a constant added to every logit, $p(z + c\mathbf{1}) = p(z)$, so
only the differences between logits carry information. PyTorch uses that to
subtract $\max_k z_k$ before exponentiating, which keeps the exponentials in
range. A model that applied its own softmax and passed the result to the same
loss would be softmaxing twice, with worse numerics and flatter gradients.

Differentiating with respect to the logits gives

$$
\frac{\partial \ell_{\mathrm{CE}}}{\partial z} = p - e_y ,
$$

where $e_y$ is the one-hot target. Two things follow. The norm of this gradient
is at most $\sqrt{2}$, and it stays close to that value when the model is
confident and wrong, so no learning signal is lost on the hardest examples. And
it reaches zero only at $p = e_y$, which finite logits cannot attain, so the
objective keeps pushing the correct logit higher without limit. Label smoothing
exists to stop that.

### 3.2 Label smoothing

Replace the one-hot target with
$q = (1 - \varepsilon) e_y + \frac{\varepsilon}{K}\mathbf{1}$, giving
$\ell = -\sum_k q_k \log p_k$ and

$$
\frac{\partial \ell}{\partial z} = p - q .
$$

The stationary point can now be reached, at $p^\star = q$, and the probability
assigned to the true class settles at

$$
p^\star_y = 1 - \varepsilon + \frac{\varepsilon}{K} = 0.91
\quad \text{for the configured } \varepsilon = 0.1 .
$$

Past that point, extra confidence is penalised rather than rewarded. Label
smoothing is a calibration device written as a loss function. It is one level of
the `loss` factor in the sweep, so its effect on accuracy and its effect on the
calibration error of section 11 can be looked at separately.

### 3.3 `nll` is the same objective

`nll` applies `log_softmax` and then the negative log-likelihood:

$$
-\log p_y = -\log \mathrm{softmax}(z)_y = \ell_{\mathrm{CE}}(z, y).
$$

It is the same function as cross-entropy, composed in two steps instead of one.
Both are in the sweep on purpose. At the same seed they should return the same
numbers up to kernel nondeterminism, so a visible gap between them points at a
fault in the pipeline rather than at a property of the objective. It is the one
comparison in the study whose answer is known in advance.

### 3.4 Squared error on probabilities

$$
\ell_{\mathrm{MSE}}(z, y) = \frac{1}{K} \lVert p - e_y \rVert_2^2 .
$$

Differentiating through the softmax brings in its Jacobian
$J = \mathrm{diag}(p) - p p^{\top}$:

$$
\frac{\partial \ell_{\mathrm{MSE}}}{\partial z} = \frac{2}{K} J (p - e_y).
$$

Set that beside $p - e_y$ from cross-entropy. Two factors shrink it.

$J$ collapses under saturation. If the model is confident and wrong, then
$p \approx e_k$ for some $k \neq y$, so $J \approx 0$ and the gradient nearly
disappears on exactly the examples where the error is largest. Cross-entropy
still supplies a gradient of norm about $\sqrt{2}$ at the same point.

`F.mse_loss` also averages over the class axis as well as the batch. The extra
factor $1/K$ divides the effective step by ten, and the diagonal entries of $J$
are bounded by $\max_p p(1-p) = 1/4$ on top of that.

The prediction is about speed rather than about a different solution: `mse`
should need more epochs to reach a comparable accuracy. The `epochs_run` column
of the sweep is where that gets checked.

---

## 4. Gradients, and what the logged norms mean

### 4.1 Backpropagation as a product of Jacobians

With $z_l = W_l h_{l-1} + b_l$ and $D_l = \mathrm{diag}(\varphi'(z_l))$, the
backward pass carries

$$
\frac{\partial \mathcal{L}}{\partial h_{l-1}}
= W_l^{\top} D_l \frac{\partial \mathcal{L}}{\partial h_l},
\qquad
\frac{\partial \mathcal{L}}{\partial W_l}
= \Big( D_l \frac{\partial \mathcal{L}}{\partial h_l} \Big) h_{l-1}^{\top} .
$$

The signal arriving at layer $l$ is a product of $L - l$ factors of this kind. A
product of many numbers shrinks to nothing or grows without bound unless the
factors stay near 1, which is where vanishing and exploding gradients come from.

### 4.2 The condition, per activation

Taking norms on both sides,

$$
\Big\lVert \frac{\partial \mathcal{L}}{\partial h_{l-1}} \Big\rVert
\le \lVert W_l \rVert_2 \cdot \max_t \lvert \varphi'(t) \rvert \cdot
\Big\lVert \frac{\partial \mathcal{L}}{\partial h_l} \Big\rVert .
$$

| $\varphi$ | $\max \lvert \varphi' \rvert$ | consequence over $L$ layers |
|---|---|---|
| sigmoid | $1/4$ | contraction of at least $4^{-L}$ from the activation alone |
| tanh | $1$ | attained only at $t = 0$, and it saturates away from there |
| ReLU / leaky / GELU | $1$ | attained on the whole active half-line |

The `activation` factor in the sweep should reproduce that ordering. Rather than
take it on trust, every run writes the per-layer gradient norms into
`reports/history_<name>.csv`:

$$
\lVert \nabla_{W_l} \mathcal{L} \rVert_F \quad \text{for every linear layer, averaged over the batches of the epoch.}
$$

Reading those columns down the layers of a sigmoid run shows the decay.

### 4.3 A caveat on reading them

The norms cannot be compared across layers of different width. A
$1024 \times 784$ matrix has more entries than a $256 \times 512$ one, and the
Frobenius norm grows with the number of entries. The comparisons that carry
information are one layer across epochs, and the ratio between adjacent layers
within a single epoch.

---

## 5. Initialisation

Push the variance of the activations through one linear layer, assuming the
weights and inputs are independent with zero mean:

$$
\mathrm{Var}(z_l) = d_{l-1} \, \mathrm{Var}(W_l) \, \mathrm{Var}(h_{l-1}).
$$

Holding the forward variance constant needs
$\mathrm{Var}(W_l) = 1/d_{l-1}$; holding the backward variance constant needs
$1/d_l$. Xavier compromises between the two,

$$
\mathrm{Var}(W_l) = \frac{2}{d_{l-1} + d_l},
$$

and the code uses it for tanh and sigmoid. ReLU discards the negative half of
its input and halves the variance in doing so, which is recovered by doubling
the target:

$$
\mathrm{Var}(W_l) = \frac{2}{d_{l-1}},
$$

used for `relu`, `leaky_relu` and `gelu`. `_initialise_weights` chooses between
the two from the configured activation, so sweeping the activation factor cannot
leave a mismatched initialiser behind.

Two limitations. The Xavier call uses a gain of 1, whereas the usual
recommendation for tanh is $5/3$, so the scheme is applied as a default rather
than tuned per activation. And when batch normalisation is on, which is the
baseline, the pre-activations are renormalised anyway, so the scale of the
initial weights matters much less for the hidden layers. What remains of the
effect shows up in runs with `batch_norm` off, and in the final layer, which has
no normalisation after it.

---

## 6. Normalisation and regularisation

### 6.1 Batch normalisation

Per feature $j$, over a batch,

$$
\hat{z}_j = \frac{z_j - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}}, \qquad
\mathrm{BN}(z)_j = \gamma_j \hat{z}_j + \beta_j .
$$

At training time $\mu$ and $\sigma^2$ come from the batch; at evaluation time
they are replaced by running averages. Training and evaluation are therefore
different functions, and `model.eval()` switches between them. While training,
the prediction for one image depends on the other images that shared its batch.

A batch of one has no usable variance, so `BatchNorm1d` raises on it.
`make_loaders` drops a trailing batch of size one to keep that from surfacing
mid-run.

### 6.2 Scale invariance, and why weight decay still matters

Every hidden linear layer in this model is followed directly by batch
normalisation. For any $\alpha > 0$,

$$
\mathrm{BN}\big( (\alpha W_l) h_{l-1} \big) = \mathrm{BN}\big( W_l h_{l-1} \big),
$$

since the factor cancels between the numerator and the standard deviation. The
loss therefore does not depend on the scale of $W_l$ at all. Differentiating
that identity gives

$$
\nabla_{W} \mathcal{L}(\alpha W) = \frac{1}{\alpha} \nabla_{W} \mathcal{L}(W),
\qquad
\langle \nabla_W \mathcal{L}(W), W \rangle = 0 .
$$

The gradient is orthogonal to the weight vector, so a gradient step can only
lengthen it. The component of the step that actually changes the function
behaves like $\eta / \lVert W \rVert^2$, which means the effective learning rate
falls as training proceeds, with nobody scheduling it.

Weight decay pulls the other way. Shrinking $\lVert W \rVert$ raises the
effective learning rate, even though the loss cannot see the scale. So
`weight_decay` and `batch_norm` are not independent, although the screening
design varies them separately. The confirmation run is where their combination
gets tested.

### 6.3 Dropout

PyTorch uses inverted dropout. During training each unit survives with
probability $1 - p$ and the survivors are divided by $1 - p$,

$$
\tilde{h}_j = \frac{m_j}{1-p} h_j, \quad m_j \sim \mathrm{Bernoulli}(1-p),
\qquad \mathbb{E}[\tilde{h}_j] = h_j ,
$$

so evaluation is the identity and needs no rescaling. The mean is preserved but
the variance is not. Dropout adds multiplicative noise of variance $p/(1-p)$ per
unit, and the regularisation comes from that noise rather than from the
expectation. At $p = 0.5$ the injected variance equals the signal. That is the
top of the swept range, and the level where a model this size is expected to
underfit.

### 6.4 Weight decay, and why `adam` and `adamw` are separate levels

L2 regularisation adds $\frac{\lambda}{2}\lVert\theta\rVert^2$ to the loss and
$\lambda\theta$ to the gradient. Under SGD that is the same thing as shrinking
the weights by a fixed fraction at every step. Under Adam it is not, because
everything in the gradient gets divided by $\sqrt{\hat{v}}$:

$$
\text{Adam (L2)}: \quad
g_t \leftarrow g_t + \lambda \theta_t \; \text{ before the moments,} \quad
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

$$
\text{AdamW}: \quad
g_t \text{ left alone,} \quad
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \lambda \theta_t
$$

Adam therefore divides each parameter's decay by that parameter's own gradient
scale. Parameters with large, noisy gradients are decayed least, which inverts
what regularisation is supposed to do. AdamW applies the decay outside the
adaptive step. The sweep carries `adam` and `adamw` as two levels with the same
`weight_decay` setting, so this term is the only difference between them.

---

## 7. Optimizers

All five are `torch.optim` objects. Below, $g_t$ is the gradient at step $t$ and
$\lambda$ the weight decay.

**`sgd`**

$$
\theta_{t+1} = \theta_t - \eta (g_t + \lambda \theta_t)
$$

**`sgd_momentum`**

$$
b_t = \mu b_{t-1} + (g_t + \lambda\theta_t), \qquad
\theta_{t+1} = \theta_t - \eta b_t
$$

**`rmsprop`**

$$
v_t = \alpha v_{t-1} + (1-\alpha) g_t^2, \qquad
b_t = \mu b_{t-1} + \frac{g_t}{\sqrt{v_t} + \epsilon}, \qquad
\theta_{t+1} = \theta_t - \eta b_t
$$

**`adam`**

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}, \qquad
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

The bias correction matters for the first few dozen steps only. $m_t$ starts at
zero, so without the division by $1 - \beta_1^t$ the early updates would be
pulled towards zero by a factor that decays geometrically.

**Why a learning rate does not carry across optimizers.** Ignore $\epsilon$ for
a moment. Adam's per-coordinate step is

$$
\left| \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}} \right| \approx \eta
\quad \text{whenever the gradient sign is consistent,}
$$

because $\hat{m}$ and $\sqrt{\hat{v}}$ carry the same units and cancel. Adam's
step size is set by $\eta$ and is insensitive to the size of the gradient. SGD's
step is $\eta \lvert g \rvert$ and scales with it. A rate that suits one is one
or two orders of magnitude off for the other. That is why `optimizer` and
`learning_rate` are separate factors in the sweep even though what matters is
their combination, and section 10.2 covers what the design does about it.

---

## 8. Learning-rate schedules

With $T$ = `max_epochs` and $\eta_0$ the configured rate:

$$
\textbf{step:} \quad \eta_t = \eta_0 \cdot 0.1^{\lfloor t / s \rfloor},
\qquad s = \max(\lfloor T/3 \rfloor, 1)
$$

$$
\textbf{cosine:} \quad
\eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_0 - \eta_{\min})\Big(1 + \cos\frac{\pi t}{T}\Big),
\qquad \eta_{\min} = 0.01\,\eta_0
$$

$$
\textbf{onecycle:} \quad
\eta_0/25 \;\nearrow\; \eta_0 \text{ over the first } 30\% \text{ of steps},
\; \searrow\; \eta_0 / 25 \cdot 10^{-4} \text{ over the rest.}
$$

One-cycle steps once per batch rather than once per epoch. `build_scheduler`
returns a flag saying so, and `_train_one_epoch` receives the scheduler only in
that case.

### Schedules under early stopping

Each schedule is laid out over $T$ epochs, but early stopping can end a run at
any $t^\star \le T$, and such a run is scored part way through its own
annealing. Take the sweep defaults, $T = 50$, and a run that stops at epoch 20:

$$
\frac{\eta_{20}}{\eta_0} \approx \frac{1}{2}\Big(1 + \cos\frac{20\pi}{50}\Big) = 0.65 .
$$

It finishes training at roughly 65% of its initial rate rather than the 1% the
schedule was heading for. The `scheduler` rows of the summary therefore compare
schedules under early stopping, which is a fair question, but not the same
question as which schedule is best when allowed to finish. Runs whose numbers
get reported, the ensemble and the CNN comparison, set `patience` equal to
`max_epochs` so that the schedule completes. The rate each epoch trained at goes
into the history file, so any run can be checked afterwards.

---

## 9. The data pipeline

### 9.1 Standardisation

$$
\tilde{x} = \frac{x / 255 - \mu}{\sigma},
\qquad
\mu = \frac{1}{784\,n_{\mathrm{train}}}\sum_{i \in \mathrm{train}} \sum_{j=1}^{784} \frac{x_{ij}}{255},
$$

with $\sigma$ the matching standard deviation. Two details are deliberate.

The statistics are computed on the training split and stored on the `DataSplits`
object, so validation and test are transformed with the same two numbers.
Computing them over all rows would let the test data influence the model through
the scaling.

$\mu$ and $\sigma$ are scalars rather than per-pixel vectors. The border pixels
of MNIST are zero in every image, so a per-pixel $\sigma_j$ would be zero there
and the division undefined. A single global scalar has no such coordinate.

That second choice has a consequence for augmentation. After standardisation the
background sits at $(0 - \mu)/\sigma$, which is negative, so filling the corners
that rotate into view with a literal zero would draw a grey border around every
digit. `augment_batch` offsets by the batch minimum instead.

### 9.2 The split

Stratified 60 / 20 / 20 with a fixed seed, done in two calls. The second takes
the validation share of what the first left behind:

$$
\text{val share of the remainder} = \frac{f_{\mathrm{val}}}{f_{\mathrm{val}} + f_{\mathrm{test}}}
= \frac{0.2}{0.4} = 0.5 .
$$

Stratification keeps $P(y = k)$ the same in all three splits. That matters for
the per-class metrics of section 11: a class whose frequency drifted between
splits would show a change in recall that had nothing to do with the model.

Cleaning happens before the split. Rows are grouped by exact pixel equality
using `np.unique(..., axis=0)`, a lexicographic sort costing
$O(n \log n \cdot d)$. A group carrying more than one distinct label is dropped
in full rather than resolved, since its labels contradict each other and keeping
one copy would amount to picking an answer at random. Because the grouping runs
first, no image can end up in two splits, and a test checks that on the group
ids instead of relying on the order in which the operations happen to be
written.

### 9.3 The affine warp

`random_affine_matrices` draws, per image,

$$
\phi \sim \mathcal{U}(-\phi_{\max}, \phi_{\max}), \qquad
(t_x, t_y) \sim \mathcal{U}(-s, s)^2,
$$

and builds

$$
\Theta =
\begin{bmatrix}
\cos\phi & -\sin\phi & 2 t_x / W \\
\sin\phi & \cos\phi & 2 t_y / W
\end{bmatrix},
\qquad W = 28 .
$$

The factor $2/W$ converts pixels into the normalised coordinates `affine_grid`
expects, where the image spans $[-1, 1]$ and one pixel is $2/W$ wide. The code
writes this as a division by $W/2$.

One point here is easy to get backwards. `affine_grid` produces a sampling grid,
so $\Theta$ maps output coordinates to input coordinates and the image ends up
transformed by $\Theta^{-1}$. It makes no difference in this case, because
$\phi$ and $(t_x, t_y)$ are drawn from ranges symmetric about zero and the
family of warps is closed under inversion. With an asymmetric range it would
matter, which is the thing to remember before extending the augmentation.

Sampling is bilinear, so the warp is differentiable and the whole batch goes
through as one operation on the training device. The background level is
subtracted before sampling and added back afterwards, so the zero padding fills
with the true background.

---

## 10. The experimental design

### 10.1 Why not a grid

The ten factors carry $(5,5,5,5,5,5,4,4,4,2)$ levels. A full factorial grid is
their product:

$$
\prod_i L_i = 5^6 \cdot 4^3 \cdot 2 = 2{,}000{,}000 \text{ configurations},
$$

which at three seeds each is six million training runs, roughly 45,000 times the
sweep that actually runs. A one-factor-at-a-time design costs the sum instead:

$$
\sum_i L_i = 44 \text{ levels}, \qquad 44 \times 3 = 132 \text{ runs} .
$$

Sum against product is the argument for the design, and it is also where its
limitation comes from.

### 10.2 What the design can and cannot estimate

Write the response as main effects plus interactions:

$$
\mathbb{E}[\, \mathrm{acc} \mid \text{config} \,]
= \beta_0 + \sum_i \beta_i(\text{level}_i) + \sum_{i < j} \gamma_{ij}(\cdot, \cdot) + \dots
$$

Varying one factor at a time from a fixed baseline estimates each $\beta_i$ at
the baseline levels of everything else. None of the $\gamma_{ij}$ can be
estimated, because no configuration in the design moves two factors at once.
Taking the best level of each factor and combining them is valid when the
response is additively separable, which is to say when the $\gamma_{ij}$ are
negligible.

At least two pairs here are known not to satisfy that, both for reasons derived
above. Learning rate with optimizer, from section 7. Learning rate with batch
size, since a larger batch lowers the variance of the gradient estimate, so a
rate that was stable at batch 32 is conservative at batch 512.

The design handles this by testing the combination rather than assuming it.
`best_levels` reads the winning level of each factor out of the summary table,
`configuration_from` applies them together, and the confirmation run scores the
result on validation. If additivity fails badly, the combined run does worse
than its parts, and that is visible.

### 10.3 Reading a difference

Each level is run at three seeds. The standard error of a level mean is

$$
\mathrm{SE} = \frac{s}{\sqrt{3}} \approx 0.58 \, s ,
$$

and for the difference between two level means, each from $n = 3$ runs,

$$
\mathrm{SE}_{\mathrm{diff}} = s_p \sqrt{\tfrac{1}{3} + \tfrac{1}{3}} = 0.82 \, s_p .
$$

A two-sided $t$-test at 95% with 4 degrees of freedom needs $t > 2.776$, so the
gap between two levels has to exceed

$$
2.776 \times 0.82 \, s_p \approx 2.3 \, s_p
$$

before it can be called significant.

The rule stated in the README, that a gap smaller than the standard deviation is
not a result, is therefore a screen rather than a test. It is a reasonable way
to discard noise, and it is about 2.3 times more permissive than significance at
95%. Two further reasons to treat the sweep as a screen and the confirmation run
as the finding:

- Thirty-four levels are compared against the baseline. At a nominal 5% per
  comparison, one or two of them clearing the bar by chance is what one should
  expect.
- Three seeds give a poor estimate of $s$. With 2 degrees of freedom the sample
  standard deviation is itself noisy, so a level that happens to draw a small
  $s$ clears any fixed multiple of it too easily.

### 10.4 An efficiency note the arithmetic exposes

Every factor's level list contains that factor's own baseline level. The number
of distinct configurations in the sweep is therefore

$$
1 + \sum_i (L_i - 1) = 1 + 34 = 35,
$$

rather than 44. At three seeds that is 105 distinct runs out of the 132
executed. The remaining 27 retrain the baseline, once per factor, at seeds it
has already been trained at.

Those runs are not replication, since an identical configuration at an identical
seed should return an identical number. What they give is a determinism check:
if the baseline runs disagree, the fault is in the pipeline rather than in the
model. Dropping them would save about a fifth of the sweep. Giving each of them
a fresh seed would turn them into real replication. Both are listed under
further work.

---

## 11. Metrics

### 11.1 Accuracy and macro F1

Accuracy is the mean of $\mathbf{1}[\hat{y} = y]$. Per class,

$$
\mathrm{precision}_k = \frac{TP_k}{TP_k + FP_k}, \qquad
\mathrm{recall}_k = \frac{TP_k}{TP_k + FN_k}, \qquad
F1_k = \frac{2\,\mathrm{precision}_k\,\mathrm{recall}_k}{\mathrm{precision}_k + \mathrm{recall}_k},
$$

and macro F1 averages $F1_k$ over the ten digits without weighting by class
size. On a roughly balanced problem it tracks accuracy closely, and the reported
run gives 0.9856 accuracy against 0.9855 macro F1. Both are reported for the
case where they separate, which would mean the errors had concentrated in one or
two digits. The confusion matrix is the next place to look when that happens.

### 11.2 Calibration

Accuracy says nothing about whether a stated confidence of 0.9 can be believed.
Sort the predictions into $B = 10$ equal-width bins $B_b$ by their top
probability:

$$
\mathrm{conf}(B_b) = \frac{1}{|B_b|}\sum_{i \in B_b} \max_k p_{ik}, \qquad
\mathrm{acc}(B_b) = \frac{1}{|B_b|}\sum_{i \in B_b} \mathbf{1}[\hat{y}_i = y_i],
$$

$$
\mathrm{ECE} = \sum_{b=1}^{B} \frac{|B_b|}{n} \Big| \mathrm{acc}(B_b) - \mathrm{conf}(B_b) \Big| .
$$

A perfectly calibrated model scores zero. The reported run scores 0.0787, and
the sign is the part worth reading: every populated bin above $0.4$ is
*under*-confident, the top one by the most. It holds 7,682 of the 8,400 rows at
a mean confidence of 0.9209 and an accuracy of 0.9987. That is the label
smoothing of section 3.2 doing what it is defined to do -- the target is never
$1$, so the logit gap it rewards is bounded and the softmax never saturates.
The argmax is untouched, which is why accuracy rose while this number did.
Temperature scaling fitted on validation would rescale the logits and recover
the calibration without moving a single prediction.

Two properties of the estimator affect how that number should be read. Errors in
opposite directions inside the same bin cancel before the absolute value is
taken, so the figure is a lower bound on the miscalibration; narrower bins
expose more of it but hold fewer samples each, and numbers computed with
different $B$ are not comparable. And on an accurate classifier nearly all the
mass lands in $[0.9, 1.0]$, so that one bin dominates the sum while the rest
contribute very little. The full reliability table in `reports/test_report.md`
shows the distribution the single number hides.

Label smoothing, section 3.2, is the lever in the sweep that acts on calibration
directly. Temperature scaling fitted on the validation split is the usual next
step; it is listed under further work and has not been implemented.

### 11.3 How precise the test number is

The test split holds $n = 8{,}400$ rows, and accuracy is an average of Bernoulli
outcomes, so

$$
\mathrm{SE} = \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}
= \sqrt{\frac{0.9856 \times 0.0144}{8400}} = 0.0013,
$$

giving an approximate 95% interval of $[0.9831, 0.9881]$, about $\pm 0.25$
percentage points. Two models whose test accuracies differ by less than half a
point cannot be separated on this split. That is the scale to keep in mind when
reading the ensemble table and the CNN comparison, and the reason neither is
used to rank models by the fourth decimal.

When two models are scored on the same rows there is a better test available.
McNemar's test uses the counts of examples where one model is right and the
other wrong, which is information the interval above throws away. It is not
implemented here.

---

## 12. Ensembling and test-time augmentation

### 12.1 Why averaging probabilities cannot hurt the likelihood

For $M$ members with probabilities $p_m$, the ensemble predicts

$$
\bar{p}(y \mid x) = \frac{1}{M} \sum_{m=1}^{M} p_m(y \mid x) .
$$

Since $-\log$ is convex, Jensen's inequality gives, for each example,

$$
-\log \bar{p}(y \mid x) \;\le\; \frac{1}{M}\sum_{m=1}^{M} -\log p_m(y \mid x) .
$$

The ensemble's negative log-likelihood is never worse than the average of its
members', with equality only when the members agree exactly. The size of the gap
is the size of their disagreement, which is why diversity between members is
what makes ensembling work at all. Members that differ only in their random seed
are the weakest form of diversity available, so a large gain from seed
ensembling alone would be a surprise.

The bound comes with two limits. It applies to the likelihood, not to accuracy.
And it compares the ensemble with the average member rather than with the best
one, so an ensemble can be beaten by its strongest member. The report lists the
members individually next to the average for that reason.

### 12.2 Probabilities, not logits

Averaging logits and then applying a softmax computes a normalised geometric
mean of the member probabilities. Logits are unbounded, and softmax is invariant
to a shift but not to a scale, so a member that happens to train to larger
logits dominates that average. Averaging probabilities normalises each member
first, and every member then contributes the same total mass.
`average_probabilities` applies the softmax before it sums for this reason.

### 12.3 Test-time augmentation

TTA averages one model over $V$ views of the same image,

$$
\bar{p}(y \mid x) = \frac{1}{V}\sum_{v=1}^{V} p\big(y \mid t_v(x)\big), \qquad t_1 = \mathrm{id} .
$$

The same Jensen bound holds, and so does the same caveat in a sharper form: the
guarantee is relative to the average view, and warped views are harder than the
clean one. TTA helps when the model's sensitivity to small shifts and rotations
behaves like noise that averages out, and hurts when the warps move images away
from the data the model was trained on. The views therefore use the same shift
and rotation budget as training, and the first view is always the unmodified
image, so the average stays anchored to the input the model was asked about.

Section 2.3 explains why TTA has anything to do here. An MLP has no built-in
translation invariance, so averaging over translations supplies from outside
what the architecture lacks.

### 12.4 What the run measured

Three members at seeds 0, 1 and 2 scored 0.9848, 0.9856 and 0.9846 on the test
split; averaging their probabilities gave 0.9867, and TTA over four views of the
first member gave 0.9837.

Both numbers land where the sections above say they should, and neither is large
enough to lean on. The ensemble beat its best member by 0.11 percentage points,
which is *inside* the $\pm 0.25$ interval of section 11.3: the direction agrees
with the Jensen bound, the margin does not clear the noise of a single 8,400-row
split.

TTA lost 0.11 points against the same member, and the command-line run in
`reports/ensemble.md` shows why that is mild. Its members train without
augmentation, and there TTA costs 0.82 points, seven times as much. Section 12.3
predicts the difference: the average is only as good as the views it is taken
over, and a model never trained on shifted digits is being asked about images
that sit outside the distribution its weights were fitted to. The reported model
trains *with* the same shift and rotation budget TTA samples from, which puts
those views back in distribution and shrinks the cost to almost nothing. It does
not become a gain, because a model already invariant to the warps has little
sensitivity left for the averaging to cancel.

---

## 13. What a convolution adds

A convolutional layer replaces the dense product $W h$ with one kernel applied
at every position:

$$
(\mathcal{K} * h)_{c', u, v} = \sum_{c} \sum_{(a,b) \in \mathcal{W}} K_{c', c, a, b} \; h_{c,\, u+a,\, v+b} .
$$

Two properties follow that the model class of section 2.3 does not have.

**Equivariance.** For a translation $T_\delta$,
$\mathcal{K} * (T_\delta h) = T_\delta(\mathcal{K} * h)$, so a shifted digit
produces shifted features rather than unrelated ones. Max pooling then turns
that into approximate invariance across its window. An MLP has to learn the same
thing from examples, one shifted copy at a time, which is what the augmentation
factor buys it.

**Parameter cost.** A $3\times3$ kernel with $c$ input and $c'$ output channels
costs $9 c c' + c'$ parameters whatever the image size, against $d_{l-1}d_l$ for
a dense layer. The reference CNN here holds 421,834 parameters against the
MLP's 1,199,114, about a third, and most of the CNN's sit in its dense head
rather than in the convolutions.

The prediction is specific: better accuracy with fewer parameters, on the same
split, the same loop and the same epoch budget. It holds, and by a margin that
clears the noise for once: 0.9930 against the MLP's 0.9856, on 421,834
parameters against 1,199,114. That gap is 0.74 percentage points, three times
the interval of section 11.3, so unlike the ensembling numbers above it is a
difference rather than a coin flip. `make compare-cnn` writes both
rows into `reports/mlp_vs_cnn.md`. Read against the interval in section 11.3,
the comparison supports a sentence in the conclusions rather than a ranking by
decimal places.

---

## 14. Where each result lives in the code

| Section | Result | Implemented in |
|---|---|---|
| 2.1 | the layer stack | `MLP._build_layers` |
| 2.2 | parameter count | `count_parameters` |
| 3.1 to 3.4 | the four objectives | `build_loss_fn` |
| 4.1 to 4.3 | per-layer gradient norms | `_layer_gradient_norms`, `EpochRecord` |
| 5 | Kaiming and Xavier by activation | `MLP._initialise_weights` |
| 6.1 | batch statistics, batch of one | `MLP._build_layers`, `make_loaders` |
| 6.4 | decoupled decay | `build_optimizer` |
| 7 | the five update rules | `build_optimizer` |
| 8 | the schedules and the per-batch flag | `build_scheduler` |
| 9.1 | training-split statistics | `build_splits` |
| 9.2 | cleaning before splitting | `clean`, `_drop_duplicate_images` |
| 9.3 | the affine matrices | `random_affine_matrices`, `augment_batch` |
| 10.2 | winners combined and retested | `best_levels`, `configuration_from` |
| 10.3 | mean and spread per level | `summarise` |
| 11.1 | per-class precision, recall, F1 | `per_class_report` |
| 11.2 | reliability bins and ECE | `calibration_table`, `expected_calibration_error` |
| 12.1, 12.2 | probability averaging | `average_probabilities` |
| 12.3 | test-time augmentation | `tta_probabilities` |
| 13 | the reference CNN | `SmallCNN` |

Numbers quoted as results come from the README, for the reported run, and from
the files in `reports/`. Each of them can be reproduced with the commands the
README lists.
