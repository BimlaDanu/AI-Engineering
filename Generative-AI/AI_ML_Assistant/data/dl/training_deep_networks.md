---
title: Training Deep Networks in Practice
subject: dl
topic: training
difficulty: advanced
year: 2026
source: in-house study notes
---

# Training deep networks in practice

Getting a deep network to train well is mostly about controlling the *statistics of signals
and gradients* as they flow through many layers. The standard toolkit:

## Weight initialisation

Random initialisation must keep activation variance roughly constant across layers.
**Xavier/Glorot** initialisation scales weights by $\sqrt{1/n_{in}}$ (for tanh/sigmoid);
**He** initialisation uses $\sqrt{2/n_{in}}$ to compensate for ReLU zeroing half its inputs.
Bad initialisation shows up as instantly saturated activations or a loss that never moves.

## Normalisation layers

**Batch normalisation** standardises each channel over the mini-batch, then rescales with
learned parameters:

$$ \hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \qquad y = \gamma \hat{x} + \beta $$

It smooths the loss landscape, permits higher learning rates, and adds slight regularisation —
but couples examples within a batch, which is awkward for small batches and sequence models.
**Layer normalisation** normalises across features of each example independently, which is why
transformers use LayerNorm rather than BatchNorm. RMSNorm is a cheaper variant used in modern
LLMs (e.g. Llama).

## Optimizers and learning-rate schedules

- **SGD with momentum** accumulates a velocity: robust, often best final accuracy in vision.
- **Adam** keeps per-parameter estimates of the gradient's first moment $m_t$ and second
  moment $v_t$ and updates $\theta \leftarrow \theta - \eta\, \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$
  — the default for transformers and most new problems. **AdamW** decouples weight decay from
  the gradient update and is the standard for LLM training.
- **Schedules**: warmup (linearly increase $\eta$ for the first few hundred steps, essential
  for transformers) followed by cosine decay is the common modern recipe. Too high a learning
  rate diverges; too low wastes compute and can generalise worse.

## Regularisation

- **Dropout**: randomly zero a fraction $p$ of activations during training, forcing redundant
  representations; scale at inference. Less used in modern very-large models, still valuable
  in small/medium ones.
- **Weight decay** (L2): penalise large weights; interacts with Adam (use AdamW).
- **Early stopping**: monitor validation loss, stop when it rises.
- **Data augmentation** and simply **more data** beat all of the above when available.
- **Label smoothing**: soften one-hot targets (e.g. 0.9/0.1) to reduce overconfidence.

## Diagnosing a failing run

| Symptom | Likely cause | First fix |
|---|---|---|
| Loss is NaN | LR too high / exploding gradients | Lower LR, clip gradient norm |
| Loss flat from step 0 | Bad init, dead ReLUs, wrong labels | Check init, try LeakyReLU, verify data |
| Train loss ≪ validation loss | Overfitting | More augmentation/dropout/weight decay |
| Both losses high | Underfitting | Bigger model, train longer, raise LR |
| Loss oscillates wildly | LR too high or batch too small | Lower LR or increase batch size |

A classic sanity check: overfit a *single batch* to ~zero loss first. If the network cannot
even memorise 32 examples, the bug is in the code, not the hyperparameters.

## Key takeaways

- Initialisation + normalisation keep signal statistics healthy; that is what makes depth
  trainable.
- AdamW with warmup + cosine decay is the modern default recipe.
- Regularise in this order: more data → augmentation → weight decay/dropout → early stopping.
