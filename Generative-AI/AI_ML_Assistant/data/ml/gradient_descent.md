---
title: Gradient Descent
subject: ml
topic: optimization
difficulty: beginner
year: 2026
source: In-house study note
---

# Gradient Descent

Gradient descent is the workhorse algorithm for training machine-learning models. The idea:
a model has parameters (weights) $\theta$, and a **loss function** $L(\theta)$ measures how
wrong the model currently is. Training means finding the $\theta$ that makes the loss small.

Picture standing on a foggy hillside and wanting to reach the valley floor. You cannot see
the whole landscape, but you can feel the slope under your feet. The sensible move is a small
step downhill, then re-check the slope, then step again. Gradient descent does exactly this:

$$\theta_{t+1} = \theta_t - \eta \, \nabla_\theta L(\theta_t)$$

where $\nabla_\theta L$ is the gradient (the direction of steepest increase) and $\eta$ is
the **learning rate** — the step size.

## Learning rate trade-off

- Too small: training crawls; you take thousands of tiny steps.
- Too large: you overshoot the valley and the loss oscillates or diverges.
- Schedules (decay, warmup, cosine annealing) change $\eta$ during training to get the best
  of both.

## Variants

**Batch gradient descent** computes the gradient over the whole dataset per step — accurate
but slow. **Stochastic gradient descent (SGD)** uses one example per step — noisy but cheap,
and the noise can help escape shallow local minima. **Mini-batch SGD** (the practical
default) uses small batches (e.g. 32–512 examples), balancing gradient quality against
hardware efficiency.

**Momentum** accumulates an exponential moving average of past gradients so the update keeps
moving through flat regions and damps oscillations, like a heavy ball rolling downhill.
**Adam** combines momentum with a per-parameter adaptive learning rate scaled by the running
variance of each gradient coordinate; it is the default optimizer for most deep-learning
work, including transformer training.

## Backpropagation connection

For neural networks, the gradient of the loss with respect to every weight is computed by
**backpropagation** — the chain rule applied efficiently layer by layer, reusing intermediate
results from the forward pass. Gradient descent is the update rule; backpropagation is the
gradient computation that feeds it.

## Convergence intuition

For convex losses (linear/logistic regression), gradient descent with a suitable learning
rate reaches the global minimum. Deep networks are non-convex, yet SGD in practice finds
low-loss regions that generalise well — understanding why is still an active research area
(loss-landscape geometry, implicit regularisation of SGD noise).
