---
title: Neural Networks and Backpropagation
subject: dl
topic: neural-networks
difficulty: beginner
year: 2026
source: in-house study notes
---

# Neural networks and backpropagation

## The idea in one sentence

A neural network is a stack of simple functions — linear maps followed by nonlinearities —
whose parameters are adjusted so that the whole stack transforms inputs into useful outputs.

## Beginner view: a bucket brigade of tiny decisions

Imagine a bucket brigade where each person slightly transforms what they pass along. Each
"person" is a **layer**: it takes numbers in, multiplies them by learned **weights**, adds a
**bias**, and squashes the result through an **activation function**. Stacking layers lets the
network build complicated decisions out of simple ones — edges become shapes, shapes become
objects, objects become "this is a cat".

## The forward pass

For one layer with input $x$, weights $W$, bias $b$, and activation $\sigma$:

$$ h = \sigma(Wx + b) $$

Common activations:

- **ReLU**: $\mathrm{ReLU}(z) = \max(0, z)$ — the default hidden-layer choice; cheap and
  avoids saturating gradients for positive inputs.
- **Sigmoid**: $\sigma(z) = 1/(1 + e^{-z})$ — outputs in $(0,1)$; used for binary outputs,
  rarely in hidden layers because its gradient vanishes for large $|z|$.
- **Softmax**: turns a vector of scores into a probability distribution for classification.

Without nonlinear activations, a stack of linear layers collapses into a single linear map —
depth would buy nothing.

## Backpropagation: the learning algorithm

Training minimises a **loss** $L(\theta)$ over the parameters $\theta$ (all weights and
biases). Backpropagation computes the gradient $\nabla_\theta L$ efficiently by applying the
**chain rule** backwards through the network:

$$ \frac{\partial L}{\partial W^{(l)}} = \frac{\partial L}{\partial h^{(l)}} \cdot \frac{\partial h^{(l)}}{\partial W^{(l)}} $$

Each layer receives "how much did my output contribute to the error" from the layer above,
computes its own parameter gradients, and passes the blame further down. One forward pass plus
one backward pass costs roughly twice the forward computation — this efficiency is why deep
learning is feasible at all.

The gradients then feed an optimizer (SGD, Adam) that nudges every parameter downhill:

$$ \theta \leftarrow \theta - \eta \, \nabla_\theta L $$

where $\eta$ is the learning rate.

## Vanishing and exploding gradients

In deep stacks the chain rule multiplies many Jacobians together. If their norms are
consistently below 1 the gradient **vanishes** (early layers stop learning); above 1 it
**explodes** (training diverges). Standard remedies: ReLU-family activations, careful weight
initialisation (Xavier/He), **residual connections** (ResNet's $h = x + f(x)$ shortcut lets
gradients flow straight through), normalisation layers, and gradient clipping.

## Universal approximation, in practice

A single hidden layer can in theory approximate any continuous function, but *depth* gives an
exponentially more efficient representation for hierarchical structure — which is what images,
audio, and language have. That is the practical argument for "deep" learning.

## Key takeaways

- A network = linear maps + nonlinearities, composed.
- Backpropagation = chain rule, organised so gradients cost about one extra forward pass.
- Vanishing/exploding gradients are the central obstacle of depth; ReLU, good initialisation,
  and residual connections are the standard fixes.
