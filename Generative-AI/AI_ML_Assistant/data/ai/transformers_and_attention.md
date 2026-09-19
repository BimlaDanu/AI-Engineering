---
title: Transformers and Attention
subject: ai
topic: transformers
difficulty: intermediate
year: 2026
source: In-house study note (after Vaswani et al. 2017, "Attention Is All You Need")
---

# Transformers and the Attention Mechanism

The transformer architecture ("Attention Is All You Need", Vaswani et al., 2017) replaced
recurrence entirely with **self-attention**, enabling parallel training over whole sequences
and becoming the foundation of every modern LLM.

## Self-attention in one formula

Each token embedding is projected into three vectors: a **query** $Q$, a **key** $K$, and a
**value** $V$. Attention lets every token gather information from every other token, weighted
by relevance:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

Intuition: the query is "what am I looking for", keys are "what do I contain", and the
softmax of their dot products gives a probability distribution over which tokens to attend
to. The output is a relevance-weighted average of the values. The $\sqrt{d_k}$ scaling stops
dot products growing with dimension and saturating the softmax (which would kill gradients).

In the sentence "The animal didn't cross the street because *it* was too tired", the token
*it* learns high attention weight on *animal* — the mechanism resolves references by content,
not by position.

## Multi-head attention

Instead of one attention over the full dimension, the transformer runs $h$ smaller attention
"heads" in parallel and concatenates the results. Different heads specialise: some track
syntax, some track positions, some track coreference. This gives the model several
representation subspaces to relate tokens in simultaneously.

## The full block

A transformer layer stacks: multi-head self-attention → residual connection + layer norm →
position-wise feed-forward network (an MLP applied to each token independently) → another
residual + norm. Dozens of these layers are stacked. Because attention itself is
permutation-invariant, **positional encodings** (sinusoidal in the original paper; learned
or rotary — RoPE — in modern LLMs) inject token-order information.

## Encoder, decoder, and causal masking

The original transformer had an encoder (bidirectional attention — each token sees the whole
sentence; BERT descends from this) and a decoder (causal/masked attention — each token sees
only its predecessors, enabling autoregressive generation; GPT-style models are
decoder-only). **Cross-attention** in the decoder attends to encoder outputs, and reappears
in multimodal models.

## Why transformers won

- Full parallelism across sequence positions during training (RNNs are sequential).
- Constant path length between any two tokens — long-range dependencies are one attention
  hop away, versus many recurrent steps.
- Scaling laws: loss falls predictably as parameters, data, and compute grow, which
  justified the leap to GPT-scale models.

The cost is attention's $O(n^2)$ memory/compute in sequence length $n$ — motivating
FlashAttention (exact but IO-efficient), sliding-window and sparse attention, and
state-space alternatives such as Mamba.
