---
title: RNNs, LSTMs, and Sequence Models
subject: dl
topic: sequence-models
difficulty: intermediate
year: 2026
source: in-house study notes
---

# RNNs, LSTMs, and sequence models

## The problem: inputs with order

Text, audio, and time series are sequences — the meaning depends on order and on context that
may be far away. Feed-forward networks see a fixed-size input with no notion of "before".

## Recurrent neural networks

An RNN processes a sequence one step at a time, carrying a **hidden state** $h_t$ that
summarises everything seen so far:

$$ h_t = \tanh(W_h h_{t-1} + W_x x_t + b) $$

The same weights are reused at every step (weight sharing in *time*, like CNNs share in
*space*). Training uses **backpropagation through time (BPTT)**: unroll the recurrence and
apply the chain rule across steps.

The catch: gradients flow through the same $W_h$ at every step, so over long sequences they
shrink or blow up geometrically — the **vanishing/exploding gradient** problem. Vanilla RNNs
struggle to connect words more than ~10–20 steps apart.

## LSTM: gated memory

The **Long Short-Term Memory** cell (Hochreiter & Schmidhuber, 1997) adds a separate **cell
state** $c_t$ — a conveyor belt that information can ride along almost unchanged — controlled
by three learned gates:

- **Forget gate** $f_t = \sigma(W_f [h_{t-1}, x_t])$ — what to erase from the cell state,
- **Input gate** $i_t$ — what new information to write,
- **Output gate** $o_t$ — what to expose as the hidden state.

$$ c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t, \qquad h_t = o_t \odot \tanh(c_t) $$

Because the cell state update is *additive* (not repeatedly multiplied by a weight matrix),
gradients survive across hundreds of steps. The **GRU** is a popular simplification with two
gates and no separate cell state — similar quality, fewer parameters.

## From recurrence to attention

Sequence-to-sequence (seq2seq) models paired an encoder RNN with a decoder RNN for machine
translation, but squeezing a whole sentence into one fixed vector was a bottleneck.
**Attention** (Bahdanau et al., 2014) let the decoder look back at *all* encoder states,
weighting the relevant ones — and worked so well that the **transformer** (2017) dropped the
recurrence entirely and kept only attention. Transformers process all positions in parallel
(faster training) and connect any two positions in one step (no vanishing path), which is why
they replaced RNNs for language.

## Where recurrence still matters

RNN/LSTM models remain relevant for streaming and low-latency inference, small on-device
models, and some time-series tasks. Modern **state-space models** (e.g. Mamba) revive the
recurrent idea with linear-time processing as a transformer alternative for long sequences.

## Key takeaways

- RNNs share weights across time; BPTT trains them but gradients decay over long spans.
- LSTMs/GRUs use gates and an additive cell state to preserve long-range information.
- Attention removed the fixed-vector bottleneck; transformers removed recurrence itself.
