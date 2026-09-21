---
title: Machine learning and the Ising model, in both directions
source: "Carrasquilla and Melko, Machine learning phases of matter, Nature Physics 13, 431 (2017)"
arxiv: 1605.01735
topics: [machine-learning, neural-networks, boltzmann-machines, applications, benchmarking]
---

# Machine learning and the Ising model, in both directions

The relationship runs both ways, and conflating the two directions is the usual
source of confusion in this area.

## Learning applied to the model

Carrasquilla and Melko (2017) trained ordinary supervised classifiers on spin
configurations of Ising models labelled only by which side of the transition they
came from. The networks separated ordered from disordered configurations and
located the critical coupling without being told what an order parameter is, which
is the interesting part: the quantity a physicist would have constructed by hand
was recovered from the raw configurations.

For the transverse-field chain the configurations come either from sampling the
wavefunction or from the classical two-dimensional model the quantum-to-classical
mapping produces. The chain is the standard test case here for the same reason it
is everywhere else — the answer being known is what makes it a test.

## The model as the learning architecture

The other direction is older, and it is a stronger statement than an analogy.

- **Hopfield networks.** An associative memory stores patterns in the couplings of
  an Ising energy function and recalls them by descending it. The energy is
  $E = -\sum_{ij} J_{ij} s_i s_j - \sum_i h_i s_i$ — the classical Ising model with
  learned couplings.
- **Boltzmann machines.** The same energy with a Boltzmann distribution over its
  configurations; training adjusts $J_{ij}$ so the model distribution matches the
  data. A restricted Boltzmann machine is the bipartite special case that made this
  trainable in practice.
- **Quantum Boltzmann machines.** Amin and colleagues (2018) added a transverse
  field to that energy, so the model being trained *is* a transverse-field Ising
  Hamiltonian and the distribution is the quantum Boltzmann distribution over its
  eigenstates. This project's Hamiltonian is not a toy analogy of that
  architecture; it is a member of the family, in the uniform one-dimensional case.

## Neural networks as wavefunctions

Carleo and Troyer (2017) went the other way again and used a restricted Boltzmann
machine as a variational ansatz for the ground state of a quantum spin model,
optimising its weights by stochastic reconfiguration. The transverse-field chain
is one of the benchmarks in that paper, and it is there because the exact answer
exists to compare against. This is now a research field of its own, neural-network
quantum states.

## What the shared mathematics does and does not buy

An energy function in common means the vocabulary transfers and the software
sometimes does. It does not mean the difficulty transfers: training a Boltzmann
machine and finding a ground state are both hard, and neither being expressible as
an Ising model makes either one easier. The chain earns its place in this
literature as a benchmark whose answer is checkable, not as evidence that a
quantum device will train a network faster.
