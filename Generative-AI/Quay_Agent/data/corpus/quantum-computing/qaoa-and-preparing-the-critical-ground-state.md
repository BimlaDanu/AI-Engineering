---
title: QAOA, discretised annealing, and preparing the critical ground state
source: "Farhi, Goldstone and Gutmann, A Quantum Approximate Optimization Algorithm, arXiv:1411.4028 (2014)"
arxiv: 1411.4028
topics: [qaoa, state-preparation, quantum-annealing, ansatz, quantum-computing, critical-point]
---

# QAOA, discretised annealing, and preparing the critical ground state

The quantum approximate optimisation algorithm alternates two Hamiltonians — a
problem Hamiltonian and a mixer — for $p$ rounds with variational angles, and
optimises the angles classically. For the Ising chain the problem Hamiltonian is
the $\hat\sigma^z\hat\sigma^z$ coupling and the mixer is the transverse field, so QAOA and the
transverse-field Ising model are not related subjects; they are the same
expression read twice.

## It is an annealing schedule with the clock removed

A Trotterised adiabatic sweep is a sequence of alternating $\hat\sigma^z\hat\sigma^z$ and $\hat\sigma^x$ evolutions
whose durations follow a fixed schedule. QAOA is the same circuit with the
schedule discarded and the durations promoted to free parameters. Anything
annealing can do with $p$ Trotter steps, QAOA can do at depth $p$ by choosing
those angles — and optimisation can only improve on them. This is why results
about the annealing gap carry over: the depth QAOA needs is governed by the same
critical slowing-down that sets the annealing runtime.

The difference is what happens at fixed small depth. An annealing schedule that is
too fast fails; a variational circuit that is too shallow returns the best state
reachable at that depth, which is a useful answer with a computable error rather
than a failure.

## Depth needed to cross the critical point

For this chain the state-preparation question has a sharp answer. Ho and Hsieh
(2019) showed that the alternating ansatz prepares the exact ground state with a
depth that grows linearly in the chain length, and that the resources needed peak
at the critical point $h = J$ — the same place the gap is smallest and the same
place a sweep has to slow down. Away from criticality, in either phase, a shallow
circuit suffices because the correlation length is short and the state is close to
a product state.

That correspondence is the reason this model is used to study circuit depth at
all: "how deep must the circuit be?" becomes "how far is the target from a product
state?", and for this chain the answer is known analytically as a function of
$h/J$.

## What the chain is used to measure

- **Ansatz quality.** The exact energy is available, so the gap between a
  variational bound and the truth is a number rather than an impression. Two
  ansätze can be compared on the same axis.
- **Optimiser behaviour.** The landscape of the two-parameter, single-round case
  can be drawn completely, which makes this the standard setting for studying how
  the classical optimiser gets stuck.
- **Transferability of angles.** Optimal angles found at one size often work at
  another for this model, which is studied here precisely because the reference
  values exist at every size.

## The limitation worth stating

QAOA is aimed at combinatorial optimisation, where the problem Hamiltonian is
disordered and frustrated. The uniform ferromagnetic chain is not an optimisation
problem — its minimum is "all spins aligned" and finding it needs no algorithm. It
is a *state-preparation* problem used as a testbed, and a depth requirement
established on a uniform chain says little about a frustrated instance. Treating
performance here as evidence about optimisation elsewhere is the error the
literature repeatedly warns against.
