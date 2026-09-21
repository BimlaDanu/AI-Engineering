---
title: Verifying a quantum computer with a model you can already solve
source: "Eisert et al., Quantum certification and benchmarking, Nature Reviews Physics 2, 382 (2020)"
arxiv: 1910.06343
topics: [benchmarking, certification, verification, quantum-computing, quantum-advantage]
---

# Verifying a quantum computer with a model you can already solve

The central awkwardness of quantum computing is that the regime worth reaching is
the regime nobody can check. If a classical computer can confirm the output, the
quantum device was not needed; if it cannot, the output is an assertion. Eisert et
al. (2020) frame certification and benchmarking around exactly this tension, and
solvable models are one of the few honest ways through it.

## Two kinds of benchmark, and only one of them is physics

**Device benchmarks** measure the machine: gate fidelities from randomised
benchmarking, coherence times, and sampling scores such as cross-entropy
benchmarking. They say nothing about whether an algorithm will produce the right
physics, because a device can pass them and still return a wrong energy through a
mis-specified Hamiltonian, a bad measurement grouping, or a stalled optimiser.

**Application benchmarks** measure the whole stack on a problem with a known
answer. That requires a Hamiltonian that is simultaneously interesting enough to
be a real test and solvable enough to have a reference value. The
transverse-field Ising chain is the canonical member of that small set, which is
why it appears in the validation section of papers whose subject is something
else entirely.

## What "known answer" has to mean

A reference value is only a reference if it was obtained independently of the
thing being tested. Three levels of independence are available here, and they are
worth distinguishing:

- **A closed form.** The free-fermion solution gives the energy of the periodic
  chain with $O(L)$ arithmetic at any size. Nothing iterative, nothing to
  converge.
- **An independent numerical method.** Exact diagonalisation of the same
  Hamiltonian shares no algebra with the closed form, so agreement between them
  is evidence rather than a tautology. It also covers the cases the closed form
  does not — open boundaries, odd length.
- **An exact limit.** At zero field the ground state is fully ordered and the
  energy is $-J$ per bond; with no coupling every spin aligns with the field.
  These are checkable by arithmetic with no solver involved at all.

A benchmark quoting one number from one method has assumed what it set out to
test.

## Falsification is the useful direction

The value of a solvable benchmark is asymmetric. Reproducing the exact energy does
not establish that a device will succeed on an unsolvable Hamiltonian —
extrapolation from an easy case to a hard one is precisely what nobody can
justify. Failing to reproduce it does establish that something in the stack is
wrong, cheaply and unambiguously.

So the right use of this model is as a filter rather than as a proof: it removes
implementations that do not work before device time is spent on problems where
being wrong is undetectable.

## Where advantage claims get settled

Claims of quantum advantage on Ising-type Hamiltonians have repeatedly been
narrowed or overturned by better classical methods — tensor networks, improved
Monte Carlo, Pauli-path simulation — rather than by other quantum devices. That
pattern is a feature of choosing a well-studied model: the classical frontier is
sharp enough that a claim can be contested with a calculation. The lesson carried
by the literature is not that the claims were dishonest but that a benchmark whose
classical difficulty is well characterised is the only kind whose difficulty can
be argued about at all.
