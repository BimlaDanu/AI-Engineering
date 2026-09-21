---
title: Where the Ising chain is physically realised, platform by platform
source: "Monroe et al., Programmable quantum simulations of spin systems with trapped ions, Reviews of Modern Physics 93, 025001 (2021)"
arxiv: 1912.07845
topics: [hardware, quantum-simulation, rydberg-atoms, trapped-ions, superconducting-qubits, quantum-computing]
---

# Where the Ising chain is physically realised, platform by platform

The transverse-field Ising chain is unusual among model Hamiltonians in that
several quite different technologies implement it directly rather than encoding
it. That is why it recurs in hardware papers across platforms that otherwise share
no engineering.

## Trapped ions

A chain of ions in a linear trap carries a qubit in each ion's internal states.
Laser fields drive spin-dependent forces through the shared vibrational modes,
which produces an effective spin-spin coupling; a transverse field is a direct
drive on each ion. The resulting Hamiltonian is an Ising model whose coupling
decays as a power law with distance, tunable between nearly nearest-neighbour and
nearly all-to-all by choosing the detuning. Monroe et al. (2021) review the
programme in detail. Ion chains have the cleanest state preparation and readout of
any platform and the smallest qubit counts, which makes them the natural place to
study a model whose exact answer is known.

## Neutral atoms in Rydberg arrays

Atoms held in optical tweezers interact only when excited to a Rydberg state, and
the blockade that follows realises an Ising-type interaction between the ground
and Rydberg states with the driving laser playing the role of the transverse
field. Bernien et al. (2017) assembled a 51-atom chain and observed the
Ising-model quantum phase transition and its non-equilibrium dynamics directly.
The array geometry is programmable, so the same apparatus gives a chain, a ladder
or a two-dimensional lattice — but the chain is the case with an exact solution to
check against.

## Superconducting circuits

Two distinct machines are built from superconducting circuits, and they use this
model differently.

- **Gate-based processors** realise the chain digitally, as a circuit of $\hat\sigma^z\hat\sigma^z$ and
  $X$ rotations. The Hamiltonian is not implemented by the hardware at all; it is
  approximated by gates, with the Trotter error that implies.
- **Annealers** implement the Hamiltonian in analogue form, as flux qubits with
  tunable couplers, evolving continuously under a schedule. The model is the
  machine's native language rather than something compiled into it.

The same physical chain therefore appears once as a target and once as a
substrate, and results from the two are not directly comparable — a point easy to
lose when both are described as "simulating the Ising model".

## Why one dimension keeps being chosen

A one-dimensional chain is the geometry every platform can build first: ions form
a line in a trap, tweezers place atoms in a row, and a chain needs only
nearest-neighbour connectivity on a chip. It is also the geometry with an exact
solution. Those two facts together are why hardware papers open with the chain and
then move to lattices where the theory runs out — the chain establishes that the
apparatus does what its authors think it does.

## What differs between platforms, and why it matters for comparison

The Hamiltonian is nominally the same and the fine print is not. Coupling range
differs — nearest-neighbour on a chip, power-law in a trap. Boundary conditions
differ: a trapped-ion chain and a tweezer array have open ends and unequal
couplings near them, while an annealing ring can be closed. Both details change
the finite-size energies, so a hardware number and a textbook number disagree
unless the same boundary and the same coupling profile were used. This project
keeps boundary and length explicit for that reason: comparing an open chain
against a periodic closed-form result is a common and entirely avoidable error.
