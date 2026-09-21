---
title: The quantum circuit for the chain, gate by gate
source: "Jozsa and Miyake, Matchgates and classical simulation of quantum circuits, Proceedings of the Royal Society A 464, 3089 (2008)"
arxiv: 0804.4050
topics: [quantum-circuit, gate-decomposition, matchgates, classical-simulation, trotterization, quantum-computing]
---

# The quantum circuit for the chain, gate by gate

Turning $\hat H = -J \sum_{i=1}^{L} \hat\sigma^z_i \hat\sigma^z_{i+1} - h \sum_{i=1}^{L} \hat\sigma^x_i$ into a circuit is elementary,
and worth doing explicitly because the circuit is short enough to hold in your
head — which is what makes this model the place to learn what a quantum circuit
for a physical Hamiltonian actually looks like.

## One Trotter layer

A single time step needs two layers of gates.

- **The field term.** $e^{i \theta \hat\sigma^x_i}$ with $\theta = h\,\Delta t$ is a single-qubit
  rotation about the $x$ axis, $R_x(-2\theta)$, applied to every qubit. All $L$ of
  them commute, so this is one layer of depth 1.
- **The coupling term.** $e^{i \phi \hat\sigma^z_i \hat\sigma^z_{i+1}}$ with $\phi = J\,\Delta t$ is the
  standard two-qubit rotation, decomposed as
  $\mathrm{CNOT}_{i,i+1} \cdot R_z(-2\phi)_{i+1} \cdot \mathrm{CNOT}_{i,i+1}$ — or
  applied directly on hardware whose native interaction is already $\hat\sigma^z\hat\sigma^z$.
  Neighbouring bonds overlap, so the layer splits into even and odd bonds: two
  sub-layers, each of constant depth, independent of chain length.

A Trotter step is therefore **constant depth** and the whole evolution is depth
proportional to the number of steps, with no dependence on $L$. That is unusual
and is a large part of why one-dimensional nearest-neighbour models are what
current hardware runs.

## Measuring the energy takes two settings

The expectation value $\langle \hat H \rangle$ needs $\langle \hat\sigma^z_i \hat\sigma^z_{i+1}\rangle$ for every
bond and $\langle \hat\sigma^x_i \rangle$ for every site. All the $\hat\sigma^z\hat\sigma^z$ terms commute and are
measured together in the computational basis; all the $X$ terms commute and are
measured together after a layer of Hadamards. Two measurement settings, whatever
the chain length. A quantum-chemistry Hamiltonian of comparable qubit count needs
hundreds of groups, so measurement cost is a solved problem here and an open one
there — which is why algorithmic studies that want to isolate *optimiser*
behaviour use this model.

## State preparation, before any evolution

Two ground states are free. At $h = 0$ the ground state is a product state along
$Z$ — the two ferromagnetic configurations and their superpositions, prepared with
no gates or one layer of them. At $J = 0$ it is a product state along $X$, one
Hadamard per qubit. Everything interesting is the interpolation between those two
product states, and the depth needed to reach it is largest at the critical point.

## The catch: these circuits are classically simulable

This is the fact that keeps the model honest as a benchmark. Under the
Jordan–Wigner map the chain is free fermions, and the corresponding circuits are
built from nearest-neighbour **matchgates**. Valiant (2002) and Terhal and
DiVincenzo (2002) showed that such circuits are classically simulable in
polynomial time, and Jozsa and Miyake (2008) gave the correspondence in the form
now usually quoted: matchgate circuits on a line are exactly the free-fermion
evolutions, and nothing about them is hard for a classical computer.

So a device running this circuit is not doing anything a laptop cannot check, at
any qubit count. Two consequences follow, and they point in opposite directions:

- **As a benchmark it is ideal.** The reference value exists at every size, so
  fidelity can be measured rather than estimated, and a wrong answer is
  attributable.
- **As a demonstration of advantage it is worthless.** A correct result on a
  matchgate circuit is evidence about the hardware, never about the hardness of the
  problem.

Adding anything that breaks the free-fermion structure — a longitudinal field
$\sum_i \hat\sigma^z_i$, a second dimension, disorder in the couplings, a non-integrable
kick — destroys the classical simulability and, with it, the reference value. The
choice between "checkable" and "hard" is made by that one term, and this model sits
on the checkable side deliberately.
