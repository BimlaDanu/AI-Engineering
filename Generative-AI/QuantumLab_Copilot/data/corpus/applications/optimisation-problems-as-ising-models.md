---
title: How a business problem becomes an Ising model
source: "Lucas, Ising formulations of many NP problems, Frontiers in Physics 2, 5 (2014)"
arxiv: 1302.5843
topics: [combinatorial-optimisation, qubo, business-applications, max-cut, np-hard, applications]
---

# How a business problem becomes an Ising model

The reason a magnet model turns up in scheduling, routing and allocation is a
change of variables, not an analogy. A decision that is yes-or-no is a bit
$x_i \in \{0, 1\}$, and a bit is a spin:

$$ \hat\sigma^z_i = 1 - 2 x_i . $$

Substitute that into any cost function which is at most quadratic in the
decisions and what comes out is an Ising energy — couplings $J_{ij}$ from the
terms that pair two decisions, local fields $h^z_i$ from the terms that price one.
Minimising cost and finding the ground state become the same sentence. The same
object written with bits rather than spins is called QUBO, quadratic
unconstrained binary optimisation; the two are one formulation in two notations.

## Constraints are priced, not imposed

Business problems are mostly constraints — every delivery served once, every
shift covered, the budget spent exactly. A ground state cannot be told about a
constraint, so each one is added to the energy as a penalty that is zero when
satisfied and positive when not. Lucas (2014) gives explicit encodings this way
for max-cut, graph colouring, the travelling salesman, number partitioning,
knapsack and covering problems, with the penalty weights needed for the intended
solution to be the ground state.

The weights are where the practical difficulty lives. Too small and the cheapest
configuration cheats the constraint; too large and the penalties dominate the
landscape, flattening the differences between valid solutions that the solver was
supposed to resolve.

## The transverse field is not part of the problem

An optimisation cost function is built from $\hat\sigma^z$ operators only. They all commute,
so that Hamiltonian is classical: its ground state is one configuration of spins,
and nothing about it requires a quantum device.

The transverse field $-h \sum_i \hat\sigma^x_i$ is what a quantum solver adds. It does not
encode any part of the business problem. It is the term that makes configurations
mix — the driver in adiabatic annealing, the mixer in QAOA — and it is switched
off by the end of a run, leaving the classical cost function whose ground state
was wanted. This is the point where the vocabulary of this project and the
vocabulary of commercial optimisation meet: the Hamiltonian studied here,
$\hat H = -J \sum_i \hat\sigma^z_i \hat\sigma^z_{i+1} - h \sum_i \hat\sigma^x_i$, is the smallest complete example
of the pair — a $\hat\sigma^z\hat\sigma^z$ cost function and a transverse driver over it.

## Why the uniform chain, of all instances

Worth being blunt about, because the marketing rarely is: the one-dimensional
uniform chain is not a hard optimisation problem. Its cost function is minimised
by inspection, and no company needs a quantum computer to align a line of spins.

Its value is the opposite of hardness. It is the instance whose full spectrum is
known in closed form, so it is the one where a solver's answer can be *checked*
rather than merely compared with another heuristic's. Every claim about a device
solving a hard instance rests on the device first being trusted on an easy one.
