---
title: Routing, the travelling salesman and scheduling as spin problems
source: "Neukart, Compostella, Seidel, von Dollen, Yarkoni and Parney, Traffic flow optimization using a quantum annealer, Frontiers in ICT 4, 29 (2017)"
arxiv: 1708.01625
topics: [travelling-salesman, routing, scheduling, np-hard, business-applications, combinatorial-optimisation, applications]
---

# Routing, the travelling salesman and scheduling as spin problems

Routing and scheduling are the operational problems most often brought to an
Ising solver, and they are the ones where the encoding cost is most visible. Both
are NP-hard, which is the reason they are interesting and also the reason no
mapping makes them easy: turning an NP-hard problem into a ground-state problem
moves the difficulty into finding the ground state, where it stays.

## The travelling salesman encoding

A tour of $n$ cities is written with one binary variable per city-per-position:
$x_{v,j}$ is one when city $v$ is visited at step $j$. The objective is the sum of
the distances between cities in consecutive positions, which is quadratic in those
variables, so it becomes the couplings of an Ising model. Three families of
penalty term enforce validity — each city appears exactly once, each position holds
exactly one city, and only legal edges are used.

Two consequences follow immediately, and they generalise to vehicle routing and to
job-shop scheduling, which use the same one-variable-per-assignment trick:

- **The variable count is quadratic.** $n$ cities need about $n^2$ spins, so a
  fifty-stop route is already a few thousand variables before any hardware
  embedding overhead. This, not the algorithm, is what limits problem size today.
- **Most of the energy is constraint.** The penalty terms are usually much larger
  than the differences between valid tours, so a solver spends its resolution on
  legality and has little left for optimality. Returning invalid tours is the
  characteristic failure mode.

## What a real deployment looked like

Neukart and colleagues (2017), working at Volkswagen, took taxi GPS traces in
Beijing and assigned each vehicle one of several precomputed routes so as to
minimise congestion — the objective being a quadratic penalty on shared road
segments. It is a genuine operational problem, it was mapped to QUBO, and part of
it was run on a quantum annealer with the remainder handled classically.

It is also a fair illustration of the state of the field: the instance was reduced
to fit the hardware, the comparison against classical solvers was not a defeat for
the classical side, and the enduring output was a clean formulation of traffic flow
as a cost function. That formulation is reusable regardless of what solves it.

## Why this project studies a chain instead

A routing instance has an irregular, dense coupling graph and no known answer. The
uniform chain has a regular graph and an exactly known spectrum. A solver, a
schedule or a device is debugged on the second and then applied to the first —
which is the whole reason a one-dimensional magnet appears in the same conversation
as delivery vans.
