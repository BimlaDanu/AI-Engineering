---
title: How a circuit runs imaginary time, which is not a thing circuits can do
source: "McArdle et al., Variational ansatz-based quantum simulation of imaginary time evolution, npj Quantum Information 5, 75 (2019)"
arxiv: 1804.03023
topics: [varqite, imaginary-time, mclachlan, quantum-geometric-tensor, state-preparation, quantum-computing, ansatz]
---

# How a circuit runs imaginary time, which is not a thing circuits can do

Imaginary-time evolution is the most reliable way to find a ground state on a
classical computer, and it is the one method in this comparison that a quantum
circuit cannot execute directly. Understanding why, and what is done instead, is
the whole content of the method.

## What imaginary time does

Replace the time $t$ in the Schrödinger evolution $e^{-i\hat Ht}$ by $-i\tau$.
The oscillating phases become decaying exponentials:

$$ |\psi(\tau)\rangle \;=\; \frac{e^{-\hat H \tau}\,|\psi(0)\rangle}{\bigl\|\,e^{-\hat H \tau}\,|\psi(0)\rangle\,\bigr\|} $$

Every eigenstate is damped by $e^{-E_n \tau}$, so the lowest-energy component
decays slowest and eventually dominates. As $\tau$ grows the state converges to
the ground state, and the rate is set by the energy gap: the first excited state
dies off relative to the ground state as $e^{-\Delta\tau}$, where $\Delta$ is the
gap. Any starting state works provided it has some overlap with the answer.

This is why it is attractive. There is no optimiser, no landscape, no local minima
— the trajectory is a straight run downhill, defined by the Hamiltonian rather
than by a search.

## Why a circuit cannot do it

Quantum circuits are unitary: they preserve the length of the state vector.
$e^{-\hat H\tau}$ does not — it shrinks some components more than others, which is
exactly the mechanism that makes it useful. Non-unitary means "not a gate". The
normalisation in the formula above is not cosmetic bookkeeping; it is the part the
hardware cannot supply.

There are three families of workaround, and they trade different things.

## Route one: variational imaginary time (the one this project uses)

Keep the state inside a parameterised circuit, $|\psi(\theta)\rangle$, and ask
what change in $\theta$ best imitates one imaginary-time step. McLachlan's
variational principle answers it: minimise the distance between the imaginary-time
motion of the true state and the motion reachable by moving the parameters. The
result is a linear system solved once per step,

$$ \sum_j A_{ij}\, \dot\theta_j \;=\; -\,C_i $$

where $C_i = \partial \langle \hat H \rangle / \partial \theta_i$ is an ordinary
energy gradient, and $A_{ij}$ is the **quantum geometric tensor** — the metric on
the space of states the circuit can reach.

That tensor is the whole difference from gradient descent, and it is worth
stating in plain terms. A gradient tells you which direction lowers the energy in
*parameter* space. Two different parameters can move the state by very different
amounts, so a step that looks small in parameters may be a large move in state
space, and vice versa. $A_{ij}$ measures how much the state actually moves, and
dividing the gradient by it converts a parameter-space direction into a
state-space one. The trajectory then stops depending on how the circuit happened
to be parameterised.

The cost is measurement. A gradient has $O(m)$ entries for $m$ parameters; the
metric has $O(m^2)$, and each entry needs its own circuit — typically a
Hadamard-test-style circuit with an ancilla qubit. Quadratic scaling in the number
of parameters is the reason this method is not the default, and it is the number
that has to appear in any honest resource estimate. Stochastic and
subspace-truncated approximations of $A_{ij}$ exist and cut the constant, not the
scaling.

Two further practical points:

- $A_{ij}$ is often near-singular, because parameters can be redundant. It is
  solved with regularisation, and the regularisation strength is a real knob that
  changes the trajectory.
- Yuan et al. (2019) placed McLachlan's principle, the related Dirac–Frenkel
  principle and the time-dependent variational principle in one framework, and
  showed they differ once the state cannot follow the exact evolution — which is
  always. Which principle was used is a detail worth reporting.

## Route two: measure and post-select

Implement the non-unitary operator by embedding it in a larger unitary acting on
the system plus an ancilla, then measure the ancilla and keep only the runs where
it gives the right outcome. This is exact, and it is probabilistic: the success
probability falls as the step becomes more strongly damping, so the shot cost
grows with $\tau$. Probabilistic imaginary-time evolution takes this route
directly.

## Route three: rebuild the state from measurements (QITE)

Motta et al. (2020) observed that if correlations are short-ranged, the action of
$e^{-\hat H \delta\tau}$ on a small region can be reproduced by a *unitary* acting
on a slightly larger region, whose parameters are found by measuring the state and
solving a small linear system classically. No ancilla, no post-selection, and the
circuit stays unitary throughout. The cost is that the required region grows with
the correlation length — so near a critical point, where correlations reach across
the whole system, the method's advantage erodes exactly where the problem gets
hard.

## How this compares with VQE on the same chain

Both end up minimising the energy of a parameterised circuit, and on a good day
they reach the same state. The difference is what decides the next step.

| | VQE | Variational imaginary time |
| --- | --- | --- |
| Next step chosen by | a classical optimiser | the Hamiltonian, through McLachlan's principle |
| Measurements per step | energy plus $O(m)$ gradients | energy, $O(m)$ gradients, $O(m^2)$ metric entries |
| Can it stall? | yes — local minima, flat regions | it follows the flow, but the flow can leave the circuit's reach |
| Extra hardware | none | usually an ancilla qubit and controlled operations |

The honest summary is that imaginary time buys a principled trajectory and pays
for it in shots, and that the purchase is worth making precisely when the
optimiser is the thing that failed. On the transverse-field Ising chain at its
critical point — where the gap is smallest and the energy landscape is at its
flattest — that is a live possibility rather than a hypothetical, which is why
the chain is a fair test bed for the comparison.
