---
title: VQE, QAOA and variational imaginary time — three variational methods, three different jobs
source: "Cerezo et al., Variational quantum algorithms, Nature Reviews Physics 3, 625 (2021)"
arxiv: 2012.09265
topics: [vqe, qaoa, varqite, imaginary-time, variational, quantum-computing, ansatz, circuit-depth, benchmarking]
---

# VQE, QAOA and variational imaginary time — three variational methods, three different jobs

All three are **hybrid quantum-classical variational** algorithms. Each runs a short
parameterised circuit on a quantum processor, measures something, and hands the number
to a classical optimiser that proposes the next set of angles. Stated at that level of
generality they are indistinguishable, and that generality is why they get conflated.

They differ in three places that matter: **what they are trying to produce**, **how the
circuit is built**, and **what the classical half is actually computing**. This note sets
out those three axes, because a feasibility question about the transverse-field Ising
chain is usually really a question about which of these three is being proposed.

## The one-line distinction

- **VQE** minimises an energy. The target is the lowest eigenvalue of a Hamiltonian, and
  the answer is a *variational upper bound* on it.
- **QAOA** minimises a classical cost function. The target is a good assignment of
  discrete variables, and the answer is a *bit string* sampled from the final state.
- **Variational imaginary-time evolution** follows a trajectory. The target is not a
  minimum reached by search but a *state at a later imaginary time*, and the classical
  half solves a differential equation rather than descending a landscape.

The third is the one most often misfiled as "VQE with a different optimiser". It is not:
its update rule is derived, not chosen, and it has no cost landscape to get stuck in.

## Axis 1 — core target

**VQE: many-body eigenstate estimation.** Prepare $|\psi(\theta)\rangle$, measure
$E(\theta) = \langle \psi(\theta)| \hat H |\psi(\theta)\rangle$, minimise. Introduced by
Peruzzo et al. (2014) and given its general form by McClean et al. (2016). The
Rayleigh–Ritz principle guarantees $E(\theta) \ge E_0$ for every $\theta$, which is what
makes the output checkable: a reported energy *below* the exact ground-state energy is a
bug, never a good run.

**QAOA: a discrete assignment.** Farhi, Goldstone and Gutmann (2014) posed it for
combinatorial optimisation. The objective is a classical function $C(z)$ over bit strings
$z$, written as a diagonal operator $\hat H_C$ whose eigenvalues *are* the objective
values. The circuit's final state is measured, the bit string is read off, and $C$ is
evaluated classically on it. The expectation value is a means of steering; the deliverable
is a candidate solution.

**Imaginary time: a state at a specified time.** Applying $e^{-\tau \hat H}$ to almost any
starting state and normalising drives it towards the ground state as $\tau$ grows, because
higher eigenvalues are suppressed faster. That operator is *not unitary*, so no quantum
circuit implements it directly, and every method in this family is a way of approximating
its action. McArdle et al. (2019) and Yuan et al. (2019) give the variational version
(often VarQITE); Motta et al. (2020) give a non-variational route (QITE) that reconstructs
each step from measured correlations.

## Axis 2 — circuit style

**VQE: flexible.** The ansatz is a free design choice and the choice carries the physics.
A *hardware-efficient* ansatz is layers of single-qubit rotations plus whatever entangling
gate the device does natively — cheap, and blind to the problem (Kandala et al., 2017). A
*chemistry-inspired* ansatz such as unitary coupled cluster is built from the structure of
the molecule and is expensive. A *Hamiltonian variational ansatz* is built from the terms
of $\hat H$ itself (Wecker, Hastings and Troyer, 2015), which on this chain is structurally
identical to QAOA — the same gates and the same angles, pointed at a different objective.

**QAOA: standardised alternating blocks.** The structure is fixed rather than designed.
Starting from the uniform superposition, apply $p$ repetitions of

$$
e^{-i\beta_k \hat H_M}\, e^{-i\gamma_k \hat H_C}
$$

where $\hat H_C$ is the diagonal **cost** operator encoding the objective and $\hat H_M$ is
the **mixer** that moves amplitude between bit strings. In the original construction
$\hat H_M = \sum_i \hat\sigma^x_i$ — a transverse field, used as a tool rather than as
physics. Two angles per layer, $2p$ parameters in total, and no other freedom. Hadfield et
al. (2019) generalise the mixer to enforce constraints, which is where QAOA becomes a
family rather than one circuit.

**Imaginary time: sequential time-step updates.** There is no landscape to search. The
parameters follow a trajectory: at each small step, solve a linear system for
$\dot\theta$, take the step, repeat. Each step is a *derived* update rather than a
proposal from an optimiser, so the circuit shape need not be expressive enough to hold the
answer at a global minimum — only expressive enough to track the path.

## Axis 3 — what the classical half computes

This is the axis that most changes the engineering, and it is usually left out of
comparison tables.

**VQE and QAOA: gradient descent, or something like it.** The classical half evaluates a
scalar and proposes new angles. It needs no information about the quantum state beyond
that number, so the classical cost is negligible and the classical optimiser is
interchangeable.

**Imaginary time: a linear system with a metric.** McLachlan's variational principle
turns "follow $e^{-\tau \hat H}$ as closely as this circuit allows" into

$$
A(\theta)\, \dot\theta = -\, b(\theta),
$$

where $b$ holds energy derivatives and $A$ is the **quantum geometric tensor** — a matrix
of overlaps between derivatives of the state, measuring how far the state actually moves
when a parameter does. Its entries must be estimated on hardware, so the measurement cost
per step scales with the *square* of the parameter count rather than linearly. Stokes et
al. (2020) show the same matrix appears in quantum natural gradient, which is exactly this
update used as a preconditioner for ordinary VQE.

That extra cost buys something specific: an update that respects the geometry of the state
space. Two parameters that barely change the state get large steps and two that change it
sharply get small ones, which is why this family is far less troubled by flat regions than
plain gradient descent.

## Where each one struggles

**VQE and QAOA share the barren plateau.** McClean et al. (2018) showed that for
sufficiently deep, expressive, randomly initialised circuits the gradient vanishes
exponentially in qubit number; Cerezo et al. (2021, *Nature Communications*) showed that
a global cost function produces the same failure even at shallow depth. The optimiser then
has nothing to follow while every individual gate works correctly. On this chain the stall
is *diagnosable*, because the exact energy is known and the gap to it is a number — on a
Hamiltonian with no reference value the same run looks like convergence.

**QAOA additionally has a classical competitor problem.** Hastings (2019) gave local
classical algorithms matching or beating QAOA at low $p$ on several problem families, and
Bravyi et al. (2020) showed a symmetry obstruction that caps QAOA's approximation ratio
on some graphs at any fixed depth. Guerreschi and Matsuura (2019) estimated that QAOA
would need several hundred qubits before it beat a good classical solver on time to
solution. None of these are settled results about the whole algorithm, and all of them are
reasons a business case for QAOA must be argued against a specific classical baseline
rather than against "classical computing".

**Imaginary time pays in measurements.** The geometric tensor is the cost: quadratic in
parameter count per time step, and every step needs it. Its practical bottleneck is shot
budget rather than trainability, which is a different problem with different fixes.

## Why this chain is the right place to compare all three

The transverse-field Ising chain
$\hat H = -J \sum_i \hat\sigma^z_i \hat\sigma^z_{i+1} - h \sum_i \hat\sigma^x_i$
is the only common target of all three:

- Its Hamiltonian is a sum of one- and two-qubit Pauli terms on a line, so **VQE**
  measures it in two commuting settings and needs no fermionic encoding.
- Its coupling term is already a diagonal $\hat\sigma^z\hat\sigma^z$ cost operator and its
  field term is already the standard mixer, so a **QAOA** circuit for it is the natural
  ansatz rather than a translation.
- It is exactly solvable, so **any** of the three can be scored against the truth instead
  of against each other.

Ho and Hsieh (2019) used exactly this to show that a QAOA-form circuit prepares the
chain's ground state exactly with a layer count growing linearly in system size, and that
the critical point $h = J$ is the hardest case. That result is the reason a comparison run
on this model is informative rather than decorative: the answer for one of the three is
known analytically, so the others can be held against it.
