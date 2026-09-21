---
title: Quantum annealing, the adiabatic theorem, and the minimum gap
source: "Albash and Lidar, Adiabatic quantum computation, Reviews of Modern Physics 90, 015002 (2018)"
arxiv: 1611.04471
topics: [quantum-annealing, adiabatic-quantum-computing, energy-gap, kibble-zurek, quantum-computing, hardware]
---

# Quantum annealing, the adiabatic theorem, and the minimum gap

Quantum annealing computes by staying in the ground state of a Hamiltonian that
is slowly changed from one whose ground state is easy to prepare into one whose
ground state encodes the answer. The transverse-field Ising model is not an
analogy for this — it *is* the Hamiltonian the machines implement.

## The annealing schedule is this model

The standard schedule interpolates

$$ \hat H(s) = -A(s) \sum_i \hat\sigma^x_i - B(s) \sum_{i<j} J_{ij} \hat\sigma^z_i \hat\sigma^z_j, \qquad s = t/t_f $$

from a dominant transverse field at $s = 0$ to a dominant Ising coupling at
$s = 1$. Restricted to a chain with uniform nearest-neighbour coupling, that is
exactly the model this project solves, swept along a line through the phase
diagram. Kadowaki and Nishimori proposed the method in this form in 1998, and
Farhi et al. (2000) gave the equivalent adiabatic formulation.

The ground state at $s = 0$ is a product state — every spin along $X$ — and is
trivially prepared. The ground state at $s = 1$ is the classical solution. The
whole question is what happens in between.

## Why the gap is the cost

The adiabatic theorem says the evolution stays in the instantaneous ground state
provided it is slow compared with the inverse square of the energy gap $\Delta$
between the ground state and the first excited state. The runtime therefore
scales as

$$ t_f \sim \frac{1}{\Delta_{\min}^2} $$

where $\Delta_{\min}$ is the *smallest* gap encountered along the path. Every
other detail of the schedule is a constant factor beside this. So the interesting
quantity in an annealing study is not an energy at all: it is a gap, evaluated
along a sweep, and its minimum.

For the uniform chain the minimum sits at the quantum critical point $h = J$,
where the gap of the infinite system closes. At finite size it does not reach
zero — it closes like the inverse of the chain length, which is the finite-size
signature of a critical point with dynamical exponent $z = 1$. That distinction
is the whole difficulty of annealing: the obstruction is a property of the
thermodynamic limit that a finite machine feels as an anomalously small gap
rather than as an actual degeneracy.

## Sweeping too fast: Kibble–Zurek

Below the adiabatic speed limit the system cannot follow, and it emerges with
excitations. For this chain the outcome is known exactly: Dziarmaga (2005) and
Zurek, Dorner and Zoller (2005) showed that the density of kinks left behind
scales as an inverse square root of the sweep time, with an exponent fixed by the
critical exponents of the transition rather than by the details of the schedule.

This makes the chain a rare thing — an annealing problem where the *failure* is
predicted analytically as well as the success. A machine that anneals this chain
and reports the wrong kink scaling is not merely imprecise; it is not evolving
coherently.

## What hardware experiments use it for

King et al. (2022) annealed a ring of nearly two thousand superconducting flux
qubits configured as exactly this model on a D-Wave processor and recovered the
predicted Kibble–Zurek scaling, which was presented as evidence of coherent
rather than thermal dynamics. The 1D chain was chosen for the reason it is always
chosen: the theory is exact, so a disagreement is attributable to the device.

## The honest limitation

Annealing a uniform ferromagnetic chain solves nothing of value — the answer is
"all spins aligned", and no computation was needed to find it. Its role is
calibration and validation. The problems annealers are aimed at are disordered
and frustrated, where $\Delta_{\min}$ can shrink exponentially with size and
where nobody knows the gap in advance. Reading a runtime advantage off a solvable
chain would be a category error, and Albash and Lidar are careful to separate the
two questions.
