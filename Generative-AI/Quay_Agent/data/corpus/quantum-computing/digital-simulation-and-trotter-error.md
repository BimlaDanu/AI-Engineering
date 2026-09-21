---
title: Digital quantum simulation of the chain, and where Trotter error enters
source: "Smith, Kim, Pollmann and Knolle, Simulating quantum many-body dynamics on a current digital quantum computer, npj Quantum Information 5, 106 (2019)"
arxiv: 1906.06343
topics: [digital-quantum-simulation, trotterization, quench-dynamics, quantum-computing, error-mitigation, hardware]
---

# Digital quantum simulation of the chain, and where Trotter error enters

A gate-based quantum computer cannot apply $e^{-i\hat Ht}$ directly. It applies a
circuit, so continuous time evolution has to be cut into gates, and the standard
way of cutting it is the one Lloyd (1996) gave when he showed that a universal
quantum computer can simulate any local Hamiltonian efficiently.

## Trotterisation

Split the Hamiltonian into the two pieces that are individually easy —
$\hat H = \hat H_{zz} + \hat H_x$ — and alternate short evolutions under each:

$$ e^{-i\hat Ht} \approx \left( e^{-i\hat H_{zz}t/n}\, e^{-i\hat H_{x}t/n} \right)^n . $$

Each factor is a layer of two-qubit $\hat\sigma^z\hat\sigma^z$ rotations and a layer of single-qubit
$X$ rotations, which is about as shallow as a useful circuit gets. The two pieces
do not commute, so the approximation carries an error that shrinks with the number
of steps and grows with the evolution time. Childs et al. (2021) put tight bounds
on it in terms of commutators of the terms, which for a one-dimensional
nearest-neighbour model are unusually favourable.

The engineering tension is immediate and is the whole subject: more Trotter steps
reduce the algorithmic error and lengthen the circuit, which increases the
hardware error. The optimum is not at either extreme, and finding it requires
knowing the true answer — which, for this chain, is available.

## What is actually simulated: a quench

The experiment that gets run is rarely a ground state. It is a **quench**:
prepare a product state, evolve under $H$, and watch an observable relax. The
usual choice is all spins aligned along $Z$ evolving under a Hamiltonian with a
transverse field, and the observable is the magnetisation or a two-site
correlator.

This is the right experiment for hardware because it is hard for classical
computers in general and easy for this model in particular. Free fermions give
the exact dynamics of the chain at any size, so a device's output can be compared
with the truth point by point in time — and the time at which the two curves part
company is a direct, quantitative measure of how long the device stayed coherent.

## Error mitigation, and how the chain validates it

Present devices are too noisy to trust a deep circuit, so results are corrected
rather than merely reported. Zero-noise extrapolation, introduced by Temme,
Bravyi and Gambetta (2017), deliberately amplifies the noise, measures the
observable at several noise levels, and extrapolates back to the noiseless limit.

Mitigation is where a solvable model earns its keep for the second time. An
extrapolation is a fit, and a fit can be wrong in a way that looks smooth and
convincing. Running the same pipeline on this chain, where the exact curve is
known, is what distinguishes a mitigation scheme that recovers the physics from
one that recovers a plausible-looking artefact. Cai et al. (2023) survey the
family and are explicit that validation against classically solvable instances is
part of the method, not an optional check.

## The scale claims are built on Ising models

Kim et al. (2023) ran a Trotterised Ising evolution on 127 superconducting qubits
with error mitigation and argued that the result was beyond the reach of the
brute-force classical methods available at the time. The claim was contested
within months by improved tensor-network and Pauli-propagation simulations, which
is itself the point: because the model is a spin model on a fixed lattice with
few parameters, a claim about it is *checkable*, and it was checked. A comparable
claim about an unsolvable Hamiltonian could not have been settled either way.

The one-dimensional chain sits underneath all of this as the calibration case. It
is where the Trotter step is chosen, where the mitigation is tuned, and where a
circuit that produces confident nonsense is caught before it is pointed at
something nobody can verify.
