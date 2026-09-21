---
title: The variational quantum eigensolver, and why it is tested on this chain
source: "Tilly et al., The Variational Quantum Eigensolver: a review of methods and best practices, Physics Reports 986, 1 (2022)"
arxiv: 2111.05176
topics: [vqe, variational-quantum-eigensolver, quantum-computing, ansatz, benchmarking, near-term-hardware]
---

# The variational quantum eigensolver, and why it is tested on this chain

The variational quantum eigensolver (VQE) is the algorithm that made ground-state
estimation a near-term problem rather than a fault-tolerant one. Peruzzo et al.
introduced it on photonic hardware in 2014, and it remains the template for almost
every ground-state experiment on current devices.

## The algorithm in one paragraph

Prepare a parameterised state $|\psi(\theta)\rangle$ on the quantum processor,
measure the expectation value $\langle \psi(\theta)| \hat H |\psi(\theta)\rangle$, and
hand that number to a classical optimiser which proposes the next $\theta$. The
loop is hybrid: the quantum device only ever prepares states and reports
expectation values, and every decision is made classically. The output is
therefore a **variational upper bound** on the true ground-state energy, never a
value that can come out below it.

That last property is what makes VQE checkable at all. Rayleigh–Ritz guarantees
$E(\theta) \ge E_0$ for every $\theta$, so a reported energy below the exact
ground-state energy is not a lucky run — it is a bug, a mis-specified
Hamiltonian, or a measurement calibrated wrongly.

## Why the transverse-field Ising chain

Three features make this model the standard first target, and they are the same
three features that make it a *toy model of quantum computing* rather than only a
model of magnetism.

- **The Hamiltonian is native.** $\hat H = -J \sum_{i=1}^{L} \hat\sigma^z_i \hat\sigma^z_{i+1} - h \sum_{i=1}^{L} \hat\sigma^x_i$ is a
  sum of one- and two-qubit Pauli terms on a line. It needs no fermionic encoding
  and no basis transformation, so the measured operator on hardware is the
  operator on paper. Molecular Hamiltonians require Jordan–Wigner or
  Bravyi–Kitaev mappings first, and a mapping is another place to be wrong.
- **Measurement is cheap.** The energy needs exactly two commuting measurement
  settings — all qubits in the $Z$ basis for the coupling term, all in the $X$
  basis for the field term. Chemistry Hamiltonians need hundreds of measurement
  groups, so shot noise dominates the study of the optimiser rather than the
  other way round.
- **The answer is known.** The chain is exactly solvable, so the bound can be
  compared against the truth at any size a classical machine can reach. A VQE
  implementation that cannot reproduce this energy has been falsified before
  anybody spends device time on a molecule nobody can check.

## Ansatz choice is where the physics enters

A hardware-efficient ansatz — layers of single-qubit rotations and whatever
entangling gate the device does natively, as used by Kandala et al. (2017) — is
cheap to run and blind to the problem. It optimises well at small size and
degrades as the circuit deepens.

The alternative is to build the ansatz out of the Hamiltonian itself. Wecker,
Hastings and Troyer (2015) proposed alternating the two terms of $H$,
$e^{-i\beta \sum_i \hat\sigma^x_i} e^{-i\gamma \sum_i \hat\sigma^z_i \hat\sigma^z_{i+1}}$, repeated $p$ times —
the Hamiltonian variational ansatz, structurally identical to QAOA. On this chain
this is unusually well understood: Ho and Hsieh (2019) showed that the ground
state can be prepared exactly with a number of layers that grows linearly with
system size, and that the critical point is the hardest case.

## The failure mode the chain exposes cleanly

McClean et al. (2018) showed that for sufficiently deep and expressive random
circuits the gradient of the cost function vanishes exponentially with qubit
number — the **barren plateau**. The optimiser then has nothing to follow and the
run stalls at an energy far above the ground state while every individual gate
works correctly.

Because the exact energy of this chain is known, a barren plateau here is
diagnosable: the gap between the variational bound and the exact value is
measurable, so the stall is visible as a number rather than inferred from an
optimiser that stopped improving. On a Hamiltonian with no reference value, the
same run looks like convergence.

## What a VQE result on this model does and does not establish

It establishes that the device can prepare a correlated state of the required
symmetry and that the measurement and optimisation stack is calibrated. It does
not establish an advantage of any kind: the classical cost of solving this chain
is negligible, which is exactly why it is a benchmark. Cerezo et al. (2021)
survey the wider family of variational algorithms and are explicit that
near-term claims rest on problems where the classical answer is *not* available —
so a solvable chain is the place to validate the method, never the place to claim
it beats anything.
