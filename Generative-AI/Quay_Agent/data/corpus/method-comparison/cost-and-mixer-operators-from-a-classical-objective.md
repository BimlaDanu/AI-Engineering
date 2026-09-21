---
title: Turning a classical objective into a cost operator and a mixer
source: "Hadfield et al., From the quantum approximate optimization algorithm to a quantum alternating operator ansatz, Algorithms 12, 34 (2019)"
arxiv: 1709.03489
topics: [qaoa, cost-operator, mixer, ising-encoding, combinatorial-optimisation, quantum-computing, circuit-design]
---

# Turning a classical objective into a cost operator and a mixer

QAOA — the quantum approximate optimisation algorithm — does not take a business
problem as input. It takes two operators, and everything difficult happens before
the circuit exists. This note is about that translation step, because it is the
one a reader from outside physics actually has to perform, and the one where a
project most often discovers that its problem does not fit.

## The two operators

A QAOA circuit alternates two blocks, $p$ times:

$$ |\psi(\gamma,\beta)\rangle \;=\; \prod_{k=1}^{p} e^{-i\beta_k \hat H_M}\, e^{-i\gamma_k \hat H_C}\;|+\rangle^{\otimes n} $$

- $\hat H_C$, the **cost operator**, encodes the thing being minimised. It is
  diagonal: every bit string is one of its eigenstates, and the eigenvalue is that
  bit string's score. It does not move probability between candidate answers, it
  only stamps each one with a phase proportional to how bad it is.
- $\hat H_M$, the **mixer**, is the part that moves probability around. The
  standard choice is a transverse field, $\hat H_M = \sum_i \hat\sigma^x_i$, which
  flips single bits. Without it the circuit would never leave the state it started
  in and the phases would mean nothing.

The starting state $|+\rangle^{\otimes n}$ is the uniform superposition — every
candidate answer, equally weighted. It is the ground state of the mixer, which is
not a coincidence: it is the $p \to \infty$ adiabatic starting point.

## Step one: write the objective in $\pm 1$ variables

A classical objective is usually written over binary variables $x_i \in \{0,1\}$.
The substitution

$$ x_i \;=\; \frac{1 - s_i}{2}, \qquad s_i \in \{-1, +1\} $$

turns it into a polynomial in spin variables $s_i$. If the objective was quadratic
in $x$ — and most useful ones are — the result is quadratic in $s$:

$$ C(s) \;=\; \sum_{i<j} J_{ij}\, s_i s_j \;+\; \sum_i h_i\, s_i \;+\; \text{constant} $$

That is an Ising energy. The constant can be dropped; it shifts every score
equally and changes no ranking.

## Step two: promote the variables to operators

Replace each $s_i$ by the Pauli operator $\hat\sigma^z_i$, whose eigenvalues are
exactly $\pm 1$:

$$ \hat H_C \;=\; \sum_{i<j} J_{ij}\, \hat\sigma^z_i \hat\sigma^z_j \;+\; \sum_i h_i\, \hat\sigma^z_i $$

Nothing quantum has happened yet. $\hat H_C$ is diagonal, it commutes with itself,
and evaluating it on a bit string is the same arithmetic the classical objective
did. The quantum content arrives only when the mixer is applied, because
$\hat\sigma^x$ and $\hat\sigma^z$ do not commute.

## Step three: the circuit that applies it

Each quadratic term $e^{-i\gamma J_{ij} \hat\sigma^z_i \hat\sigma^z_j}$ is one
two-qubit gate sandwich: CNOT, a $z$-rotation by $2\gamma J_{ij}$, CNOT. Each
linear term is a single $z$-rotation. The mixer is one $x$-rotation per qubit.

So the two-qubit gate count of one QAOA layer equals the number of non-zero
$J_{ij}$ — the number of edges in the problem. This is the arithmetic that decides
feasibility, and it is arithmetic a software engineer can do without any physics:
count the edges, multiply by $p$, multiply by two for the CNOT pair, compare
against the device's coherence budget.

## Where constraints go, and why this is the hard part

Real objectives come with constraints — a vehicle visits each city once, a
portfolio spends its whole budget, a schedule assigns each shift exactly one
worker. There are two ways to handle them and they are not equivalent.

**Penalty terms.** Add $\lambda\,(\text{violation})^2$ to the objective. Simple,
and it keeps the standard transverse-field mixer. The costs are real: the penalty
weight $\lambda$ has to be large enough to dominate a genuine improvement but
small enough not to flatten the landscape, and a squared constraint over $k$
variables generates $\binom{k}{2}$ new quadratic terms — new edges, new gates,
often on qubit pairs the hardware does not connect. Lucas (2014) gives the
standard penalty encodings for the classic NP-hard problems and is the reference
most projects start from.

**Constraint-preserving mixers.** Choose $\hat H_M$ so that it can only move
between states that already satisfy the constraint — an XY mixer that swaps a pair
of bits conserves the number of ones, so a "choose exactly $k$" constraint holds
by construction. Hadfield et al. (2019) generalised QAOA in exactly this
direction, renaming it the *quantum alternating operator ansatz* because the
blocks need not be time evolutions of any Hamiltonian at all. The search space
shrinks to the feasible set, which is the point. The price is a harder initial
state and a deeper mixer.

The choice between the two is an engineering decision with a measurable
consequence, and it belongs in a feasibility study rather than in a footnote.

## Why this project's chain is the calibration case

For the transverse-field Ising chain the translation is already done: the cost
operator is the chain's own $\hat\sigma^z\hat\sigma^z$ coupling and the mixer is
its own transverse field. There is no encoding step to get wrong, the edges are
the bonds of a line, and the exact answer is known. That makes it the right place
to check that a QAOA implementation is correct before pointing it at a problem
where a wrong answer would be indistinguishable from a hard one.

It also makes the chain a poor advertisement for QAOA's value. A one-dimensional
uniform chain is solved instantly by classical methods; the interesting instances
are dense, disordered and frustrated. Farhi, Goldstone and Gutmann (2014)
introduced QAOA on MaxCut for that reason, and the honest framing of a chain
result is "the implementation is correct", never "the method is competitive".

## What is known about how well it works

- Farhi et al. (2014) gave a provable approximation ratio for MaxCut on
  3-regular graphs at $p = 1$, which classical algorithms then beat.
- Bravyi et al. (2020) showed that at fixed $p$ the locality of the circuit itself
  limits what QAOA can see, so obstructions exist that more shots cannot fix.
- Symmetry-informed parameter transfer means good angles found on one instance
  often work on another of the same family, which turns the classical optimisation
  from a per-instance cost into a one-off one — a large practical saving that is
  easy to miss when costing a pilot.
