---
title: Portfolio optimisation and other financial problems as spin models
source: "Orus, Mugel and Lizaso, Quantum computing for finance: overview and prospects, Reviews in Physics 4, 100028 (2019)"
arxiv: 1807.03890
topics: [finance, portfolio-optimisation, business-applications, quantum-annealing, applications]
---

# Portfolio optimisation and other financial problems as spin models

Finance is the application area where the mapping to an Ising model is most
direct, because the textbook problem is already a quadratic form. Markowitz
portfolio selection asks for the holdings that maximise expected return minus a
risk aversion times the variance of the portfolio, subject to a budget.

- The **variance** is a quadratic form in the holdings with the covariance matrix
  in the middle. Covariances between assets become the couplings $J_{ij}$.
- The **expected returns** are linear in the holdings, so they become local fields
  $h^z_i$.
- The **budget** is a constraint, and constraints become penalty terms, as in any
  other encoding.

Orus, Mugel and Lizaso (2019) survey this and the neighbouring problems — arbitrage
cycles in a currency graph, optimal trading trajectories, credit scoring, option
pricing — and separate the ones that reduce to an Ising ground state from the ones
that need amplitude estimation or linear algebra instead. Only the first family is
what an annealer or a QAOA circuit can be pointed at.

## What makes the financial instances awkward

Two properties of real portfolios push against the hardware rather than the
mathematics.

**Holdings are not bits.** An allocation is a quantity, so each asset needs
several spins to express it, and the encoding cost multiplies the number of
qubits by the resolution demanded.

**Covariance matrices are dense.** Every asset is correlated with every other, so
the coupling graph is close to all-to-all. Hardware whose qubits are connected
only to their neighbours has to embed that graph across many physical qubits per
logical one, and the embedding overhead grows faster than the problem.

## Where the transverse field enters, and the honest limit

A quantum annealer starts in the ground state of the transverse field alone and
turns the field down until only the financial cost function is left. How slowly it
must be turned down is set by the smallest energy gap encountered along the way,
which is why the gap is the quantity that matters commercially and not only
theoretically.

The reviews are consistent about the current state: these are formulations and
pilot studies. No demonstration on a production-sized portfolio has beaten a good
classical solver, and several reported successes were later matched by classical
heuristics on ordinary hardware. What the exercise reliably produces is a clean
statement of the problem, which is worth something on its own — a portfolio
problem written as an explicit cost function is easier to solve by any method.
