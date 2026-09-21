---
title: Exact diagonalisation in practice, and where it stops
source: "Weisse and Fehske, Exact Diagonalization Techniques, Lect. Notes Phys. 739, 529 (2008)"
arxiv: null
topics: [exact-diagonalisation, lanczos, sparse-methods, computational-limits]
---

# Exact diagonalisation in practice, and where it stops

Exact diagonalisation makes no approximation about the physics. Its only
approximation is the system size, and that one is severe.

## The wall

A spin-1/2 chain of $L$ sites has a Hilbert space of dimension $2^L$. Each added
site doubles the vector length and quadruples the memory needed for anything
storing a matrix densely. There is no algorithmic escape: the wall is the size of
the state itself, so a better solver postpones it by a site or two and no more.

The practical consequences follow directly:

- **Never build the Hamiltonian densely.** It is extremely sparse — for this model
  each basis state couples to only $L$ others via the transverse field, plus a
  diagonal entry. Sparse storage plus an iterative eigensolver is the only viable
  route beyond very small chains.
- **The ground state alone is usually enough**, and iterative methods give exactly
  that: Lanczos converges the extremal eigenvalues fastest, so the quantity
  physicists most want is the one the method is best at.

## Reproducibility of iterative solvers

Lanczos and its variants start from a random vector by default. Two runs on
identical input then differ in their last digits, and worse, occasionally differ in
which state they converge to when levels are nearly degenerate. Fixing the start
vector is what makes a numerical result a reproducible one — a necessity for any
result that will be compared against an independent calculation or stored and
re-checked later.

## Near-degeneracy in the ordered phase

Deep in the ordered phase the lowest two eigenvalues are exponentially close in
$L$: they are the even and odd combinations of the two ordered configurations, and
their splitting vanishes rapidly as the chain lengthens. An eigensolver asked for
two eigenvalues here can legitimately return the same state twice, or return the
pair in either order.

Any procedure that reads the gap as the difference of the two lowest eigenvalues
must therefore check that the two states it found are genuinely distinct — by
overlap, or by symmetry sector — rather than assuming the solver returned two
different states because it was asked for two.

## Cross-checks that cost nothing

Exact identities are better tests than stored reference numbers, because they hold
at every parameter value rather than at one and they cannot go stale:

- At $h = 0$ the Hamiltonian is classical and diagonal, so the ground-state energy
  is $-J$ times the number of bonds, exactly.
- Kramers–Wannier self-duality requires $E_0(J, h) = E_0(h, J)$.
- The energy must reconstruct from its own expectation values:
  $E = -J \, n_{\text{bonds}} \langle \hat\sigma^z \hat\sigma^z \rangle - h L \langle \hat\sigma^x \rangle$. This
  catches a mismatch between the Hamiltonian used to solve and the operators used
  to measure, which is a common and otherwise silent error.
- Any variational energy is bounded below by the true ground-state energy, so a
  trial state that comes out lower indicates a bug and not a discovery.
