---
title: The critical point at h = J and its universal exponents
source: "Sachdev, Quantum Phase Transitions, 2nd ed., Cambridge University Press (2011), ch. 4-5"
arxiv: null
topics: [quantum-criticality, critical-exponents, universality, conformal-field-theory]
---

# The critical point at h = J and its universal exponents

The transverse-field Ising chain has a quantum critical point at $h = J$
separating two phases that differ in symmetry rather than in energy scale.

## The two phases

For $h < J$ the coupling dominates. In the thermodynamic limit the ground state is
one of two symmetry-broken states with non-zero magnetisation along $Z$ — the
ordered, ferromagnetic phase. For $h > J$ the field dominates: the ground state is
a unique paramagnet, adiabatically connected to all spins pointing along $X$, with
no $Z$ magnetisation. The transition between them happens at zero temperature and
is driven by quantum fluctuations, not thermal ones.

## Finite-size caution: the symmetry is never broken

At any finite $L$ the Hamiltonian commutes with the global spin flip
$P = \prod_i \hat\sigma^x_i$, and its ground state is a $P$ eigenstate. Consequently
$\langle \hat\sigma^z_i \rangle = 0$ **identically** for every $L$, $J$ and $h$ — including
deep in the ordered phase. The finite chain's ground state is a symmetric
superposition (a "cat state") of the two ordered configurations, not one of them.

This is a trap rather than a subtlety. A calculation that measures order by
computing $\langle \hat\sigma^z_i \rangle$ returns zero everywhere and looks like a bug in the
solver. Order at finite $L$ must be read from the **correlator**
$\langle \hat\sigma^z_0 \hat\sigma^z_r \rangle$, which is insensitive to the overall flip and does
distinguish the phases: it saturates to a constant at large $r$ in the ordered
phase and decays exponentially in the disordered one.

## Self-duality fixes the location

The model is self-dual under the Kramers–Wannier transformation, which exchanges
$J$ and $h$. If a single critical point exists it must therefore sit on the
self-dual line $h = J$. This is a symmetry argument, not a numerical estimate,
which is what makes it usable as a test: the ground-state energy must satisfy
$E_0(J, h) = E_0(h, J)$ exactly, for every chain length.

## What conformal field theory predicts

At the critical point the correlation length diverges, the gap closes, and the
long-distance physics is described by a conformal field theory — specifically the
Ising CFT, with central charge $c = 1/2$. The universal predictions are:

- central charge $c = 1/2$
- correlation-function exponent $\eta = 1/4$, so $\langle \hat\sigma^z_0 \hat\sigma^z_r \rangle$ decays
  as a power law $r^{-\eta}$ at criticality rather than exponentially
- order-parameter exponent $\beta = 1/8$
- dynamical critical exponent $z = 1$, reflecting the linear closing of the gap

These are properties of the universality class, not of the model: any system in it
shares them, and they do not depend on $J$, on the lattice, or on microscopic
detail.

## Extracting an exponent on a short chain

Two finite-size corrections dominate and both must be handled, or a fit will
converge cleanly on the wrong answer.

**Use the chord distance, not the separation.** On a ring of circumference $L$ the
conformal distance between two sites separated by $r$ is
$(L/\pi)\sin(\pi r / L)$, not $r$. The two agree only for $r \ll L$, which on a
short chain is no separation at all. Fitting a power law against $r$ on a periodic
chain produces a systematically wrong exponent, and the fit quality gives no
warning.

**Discard the near and far points.** Small $r$ is contaminated by lattice-scale
detail that the continuum theory does not describe; $r$ near $L/2$ is contaminated
by the two images of the same pair meeting around the ring. Only the middle range
is asymptotic, and on a short chain that range is a handful of points.

The honest consequence is that a chain short enough to diagonalise exactly gives
an exponent estimate with real finite-size error. A result quoted from such a fit
belongs with its residual and its fit window, and a chain too short to resolve the
exponent at all warrants saying so rather than quoting a number.
