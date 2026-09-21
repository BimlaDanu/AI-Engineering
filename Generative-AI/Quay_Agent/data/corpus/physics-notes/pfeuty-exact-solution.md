---
title: Exact solution of the transverse-field Ising chain
source: "Pfeuty, The one-dimensional Ising model with a transverse field, Annals of Physics 57, 79 (1970)"
arxiv: null
topics: [exact-solution, free-fermions, jordan-wigner, ground-state-energy]
---

# Exact solution of the transverse-field Ising chain

The one-dimensional transverse-field Ising model is one of the few interacting
quantum many-body systems that can be solved in closed form. With the convention

$$
\hat H = -J \sum_{i=1}^{L} \hat\sigma^z_i \hat\sigma^z_{i+1} - h \sum_{i=1}^{L} \hat\sigma^x_i
$$

the solution proceeds in three steps, none of which involves an approximation.

## Step 1 — Jordan–Wigner: the explicit transformation

Because the field couples to $\hat\sigma^x$ in this convention, the fermion
occupation is counted along $x$. Introduce spinless fermions $\hat c_i$ by

$$ \hat\sigma^x_i = 1 - 2\hat c^\dagger_i \hat c_i, \qquad
   \hat\sigma^z_i = -\Big(\prod_{j<i} \hat\sigma^x_j\Big)\big(\hat c_i + \hat c^\dagger_i\big) $$

The product is the **Jordan–Wigner string**. It is what repairs the statistics:
$\hat\sigma^z_i$ commutes on different sites, $\hat c_i$ must anticommute, and the
string of $\hat\sigma^x$ supplies the sign that converts one into the other.

The string is non-local, but in the Ising coupling it very nearly cancels. The
strings of sites $i$ and $i+1$ share every factor with $j < i$, and what survives
between the two operators collapses to

$$ \hat\sigma^z_i \hat\sigma^z_{i+1} =
   \big(\hat c^\dagger_i - \hat c_i\big)\big(\hat c^\dagger_{i+1} + \hat c_{i+1}\big) $$

**A non-local map with a local image: that is the whole trick.** Two strings
overlap in all but one factor, so a transformation that looks hopeless term by term
leaves a nearest-neighbour bilinear behind.

## Step 2 — the quadratic fermion Hamiltonian

Expanding the product and using $-\hat c_i \hat c^\dagger_{i+1} = \hat c^\dagger_{i+1}\hat c_i$:

$$ \hat H = -J \sum_i \big(
     \underbrace{\hat c^\dagger_i \hat c_{i+1} + \hat c^\dagger_{i+1} \hat c_i}_{\text{hopping}}
   + \underbrace{\hat c^\dagger_i \hat c^\dagger_{i+1} + \hat c_{i+1} \hat c_i}_{\text{pairing}}
   \big) \; + \; 2h \sum_i \hat c^\dagger_i \hat c_i \; - \; hL $$

Every term is a product of exactly two fermion operators. The chain is
**quadratic** — no interaction survives — and that, not any small parameter, is
where the solvability comes from. The pairing term is why particle number is not
conserved and a Bogoliubov rotation rather than a plain Fourier transform is
needed.

The one term that does not cancel is the bond closing the ring, $i = L$, whose
string wraps the whole chain and leaves the parity operator

$$ \hat P = \prod_{i=1}^{L} \hat\sigma^x_i = (-1)^{\hat N}, \qquad \hat N = \sum_i \hat c^\dagger_i \hat c_i $$

$\hat P$ commutes with $\hat H$, so the ring splits into two sectors and the
fermions obey a different boundary condition in each: **antiperiodic** in the
even-parity sector, periodic in the odd. This is bookkeeping, not a detail — it
fixes which momenta are allowed, and it is the usual source of small discrepancies
when a closed-form result is compared against a numerical one.

## Step 3 — Fourier transform and the Bogoliubov rotation

With $\hat c_j = L^{-1/2} \sum_k e^{ikj} \hat c_k$ the Hamiltonian becomes a sum of
independent $(k, -k)$ blocks:

$$ \hat H = \sum_{k>0} \Big[ \xi_k \big(\hat c^\dagger_k \hat c_k + \hat c^\dagger_{-k} \hat c_{-k} - 1\big)
   + \Delta_k \big(\hat c^\dagger_k \hat c^\dagger_{-k} + \hat c_{-k} \hat c_k\big)\Big] $$

$$ \xi_k = 2(h - J\cos k), \qquad \Delta_k = 2J\sin k $$

Each block is a two-level problem, and one rotation diagonalises it. Define
quasiparticles $\hat\gamma_k$ mixing $\hat c_k$ with $\hat c^\dagger_{-k}$ through
an angle

$$ \tan\theta_k = \frac{\Delta_k}{\xi_k} = \frac{J \sin k}{h - J\cos k} $$

which leaves $\hat H = \sum_k \epsilon_k \big(\hat\gamma^\dagger_k \hat\gamma_k - \tfrac{1}{2}\big)$ with

$$ \epsilon_k = \sqrt{\xi_k^2 + \Delta_k^2} = 2\sqrt{J^2 + h^2 - 2Jh\cos k} $$

so that $\cos\theta_k = \xi_k/\epsilon_k$ and $\sin\theta_k = \Delta_k/\epsilon_k$.
The interacting spin chain has become $L$ free quasiparticles, exactly.

## Step 4 — the ground state and its energy

The ground state is the quasiparticle vacuum, $\hat\gamma_k|0\rangle = 0$ for every
$k$, and its energy is minus one half the sum of the dispersion:

$$ E_0 = -\frac{1}{2}\sum_k \epsilon_k = -\sum_{k>0} \epsilon_k $$

the second form because $k$ and $-k$ contribute identically. In the even-parity
sector the antiperiodic condition quantises the momenta as

$$ k = \frac{(2n+1)\pi}{L}, \qquad n = 0, 1, \ldots, \tfrac{L}{2} - 1 \ \text{for the positive half} $$

For a ferromagnetic ring of even length the ground state lies in this sector, and
these momenta never land on $k = 0$ or $k = \pi$, so no zero mode needs separate
treatment. The energy is then a finite sum of $L/2$ terms evaluated to machine
precision — **no diagonalisation of a $2^L$ matrix required**.

The transverse magnetisation follows from the same angle without any new work,
since each mode contributes $\cos\theta_k$ to it:

$$ \langle \hat\sigma^x \rangle = \frac{1}{L}\sum_k \cos\theta_k
   = \frac{1}{L}\sum_k \frac{h - J\cos k}{\sqrt{J^2 + h^2 - 2Jh\cos k}} $$

Any ground-state quantity that is quadratic in the fermions is a closed-form
momentum sum of this kind. That is the practical content of "exactly solvable":
not that one number is known, but that a whole class of them is a sum rather than
an eigenproblem.

## Why this matters here

This gives an $O(L)$ route to a quantity that exact diagonalisation obtains in
$O(2^L)$, **sharing no algebra with it**. That is the property this project
depends on: two solvers that agree are two solvers that are almost certainly both
right, because a bug would have to be replicated in a linear-algebra eigensolver
and a closed-form momentum sum simultaneously.

The closed form applies to the uniform chain only. Add disorder, a longitudinal
field, or a second-neighbour coupling and the free-fermion structure is destroyed;
diagonalisation still works, but the independent check is gone, and any claim then
rests on one code path alone.

## The gap

The dispersion is minimised at $k = 0$, where $\epsilon_0 = 2|J - h|$. The gap
therefore closes **linearly** in the distance from $h = J$ and vanishes there.
A vanishing gap in the thermodynamic limit is the definition of a quantum critical
point, and its linear closing is what makes the dynamical critical exponent unity
for this model.

That statement is about the infinite chain, and a finite ring does not reproduce it
naively — for two reasons, both visible in the steps above. The allowed momenta
$k = (2n+1)\pi/L$ exclude $k = 0$, so the smallest single-quasiparticle energy is
$\epsilon_{\pi/L}$ rather than $\epsilon_0$; and because $\hat P$ is conserved, a
state reached from the ground state by adding one quasiparticle lies in the other
parity sector, so the lowest excitation *within* the sector costs a **pair**. The
finite-$L$ gap a diagonalisation reports is therefore neither $2|J-h|$ nor
$\epsilon_{\pi/L}$ in general, and at $h = J$ it is small but strictly positive,
shrinking as $L$ grows. A finite gap at the critical point is the expected
finite-size behaviour, not a numerical error.
