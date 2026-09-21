---
title: The low-lying spectrum, and the two parity sectors it comes from
source: "Lieb, Schultz and Mattis, Two soluble models of an antiferromagnetic chain, Annals of Physics 16, 407 (1961); Pfeuty, Annals of Physics 57, 79 (1970)"
arxiv: null
topics: [low-lying-spectrum, parity-sectors, jordan-wigner, energy-gap, symmetry-breaking, finite-size-effects]
---

# The low-lying spectrum, and the two parity sectors it comes from

"Plot the low-lying spectrum against the field" is one of the two or three most
useful figures this model has, and it is also the one where a careless free-fermion
calculation goes wrong in a way that looks right in one phase. The reason is a
single conserved quantity.

## The conserved parity, and why the ring splits in two

The Hamiltonian

$$
\hat H = -J \sum_{i=1}^{L} \hat\sigma^z_i \hat\sigma^z_{i+1} - h \sum_{i=1}^{L} \hat\sigma^x_i
$$

commutes with

$$
\hat P = \prod_{i=1}^{L} \hat\sigma^x_i , \qquad \hat P^2 = 1, \qquad [\hat H, \hat P] = 0,
$$

so every eigenstate carries a definite parity $P = \pm 1$ and the Hilbert space
splits into two sectors that never mix. Under the Jordan–Wigner transformation
$\hat P = (-1)^{\hat N}$ with $\hat N$ the fermion number, so parity is fermion
number modulo two — and the Jordan–Wigner string that closes the ring contributes
exactly $\hat P$. That is why the two sectors carry **different allowed momenta**:

$$
P = +1: \quad k = \frac{(2n+1)\pi}{L} \ \ (\text{antiperiodic}),
\qquad
P = -1: \quad k = \frac{2n\pi}{L} \ \ (\text{periodic}),
$$

for $n = 0, \dots, L-1$ over the full Brillouin zone. Both sets have $L$ members,
one mode per momentum: occupying $+k$ and $-k$ are two different excitations, each
costing $\epsilon(k)$.

## The energies, sector by sector

Within each sector the Hamiltonian is a sum of independent modes,

$$
\hat H = \sum_m \epsilon_m \left( \hat n_m - \tfrac{1}{2} \right),
\qquad
E = -\frac{1}{2}\sum_m \lvert \epsilon_m \rvert
    + \sum_{m\ \text{flipped}} \lvert \epsilon_m \rvert ,
$$

where the sum runs over that sector's momenta and the number of flips is
constrained to preserve the sector's parity. For a mode with pairing amplitude
$\Delta_k = 2J\sin k \neq 0$ the Bogoliubov rotation gives the familiar

$$
\epsilon(k) = \sqrt{\xi_k^2 + \Delta_k^2} = 2\sqrt{J^2 + h^2 - 2Jh\cos k},
\qquad \xi_k = 2(h - J\cos k).
$$

**The exception is where the whole subtlety lives.** The momenta $k = 0$ and
$k = \pi$ appear only in the *periodic* set, and there $\Delta_k = 2J\sin k = 0$:
nothing is paired, no rotation happens, and the mode keeps its bare, **signed**
energy $\xi_k$. At $k = 0$ that is

$$
\xi_0 = 2(h - J),
$$

which is **negative throughout the ordered phase** $h < J$. A negative mode energy
means the cheapest configuration has that mode *filled* — and filling it changes
the fermion parity at a cost that goes to zero as $h \to J$ from below.

## What that one sign produces: symmetry breaking

Everything a spectrum plot shows follows from it.

- **Below $h = J$** the lowest state of the odd sector is essentially degenerate
  with the lowest state of the even sector. These are the two symmetry-broken
  ground states — all spins up along $\sigma^z$, all spins down — and their
  splitting falls **exponentially** with $L$: at $L = 10$, $h = 0.2J$ it is under
  $10^{-5}$. A spectrum figure therefore shows $E_1 - E_0$ sitting on the axis
  throughout the ordered phase. That is the physics, not a numerical artefact, and
  a plot that shows a large gap there has the sector bookkeeping wrong.
- **Above $h = J$** $\xi_0$ turns positive, the doublet separates, and the cheapest
  excitation becomes a genuine quasiparticle. The gap grows roughly linearly with
  the field.
- **At $h = J$** the two lowest levels are as close as a finite ring ever brings
  them without touching, which is the finite-size version of a level crossing that
  becomes exact only as $L \to \infty$.

## Three different numbers, all called "the gap"

They differ by factors of two and by whole levels, and mixing them up is a standing
source of wrong convergence estimates.

$$
\underbrace{\epsilon(\pi/L)}_{\text{one quasiparticle}}
\qquad
\underbrace{2\,\epsilon(\pi/L)}_{\text{cheapest even excitation}}
\qquad
\underbrace{E_1 - E_0}_{\text{the ring's actual first gap}}
$$

The first is the smallest single-quasiparticle energy and is what the phase diagram
is drawn from. The second is what a *parity-conserving* method can actually reach:
imaginary-time evolution from $|+\rangle^{\otimes L}$, a hardware-efficient ansatz
and the variational imaginary-time baseline all conserve $\hat P$, so the states
they can mix into the ground state are the even ones, quasiparticles come in pairs,
and the energy that sets how fast such an evolution converges is twice the first.
Getting that wrong is a factor of two in every convergence-time estimate, always in
the optimistic direction. The third is the spacing a diagonalisation reports, and
below $h = J$ it is the exponentially small doublet splitting rather than either of
the others.

## The infinite chain, for comparison

Momentum is continuous when $L \to \infty$, so $k = 0$ is available and

$$
\Delta = \epsilon(0) = 2\lvert J - h \rvert
$$

is the cheapest excitation there is. It closes **linearly** in the distance from
$h = J$ and vanishes there, which is the definition of a quantum critical point and
what makes the dynamical critical exponent $z = 1$ for this model.

A finite ring has no gapless point at all: its smallest allowed momentum is $\pi/L$,
not zero, so $\epsilon(\pi/L) > 0$ at every field. Watching that number shrink as
$L$ grows is how a finite calculation sees a phase transition. Plotting the finite
gap against $2|J-h|$ on the same axes is the honest way to show both facts at once:
the curves agree away from the critical field and separate near it, and the
separation narrows with $L$.

## A practical warning about sparse eigensolvers

Because the ordered phase has an exponentially small doublet splitting, the lowest
levels of this model are nearly degenerate over a whole phase — and iterative sparse
eigensolvers (Lanczos, and `scipy.sparse.linalg.eigsh` with it) do not reliably
return every copy of a degenerate or near-degenerate level. Comparing a level-by-
level list from such a solver against a closed-form construction can therefore fail
while **both are right**: the closed form returns three copies of a level and the
iterative solver returns two, so every entry after that point is shifted.

Two ways out, and both are worth knowing: compare against a *dense* diagonalisation
where the matrix is small enough that every eigenvalue is returned with its
multiplicity, or compare the level *sets* rather than the ordered lists. The failure
is not in the physics and it is not in either solver; it is in the assumption that
"the $n$ lowest eigenvalues" is a well-defined list when levels coincide.
