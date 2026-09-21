---
title: The transfer matrix and the quantum-to-classical mapping
source: "Suzuki, Relationship among Exactly Soluble Models of Critical Phenomena, Progress of Theoretical Physics 56, 1454 (1976)"
arxiv: null
topics: [transfer-matrix, quantum-to-classical-mapping, suzuki-trotter, imaginary-time, exact-solution, quantum-criticality]
---

# The transfer matrix and the quantum-to-classical mapping

The transverse-field Ising chain is the standard example of a correspondence that
runs through all of statistical and quantum physics: a quantum system in $d$
dimensions is a classical system in $d+1$. Understanding it on this model is how
most people understand it at all, which is another sense in which the chain is a
toy model — the toy is the *mapping*, not only the magnet.

## The classical transfer matrix

For the classical Ising chain in one dimension, the partition function factorises
into a product of identical $2 \times 2$ matrices, one per bond:

$$ Z = \operatorname{Tr} T^N, \qquad T_{ss'} = e^{K s s' + \frac{H}{2}(s + s')} $$

with $K = J/k_BT$. The free energy per site in the thermodynamic limit is set by
the largest eigenvalue of $T$ alone, and the correlation length by the ratio of the
two eigenvalues. That is the whole solution of the 1D classical chain: a
two-by-two eigenvalue problem, with no phase transition at any non-zero
temperature because the largest eigenvalue never becomes degenerate.

The transfer matrix is a linear operator that advances the system one step along
the lattice. That is the observation the mapping is built on.

## Imaginary time as an extra dimension

A quantum system's Boltzmann operator $e^{-\beta H}$ has exactly the same
structure if $\beta$ is read as a length. Slice it into $M$ pieces,

$$ e^{-\beta H} = \left( e^{-\Delta\tau H} \right)^{M}, \qquad \Delta\tau = \beta/M, $$

and split each factor into its commuting parts. Each slice is then a transfer
matrix acting along a new direction — imaginary time. Suzuki (1976) worked this
out for exactly this model: the 1D transverse-field Ising chain becomes an
**anisotropic 2D classical Ising model**, with the original chain as one axis and
the imaginary-time slices as the other. The classical coupling along the new
direction is fixed by the transverse field through

$$ K_\tau = -\tfrac{1}{2} \ln \tanh(h \, \Delta\tau), $$

so a *strong* transverse field becomes a *weak* classical coupling in the time
direction. The zero-temperature limit of the quantum chain is the infinite-extent
limit of the classical strip.

## What the mapping explains

- **Why there is a transition at all.** The 1D quantum chain has a phase
  transition at zero temperature while the 1D classical chain has none at finite
  temperature. The mapping resolves the apparent contradiction: the quantum chain
  is a *two*-dimensional classical model, and two-dimensional Ising models order.
- **Where the exponents come from.** Universal quantities of the quantum critical
  point at $h = J$ are those of the 2D classical Ising model, which is why they are
  the same numbers found in a completely different context and why the central
  charge is the one of a free Majorana field.
- **Why self-duality pins the critical point.** The Kramers–Wannier duality of the
  2D classical model becomes the statement that the quantum chain maps to itself
  under exchanging $J$ and $h$, so a single transition must sit at $h = J$.
- **The direction of the error in a slicing.** The splitting above is exact only as
  $\Delta\tau \to 0$; at finite $\Delta\tau$ it carries a discretisation error set by
  the commutator of the two terms.

## The same decomposition, used forwards

That last point is the bridge to computation. Replace imaginary time $\Delta\tau$
with real time and the identical decomposition becomes the Trotterised circuit a
gate-based quantum computer runs to simulate this model: alternating layers of
coupling and field, with an error controlled by the slice thickness. Quantum Monte
Carlo methods sample the classical model the mapping produces; digital quantum
simulation runs the same factorisation as gates. One identity, read in imaginary
time by the classical algorithm and in real time by the quantum circuit.

## Practical caution

The mapping is exact but it is not free. The classical model it produces is
anisotropic, and the two couplings must be scaled together in any finite-size
study — comparing a quantum chain of length $L$ against a square classical lattice
is a different model. In this project the mapping is used as an explanation rather
than as a solver: the numbers come from the free-fermion solution and from exact
diagonalisation, which need no discretisation and therefore have no $\Delta\tau$ to
extrapolate away.
