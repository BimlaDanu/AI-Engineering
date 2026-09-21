---
title: The derivatives of the ground-state energy, and what each one shows
source: "Pfeuty, The one-dimensional Ising model with a transverse field, Annals of Physics 57, 79 (1970); Sachdev, Quantum Phase Transitions, 2nd ed., CUP (2011), ch. 4"
arxiv: null
topics: [ground-state-energy, magnetisation, hellmann-feynman, susceptibility, critical-exponents, finite-size-scaling]
---

# The derivatives of the ground-state energy, and what each one shows

A common request is "plot the ground-state energy against the field". Done on its
own it is the least informative picture available, and the reason is worth stating
before the equations: **the ground-state energy is smooth through the phase
transition.** There is no kink at $h = J$, no jump, and nothing a reader can point
at. Everything the transition does to this model appears in the *derivatives*, and
each successive one shows more of it.

## The three curves, and what they are

With the convention

$$
\hat H = -J \sum_i \hat\sigma^z_i \hat\sigma^z_{i+1} - h \sum_i \hat\sigma^x_i ,
$$

and $\epsilon(k) = 2\sqrt{J^2 + h^2 - 2Jh\cos k}$ the quasiparticle energy of the
free-fermion solution, the energy density of a ring of $L$ sites is

$$
\frac{E_0}{L} = -\frac{1}{L}\sum_{k>0} \epsilon(k),
\qquad k = \frac{(2n+1)\pi}{L},
$$

and in the thermodynamic limit the sum becomes an elliptic integral,

$$
\frac{E_0}{L} \;\xrightarrow[L\to\infty]{}\;
-\frac{2}{\pi}\,(J+h)\;E\!\left(\frac{4Jh}{(J+h)^2}\right),
$$

with $E(\cdot)$ the complete elliptic integral of the second kind.

## First derivative: it *is* the magnetisation

Not "is related to" — is. The Hellmann–Feynman theorem states that for a normalised
eigenstate $|\psi_0\rangle$ of a Hamiltonian depending on a parameter $\lambda$,

$$
\frac{\partial E_0}{\partial \lambda}
  = \Big\langle \psi_0 \Big| \frac{\partial \hat H}{\partial \lambda} \Big| \psi_0 \Big\rangle .
$$

Here $\lambda = h$ and $\partial \hat H/\partial h = -\sum_i \hat\sigma^x_i$, so

$$
\frac{\partial}{\partial h}\frac{E_0}{L} = -\langle \sigma^x \rangle,
\qquad
\langle \sigma^x \rangle = \frac{1}{L}\sum_i \langle \hat\sigma^x_i \rangle
 = \frac{1}{L}\sum_k \frac{h - J\cos k}{\sqrt{J^2 + h^2 - 2Jh\cos k}} .
$$

Two consequences follow, and both are practical rather than decorative.

**No numerical differentiation is ever needed for this curve.** The derivative is a
closed-form momentum sum, exact to machine precision rather than to a step size. A
finite difference taken across a sweep would depend on how many points were sampled,
so asking for a coarser plot would change the physics the plot reported.

**Two curves that look independent are one quantity.** A figure showing
$\langle\sigma^x\rangle$ and a figure showing $\partial(E_0/L)/\partial h$ are the
same numbers with a sign between them. That makes the pair a *check*: if a
calculation produces them by different routes and they do not mirror each other,
one route is wrong. It also makes a specific mistake possible — answering a question
about the magnetisation's derivatives with the energy's, which is showing the right
numbers under the wrong name.

The magnetisation runs from $0$ at $h = 0$, where every spin lies along $\sigma^z$
and the transverse alignment is nil, to $1$ as $h$ dominates and every spin is
knocked over.

## Second derivative: the transverse susceptibility, and the first sharp feature

Differentiating once more,

$$
\frac{\partial^2}{\partial h^2}\frac{E_0}{L}
 = -\frac{\partial \langle \sigma^x \rangle}{\partial h}
 = -\frac{1}{L}\sum_k \frac{2J^2\sin^2 k}{\left(J^2 + h^2 - 2Jh\cos k\right)^{3/2}} ,
$$

which is negative everywhere — the ground-state energy is concave in the field, as
any ground-state energy must be — and **dips sharply near $h = J$**. Its negative,
$\partial\langle\sigma^x\rangle/\partial h$, is the transverse susceptibility: how
much more aligned the chain becomes for a little more push.

In the infinite chain this quantity diverges at $h = J$, logarithmically rather than
as a power law. That is the statement $\alpha = 0$ for the specific-heat-like
exponent of this transition, and it is the same logarithm as in the specific heat of
the two-dimensional classical Ising model — which the quantum-to-classical mapping
says it must be.

On a **finite** ring nothing diverges. What happens instead is that the peak of the
susceptibility grows and moves towards $h/J = 1$ as the chain lengthens. Computed
from the closed form on a fine grid:

| $L$ | peak at $h/J$ | peak height |
| --- | --- | --- |
| 6 | 0.835 | 0.83 |
| 12 | 0.949 | 0.99 |
| 20 | 0.979 | 1.13 |
| 40 | 0.994 | 1.34 |
| 100 | 0.999 | 1.63 |

Both halves of that table matter. A reader shown only the peak *height* would think
the feature is merely growing; a reader shown only its *position* would think it is
merely moving. It is doing both, and doing both is how a chain of a dozen magnets
points at a singularity it does not itself contain.

## Third derivative: where the peak is, without looking for it

The second derivative of the magnetisation — equivalently the third of the energy
density — is

$$
\frac{\partial^2 \langle \sigma^x \rangle}{\partial h^2}
 = -\frac{1}{L}\sum_k \frac{6J^2\sin^2 k\,(h - J\cos k)}
        {\left(J^2 + h^2 - 2Jh\cos k\right)^{5/2}} ,
$$

positive below the peak of the susceptibility, where the magnetisation curve is
still steepening, and negative above it, where it is flattening towards saturation.
It **crosses zero exactly at the peak**, which turns "where is the transition on
this finite chain?" from a maximum to be scanned for into a root to be solved for.

## Reading the three panels together

Stacked on a shared axis, the three panels say one thing that three separate figures
do not:

1. $E_0/L$ — smooth, featureless, no visible sign of the transition.
2. $\partial(E_0/L)/\partial h = -\langle\sigma^x\rangle$ — bends, and the bend
   steepens with $L$.
3. $\partial^2 (E_0/L)/\partial h^2$ — a sharp dip that sharpens and migrates to
   $h/J = 1$.

The singularity of the infinite chain is not absent from the finite one; it is
*two derivatives down*. That is the general lesson of this figure and it is not
specific to this model: at a continuous phase transition the free energy stays
smooth and a derivative of it does not.

## The order parameter this does not show

$\langle\sigma^x\rangle$ is the alignment with the field, and it is non-zero on both
sides of the transition — it is not the order parameter. The order parameter is the
longitudinal magnetisation $\langle\sigma^z\rangle$, which is non-zero in the
ordered phase ($h < J$) and zero above it, and on a finite ring it vanishes
identically by symmetry unless the symmetry is broken by hand: the two ground states
related by flipping every spin are degenerate, so the ring averages over both. The
usual finite-size substitutes are the correlation function
$\langle\sigma^z_i\sigma^z_{i+r}\rangle$ at large $r$ or the square
$\langle(\sum_i\sigma^z_i/L)^2\rangle$, neither of which is a first derivative of
anything and both of which need the state rather than only its energy.
