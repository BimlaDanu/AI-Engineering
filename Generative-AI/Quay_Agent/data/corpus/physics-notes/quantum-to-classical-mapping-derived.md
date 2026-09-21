---
title: The quantum-to-classical mapping, derived line by line
source: "Suzuki, Relationship among Exactly Soluble Models of Critical Phenomena, Progress of Theoretical Physics 56, 1454 (1976); Kogut, An introduction to lattice gauge theory and spin systems, Rev. Mod. Phys. 51, 659 (1979)"
arxiv: null
topics: [quantum-to-classical-mapping, suzuki-trotter, transfer-matrix, imaginary-time, kramers-wannier-duality, quantum-monte-carlo]
---

# The quantum-to-classical mapping, derived line by line

A $d$-dimensional quantum system at zero temperature is a $(d+1)$-dimensional
classical system. Stated that way it is a slogan. This note carries it out
explicitly for the transverse-field Ising chain, where every step is elementary and
the result is exact: no large-$L$ limit, no small parameter, and no approximation
beyond one that is controlled and vanishes.

Companion to *The transfer matrix and the quantum-to-classical mapping*, which says
what the correspondence is *for*. This one is the algebra.

## Step 0 — the object to be rewritten

Take the chain

$$
\hat H = \underbrace{-J \sum_{i=1}^{L} \hat\sigma^z_i \hat\sigma^z_{i+1}}_{\hat H_{zz}}
       \; \underbrace{-\, h \sum_{i=1}^{L} \hat\sigma^x_i}_{\hat H_x}
$$

and its partition function at inverse temperature $\beta$,

$$
Z = \operatorname{Tr} e^{-\beta \hat H}.
$$

The two pieces do not commute, $[\hat H_{zz}, \hat H_x] \neq 0$, which is the only
reason there is anything to do.

## Step 1 — slice the exponential (Suzuki–Trotter)

Write $\Delta\tau = \beta/M$ and split each slice:

$$
e^{-\beta \hat H} = \left( e^{-\Delta\tau \hat H} \right)^{M}
 = \left( e^{-\Delta\tau \hat H_{zz}} e^{-\Delta\tau \hat H_x} \right)^{M}
   + \mathcal{O}\!\left(\Delta\tau^2\right),
$$

the error per slice coming from the Baker–Campbell–Hausdorff commutator
$\tfrac{1}{2}\Delta\tau^2[\hat H_{zz}, \hat H_x]$. The total error is
$\mathcal{O}(\beta \Delta\tau)$ and vanishes as $M \to \infty$ at fixed $\beta$. This
is the **only** approximation in the whole derivation, and it is the same
factorisation a Trotterised quantum circuit runs — in real time instead of
imaginary.

## Step 2 — insert a complete set of states between every slice

Let $|s\rangle = |s_1, \dots, s_L\rangle$ be a $\sigma^z$ basis state, $s_i = \pm 1$,
and insert $\sum_{s} |s\rangle\langle s| = \mathbb{1}$ between all $2M$ exponentials.
Labelling the slices by $\tau = 1, \dots, M$ with periodic identification
$\tau + M \equiv \tau$,

$$
Z = \sum_{\{s_{i,\tau}\}} \prod_{\tau=1}^{M}
      \langle s_{\tau} | e^{-\Delta\tau \hat H_{zz}} | s_{\tau} \rangle \,
      \langle s_{\tau} | e^{-\Delta\tau \hat H_x} | s_{\tau+1} \rangle .
$$

The sum now runs over $\pm 1$ variables on an $L \times M$ **lattice**: the original
chain along one axis, the imaginary-time slices along a new one. Every quantum
operator has gone; what is left is a sum over classical configurations.

## Step 3 — the diagonal factor gives the spatial coupling

$\hat H_{zz}$ is diagonal in this basis, so its matrix element is just its
eigenvalue:

$$
\langle s_{\tau} | e^{-\Delta\tau \hat H_{zz}} | s_{\tau} \rangle
 = \exp\!\Big( \Delta\tau J \sum_i s_{i,\tau}\, s_{i+1,\tau} \Big)
 \equiv \exp\!\Big( K_x \sum_i s_{i,\tau} s_{i+1,\tau} \Big),
 \qquad \boxed{K_x = J\,\Delta\tau}.
$$

## Step 4 — the off-diagonal factor gives the time coupling

$\hat H_x$ is a sum of single-site terms, so the matrix element factorises over
sites, and for one site the whole calculation is two lines. Since
$(\hat\sigma^x)^2 = \mathbb{1}$,

$$
e^{a \hat\sigma^x} = \cosh a + \hat\sigma^x \sinh a
\quad\Longrightarrow\quad
\langle s' | e^{a\hat\sigma^x} | s \rangle =
\begin{cases}
\cosh a, & s' = s,\\
\sinh a, & s' = -s,
\end{cases}
\qquad a = h\,\Delta\tau .
$$

Both cases are captured by a single Ising-like weight. Writing
$\langle s'|e^{a\hat\sigma^x}|s\rangle = C\,e^{K_\tau s s'}$ and solving the two
equations $Ce^{K_\tau} = \cosh a$, $Ce^{-K_\tau} = \sinh a$ gives

$$
\boxed{K_\tau = -\tfrac{1}{2}\ln \tanh\!\left(h\,\Delta\tau\right)
              = \tfrac{1}{2}\ln \coth\!\left(h\,\Delta\tau\right)},
\qquad
C = \sqrt{\cosh a \sinh a} = \sqrt{\tfrac{1}{2}\sinh 2a}.
$$

Note the direction of the map, which is the counter-intuitive part: a **strong**
transverse field is a **weak** classical coupling along the time axis, since
$K_\tau \to 0$ as $h\Delta\tau \to \infty$. A strong field decorrelates
neighbouring time slices, which is what "quantum fluctuations destroy order" means
once the mapping is in hand.

## Step 5 — the result

Collecting the two factors,

$$
Z = C^{LM} \sum_{\{s_{i,\tau}\}} \exp\!\left[
   \sum_{i,\tau} \Big( K_x\, s_{i,\tau} s_{i+1,\tau}
                     + K_\tau\, s_{i,\tau} s_{i,\tau+1} \Big) \right],
$$

which is exactly the partition function of an **anisotropic two-dimensional
classical Ising model** at unit temperature, with coupling $K_x$ along the chain and
$K_\tau$ along imaginary time. One quantum chain, one classical sheet.

Correlation functions map the same way. A quantum equal-time correlator becomes a
classical correlator within one row,

$$
\langle \hat\sigma^z_i \hat\sigma^z_j \rangle
 = \big\langle s_{i,\tau} s_{j,\tau} \big\rangle_{\text{classical}},
$$

and an imaginary-time-separated quantum correlator becomes a classical correlator
between rows — which is why the quantum system's *dynamics* is encoded in the
classical system's geometry.

## Step 6 — the critical point, exactly, for any slice thickness

The anisotropic square-lattice Ising model is critical when

$$
\sinh(2K_x)\,\sinh(2K_\tau) = 1 .
$$

Substituting the two couplings, and using
$\sinh\!\big(2K_\tau\big) = 1/\sinh(2h\Delta\tau)$ — which follows from
$e^{2K_\tau} = \coth(h\Delta\tau)$ — the condition becomes

$$
\frac{\sinh(2J\Delta\tau)}{\sinh(2h\Delta\tau)} = 1
\qquad\Longleftrightarrow\qquad
\boxed{h = J}.
$$

The $\Delta\tau$ cancels **identically**. The mapping puts the quantum critical
point at $h = J$ at every slice thickness, which is a strong internal check on the
whole construction: a step with an algebra error would leave a $\Delta\tau$ behind
and predict a critical field that depended on how finely the exponential was cut.

The same statement read backwards is the Kramers–Wannier duality: exchanging $J$ and
$h$ maps the chain to itself, so a single transition can only sit on the self-dual
line.

## Step 7 — what has to be scaled with what

The classical model is anisotropic, and taking $\Delta\tau \to 0$ makes it extremely
so: $K_x \to 0$ while $K_\tau \to \infty$. Two consequences for anyone using this
numerically:

- **The continuum limit is a direction, not a shape.** Comparing a quantum chain of
  $L$ sites against a *square* $L \times L$ classical lattice is comparing two
  different models. What must be held fixed is the aspect ratio in units of the
  correlation length, $M\Delta\tau \propto L^{z}$ with $z = 1$ for this model.
- **The universal numbers are the two-dimensional classical ones.** The critical
  exponents of the quantum chain at $h = J$ are those of the 2D classical Ising
  model — $\beta = 1/8$, $\nu = 1$, $\eta = 1/4$, and a logarithmic rather than
  power-law specific-heat singularity, $\alpha = 0$. This is why the second
  derivative of the ground-state energy with respect to $h$ diverges
  logarithmically and not as a power.

## Where this is used in this project

Twice, in opposite directions.

**As a sampler.** The classical baseline here is a variational imaginary-time
ansatz whose expectation values are evaluated by exactly this construction: a
resolution of the identity between every exponential turns the quantum expectation
value into a classical Ising average on a lattice of imaginary-time slices, sampled
by Metropolis. Every weight is real and positive — there is no sign problem for this
model — which is precisely why the chain is the honest place to ask whether a
quantum computer earns its keep: a classical sampler can reach arbitrary statistical
precision on the same variational family, given enough sweeps.

**As a circuit.** Replace imaginary time with real time and the Step 1 factorisation
is the Trotterised circuit a gate-based machine runs: alternating layers of coupling
and field, with an error controlled by the layer thickness. The alternating structure
of the hardware-efficient and QAOA-style ansätze used here is that same
decomposition, with the slice thicknesses promoted to free variational parameters
instead of being fixed at $\Delta\tau$.

One identity, run in imaginary time by the classical algorithm and in real time by
the quantum circuit. That symmetry is the reason this chain is a fair test bed
rather than a rigged one.
