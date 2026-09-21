---
title: Quantum annealing as the fourth method, and what QAOA adds over it
source: "Albash and Lidar, Adiabatic quantum computation, Reviews of Modern Physics 90, 015002 (2018)"
arxiv: 1611.04471
topics: [quantum-annealing, adiabatic-quantum-computing, qaoa, energy-gap, analogue-versus-digital, quantum-computing, hardware]
---

# Quantum annealing as the fourth method, and what QAOA adds over it

Three of the methods in this shelf are variational: a parameterised circuit, a
classical loop, a search. Quantum annealing is not. It belongs in the comparison
anyway, for two reasons — it is the oldest of the four and the one with the most
hardware actually deployed, and QAOA is best understood as its digital descendant.

## The method in one paragraph

Prepare the ground state of a Hamiltonian that is easy to prepare, then change the
Hamiltonian slowly into one whose ground state is the answer. If the change is
slow enough the system stays in the instantaneous ground state throughout — this
is the adiabatic theorem — and reading out the final state reads out the solution.
The standard schedule is

$$ \hat H(s) \;=\; -A(s)\sum_i \hat\sigma^x_i \;-\; B(s)\sum_{i<j} J_{ij}\,\hat\sigma^z_i \hat\sigma^z_j, \qquad s = t/t_f $$

running from $A \gg B$ at $s=0$ to $B \gg A$ at $s=1$. Kadowaki and Nishimori
(1998) proposed it in this form; Farhi et al. (2000) gave the equivalent adiabatic
formulation.

Note what that Hamiltonian is. It is the transverse-field Ising model, swept along
a line through its own phase diagram. For an annealer, this project's model is not
an analogy — it is the machine's native language.

## No parameters, no optimiser, one knob

There is nothing to train. The user supplies the couplings $J_{ij}$, which are the
problem, and the total time $t_f$, which is the effort. There is no classical
optimisation loop, no gradient, no barren plateau, and no ansatz to choose badly.
That is a genuine simplification and it is the reason annealers reached useful
problem sizes years before gate-based machines did.

The cost of having no parameters is having no recourse. When a variational method
underperforms you can change the ansatz, the optimiser or the cost function. When
an anneal underperforms, the only knobs are "go slower" and "change the encoding".

## The runtime is a gap, not an energy

The adiabatic theorem's speed limit is governed by the smallest energy gap
encountered along the sweep — the gap between the ground state and the first
excited state, at its narrowest point:

$$ t_f \;\sim\; \frac{1}{\Delta_{\min}^{2}} $$

Everything else is a constant factor. So the interesting quantity in an annealing
study is not an energy at all; it is a gap, evaluated along a sweep, and its
minimum. For the uniform chain that minimum sits at the critical point $h = J$,
where the gap of the infinite chain closes; at finite size it closes only as fast
as the inverse chain length, which is the finite-size fingerprint of a critical
point rather than a genuine degeneracy.

For the hard instances annealers are aimed at — disordered, frustrated — the
minimum gap can shrink exponentially with problem size, and nobody knows it in
advance. That is the honest statement of where the method's guarantee goes.

## QAOA is this method, discretised

Trotterise the sweep: chop it into steps and alternate the two terms rather than
applying their sum. One step of a dominant-coupling phase followed by one step of
a dominant-field phase is precisely one QAOA layer, with $\gamma_k$ and $\beta_k$
set by the schedule. As the number of layers $p$ grows with correspondingly small
angles, QAOA reproduces the anneal, which is where its $p \to \infty$ exactness
comes from.

What QAOA adds is that the angles are *not* required to follow a schedule. They
are free parameters, fitted by a classical optimiser, and the optimiser routinely
finds angle sequences that beat the adiabatic ones at the same depth. QAOA is
therefore annealing with the schedule treated as something to learn instead of
something to assume — and the price of that freedom is the classical optimisation
loop, the shot budget it consumes, and the trainability problems that come with
it.

The comparison is not "digital versus analogue" so much as **fixed schedule versus
learned schedule**, and the practical question is whether the learning pays for
itself at the depth the hardware allows.

## Where annealing wins and where it does not

Wins:

- Problem size. Deployed annealers hold thousands of qubits, orders of magnitude
  beyond what a gate-based device can run a deep variational circuit on.
- Simplicity of interface. Submit couplings, receive bit strings. No circuit, no
  transpilation, no ansatz design.
- Native problem fit. A quadratic Ising objective is what the hardware takes,
  which is exactly what the mapping in this shelf's cost-and-mixer note produces.

Does not win:

- **Connectivity, via minor embedding.** Annealing hardware has a fixed sparse
  coupling graph. A problem denser than that graph must be *minor-embedded*:
  each logical variable becomes a chain of physical qubits held together by strong
  ferromagnetic couplings. This costs qubits — often quadratically in the number of
  logical variables for dense problems — and introduces a new failure mode, chain
  breaking, plus a new parameter, the chain strength, that has to be tuned. It is
  the practical reason a nominal thousands-of-qubits machine solves problems with
  hundreds of variables.
- **Coherence.** Deployed annealers are open systems in contact with a thermal
  environment. Some of what they do is thermal relaxation rather than coherent
  quantum evolution, which is why demonstrations work hard to show coherence
  specifically.
- **No variational bound.** VQE returns an energy that provably cannot fall below
  the true ground-state energy, so a value below it is a bug. An anneal returns a
  bit string and its classical cost; there is no comparable guarantee, and a wrong
  answer looks like a valid one.
- **Provable speedup.** For the general case, none is established. Comparisons
  against well-tuned classical heuristics — simulated annealing, tensor networks,
  specialised solvers — have repeatedly failed to show a scaling advantage.

## Why this chain is the calibration instance for annealing too

The uniform ferromagnetic chain is worthless as an annealing *problem*: the answer
is "all spins aligned", and no computation is needed. Its value is validation.
Sweeping too fast leaves a predictable density of defects behind — Kibble–Zurek
scaling, derived exactly for this chain by Dziarmaga (2005) and by Zurek, Dorner
and Zoller (2005) — so the *failure* mode is predicted analytically as well as the
success. King et al. (2022) annealed a ring of nearly two thousand flux qubits
configured as this model and recovered the predicted scaling, which was presented
as evidence of coherent rather than thermal dynamics.

A machine that anneals this chain and reports the wrong defect scaling is not
merely imprecise; it is not evolving coherently. That is the strongest kind of
statement an exactly solvable model can license, and it is why the chain keeps
appearing in hardware papers.

## The four-way summary

| | Anneal | QAOA | VQE | Variational imaginary time |
| --- | --- | --- | --- | --- |
| Control | fixed schedule | learned angles | learned angles | set by the Hamiltonian |
| Classical loop | none | optimiser | optimiser | linear solve per step |
| Output | bit string | bit string | energy (upper bound) | a state, and its energy |
| Runtime set by | minimum gap | depth and shots | depth and shots | steps and $O(m^2)$ metric measurements |
| Hardware | analogue annealer | gate-based | gate-based | gate-based, usually with an ancilla |
| Fails by | closing gap, thermal noise | flat landscape, shallow depth | local minima, flat landscape | trajectory leaving the circuit's reach |
