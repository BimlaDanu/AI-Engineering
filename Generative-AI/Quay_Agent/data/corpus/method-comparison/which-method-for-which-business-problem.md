---
title: Which method for which problem, including when the answer is a classical solver
source: "Lucas, Ising formulations of many NP problems, Frontiers in Physics 2, 5 (2014)"
arxiv: 1302.5843
topics: [use-cases, combinatorial-optimisation, business, decision-guide, qaoa, vqe, quantum-annealing, quantum-speedups]
---

# Which method for which problem, including when the answer is a classical solver

Four methods, and the question a decision-maker asks is not how they work but
which one applies. This note is the decision guide, and it is written to be usable
by somebody who does not intend to learn any physics. It ends where an honest guide
has to end: with the cases where the answer is a classical solver.

## First, sort the problem into one of two families

Everything these methods do falls into one of two shapes, and the shapes are not
close.

**Family A — find the lowest energy of a quantum system.** Molecules, catalysts,
magnetic materials, batteries. The unknown is a *quantum state*, the answer is a
number derived from it, and the reason a classical computer struggles is that the
state does not fit in memory. This is VQE's family, and variational imaginary
time's.

**Family B — find the best of exponentially many discrete choices.** Routing,
scheduling, portfolio selection, network design, facility placement, graph
partitioning. The unknown is a *set of decisions*, the answer is a list of them,
and the reason a classical computer struggles is combinatorial explosion. This is
QAOA's family, and annealing's.

The two are connected by a technical fact rather than by a similarity: a Family B
objective, once written in $\pm 1$ variables, has the same algebraic form as a
Family A Hamiltonian. That is why one piece of quantum hardware can be aimed at
both, and it is also the source of most of the field's overselling — the shared
form does not mean shared difficulty or shared prospects.

## Family B: what the mapping requires of your problem

Lucas (2014) is the standard catalogue of encodings, and it is worth knowing what
it asks for before costing a pilot. A problem maps cleanly when:

- the decisions are **binary**, or can be written as binary indicators;
- the objective is **quadratic** in those indicators, or can be made quadratic by
  introducing auxiliary variables (which cost qubits);
- the constraints can be written as penalty terms, or handled by a
  constraint-preserving mixer.

It maps badly when the objective involves continuous quantities, ratios,
conditional logic, or long chains of implications — each of which becomes
auxiliary variables and extra couplings, and the qubit count grows faster than the
problem description does.

**The number that decides a pilot.** Count the non-zero couplings after encoding.
On gate-based hardware that is the two-qubit gate count per QAOA layer, so
multiply by the depth and compare against the device's gate-error and coherence
ceilings. On an annealer it is the density of the coupling graph, which sets the
minor-embedding overhead — how many physical qubits each logical variable
consumes. In both cases the answer is arithmetic, available before any hardware is
touched, and it disqualifies most first attempts.

## Family A: what the payoff depends on

The system size that is classically intractable is a moving target, and it moves
in the classical direction. Density functional theory, coupled cluster, quantum
Monte Carlo and tensor networks each handle large classes of problem well. A
quantum advantage in this family needs a system that is *strongly correlated* —
where those approximations break — and small enough to fit on a device with enough
depth to represent it. Current careful resource estimates put the genuinely
valuable cases beyond NISQ hardware and inside the error-corrected era.

That is not an argument against work in this family now. It is an argument for
being explicit about which era a result belongs to.

## The decision table

| If your problem is… | Start with | Why | The thing that will bite |
| --- | --- | --- | --- |
| a molecular or material ground-state energy | VQE | it is what the method was built for, and its answer is a bound | measurement count, and chemical accuracy is a hard target |
| the same, but the optimiser stalls | variational imaginary time | the trajectory is set by the Hamiltonian, not by a search | $O(m^2)$ metric measurements per step |
| a discrete optimisation with a sparse quadratic objective | QAOA | depth scales with the number of couplings, and shallow circuits are all NISQ allows | classical heuristics are very good at these |
| a discrete optimisation, large, and you want an answer this week | quantum annealing | thousands of qubits, and the interface is just the couplings | minor embedding, and no certificate on the answer |
| a spin-model phase diagram | analogue simulation, or VQE | some hardware realises spin Hamiltonians natively | analogue and digital results are not comparable |
| anything, and you need to know your stack is correct | the transverse-field Ising chain | exactly solvable, so a residual can be attributed | it proves correctness, never advantage |

## When the answer is a classical solver

A guide that never says this is not a guide. Reach for classical methods when:

- **The problem is one dimensional, uniform, or otherwise structured.** This
  project's own chain is the example. Exactly solvable means solved.
- **A good heuristic already exists.** Modern mixed-integer programming solvers,
  simulated annealing and specialised routing solvers handle industrially relevant
  instances routinely. The benchmark for a quantum method is the best available
  classical method, not brute-force enumeration, and comparisons against the
  latter are the most common way a quantum result is inflated.
- **The instance is small.** Anything up to roughly 20–30 qubits is exactly
  simulable, so a "quantum" result at that size is a simulation result and should
  be reported as one.
- **Approximate is not acceptable.** QAOA and annealing return good candidates,
  not certified optima. If the application needs a proof of optimality, an
  exact classical solver is answering a different and stronger question.
- **The economics do not close.** A variational run is thousands of circuit
  executions in a classical loop, on metered and queued hardware, with a classical
  optimiser in between. Wall-clock cost per answer is frequently the deciding
  number and is frequently left out of the pitch.

## What a defensible pilot looks like

1. State the problem and the objective in ordinary language, then in $\pm 1$
   variables. If step two is hard, that is the finding.
2. Count variables and couplings after encoding. Compare against a device's
   ceilings. Most candidates stop here, cheaply.
3. Establish the classical baseline properly, with a well-tuned solver, and record
   its runtime and its answer quality.
4. Validate the quantum stack on an instance with a known answer — which is what
   an exactly solvable model is for.
5. Only then run the real instance, and report the comparison against step 3
   including hardware time.

Steps 2 and 3 are the ones most often skipped, and they are the two that determine
the verdict. A study that does them and concludes "no advantage" has produced a
real result and saved real money.
