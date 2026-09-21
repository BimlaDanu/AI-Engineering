---
title: Which quantum speedups are real, and what that means for AI and ML
source: "Montanaro, Quantum algorithms: an overview, npj Quantum Information 2, 15023 (2016)"
arxiv: 1511.04206
topics: [quantum-algorithms, speedup, machine-learning, business-applications, applications, benchmarking]
---

# Which quantum speedups are real, and what that means for AI and ML

The question behind most commercial interest is "what does quantum hardware make
faster". Montanaro (2016) surveys the algorithms with an eye to exactly that, and
the useful thing about the survey is how sharply it separates the categories.

## Proven asymptotic speedups, and their scope

- **Factoring and discrete logarithms.** Shor's algorithm is exponentially faster
  than the best known classical method. Its commercial significance is cryptography
  — that is, a threat model and a migration programme, not a product.
- **Unstructured search.** Grover's algorithm gives a quadratic speedup, provably
  optimal for the black-box problem. Quadratic is real but modest: it is eaten by
  the constant-factor overhead of error correction for problem sizes anyone runs
  today.
- **Hamiltonian simulation.** Simulating the dynamics of a local quantum
  Hamiltonian is efficient on a quantum computer and believed hard classically in
  general. This is the clearest case, and it is the one this project sits inside:
  the transverse-field chain is a local Hamiltonian, and a circuit that evolves it
  is the textbook instance of the algorithm.
- **Linear systems.** The HHL algorithm solves sparse, well-conditioned systems
  with exponentially better scaling in dimension — subject to fine print that
  Aaronson (2015) set out plainly: the input must be loadable as a quantum state,
  the matrix must be sparse and well conditioned, and the output is a quantum state
  rather than a vector you can read. Each condition can silently remove the
  advantage.

## Machine learning specifically

Quantum machine learning is an active field (Biamonte and colleagues, 2017) with
three honest categories in it.

**Speedups that were withdrawn.** Several early claims of exponential advantage —
recommendation systems, principal component analysis, low-rank algebra — assumed a
quantum data structure for input. Tang and others then produced classical
algorithms with comparable scaling under the analogous classical assumption, which
removed the exponential gap rather than the usefulness of the work. The lesson is
structural: an advantage that depends on a data-loading assumption must be compared
against a classical method given the same assumption.

**Speedups that are quadratic or heuristic.** Grover-style amplitude amplification
inside a training loop, and variational models — the quantum analogue of a neural
network, trained by a classical optimiser. These have no proven advantage. Barren
plateaus, where gradients vanish exponentially with system size, are a known
obstacle to scaling them.

**Sampling and generative models.** Quantum Boltzmann machines and related models
propose to sample distributions that are hard to sample classically. This is where
a transverse-field Ising Hamiltonian appears as the model being trained rather than
as a physics problem.

## Optimisation, which is what most businesses ask about

Annealing and QAOA are heuristics. No proven asymptotic speedup exists for either
on general optimisation problems, and the empirical record against strong classical
heuristics is mixed at best. This is the gap between the two halves of the field
worth carrying away: quantum *simulation* of quantum systems has a solid
theoretical case and the clearest experimental progress, while quantum
*optimisation* for business problems has the larger market and the weaker evidence.

## Why an exactly solvable model matters to all of this

Every claim above is eventually an empirical claim about a device. A model whose
answer is known in closed form is where such a claim is calibrated — it cannot
demonstrate an advantage, by construction, but it is the only place a device's
output can be declared right or wrong without argument.
