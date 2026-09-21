---
title: Real machines these methods can be run and tested on
source: "Bharti et al., Noisy intermediate-scale quantum algorithms, Reviews of Modern Physics 94, 015004 (2022)"
arxiv: 2101.08448
topics: [hardware, nisq, superconducting-qubits, trapped-ions, rydberg-atoms, quantum-annealing, cloud-access, benchmarking, quantum-computing]
---

# Real machines these methods can be run and tested on

A feasibility study that names no machine is not a feasibility study. This note is
the inventory: which technologies exist, which of the four methods each one can
run, what limits it, and how a reader gets access. Figures below are
**representative magnitudes for the mid-2020s generation of hardware, not
calibration data** — every vendor publishes live per-qubit numbers, those move
weekly, and any real estimate should be recomputed against the day's snapshot.

## Superconducting circuits — the default gate-based platform

Qubits are microwave resonators on a chip, controlled by pulses; the whole thing
sits in a dilution refrigerator. Built by IBM, Google, Rigetti, IQM, OQC and
others.

- **Scale**: hundreds of physical qubits on one chip, with a few devices past a
  thousand.
- **Speed**: two-qubit gates in the low hundreds of nanoseconds; single-qubit
  gates a few tens of nanoseconds. This is the fastest platform by a wide margin,
  which matters because a variational method runs the same circuit many thousands
  of times.
- **Errors**: two-qubit gate error around $10^{-3}$ to $10^{-2}$; coherence times
  $T_1$ and $T_2$ of order $100\ \mu$s.
- **Connectivity**: fixed and sparse. IBM's heavy-hexagon lattice gives most
  qubits only two or three neighbours; the sparsity is deliberate, because fewer
  neighbours means less crosstalk and is part of what buys the error rates above.
- **Runs which methods**: VQE, QAOA and variational imaginary time. All three are
  gate-based and all three are limited by the same depth budget.
- **Watch for**: SWAP overhead from transpilation on anything denser than the
  chip's own graph, and the fact that a nominal qubit count is not a usable one —
  a variational circuit needs a well-calibrated connected patch, not the whole
  device.

A one-dimensional chain is the friendliest possible problem here: its bonds are
already edges on any linear or lattice topology, so routing costs nothing and the
whole difficulty is depth against coherence.

## Trapped ions — the cleanest gates, the fewest qubits

Ions held in an electromagnetic trap, each carrying a qubit in its internal
electronic states, coupled through their shared motion. Built by Quantinuum,
IonQ, AQT and others.

- **Scale**: tens of qubits, into the low hundreds.
- **Speed**: two-qubit gates in the tens to hundreds of microseconds — roughly
  a thousand times slower than superconducting. For a method that needs $10^5$
  circuit repetitions, that difference is the wall-clock budget.
- **Errors**: the best in the field, two-qubit error at or below $10^{-3}$, with
  coherence times of seconds. State preparation and readout are also the cleanest.
- **Connectivity**: effectively all-to-all within a register, because the coupling
  runs through shared vibrational modes rather than through wires. No SWAP
  overhead at all.
- **Runs which methods**: all three gate-based methods, and it is the natural
  choice for variational imaginary time, whose $O(m^2)$ metric measurements and
  ancilla-based circuits reward gate quality over gate speed.
- **Watch for**: throughput. High fidelity per shot does not help if the shots
  cannot be taken.

The Ising chain is realised directly here rather than compiled: laser-driven
spin-dependent forces produce the $\hat\sigma^z\hat\sigma^z$ coupling and a direct
drive supplies the transverse field, with the coupling range tunable from nearly
nearest-neighbour to nearly all-to-all. Monroe et al. (2021) review that programme.

## Neutral atoms in Rydberg arrays — geometry as a free parameter

Atoms held in optical tweezers, interacting only when excited to a large-radius
Rydberg state. Built by QuEra, Pasqal, Atom Computing, planqc.

- **Scale**: hundreds to thousands of atoms, the largest qubit registers in the
  gate-based world.
- **Errors**: two-qubit fidelities have improved rapidly and now approach the
  superconducting range; atom loss during a run is a distinctive extra failure
  mode.
- **Connectivity**: set by where the tweezers are placed, and some systems can
  move atoms mid-circuit. Geometry is a design choice rather than a constraint,
  which is unusual and valuable.
- **Runs which methods**: gate-based variational methods on the digital machines,
  and — importantly — **analogue Ising simulation** on the rest. A Rydberg array
  under continuous drive is an Ising-type Hamiltonian with the laser as the
  transverse field, so it can anneal or quench the model without any circuit.
  Bernien et al. (2017) assembled a 51-atom chain and observed this model's
  quantum phase transition directly.
- **Watch for**: which mode a paper is using. An analogue Rydberg result and a
  digital circuit result are not comparable, even when both say "Ising".

## Quantum annealers — analogue, large, and only for one of the four methods

Superconducting flux qubits with tunable couplers, evolving continuously under a
schedule. D-Wave is the sole large-scale commercial supplier.

- **Scale**: thousands of physical qubits, far beyond gate-based devices.
- **Connectivity**: fixed and sparse; denser problems require minor embedding,
  where one logical variable becomes a chain of physical qubits. Dense problems
  can consume qubits quadratically in the number of variables, which is why a
  5,000-qubit machine handles a few hundred fully-connected variables.
- **Runs which methods**: annealing only. It cannot run VQE, QAOA or imaginary
  time, because it has no gates.
- **Watch for**: thermal versus coherent dynamics, and the absence of a
  variational bound — the machine returns a bit string, and nothing certifies it.

## Photonic and other platforms

Photonic processors (Xanadu, PsiQuantum) and spin qubits in silicon are active
programmes with different roadmaps. Neither is currently a routine destination for
a small variational spin-chain study, and a study that names them should say what
it is buying by doing so.

## How a reader actually gets access

All the major platforms are reachable over a cloud API, either directly from the
vendor or through a broker — IBM Quantum, Amazon Braket and Microsoft Azure
Quantum each front several hardware providers behind one interface. In practice:

1. **Simulate first, exactly.** Anything up to roughly 20–30 qubits fits in a
   state-vector simulator on a laptop or a workstation, and for this chain the
   answer is known in closed form. Every bug should be found here, where a run
   costs nothing and the truth is available.
2. **Simulate with a noise model.** Vendors publish device noise models. This is
   where the depth ceiling from the depth note gets its first real test.
3. **Transpile against the real coupling map** and re-count the gates. This is the
   step that most often changes the verdict.
4. **Then queue.** Hardware time is metered, queued, and shared. A variational run
   is not one job but thousands of circuit executions in a classical loop, and
   queue latency between iterations is a real cost that simulation does not show.

## Which platform for which method

| Method | Best-fitted platform | Because |
| --- | --- | --- |
| VQE | superconducting | needs many fast repetitions of a shallow circuit |
| QAOA | superconducting, or neutral atoms | depth scales with problem edges; speed and register size dominate |
| Variational imaginary time | trapped ions | $O(m^2)$ metric measurements and ancilla circuits reward fidelity over speed |
| Annealing | annealer, or analogue Rydberg array | the Hamiltonian is the hardware, no circuit involved |

## How this project's device models line up

This project does not call a vendor API; it models three machines and computes
against them, so that a verdict is reproducible and needs no credential.

- **Ideal** — all-to-all, no error, unbounded coherence. Not a machine: a control.
  Running a configuration here and on a real model isolates algorithmic error from
  device error, which is the subtraction the whole comparison depends on.
- **Linear** — a row of qubits with representative superconducting figures. The
  best case that is still a real machine: an open chain maps on with every bond
  already an edge, routing costs nothing, and the whole difficulty is depth against
  coherence. A "no" here is a "no" that better connectivity would not have saved.
- **Heavy-hex-27** — a published sparse lattice, the topology most people mean by
  "a real quantum computer". Its sparsity puts a number on the routing cost for a
  problem shaped like a chain.

The figures carried by those models are representative magnitudes of the platforms
described above, and the code says so in each device's provenance field. That
honesty is load-bearing: a study that presented modelled numbers as a calibration
snapshot would be making a claim it cannot support, and the distinction between
"this is the order of magnitude" and "this is what the machine did on Tuesday" is
exactly the one a feasibility report exists to keep straight.
