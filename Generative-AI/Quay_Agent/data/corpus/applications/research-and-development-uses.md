---
title: What industrial R&D actually uses a solvable chain for
source: "Yarkoni, Raponi, Back and Schmitt, Quantum annealing for industry applications: introduction and review, Reports on Progress in Physics 85, 104001 (2022)"
arxiv: 2112.07491
topics: [research-and-development, business-applications, benchmarking, quantum-annealing, applications]
---

# What industrial R&D actually uses a solvable chain for

The commercial questions about this model are rarely "what is its ground-state
energy". They are about what a team can do with a problem whose answer is already
known, and there are four recurring uses.

## Acceptance testing a device or a service

A quantum processor, or an access contract for one, is evaluated before it is
trusted. The transverse-field chain is what gets run: the closed-form spectrum
turns "is this device working" into a comparison rather than a judgement. A
discrepancy is attributable — miscalibrated couplings, a schedule that is too fast,
readout bias — because the correct answer is not in question.

## Validating a pipeline end to end

An industrial quantum workflow is a stack: a formulation layer, a compiler, an
embedder or transpiler, error mitigation, and classical post-processing. Every
layer can be wrong in a way that still returns plausible numbers. Running the stack
on an instance with a known answer is the only cheap way to establish that the
plumbing is right before it is pointed at a problem where no one can tell.

## Formulating the business problem, which is often the deliverable

Yarkoni and colleagues (2022) review deployments across logistics, traffic-flow
routing, scheduling, and manufacturing planning. A consistent finding is that the
work of writing an operational problem as an explicit cost function — decision
variables, objective, penalties — improves the classical solution too, and is
frequently the part that survives the pilot. Several published projects reported
their main gain as a better-specified problem rather than a faster solver.

## Training people, and reading the literature

The chain is the shared example across the field, so it is where a new team member
starts and what a vendor's benchmark almost certainly means. Being able to read a
hardware paper's Ising results critically is itself an R&D capability.

## The honest summary of the commercial state

Review after review lands in the same place. There are many formulations, a growing
number of small pilots, and no established commercial advantage at production scale
for optimisation problems on quantum hardware. Where a quantum device was reported
to win, a classical heuristic on ordinary hardware has usually caught up within a
year or two. The value being delivered now is capability, formulation and
benchmarking — not throughput. A project that says so plainly is easier to trust
about the parts that do work.
