---
title: "An Implementation of the Quantum Verification of Matrix Products Algorithm"
source: "Elton Pinto, arXiv:2208.09914v1 (2022-08-21)"
arxiv: 2208.09914v1
topics: [transpilation, connectivity, hardware]
---

# An Implementation of the Quantum Verification of Matrix Products Algorithm

## Abstract

We present a space-efficient implementation of the quantum verification of matrix products (QVMP) algorithm and demonstrate its functionality by running it on the Aer simulator with two simulation methods: statevector and matrix product state (MPS). We report circuit metrics (gate count, qubit count, circuit depth), transpilation time, simulation time, and a proof of Grover oracle correctness. Our study concludes that while QVMP can be simulated on moderately sized inputs, it cannot scale to a degree where we can observe any quantum advantage on current quantum hardware due to circuit depth and qubit count constraints. Further, the choice of simulation method has a noticeable impact on the size of the transpiled circuit which slows down development.

## Citation

Elton Pinto, arXiv:2208.09914v1 (2022-08-21). Retrieved from the arXiv API; the text above is the authors' own abstract, reproduced without alteration.
