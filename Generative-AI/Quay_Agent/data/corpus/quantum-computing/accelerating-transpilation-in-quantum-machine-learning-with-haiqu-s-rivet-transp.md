---
title: "Accelerating Transpilation in Quantum Machine Learning with Haiqu's Rivet-transpiler"
source: "Aleksander Kaczmarek, Dikshant Dulal, arXiv:2508.21342v1 (2025-08-29)"
arxiv: 2508.21342v1
topics: [transpilation, connectivity, hardware]
---

# Accelerating Transpilation in Quantum Machine Learning with Haiqu's Rivet-transpiler

## Abstract

Transpilation is a crucial process in preparing quantum circuits for execution on hardware, transforming virtual gates to match device-specific topology by introducing swap gates and basis gates, and applying optimizations that reduce circuit depth and gate count, particularly for two-qubit gates. As the number of qubits increases, the cost of transpilation escalates significantly, especially when trying to find the optimal layout with minimal noise under the qubit connectivity constraints imposed by device topology. In this work, we use the Rivet transpiler, which accelerates transpilation by reusing previously transpiled circuits. This approach is relevant for cases such as quantum chemistry, where multiple Pauli terms need to be measured by appending a series of rotation gates at the end for non-commuting Paulis, and for more complex cases when quantum circuits need to be modified iteratively, as occurs in quantum layerwise learning. We demonstrate up to 600% improvement in transpilation time for quantum layerwise learning using the Rivet transpiler compared to standard transpilation without reuse.

## Citation

Aleksander Kaczmarek, Dikshant Dulal, arXiv:2508.21342v1 (2025-08-29). Retrieved from the arXiv API; the text above is the authors' own abstract, reproduced without alteration.
