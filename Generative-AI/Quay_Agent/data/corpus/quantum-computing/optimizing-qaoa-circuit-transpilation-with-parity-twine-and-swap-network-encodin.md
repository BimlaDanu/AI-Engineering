---
title: "Optimizing QAOA circuit transpilation with parity twine and SWAP network encodings"
source: "J. A. Montanez-Barrera, Yanjun Ji, Michael R. von Spakovsky, David E. Bernal Neira, Kristel Michielsen, arXiv:2505.17944v2 (2025-05-23)"
arxiv: 2505.17944v2
topics: [transpilation, connectivity, hardware]
---

# Optimizing QAOA circuit transpilation with parity twine and SWAP network encodings

## Abstract

Mapping quantum approximate optimization algorithm (QAOA) circuits with non-trivial connectivity in fixed-layout quantum platforms, such as superconducting quantum processing units (QPUs), requires a transpilation process to match the circuit to the hardware layout. This step is critical for reducing error rates on noisy QPUs. Two approaches that improve the resources required for such transpilation are the SWAP network and parity twine chains (PTC), which reduce the two-qubit gate count and circuit depth needed to represent fully connected circuits. In this work, we introduce a simulated annealing-based method that further reduces the encoding overhead of PTC and SWAP networks for QAOA circuits with non-fully connected two-qubit interactions. The method is benchmarked against various transpilers, including the Qiskit SAT mapper, demonstrating that beyond specific connectivity thresholds it achieves significant reductions in both two-qubit gate count and circuit depth. For example, for a 120-qubit QAOA instance with 25% connectivity, our method achieves an 87\% reduction in depth and a 29% reduction in two-qubit gates compared to the Qiskit transpiler. Finally, the practical impact of PTC encoding is validated by benchmarking QAOA on the ibm_fez and ibm_kingston devices, showing improved performance for systems of up to 20 qubits.

## Citation

J. A. Montanez-Barrera, Yanjun Ji, Michael R. von Spakovsky, David E. Bernal Neira, Kristel Michielsen, arXiv:2505.17944v2 (2025-05-23). Retrieved from the arXiv API; the text above is the authors' own abstract, reproduced without alteration.
