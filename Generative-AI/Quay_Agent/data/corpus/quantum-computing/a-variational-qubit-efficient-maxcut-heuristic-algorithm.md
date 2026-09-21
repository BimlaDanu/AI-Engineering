---
title: "A Variational Qubit-Efficient MaxCut Heuristic Algorithm"
source: "Yovav Tene-Cohen, Tomer Kelman, Ohad Lev, Adi Makmal, arXiv:2308.10383v2 (2023-08-20)"
arxiv: 2308.10383v2
topics: [qaoa, variational, optimisation]
---

# A Variational Qubit-Efficient MaxCut Heuristic Algorithm

## Abstract

MaxCut is a key NP-Hard combinatorial optimization graph problem with extensive theoretical and industrial applications, including the Ising model and chip design. While quantum computing offers new solutions for such combinatorial challenges which are potentially better than classical schemes, with the Quantum Approximate Optimization Algorithm (QAOA) being a state-of-the-art example, its performance is currently hindered by hardware noise and limited qubit number. Here, we present a new variational Qubit-Efficient MaxCut (QEMC) algorithm that requires a logarithmic number of qubits with respect to the graph size, an exponential reduction compared to QAOA. We demonstrate cutting-edge performance for graph instances consisting of up to 32 nodes (5 qubits) on real superconducting hardware, and for graphs with up to 2048 nodes (11 qubits) using noiseless simulations, outperforming the established classical algorithm of Goemans and Williamson (GW). The QEMC algorithm's innovative encoding scheme empowers it with great noise-resiliency on the one hand, but also enables its efficient classical simulation on the other, thus obscuring a distinct quantum advantage. Nevertheless, even in the absence of quantum advantage, the QEMC algorithm serves as a potential quantum-inspired algorithm, provides a challenging benchmark for QAOA, and presents a novel encoding paradigm with potential applications extending to other quantum and classical algorithms.

## Citation

Yovav Tene-Cohen, Tomer Kelman, Ohad Lev, Adi Makmal, arXiv:2308.10383v2 (2023-08-20). Retrieved from the arXiv API; the text above is the authors' own abstract, reproduced without alteration.
