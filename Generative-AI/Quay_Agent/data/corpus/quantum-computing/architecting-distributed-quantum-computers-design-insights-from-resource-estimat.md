---
title: "Architecting Distributed Quantum Computers: Design Insights from Resource Estimation"
source: "Dmitry Filippov, Peter Yang, Prakash Murali, arXiv:2508.19160v2 (2025-08-26)"
arxiv: 2508.19160v2
topics: [resource-estimation, quantum-advantage, hardware]
---

# Architecting Distributed Quantum Computers: Design Insights from Resource Estimation

## Abstract

In the emerging field of Fault Tolerant Quantum Computation (FTQC), resource estimation is an important tool for quantitatively comparing prospective architectures, identifying hardware bottlenecks and informing which research paths are most valuable. Despite a recent increase in attention on FTQC, there is currently a lack of resource estimation research for architectures that can realistically offer quantum advantage. In particular, current modelling efforts focus on monolithic quantum computers where all qubits reside on a single device. Constraints on fabrication yield, wiring density, and cooling power make monolithic devices unlikely to scale to fault-tolerant sizes in the foreseeable future. Distributed quantum supercomputers offer a path to overcome these limitations. We propose a prospective distributed quantum computing architecture based on lattice surgery with support for modular and distributed operations, with a focus on superconducting qubits. We develop a resource-estimation framework and software tool tailored to distributed FTQC, enabling end-to-end analysis of practical quantum algorithms on our proposed architecture with various hardware configurations, spanning different node sizes, inter-node entanglement generation rates and distillation protocols. Our extensive benchmarking across eight applications and thousands of hardware configurations, shows that resource estimation driven architecture design is crucial for scalability. We provide concrete design configurations that have feasible resource requirements, recommendations for hardware design and system organization. More broadly, our work provides a rigorous methodology for architectural pathfinding, capable of informing system designs and guiding future research priorities.

## Citation

Dmitry Filippov, Peter Yang, Prakash Murali, arXiv:2508.19160v2 (2025-08-26). Retrieved from the arXiv API; the text above is the authors' own abstract, reproduced without alteration.
