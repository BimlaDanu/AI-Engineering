---
title: "Resource-efficient Generalized Quantum Subspace Expansion"
source: "Bo Yang, Nobuyuki Yoshioka, Hiroyuki Harada, Shigeo Hakkaku, Yuuki Tokunaga, Hideaki Hakoshima et al., arXiv:2309.14171v3 (2023-09-25)"
arxiv: 2309.14171v3
topics: [error-mitigation, shot-budget, hardware]
---

# Resource-efficient Generalized Quantum Subspace Expansion

## Abstract

Realizing practical quantum computing requires overcoming a number of computation errors and the limitation of device size, which have intensively been tackled by quantum error mitigation (QEM) these days. As a unified approach of noise-agnostic QEM, generalized quantum subspace expansion (GSE) has lately been proposed to be remarkably robust against stochastic and coherent errors, integrating quantum subspace expansion and virtual state purification. However, the requirement in GSE to perform entangled measurements between copies of the quantum states remains a significant drawback under the current situation of quantum devices with a restricted number of qubits and their connectivity. In this work, we propose ``Dual-GSE'', a resource-efficient implementation of GSE to circumvent this overhead by constructing an ansatz of error-mitigated quantum states via dual-state purification without state copies. Remarkably, the proposed method can further simulate larger quantum systems beyond the size of available quantum hardware, achieved by a suitable ansatz construction inspired by the divide-and-conquer strategy that classically reintroduces the effect of entanglement. While classically forging the entanglement comes with additional cost, the total sampling overhead can be notably reduced by reusing the same Pauli expectation values among divided-and-conquered subsystems. We comprehensively analyze the advantages and overhead of Dual-GSE and perform numerical simulations of the eight-qubit transverse-field Ising model under various setups. Our results demonstrate that Dual-GSE estimates the ground state energy with high accuracy under gate noise with low mitigation overhead and practical sampling cost.

## Citation

Bo Yang, Nobuyuki Yoshioka, Hiroyuki Harada, Shigeo Hakkaku, Yuuki Tokunaga, Hideaki Hakoshima et al., arXiv:2309.14171v3 (2023-09-25). Retrieved from the arXiv API; the text above is the authors' own abstract, reproduced without alteration.
