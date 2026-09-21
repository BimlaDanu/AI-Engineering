---
title: "Virtual Channel Purification"
source: "Zhenhuan Liu, Xingjian Zhang, Yue-Yang Fei, Zhenyu Cai, arXiv:2402.07866v3 (2024-02-12)"
arxiv: 2402.07866v3
topics: [error-mitigation, shot-budget, hardware]
---

# Virtual Channel Purification

## Abstract

Quantum error mitigation is a key approach for extracting target state properties on state-of-the-art noisy machines and early fault-tolerant devices. Using the ideas from flag fault tolerance and virtual state purification, we develop the virtual channel purification (VCP) protocol, which consumes similar qubit and gate resources as virtual state purification but offers stronger error suppression with increased system size and more noisy operation copies. The application of VCP does not require specific knowledge about the target quantum state, the target problem and the gate noise model in the target circuit, and can still offer rigorous performance guarantees for practical noise regimes as long as the noise is incoherent. Further connections are made between VCP and quantum error correction to produce the virtual error correction (VEC) protocol, one of the first protocols that combine quantum error correction (QEC) and quantum error mitigation beyond directly applying error mitigation protocols on top of logical qubits. Assuming perfect syndrome extraction, VEC can virtually remove all correctable noise in the channel while paying only the same sampling cost as low-order purification. It can achieve QEC-level protection on an unencoded register when transmitting it through a noisy channel, removing the associated encoding qubit overhead. Another variant of VEC can mimic the error suppression power of the surface code by inputting only a bit-flip and a phase-flip code. Our protocol can also be adapted to key tasks in quantum networks like channel capacity activation and entanglement distribution.

## Citation

Zhenhuan Liu, Xingjian Zhang, Yue-Yang Fei, Zhenyu Cai, arXiv:2402.07866v3 (2024-02-12). Retrieved from the arXiv API; the text above is the authors' own abstract, reproduced without alteration.
