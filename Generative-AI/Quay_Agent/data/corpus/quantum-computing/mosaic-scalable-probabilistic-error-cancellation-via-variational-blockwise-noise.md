---
title: "MoSAIC: Scalable Probabilistic Error Cancellation via Variational Blockwise Noise Aggregation"
source: "Maya Ma, Rimika Jaiswal, Murphy Yuezhen Niu, arXiv:2603.26063v1 (2026-03-27)"
arxiv: 2603.26063v1
topics: [error-mitigation, shot-budget, hardware]
---

# MoSAIC: Scalable Probabilistic Error Cancellation via Variational Blockwise Noise Aggregation

## Abstract

Quantum error mitigation is essential for extracting trustworthy results from noisy intermediate-scale quantum (NISQ) processors. Yet, current approaches face a core scalability bottleneck: unbiased methods such as probabilistic error cancellation (PEC) incur exponential sampling overhead, while approximate techniques like zero-noise extrapolation trade accuracy for efficiency. We introduce and experimentally demonstrate MoSAIC (Modular Spatio-temporal Aggregation for Inverted Channels), a scalable quantum error mitigation framework that preserves the unbiasedness of PEC while dramatically reducing sampling costs. MoSAIC partitions a circuit into noise-aligned blocks, learns an effective block noise model using classical variational optimization, and applies quasi-probabilistic inversion once per block instead of after every layer. This blockwise aggregation reduces both sampling overhead and circuit-depth overhead, enabling mitigation far beyond the operating regime of standard PEC. We also experimentally validate MoSAIC on IBM's 156-qubit Heron processors, performing the largest PEC-based mitigation demonstration on hardware to date. As a physically meaningful benchmark, we prepare the critical one-dimensional transverse-field Ising (TFIM) ground state for system sizes up to 50 qubits. We show that MoSAIC can achieve at least 1 to 2 orders of magnitude better accuracy than standard PEC under identical sampling budgets. This enables MoSAIC to recover accurate observables for larger system sizes, even when standard PEC fails due to its prohibitive sampling overhead. We also present CUDA-Q accelerated simulations to validate performance trends under a range of different noise models. These results demonstrate that MoSAIC is not only theoretically scalable but also practically deployable for high-accuracy, large-scale quantum experiments on today's quantum hardware.

## Citation

Maya Ma, Rimika Jaiswal, Murphy Yuezhen Niu, arXiv:2603.26063v1 (2026-03-27). Retrieved from the arXiv API; the text above is the authors' own abstract, reproduced without alteration.
