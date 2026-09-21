---
title: "Beyond Logical Circuits: Hardware-Aware Analysis of Expressibility and Trainability in Variational Quantum Algorithms"
source: "Muhammad Kashif, Muhammad Shafique, arXiv:2605.25552v1 (2026-05-25)"
arxiv: 2605.25552v1
topics: [transpilation, connectivity, hardware]
---

# Beyond Logical Circuits: Hardware-Aware Analysis of Expressibility and Trainability in Variational Quantum Algorithms

## Abstract

Variational quantum algorithms (VQAs) rely on parameterized quantum circuits (PQCs), whose performance is governed by expressibility and trainability. Existing studies typically evaluate these properties at the logical circuit level, implicitly assuming that designed PQCs remain unchanged during hardware execution. In practice, however, hardware-aware transpilation modifies circuit structure through qubit mapping, routing, and basis decomposition, potentially altering PQC behavior. In this paper, we perform a systematic hardware-aware analysis of expressibility and trainability by comparing logical and transpiled PQCs across multiple ansatz families, qubit counts, and circuit depths. Expressibility is measured using fidelity-based KL divergence, while trainability is quantified through gradient variance. Our results show that transpilation acts as an implicit architectural perturbation, producing strongly ansatz-dependent effects. Expressibility deviations exceed upto 125% in some cases, while trainability variations reach up to 25%. Structured ansatzes are generally more robust, whereas highly entangled architectures are more sensitive to transpilation-induced transformations. We further show that transpilation can alter the commonly assumed expressibility-trainability trade-off, demonstrating that logical-level analyses may not reliably predict hardware-level behavior. These findings highlight the importance of hardware-aware evaluation for accurate characterization of VQAs.

## Citation

Muhammad Kashif, Muhammad Shafique, arXiv:2605.25552v1 (2026-05-25). Retrieved from the arXiv API; the text above is the authors' own abstract, reproduced without alteration.
