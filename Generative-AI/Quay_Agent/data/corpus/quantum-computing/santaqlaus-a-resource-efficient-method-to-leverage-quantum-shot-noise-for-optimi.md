---
title: "SantaQlaus: A resource-efficient method to leverage quantum shot-noise for optimization of variational quantum algorithms"
source: "Kosuke Ito, Keisuke Fujii, arXiv:2312.15791v1 (2023-12-25)"
arxiv: 2312.15791v1
topics: [shot-budget, estimation, variational]
---

# SantaQlaus: A resource-efficient method to leverage quantum shot-noise for optimization of variational quantum algorithms

## Abstract

We introduce SantaQlaus, a resource-efficient optimization algorithm tailored for variational quantum algorithms (VQAs), including applications in the variational quantum eigensolver (VQE) and quantum machine learning (QML). Classical optimization strategies for VQAs are often hindered by the complex landscapes of local minima and saddle points. Although some existing quantum-aware optimizers adaptively adjust the number of measurement shots, their primary focus is on maximizing gain per iteration rather than strategically utilizing quantum shot-noise (QSN) to address these challenges. Inspired by the classical Stochastic AnNealing Thermostats with Adaptive momentum (Santa) algorithm, SantaQlaus explicitly leverages inherent QSN for optimization. The algorithm dynamically adjusts the number of quantum measurement shots in an annealing framework: fewer shots are allocated during the early, high-temperature stages for efficient resource utilization and landscape exploration, while more shots are employed later for enhanced precision. Numerical simulations on VQE and QML demonstrate that SantaQlaus outperforms existing optimizers, particularly in mitigating the risks of converging to poor local optima, all while maintaining shot efficiency. This paves the way for efficient and robust training of quantum variational models.

## Citation

Kosuke Ito, Keisuke Fujii, arXiv:2312.15791v1 (2023-12-25). Retrieved from the arXiv API; the text above is the authors' own abstract, reproduced without alteration.
