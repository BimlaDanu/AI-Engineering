---
title: "QAOA-in-QAOA: solving large-scale MaxCut problems on small quantum machines"
source: "Zeqiao Zhou, Yuxuan Du, Xinmei Tian, Dacheng Tao, arXiv:2205.11762v1 (2022-05-24)"
arxiv: 2205.11762v1
topics: [qaoa, variational, optimisation]
---

# QAOA-in-QAOA: solving large-scale MaxCut problems on small quantum machines

## Abstract

The design of fast algorithms for combinatorial optimization greatly contributes to a plethora of domains such as logistics, finance, and chemistry. Quantum approximate optimization algorithms (QAOAs), which utilize the power of quantum machines and inherit the spirit of adiabatic evolution, are novel approaches to tackle combinatorial problems with potential runtime speedups. However, hurdled by the limited quantum resources nowadays, QAOAs are infeasible to manipulate large-scale problems. To address this issue, here we revisit the MaxCut problem via the divide-and-conquer heuristic: seek the solutions of subgraphs in parallel and then merge these solutions to obtain the global solution. Due to the $\mathbb{Z}_2$ symmetry in MaxCut, we prove that the merging process can be further cast into a new MaxCut problem and thus be addressed by QAOAs or other MaxCut solvers. With this regard, we propose QAOA-in-QAOA ($\text{QAOA}^2$) to solve arbitrary large-scale MaxCut problems using small quantum machines. We also prove that the approximation ratio of $\text{QAOA}^2$ is lower bounded by 1/2. Experiment results illustrate that under different graph settings, $\text{QAOA}^2$ attains a competitive or even better performance over the best known classical algorithms when the node count is around 2000. Our method can be seamlessly embedded into other advanced strategies to enhance the capability of QAOAs in large-scale combinatorial optimization problems.

## Citation

Zeqiao Zhou, Yuxuan Du, Xinmei Tian, Dacheng Tao, arXiv:2205.11762v1 (2022-05-24). Retrieved from the arXiv API; the text above is the authors' own abstract, reproduced without alteration.
