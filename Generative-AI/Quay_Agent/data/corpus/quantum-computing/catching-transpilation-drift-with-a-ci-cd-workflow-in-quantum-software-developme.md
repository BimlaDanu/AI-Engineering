---
title: "Catching Transpilation Drift with a CI/CD Workflow in Quantum Software Development"
source: "Jouni Peltonen, Otso Kinanen, Vlad Stirbu, arXiv:2608.08248v1 (2026-08-08)"
arxiv: 2608.08248v1
topics: [transpilation, connectivity, hardware]
---

# Catching Transpilation Drift with a CI/CD Workflow in Quantum Software Development

## Abstract

Quantum software workflows rely on compiler and provider toolchains that evolve independently of application source code. Consequently, an unchanged quantum circuit may transpile into a different target-specific realization after changes in SDK versions, optimization settings, basis gates, coupling maps, or backend descriptions. Such transpilation drift can affect circuit depth, gate composition, qubit mapping, and execution behavior, yet it is rarely monitored in CI/CD pipelines. This paper proposes a Quantum DevOps workflow for detecting transpilation drift before execution. The workflow transpiles source circuits against configured target profiles, computes structural drift metrics, records provenance and artifacts in MLflow, and raises configurable warnings or failures in GitHub Actions. Using representative circuits and target profiles, we show how drift checks can expose toolchain-induced changes and support reproducibility audits. The contribution is a practical CI/CD guardrail for making quantum compilation behavior observable, testable, and auditable.

## Citation

Jouni Peltonen, Otso Kinanen, Vlad Stirbu, arXiv:2608.08248v1 (2026-08-08). Retrieved from the arXiv API; the text above is the authors' own abstract, reproduced without alteration.
