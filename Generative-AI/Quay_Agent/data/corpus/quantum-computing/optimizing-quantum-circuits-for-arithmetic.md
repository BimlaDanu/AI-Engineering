---
title: "Optimizing Quantum Circuits for Arithmetic"
source: "Thomas Häner, Martin Roetteler, Krysta M. Svore, arXiv:1805.12445v1 (2018-05-31)"
arxiv: 1805.12445v1
topics: [resource-estimation, quantum-advantage, hardware]
---

# Optimizing Quantum Circuits for Arithmetic

## Abstract

Many quantum algorithms make use of oracles which evaluate classical functions on a superposition of inputs. In order to facilitate implementation, testing, and resource estimation of such algorithms, we present quantum circuits for evaluating functions that are often encountered in the quantum algorithm literature. This includes Gaussians, hyperbolic tangent, sine/cosine, inverse square root, arcsine, and exponentials. We use insights from classical high-performance computing in order to optimize our circuits and implement a quantum software stack module which allows to automatically generate circuits for evaluating piecewise smooth functions in the computational basis. Our circuits enable more detailed cost analyses of various quantum algorithms, allowing to identify concrete applications of future quantum computing devices. Furthermore, our resource estimates may guide future research aiming to reduce the costs or even the need for arithmetic in the computational basis altogether.

## Citation

Thomas Häner, Martin Roetteler, Krysta M. Svore, arXiv:1805.12445v1 (2018-05-31). Retrieved from the arXiv API; the text above is the authors' own abstract, reproduced without alteration.
