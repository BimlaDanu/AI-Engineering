---
title: "Using gradient-based algorithms to determine ground state energies on a quantum computer"
source: "Tomislav Piskor, Florian G. Eich, Jan-Michael Reiner, Sebastian Zanker, Nicolas Vogt, Michael Marthaler et al., arXiv:2109.08420v1 (2021-09-17)"
arxiv: 2109.08420v1
topics: [shot-budget, estimation, variational]
---

# Using gradient-based algorithms to determine ground state energies on a quantum computer

## Abstract

Variational algorithms are promising candidates to be implemented on near-term quantum computers. The variational quantum eigensolver (VQE) is a prominent example, where a parametrized trial state of the quantum mechanical wave function is optimized to obtain the ground state energy. In our work, we investigate the variational Hamiltonian Ansatz (VHA), where the trial state is given by a non-interacting reference state modified by unitary rotations using generators that are part of the Hamiltonian describing the system. The lowest energy is obtained by optimizing the angles of those unitary rotations. A standard procedure to optimize the variational parameters is to use gradient-based algorithms. However, shot noise and the intrinsic noise of the quantum device affect the evaluation of the required gradients. We studied how different methods for obtaining the gradient, specifically the finite-difference and the parameter-shift rule, are affected by shot noise and noise of the quantum computer. To this end, we simulated a simple quantum circuit, as well as the 2-site and 6-site Hubbard model.

## Citation

Tomislav Piskor, Florian G. Eich, Jan-Michael Reiner, Sebastian Zanker, Nicolas Vogt, Michael Marthaler et al., arXiv:2109.08420v1 (2021-09-17). Retrieved from the arXiv API; the text above is the authors' own abstract, reproduced without alteration.
