---
title: "Robustness of quantum reinforcement learning under hardware errors"
source: "Andrea Skolik, Stefano Mangini, Thomas Bäck, Chiara Macchiavello, Vedran Dunjko, arXiv:2212.09431v1 (2022-12-19)"
arxiv: 2212.09431v1
topics: [shot-budget, estimation, variational]
---

# Robustness of quantum reinforcement learning under hardware errors

## Abstract

Variational quantum machine learning algorithms have become the focus of recent research on how to utilize near-term quantum devices for machine learning tasks. They are considered suitable for this as the circuits that are run can be tailored to the device, and a big part of the computation is delegated to the classical optimizer. It has also been hypothesized that they may be more robust to hardware noise than conventional algorithms due to their hybrid nature. However, the effect of training quantum machine learning models under the influence of hardware-induced noise has not yet been extensively studied. In this work, we address this question for a specific type of learning, namely variational reinforcement learning, by studying its performance in the presence of various noise sources: shot noise, coherent and incoherent errors. We analytically and empirically investigate how the presence of noise during training and evaluation of variational quantum reinforcement learning algorithms affect the performance of the agents and robustness of the learned policies. Furthermore, we provide a method to reduce the number of measurements required to train Q-learning agents, using the inherent structure of the algorithm.

## Citation

Andrea Skolik, Stefano Mangini, Thomas Bäck, Chiara Macchiavello, Vedran Dunjko, arXiv:2212.09431v1 (2022-12-19). Retrieved from the arXiv API; the text above is the authors' own abstract, reproduced without alteration.
