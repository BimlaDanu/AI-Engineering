r"""The classical baseline the quantum answer has to beat.

A feasibility verdict is only as honest as the alternative it was measured
against. If the classical arm is a strawman -- mean field, or a two-site
cluster -- then "the quantum circuit wins" says nothing, because a competent
classical practitioner was never in the room.

So the baseline here is the variational imaginary-time ansatz (VITA), evaluated
by Monte Carlo on the classical Ising model dual to it. It is chosen for one
reason above all others: **it is the same variational family as the quantum
ansatz.** VITA prepares

.. math::

    |\psi_P\rangle \propto \prod_{p=P}^{1}
        e^{-\beta_p \hat H_{\rm field}} e^{-\alpha_p \hat H_{\rm diag}} |+\rangle^{\otimes L},

and QAOA prepares the same product with imaginary angles,
:math:`e^{-i\gamma_p \hat H_{\rm diag}}`. Same layers, same alternation, same depth
parameter; one runs on a sampler, the other on a device. Both are driven by
stochastic reconfiguration -- :math:`S\,\delta\theta = -F` -- with the metric
:math:`S` estimated from the covariance of the log-derivatives here and from the
Fubini-Study metric in
:func:`src.physics.quantum.imaginary_time_evolution.fubini_study_metric`.

That correspondence is what makes the comparison sharp. "Is the quantum version
worth it?" stops being a comparison between two unrelated algorithms and becomes
a question about one algorithm and two ways of evaluating it.

This package may not see an exact answer. It is the agent's opponent, not
the grader: a baseline that could read :mod:`src.physics.reference.free_fermions` would report the
true ground-state energy as its own result and the whole comparison would
collapse. The architecture test enforces this alongside the agent
packages.
"""
