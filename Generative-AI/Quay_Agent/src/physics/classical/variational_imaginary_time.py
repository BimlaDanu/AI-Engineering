r"""VITA: the variational imaginary-time ansatz, optimised by sampling.

This is the classical arm of the project's central comparison, and the number a
quantum feasibility verdict has to be issued against. It answers one question --
*how well can a competent classical method do on this chain, and at what cost?*
-- and it answers it with the same variational family the quantum side uses, so
that the comparison is about the machine and not about the algorithm.

The loop
--------
At fixed parameters :math:`\theta = (\alpha, \beta)`, a Markov chain on the dual
classical lattice estimates three things at once: the energy, its gradient, and
the metric

.. math::

    S_{jk} = \langle O_j O_k \rangle - \langle O_j \rangle \langle O_k \rangle,
    \qquad O_j = \partial_j \ln \psi ,

which is the Fisher information of the variational manifold. Parameters are then
moved along the *natural* gradient

.. math::

    \theta \leftarrow \theta - \eta\, S^{-1} \nabla_\theta \langle \hat H \rangle ,

which is stochastic reconfiguration. A plain gradient step is badly conditioned
here, because :math:`\alpha` and :math:`\beta` move the state by very different
amounts for the same change in the number; :math:`S` is exactly the object that
knows the difference.

This is the same update the quantum side performs in
:mod:`src.physics.quantum.imaginary_time_evolution`, where :math:`S` is the
Fubini-Study metric measured on a device instead of the covariance of
log-derivatives measured on a sampler.
Two estimators of one geometry.

Two honesty rules, learned the hard way
---------------------------------------
Both are enforced in :func:`ground_state_energy` rather than left to the caller,
because both are silent when violated.

1. Never report the lowest sampled energy. Selecting the iteration with the
   smallest sampled value systematically picks a downward fluctuation, and can
   report an energy *below* the true ground state -- impossible for a
   variational state, and a defect this exact code has produced before. The
   parameters are tail-averaged instead, which is unbiased, and then measured
   afresh.
2. The reported energy comes from a measurement that took no part in choosing
   the parameters. Re-using the optimisation's own samples would
   correlate the estimate with the selection, which is the same bias wearing a
   different hat.

What it does not do
-------------------
It does not know the exact answer, and must not: this package is the agent's
opponent, not the grader. A baseline able to import :mod:`src.physics.reference.free_fermions`
would return the true ground-state energy as its own result and the entire
comparison would be vacuous. The architecture test enforces that.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.physics.classical.dual_lattice import (
    FloatArray,
    Lattice,
    build_couplings,
    clip_parameters,
    total_imaginary_time,
)
from src.physics.classical.metropolis import Sampler, log_derivatives
from src.physics.classical.regime import BaselineUnavailableError, Regime, assess
from src.physics.lattice import Lattice as SpatialLattice
from src.physics.model import TFIMSpec


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """How much Monte Carlo goes into a single energy and gradient estimate.

    Attributes:
        warmup: Sweeps discarded before measuring, so the chain forgets where it
            started. Paid in full on the first call of an optimisation and then
            largely wasted on later ones, since the chain is carried over --
            which is why it can be modest.
        sweeps: Sweeps per bin. One measurement is taken at the end of each.
        bins: Number of bins. Bin means are treated as independent, and it is
            their spread that becomes the error bar. This is an assumption, not
            a measurement: autocorrelation is absorbed rather than estimated, so
            a short bin under-reports the error rather than over-reporting it.
    """

    warmup: int = 200
    sweeps: int = 20
    bins: int = 20

    def __post_init__(self) -> None:
        """Reject settings that cannot produce a usable estimate.

        Raises:
            ValueError: If any count is negative, or no measurement is taken.
        """
        if self.warmup < 0:
            raise ValueError(f"warmup must be non-negative, got {self.warmup}")
        if self.sweeps < 1 or self.bins < 1:
            raise ValueError(f"need at least one sweep and one bin, got {self.sweeps}/{self.bins}")

    @property
    def n_samples(self) -> int:
        """Total number of measurements, ``sweeps * bins``."""
        return self.sweeps * self.bins


@dataclass(frozen=True, slots=True)
class SampleStatistics:
    r"""One Monte Carlo estimate at fixed variational parameters.

    Energy and magnetisation are reported **per spin** so that different chain
    lengths can be compared directly; the gradient and the metric are left
    extensive, because that is the scale the optimiser steps in.

    Attributes:
        energy: Mean energy per spin.
        energy_error: Standard error of that mean, from the spread of the bin
            means. ``nan`` when there is only one bin, which is honest: a single
            bin carries no information about its own error.
        magnetisation: Mean ``|magnetisation|`` per spin, the order parameter.
        magnetisation_error: Standard error of the same.
        gradient: Estimate of :math:`\nabla_\theta \langle \hat H \rangle`.
        metric: The stochastic-reconfiguration matrix :math:`S`.
    """

    energy: float
    energy_error: float
    magnetisation: float
    magnetisation_error: float
    gradient: FloatArray
    metric: FloatArray


def estimate(sampler: Sampler, theta: FloatArray, config: SamplingConfig) -> SampleStatistics:
    """Estimate the energy, gradient and metric at one set of parameters.

    Args:
        sampler: Markov chain to draw from. It is advanced in place and left
            equilibrated, so that the next call along an optimisation starts
            from a configuration already close to its new distribution.
        theta: The variational parameters to measure at.
        config: Warm-up length, sweeps per bin, and number of bins.

    Returns:
        The sampled energy and its error, the order parameter, the gradient of
        the energy, and the metric.
    """
    couplings = build_couplings(theta, sampler.field_ratio)
    sampler.warmup(couplings, config.warmup)

    n_parameters = couplings.n_parameters
    energies = np.empty(config.n_samples)
    magnetisations = np.empty(config.n_samples)
    classical_gradients = np.empty((config.n_samples, n_parameters))
    estimator_gradients = np.empty_like(classical_gradients)

    for index in range(config.n_samples):
        sampler.sweep(couplings)
        measurement = sampler.measure(couplings)
        energies[index] = measurement.energy
        magnetisations[index] = measurement.magnetisation
        classical_gradients[index] = measurement.classical_gradient
        estimator_gradients[index] = measurement.estimator_gradient

    mean_energy = float(np.mean(energies))
    mean_classical = classical_gradients.mean(axis=0)
    gradient = (
        estimator_gradients.mean(axis=0)
        - (energies[:, None] * classical_gradients).mean(axis=0)
        + mean_energy * mean_classical
    )

    centred = log_derivatives(classical_gradients - mean_classical)
    metric = (centred.T @ centred) / config.n_samples

    n_sites = sampler.lattice.n_sites
    return SampleStatistics(
        energy=mean_energy / n_sites,
        energy_error=_bin_error(energies, config) / n_sites,
        magnetisation=float(np.mean(magnetisations)) / n_sites,
        magnetisation_error=_bin_error(magnetisations, config) / n_sites,
        gradient=gradient,
        metric=metric,
    )


def natural_gradient(
    gradient: FloatArray, metric: FloatArray, diagonal_shift: float = 1e-3
) -> FloatArray:
    r"""Solve :math:`S x = \nabla E` for the preconditioned step direction.

    :math:`S` is positive semi-definite by construction but is frequently
    singular in practice, because two adjacent pulses can become redundant and
    the manifold loses a direction. A diagonal shift proportional to the largest
    diagonal entry keeps the solve well posed, and the pseudo-inverse absorbs
    whatever singularity survives it -- returning the minimum-norm solution,
    which is the right thing to do with a direction the state cannot move in.

    Args:
        gradient: The sampled energy gradient.
        metric: The sampled metric :math:`S`.
        diagonal_shift: Regularisation, relative to the scale of ``metric``.

    Returns:
        The step direction, before any learning rate is applied.
    """
    scale = float(np.max(np.diag(metric))) if metric.size else 0.0
    regularised = metric + diagonal_shift * max(scale, 1.0) * np.eye(metric.shape[0])
    return np.linalg.pinv(regularised) @ gradient


@dataclass(frozen=True, slots=True)
class OptimiserConfig:
    r"""Settings of the stochastic-reconfiguration loop.

    Attributes:
        iterations: Number of parameter updates.
        learning_rate: Initial step size.
        decay_factor: Multiplier applied to the step size every
            ``decay_every`` iterations, so late steps are small enough to
            average out sampling noise rather than chase it.
        decay_every: How often the decay is applied.
        diagonal_shift: Regularisation of the metric before inversion.
        normalise_step: Rescale the update to unit norm before applying the
            learning rate. This makes the step length independent of the
            gradient's (noisy) magnitude, which is what stops a short chain
            from taking a wild step early on.
        min_parameter: Floor on every imaginary time. This is a **conditioning
            choice, not physics**: the transverse-field estimator is
            :math:`e^{-2 J_\tau(m) s s'}` with :math:`J_\tau = \frac12 \ln
            \coth\beta`, so as :math:`\beta \to 0` its two possible values
            separate without bound and its variance diverges. The default is far
            from any optimum found so far, so it does not bias the result -- but
            if a genuinely small optimal :math:`\beta` is ever expected, lower
            it and raise the statistics together, and check that the sampled
            energy stays above the ground state.
        tail_fraction: Fraction of the run averaged to fix the final
            parameters. See rule 1 in the module docstring.
    """

    iterations: int = 60
    learning_rate: float = 0.1
    decay_factor: float = 0.45
    decay_every: int = 10
    diagonal_shift: float = 1e-3
    normalise_step: bool = True
    min_parameter: float = 0.05
    tail_fraction: float = 0.25

    def step_size(self, iteration: int) -> float:
        """Return the learning rate in force at a given iteration."""
        return self.learning_rate * self.decay_factor ** (iteration // self.decay_every)


@dataclass
class OptimisationHistory:
    """Trace of an optimisation, one entry per iteration.

    Kept in full rather than reduced on the fly, because the shape of the
    descent is what tells a reader whether the run converged or merely stopped.

    Attributes:
        theta: Parameters at the start of each iteration.
        energy: Sampled energy per spin at those parameters.
        energy_error: Its statistical error.
        magnetisation: Sampled order parameter per spin.
    """

    theta: list[FloatArray] = field(default_factory=list)
    energy: list[float] = field(default_factory=list)
    energy_error: list[float] = field(default_factory=list)
    magnetisation: list[float] = field(default_factory=list)

    def record(self, theta: FloatArray, statistics: SampleStatistics) -> None:
        """Append one iteration to the trace."""
        self.theta.append(np.array(theta, copy=True))
        self.energy.append(statistics.energy)
        self.energy_error.append(statistics.energy_error)
        self.magnetisation.append(statistics.magnetisation)

    def converged_theta(self, tail_fraction: float = 0.25) -> FloatArray:
        """Average the parameters over the tail of the run.

        Once the optimisation has converged the parameters fluctuate about their
        optimum, so averaging the tail removes that noise. Unlike picking the
        iteration with the lowest sampled energy, it is unbiased -- selecting on
        a noisy energy systematically chooses a downward fluctuation and reports
        a variational energy below the true ground state, which is impossible
        and has happened.

        Args:
            tail_fraction: Fraction of iterations to average over.

        Returns:
            The averaged parameter vector.

        Raises:
            ValueError: If nothing has been recorded yet.
        """
        if not self.theta:
            raise ValueError("history is empty: nothing has been recorded")
        tail = max(1, int(tail_fraction * len(self.theta)))
        return np.asarray(np.mean(self.theta[-tail:], axis=0), dtype=float)


@dataclass(frozen=True, slots=True)
class VitaResult:
    """The classical baseline's answer, in the form the verdict needs it.

    Attributes:
        energy: Variational energy per spin, from a measurement taken *after*
            the parameters were fixed and taking no part in fixing them.
        energy_error: Statistical error of that energy, per spin. The baseline
            is stochastic, so it is a number with an error bar and has to be
            compared as one; a verdict that ignores the bar is not a verdict.
        theta: The converged parameters ``[alpha_1..alpha_P, beta_1..beta_P]``.
        depth: The ansatz depth ``P``, which is what a quantum circuit would
            have to match layer for layer.
        imaginary_time: Total imaginary time propagated. How this grows with
            chain length is the substantive claim the ansatz makes, and it is
            the classical counterpart of circuit depth.
        magnetisation: Sampled order parameter per spin.
        n_measurements: Total single-configuration measurements spent. The
            classical counterpart of a shot count, and the number that belongs
            on the other side of the cost comparison.
        regime: How far this number may be trusted, and why. Carried on the
            result rather than left for the caller to look up, because a
            degraded baseline fails in the direction that flatters the quantum
            arm -- an under-reported error bar is a lowered bar for claiming a
            lead -- so the qualification has to travel with the number it
            qualifies. See :mod:`src.physics.classical.regime`.
    """

    energy: float
    energy_error: float
    theta: FloatArray
    depth: int
    imaginary_time: float
    magnetisation: float
    n_measurements: int
    regime: Regime


def optimise(
    spec: TFIMSpec,
    depth: int = 1,
    longitudinal_field: float = 0.0,
    theta: FloatArray | None = None,
    sampling: SamplingConfig | None = None,
    optimiser: OptimiserConfig | None = None,
    seed: int = 0,
    geometry: SpatialLattice | None = None,
) -> tuple[OptimisationHistory, Sampler]:
    """Run the stochastic-reconfiguration loop on one Markov chain.

    Args:
        spec: The chain to solve.
        depth: Ansatz depth ``P``, giving ``2P`` parameters and ``2P + 1``
            imaginary-time slices. ``P = 1`` converges readily; ``P >= 2``
            needs substantially more statistics, because the gradient of a
            nearly redundant pulse is small next to the sampling noise.
        longitudinal_field: The ``g`` of the project's Hamiltonian. Not carried
            by :class:`~src.physics.model.TFIMSpec`, and so passed separately.
        theta: Initial parameters. Drawn at random if omitted.
        sampling: Monte Carlo settings; defaults if omitted.
        optimiser: Step-size settings; defaults if omitted.
        seed: Seeds both the initial parameters and the Markov chain.
        geometry: The physical shape to solve on. Defaults to the chain
            :class:`~src.physics.model.TFIMSpec` describes, which is the only
            shape this sampler implements -- see the refusal below.

    Returns:
        The trace of the run and the equilibrated sampler, so that the caller
        can take a fresh, uncorrelated measurement at the converged parameters
        without paying the warm-up again.

    Raises:
        ValueError: If ``depth`` is not positive.
        BaselineUnavailableError: If the shape asked for is one this sampler cannot
            represent. The refusal is deliberate and its reasoning is in
            :mod:`src.physics.classical.regime`; the short version is that the dual
            lattice below is built from ``spec.n_sites`` alone, so without the check
            a two-dimensional lattice would be sampled as a *line* of the same
            number of sites and reported as the classical result for the lattice.
    """
    if depth < 1:
        raise ValueError(f"depth P must be at least 1, got {depth}")
    shape = geometry or SpatialLattice("chain", 1, spec.n_sites, spec.boundary)
    standing = assess(shape, spec.coupling, spec.field)
    if not standing.can_run:
        raise BaselineUnavailableError(standing)
    sampling = sampling or SamplingConfig()
    optimiser = optimiser or OptimiserConfig()
    rng = np.random.default_rng(seed)

    start = rng.uniform(0.1, 1.0, size=2 * depth) if theta is None else np.asarray(theta, float)
    current = clip_parameters(start)

    sampler = Sampler(
        lattice=Lattice(spec.n_sites, 2 * depth + 1, spec.boundary),
        transverse_field=spec.field,
        coupling=spec.coupling,
        longitudinal_field=longitudinal_field,
        rng=rng,
        geometry=shape,
    )

    history = OptimisationHistory()
    for iteration in range(optimiser.iterations):
        statistics = estimate(sampler, current, sampling)
        history.record(current, statistics)

        step = natural_gradient(statistics.gradient, statistics.metric, optimiser.diagonal_shift)
        if optimiser.normalise_step:
            norm = float(np.linalg.norm(step))
            if norm > 0.0:
                step = step / norm
        current = np.maximum(
            clip_parameters(current - optimiser.step_size(iteration) * step),
            optimiser.min_parameter,
        )

    return history, sampler


def ground_state_energy(
    spec: TFIMSpec,
    depth: int = 1,
    longitudinal_field: float = 0.0,
    sampling: SamplingConfig | None = None,
    optimiser: OptimiserConfig | None = None,
    final_sampling: SamplingConfig | None = None,
    seed: int = 0,
    geometry: SpatialLattice | None = None,
) -> VitaResult:
    """Run the classical baseline end to end and report its answer.

    Optimises the ansatz, fixes the parameters by tail-averaging the trace, and
    then measures once more at those parameters with fresh statistics. Both of
    those choices are bias controls rather than conveniences; see the module
    docstring.

    The result is an upper bound on the ground-state energy with an error bar
    attached, and it is deliberately never compared here against the exact
    answer -- this module cannot see it.

    Args:
        spec: The chain to solve.
        depth: Ansatz depth ``P``.
        longitudinal_field: The ``g`` of the project's Hamiltonian.
        sampling: Statistics used during the optimisation.
        optimiser: Step-size settings.
        final_sampling: Statistics for the final measurement. Defaults to four
            times the optimisation's bins, because this is the one number that
            gets reported and it is cheap relative to the loop that preceded it.
        seed: Seeds the run.
        geometry: The physical shape to solve on. Defaults to the chain the spec
            describes.

    Returns:
        The variational energy per spin, its error, what was spent to get it,
        and -- on the same object -- how far it may be trusted. The standing
        travels with the number because the two are read by different code in
        different places: :func:`src.agent.verdict.judge` needs it to know
        whether it may be confident, and
        :func:`src.agent.report._classical_section` needs it to say what could
        not be checked.

    Raises:
        BaselineUnavailableError: If the shape is one this sampler cannot represent.
            See :func:`optimise`.
    """
    sampling = sampling or SamplingConfig()
    optimiser = optimiser or OptimiserConfig()
    history, sampler = optimise(
        spec,
        depth=depth,
        longitudinal_field=longitudinal_field,
        sampling=sampling,
        optimiser=optimiser,
        seed=seed,
        geometry=geometry,
    )

    final_sampling = final_sampling or SamplingConfig(
        warmup=sampling.warmup, sweeps=sampling.sweeps, bins=4 * sampling.bins
    )
    converged = np.maximum(
        history.converged_theta(optimiser.tail_fraction), optimiser.min_parameter
    )
    final = estimate(sampler, converged, final_sampling)

    return VitaResult(
        energy=final.energy,
        energy_error=final.energy_error,
        theta=converged,
        depth=depth,
        imaginary_time=total_imaginary_time(converged),
        magnetisation=final.magnetisation,
        n_measurements=optimiser.iterations * sampling.n_samples + final_sampling.n_samples,
        regime=sampler.regime,
    )


def _bin_error(values: FloatArray, config: SamplingConfig) -> float:
    """Standard error of the mean, from the spread of the bin means.

    Returns ``nan`` for a single bin rather than zero. A single bin genuinely
    carries no information about its own error, and reporting ``0.0`` would let
    a caller divide by it and call the result significant.
    """
    if config.bins < 2:
        return float("nan")
    bin_means = values.reshape(config.bins, config.sweeps).mean(axis=1)
    return float(np.std(bin_means, ddof=1) / np.sqrt(config.bins))
