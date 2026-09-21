r"""The classical baseline, held against dense linear algebra it shares nothing with.

The Monte Carlo code draws random numbers on a classical Ising lattice and never
forms a matrix; the reference side here builds :math:`2^L \times 2^L` Kronecker
products of Pauli matrices, exponentiates them, and never draws a random number.
The only thing the two share is the *claim* that they describe the same
variational state. Agreement is therefore evidence about the quantum-to-classical
mapping itself, which is the part of this module that could be subtly wrong
while every self-consistency check inside it still passed.

The Pauli matrices here are written out by hand rather than imported from
``src.physics.quantum.hamiltonians``, deliberately: a cross-check that reuses the code
it is checking tests arithmetic, not physics.

Sampling can only ever be checked statistically, so comparisons are made in
units of the quoted error bar. A systematic mistake -- a sign, a boundary
condition, a slice index off by one -- shifts the sampled energy by a small
constant that looks perfectly plausible in isolation and only shows up against a
number computed another way.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.linalg import expm
from scipy.optimize import OptimizeResult, minimize

from src.physics.classical import variational_imaginary_time as vita
from src.physics.classical.dual_lattice import (
    Lattice,
    build_couplings,
    classical_energy,
    clip_parameters,
    join_parameters,
    local_field,
    split_parameters,
    total_imaginary_time,
)
from src.physics.classical.metropolis import Sampler, log_derivatives
from src.physics.model import DEFAULT_SITES, TFIMSpec
from src.physics.reference import free_fermions

FloatMatrix = NDArray[np.float64]

SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]])
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]])
IDENTITY = np.eye(2)

#: How many error bars a correct estimator may stray from an independent answer.
TOLERANCE_SIGMA = 4.0

#: Enough statistics to resolve a systematic error, few enough to stay quick.
HEAVY = vita.SamplingConfig(warmup=1500, sweeps=200, bins=60)


# --------------------------------------------------------------------------
# The independent reference: dense Pauli algebra, no sampling anywhere.
# --------------------------------------------------------------------------
def operator_on(site: int, matrix: FloatMatrix, n_sites: int) -> FloatMatrix:
    """Embed a single-site operator into the full 2**L space."""
    factors = [matrix if index == site else IDENTITY for index in range(n_sites)]
    out = np.asarray(factors[0], dtype=np.float64)
    for factor in factors[1:]:
        out = np.asarray(np.kron(out, factor), dtype=np.float64)
    return out


def bonds_of(n_sites: int, boundary: str) -> list[tuple[int, int]]:
    """Nearest-neighbour bonds of the chain."""
    pairs = [(i, i + 1) for i in range(n_sites - 1)]
    if boundary == "periodic":
        pairs.append((n_sites - 1, 0))
    return pairs


@cache
def diagonal_generator(n_sites: int, boundary: str, field_ratio: float) -> FloatMatrix:
    r"""The ansatz generator :math:`\hat H_{\rm diag}`, with ``J`` divided out."""
    total = np.zeros((2**n_sites, 2**n_sites))
    for left, right in bonds_of(n_sites, boundary):
        total -= operator_on(left, SIGMA_Z, n_sites) @ operator_on(right, SIGMA_Z, n_sites)
    for site in range(n_sites):
        total -= field_ratio * operator_on(site, SIGMA_Z, n_sites)
    return total


@cache
def field_generator(n_sites: int) -> FloatMatrix:
    r"""The ansatz generator :math:`\hat H_{\rm field} = -\sum_i \hat\sigma^x_i`."""
    total = np.zeros((2**n_sites, 2**n_sites))
    for site in range(n_sites):
        total -= operator_on(site, SIGMA_X, n_sites)
    return total


def hamiltonian_of(spec: TFIMSpec, longitudinal_field: float) -> FloatMatrix:
    r"""The project's Hamiltonian as a dense matrix, longitudinal term included."""
    return spec.coupling * diagonal_generator(
        spec.n_sites, spec.boundary, longitudinal_field / spec.coupling
    ) + spec.field * field_generator(spec.n_sites)


def vita_state(spec: TFIMSpec, theta: FloatMatrix, longitudinal_field: float) -> FloatMatrix:
    r"""Build :math:`|\psi_P(\alpha,\beta)\rangle` by dense matrix exponentials.

    The pulses are applied innermost first, which is what
    :math:`\prod_{p=P}^{1} e^{-\beta_p \hat H_{\rm field}} e^{-\alpha_p \hat
    H_{\rm diag}}` means read right to left. Getting that order backwards is
    invisible at depth one and wrong at every greater depth, so the depth-two
    cases in this file are the ones that pin it.
    """
    alpha, beta = split_parameters(np.asarray(theta, dtype=float))
    diagonal = diagonal_generator(spec.n_sites, spec.boundary, longitudinal_field / spec.coupling)
    transverse = field_generator(spec.n_sites)

    state = np.ones(2**spec.n_sites) / np.sqrt(2**spec.n_sites)  # |+>^{otimes L}
    for pulse in range(alpha.size):
        state = expm(-alpha[pulse] * diagonal) @ state
        state = expm(-beta[pulse] * transverse) @ state
    return state / np.linalg.norm(state)


def variational_energy_per_spin(
    spec: TFIMSpec, theta: FloatMatrix, longitudinal_field: float = 0.0
) -> float:
    """Exact energy per spin of the VITA state, with no sampling involved."""
    state = vita_state(spec, theta, longitudinal_field)
    return float(state @ hamiltonian_of(spec, longitudinal_field) @ state) / spec.n_sites


@cache
def dense_optimum(spec: TFIMSpec) -> OptimizeResult:
    """Best depth-one parameters, found by dense algebra with no sampling at all.

    Cached because it is the expensive half of every comparison and does not
    depend on the seed.
    """
    return minimize(
        lambda theta: variational_energy_per_spin(spec, np.abs(theta)),
        x0=np.array([0.5, 0.3]),
        method="Nelder-Mead",
        options={"xatol": 1e-7, "fatol": 1e-13},
    )


def sampled(
    spec: TFIMSpec,
    theta: list[float],
    longitudinal_field: float = 0.0,
    seed: int = 0,
    config: vita.SamplingConfig = HEAVY,
) -> vita.SampleStatistics:
    """Sample a fixed variational state of the given chain."""
    depth = len(theta) // 2
    sampler = Sampler(
        lattice=Lattice(spec.n_sites, 2 * depth + 1, spec.boundary),
        transverse_field=spec.field,
        coupling=spec.coupling,
        longitudinal_field=longitudinal_field,
        rng=np.random.default_rng(seed),
    )
    return vita.estimate(sampler, np.array(theta), config)


# --------------------------------------------------------------------------
# The coupling map
# --------------------------------------------------------------------------
def test_the_coupling_map_mirrors_about_the_operator_slice() -> None:
    couplings = build_couplings(np.array([0.3, 0.7, 0.4, 0.9]))
    assert couplings.n_slices == 5
    assert couplings.middle == 2
    # Spatial couplings mirror, and the middle slice carries no diagonal factor.
    assert couplings.spatial.tolist() == [0.3, 0.7, 0.0, 0.7, 0.3]
    # Temporal couplings mirror about the middle *bond*, which is one index
    # earlier than the middle slice: 2P + 1 slices are joined by 2P bonds.
    b_one, b_two = 0.5 * np.log(1.0 / np.tanh(np.array([0.4, 0.9])))
    assert np.allclose(couplings.temporal[:4], [b_one, b_two, b_two, b_one])
    # The time direction is open: the last bond does not exist.
    assert couplings.temporal[-1] == 0.0
    assert couplings.temporal_derivative[-1] == 0.0


def test_the_temporal_coupling_derivative_matches_finite_differences() -> None:
    beta = np.array([0.4, 0.9])
    step = 1e-6
    for index, value in enumerate(beta):
        shifted_up = build_couplings(
            join_parameters(np.ones(2), _replace(beta, index, value + step))
        )
        shifted_down = build_couplings(
            join_parameters(np.ones(2), _replace(beta, index, value - step))
        )
        numerical = (shifted_up.temporal[index] - shifted_down.temporal[index]) / (2 * step)
        analytic = build_couplings(join_parameters(np.ones(2), beta)).temporal_derivative[index]
        assert abs(numerical - analytic) < 1e-5


def _replace(values: FloatMatrix, index: int, value: float) -> FloatMatrix:
    out = np.array(values, dtype=float, copy=True)
    out[index] = value
    return out


def test_the_longitudinal_coupling_follows_the_physical_ratio() -> None:
    couplings = build_couplings(np.array([0.3, 0.7, 0.4, 0.9]), field_ratio=0.25)
    assert np.allclose(couplings.longitudinal, 0.25 * couplings.spatial)
    assert couplings.has_longitudinal_field
    assert not build_couplings(np.array([0.3, 0.4])).has_longitudinal_field


def test_parameters_are_reflected_into_the_physical_domain() -> None:
    assert np.all(clip_parameters(np.array([-0.5, 0.0, 0.3])) > 0.0)
    assert clip_parameters(np.array([-0.5]))[0] == pytest.approx(0.5)
    assert total_imaginary_time(np.array([0.6, 0.2, 0.4, 0.8])) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# The lattice and the sweep
# --------------------------------------------------------------------------
def test_no_two_sites_of_one_sublattice_share_a_bond() -> None:
    """The whole vectorised sweep is invalid if this fails, and silently so."""
    lattice = Lattice(8, 5, "periodic")
    mask = lattice.sublattice_mask(0)
    # Space wraps, so the ring is checked with a roll; imaginary time is open,
    # so slice 0 and slice 2P are not neighbours and must not be compared.
    assert not np.any(mask & np.roll(mask, 1, axis=0))
    assert not np.any(mask[:, :-1] & mask[:, 1:])
    # Every spin belongs to exactly one sublattice, or the sweep misses some.
    assert np.all(mask ^ lattice.sublattice_mask(1))


def test_a_periodic_chain_of_odd_length_is_refused_rather_than_sampled_wrongly() -> None:
    with pytest.raises(ValueError, match="even length"):
        Lattice(7, 3, "periodic")
    Lattice(7, 3, "open")  # the same length is fine once the ring is cut


@pytest.mark.parametrize("field_ratio", [0.0, 0.35])
def test_flipping_one_spin_costs_exactly_twice_the_local_field(field_ratio: float) -> None:
    """The one identity the Metropolis acceptance rests on.

    ``local_field`` counts bonds from both ends but a site field only once. If
    those two conventions are conflated the longitudinal field is wrong by a
    factor of two -- and the sampler still runs, and still converges.
    """
    lattice = Lattice(6, 5, "open")
    couplings = build_couplings(np.array([0.3, 0.7, 0.4, 0.9]), field_ratio=field_ratio)
    rng = np.random.default_rng(4)
    spins = lattice.random_spins(rng)

    before = classical_energy(spins, lattice, couplings)
    field = local_field(spins, lattice, couplings)
    for site, slice_index in ((0, 0), (3, 2), (5, 4), (2, 3)):
        flipped = spins.copy()
        flipped[site, slice_index] *= -1
        after = classical_energy(flipped, lattice, couplings)
        predicted = 2.0 * spins[site, slice_index] * field[site, slice_index]
        assert after - before == pytest.approx(predicted, abs=1e-10)


def test_the_chain_samples_the_boltzmann_distribution_it_claims_to() -> None:
    """Enumerate every configuration of a tiny lattice and compare frequencies.

    Six classical spins is 64 configurations, so the exact distribution
    ``exp(-H_cl)/Z`` can be written down in full. This is the only test that
    checks the Markov chain itself rather than what is measured on it.
    """
    lattice = Lattice(2, 3, "open")
    couplings = build_couplings(np.array([0.6, 0.5]), field_ratio=0.3)

    states = np.array(
        [[(-1) ** ((index >> bit) & 1) for bit in range(6)] for index in range(64)],
        dtype=np.int8,
    ).reshape(64, 2, 3)
    weights = np.exp([-classical_energy(state, lattice, couplings) for state in states])
    expected = weights / weights.sum()

    sampler = Sampler(lattice, transverse_field=1.0, rng=np.random.default_rng(11))
    sampler.warmup(couplings, 200)
    keys = {state.tobytes(): index for index, state in enumerate(states)}
    counts = np.zeros(64)
    draws = 200_000
    for _ in range(draws):
        sampler.sweep(couplings)
        counts[keys[np.ascontiguousarray(sampler.spins).tobytes()]] += 1

    observed = counts / draws
    # Binomial error on each bin, loosened for the autocorrelation of the chain.
    error = np.sqrt(expected * (1 - expected) / draws)
    assert np.max(np.abs(observed - expected) / error) < 8.0


def test_a_vanishing_imaginary_time_parameter_orders_the_time_direction() -> None:
    r"""As :math:`\beta \to 0`, :math:`J_\tau` diverges and the slices must align.

    This pins the *sign* with which the temporal coupling enters the local
    field: aligned neighbours have to be the favourable configuration.
    """
    lattice = Lattice(4, 3, "open")
    sampler = Sampler(lattice, transverse_field=1.0, rng=np.random.default_rng(0))
    couplings = build_couplings(np.array([1e-4, 1e-4]))
    sampler.warmup(couplings, 200)
    aligned = np.mean(sampler.spins[:, :-1] == sampler.spins[:, 1:])
    assert aligned > 0.95


def test_the_sweep_stays_in_the_spin_alphabet() -> None:
    lattice = Lattice(4, 5, "periodic")
    sampler = Sampler(lattice, transverse_field=1.0, rng=np.random.default_rng(0))
    couplings = build_couplings(np.array([0.3, 0.7, 0.4, 0.9]))
    sampler.warmup(couplings, 10)
    assert set(np.unique(sampler.spins)) <= {-1, 1}
    assert sampler.spins.shape == lattice.shape


# --------------------------------------------------------------------------
# The mapping itself: sampled expectation values against dense algebra
# --------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize(
    ("n_sites", "boundary", "coupling", "field", "theta"),
    [
        (8, "periodic", 1.0, 1.0, [0.5, 0.5]),
        (8, "periodic", 1.0, 0.5, [0.4, 0.7]),
        (8, "periodic", 1.0, 2.0, [0.35, 0.55, 0.65, 0.45]),
        (6, "open", 1.0, 1.0, [0.45, 0.35, 0.55, 0.25]),
        (6, "open", 1.7, 0.8, [0.3, 0.6]),
    ],
)
def test_the_sampled_energy_matches_the_dense_variational_energy(
    n_sites: int, boundary: str, coupling: float, field: float, theta: list[float]
) -> None:
    spec = TFIMSpec(n_sites=n_sites, coupling=coupling, field=field, boundary=boundary)  # type: ignore[arg-type]
    statistics = sampled(spec, theta)
    reference = variational_energy_per_spin(spec, np.array(theta))
    deviation = abs(statistics.energy - reference) / statistics.energy_error
    assert deviation < TOLERANCE_SIGMA, (
        f"sampled {statistics.energy:+.6f} +- {statistics.energy_error:.6f} "
        f"vs dense {reference:+.6f} ({deviation:.1f} sigma)"
    )


@pytest.mark.slow
@pytest.mark.parametrize("longitudinal_field", [0.3, 0.8])
def test_the_mapping_carries_the_longitudinal_field(longitudinal_field: float) -> None:
    """The one term no reference solver in this repository carries.

    ``reference/free_fermions.py`` is the free-fermion transverse-field model and
    ``reference/exact_diagonalisation.py`` builds the same Hamiltonian, so at
    ``g != 0`` neither can grade anything. The dense construction in this file
    can, which is what makes the longitudinal field usable rather than merely
    implemented.
    """
    spec = TFIMSpec(n_sites=6, coupling=1.0, field=1.0, boundary="open")
    statistics = sampled(spec, [0.4, 0.5], longitudinal_field=longitudinal_field, seed=5)
    reference = variational_energy_per_spin(spec, np.array([0.4, 0.5]), longitudinal_field)
    deviation = abs(statistics.energy - reference) / statistics.energy_error
    assert deviation < TOLERANCE_SIGMA, (
        f"sampled {statistics.energy:+.6f} +- {statistics.energy_error:.6f} "
        f"vs dense {reference:+.6f} ({deviation:.1f} sigma)"
    )


@pytest.mark.slow
def test_the_sampled_gradient_reproduces_the_derivative_of_the_dense_energy() -> None:
    """A wrong gradient still converges -- somewhere else. Only this catches it."""
    spec = TFIMSpec(n_sites=DEFAULT_SITES, coupling=1.0, field=1.0, boundary="periodic")
    theta = np.array([0.35, 0.55, 0.65, 0.45])

    step = 1e-5
    numerical = np.empty_like(theta)
    for index in range(theta.size):
        shift = np.zeros_like(theta)
        shift[index] = step
        numerical[index] = (
            variational_energy_per_spin(spec, theta + shift)
            - variational_energy_per_spin(spec, theta - shift)
        ) / (2 * step)

    statistics = sampled(spec, list(theta), seed=3)
    assert np.allclose(statistics.gradient / spec.n_sites, numerical, atol=0.05)


@pytest.mark.slow
def test_the_sampled_state_never_sits_below_the_true_ground_state() -> None:
    """A variational energy below the ground state is impossible, so it is a bug.

    It is also a bug this code has produced before: the transverse-field
    estimator is heavy-tailed as its parameter shrinks, and a short chain then
    reports a number that is not merely noisy but unphysical.
    """
    spec = TFIMSpec(n_sites=DEFAULT_SITES, coupling=1.0, field=1.0, boundary="periodic")
    statistics = sampled(spec, [0.5, 0.5], seed=2)
    ground = free_fermions.energy_density(spec)
    assert statistics.energy > ground - TOLERANCE_SIGMA * statistics.energy_error


# --------------------------------------------------------------------------
# Stochastic reconfiguration
# --------------------------------------------------------------------------
def test_the_metric_is_symmetric_and_positive_semidefinite() -> None:
    spec = TFIMSpec(n_sites=DEFAULT_SITES, coupling=1.0, field=1.0, boundary="periodic")
    statistics = sampled(spec, [0.35, 0.55, 0.65, 0.45], seed=1, config=vita.SamplingConfig())
    assert np.allclose(statistics.metric, statistics.metric.T)
    assert np.min(np.linalg.eigvalsh(statistics.metric)) > -1e-10


def test_the_log_derivative_carries_its_factor_of_one_half() -> None:
    assert log_derivatives(np.array([2.0, -4.0])).tolist() == [-1.0, 2.0]


def test_the_natural_gradient_is_the_plain_gradient_under_a_flat_metric() -> None:
    gradient = np.array([1.0, -2.0, 0.5])
    step = vita.natural_gradient(gradient, np.eye(3), diagonal_shift=0.0)
    assert np.allclose(step, gradient)


def test_the_natural_gradient_survives_a_singular_metric() -> None:
    """Two redundant pulses make the metric singular; the solve must still work."""
    step = vita.natural_gradient(np.ones(3), np.zeros((3, 3)), diagonal_shift=1e-3)
    assert np.all(np.isfinite(step))


# --------------------------------------------------------------------------
# The loop, end to end
# --------------------------------------------------------------------------
def test_a_single_bin_reports_no_error_rather_than_a_wrong_one() -> None:
    spec = TFIMSpec(n_sites=DEFAULT_SITES, coupling=1.0, field=1.0, boundary="periodic")
    statistics = sampled(spec, [0.5, 0.5], config=vita.SamplingConfig(warmup=10, sweeps=5, bins=1))
    assert np.isnan(statistics.energy_error)


def test_the_history_refuses_to_average_an_empty_trace() -> None:
    with pytest.raises(ValueError, match="history is empty"):
        vita.OptimisationHistory().converged_theta()


def test_the_tail_average_ignores_the_lowest_sampled_energy() -> None:
    """The bias control, stated as a test rather than as a comment.

    A run that wanders to a wildly low sampled energy at one iteration must not
    be pulled towards the parameters that produced it.
    """
    history = vita.OptimisationHistory()
    statistics = vita.SampleStatistics(0.0, 0.0, 0.0, 0.0, np.zeros(2), np.zeros((2, 2)))
    for theta in ([1.0, 1.0], [9.0, 9.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]):
        history.record(np.array(theta), statistics)
    history.energy[1] = -1e6  # an impossible downward fluctuation
    assert history.converged_theta(tail_fraction=0.6).tolist() == [1.0, 1.0]


@pytest.mark.slow
@pytest.mark.parametrize("field", [0.5, 1.0, 2.0])
def test_the_optimisation_reaches_the_dense_variational_optimum(field: float) -> None:
    """A converged run must reach the best energy the ansatz can reach.

    The comparison is against the *variational* optimum found by a dense
    optimiser, not against the exact ground state: depth one cannot represent
    the ground state, and a test that demanded it would be testing the wrong
    thing. What it does demand is that sampling costs nothing in quality --
    that the Monte Carlo route finds what dense linear algebra finds.
    """
    spec = TFIMSpec(n_sites=DEFAULT_SITES, coupling=1.0, field=field, boundary="periodic")
    result = vita.ground_state_energy(spec, depth=1, seed=1)
    best = dense_optimum(spec)

    deviation = (result.energy - best.fun) / result.energy_error
    assert deviation < TOLERANCE_SIGMA, (
        f"sampled {result.energy:+.6f} +- {result.energy_error:.6f} sits "
        f"{deviation:.1f} sigma above the dense optimum {best.fun:+.6f}"
    )
    # And the variational bound holds against the answer this module cannot see.
    assert (
        result.energy > free_fermions.energy_density(spec) - TOLERANCE_SIGMA * result.energy_error
    )


@pytest.mark.slow
def test_the_optimisation_finds_the_right_parameters_where_the_minimum_is_sharp() -> None:
    """Parameters, not just energy -- but only where parameters are identifiable.

    At criticality the minimum is sharp in both directions and a converged run
    has nowhere else to sit. Deep in the ferromagnetic phase it is not: at
    ``h/J = 0.5`` the energy changes by less than one error bar as ``alpha``
    moves by 15%, so the optimiser stops somewhere along a nearly flat valley
    and the *parameters* disagree while the *energy* agrees. That is a property
    of the landscape rather than a defect in the sampler, and it is the same
    flatness that makes deep variational circuits hard to train -- which is why
    it is recorded here as a test rather than filed as a bug.
    """
    spec = TFIMSpec(n_sites=DEFAULT_SITES, coupling=1.0, field=1.0, boundary="periodic")
    result = vita.ground_state_energy(spec, depth=1, seed=1)
    assert np.allclose(result.theta, np.abs(dense_optimum(spec).x), atol=0.1), (
        f"sampled {np.round(result.theta, 3)} vs dense {np.round(np.abs(dense_optimum(spec).x), 3)}"
    )


@pytest.mark.slow
def test_the_ferromagnetic_valley_really_is_flat() -> None:
    """The claim the test above rests on, checked rather than asserted in prose."""
    spec = TFIMSpec(n_sites=DEFAULT_SITES, coupling=1.0, field=0.5, boundary="periodic")
    best = np.abs(dense_optimum(spec).x)
    shifted = best * np.array([0.85, 1.0])
    cost = variational_energy_per_spin(spec, shifted) - variational_energy_per_spin(spec, best)
    assert cost < 0.003, f"a 15% move in alpha costs {cost:.2e} per spin"


def test_the_result_reports_what_it_spent() -> None:
    spec = TFIMSpec(n_sites=6, coupling=1.0, field=1.0, boundary="open")
    sampling = vita.SamplingConfig(warmup=20, sweeps=4, bins=4)
    optimiser = vita.OptimiserConfig(iterations=5)
    result = vita.ground_state_energy(spec, depth=1, sampling=sampling, optimiser=optimiser)

    assert result.depth == 1
    assert result.theta.size == 2
    assert result.imaginary_time == pytest.approx(0.5 * float(np.sum(result.theta)))
    # Five iterations of 16 measurements, plus a final measurement of 4x the bins.
    assert result.n_measurements == 5 * 16 + 4 * 16


def test_a_zero_depth_ansatz_is_refused() -> None:
    spec = TFIMSpec(n_sites=6, coupling=1.0, field=1.0, boundary="open")
    with pytest.raises(ValueError, match="depth P"):
        vita.optimise(spec, depth=0)


def test_sampling_settings_that_measure_nothing_are_refused() -> None:
    with pytest.raises(ValueError, match="at least one sweep"):
        vita.SamplingConfig(sweeps=0)
    with pytest.raises(ValueError, match="non-negative"):
        vita.SamplingConfig(warmup=-1)
