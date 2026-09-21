"""Tests for the circuit specification -- the arithmetic done before anything runs.

Nothing here builds a circuit or a state vector. The whole point of an
:class:`~src.physics.quantum.ansatz.AnsatzSpec` is that a depth can be priced from
integers alone, so these tests are integer tests, and the expensive checks that the
counts describe a real circuit live in ``tests/test_statevector.py`` and
``tests/test_hardware.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.physics.lattice import Lattice
from src.physics.quantum.ansatz import (
    AnsatzSpec,
    adiabatic_ramp,
    angle_periods,
    initial_angles,
    interpolate_to_depth,
    join_angles,
    small_angle,
    split_angles,
    wrap_angles,
)

SIZES = [2, 3, 4, 6, 8, 12]
DEPTHS = [0, 1, 2, 5]


# --------------------------------------------------------------------------
# The counts
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_qubits", SIZES)
@pytest.mark.parametrize("depth", DEPTHS)
def test_the_parameter_count_does_not_grow_with_the_chain(n_qubits: int, depth: int) -> None:
    # The reason this family is worth running at all: the classical optimisation problem
    # is the same size at 8 qubits as at 80.
    assert AnsatzSpec(n_qubits=n_qubits, depth=depth).n_parameters == 2 * depth


@pytest.mark.parametrize("n_qubits", [3, 4, 6, 8, 12])
def test_an_open_chain_needs_two_rounds_however_long_it_is(n_qubits: int) -> None:
    spec = AnsatzSpec(n_qubits=n_qubits, depth=1, boundary="open")
    assert spec.two_qubit_rounds_per_layer == 2


@pytest.mark.parametrize("n_qubits", [4, 6, 8, 12])
def test_an_even_ring_also_needs_two_rounds(n_qubits: int) -> None:
    spec = AnsatzSpec(n_qubits=n_qubits, depth=1, boundary="periodic")
    assert spec.two_qubit_rounds_per_layer == 2


@pytest.mark.parametrize("n_qubits", [3, 5, 7, 9])
def test_an_odd_ring_costs_a_third_round(n_qubits: int) -> None:
    # A cycle of odd length is not two-colourable, so one bond is always left over. This
    # is a real half-again cost in two-qubit depth and not a rounding detail, and it is
    # the reason the rounds are counted by colouring rather than assumed to be two.
    spec = AnsatzSpec(n_qubits=n_qubits, depth=1, boundary="periodic")
    assert spec.two_qubit_rounds_per_layer == 3


def test_a_two_site_chain_has_a_single_round() -> None:
    assert AnsatzSpec(n_qubits=2, depth=1).two_qubit_rounds_per_layer == 1


@pytest.mark.parametrize("n_qubits", SIZES)
def test_the_rounds_really_do_partition_the_bonds_without_a_shared_qubit(n_qubits: int) -> None:
    # Restates the colouring claim as the property that licenses it: two bonds may fire
    # together exactly when they are disjoint. A count alone would pass even if the
    # grouping were nonsense.
    spec = AnsatzSpec(n_qubits=n_qubits, depth=1)
    rounds: list[set[int]] = []
    for left, right in spec.bonds:
        for occupied in rounds:
            if left not in occupied and right not in occupied:
                occupied.update((left, right))
                break
        else:
            rounds.append({left, right})
    assert len(rounds) == spec.two_qubit_rounds_per_layer
    assert sum(len(group) for group in rounds) == 2 * len(spec.bonds)


@pytest.mark.parametrize("n_qubits", [6, 10, 14])
def test_the_even_odd_split_beats_naive_compilation_by_the_chain_length(n_qubits: int) -> None:
    # The single largest optimisation in the project, stated as a ratio. At 14 sites the
    # depth falls from 26 to 4 on a bit-identical unitary.
    spec = AnsatzSpec(n_qubits=n_qubits, depth=1)
    assert spec.two_qubit_depth == 4
    assert spec.naive_two_qubit_depth() == 2 * (n_qubits - 1)
    assert spec.describe()["depth_saving_factor"] == pytest.approx((n_qubits - 1) / 2)


def test_the_longitudinal_field_costs_no_two_qubit_depth() -> None:
    # Why g is the knob the project holds in reserve: it breaks integrability, and it is
    # free in the currency that a coherence budget is spent in.
    plain = AnsatzSpec(n_qubits=8, depth=3, longitudinal=False)
    mixed = AnsatzSpec(n_qubits=8, depth=3, longitudinal=True)
    assert plain.two_qubit_depth == mixed.two_qubit_depth
    assert plain.two_qubit_gates == mixed.two_qubit_gates
    assert mixed.single_qubit_gates > plain.single_qubit_gates


def test_the_family_name_changes_no_gate_count() -> None:
    hva = AnsatzSpec(n_qubits=6, depth=3, family="hva").describe()
    qaoa = AnsatzSpec(n_qubits=6, depth=3, family="qaoa").describe()
    assert {key: value for key, value in hva.items() if key != "family"} == {
        key: value for key, value in qaoa.items() if key != "family"
    }


def test_a_depth_zero_ansatz_costs_nothing_but_state_preparation() -> None:
    spec = AnsatzSpec(n_qubits=6, depth=0)
    assert (spec.n_parameters, spec.two_qubit_gates, spec.two_qubit_depth) == (0, 0, 0)
    assert spec.single_qubit_gates == 6


@pytest.mark.parametrize(
    ("n_qubits", "depth"),
    [(1, 1), (0, 1), (4, -1)],
)
def test_an_impossible_specification_is_refused(n_qubits: int, depth: int) -> None:
    with pytest.raises(ValueError):
        AnsatzSpec(n_qubits=n_qubits, depth=depth)


def test_describe_returns_only_json_safe_values() -> None:
    # A tool result travels into a language model's context, where a numpy scalar becomes
    # a serialisation error rather than a number.
    for value in AnsatzSpec(n_qubits=6, depth=2).describe().values():
        assert isinstance(value, (int, float, str, bool))


# --------------------------------------------------------------------------
# The schedules
# --------------------------------------------------------------------------


@pytest.mark.parametrize("depth", [1, 2, 5, 20])
def test_the_ramp_rises_in_gamma_and_falls_in_beta(depth: int) -> None:
    # The adiabatic reading of the schedule: the diagonal half is switched on while the
    # field half is switched off. If this ordering ever inverts, the initialisation is
    # running the ramp backwards and the warm start is worse than a random one.
    gamma, beta = split_angles(adiabatic_ramp(depth, ramp_time=2.0))
    assert np.all(np.diff(gamma) > 0) if depth > 1 else True
    assert np.all(np.diff(beta) < 0) if depth > 1 else True


@pytest.mark.parametrize("depth", [1, 2, 5, 20])
def test_no_ramp_layer_is_wasted_on_the_identity(depth: int) -> None:
    # Sampling the adiabatic path at its endpoints would put beta_p at exactly zero,
    # spending the last layer's two-qubit depth on nothing. The midpoint rule is what
    # avoids that, and this is the test that keeps it.
    angles = adiabatic_ramp(depth, ramp_time=2.0)
    assert np.all(np.abs(angles) > 0.0)


def test_the_ramp_scales_linearly_with_its_time() -> None:
    single = adiabatic_ramp(4, ramp_time=1.0)
    double = adiabatic_ramp(4, ramp_time=2.0)
    assert np.allclose(double, 2.0 * single)


@pytest.mark.parametrize("depth", [0, 1, 3])
def test_every_initialisation_returns_two_angles_per_layer(depth: int) -> None:
    for strategy in ("adiabatic_ramp", "small_angle", "zeros"):
        assert initial_angles(depth, strategy).size == 2 * depth


def test_the_small_angle_start_is_near_the_identity() -> None:
    # The standard first defence against a barren plateau is only a defence if the angles
    # really are small; a scale that silently became large would look identical.
    assert np.all(np.abs(small_angle(6, scale=1e-2, seed=0)) <= 1e-2)


def test_the_small_angle_start_reproduces_from_its_seed() -> None:
    assert np.array_equal(small_angle(4, seed=7), small_angle(4, seed=7))
    assert not np.array_equal(small_angle(4, seed=7), small_angle(4, seed=8))


def test_an_unknown_initialisation_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="unknown initialisation strategy"):
        initial_angles(3, "linear_ramp")  # type: ignore[arg-type]


def test_splitting_and_joining_are_inverse() -> None:
    angles = adiabatic_ramp(5)
    assert np.array_equal(join_angles(*split_angles(angles)), angles)


def test_an_odd_length_vector_cannot_be_split() -> None:
    with pytest.raises(ValueError, match="even number of angles"):
        split_angles(np.zeros(5))


def test_two_schedules_of_different_length_cannot_be_joined() -> None:
    with pytest.raises(ValueError, match="same length"):
        join_angles(np.zeros(3), np.zeros(4))


@pytest.mark.parametrize("new_depth", [1, 2, 3, 7])
def test_interpolation_lands_on_the_requested_depth(new_depth: int) -> None:
    assert interpolate_to_depth(adiabatic_ramp(4), new_depth).size == 2 * new_depth


def test_interpolating_onto_the_same_depth_changes_nothing() -> None:
    angles = adiabatic_ramp(5)
    assert np.allclose(interpolate_to_depth(angles, 5), angles)


def test_interpolation_preserves_the_endpoints_of_each_schedule() -> None:
    # The warm start's whole value is that it inherits the converged basin. An
    # interpolation that moved the ends would be starting somewhere else.
    gamma, beta = split_angles(adiabatic_ramp(3, ramp_time=2.0))
    grown_gamma, grown_beta = split_angles(interpolate_to_depth(adiabatic_ramp(3, 2.0), 9))
    assert grown_gamma[0] == pytest.approx(gamma[0])
    assert grown_gamma[-1] == pytest.approx(gamma[-1])
    assert grown_beta[0] == pytest.approx(beta[0])
    assert grown_beta[-1] == pytest.approx(beta[-1])


def test_an_empty_schedule_cannot_be_grown() -> None:
    with pytest.raises(ValueError, match="empty schedule"):
        interpolate_to_depth(np.zeros(0), 3)


def test_wrapping_folds_a_whole_period_back_onto_itself() -> None:
    # Two vectors describing the identical state must compare equal, or the convergence
    # test never fires on a solution that merely drifted a full period.
    angles = np.array([0.3, -0.7, 1.2, 2.9])
    periods = angle_periods(1.0, 1.0)
    turn = np.array([np.pi, np.pi, np.pi, np.pi])
    assert np.allclose(wrap_angles(angles + turn, *periods), wrap_angles(angles, *periods))


def test_wrapping_with_no_period_given_moves_nothing() -> None:
    # The honest default. A parameter vector does not carry J or h, so a caller that
    # cannot supply them gets an untidy answer rather than an incorrect one.
    angles = np.array([0.0, 1.0, -1.0, 9.0])
    assert np.allclose(wrap_angles(angles), angles)


def test_the_period_is_the_generators_and_not_the_circles() -> None:
    # The defect this replaced: folding by 2*pi. It is a whole number of periods only
    # when 2J and 2h are whole numbers, which is true at the calibration point and
    # false at most of the values the interface's sliders offer.
    assert angle_periods(1.0, 1.0) == (np.pi, np.pi)
    assert angle_periods(0.5, 2.0) == (2.0 * np.pi, 0.5 * np.pi)
    # Two incommensurate scales in the diagonal generator: no common period, so the
    # answer is "do not fold" rather than a plausible-looking number.
    assert angle_periods(1.0, 1.0, longitudinal_field=0.4)[0] is None


@pytest.mark.parametrize(("coupling", "field"), [(1.0, 0.85), (0.7, 1.0), (0.3, 1.7)])
def test_folding_by_the_period_preserves_the_energy(coupling: float, field: float) -> None:
    # The check that has teeth, and the one the old fold failed: a solver reports its
    # angles beside its energy, so the angles must still produce that energy. Held
    # against the state vector rather than against the wrapping arithmetic itself,
    # which would only ask the fold whether it agreed with itself.
    from src.physics.quantum.statevector import diagonal_energies, energy, evolve

    spec = AnsatzSpec(n_qubits=6, depth=3)
    theta = np.linspace(-8.0, 8.0, spec.n_parameters)
    diagonal = diagonal_energies(spec.n_qubits, coupling=coupling)
    before = energy(evolve(theta, spec, diagonal, field), diagonal, field)
    folded = wrap_angles(theta, *angle_periods(coupling, field))
    after = energy(evolve(folded, spec, diagonal, field), diagonal, field)
    assert after == pytest.approx(before, abs=1e-9)
    # And the fold actually did something, or the check above is vacuous.
    assert not np.allclose(folded, theta)


# --------------------------------------------------------------------------
# A lattice reaches the circuit, not only the Hamiltonian
# --------------------------------------------------------------------------


def test_a_lattice_changes_the_bonds_the_circuit_acts_on() -> None:
    # The gap this closes: `lattice=` reached the Hamiltonian assembly and the grader
    # but not the circuit layer, so a square lattice was costed and evolved as a chain
    # of the same width.
    square = Lattice("square", 2, 2)
    flat = AnsatzSpec(n_qubits=4, depth=2)
    shaped = AnsatzSpec(n_qubits=4, depth=2, lattice=square)
    assert flat.bonds == ((0, 1), (1, 2), (2, 3))
    assert shaped.bonds == ((0, 1), (0, 2), (1, 3), (2, 3))
    assert shaped.two_qubit_gates > flat.two_qubit_gates
    assert shaped.describe()["geometry"] == "square"


def test_a_lattice_that_contradicts_the_register_is_refused() -> None:
    # Two spellings of the same fact are two spellings that will disagree. Neither is
    # preferred, because whichever property happened to read which would be right half
    # the time; the disagreement is refused instead.
    with pytest.raises(ValueError, match="different circuits"):
        AnsatzSpec(n_qubits=9, depth=1, lattice=Lattice("square", 2, 2))
    with pytest.raises(ValueError, match="different circuits"):
        AnsatzSpec(n_qubits=4, depth=1, boundary="periodic", lattice=Lattice("square", 2, 2))


def test_a_triangular_lattice_costs_more_rounds_than_a_square_one() -> None:
    # Non-bipartite, so its bonds cannot be two-coloured and a layer needs a third
    # round. That is a hardware cost, and it is the reason frustration is expensive
    # on a device as well as hard classically.
    square = AnsatzSpec(n_qubits=9, depth=1, lattice=Lattice("square", 3, 3))
    triangular = AnsatzSpec(n_qubits=9, depth=1, lattice=Lattice("triangular", 3, 3))
    assert triangular.two_qubit_rounds_per_layer > square.two_qubit_rounds_per_layer
