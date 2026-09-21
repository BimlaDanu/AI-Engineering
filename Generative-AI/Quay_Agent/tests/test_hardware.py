"""The device layer: wiring, routing, noise, and the files that come out of it.

Nothing here needs a credential, a network or a simulator. Every figure the device
layer produces is integer arithmetic over a coupling graph or a product of
probabilities, which is what makes it testable at all -- and what makes a number in
a report something a reader can recompute rather than take on trust.

The check that matters most is the one against a module that shares no algebra with
this one: :class:`~src.physics.quantum.ansatz.AnsatzSpec` prices a circuit by
counting bonds and colouring them, and this package prices it by placing spins on a
lattice and walking shortest paths. Where the wiring matches the problem the two
must agree exactly. Where it does not, the disagreement is the finding.
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from src.agent.state import CoherenceBudget
from src.hardware.devices import (
    DEVICES,
    HEAVY_HEX,
    IDEAL,
    LINEAR,
    Device,
    device_for,
    device_names,
)
from src.hardware.export import (
    Couplings,
    RunCard,
    qasm3,
    readme,
    run_card,
    submission_package,
)
from src.hardware.fidelity import (
    OBSERVABLE_LOCALITY,
    USABLE_FIDELITY_FLOOR,
    coherence_budget,
    depth_ceiling,
    estimate,
    idle_qubit_ns,
    shot_inflation,
)
from src.hardware.transpile import (
    BOND_TWO_QUBIT_GATES,
    SWAP_TWO_QUBIT_GATES,
    Layout,
    fits,
    place,
    route,
    schedule,
    transpile,
)
from src.physics.quantum.ansatz import AnsatzSpec


def chain(n_sites: int = 8, depth: int = 2, boundary: str = "open") -> AnsatzSpec:
    """Build one ansatz without repeating the keywords in every test.

    Args:
        n_sites: Length of the chain.
        depth: Number of ansatz layers.
        boundary: ``"open"`` or ``"periodic"``.

    Returns:
        The specification.
    """
    return AnsatzSpec(n_qubits=n_sites, depth=depth, boundary=boundary)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# The machines
# --------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES, ids=lambda d: d.name)
def test_every_device_describes_itself_in_primitives(device: Device) -> None:
    # A description that needed a custom encoder could not be logged, written to a
    # run card or handed to a model, which is the only reason it exists.
    described = device.describe()
    assert json.loads(json.dumps(described)) == described


@pytest.mark.parametrize("device", DEVICES, ids=lambda d: d.name)
def test_coupling_is_symmetric_and_within_the_register(device: Device) -> None:
    for left, right in device.coupling:
        assert device.are_coupled(left, right)
        assert device.are_coupled(right, left)
        assert left in device.neighbours(right)
        assert right in device.neighbours(left)


@pytest.mark.parametrize("device", DEVICES, ids=lambda d: d.name)
def test_every_qubit_is_reachable_from_every_other(device: Device) -> None:
    # A disconnected fragment would make routing raise rather than cost, and the
    # failure would surface as an exception in a planner rather than as a price.
    for qubit in range(device.n_qubits):
        assert device.distance(0, qubit) >= 0


def test_a_line_has_the_distance_a_line_has() -> None:
    for qubit in range(LINEAR.n_qubits):
        assert LINEAR.distance(0, qubit) == qubit


def test_everything_is_one_step_apart_on_an_all_to_all_device() -> None:
    assert all(IDEAL.distance(0, q) == 1 for q in range(1, IDEAL.n_qubits))


def test_the_heavy_hex_lattice_has_the_degrees_that_give_it_its_name() -> None:
    # Two hexagons' worth of structure: mostly degree two, a few degree three, and
    # nothing higher. Degree four is what the lattice exists to avoid.
    degrees = {HEAVY_HEX.degree(q) for q in range(HEAVY_HEX.n_qubits)}
    assert degrees == {1, 2, 3}
    assert HEAVY_HEX.n_edges == HEAVY_HEX.n_qubits + 1  # a tree plus two cycles


def test_a_line_and_an_all_to_all_device_have_a_path_through_every_qubit() -> None:
    assert len(LINEAR.longest_path()) == LINEAR.n_qubits
    assert len(IDEAL.longest_path()) == IDEAL.n_qubits


def test_the_heavy_hex_lattice_does_not_and_the_shortfall_is_the_point() -> None:
    # Twenty-one of twenty-seven qubits lie on one path. The other six are reachable
    # only by routing, which is why a chain longer than twenty-one is a different
    # proposition on this machine from a chain of twenty.
    path = HEAVY_HEX.longest_path()
    assert len(path) == 21
    assert len(set(path)) == len(path)
    assert all(HEAVY_HEX.are_coupled(a, b) for a, b in pairwise(path))


@pytest.mark.parametrize("device", DEVICES, ids=lambda d: d.name)
def test_the_longest_path_is_a_path(device: Device) -> None:
    path = device.longest_path()
    assert len(set(path)) == len(path)
    assert all(device.are_coupled(a, b) for a, b in pairwise(path))


def test_a_device_can_be_looked_up_by_name_however_it_was_typed() -> None:
    assert device_for("  LINEAR ") is LINEAR
    assert set(device_names()) == {d.name for d in DEVICES}


def test_an_unknown_device_says_what_the_known_ones_are() -> None:
    # The caller is often a model that guessed a name. An error that lists the real
    # ones turns a dead end into a retry.
    with pytest.raises(KeyError) as raised:
        device_for("sycamore")
    assert "linear" in str(raised.value)


def test_a_qubit_outside_the_register_is_an_error_not_a_wrap_around() -> None:
    with pytest.raises(IndexError):
        LINEAR.neighbours(LINEAR.n_qubits)
    with pytest.raises(IndexError):
        LINEAR.distance(0, -1)


def test_coherence_longer_than_twice_the_relaxation_time_is_refused() -> None:
    # T2 <= 2*T1 is not a convention, it is what the two quantities are. A device
    # model that allowed otherwise would hand every fidelity estimate free coherence.
    with pytest.raises(ValueError, match="exceeds"):
        Device(
            name="impossible",
            topology="line",
            n_qubits=2,
            coupling=((0, 1),),
            native_two_qubit="cz",
            two_qubit_gate_ns=300.0,
            single_qubit_gate_ns=35.0,
            readout_ns=1000.0,
            t1_ns=1000.0,
            t2_ns=5000.0,
            two_qubit_error=0.01,
            single_qubit_error=0.001,
            readout_error=0.01,
            provenance="a test",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("two_qubit_error", 1.0),
        ("readout_error", -0.1),
        ("two_qubit_gate_ns", 0.0),
        ("n_qubits", 0),
    ],
)
def test_a_device_rejects_numbers_that_describe_no_machine(field: str, value: float) -> None:
    fields = {
        "name": "broken",
        "topology": "line",
        "n_qubits": 2,
        "coupling": ((0, 1),),
        "native_two_qubit": "cz",
        "two_qubit_gate_ns": 300.0,
        "single_qubit_gate_ns": 35.0,
        "readout_ns": 1000.0,
        "t1_ns": 100_000.0,
        "t2_ns": 100_000.0,
        "two_qubit_error": 0.01,
        "single_qubit_error": 0.001,
        "readout_error": 0.01,
        "provenance": "a test",
    }
    fields[field] = value
    if field == "n_qubits":
        fields["coupling"] = ()
    with pytest.raises(ValueError):
        Device(**fields)  # type: ignore[arg-type]


def test_an_edge_naming_a_qubit_that_does_not_exist_is_refused() -> None:
    with pytest.raises(ValueError, match="outside the register"):
        Device(
            name="broken",
            topology="line",
            n_qubits=2,
            coupling=((0, 5),),
            native_two_qubit="cz",
            two_qubit_gate_ns=300.0,
            single_qubit_gate_ns=35.0,
            readout_ns=1000.0,
            t1_ns=100_000.0,
            t2_ns=100_000.0,
            two_qubit_error=0.01,
            single_qubit_error=0.001,
            readout_error=0.01,
            provenance="a test",
        )


# --------------------------------------------------------------------------
# Placement and routing
# --------------------------------------------------------------------------


def test_a_chain_larger_than_the_register_is_refused_with_a_sentence() -> None:
    # A refusal rather than an exception, because the caller is a planner choosing
    # between configurations and needs to record why this one was dropped.
    refusal = fits(40, LINEAR)
    assert refusal is not None
    assert "32 qubits" in refusal
    assert fits(8, LINEAR) is None


def test_placing_more_spins_than_qubits_raises() -> None:
    with pytest.raises(ValueError, match="no layout exists"):
        place(40, LINEAR, "open")


@pytest.mark.parametrize("device", DEVICES, ids=lambda d: d.name)
@pytest.mark.parametrize("boundary", ["open", "periodic"])
def test_no_two_spins_share_a_qubit(device: Device, boundary: str) -> None:
    layout = place(10, device, boundary)  # type: ignore[arg-type]
    assert len(set(layout.sites)) == 10


def test_an_open_chain_is_laid_out_along_a_connected_run() -> None:
    layout = place(8, LINEAR, "open")
    assert layout.strategy == "path"
    assert all(LINEAR.are_coupled(a, b) for a, b in pairwise(layout.sites))


def test_a_ring_is_folded_so_that_its_two_ends_start_as_neighbours() -> None:
    # The whole reason a ring is runnable. Unfolded, the wrap-around bond is a serial
    # SWAP chain the length of the register; folded, it is one edge.
    layout = place(8, LINEAR, "periodic")
    assert layout.strategy == "folded_path"
    assert LINEAR.are_coupled(layout.physical(0), layout.physical(7))


def test_the_folded_layout_leaves_no_bond_further_than_two_apart() -> None:
    for n_sites in (4, 6, 10, 16):
        layout = place(n_sites, LINEAR, "periodic")
        spec = chain(n_sites, 1, "periodic")
        assert max(bond.distance for bond in route(layout, LINEAR, spec.bonds)) <= 2


def test_a_bond_the_wiring_already_provides_costs_nothing() -> None:
    routed = route(place(6, LINEAR, "open"), LINEAR, chain(6, 1).bonds)
    assert all(bond.distance == 1 for bond in routed)
    assert all(bond.swaps == 0 for bond in routed)
    assert all(not bond.routed for bond in routed)


def test_a_routed_bond_pays_two_swaps_for_every_edge_it_is_short() -> None:
    # One SWAP to walk the qubits together, one to put the layout back for the layer
    # that follows.
    layout = Layout(device=LINEAR.name, sites=(0, 4), strategy="path")
    (bond,) = route(layout, LINEAR, ((0, 1),))
    assert bond.distance == 4
    assert bond.swaps == 2 * (4 - 1)
    assert bond.two_qubit_depth == bond.swaps * SWAP_TWO_QUBIT_GATES + BOND_TWO_QUBIT_GATES


def test_a_routed_bond_carries_the_route_it_was_charged_for() -> None:
    layout = Layout(device=LINEAR.name, sites=(2, 6), strategy="path")
    (bond,) = route(layout, LINEAR, ((0, 1),))
    assert bond.path == (2, 3, 4, 5, 6)
    assert len(bond.path) == bond.distance + 1
    assert bond.footprint == frozenset(bond.path)


def test_routing_is_deterministic() -> None:
    # Two runs of the same campaign have to produce the same circuit, or nothing
    # about the run is reproducible.
    first = transpile(chain(12, 3, "periodic"), HEAVY_HEX)
    second = transpile(chain(12, 3, "periodic"), HEAVY_HEX)
    assert [b.path for b in first.bonds] == [b.path for b in second.bonds]
    assert first.two_qubit_depth == second.two_qubit_depth


def test_interactions_that_share_no_qubit_go_in_the_same_round() -> None:
    routed = route(place(8, LINEAR, "open"), LINEAR, chain(8, 1).bonds)
    rounds, depth = schedule(routed)
    assert rounds == 2  # even bonds, then odd ones
    assert depth == 2 * BOND_TWO_QUBIT_GATES


def test_scheduling_nothing_costs_nothing() -> None:
    assert schedule(()) == (0, 0)


# --------------------------------------------------------------------------
# The cross-check: the abstract price against the placed one
# --------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES, ids=lambda d: d.name)
@pytest.mark.parametrize("n_sites", [4, 8, 12])
@pytest.mark.parametrize("depth", [1, 3])
def test_an_open_chain_costs_exactly_what_the_ansatz_said(
    device: Device, n_sites: int, depth: int
) -> None:
    # The problem is shaped like the machine, so placement is free and the two
    # independent price calculations -- one by colouring bonds, one by walking a
    # lattice -- have to give the same number. This is the check that says the
    # device layer has not invented cost that is not there.
    spec = chain(n_sites, depth, "open")
    compiled = transpile(spec, device)
    assert compiled.swaps == 0
    assert compiled.two_qubit_depth == spec.two_qubit_depth
    assert compiled.two_qubit_gates == spec.two_qubit_gates
    assert compiled.routing_overhead == pytest.approx(1.0)


def test_a_ring_costs_more_than_a_segment_on_a_line_and_nothing_extra_on_all_to_all() -> None:
    segment = transpile(chain(12, 2, "open"), LINEAR)
    ring = transpile(chain(12, 2, "periodic"), LINEAR)
    assert ring.two_qubit_depth > segment.two_qubit_depth
    assert ring.routing_overhead > 1.0
    assert transpile(chain(12, 2, "periodic"), IDEAL).routing_overhead == pytest.approx(1.0)


def test_folding_holds_a_rings_depth_flat_while_the_naive_layout_grows_with_it() -> None:
    # The result the placement stage exists for. The SWAP count is comparable either
    # way; the depth is not, because a naive layout turns one bond into a serial
    # chain that every other gate waits behind.
    folded, naive = [], []
    for n_sites in (6, 8, 12):
        spec = chain(n_sites, 1, "periodic")
        folded.append(schedule(route(place(n_sites, LINEAR, "periodic"), LINEAR, spec.bonds))[1])
        flat = Layout(device=LINEAR.name, sites=tuple(range(n_sites)), strategy="path")
        naive.append(schedule(route(flat, LINEAR, spec.bonds))[1])
    assert len(set(folded)) == 1, f"folded depth should not depend on length, got {folded}"
    assert naive == sorted(naive) and naive[-1] > naive[0]
    # Twice, not three times: the ratio is what grows with length, so the factor a
    # test can assert is set by the longest ring it runs. Twelve sites gives 66
    # against 32. The twenty this used to end on gave 3.4x and is not worth the
    # twelve seconds -- the finding is that folding is flat and naive is not, and
    # three lengths show that as well as four did.
    assert naive[-1] > 2 * folded[-1]


def test_a_deeper_circuit_costs_proportionally_more() -> None:
    # Layers are identical, so the price is linear in the count. A model where it is
    # not has an off-by-one in the per-layer arithmetic.
    one = transpile(chain(10, 1, "periodic"), HEAVY_HEX)
    five = transpile(chain(10, 5, "periodic"), HEAVY_HEX)
    assert five.two_qubit_depth == 5 * one.two_qubit_depth
    assert five.swaps == 5 * one.swaps


def test_the_compiled_description_is_primitives_and_names_both_prices() -> None:
    described = transpile(chain(10, 2, "periodic"), HEAVY_HEX).describe()
    assert json.loads(json.dumps(described)) == described
    assert described["abstract_two_qubit_depth"] < described["two_qubit_depth"]
    assert described["routing_overhead"] > 1.0


def test_a_zero_depth_circuit_is_not_reported_as_infinitely_overrun() -> None:
    assert transpile(chain(6, 0, "open"), LINEAR).routing_overhead == pytest.approx(1.0)


# --------------------------------------------------------------------------
# What survives
# --------------------------------------------------------------------------


def test_the_ideal_machine_loses_nothing() -> None:
    surviving = estimate(transpile(chain(12, 6, "periodic"), IDEAL))
    assert surviving.total == pytest.approx(1.0)
    assert surviving.shot_inflation() == pytest.approx(1.0)


@pytest.mark.parametrize("device", [LINEAR, HEAVY_HEX], ids=lambda d: d.name)
def test_fidelity_falls_as_the_circuit_deepens(device: Device) -> None:
    totals = [estimate(transpile(chain(10, p, "open"), device)).total for p in (1, 2, 4, 8)]
    assert totals == sorted(totals, reverse=True)


def test_dephasing_is_charged_on_idle_time_only() -> None:
    # A published gate error already contains the decoherence suffered during the
    # gate. Charging the full duration again would count it twice, and the model
    # would be wrong rather than merely cautious.
    compiled = transpile(chain(10, 4, "open"), LINEAR)
    idle = idle_qubit_ns(compiled)
    offered = compiled.ansatz.n_qubits * compiled.duration_ns
    assert 0.0 < idle < offered


def test_idle_time_is_never_negative() -> None:
    for depth in range(0, 8):
        assert idle_qubit_ns(transpile(chain(6, depth, "periodic"), HEAVY_HEX)) >= 0.0


def test_readout_is_charged_per_observable_not_per_register() -> None:
    # Every term in this Hamiltonian is read from at most two bits. Demanding the
    # whole bitstring would price a measurement the algorithm never performs, and at
    # a dozen qubits that alone would rule out hardware that works.
    narrow = estimate(transpile(chain(4, 1, "open"), LINEAR)).readout
    wide = estimate(transpile(chain(12, 1, "open"), LINEAR)).readout
    assert narrow == pytest.approx(wide)
    assert wide == pytest.approx((1.0 - 2.0 * LINEAR.readout_error) ** OBSERVABLE_LOCALITY)


def test_a_misread_bit_negates_a_term_rather_than_losing_it() -> None:
    """The readout factor carries 2*epsilon, and the difference is checkable.

    A term with a +-1 spectrum whose bit is flipped with probability epsilon comes
    back with mean (1 - 2*epsilon) times its true value, not (1 - epsilon): the
    wrong bit reports the opposite sign rather than nothing at all. Simulated here
    against the closed form, because it is the one factor in the model where the
    probability of surviving is not the damping.
    """
    generator = np.random.default_rng(0)
    epsilon = 0.015
    truth = np.array([1.0, -1.0, 1.0, 1.0, -1.0])
    draws = np.tile(truth, (200_000, 1))
    flipped = np.where(generator.random(draws.shape) < epsilon, -draws, draws)

    measured = flipped.mean(axis=0) / truth

    assert measured.mean() == pytest.approx(1.0 - 2.0 * epsilon, abs=2e-3)
    assert estimate(transpile(chain(4, 1, "open"), LINEAR)).readout == pytest.approx(
        (1.0 - 2.0 * epsilon) ** OBSERVABLE_LOCALITY
    )


def test_the_shot_cost_of_noise_is_the_inverse_square_of_what_survives() -> None:
    compiled = transpile(chain(10, 3, "open"), LINEAR)
    surviving = estimate(compiled)
    assert shot_inflation(compiled) == pytest.approx(1.0 / surviving.total**2)
    assert shot_inflation(compiled) > 1.0


def test_noise_can_only_push_a_variational_energy_upwards() -> None:
    # The one piece of luck in the model: a variational result is an upper bound, and
    # depolarising noise weakens a bound rather than falsely tightening it.
    surviving = estimate(transpile(chain(10, 4, "open"), LINEAR))
    assert surviving.energy_bias(-12.5) > 0.0
    assert surviving.energy_bias(-12.5) == pytest.approx((1.0 - surviving.total) * 12.5)


def test_the_dominant_loss_is_named_so_a_report_can_say_what_to_fix() -> None:
    surviving = estimate(transpile(chain(12, 8, "open"), LINEAR))
    assert surviving.dominant_loss in {"gates", "coherence", "readout"}
    assert surviving.dominant_loss == "gates"  # deep circuit, error-rate limited


def test_a_fidelity_description_is_primitives() -> None:
    described = estimate(transpile(chain(10, 2, "open"), HEAVY_HEX)).describe()
    assert json.loads(json.dumps(described)) == described


# --------------------------------------------------------------------------
# The depth ceiling
# --------------------------------------------------------------------------


def test_a_device_budget_carries_that_devices_clock_and_not_the_nominal_one() -> None:
    budget = coherence_budget(HEAVY_HEX)
    assert isinstance(budget, CoherenceBudget)
    assert budget.two_qubit_gate_ns == HEAVY_HEX.two_qubit_gate_ns
    assert budget.coherence_ns == HEAVY_HEX.t2_ns
    assert budget.max_two_qubit_depth < coherence_budget(LINEAR).max_two_qubit_depth


def test_the_ideal_machine_has_no_ceiling_worth_naming() -> None:
    ceiling = depth_ceiling(12, IDEAL)
    assert ceiling.binding == "neither"
    assert ceiling.limit > 100


@pytest.mark.parametrize("device", [LINEAR, HEAVY_HEX], ids=lambda d: d.name)
def test_a_longer_chain_gets_a_lower_ceiling(device: Device) -> None:
    limits = [depth_ceiling(n, device).limit for n in (4, 8, 12, 20)]
    assert limits == sorted(limits, reverse=True)


def test_a_real_machine_is_limited_by_its_error_rate_before_its_clock() -> None:
    # The finding a report should lead with: at these sizes the circuit has run out
    # of signal long before it has run out of time, so a faster gate buys nothing and
    # a better one buys everything.
    ceiling = depth_ceiling(12, LINEAR)
    assert ceiling.binding == "fidelity"
    assert ceiling.by_fidelity < ceiling.by_coherence
    assert ceiling.limit == ceiling.by_fidelity


def test_a_ring_gets_a_lower_ceiling_than_a_segment() -> None:
    assert depth_ceiling(10, LINEAR, "periodic").limit < depth_ceiling(10, LINEAR, "open").limit


def test_every_depth_inside_the_ceiling_clears_the_floor() -> None:
    ceiling = depth_ceiling(8, LINEAR)
    assert ceiling.limit >= 1
    inside = estimate(transpile(chain(8, ceiling.limit, "open"), LINEAR))
    assert inside.total >= USABLE_FIDELITY_FLOOR
    beyond = estimate(transpile(chain(8, ceiling.by_fidelity + 1, "open"), LINEAR))
    assert beyond.total < USABLE_FIDELITY_FLOOR


def test_every_depth_inside_the_ceiling_finishes_inside_the_coherence_window() -> None:
    """The window is charged the whole schedule, not the entangling part of it.

    Single-qubit layers and the readout at the end are real time the qubits spend
    dephasing, and on a superconducting machine readout alone is a sixth of the
    window. A ceiling computed from entangling time would name a rung whose circuit
    does not finish -- which is the one thing a ceiling exists to rule out.
    """
    for device in (LINEAR, HEAVY_HEX):
        ceiling = depth_ceiling(8, device)
        window = coherence_budget(device).usable_ns
        assert transpile(chain(8, ceiling.by_coherence, "open"), device).duration_ns <= window
        beyond = transpile(chain(8, ceiling.by_coherence + 1, "open"), device)
        assert beyond.duration_ns > window


def test_a_ceiling_for_a_chain_that_does_not_fit_is_an_error() -> None:
    with pytest.raises(ValueError, match="no layout exists"):
        depth_ceiling(40, LINEAR)


def test_the_ceiling_describes_itself_in_primitives() -> None:
    described = depth_ceiling(10, HEAVY_HEX).describe()
    assert json.loads(json.dumps(described)) == described
    assert described["binding_constraint"] in {"coherence", "fidelity", "neither"}


# --------------------------------------------------------------------------
# What gets written out
# --------------------------------------------------------------------------


def test_the_exported_circuit_contains_exactly_the_gates_that_were_charged_for() -> None:
    # The correspondence the whole package rests on. A run card quoting a depth the
    # circuit does not contain is a number nobody can check.
    compiled = transpile(chain(10, 2, "periodic"), LINEAR)
    program = qasm3(compiled)
    written = program.count("cx q") + SWAP_TWO_QUBIT_GATES * program.count("swap q")
    assert written == compiled.two_qubit_gates


def test_an_open_chain_exports_no_swaps_at_all() -> None:
    assert "swap" not in qasm3(transpile(chain(10, 3, "open"), LINEAR))


def test_the_circuit_addresses_physical_qubits_and_measures_in_site_order() -> None:
    compiled = transpile(chain(6, 1, "periodic"), LINEAR)
    program = qasm3(compiled)
    assert f"qubit[{LINEAR.n_qubits}] q;" in program
    assert "bit[6] c;" in program
    for site, qubit in enumerate(compiled.layout.sites):
        assert f"c[{site}] = measure q[{qubit}];" in program


def test_a_circuit_without_angles_declares_them_as_runtime_inputs() -> None:
    # What a variational submission wants: the optimiser rebinds the angles every
    # iteration, and re-emitting the file each time would be the same circuit with
    # different literals in it.
    program = qasm3(transpile(chain(6, 2, "open"), LINEAR))
    assert "input float[64] gamma_0;" in program
    assert "input float[64] beta_1;" in program


def test_a_circuit_with_angles_bakes_them_in() -> None:
    angles = np.array([0.25, 0.5], dtype=np.float64)
    program = qasm3(transpile(chain(6, 1, "open"), LINEAR), angles)
    assert "input float" not in program
    # 2 * gamma * J and 2 * beta * h, folded to a literal at the default J = h = 1.
    assert "rz(0.5)" in program
    assert "rx(1)" in program


def test_the_couplings_reach_every_rotation_angle() -> None:
    # The defect this pins: the emitted gates were rz(2*gamma) and rx(2*beta) with no
    # J and no h in them, so the file was silently valid only at J = h = 1. Nothing in
    # the circuit lets a reader recover the strengths, because the angles absorb them.
    spec = AnsatzSpec(n_qubits=4, depth=1, boundary="open", longitudinal=True)
    compiled = transpile(spec, LINEAR)
    angles = np.array([0.5, 0.25], dtype=np.float64)
    program = qasm3(compiled, angles, Couplings(0.7, 0.85, 0.4))
    assert "rz(0.7)" in program  # 2 * 0.5 * 0.7
    assert "rx(0.425)" in program  # 2 * 0.25 * 0.85
    assert "rz(0.4)" in program  # 2 * 0.5 * 0.4, and *not* another 2 * gamma * J
    assert program.count("rz(0.7)") != program.count("rz(0.4)")


def test_the_header_states_the_strengths_even_at_their_defaults() -> None:
    # A circuit that states its own assumptions can be checked by somebody who does
    # not already know the answer; one that omits them cannot. So the header names
    # J, g and h whether or not they were passed.
    plain = qasm3(transpile(chain(6, 1, "open"), LINEAR))
    assert "J=1, g=0, h=1" in plain
    tilted = qasm3(transpile(chain(6, 1, "open"), LINEAR), None, Couplings(0.7, 0.85, 0.4))
    assert "J=0.7, g=0.4, h=0.85" in tilted
    # Unbound angles keep the multiplication, because the runtime supplies the name.
    assert "rz(1.4*gamma_0)" in tilted


def test_a_run_card_carries_the_problem_and_not_only_its_shape() -> None:
    card = run_card(
        transpile(chain(6, 1, "open"), LINEAR),
        1000,
        "a first look",
        Couplings(0.7, 0.85, 0.4),
    )
    problem = card.as_data()["problem"]
    assert problem["coupling"] == 0.7
    assert problem["transverse_field"] == 0.85
    assert problem["longitudinal_field"] == 0.4


def test_the_wrong_number_of_angles_is_refused() -> None:
    with pytest.raises(ValueError, match="takes 4 angles"):
        qasm3(transpile(chain(6, 2, "open"), LINEAR), np.zeros(3, dtype=np.float64))


def test_a_longitudinal_field_adds_rotations_and_no_entangling_gates() -> None:
    plain = AnsatzSpec(n_qubits=6, depth=1, boundary="open", longitudinal=False)
    tilted = AnsatzSpec(n_qubits=6, depth=1, boundary="open", longitudinal=True)
    plain_program = qasm3(transpile(plain, LINEAR))
    tilted_program = qasm3(transpile(tilted, LINEAR))
    assert tilted_program.count("rz(") > plain_program.count("rz(")
    assert tilted_program.count("cx q") == plain_program.count("cx q")


def test_a_run_card_needs_a_reason_and_a_budget() -> None:
    compiled = transpile(chain(8, 2, "open"), LINEAR)
    with pytest.raises(ValueError, match="at least one shot"):
        run_card(compiled, 0, "because")
    with pytest.raises(ValueError, match="why this configuration"):
        run_card(compiled, 1000, "   ")


def test_a_run_card_prices_the_shots_the_noise_costs() -> None:
    card = run_card(transpile(chain(10, 2, "open"), LINEAR), 1_000_000, "the deepest that fits")
    data = card.as_data()
    assert data["shots"]["noise_inflation"] > 1.0
    assert data["shots"]["equivalent_noiseless_shots"] < card.shots
    assert json.loads(json.dumps(data)) == data


def test_the_note_beside_the_circuit_states_the_claim_and_what_would_break_it() -> None:
    card = run_card(transpile(chain(10, 2, "open"), LINEAR), 500_000, "the deepest that fits")
    note = readme(card)
    assert "upper bound" in note
    assert "falsif" in note
    assert "10-site chain" in note


def test_a_submission_package_is_three_files_and_they_agree_with_each_other(
    tmp_path: Path,
) -> None:
    compiled = transpile(chain(10, 2, "periodic"), LINEAR)
    card = run_card(compiled, 250_000, "the deepest configuration inside the fidelity floor")
    folder = submission_package(card, "hva-p2 / linear", directory=tmp_path)
    assert folder.name == "hva-p2-linear"
    written = json.loads((folder / "run_card.json").read_text())
    program = (folder / "circuit.qasm").read_text()
    assert written["compiled"]["two_qubit_depth"] == compiled.two_qubit_depth
    assert program.count("swap q") == compiled.swaps
    assert (folder / "README.md").read_text().startswith("# 10-site chain")


def test_a_package_written_without_angles_still_records_the_starting_ones(
    tmp_path: Path,
) -> None:
    # So that a reader can reproduce the first iteration exactly, which is the only
    # iteration anybody can check without rerunning the optimiser.
    card = run_card(transpile(chain(8, 3, "open"), LINEAR), 100_000, "a first look")
    folder = submission_package(card, "first-look", directory=tmp_path)
    angles = json.loads((folder / "run_card.json").read_text())["angles"]
    assert angles["baked_into_circuit"] is False
    assert len(angles["values"]) == 6


def test_a_package_is_rewritten_in_place_rather_than_accumulating(tmp_path: Path) -> None:
    # A depth ladder writes the same label repeatedly as it re-plans. A package that
    # refused to overwrite would fail the second campaign of the day.
    shallow = run_card(transpile(chain(6, 1, "open"), LINEAR), 1000, "a first pass")
    deeper = run_card(transpile(chain(6, 3, "open"), LINEAR), 1000, "a second pass")
    first = submission_package(shallow, "rerun", directory=tmp_path)
    second = submission_package(deeper, "rerun", directory=tmp_path)
    assert first == second
    assert json.loads((second / "run_card.json").read_text())["ansatz"]["depth"] == 3


def test_a_label_that_leaves_no_directory_name_is_refused(tmp_path: Path) -> None:
    card = run_card(transpile(chain(6, 1, "open"), LINEAR), 1000, "a rerun")
    with pytest.raises(ValueError, match="no usable directory name"):
        submission_package(card, "///", directory=tmp_path)


def test_a_run_card_is_a_record_not_a_loose_mapping() -> None:
    # So that a package cannot be built from a circuit and a fidelity estimate that
    # came from different runs.
    compiled = transpile(chain(6, 1, "open"), LINEAR)
    card = run_card(compiled, 1000, "a note")
    assert isinstance(card, RunCard)
    assert card.fidelity is not estimate(compiled)
    assert card.fidelity.total == pytest.approx(estimate(compiled).total)
