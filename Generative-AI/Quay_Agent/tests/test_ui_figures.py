"""The cartoons draw, and they draw the schedule the costing actually used.

:mod:`src.ui.figures` imports no Streamlit, which is the whole reason it is a module
of its own -- so it can be exercised here directly, with no ``AppTest`` and no
session, and a broken figure is a fast red test rather than a blank panel in
somebody's browser.

Two kinds of check. The first is that every figure draws at all, including at the
edges the interface can actually reach: a two-magnet chain, a chain with the push
switched off, a machine with no wiring problems and a machine with plenty. The
second is the one worth having -- that the circuit drawing is built from
:attr:`~src.physics.quantum.ansatz.AnsatzSpec.rounds`, the same grouping the depth
arithmetic prices. A picture that showed a schedule nobody costed would be a
confident, well-drawn lie, and it is exactly the kind that survives review.
"""

from __future__ import annotations

import threading
from io import BytesIO
from typing import Any

import matplotlib
import pytest

matplotlib.use("Agg")

import numpy as np
from matplotlib.figure import Figure

from src.hardware.devices import device_for, device_names
from src.hardware.transpile import fits, transpile
from src.physics import quantumness
from src.physics.model import MAX_SITES_STATEVECTOR, TFIMSpec
from src.physics.quantum.ansatz import AnsatzSpec, bond_rounds
from src.physics.reference import free_fermions
from src.ui import figures

RATIOS = (0.2, 0.7, 1.0, 2.0)
"""The fields the spreading strip is drawn at, matching the Lab page."""


def spec(n_sites: int = 6, field: float = 1.0, boundary: str = "periodic") -> TFIMSpec:
    """Build a chain to draw.

    Args:
        n_sites: Chain length.
        field: The transverse field.
        boundary: Ring or segment.

    Returns:
        The specification.
    """
    return TFIMSpec(n_sites=n_sites, coupling=1.0, field=field, boundary=boundary)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Every cartoon draws
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_sites", (2, 4, 6, 8))
@pytest.mark.parametrize("field", (0.0, 1.0, 5.0))
def test_the_superposition_cartoon_draws_at_every_size_and_field(
    n_sites: int, field: float
) -> None:
    # Zero field is the classical limit, where the state is one arrangement and the
    # share bars are one full bar and seven empty ones -- the case where a naive
    # axis limit divides by zero.
    drawn = figures.superposition_figure(quantumness.superposition(spec(n_sites, field)))
    assert isinstance(drawn, Figure)


def test_the_spreading_strip_draws_one_panel_per_field() -> None:
    pictures = tuple(quantumness.superposition(spec(field=ratio)) for ratio in RATIOS)
    drawn = figures.spreading_figure(pictures, RATIOS)
    assert len(drawn.axes) == len(RATIOS)


@pytest.mark.parametrize("boundary", ("open", "periodic"))
def test_the_phase_diagram_draws_with_and_without_this_chains_own_gap(boundary: str) -> None:
    # An open chain has no closed-form gap, so the second curve is absent. The
    # figure has to be a figure either way rather than raising on a None.
    chain = spec(boundary=boundary)
    ratios = np.linspace(0.0, 2.5, 41)
    infinite = np.asarray([free_fermions.gap_thermodynamic(1.0, ratio) for ratio in ratios])
    finite = (
        None
        if free_fermions.unsupported_reason(chain) is not None
        else np.asarray(
            [free_fermions.gap(spec(field=ratio, boundary=boundary)) for ratio in ratios]
        )
    )
    assert isinstance(figures.phase_diagram_figure(chain, ratios, infinite, finite), Figure)


def test_the_loop_and_the_lattice_need_nothing_but_their_shape() -> None:
    # Neither depends on a solver, which is why they are the two cartoons a reader
    # sees before any number has been computed.
    assert isinstance(figures.hybrid_loop_figure(), Figure)
    assert isinstance(figures.dual_lattice_figure(6, 5, periodic=True), Figure)
    assert isinstance(figures.dual_lattice_figure(2, 1, periodic=False), Figure)


def test_the_method_map_needs_nothing_at_all() -> None:
    # The one cartoon on the page that is not about this chain. It is the family of
    # four methods, so it takes no arguments and cannot go stale against a setting.
    assert isinstance(figures.method_map_figure(), Figure)
    assert len(figures.METHODS) == 4


@pytest.mark.parametrize("exact", (None, -7.0))
def test_the_race_draws_curves_of_different_lengths(exact: float | None) -> None:
    # The third arm of the race stops on its first step, so one history has a single
    # point in it. A line through one point draws nothing, which would read as a
    # missing curve rather than as the result -- the figure marks it instead, and
    # this is the case that would regress silently.
    runs = (
        ("informed", (-5.0, -6.0, -6.8)),
        ("less informed", (-4.0, -5.0, -6.0, -6.5, -6.8)),
        ("cold", (-4.0,)),
    )
    assert isinstance(figures.method_race_figure(runs, exact), Figure)


def test_the_race_survives_an_arm_that_produced_no_history() -> None:
    # A run that never recorded a step is legal -- depth zero returns a closed form
    # and optimises nothing -- and it must not take the other curves down with it.
    runs = (("ran", (-5.0, -6.0)), ("did not run", ()))
    assert isinstance(figures.method_race_figure(runs, -7.0), Figure)


@pytest.mark.parametrize("name", device_names())
def test_every_machine_can_be_drawn_with_a_chain_on_it(name: str) -> None:
    # Three topologies -- a row, a heavy hexagon and all-to-all -- and each is laid
    # out by a different branch of `_positions`. A layout that silently returned
    # the wrong shape would draw every qubit on top of every other.
    machine = device_for(name)
    ansatz = AnsatzSpec(n_qubits=6, depth=2, boundary="open")
    if fits(6, machine) is not None:
        pytest.skip(f"{name} cannot hold a 6-magnet chain")
    compiled = transpile(ansatz, machine)
    drawn = figures.wiring_figure(
        machine.n_qubits,
        machine.coupling,
        compiled.layout.sites,
        tuple(bond.path for bond in compiled.bonds if bond.distance > 1),
        machine.name,
    )
    assert isinstance(drawn, Figure)


@pytest.mark.parametrize("exact", (None, -7.0))
def test_the_result_curves_draw_with_and_without_a_reference(exact: float | None) -> None:
    # Above the exact-diagonalisation cap there is no true answer to draw, and the
    # curves have to degrade to "here is what it reached" rather than plotting a
    # distance to a number nobody has.
    history = (-5.0, -6.0, -6.5, -6.9)
    assert isinstance(figures.convergence_figure(history, exact), Figure)
    assert isinstance(figures.ladder_figure((1, 2, 3), (-5.0, -6.5, -6.9), exact), Figure)
    times = np.linspace(0.0, 3.0, 10)
    assert isinstance(
        figures.imaginary_time_figure(times, tuple(history) * 2 + (-6.9,) * 2, exact), Figure
    )


def test_a_rung_that_landed_exactly_on_the_answer_still_appears() -> None:
    # The ladder is drawn on a log axis, and a distance of exactly zero has no place
    # on one. Clipped rather than dropped: a missing point reads as a failed run.
    drawn = figures.ladder_figure((1, 2), (-7.0, -7.0), -7.0)
    plotted = np.asarray(drawn.axes[0].lines[0].get_ydata())
    assert plotted.size == 2
    assert bool(np.all(plotted > 0.0))


# --------------------------------------------------------------------------
# The drawing shows the schedule that was priced
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_qubits", "boundary"),
    ((6, "open"), (6, "periodic"), (7, "periodic"), (2, "open")),
)
def test_the_rounds_the_picture_draws_are_the_rounds_the_depth_prices(
    n_qubits: int, boundary: str
) -> None:
    # The point of exposing `rounds` rather than recomputing a colouring in the
    # interface. An odd ring needs three rounds and an even one needs two; a second
    # colouring written for the drawing could easily disagree, and then the picture
    # would show a circuit half the depth the feasibility verdict was based on.
    ansatz = AnsatzSpec(n_qubits=n_qubits, depth=2, boundary=boundary)  # type: ignore[arg-type]
    assert len(ansatz.rounds) == ansatz.two_qubit_rounds_per_layer
    assert sorted(bond for group in ansatz.rounds for bond in group) == sorted(ansatz.bonds)


@pytest.mark.parametrize(
    ("n_qubits", "boundary"),
    ((6, "open"), (6, "periodic"), (7, "periodic")),
)
def test_no_round_ever_puts_two_gates_on_the_same_magnet(n_qubits: int, boundary: str) -> None:
    # The property that makes a round a round. If it failed the depth arithmetic
    # would be wrong everywhere downstream, not just in the drawing.
    ansatz = AnsatzSpec(n_qubits=n_qubits, depth=1, boundary=boundary)  # type: ignore[arg-type]
    for group in ansatz.rounds:
        touched = [qubit for bond in group for qubit in bond]
        assert len(touched) == len(set(touched)), f"a round reuses a magnet: {group}"


def test_the_grouping_is_the_one_the_greedy_colouring_gives() -> None:
    assert bond_rounds(((0, 1), (1, 2), (2, 3))) == (((0, 1), (2, 3)), ((1, 2),))


@pytest.mark.parametrize("depth", (1, 2, 5, 12))
def test_the_circuit_drawing_never_grows_past_what_can_be_read(depth: int) -> None:
    # A twelve-layer circuit on twenty magnets is not a picture, it is a texture.
    # The drawing caps both and says in its own title what it left out, which is the
    # difference between an abbreviation and a misrepresentation.
    ansatz = AnsatzSpec(n_qubits=12, depth=depth, boundary="open")
    drawn = figures.circuit_figure(20, depth, ansatz.bonds, ansatz.rounds)
    title = drawn.axes[0].get_title()
    assert "more magnets" in title
    assert ("more identical layers" in title) == (depth > figures.MAX_DRAWN_LAYERS)


def test_a_small_circuit_claims_nothing_is_missing_from_its_drawing() -> None:
    ansatz = AnsatzSpec(n_qubits=4, depth=2, boundary="open")
    title = figures.circuit_figure(4, 2, ansatz.bonds, ansatz.rounds).axes[0].get_title()
    assert "not drawn" not in title


def test_a_ring_says_that_its_closing_bond_is_not_in_the_picture() -> None:
    # It cannot be drawn without a connector running through every gate between the
    # two ends, so it is left out -- and a bond left out silently is a gate count a
    # reader cannot reconcile with the picture in front of them.
    ansatz = AnsatzSpec(n_qubits=6, depth=1, boundary="periodic")
    title = figures.circuit_figure(6, 1, ansatz.bonds, ansatz.rounds).axes[0].get_title()
    assert "closes the ring" in title


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------


def test_a_machine_wired_in_a_row_is_drawn_as_a_row() -> None:
    # Not cosmetic. A force simulation lays a path out as a wiggle, and the shape of
    # the machine's wiring is the information the picture exists to carry.
    machine = device_for("linear")
    places = figures._positions(machine.n_qubits, machine.coupling)
    assert np.allclose(places[:, 1], 0.0)


def test_the_same_machine_is_drawn_the_same_way_twice() -> None:
    # The layout is seeded. A diagram that rearranged itself between reruns is one
    # nobody can get familiar with, and every rerun of a Streamlit page redraws it.
    machine = device_for("heavy-hex-27")
    first = figures._positions(machine.n_qubits, machine.coupling)
    second = figures._positions(machine.n_qubits, machine.coupling)
    assert np.array_equal(first, second)


# --------------------------------------------------------------------------
# The exact solution, traced across the field
# --------------------------------------------------------------------------


def swept_arrays(n_sites: int = 8, points: int = 21) -> dict[str, Any]:
    """One exact sweep, as the arrays the figures take.

    Args:
        n_sites: Chain length.
        points: How many field values.

    Returns:
        The payload. Taken from the grader's own bench rather than made up, so the
        figures are exercised on the shape the interface actually hands them.
    """
    from src.physics.registry import field_sweep_bench

    return field_sweep_bench()(
        n_sites=n_sites,
        curves=[
            "spectrum",
            "energy",
            "magnetisation",
            "energy_derivatives",
            "magnetisation_derivatives",
        ],
        points=points,
    )


def test_the_spectrum_figure_draws_every_level_and_leaves_room_for_its_legend() -> None:
    payload = swept_arrays()
    ratios = np.array(payload["ratio"], dtype=float)
    excitations = tuple(tuple(row) for row in payload["excitations"])
    drawn = figures.spectrum_sweep_figure(
        ratios, excitations, np.array([2.0 * abs(1.0 - r) for r in ratios])
    )
    assert isinstance(drawn, Figure)
    axes = drawn.axes[0]
    # One line per level above the ground state, plus the thermodynamic reference.
    assert len(axes.lines) >= len(excitations[0]) - 1
    bottom, top = axes.get_ylim()
    assert bottom == 0.0
    # Headroom above the highest curve, so the legend does not sit on it.
    highest = max(max(row[1:]) for row in excitations)
    assert top > highest


def test_the_spectrum_figure_is_measured_from_the_ground_state() -> None:
    # Absolute levels all slide down the page together as the field rises, which
    # squeezes the structure the question is about into the width of a line.
    payload = swept_arrays()
    excitations = tuple(tuple(row) for row in payload["excitations"])
    assert all(row[0] == 0.0 for row in excitations)


def test_the_observable_figure_draws_a_reference_only_when_given_one() -> None:
    payload = swept_arrays()
    ratios = np.array(payload["ratio"], dtype=float)
    values = np.array(payload["energy_density"], dtype=float)
    alone = figures.observable_sweep_figure(ratios, values, "$E_0/L$")
    paired = figures.observable_sweep_figure(ratios, values, "$E_0/L$", values * 0.99)
    assert len(paired.axes[0].lines) == len(alone.axes[0].lines) + 1


def test_the_derivative_figure_stacks_three_panels_on_one_axis() -> None:
    payload = swept_arrays()
    ratios = np.array(payload["ratio"], dtype=float)
    drawn = figures.derivative_sweep_figure(
        ratios,
        np.array(payload["magnetisation"], dtype=float),
        np.array(payload["magnetisation_slope"], dtype=float),
        np.array(payload["magnetisation_curvature"], dtype=float),
        ("a", "b", "c"),
    )
    assert len(drawn.axes) == 3
    # The shared axis is labelled once, at the bottom, and the panels are labelled
    # in the order they were passed.
    assert [axes.get_xlabel() for axes in drawn.axes][:2] == ["", ""]
    assert "h/J" in drawn.axes[-1].get_xlabel()
    assert [axes.get_ylabel() for axes in drawn.axes] == ["a", "b", "c"]


def test_every_sweep_figure_marks_the_critical_field() -> None:
    # The whole point of these curves is what happens near h/J = 1, and a reader who
    # has to find it by eye on the axis has been handed a chart rather than a figure.
    payload = swept_arrays()
    ratios = np.array(payload["ratio"], dtype=float)
    values = np.array(payload["magnetisation"], dtype=float)
    drawn = [
        figures.observable_sweep_figure(ratios, values, "$m$"),
        figures.spectrum_sweep_figure(ratios, tuple(tuple(row) for row in payload["excitations"])),
        figures.derivative_sweep_figure(ratios, values, values, values, ("a", "b", "c")),
    ]
    for one in drawn:
        for axes in one.axes:
            marks = [
                line
                for line in axes.lines
                if list(np.asarray(line.get_xdata(), dtype=float))
                == [figures.CRITICAL_RATIO, figures.CRITICAL_RATIO]
            ]
            assert marks, "no critical-field marker on a sweep panel"


def test_no_problem_this_project_can_pose_is_drawn_short() -> None:
    """A 12-magnet question answered under an 8-magnet diagram is the wrong picture.

    The wire cap was 8 while the model's own ceiling is 16, so every chain longer
    than eight -- which is most of the ones the README advertises -- was drawn
    truncated. It was reported twice by a reader before it was found here. The
    figure's height grows with the wire count and its width does not, so there was
    never a legibility reason for the cap to be the lower number.
    """
    assert figures.MAX_DRAWN_QUBITS >= MAX_SITES_STATEVECTOR, (
        "a chain the application will happily solve cannot be drawn in full"
    )


def test_a_chain_at_the_ceiling_is_drawn_whole() -> None:
    ansatz = AnsatzSpec(n_qubits=MAX_SITES_STATEVECTOR, depth=4, boundary="open")
    title = (
        figures.circuit_figure(MAX_SITES_STATEVECTOR, 4, ansatz.bonds, ansatz.rounds)
        .axes[0]
        .get_title()
    )
    assert "not drawn" not in title, title


def test_the_drawing_widens_for_layers_rather_than_squeezing_them() -> None:
    """Otherwise raising the layer cap just makes every layer thinner."""
    ansatz = AnsatzSpec(n_qubits=8, depth=6, boundary="open")
    narrow = figures.circuit_figure(8, 2, ansatz.bonds, ansatz.rounds).get_size_inches()[0]
    wide = figures.circuit_figure(8, 6, ansatz.bonds, ansatz.rounds).get_size_inches()[0]
    assert wide > narrow


def test_two_pages_can_draw_maths_at_the_same_time() -> None:
    """Maths is parsed by one object shared by the whole process.

    Streamlit gives each script run its own thread. Two figures carrying maths,
    drawn at once, corrupted that parser's state and one of them died with an empty
    ``ParseException`` raised from inside matplotlib -- a traceback with nothing in
    it naming this project. It took the Lab page down on load. The lock is the fix;
    this is the test that would have caught it.
    """
    spec = TFIMSpec(n_sites=6, coupling=1.0, field=1.0, boundary="open")
    ratios = np.linspace(0.0, 2.0, 40)
    gap = np.abs(1.0 - ratios)
    failures: list[str] = []

    def draw() -> None:
        for _ in range(6):
            try:
                drawn = figures.phase_diagram_figure(spec, ratios, gap, None)
                with figures.DRAWING:
                    drawn.savefig(BytesIO(), format="png")
            except Exception as error:  # the point is to record any of them
                failures.append(f"{type(error).__name__}: {error}")

    threads = [threading.Thread(target=draw) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not failures, failures[:3]


def test_every_figure_the_interface_draws_holds_the_lock() -> None:
    """Every public builder is decorated.

    One added without it brings the crash straight back, as an intermittent failure
    on somebody else's machine. Public names only: ``_styled_figure`` hands back an
    empty canvas and is only reached from inside a builder already holding the lock.
    """
    unguarded = [
        name
        for name in dir(figures)
        if name.endswith("_figure")
        and not name.startswith("_")
        and callable(getattr(figures, name))
        and getattr(getattr(figures, name), "__wrapped__", None) is None
    ]
    assert not unguarded, unguarded
