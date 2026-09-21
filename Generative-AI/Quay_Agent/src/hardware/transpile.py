r"""Turning a circuit that assumes anything into one a named machine can run.

Four stages, in the order a compiler does them, each of which can only make the
circuit worse: *decompose* rewrites the algorithm\'s gates in the machine\'s gate
set, *place* decides which physical qubit holds which spin, *route* pays for
every interaction the wiring does not provide, and *schedule* packs what is left
into layers and puts a duration on it. The output is two-qubit depth on this
machine, plus the audit trail that produced it.

:class:`~src.physics.quantum.ansatz.AnsatzSpec` already prices a circuit, and its
price is correct in the abstract -- an open chain needs four layers of entangling
gates per ansatz layer, however long it is. That is the number a paper quotes,
and it is unreachable on any machine whose wiring does not match the problem.
This module computes the gap.

An open chain on a line costs nothing to route: every bond is already an edge, so
the depth equals the abstract depth exactly. The tests assert that equality,
because it is the check that this module has not invented cost.

A periodic chain has one bond joining its two ends, which on a line are as far
apart as two qubits can be. Routed naively that is a serial chain of :math:`L-1`
SWAPs every other gate waits behind, so depth grows with the chain. Placed folded
-- site order :math:`0, L-1, 1, L-2, \dots` -- every bond is a distance of at
most two, nothing blocks the rest, and the routing runs in parallel: the SWAP
count barely moves while the depth falls by roughly the length of the chain.
:func:`place` makes that choice on the boundary condition alone.

Modelled: shortest-path routing over the real coupling graph, SWAPs decomposed
into the machine\'s entangling gate, parallel execution of operations sharing no
qubit, and the single-qubit cost of synthesising a ``CX``.

Not modelled: crosstalk, gate-dependent durations, pulse-level scheduling, or a
compiler clever enough to keep a permuted layout between layers instead of
undoing its own SWAPs. The last is the significant omission and it makes every
figure here pessimistic, which is the safe direction: a verdict of "this fits"
computed with too much cost is still true.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

from src.hardware.devices import Device
from src.physics.quantum.ansatz import AnsatzSpec

Placement = Literal["path", "folded_path"]
"""How logical sites were laid out on physical qubits. See :func:`place`."""

SWAP_TWO_QUBIT_GATES = 3
"""Entangling gates in one SWAP.

A SWAP is three ``CX`` gates and no amount of compilation makes it two. This is the
constant that makes routing expensive rather than merely inconvenient, and it is why
a layout that avoids a SWAP is worth more than a gate optimiser that shortens one.
"""

BOND_TWO_QUBIT_GATES = 2
r"""Entangling gates in one :math:`\hat\sigma^z\hat\sigma^z` rotation.

The rotation :math:`e^{-i\gamma\hat\sigma^z_i\hat\sigma^z_j}` compiles to
``CX``, a single-qubit :math:`\hat R^z(2\gamma)` on the target, and ``CX`` back. The
angle lives entirely in the middle rotation, which is why a whole layer of these
costs the same whatever the angles are, and why the depth ladder can be priced
before any optimisation has run.
"""

SINGLE_QUBIT_PER_CX: dict[str, int] = {
    "cx": 0,
    "cz": 2,
    "ecr": 3,
}
"""Single-qubit gates needed to build one ``CX`` from the machine's native gate.

``CZ`` needs a Hadamard on the target before and after. ``ECR`` -- the echoed
cross-resonance gate superconducting hardware actually calibrates -- needs a frame
change and a rotation on each side. Neither costs meaningful time next to the
entangling gate itself, and both are counted anyway, because a run card that says
"1,248 gates" when the machine will execute 3,100 of them is not a run card.
"""


@dataclass(frozen=True, slots=True)
class Layout:
    """Which physical qubit holds which spin.

    Attributes:
        device: Name of the machine this layout is for. A layout is meaningless
            without it, and carrying the name means a run card cannot be read
            against the wrong machine.
        sites: Physical qubit for each logical site, indexed by site. Site ``i``
            lives on qubit ``sites[i]``.
        strategy: Which rule produced it. See :func:`place`.
    """

    device: str
    sites: tuple[int, ...]
    strategy: Placement

    def __post_init__(self) -> None:
        """Reject a layout that puts two spins on one qubit.

        Raises:
            ValueError: If the same physical qubit appears twice.
        """
        if len(set(self.sites)) != len(self.sites):
            raise ValueError(f"layout reuses a physical qubit: {self.sites}")

    @property
    def n_sites(self) -> int:
        """How many spins were placed."""
        return len(self.sites)

    def physical(self, site: int) -> int:
        """Find the qubit holding one spin.

        Args:
            site: The logical site index.

        Returns:
            The physical qubit index.

        Raises:
            IndexError: If the site was not placed.
        """
        if not 0 <= site < len(self.sites):
            raise IndexError(f"site {site} was not placed; the layout holds {len(self.sites)}")
        return self.sites[site]

    def describe(self) -> dict[str, Any]:
        """Render the layout as plain data for a run card or a log line.

        Returns:
            The device name, the strategy and the site-to-qubit assignment.
        """
        return {
            "device": self.device,
            "strategy": self.strategy,
            "n_sites": self.n_sites,
            "sites_to_qubits": list(self.sites),
        }


@dataclass(frozen=True, slots=True)
class RoutedBond:
    r"""What one :math:`\hat\sigma^z\hat\sigma^z` interaction costs on this machine.

    Attributes:
        sites: The logical pair, as the model names it.
        qubits: The physical pair, after placement.
        distance: Edges between those qubits on the coupling graph. One means the
            wiring already provides the interaction and nothing is owed.
        swaps: SWAPs the interaction needs, counting the ones that put the layout
            back. Zero at distance one.
        path: The physical qubits the interaction is routed along, from the first
            endpoint to the second. Length one greater than :attr:`distance`, and
            equal to the two endpoints when the wiring already provides the
            interaction. Carried rather than recomputed because the exported circuit
            has to emit the same SWAPs that were charged for -- a run card whose
            gate count disagrees with the cost model is worse than no run card.
        two_qubit_depth: Entangling layers this interaction takes on its own. Its
            SWAPs are serial -- each one moves a qubit the next one needs -- so this
            grows with distance and does not benefit from the machine being idle
            elsewhere.
    """

    sites: tuple[int, int]
    qubits: tuple[int, int]
    distance: int
    swaps: int
    path: tuple[int, ...]
    two_qubit_depth: int

    @property
    def routed(self) -> bool:
        """Whether the wiring failed to provide this interaction directly."""
        return self.swaps > 0

    @property
    def footprint(self) -> frozenset[int]:
        """Every physical qubit this interaction occupies while it runs.

        The two endpoints plus every qubit the SWAPs walk through. Two interactions
        can share a layer exactly when their footprints are disjoint, so this is
        what the scheduler colours on.
        """
        return frozenset(self.path)


@dataclass(frozen=True, slots=True)
class Transpiled:
    """One ansatz, priced against one machine.

    Attributes:
        device: The machine it was compiled for.
        ansatz: The specification it was compiled from, unchanged.
        layout: Where the spins went.
        bonds: What each interaction cost, in the order the model lists them.
        rounds_per_layer: How many groups of mutually disjoint interactions one
            ansatz layer needs. Two for an open chain -- the even bonds, then the
            odd ones -- and more once routing corridors start colliding.
        two_qubit_depth_per_layer: Entangling layers in one ansatz layer.
        swaps_per_layer: SWAPs in one ansatz layer.
        two_qubit_depth: Entangling layers in the whole circuit.
        two_qubit_gates: Entangling gates in the whole circuit, in the machine's
            native gate.
        single_qubit_gates: Single-qubit gates in the whole circuit, including the
            ones spent synthesising each ``CX``.
        duration_ns: How long the machine is occupied, from the first gate to the
            end of readout.
    """

    device: Device
    ansatz: AnsatzSpec
    layout: Layout
    bonds: tuple[RoutedBond, ...]
    rounds_per_layer: int
    two_qubit_depth_per_layer: int
    swaps_per_layer: int
    two_qubit_depth: int
    two_qubit_gates: int
    single_qubit_gates: int
    duration_ns: float

    @property
    def swaps(self) -> int:
        """SWAPs in the whole circuit."""
        return self.swaps_per_layer * self.ansatz.depth

    @property
    def routing_overhead(self) -> float:
        """How much deeper the machine makes the circuit, as a ratio.

        One means the wiring cost nothing -- the machine ran the circuit the
        algorithm asked for. Anything above one is depth spent moving information
        rather than computing with it, and it is the honest headline of this whole
        module.

        Returns:
            Depth on this machine divided by depth in the abstract. One when the
            abstract circuit is empty, so that a zero-depth ansatz is not reported
            as infinitely overrun.
        """
        abstract = self.ansatz.two_qubit_depth
        return self.two_qubit_depth / abstract if abstract else 1.0

    @property
    def duration_us(self) -> float:
        """How long the circuit occupies the machine, in microseconds."""
        return self.duration_ns / 1000.0

    def describe(self) -> dict[str, Any]:
        """Render the compiled circuit as plain data.

        Every value is a primitive, so this survives a log line, a JSON run card
        and a language model's context without a custom encoder.

        Returns:
            A flat mapping of what the circuit costs on this machine, alongside the
            abstract figures it is being compared against.
        """
        return {
            "device": self.device.name,
            "native_two_qubit": self.device.native_two_qubit,
            "n_sites": self.ansatz.n_qubits,
            "ansatz_depth": self.ansatz.depth,
            "boundary": self.ansatz.boundary,
            "layout": self.layout.strategy,
            "routed_bonds": sum(1 for bond in self.bonds if bond.routed),
            "swaps": self.swaps,
            "rounds_per_layer": self.rounds_per_layer,
            "two_qubit_depth": self.two_qubit_depth,
            "abstract_two_qubit_depth": self.ansatz.two_qubit_depth,
            "routing_overhead": round(self.routing_overhead, 3),
            "two_qubit_gates": self.two_qubit_gates,
            "single_qubit_gates": self.single_qubit_gates,
            "duration_us": round(self.duration_us, 3),
            "t2_us": round(self.device.t2_ns / 1000.0, 1),
            "fraction_of_t2": round(self.duration_ns / self.device.t2_ns, 4),
        }


def fits(n_sites: int, device: Device) -> str | None:
    """Say why a chain cannot be put on a machine, or nothing if it can.

    A refusal rather than an exception, and a sentence rather than a flag, because
    the caller is a planner deciding between configurations: it needs to record why
    this one was dropped and try a smaller one, not stop.

    Args:
        n_sites: Length of the chain.
        device: The machine.

    Returns:
        A sentence naming the obstruction, or ``None`` when the chain fits.
    """
    if n_sites < 2:
        return f"a chain needs at least 2 sites, got {n_sites}"
    if n_sites > device.n_qubits:
        return (
            f"{device.name} has {device.n_qubits} qubits and the chain needs "
            f"{n_sites}; no layout exists"
        )
    return None


def place(n_sites: int, device: Device, boundary: Literal["open", "periodic"]) -> Layout:
    r"""Decide which physical qubit holds which spin.

    Two strategies, chosen by the boundary condition, because the boundary
    condition is what determines whether the problem's interaction graph is a path
    or a cycle:

    ``"path"`` -- for an open chain. Sites go in order along the longest run of
    connected qubits the machine has. Every nearest-neighbour bond then lands on an
    edge and routing costs nothing.

    ``"folded_path"`` -- for a ring. Sites go along the same run in the order
    :math:`0, L-1, 1, L-2, \dots`, which starts the two ends of the chain next to
    each other. The wrap-around bond, which would otherwise span the machine, becomes
    a single edge; every other bond becomes a distance of two, which costs one SWAP
    and blocks nothing. This is the whole reason a ring is runnable at all: the naive
    layout turns one bond into a serial SWAP chain the length of the register, and a
    serial chain is depth, which is coherence, which is the budget that binds.

    Args:
        n_sites: Length of the chain.
        device: The machine.
        boundary: ``"open"`` for a segment, ``"periodic"`` for a ring.

    Returns:
        The layout.

    Raises:
        ValueError: If the chain cannot be placed -- see :func:`fits` for the
            refusal a planner should use instead of catching this.
    """
    refusal = fits(n_sites, device)
    if refusal is not None:
        raise ValueError(refusal)
    corridor = device.longest_path()
    if len(corridor) < n_sites:
        corridor = corridor + tuple(q for q in range(device.n_qubits) if q not in set(corridor))
    corridor = corridor[:n_sites]
    if boundary == "open":
        return Layout(device=device.name, sites=corridor, strategy="path")
    return Layout(device=device.name, sites=_fold(corridor), strategy="folded_path")


def _fold(corridor: tuple[int, ...]) -> tuple[int, ...]:
    r"""Interleave a run of qubits so that the two ends of a ring meet.

    The corridor is a line of physical qubits; the sites are dealt onto it from both
    ends of the chain inwards, so that site :math:`0` and site :math:`L-1` occupy the
    first two qubits. Reading it the other way, physical position :math:`2k` holds
    site :math:`k` and position :math:`2k+1` holds site :math:`L-1-k`.

    Args:
        corridor: Physical qubits in order along a connected run.

    Returns:
        The physical qubit for each logical site, indexed by site.
    """
    length = len(corridor)
    sites = [0] * length
    low, high = 0, length - 1
    for position, qubit in enumerate(corridor):
        if position % 2 == 0:
            sites[low] = qubit
            low += 1
        else:
            sites[high] = qubit
            high -= 1
    return tuple(sites)


def route(
    layout: Layout, device: Device, bonds: tuple[tuple[int, int], ...]
) -> tuple[
    RoutedBond,
    ...,
]:
    """Price every interaction against the wiring that has to carry it.

    An interaction between qubits :math:`d` edges apart needs :math:`d-1` SWAPs to
    bring them together, and :math:`d-1` more to put every qubit back where the next
    layer expects it. Undoing the SWAPs is the conservative choice -- a production
    compiler tracks the permutation and carries it forward instead -- and it is taken
    deliberately, because a feasibility verdict computed with too much cost errs
    towards "no", which is the direction that cannot mislead.

    Args:
        layout: Where the spins were placed.
        device: The machine.
        bonds: The coupled pairs, as logical site indices.

    Returns:
        One record per bond, in the order given.

    Raises:
        ValueError: If a bond's endpoints have no route between them at all, which
            means the layout spans two disconnected fragments of the lattice.
    """
    routed: list[RoutedBond] = []
    for left_site, right_site in bonds:
        left = layout.physical(left_site)
        right = layout.physical(right_site)
        distance = device.distance(left, right)
        if distance < 1:
            raise ValueError(
                f"qubits {left} and {right} are not connected on {device.name}; "
                f"sites {left_site} and {right_site} cannot interact"
            )
        swaps = 2 * (distance - 1)
        routed.append(
            RoutedBond(
                sites=(left_site, right_site),
                qubits=(left, right),
                distance=distance,
                swaps=swaps,
                path=_route_path(device, left, right),
                two_qubit_depth=swaps * SWAP_TWO_QUBIT_GATES + BOND_TWO_QUBIT_GATES,
            )
        )
    return tuple(routed)


def _route_path(device: Device, left: int, right: int) -> tuple[int, ...]:
    """Walk one shortest route between two qubits, in order.

    Each step goes to whichever neighbour is one edge closer to the destination,
    which is the shortest path by construction. Where several are equally close the
    lowest-numbered wins, so the route is deterministic -- two runs of the same
    campaign have to produce the same circuit, or nothing about the run is
    reproducible.

    Args:
        device: The machine.
        left: Where the walk starts.
        right: Where it ends.

    Returns:
        The qubits in order, beginning with ``left`` and ending with ``right``.
    """
    walk = [left]
    here = left
    while here != right:
        remaining = device.distance(here, right)
        here = min(
            neighbour
            for neighbour in device.neighbours(here)
            if device.distance(neighbour, right) == remaining - 1
        )
        walk.append(here)
    return tuple(walk)


def schedule(routed: tuple[RoutedBond, ...]) -> tuple[int, int]:
    """Pack interactions into rounds and count the entangling layers.

    Two interactions run at the same time exactly when they share no qubit, so this
    is a colouring of the interaction graph by footprint -- the same argument the
    abstract ansatz makes, extended to include the corridors routing walks through.
    A round costs whatever its deepest member costs, so one expensive interaction
    sets the price of everything scheduled beside it, which is precisely why a
    layout that avoids long routes is worth more than one that merely shortens them.

    Args:
        routed: The priced interactions.

    Returns:
        The number of rounds, and the entangling depth those rounds add up to.
    """
    rounds: list[tuple[set[int], int]] = []
    for bond in routed:
        occupied = set(bond.footprint)
        for index, (taken, cost) in enumerate(rounds):
            if taken.isdisjoint(occupied):
                taken.update(occupied)
                rounds[index] = (taken, max(cost, bond.two_qubit_depth))
                break
        else:
            rounds.append((occupied, bond.two_qubit_depth))
    return len(rounds), sum(cost for _, cost in rounds)


@lru_cache(maxsize=4096)
def transpile(ansatz: AnsatzSpec, device: Device) -> Transpiled:
    """Compile one ansatz for one machine and price the result.

    Memoised. Both arguments are frozen and the result is a record, so the same
    specification on the same machine has one answer for the life of the process --
    and a planner climbing a depth ladder, a fidelity search doing the same, and a
    sweep pricing the whole grid all ask for it repeatedly.

    Args:
        ansatz: The circuit specification, in the abstract.
        device: The machine to compile it for.

    Returns:
        The compiled circuit, with the layout, the routing and the duration.

    Raises:
        ValueError: If the chain does not fit on the machine.
    """
    layout = place(ansatz.n_qubits, device, ansatz.boundary)
    routed = route(layout, device, ansatz.bonds)
    rounds, depth_per_layer = schedule(routed)
    swaps_per_layer = sum(bond.swaps for bond in routed)

    two_qubit_per_layer = sum(
        bond.swaps * SWAP_TWO_QUBIT_GATES + BOND_TWO_QUBIT_GATES for bond in routed
    )
    two_qubit_gates = two_qubit_per_layer * ansatz.depth
    single_qubit_gates = ansatz.single_qubit_gates + two_qubit_gates * SINGLE_QUBIT_PER_CX.get(
        device.native_two_qubit, 0
    )

    two_qubit_depth = depth_per_layer * ansatz.depth
    duration = _duration_ns(
        device=device,
        two_qubit_depth=two_qubit_depth,
        single_qubit_gates=single_qubit_gates,
        n_qubits=ansatz.n_qubits,
    )
    return Transpiled(
        device=device,
        ansatz=ansatz,
        layout=layout,
        bonds=routed,
        rounds_per_layer=rounds,
        two_qubit_depth_per_layer=depth_per_layer,
        swaps_per_layer=swaps_per_layer,
        two_qubit_depth=two_qubit_depth,
        two_qubit_gates=two_qubit_gates,
        single_qubit_gates=single_qubit_gates,
        duration_ns=duration,
    )


def _duration_ns(
    *, device: Device, two_qubit_depth: int, single_qubit_gates: int, n_qubits: int
) -> float:
    """Estimate how long the machine is occupied.

    Entangling depth dominates and is counted exactly. Single-qubit gates are spread
    evenly over the register, because they act on different qubits and therefore run
    at the same time -- taking the per-qubit average rather than the total is the
    difference between an estimate that is roughly right and one that is wrong by
    the width of the register. Readout is added once.

    Args:
        device: The machine.
        two_qubit_depth: Entangling layers in the whole circuit.
        single_qubit_gates: Single-qubit gates in the whole circuit.
        n_qubits: How many qubits they are spread across.

    Returns:
        The duration in nanoseconds.
    """
    single_qubit_depth = math.ceil(single_qubit_gates / n_qubits) if n_qubits else 0
    return (
        two_qubit_depth * device.two_qubit_gate_ns
        + single_qubit_depth * device.single_qubit_gate_ns
        + device.readout_ns
    )
