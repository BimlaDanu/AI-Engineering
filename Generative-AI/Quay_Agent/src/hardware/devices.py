r"""The machines a circuit can be sent to, described as data.

A device is four things a feasibility question cares about: which qubits are
wired to which, how long each operation takes, how often each is wrong, and how
long the qubits remember anything. Routing cost, circuit duration, surviving
fidelity and shot inflation all derive from those four, so they are gathered
here where they can be read and disputed.

:data:`IDEAL` is the control -- all-to-all, no error, unbounded coherence. An
energy error reported on it is the algorithm\'s own, and separating that from
hardware error is the first thing an honest study does.

:data:`LINEAR` is the fair case: a line of qubits, the shape the problem already
has, so an open chain maps on with no routing. Its answer to "how well could
this possibly go on real hardware?" is an upper bound, not a prediction.

:data:`HEAVY_HEX` is a machine that exists -- twenty-seven qubits in the
heavy-hexagon lattice IBM\'s processors use, with the two-qubit gate they run.
Its degree-two and degree-three qubits are why the lattice exists, fewer
neighbours meaning less crosstalk, and also why a long chain is awkward on it.
That trade shows up in the routing cost.

The error figures are representative published magnitudes for superconducting
hardware of this generation, not a calibration snapshot. A real backend reports
a different :math:`T_1`, :math:`T_2` and two-qubit error per qubit and per edge,
and they move between calibrations, so per-qubit numbers here would imply a
precision the study does not have. One figure per device keeps the conclusion a
scaling argument.

Worth checking first: a verdict that flips when the two-qubit error moves from
0.008 to 0.012 is a verdict about the calibration, not about the algorithm.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from functools import cache
from typing import Any, Literal

NativeTwoQubitGate = Literal["cz", "cx", "ecr"]
r"""The entangling gate the machine actually applies.

A ``CX`` written in an algorithm is a request, not an instruction. Superconducting
processors run ``CZ`` or ``ECR`` and synthesise ``CX`` from it with single-qubit
frame changes, which cost no time worth counting but do change the gate count a
report should quote. Carried so that a run card names the gate the machine will
run rather than the one the paper was written in.
"""

Topology = Literal["all_to_all", "line", "heavy_hex"]
"""Shape of the coupling graph, kept for a label and a log line, never branched on.

Every routine in this package reads :attr:`Device.coupling` and asks the graph. A
topology name that code decided on would be a second source of truth for the same
fact, and the two would eventually disagree.
"""

HEAVY_HEX_27_COUPLING: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (1, 4),
    (2, 3),
    (3, 5),
    (4, 7),
    (5, 8),
    (6, 7),
    (7, 10),
    (8, 9),
    (8, 11),
    (10, 12),
    (11, 14),
    (12, 13),
    (12, 15),
    (13, 14),
    (14, 16),
    (15, 18),
    (16, 19),
    (17, 18),
    (18, 21),
    (19, 20),
    (19, 22),
    (21, 23),
    (22, 25),
    (23, 24),
    (24, 25),
    (25, 26),
)
r"""The 27-qubit heavy-hexagon lattice, as undirected edges.

Twenty-eight edges over twenty-seven qubits: a tree would need twenty-six, so there
are exactly two independent cycles -- the two hexagons the name refers to. Most
qubits have two neighbours and a few have three, and no qubit has four. That is the
whole design: a square lattice would give every interior qubit four neighbours and
four chances to collide in frequency with one of them.

Written out rather than generated because it is a transcription of a real machine's
wiring, and a generator would be a second claim -- that the lattice follows the rule
the generator encodes -- on top of the one being made.
"""


def _line_coupling(n_qubits: int) -> tuple[tuple[int, int], ...]:
    """Wire a row of qubits to their immediate neighbours.

    Args:
        n_qubits: How many qubits are in the row.

    Returns:
        The ``n_qubits - 1`` edges, each with the lower index first.
    """
    return tuple((q, q + 1) for q in range(n_qubits - 1))


def _all_to_all_coupling(n_qubits: int) -> tuple[tuple[int, int], ...]:
    """Wire every qubit to every other one.

    Args:
        n_qubits: How many qubits there are.

    Returns:
        Every unordered pair, lower index first.
    """
    return tuple((a, b) for a in range(n_qubits) for b in range(a + 1, n_qubits))


@dataclass(frozen=True, slots=True)
class Device:
    r"""One machine: its wiring, its clock and its error rates.

    Frozen, because a device model that could be edited after a circuit was priced
    against it would make the price meaningless. Hashable for the same reason, which
    is what lets the graph searches below be cached per device.

    Attributes:
        name: What a report and a run card call this machine.
        topology: Shape of the coupling graph. A label; nothing branches on it.
        n_qubits: How many physical qubits there are.
        coupling: Undirected edges, each written with the lower index first. A pair
            absent from this tuple is a pair that cannot interact directly, and
            every unit of routing cost in this package comes from that absence.
        native_two_qubit: The entangling gate the machine runs.
        two_qubit_gate_ns: Duration of one entangling gate. It dominates everything;
            single-qubit rotations are roughly an order of magnitude faster.
        single_qubit_gate_ns: Duration of one single-qubit rotation.
        readout_ns: Duration of measuring the register, once, at the end.
        t1_ns: Energy relaxation time. How long an excited qubit stays excited.
        t2_ns: Phase coherence time. The one that binds, because an algorithm's
            information is in the relative phases and :math:`T_2 \le 2 T_1` always.
        two_qubit_error: Probability that one entangling gate is wrong.
        single_qubit_error: Probability that one rotation is wrong.
        readout_error: Probability that one qubit is read back as the other value.
        provenance: Where these figures come from, in one sentence, so a reader
            knows whether they are a measurement or a round number.
    """

    name: str
    topology: Topology
    n_qubits: int
    coupling: tuple[tuple[int, int], ...]
    native_two_qubit: NativeTwoQubitGate
    two_qubit_gate_ns: float
    single_qubit_gate_ns: float
    readout_ns: float
    t1_ns: float
    t2_ns: float
    two_qubit_error: float
    single_qubit_error: float
    readout_error: float
    provenance: str

    def __post_init__(self) -> None:
        """Reject numbers that could not describe a machine.

        The checks that matter are the two physical ones. A probability outside
        :math:`[0, 1)` is a typo; :math:`T_2 > 2 T_1` is a violation of the
        relationship between the two, and a device model that permitted it would
        quietly hand every fidelity estimate an impossible amount of coherence.

        Raises:
            ValueError: If the register is empty, an edge names a qubit that does
                not exist, a duration is not positive, an error rate is outside
                :math:`[0, 1)`, or :math:`T_2` exceeds :math:`2 T_1`.
        """
        if self.n_qubits < 1:
            raise ValueError(f"a device needs at least one qubit, got {self.n_qubits}")
        for left, right in self.coupling:
            if not 0 <= left < self.n_qubits or not 0 <= right < self.n_qubits:
                raise ValueError(f"edge ({left}, {right}) names a qubit outside the register")
            if left == right:
                raise ValueError(f"qubit {left} is coupled to itself")
        durations = {
            "two_qubit_gate_ns": self.two_qubit_gate_ns,
            "single_qubit_gate_ns": self.single_qubit_gate_ns,
            "readout_ns": self.readout_ns,
            "t1_ns": self.t1_ns,
            "t2_ns": self.t2_ns,
        }
        for field_name, value in durations.items():
            if value <= 0.0:
                raise ValueError(f"{field_name} must be positive, got {value}")
        rates = {
            "two_qubit_error": self.two_qubit_error,
            "single_qubit_error": self.single_qubit_error,
            "readout_error": self.readout_error,
        }
        for field_name, value in rates.items():
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{field_name} must be in [0, 1), got {value}")
        if self.t2_ns > 2.0 * self.t1_ns:
            raise ValueError(
                f"{self.name}: T2 of {self.t2_ns} ns exceeds 2*T1 of {2.0 * self.t1_ns} ns, "
                "which no qubit does"
            )

    @property
    def n_edges(self) -> int:
        """How many pairs can interact without help."""
        return len(self.coupling)

    @property
    def mean_degree(self) -> float:
        """Average number of neighbours per qubit.

        The one number that summarises how hard routing will be. Two is a line,
        which is exactly what a chain wants; the heavy-hexagon lattice sits just
        above it; all-to-all is ``n_qubits - 1`` and never routes.
        """
        return 2.0 * self.n_edges / self.n_qubits

    def neighbours(self, qubit: int) -> tuple[int, ...]:
        """List the qubits directly wired to one qubit.

        Args:
            qubit: The physical qubit index.

        Returns:
            Its neighbours, ascending. Empty for an isolated qubit.

        Raises:
            IndexError: If the qubit is not on this device.
        """
        if not 0 <= qubit < self.n_qubits:
            raise IndexError(f"qubit {qubit} is not on {self.name}")
        return _adjacency(self)[qubit]

    def degree(self, qubit: int) -> int:
        """Count one qubit's neighbours.

        Args:
            qubit: The physical qubit index.

        Returns:
            How many qubits it is wired to.

        Raises:
            IndexError: If the qubit is not on this device.
        """
        return len(self.neighbours(qubit))

    def are_coupled(self, left: int, right: int) -> bool:
        """Say whether two qubits can interact without a single SWAP.

        Args:
            left: One physical qubit.
            right: The other.

        Returns:
            ``True`` when an edge joins them.
        """
        return right in _adjacency(self)[left] if 0 <= left < self.n_qubits else False

    def distance(self, left: int, right: int) -> int:
        r"""Count the edges on the shortest route between two qubits.

        This is what routing costs are made of. Two qubits at distance :math:`d`
        cannot interact until :math:`d - 1` SWAPs have walked one of them to the
        other, and every SWAP is three entangling gates that do no physics.

        Args:
            left: One physical qubit.
            right: The other.

        Returns:
            The number of edges. Zero for a qubit and itself, and ``-1`` when no
            route exists, which a disconnected fragment of a lattice can produce.

        Raises:
            IndexError: If either qubit is not on this device.
        """
        if not 0 <= left < self.n_qubits:
            raise IndexError(f"qubit {left} is not on {self.name}")
        if not 0 <= right < self.n_qubits:
            raise IndexError(f"qubit {right} is not on {self.name}")
        return _distances_from(self, left)[right]

    def longest_path(self) -> tuple[int, ...]:
        """Find the longest run of qubits wired end to end.

        A one-dimensional chain of :math:`L` sites maps onto the machine for free
        exactly when the machine has a path of :math:`L` qubits, so this is the
        largest problem the device takes without any routing at all. It is the
        single most useful number about a device for this project, which is why it
        is searched for properly rather than estimated.

        Longest path is NP-hard in general, and the honest position is that this is
        exact for the machines here and best-effort beyond them. Two things make it
        cheap: the search stops the moment it has covered a whole connected
        component, which is immediate on a line and on all-to-all, and the lattices
        that do not have a spanning path are small and have maximum degree three,
        where exhaustive backtracking finishes in milliseconds. Past
        :data:`_SEARCH_BUDGET` steps the best walk found so far is returned, so a
        larger lattice degrades to an underestimate -- which is the safe direction,
        since an underestimate makes the device look harder to use, never easier.

        Returns:
            The qubits in order along the path. Never empty for a device with at
            least one qubit.
        """
        return _longest_path(self)

    def describe(self) -> dict[str, Any]:
        """Render the machine as plain data for a tool result, a log line or a card.

        Every value is a primitive, so the result survives being written to a file
        or handed to a language model without a custom encoder.

        Returns:
            A flat mapping of the wiring summary, the clock and the error rates.
        """
        path = self.longest_path()
        return {
            "name": self.name,
            "topology": self.topology,
            "n_qubits": self.n_qubits,
            "n_edges": self.n_edges,
            "mean_degree": round(self.mean_degree, 3),
            "longest_chain_without_routing": len(path),
            "native_two_qubit": self.native_two_qubit,
            "two_qubit_gate_ns": self.two_qubit_gate_ns,
            "single_qubit_gate_ns": self.single_qubit_gate_ns,
            "readout_ns": self.readout_ns,
            "t1_us": round(self.t1_ns / 1000.0, 1),
            "t2_us": round(self.t2_ns / 1000.0, 1),
            "two_qubit_error": self.two_qubit_error,
            "single_qubit_error": self.single_qubit_error,
            "readout_error": self.readout_error,
            "two_qubit_gates_within_coherence": self.gates_within_coherence(),
            "provenance": self.provenance,
        }

    def gates_within_coherence(self) -> int:
        r"""How many sequential entangling layers fit inside :math:`T_2`.

        The crudest possible measure of a machine, and the most quoted one, because
        it collapses speed and coherence into a single figure of merit: a faster
        gate and a longer-lived qubit are the same improvement to an algorithm.
        It is an optimistic ceiling -- it ignores gate error entirely, and
        :mod:`src.hardware.fidelity` puts the measured limit well below it.

        Returns:
            :math:`\lfloor T_2 / t_\text{2q} \rfloor`.
        """
        return math.floor(self.t2_ns / self.two_qubit_gate_ns)


@cache
def _adjacency(device: Device) -> tuple[tuple[int, ...], ...]:
    """Build the neighbour lists once per device.

    Args:
        device: The machine.

    Returns:
        One ascending tuple of neighbours per qubit, indexed by qubit.
    """
    lists: list[list[int]] = [[] for _ in range(device.n_qubits)]
    for left, right in device.coupling:
        lists[left].append(right)
        lists[right].append(left)
    return tuple(tuple(sorted(entry)) for entry in lists)


@cache
def _distances_from(device: Device, source: int) -> tuple[int, ...]:
    """Breadth-first distances from one qubit to every other.

    Cached because routing asks for the same source repeatedly while pricing a
    depth ladder, and the graph never changes.

    Args:
        device: The machine.
        source: The qubit to measure from.

    Returns:
        One distance per qubit, indexed by qubit, with ``-1`` where no route exists.
    """
    found = [-1] * device.n_qubits
    found[source] = 0
    queue = deque([source])
    while queue:
        here = queue.popleft()
        for neighbour in _adjacency(device)[here]:
            if found[neighbour] == -1:
                found[neighbour] = found[here] + 1
                queue.append(neighbour)
    return tuple(found)


_SEARCH_BUDGET = 400_000
"""How many extensions the path search may try before settling for what it has.

Generous enough that every machine in this file is searched exhaustively, and low
enough that a denser lattice added later cannot hang a campaign. The failure mode
it guards against is a device model that makes the application stop responding, so
it is a ceiling on work rather than a tuning parameter -- nothing improves by
raising it until a device exists that hits it.
"""


def _component_size(device: Device, start: int) -> int:
    """Count the qubits reachable from one qubit.

    The search below stops as soon as it has covered a whole component, so this is
    what makes an all-to-all device cost one descent instead of a factorial number.

    Args:
        device: The machine.
        start: The qubit to measure from.

    Returns:
        How many qubits are connected to it, counting itself.
    """
    return sum(1 for reach in _distances_from(device, start) if reach >= 0)


@cache
def _longest_path(device: Device) -> tuple[int, ...]:
    """Search every starting qubit for the longest simple path.

    Depth-first with backtracking, which is what makes it exact rather than a
    heuristic: a walk that strands itself in a corner is undone and retried, where
    a greedy walk would have reported the corner. Ordering each step by the number
    of *unvisited* neighbours the candidate has is what keeps that backtracking
    rare -- it defers dead ends, and on a lattice whose leaves are degree-one qubits
    that is most of the wasted work.

    Args:
        device: The machine.

    Returns:
        The qubits in order along the longest path found.
    """
    adjacency = _adjacency(device)
    best: list[int] = []
    steps = 0

    def descend(walk: list[int], visited: set[int], target: int) -> bool:
        """Extend one walk, recording it if it is the best so far.

        Args:
            walk: The path built so far, in order.
            visited: The same qubits, as a set.
            target: Size of the component; reaching it means nothing can beat this.

        Returns:
            ``True`` when the search should stop -- a spanning path was found, or
            the budget ran out.
        """
        nonlocal best, steps
        if len(walk) > len(best):
            best = list(walk)
            if len(best) == target:
                return True
        open_next = sorted(
            (q for q in adjacency[walk[-1]] if q not in visited),
            key=lambda q: (sum(1 for n in adjacency[q] if n not in visited), q),
        )
        for step in open_next:
            steps += 1
            if steps > _SEARCH_BUDGET:
                return True
            walk.append(step)
            visited.add(step)
            if descend(walk, visited, target):
                return True
            walk.pop()
            visited.discard(step)
        return False

    for start in range(device.n_qubits):
        if len(best) >= _component_size(device, start):
            continue
        if descend([start], {start}, _component_size(device, start)):
            break
    return tuple(best)


IDEAL = Device(
    name="ideal",
    topology="all_to_all",
    n_qubits=32,
    coupling=_all_to_all_coupling(32),
    native_two_qubit="cz",
    two_qubit_gate_ns=1.0,
    single_qubit_gate_ns=1.0,
    readout_ns=1.0,
    t1_ns=1.0e12,
    t2_ns=1.0e12,
    two_qubit_error=0.0,
    single_qubit_error=0.0,
    readout_error=0.0,
    provenance="not a machine: the control that isolates algorithmic error from device error",
)
r"""The machine that cannot be blamed.

Everything connected to everything, nothing ever wrong, coherence sixteen minutes
long. Its purpose is subtraction: run a configuration here and on a real device,
and the difference is what the hardware cost. Thirty-two qubits because the state
vector, not the device, is the limit on anything that has to be simulated.
"""

LINEAR = Device(
    name="linear",
    topology="line",
    n_qubits=32,
    coupling=_line_coupling(32),
    native_two_qubit="cz",
    two_qubit_gate_ns=300.0,
    single_qubit_gate_ns=35.0,
    readout_ns=1_000.0,
    t1_ns=120_000.0,
    t2_ns=100_000.0,
    two_qubit_error=8.0e-3,
    single_qubit_error=3.0e-4,
    readout_error=1.5e-2,
    provenance="round superconducting figures: 300 ns entangling gate, 100 us T2, 0.8% two-qubit",
)
r"""A row of qubits, which is the shape the problem already has.

The best case that is still a real machine. An open chain of :math:`L` sites places
onto qubits :math:`0 \dots L-1` with every bond already an edge, so routing costs
nothing and the whole difficulty is coherence against depth. A verdict of "no" here
is a verdict that better connectivity would not have saved.
"""

HEAVY_HEX = Device(
    name="heavy-hex-27",
    topology="heavy_hex",
    n_qubits=27,
    coupling=HEAVY_HEX_27_COUPLING,
    native_two_qubit="ecr",
    two_qubit_gate_ns=440.0,
    single_qubit_gate_ns=35.0,
    readout_ns=1_400.0,
    t1_ns=110_000.0,
    t2_ns=90_000.0,
    two_qubit_error=9.0e-3,
    single_qubit_error=2.5e-4,
    readout_error=1.8e-2,
    provenance=(
        "published heavy-hexagon lattice with representative figures for that generation "
        "of superconducting hardware -- a magnitude, not a calibration snapshot"
    ),
)
"""A lattice that exists, with the awkwardness that comes with it.

Chosen because it is the topology most people mean by "a real quantum computer"
today, and because its sparsity is not a flaw to be apologised for -- fewer
neighbours is what buys the low crosstalk that makes the error rates above
achievable at all. The cost lands on problems shaped like chains, which is this
one, and the routing stage puts a number on it.
"""

DEVICES: tuple[Device, ...] = (IDEAL, LINEAR, HEAVY_HEX)
"""Every machine, in the order a study should use them: control, best case, real."""


def device_names() -> tuple[str, ...]:
    """List what can be asked for by name.

    Returns:
        The device names, in the order of :data:`DEVICES`.
    """
    return tuple(device.name for device in DEVICES)


def device_for(name: str) -> Device:
    """Look up a machine by name.

    Args:
        name: One of :func:`device_names`. Matching ignores case and surrounding
            space, because this is reached from a command line and from a tool
            argument a language model wrote.

    Returns:
        The device.

    Raises:
        KeyError: If no device has that name. The message lists the ones that do,
            since the caller is often a model that guessed.
    """
    wanted = name.strip().lower()
    for device in DEVICES:
        if device.name == wanted:
            return device
    raise KeyError(f"no device named {name!r}; available: {', '.join(device_names())}")
