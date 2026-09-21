r"""The functions the language model is allowed to call, and nothing else.

This is the agent's whole surface onto the project. Everything it can find out
about a physics problem, and every calculation it can cause to happen, goes
through one of the tools registered here. That is the point of concentrating
them in one file: the question "what can this agent actually do?" has a
readable answer, and the question "can it see the answer it is being graded
against?" has a checkable one.

Every tool here is written to be read by someone who does not do physics. Its
description -- the text the model sees, and the text that answers "what is the
agent for?" -- states what the tool does in plain language and
names its units. The domain terms are glossed on first use in each description
rather than assumed, because the model choosing between these tools has only the
descriptions to go on, and so does a reader.

The wall. Two exact solutions to this problem live in
:mod:`src.physics.reference`. They are the grader's, and no tool here can reach
them, directly or through anything it imports; the architecture test
walks the import graph and fails if that ever stops being true. The agent can
*learn from* :func:`list_methods` that an exact answer exists and what it would
cost -- that is a fact about the problem, and an agent that could not state it
would be unable to write an honest report -- but it holds no way to evaluate
one. The only solver it can run is the classical baseline, which returns a
number it is entitled to compute for itself.

Everything is JSON. Tools return plain dictionaries of primitives, never
project dataclasses. A tool result is about to be serialised into a model's
context window, and a type that renders as ``<PauliSum object at 0x...>`` is a
silent failure that costs a whole turn.

Bad arguments are prevented, then reported, in that order. The
enumerated arguments are typed as :class:`~typing.Literal`, so the schema the
model is shown carries the permitted values and a wrong one is rejected by
pydantic before any code here runs -- constraining the choice is better than
correcting it. Everything the schema cannot express -- a chain length that is
physically meaningless, a size beyond what these tools will spend time on, a
method that declines the problem -- is checked here and returned as
``{"error": ...}``. Returned, not raised: a tool that raises kills the agent's
turn, whereas a value gives the model the chance to fix its own argument and try
again, which turns a wrong guess into one wasted call instead of a dead run.

The value-level checks are not redundant with the schema. Anything calling these
functions directly -- a test, a notebook, the Streamlit page -- bypasses
pydantic entirely, and a guard that only exists in a schema is a guard that is
absent exactly when someone is debugging.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, Protocol, cast, get_args

from langchain_core.tools import BaseTool, tool

from src.agent.usage import estimated_price_for, price_basis, price_for
from src.hardware.devices import device_for, device_names
from src.hardware.fidelity import depth_ceiling, estimate
from src.hardware.transpile import fits, transpile
from src.physics.lattice import Geometry
from src.physics.method_catalogue import (
    COST_ORDER,
    PFEUTY_EXACT,
    get_method,
    method_names,
    survey,
)
from src.physics.model import BoundaryCondition, TFIMSpec
from src.physics.quantum.ansatz import AnsatzSpec
from src.physics.quantum.hamiltonians import ising_chain

DEFAULT_TARGET_ERROR = 1e-2
"""Default accuracy, in energy units, for a shot-budget estimate.

One percent of a coupling. Chosen because it is the loosest tolerance at which
the answer is still useful for a feasibility statement, and because the shot
count scales as its inverse square -- so a reader who wants ten times the
accuracy can multiply by a hundred without being told.
"""

MAX_TOOL_SITES = 64
"""Largest chain a tool will accept.

A refusal rather than a limit on what the physics permits. The classical
baseline would happily run at a thousand sites and take an hour doing it, and an
agent that can commit the process to an hour of work on a hallucinated argument
is an agent that will eventually do so.
"""

DEFAULT_ARXIV_RESULTS = 4
"""Papers one arXiv search returns unless the model asks for fewer.

Four abstracts is a paragraph each and still leaves room for an answer. It is
also enough to show a reader that the search found a *field* rather than one
lucky hit, which is what a request for "recent work" is actually asking.
"""

MAX_ARXIV_RESULTS = 8
"""Ceiling on one arXiv search, whatever the model asks for.

The cost of a high limit is not the network call; it is that eight abstracts
crowd the answer out of the context window, and the model then summarises the
abstracts instead of answering the question.
"""

MAX_ARXIV_PAPERS = 8
"""Ceiling on the papers one call returns, however many subjects it searched.

The per-subject :data:`MAX_ARXIV_RESULTS` bounds one search; this bounds the call,
because four subjects at eight apiece is thirty-two abstracts and no reader, model
or person, is served by that. Applied *after*
:func:`_one_from_each_in_turn`, so trimming to it drops the least relevant paper of
each subject rather than every paper of the last subject.
"""

MAX_ARXIV_SEARCHES = 4
"""How many subjects one call may search for at once.

A ceiling on threads and on records both: four subjects at
:data:`MAX_ARXIV_RESULTS` apiece is already more abstract than the answering call
can read, and the point of batching is to remove round trips rather than to raise
how much comes back. Anything past this is searched, just not concurrently.
"""


def _spec_or_error(
    n_sites: int,
    coupling: float,
    transverse_field: float,
    boundary: str,
    geometry: str = "chain",
    rows: int = 1,
) -> TFIMSpec | dict[str, str]:
    """Build a validated problem specification, or the error to hand back.

    Argument validation is centralised here because every tool takes the same
    numbers, and because the model needs the *same* sentence back whichever
    tool it got them wrong in -- an error that varies by call site reads as
    several different problems.

    The shape is validated here for the same reason. Carried nowhere, it left
    every tool building a spec from four numbers and so describing a line whatever
    the question was about, and ``list_methods`` reporting the closed-form solution
    as available for a ``4 x 4`` square. That solution exists for a line and
    nothing else.

    Args:
        n_sites: Total number of spins.
        coupling: The interaction strength ``J``.
        transverse_field: The field strength ``h``.
        boundary: ``"periodic"`` or ``"open"``.
        geometry: ``"chain"``, ``"square"`` or ``"triangular"``.
        rows: Rows of sites for a two-dimensional shape; one for a chain.

    Returns:
        A :class:`~src.physics.model.TFIMSpec`, or a one-key ``{"error": ...}``
        dictionary describing what was wrong with the arguments.
    """
    if n_sites > MAX_TOOL_SITES:
        return {
            "error": (
                f"n_sites={n_sites} is above the tool limit of {MAX_TOOL_SITES}. "
                "Ask for a smaller problem, or explain in your answer why the "
                "question needs a larger one than these tools will run."
            )
        }
    if boundary not in ("periodic", "open"):
        return {"error": f"boundary must be 'periodic' or 'open', got {boundary!r}"}
    if geometry not in get_args(Geometry):
        return {"error": (f"geometry must be one of {list(get_args(Geometry))}, got {geometry!r}")}
    try:
        return TFIMSpec(
            n_sites=n_sites,
            coupling=coupling,
            field=transverse_field,
            boundary=cast(BoundaryCondition, boundary),
            geometry=cast(Geometry, geometry),
            rows=rows,
        )
    except (TypeError, ValueError) as error:
        return {"error": str(error)}


@tool(parse_docstring=True)
def describe_problem(
    n_sites: int,
    coupling: float = 1.0,
    transverse_field: float = 1.0,
    longitudinal_field: float = 0.0,
    boundary: Literal["periodic", "open"] = "open",
    geometry: Literal["chain", "square", "triangular"] = "chain",
    rows: int = 1,
) -> dict[str, Any]:
    """Report the structure of a quantum spin-lattice problem without solving it.

    The problem is `n_sites` two-state quantum systems ("spins", the same thing
    as qubits) arranged on a lattice -- a line by default, or a two-dimensional
    square or triangular lattice. Three numbers set it up: `coupling` makes
    neighbours want to point the same way, `transverse_field` pushes every spin
    sideways so that it cannot settle, and `longitudinal_field` tilts it back along
    the coupling direction. The competition between the first two is the whole
    problem; the third controls how hard it is.

    Use this first, before choosing a method. It is arithmetic on the problem
    statement and costs nothing -- no solver runs -- and it returns the sizes
    that decide what is affordable: how many separate terms make up the energy,
    how many of them can be measured at the same time on a quantum computer, and
    how large the state would be if written out in full.

    Args:
        n_sites: Number of spins in the chain. Each one doubles the size of the
            full quantum state, which is why small numbers are not timidity.
        coupling: Strength of the neighbour interaction, usually written J.
            Positive; the energy scale everything else is measured against.
        transverse_field: Strength of the sideways field, usually written h. The
            ratio h/J is the dial that matters: the chain changes character
            sharply at h/J = 1, and problems are hardest there.
        longitudinal_field: Strength of the field along the coupling direction,
            usually written g. Zero by default. At g = 0 this problem has a known
            shortcut solution and no quantum computer could beat it; g not equal
            to zero removes the shortcut and is what makes the question real.
        boundary: "open" for edges that are edges, "periodic" for edges that wrap.
        geometry: The shape the spins sit on. "chain" is a line, which is the
            default and the only shape with a shortcut solution. "square" and
            "triangular" are two-dimensional lattices, where each spin has four or
            six neighbours instead of two -- harder to solve, and the regime where a
            quantum computer has the better case. A triangular lattice with a
            negative coupling is *frustrated*: no arrangement satisfies every bond
            at once.
        rows: How many rows of sites, for a two-dimensional shape. The columns
            follow from n_sites, so a 4x4 square is n_sites=16 with rows=4. Leave at
            1 for a chain.

    Returns:
        A dictionary describing the problem's size and structure, or a single
        "error" key if the arguments were not usable.
    """
    spec = _spec_or_error(n_sites, coupling, transverse_field, boundary, geometry, rows)
    if isinstance(spec, dict):
        return spec
    operator = ising_chain(
        n_sites=spec.n_sites,
        coupling=spec.coupling,
        transverse_field=spec.field,
        longitudinal_field=longitudinal_field,
        boundary=spec.boundary,
        lattice=spec.shape_to_solve,
    )
    groups = operator.measurement_groups()
    shape = spec.lattice
    return {
        "label": spec.label(),
        "n_spins": spec.n_sites,
        "coupling": spec.coupling,
        "transverse_field": spec.field,
        "longitudinal_field": longitudinal_field,
        "boundary": spec.boundary,
        "geometry": spec.geometry,
        "shape": shape.in_words(),
        "neighbours_per_spin": shape.neighbours_per_site(),
        "is_bipartite": shape.is_bipartite,
        "frustrated": shape.frustrated_by(spec.coupling),
        "field_ratio_h_over_J": spec.ratio,
        "at_the_critical_point": abs(spec.ratio - 1.0) < 1e-9,
        "has_shortcut_solution": longitudinal_field == 0.0 and spec.is_one_dimensional,
        "n_interacting_pairs": spec.n_bonds,
        "n_energy_terms": len(operator.terms),
        "n_measurement_settings": len(groups),
        "full_state_dimension": 2**spec.n_sites,
        "note": (
            "n_measurement_settings is how many separate experiments a quantum "
            "computer needs per energy estimate: terms that can be measured "
            "together are grouped. full_state_dimension is how many numbers a "
            "classical computer would have to store to hold the state exactly. "
            "has_shortcut_solution is true only on a line with no longitudinal "
            "field -- that is the one case with a closed-form answer, and a "
            "two-dimensional lattice never has one however small it is. "
            "frustrated means no arrangement of spins can satisfy every bond, "
            "which needs a negative coupling on a lattice that is not bipartite."
        ),
    }


@tool(parse_docstring=True)
def list_methods(
    n_sites: int,
    coupling: float = 1.0,
    transverse_field: float = 1.0,
    longitudinal_field: float = 0.0,
    boundary: Literal["periodic", "open"] = "open",
    geometry: Literal["chain", "square", "triangular"] = "chain",
    rows: int = 1,
) -> dict[str, Any]:
    """List every method that could solve this problem, and why the rest cannot.

    Returns a menu, not a decision. Each entry says how the method's cost grows
    with the size of the problem, what kind of number it returns, and --
    importantly -- whether you are allowed to run it.

    Some methods are marked `runnable: false`. Those are exact reference
    solutions this agent cannot call. They are listed
    because knowing an exact answer exists, and what it would cost, is part of an
    honest report: it is the difference between "no cheaper method was available"
    and "a cheaper method existed and I did not use it". Do not claim their
    results as your own, and do not describe a problem as unsolvable because the
    only method you may run is an approximate one.

    Args:
        n_sites: Number of spins in the chain.
        coupling: Strength of the neighbour interaction, usually written J.
        transverse_field: Strength of the sideways field, usually written h.
        longitudinal_field: Strength of the field along the coupling direction,
            usually written g. **This changes the menu**, which is why it is here:
            at g = 0 the chain maps to free fermions and an exact answer costs
            O(L), and at any other g that shortcut is gone and nothing cheaper
            than an exponential method remains. This argument was missing, so the
            survey always ran at g = 0 and reported the closed form as available
            for every chain -- telling the reader an O(L) exact solution exists for
            exactly the class of problem this project defines as having none, which
            inverts the feasibility verdict.
        boundary: "open" for edges that are edges, "periodic" for edges that wrap.
        geometry: The shape the spins sit on. **This changes the menu too, and
            more sharply than anything else here.** The shortcut solution comes
            from a transformation that straightens out a line and does nothing for
            any other shape, so on "square" or "triangular" it is withheld however
            small the lattice is. The sampled classical baseline is withheld as
            well, because the lattice it walks would have to gain a dimension.
            Leaving this at its default when the question was about a lattice
            reports the wrong problem's menu.
        rows: How many rows of sites, for a two-dimensional shape. The columns
            follow from n_sites, so a 4x4 square is n_sites=16 with rows=4. Leave
            at 1 for a chain.

    Returns:
        A dictionary with an "available" list, an "unavailable" list carrying the
        reason each method declined, and "runnable" naming the subset you may
        actually call. A single "error" key if the arguments were not usable.
    """
    spec = _spec_or_error(n_sites, coupling, transverse_field, boundary, geometry, rows)
    if isinstance(spec, dict):
        return spec
    result = survey(spec)
    # The shape is now on the spec, so `survey` withholds the closed form on a
    # lattice by itself. The longitudinal field is not on the spec -- the two
    # reference solvers do not implement it, and a spec able to express a problem
    # nothing can check invites an uncheckable run -- so `survey` cannot see g and
    # this is the one place that has to. Both omissions had the same consequence
    # and it is the worst one available here: an O(L) exact answer reported as
    # available for exactly the class of problem the project defines as having
    # none, which inverts the verdict rather than blurring it.
    broken_by_g = longitudinal_field != 0.0
    applicable = [
        facts for facts in result.applicable if not (broken_by_g and facts.name == PFEUTY_EXACT)
    ]
    withheld = [
        {
            "name": item.method.name,
            "reason": item.reason,
        }
        for item in result.rejected
    ]
    if broken_by_g and any(facts.name == PFEUTY_EXACT for facts in result.applicable):
        withheld.append(
            {
                "name": PFEUTY_EXACT,
                "reason": (
                    f"the longitudinal field g = {longitudinal_field:g} is not zero, so "
                    "the chain no longer maps to free fermions and the O(L) closed form "
                    "does not apply -- nothing cheaper than an exponential method is "
                    "left, which is the whole reason g is the interesting knob"
                ),
            }
        )
    return {
        "problem": spec.label(),
        "shape": spec.lattice.in_words(),
        "longitudinal_field": longitudinal_field,
        "available": [
            {
                "name": facts.name,
                "summary": facts.summary,
                "when_to_use": facts.when_to_use,
                "cost_growth": facts.cost,
                "cost_rank": COST_ORDER[facts.cost],
                "answer_kind": facts.accuracy,
                "runnable": facts.runnable_by_agent,
                "estimated_memory_bytes": facts.memory_bytes(spec),
            }
            for facts in applicable
        ],
        "unavailable": withheld,
        "runnable": [
            facts.name
            for facts in result.runnable()
            if not (broken_by_g and facts.name == PFEUTY_EXACT)
        ],
        "note": (
            "answer_kind 'exact' is correct to machine precision. "
            "'variational_bound' is guaranteed to be at or above the true "
            "lowest energy, never below it -- so it can be trusted as an upper "
            "limit even with no reference to check it against."
        ),
    }


@tool(parse_docstring=True)
def estimate_measurement_cost(
    n_sites: int,
    coupling: float = 1.0,
    transverse_field: float = 1.0,
    longitudinal_field: float = 0.0,
    boundary: Literal["periodic", "open"] = "open",
    target_error: float = DEFAULT_TARGET_ERROR,
    geometry: Literal["chain", "square", "triangular"] = "chain",
    rows: int = 1,
) -> dict[str, Any]:
    """Estimate how many quantum measurements one energy reading would cost.

    A quantum computer does not read out an energy; it produces one random
    sample per run, and the energy is the average over many. The number of runs
    needed grows as the inverse *square* of the accuracy wanted, so asking for
    ten times the precision costs a hundred times as much. This tool does that
    arithmetic, and it is the honest way to answer "would this fit in a budget?"
    before spending anything.

    The estimate is a worst case, not a floor. It charges every term the largest
    variance a measurement of it could have, which on a prepared ground state
    over-prices by more than tenfold, and it charges no hardware noise, which
    under-prices by the inverse square of the signal that survives -- the factor
    `check_device_feasibility` reports. On the devices here the first effect is the
    larger, so this comes out high rather than low.

    Args:
        n_sites: Number of spins in the chain.
        coupling: Strength of the neighbour interaction, usually written J.
        transverse_field: Strength of the sideways field, usually written h.
        longitudinal_field: Strength of the field along the coupling direction,
            usually written g. Adds terms but not measurement settings.
        boundary: "open" for a line with two ends, "periodic" for a ring.
        target_error: Wanted accuracy of the energy, in the same units as
            coupling. Must be positive. Smaller is quadratically more expensive.
        geometry: The shape the spins sit on. It changes the answer, because the
            shot count grows with the sum of the term weights and a lattice has
            more terms: a 4x4 square has 24 interacting pairs where a line of
            sixteen has 15, so the same accuracy costs more than twice as many
            measurements.
        rows: How many rows of sites, for a two-dimensional shape. The columns
            follow from n_sites, so a 4x4 square is n_sites=16 with rows=4. Leave
            at 1 for a chain.

    Returns:
        A dictionary with the shot count, how it splits across measurement
        settings, and the assumption behind it. The split is proportional to each
        setting's share of the term weight, because that is the allocation the
        total was derived from. A single "error" key if the arguments were not
        usable.
    """
    if target_error <= 0.0:
        return {"error": f"target_error must be positive, got {target_error}"}
    spec = _spec_or_error(n_sites, coupling, transverse_field, boundary, geometry, rows)
    if isinstance(spec, dict):
        return spec
    operator = ising_chain(
        n_sites=spec.n_sites,
        coupling=spec.coupling,
        transverse_field=spec.field,
        longitudinal_field=longitudinal_field,
        boundary=spec.boundary,
        lattice=spec.shape_to_solve,
    )
    groups = operator.measurement_groups()
    weight = operator.coefficient_l1()
    total_shots = math.ceil((weight / target_error) ** 2)
    weights = [group.operator.coefficient_l1() for group in groups]
    return {
        "problem": spec.label(),
        "shape": spec.lattice.in_words(),
        "target_error": target_error,
        "total_shots": total_shots,
        "n_measurement_settings": len(groups),
        "shots_per_setting": [
            math.ceil(total_shots * share / weight) if weight else 0 for share in weights
        ],
        "weight_per_setting": weights,
        "sum_of_term_weights": weight,
        "one_energy_reading_only": True,
        "note": (
            "This is the cost of ONE energy reading. A variational search calls "
            "for one reading per optimisation step and often several hundred "
            "steps, so multiply before quoting a total. Shots scale as "
            "(sum_of_term_weights / target_error)**2, which charges every term "
            "the largest variance it could have and charges no hardware noise, so "
            "a real device needs this multiplied by the inverse square of its "
            "surviving signal. The per-setting split is proportional to each "
            "setting's share of the weight, which is the allocation the total "
            "assumes; splitting the shots evenly instead would miss the target "
            "error."
        ),
    }


@tool(parse_docstring=True)
def run_classical_baseline(
    n_sites: int,
    coupling: float = 1.0,
    transverse_field: float = 1.0,
    longitudinal_field: float = 0.0,
    boundary: Literal["periodic", "open"] = "open",
    depth: int = 1,
    seed: int = 0,
    geometry: Literal["chain", "square", "triangular"] = "chain",
    rows: int = 1,
) -> dict[str, Any]:
    """Solve the problem approximately on a classical computer, and report the cost.

    This is the number any quantum result has to beat, and running it is not
    optional: a quantum method that is not compared against a good classical one
    has not been shown to be worth anything. It is also the only solver you may
    call.

    The method builds a trial state from `depth` alternating layers and tunes
    them by random sampling until the energy stops falling. It is the same layer
    structure a quantum circuit would use, run classically -- which is what makes
    the comparison fair rather than a straw man.

    The answer it returns is an *upper bound*: guaranteed at or above the true
    lowest energy, never below. Report it with its error bar, and never describe
    it as the exact answer.

    This tool does real work and takes seconds, not milliseconds. Call it once
    per problem you actually intend to report on.

    Args:
        n_sites: Number of spins in the chain.
        coupling: Strength of the neighbour interaction, usually written J.
        transverse_field: Strength of the sideways field, usually written h.
        longitudinal_field: Strength of the field along the coupling direction,
            usually written g.
        boundary: "open" for a line with two ends, "periodic" for a ring. A ring
            must have an even number of spins.
        depth: How many layers the trial state gets. More layers can reach a
            lower energy and cost proportionally more to tune. Start at 1 and
            raise it only if you intend to report how the answer improved.
        seed: Fixes the random numbers, so the same call returns the same answer.
            Change it to check that a result is not an artefact of one sample.
        geometry: The shape the spins sit on. **This method runs on a line only**,
            and it says so rather than answering about one. The lattice it walks
            is sites by imaginary-time slices, which is two-dimensional for a
            line; a two-dimensional problem would need a three-dimensional one.
            Pass the real shape and read the refusal -- it is a fact about the
            problem worth reporting, not an obstacle to route around, and a line
            of the same size would be a different problem's answer with a
            convincingly small error bar on it.
        rows: How many rows of sites, for a two-dimensional shape. The columns
            follow from n_sites. Leave at 1 for a chain.

    Returns:
        A dictionary with the energy per spin, its statistical error, and what
        was spent. A single "error" key if the arguments were not usable or the
        method declined the problem.
    """
    if depth < 1:
        return {"error": f"depth must be at least 1, got {depth}"}
    spec = _spec_or_error(n_sites, coupling, transverse_field, boundary, geometry, rows)
    if isinstance(spec, dict):
        return spec
    facts = get_method("variational_imaginary_time")
    refusal = facts.unsupported_reason(spec)
    if refusal is not None:
        return {"error": f"the classical baseline declined this problem: {refusal}"}

    from src.physics.classical.variational_imaginary_time import ground_state_energy

    result = ground_state_energy(
        spec,
        depth=depth,
        longitudinal_field=longitudinal_field,
        seed=seed,
        geometry=spec.shape_to_solve,
    )
    return {
        "problem": spec.label(),
        "shape": spec.lattice.in_words(),
        "method": facts.name,
        "energy_per_spin": result.energy,
        "energy_error": result.energy_error,
        "total_energy": result.energy * spec.n_sites,
        "depth": result.depth,
        "n_measurements": result.n_measurements,
        "is_upper_bound": True,
        "note": (
            "energy_per_spin is an upper bound on the true lowest energy, with "
            "energy_error its one-sigma statistical uncertainty. A difference "
            "between two runs smaller than the error bars is not a difference. "
            "Quote both numbers together or neither."
        ),
    }


@tool(parse_docstring=True)
def assess_device_fit(
    n_sites: int,
    device: Literal["ideal", "linear", "heavy-hex-27"] = "linear",
    circuit_depth: int = 1,
    boundary: Literal["periodic", "open"] = "open",
    longitudinal_field: float = 0.0,
    geometry: Literal["chain", "square", "triangular"] = "chain",
    rows: int = 1,
) -> dict[str, Any]:
    """Check whether a real quantum machine could actually run this circuit.

    Every other estimate in this project assumes the machine can do whatever is
    asked of it. Real hardware cannot. Its qubits are wired to some of their
    neighbours and not the rest, so two spins the problem says interact may be
    physically far apart and have to be shuffled together first -- which costs
    extra operations that do no useful work. Its qubits also forget what they
    were doing after a fixed time, and every operation has a chance of going
    wrong. This tool applies all three limits and reports what is left.

    Call it before claiming a configuration is runnable, and call it again for a
    different machine before claiming one machine is better than another. It
    spends nothing: the answer is arithmetic over the machine's wiring diagram,
    not a simulation.

    Three machines can be asked about. "ideal" is a fictional perfect one, used
    as a control -- any error left on it belongs to the algorithm rather than the
    hardware. "linear" is a realistic row of qubits, which is the shape this
    problem already has, so it is the best case that is still honest.
    "heavy-hex-27" is the wiring pattern real superconducting processors use, and
    it is sparser, so it is the case that shows what connectivity costs.

    Args:
        n_sites: Number of spins in the chain, one qubit each.
        device: Which machine to check against.
        circuit_depth: How many repeated layers the circuit has. More layers can
            represent the answer better and cost proportionally more.
        boundary: "open" for a line with two ends, "periodic" for a ring. A ring
            is markedly more expensive on real hardware, because its two ends
            have to interact and no machine here wires them together.
        longitudinal_field: Strength of the field along the coupling direction,
            usually written g. Adds single-qubit operations only.
        geometry: The shape the spins sit on. **This changes the answer more than
            any other argument here.** A line puts two two-qubit gates on each
            spin per layer, a square lattice four and a triangular one six -- and
            every one of them is a full gate duration against a fixed coherence
            time, before the machine's wiring adds the shuffling. So a lattice
            reaches its depth ceiling several rungs earlier than a line of the
            same size, and leaving this at its default reports a line's ceiling
            under the lattice's name.
        rows: How many rows of sites, for a two-dimensional shape. The columns
            follow from n_sites, so a 4x4 square is n_sites=16 with rows=4. Leave
            at 1 for a chain.

    Returns:
        A dictionary with what the circuit costs on that machine, how much of the
        signal survives the noise, how many extra measurements that costs, and
        the deepest circuit the machine would carry for this problem. A single
        "error" key if the arguments were not usable.
    """
    if n_sites < 2:
        return {"error": f"the problem needs at least 2 sites, got {n_sites}"}
    if circuit_depth < 0:
        return {"error": f"circuit_depth cannot be negative, got {circuit_depth}"}
    if n_sites > MAX_TOOL_SITES:
        return {"error": f"n_sites={n_sites} is above the tool limit of {MAX_TOOL_SITES}"}
    if boundary not in ("periodic", "open"):
        return {"error": f"boundary must be 'periodic' or 'open', got {boundary!r}"}
    shaped = _spec_or_error(n_sites, 1.0, 1.0, boundary, geometry, rows)
    if isinstance(shaped, dict):
        return shaped
    shape = shaped.shape_to_solve
    try:
        machine = device_for(device)
    except KeyError:
        return {"error": f"unknown device {device!r}; available: {', '.join(device_names())}"}
    refusal = fits(n_sites, machine)
    if refusal is not None:
        return {"error": refusal}

    compiled = transpile(
        AnsatzSpec(
            n_qubits=n_sites,
            depth=circuit_depth,
            boundary=boundary,
            longitudinal=longitudinal_field != 0.0,
            lattice=shape,
        ),
        machine,
    )
    surviving = estimate(compiled)
    ceiling = depth_ceiling(
        n_sites,
        machine,
        boundary=boundary,
        longitudinal=longitudinal_field != 0.0,
        lattice=shape,
    )
    inflation = surviving.shot_inflation()
    return {
        "device": machine.describe(),
        "problem_shape": shaped.lattice.in_words(),
        "neighbours_per_spin": shaped.lattice.neighbours_per_site(),
        "circuit_on_this_device": compiled.describe(),
        "surviving_signal": surviving.describe(),
        "deepest_circuit_worth_running": ceiling.describe(),
        "runnable_as_asked": circuit_depth <= ceiling.limit,
        "extra_shots_the_noise_costs": None if inflation == float("inf") else round(inflation, 2),
        "note": (
            "routing_overhead is how much deeper the machine's wiring makes the "
            "circuit; 1.0 means the wiring cost nothing. surviving_signal.total is "
            "the fraction of the result that is still the intended state, and "
            "reaching a fixed accuracy costs its inverse square in extra "
            "measurements. binding_constraint says which limit stops this problem "
            "first: 'fidelity' asks for more accurate gates, 'coherence' asks for "
            "faster ones, and they are not the same request."
        ),
    }


def _one_from_each_in_turn(
    per_subject: list[list[dict[str, str]]],
) -> list[dict[str, str]]:
    """Interleave each subject's papers so the first few cover all of them.

    Interleaving is what makes batching safe; without it, batching is worse than
    the three separate calls it replaces. Everything downstream of a tool is
    clipped -- the consulting model sees
    :data:`src.agent.llm.MAX_TOOL_RESULT_CHARACTERS` of the result and the answering
    call sees :data:`src.agent.graph.MAX_RENDERED_TOOL_RESULT` of it -- and those
    clips take a prefix. Three subjects' results laid end to end therefore put every
    paper on the last two subjects past the cut: a question about VQE, QAOA and
    VarQITE came back with twelve papers, of which the four the answer could see
    were all about VQE. Three separate searches did not have that failure, because
    each got its own window.

    Interleaved, a prefix of any length is a fair sample of every subject asked
    about, which is the property a clip cannot take away.

    Args:
        per_subject: The admitted papers, one list per subject, in the order the
            subjects were asked for.

    Returns:
        One flat list: each subject's first paper, then each subject's second, and
        so on until they are exhausted.
    """
    ordered: list[dict[str, str]] = []
    for rank in range(max((len(group) for group in per_subject), default=0)):
        ordered.extend(group[rank] for group in per_subject if rank < len(group))
    return ordered


@tool(parse_docstring=True)
def search_arxiv(
    queries: list[str],
    limit: int = DEFAULT_ARXIV_RESULTS,
    shelf: Literal[
        "quantum-computing", "physics-notes", "method-comparison", "applications"
    ] = "quantum-computing",
) -> dict[str, Any]:
    """Search arXiv for papers on one or more subjects and return their abstracts.

    arXiv is the open preprint server where nearly all quantum-computing and
    condensed-matter physics work appears first. Use this when the reader asks
    for papers, references, recent work, or who has studied something -- and
    also when the project's own notes came back with nothing useful and an
    honest answer needs a source from outside them.

    What comes back is the abstract as arXiv holds it, never a summary invented
    here. Every record is screened before it is returned, and one whose text
    carries instruction-like patterns is dropped rather than shown: an abstract
    is untrusted text from an index nobody in this project controls, and it is
    about to be read by a language model. Refusals are reported as a count and a
    reason, never by quoting what was refused.

    Cite what you use by its arXiv id. Say that these came from arXiv and have
    not been reviewed by this project, which is the difference between them and
    the curated notes.

    **Ask for every subject at once.** A question about three algorithms is one
    call with three entries, not three calls. Each search is a network round trip
    and the entries here are run together, so three subjects in one call cost
    about what one costs; three separate calls cost three round trips *plus* two
    extra turns of the consultation, and the reader waits through all of it.

    Args:
        queries: What to search for, one entry per subject, in the words a
            person would use. Author names, arXiv ids and phrases in quotes all
            work. Give several only when they are genuinely different subjects:
            rephrasings of one subject return the same papers and are dropped as
            duplicates, having cost the search anyway.
        limit: How many records to keep per subject, at most. Small on purpose:
            each abstract is a paragraph of context, and eight of them crowd out
            the answer.
        shelf: Which of the project's knowledge bases these results belong
            beside. It changes nothing about the search and is reported back so
            an answer can say where the material would be filed.

    Returns:
        A dictionary with the subjects as searched, the papers found -- title,
        citation line, arXiv id, link and abstract -- and a count of anything
        the screen refused. A paper matching two of the subjects appears once:
        the same abstract twice is context spent to say nothing. An empty list
        is an ordinary result and means the index held nothing, not that the
        tool failed.
    """
    wanted = [query.strip() for query in queries if query.strip()]
    if not wanted:
        return {"error": "no query given; name at least one subject to search for"}
    if limit < 1:
        return {"error": f"limit must be at least 1, got {limit}"}
    kept = min(limit, MAX_ARXIV_RESULTS)

    from src.rag.external import admit, fetch

    def search(query: str) -> list[Any]:
        """Fetch one subject's candidates.

        Args:
            query: The subject.

        Returns:
            What the index returned, before the screen.
        """
        return list(fetch(query, shelf, limit=kept))

    # Order is preserved rather than taken from whichever request returned first,
    # so that the same question cites the same papers on two different days. See
    # `src.rag.retrieve._in_parallel`, which searches the project's own notes the
    # same way and for the same reason.
    if len(wanted) == 1:
        found = [search(wanted[0])]
    else:
        with ThreadPoolExecutor(
            max_workers=min(len(wanted), MAX_ARXIV_SEARCHES), thread_name_prefix="arxiv"
        ) as pool:
            found = list(pool.map(search, wanted))

    admitted: list[list[dict[str, str]]] = []
    refused: list[str] = []
    seen: set[str] = set()
    for candidates in found:
        kept_here: list[dict[str, str]] = []
        for candidate in candidates:
            decision = admit(candidate)
            if not decision.admitted:
                refused.append(decision.reason)
                continue
            if candidate.identifier in seen:
                continue
            seen.add(candidate.identifier)
            kept_here.append(
                {
                    "title": candidate.title,
                    "citation": candidate.source,
                    "arxiv_id": candidate.identifier,
                    "url": f"https://arxiv.org/abs/{candidate.identifier}",
                    "abstract": candidate.body,
                }
            )
        admitted.append(kept_here)
    papers = _one_from_each_in_turn(admitted)[:MAX_ARXIV_PAPERS]
    return {
        "queries": tuple(wanted),
        "shelf": shelf,
        "n_found": len(papers),
        "papers": papers,
        "n_refused": len(refused),
        "refused_because": tuple(dict.fromkeys(refused)),
        "note": (
            "Abstracts are reproduced from arXiv and have NOT been reviewed by "
            "this project. Cite each one by its arxiv_id and say so. An empty "
            "papers list means the index returned nothing for these subjects -- "
            "rephrase them or say plainly that nothing was found, and do not "
            "supply a reference from memory."
            if papers
            else "Nothing came back. The index may be unreachable, or the query "
            "may match nothing. Say that no paper was found rather than naming "
            "one from memory: a fabricated citation is the worst failure this "
            "tool can produce."
        ),
    }


@tool(parse_docstring=True)
def estimate_token_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    calls: int = 1,
) -> dict[str, Any]:
    """Estimate what a number of language-model calls would cost in dollars.

    A *token* is the unit models are billed in -- roughly three quarters of an
    English word, so a thousand tokens is about 750 words. Providers charge a
    different rate for what is sent (the prompt) and what comes back (the
    completion), and the completion rate is usually several times the prompt
    rate, which is why a long reply costs more than a long question.

    Use this before recommending a model, a retry count or a sweep, so the
    recommendation carries its price. It is the model-call twin of
    estimate_measurement_cost: one prices a quantum experiment in measurements,
    this prices a language-model run in dollars, and a feasibility answer that
    names one without the other is only half costed.

    Rates come from this project's own catalogue, and the "basis" field says how
    much to trust each one: "published" is the provider's list price,
    "estimated" is reasoned from a neighbouring model in the same family because
    this subscription lists its models and not their prices, and "unknown" means
    no rate at all -- in which case the cost is reported as null rather than
    guessed.

    Args:
        model: The model's slug, as the gateway names it, for example
            "openai/gpt-4o-mini".
        prompt_tokens: Tokens sent in one call. Count them if you can; a rule of
            thumb is characters divided by four.
        completion_tokens: Tokens the model returns in one call.
        calls: How many such calls. A campaign makes several, so this is the
            multiplier that turns a per-call rate into a run's budget.

    Returns:
        A dictionary with the per-call and total cost in US dollars, the rates
        used, and the basis for them. A single "error" key if the arguments were
        not usable.
    """
    if prompt_tokens < 0 or completion_tokens < 0:
        return {
            "error": (
                f"token counts cannot be negative, got prompt_tokens={prompt_tokens} "
                f"and completion_tokens={completion_tokens}"
            )
        }
    if calls < 1:
        return {"error": f"calls must be at least 1, got {calls}"}
    if not model.strip():
        return {"error": "model is empty; name the slug the gateway would be called with"}

    slug = model.strip()
    basis = price_basis(slug)
    price = price_for(slug) or estimated_price_for(slug)
    if price is None:
        return {
            "model": slug,
            "basis": basis,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "calls": calls,
            "cost_per_call_usd": None,
            "total_cost_usd": None,
            "note": (
                f"No rate on record for {slug}. The call would work; its cost cannot "
                "be stated. Report the token counts and say the spend is unpriced "
                "rather than substituting a rate from another model."
            ),
        }
    per_call = price.cost(prompt_tokens, completion_tokens)
    return {
        "model": slug,
        "basis": basis,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "calls": calls,
        "prompt_usd_per_million_tokens": price.input_usd,
        "completion_usd_per_million_tokens": price.output_usd,
        "cost_per_call_usd": per_call,
        "total_cost_usd": per_call * calls,
        "note": (
            "Rates are per million tokens. "
            + (
                "These are the provider's published prices."
                if basis == "published"
                else "These are ESTIMATED from a neighbouring model in the same "
                "family, because this subscription lists its models and not its "
                "prices. Quote the figure as an estimate."
            )
        ),
    }


MAX_EXPRESSION_CHARACTERS = 400
"""Longest expression the symbolic tool will accept.

A bound rather than a preference. Symbolic algebra has no useful upper limit on
how long one expression can take -- a nested integral of a few dozen characters
can run for minutes -- and a model that can hand this tool five thousand
characters is a model that can hang the process. Four hundred characters is more
than any equation in this project needs.
"""

SAFE_SYMBOLIC_NAMES: tuple[str, ...] = (
    # The constructors the parser itself emits. Without these nothing parses at
    # all, because a transformation turns "2" into Integer(2) and "J" into
    # Symbol("J") before the namespace is consulted.
    "Integer",
    "Float",
    "Rational",
    "Symbol",
    "Function",
    "Pow",
    "Mul",
    "Add",
    # Functions. Everything a Hamiltonian, a dispersion relation or a correlation
    # length in this project is written with, and nothing else.
    "sqrt",
    "exp",
    "log",
    "ln",
    "Abs",
    "sign",
    "sin",
    "cos",
    "tan",
    "cot",
    "sec",
    "csc",
    "asin",
    "acos",
    "atan",
    "atan2",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
    "Min",
    "Max",
    "factorial",
    "binomial",
    "gamma",
    "erf",
    "re",
    "im",
    "conjugate",
    # Constants.
    "pi",
    "E",
    "I",
    "oo",
    "zoo",
    "nan",
    "GoldenRatio",
    "EulerGamma",
)
"""Every name the symbolic parser may resolve, and the whole of it.

An allowlist, because the alternative is a denylist of ways to escape a Python
``eval`` and nobody has ever finished writing one of those. A name absent from
here does not resolve to a Python object at all: it either becomes an
unevaluated symbolic function or raises, and both are safe outcomes.

Adding to this list is adding to the attack surface. A name belongs here only if
it is a mathematical function whose arguments are numbers.
"""


def _safe_parse(sympy: Any, expression: str) -> Any:
    """Parse an expression to a sympy object without evaluating Python.

    ``sympy.sympify`` is the obvious way to do this and it must never be used
    here: it is documented as unsafe on untrusted input because it calls
    ``eval``, and it means what it says --
    ``sympify("__import__('os').getpid()")`` runs, and returns the pid. That is
    remote code execution on this machine, because **the argument to this tool is
    not written by the user**. It is written by the model, which reads arXiv
    abstracts fetched over the network; :func:`src.rag.external.admit` screens
    those for injection, and a screen is a filter rather than a proof.

    Three independent things have to hold, so that any one of them failing is
    still not an escape:

    1. No dunder survives the screen. ``(1).__class__.__mro__[-1]`` reaches
       every class the interpreter has loaded, and from there ``os``. Attribute
       access cannot be switched off in the parser, so the text is refused
       instead. Nothing in mathematics needs a double underscore.
    2. The namespace holds no builtins. ``__builtins__`` is set to an empty
       mapping explicitly, because Python inserts the real one into any globals
       dict that omits it -- an empty ``global_dict`` is *not* an empty
       namespace.
    3. The result must be a sympy expression. Anything that got through and
       returned a list, a class or a module is refused here rather than handed to
       the task, which is the check that does not depend on having predicted the
       route.

    Args:
        sympy: The imported module, passed in because the import is lazy.
        expression: The text to parse, already length-checked.

    Returns:
        The parsed expression, as a ``sympy.Basic``.

    Raises:
        ValueError: If the text is refused by any of the three checks. The
            message is written to be shown to the model, which can rewrite the
            expression and try again.
    """
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations

    if "__" in expression:
        raise ValueError(
            "a double underscore is not allowed in an expression. Write the "
            "mathematics with functions and symbols only."
        )

    namespace: dict[str, Any] = {"__builtins__": {}}
    for name in SAFE_SYMBOLIC_NAMES:
        namespace[name] = getattr(sympy, name)

    parsed = parse_expr(
        expression,
        global_dict=namespace,
        transformations=standard_transformations,
        evaluate=True,
    )
    if not isinstance(parsed, sympy.Basic):
        raise ValueError(
            f"that parsed to a {type(parsed).__name__} rather than to an "
            "expression. Write a mathematical expression."
        )
    return parsed


SymbolicTask = Literal[
    "solve",
    "simplify",
    "expand",
    "factor",
    "differentiate",
    "integrate",
    "evaluate",
    "limit",
]
"""What the symbolic tool may be asked to do.

An enumeration rather than free text, so the schema the model is shown carries
the whole list and a task nobody implemented is rejected by pydantic instead of
by a stack trace here.
"""


@tool(parse_docstring=True)
def solve_symbolic_maths(
    expression: str,
    task: SymbolicTask = "solve",
    variable: str = "x",
    at: str = "0",
) -> dict[str, Any]:
    """Do exact algebra or calculus on an expression, symbolically.

    Symbolic means the answer comes back as an expression rather than a decimal:
    asking for the derivative of ``cos(2*t)`` returns ``-2*sin(2*t)``, not a
    number. Use it wherever a step of algebra would otherwise be done in prose --
    solving for a critical point, differentiating an energy, checking a limit,
    simplifying a messy expression before quoting it.

    Use it rather than doing the algebra yourself. This is the one tool here that
    exists to stop a plausible-looking manipulation from reaching the reader
    unchecked, and a wrong sign in a derivative is exactly the error a fluent
    answer hides best.

    Notation is ordinary programming notation: ``*`` for multiplication, ``**``
    for powers, ``sqrt(x)``, ``exp(x)``, ``sin(x)``, ``pi``. Write an equation as
    an expression equal to zero -- for ``h**2 = J**2`` pass ``h**2 - J**2``.

    Args:
        expression: The expression, in the notation above. For "solve" it is
            taken as equal to zero.
        task: What to do with it. "solve" finds the values of the variable that
            make it zero; "simplify", "expand" and "factor" rewrite it;
            "differentiate" and "integrate" apply calculus in the named
            variable; "evaluate" reduces it to a number; "limit" takes the limit
            as the variable approaches "at".
        variable: Which symbol the task acts on. Ignored by "simplify",
            "expand", "factor" and "evaluate".
        at: Where the limit is taken, used only by "limit". Accepts a number,
            "oo" for infinity, or "-oo".

    Returns:
        A dictionary with the result in both ordinary and LaTeX notation, and the
        task and input echoed back so an answer can show its working. A single
        "error" key if the expression could not be read, the task could not be
        carried out, or the library that does this work is not installed.
    """
    if not expression.strip():
        return {"error": "expression is empty; pass something to work on"}
    if len(expression) > MAX_EXPRESSION_CHARACTERS:
        return {
            "error": (
                f"expression is {len(expression)} characters, above the tool limit of "
                f"{MAX_EXPRESSION_CHARACTERS}. Break it into steps."
            )
        }
    try:
        import sympy
    except ImportError:
        return {
            "error": (
                "symbolic algebra is unavailable: sympy is not installed in this "
                "environment. Answer without the algebra and say that the step was "
                "not checked, rather than doing it in prose."
            )
        }

    if "__" in variable:
        return {"error": "a double underscore is not allowed in a variable name."}
    try:
        symbol = sympy.Symbol(variable)
        parsed = _safe_parse(sympy, expression)
    except ValueError as refusal:
        # Raised by _safe_parse when the text is refused rather than unparseable.
        # Passed through as it is: it already says which rule it broke.
        return {"error": str(refusal)}
    except (sympy.SympifyError, SyntaxError, TypeError, NameError, AttributeError) as error:
        return {
            "error": (
                f"could not read {expression!r} as an expression ({type(error).__name__}). "
                "Use * for multiplication and ** for powers, and write an equation as "
                "an expression equal to zero."
            )
        }

    try:
        result = _apply_symbolic_task(sympy, parsed, task, symbol, at)
    except Exception as error:
        # Returned rather than raised, like every other tool failure here: the
        # model can fix its own expression and try again, and a raise would take
        # the whole turn down with it.
        return {
            "error": (
                f"{task} failed on {expression!r} ({type(error).__name__}). Try a "
                "simpler form, or name a different variable."
            )
        }

    return {
        "task": task,
        "input": expression.strip(),
        "variable": variable,
        "result": str(result),
        "result_latex": sympy.latex(result),
        "note": (
            "Exact, not numerical: the result is an expression unless 'evaluate' "
            "was asked for. Quote it as it is rather than rounding it, and use "
            "result_latex when writing it into an explanation."
        ),
    }


def _apply_symbolic_task(
    sympy: Any,
    parsed: Any,
    task: SymbolicTask,
    symbol: Any,
    at: str,
) -> Any:
    """Carry out one symbolic task on a parsed expression.

    Split out of :func:`solve_symbolic_maths` so that the eight branches are one
    readable mapping rather than a wall inside a function that is already doing
    validation, parsing and formatting. It is deliberately the only place that
    knows what each task name means, so a task added to :data:`SymbolicTask`
    without a branch here fails loudly in the tests rather than silently
    returning the input.

    Args:
        sympy: The imported module, passed in because the import is lazy -- the
            library is optional and importing it at module scope would make the
            whole tool layer unimportable without it.
        parsed: The expression to act on.
        task: Which operation to apply.
        symbol: The variable the calculus and solving act on.
        at: Where a limit is taken, as text.

    Returns:
        The result, as a sympy expression or a list of them.

    Raises:
        ValueError: If the task is not one this function implements. The caller
            turns it into a returned error like any other failure.
    """
    if task == "solve":
        return sympy.solve(sympy.Eq(parsed, 0), symbol)
    if task == "simplify":
        return sympy.simplify(parsed)
    if task == "expand":
        return sympy.expand(parsed)
    if task == "factor":
        return sympy.factor(parsed)
    if task == "differentiate":
        return sympy.diff(parsed, symbol)
    if task == "integrate":
        return sympy.integrate(parsed, symbol)
    if task == "evaluate":
        return sympy.N(parsed)
    if task == "limit":
        return sympy.limit(parsed, symbol, _safe_parse(sympy, at))
    raise ValueError(f"no branch for symbolic task {task!r}")


SWEEP_CURVES: tuple[str, ...] = (
    "spectrum",
    "energy",
    "magnetisation",
    "energy_derivatives",
    "magnetisation_derivatives",
)
"""The curves :func:`field_sweep_tool` will trace, as the model names them.

Repeated here rather than imported, and that repetition is the seal rather than an
oversight: the list lives beside the solver in
``src.physics.reference.field_sweep.Curve``, importing it would put this module one
step from an exact answer, and the architecture test would fail on the spot.
A test holds the two tuples equal instead, so they cannot drift while staying apart.
"""

DEFAULT_SWEEP_POINTS = 41
"""Field values one sweep samples unless the model asks for another number.

Enough that the peak of the susceptibility and the minimum of the gap land within a
per cent or so of where they belong, and few enough that the table of them fits in a
context window beside an answer.
"""

MAX_SWEEP_POINTS = 201
"""Ceiling on one sweep. Past this the extra rows are digits, not information."""


SWEEP_ARRAYS: frozenset[str] = frozenset(
    {
        "ratio",
        "energy_density",
        "magnetisation",
        "energy_slope",
        "energy_curvature",
        "magnetisation_slope",
        "magnetisation_curvature",
        "thermodynamic_energy_density",
        "thermodynamic_gap",
        "excitations",
    }
)
"""Sweep payload keys held back from the model's context.

One float per point per quantity, which at the default point count is several
hundred numbers -- and a model handed several hundred numbers writes about numbers
instead of about the curve they describe. They are not discarded: the interface
re-runs the same call to draw the figure, so the arrays reach the reader as a
picture, which is the form the question asked for.
"""


class FieldSweepBench(Protocol):
    """What a lender of the exact field sweep has to provide.

    A structural type, deliberately: it is satisfied by the closure
    :func:`src.physics.registry.field_sweep_bench` returns **without either module
    importing the other**, which is the only reason this tool can exist at all. The
    solver is sealed from everything under ``src/agent/``; what arrives here is a
    function that speaks primitives, handed in at run time by the host.
    """

    def __call__(
        self,
        n_sites: int,
        coupling: float = ...,
        boundary: str = ...,
        curves: object = ...,
        points: int = ...,
        ratio_max: float = ...,
        levels: int = ...,
        geometry: str = ...,
        rows: int = ...,
    ) -> dict[str, Any]:
        """Trace the field range for one problem and return it as plain data."""
        ...


def field_sweep_tool(bench: FieldSweepBench) -> BaseTool:
    r"""Build the exact-field-sweep tool around a solver lent by the grader.

    The one tool in this project that is *constructed* rather than declared, and the
    reason is the wall. Every other tool here can be written down at import time
    because everything it needs is on the agent's side of it. This one runs the
    closed-form solution of the model -- which is the grader's -- so it exists only
    when a host chooses to lend it, and the lending is visible at the call site in
    :func:`src.agent.graph.run_campaign` rather than hidden in an import.

    Half the questions a reader actually asks about this chain are questions about
    a curve: how the spectrum's levels move as the field is
    turned up, how the magnetisation switches on, what the second derivative of the
    ground-state energy does near the transition. Those have right answers, and
    before this tool existed the agent answered them from a shelf of abstracts --
    fluent, sourced, and about the neighbourhood of the question rather than the
    question. That is the failure mode this closes.

    It does not void the experiment. This tool is offered only at the
    consultation node, which sits in front of the two prose branches. The
    feasibility branch -- the one that designs circuits, spends the shot budget and
    is graded against an exact answer -- does not pass through it and cannot call
    this. See :func:`src.physics.registry.field_sweep_bench`.

    Args:
        bench: The lent solver. See :class:`FieldSweepBench`.

    Returns:
        The tool, named ``exact_field_sweep``.
    """

    @tool(parse_docstring=True)
    def exact_field_sweep(
        n_sites: int,
        curves: list[
            Literal[
                "spectrum",
                "energy",
                "magnetisation",
                "energy_derivatives",
                "magnetisation_derivatives",
            ]
        ],
        coupling: float = 1.0,
        boundary: Literal["periodic", "open"] = "periodic",
        points: int = DEFAULT_SWEEP_POINTS,
        ratio_max: float = 2.0,
        levels: int = 6,
        geometry: Literal["chain", "square", "triangular"] = "chain",
        rows: int = 1,
    ) -> dict[str, Any]:
        r"""Solve the problem exactly at many field strengths and return the curves.

        Use this for any question about how something *changes as the sideways
        push is turned up* -- "plot the low-lying spectrum against the field",
        "how does the magnetisation turn on", "show the ground-state energy and
        its first and second derivatives against h/J". Nothing here is fitted,
        sampled or approximated.

        On a line it solves the problem in closed form (the free-fermion, or
        Jordan-Wigner, solution) at every field value in the range. On a square or
        triangular lattice there is no closed form, so every point is diagonalised
        instead -- which is exact but costs 2**L, so a lattice sweep is limited to
        16 sites and the second and higher derivatives are not available. Either
        way, where the problem is small enough, each point is computed a second
        time by a completely different method and the answer reports how far apart
        the two came out.

        The reader is shown the curves you ask for as a figure beneath your
        answer, drawn from this same call. So describe what the curves *do* and
        quote the numbers in the returned table; do not try to draw them yourself,
        and do not list every row.

        Glossary, since the terms are not general knowledge. *Spectrum*: the
        energies of the lowest few states of the whole chain, the lowest being the
        ground state. *Magnetisation* here means the average alignment with the
        sideways push, written as the expectation of the Pauli operator
        \(\sigma^x\); it runs from 0 (no alignment) to 1 (fully aligned).
        *Derivatives* are with respect to the field h, and are analytic rather
        than numerical. *h/J* is the field divided by the coupling, the one ratio
        the physics depends on, with the phase transition of the infinite chain at
        h/J = 1.

        Args:
            n_sites: Total number of magnets, L. On a line this must be even and
                the line must be a ring: the closed form exists for that case, and
                the tool says so rather than guessing when it does not. On a
                lattice any size up to 16 works, because a different method runs.
            curves: Which curves to trace, one or more of: "spectrum" (the
                low-lying energy levels), "energy" (ground-state energy per
                magnet), "magnetisation", "energy_derivatives" (the first and
                second derivative of the energy per magnet with respect to the
                field) and "magnetisation_derivatives" (the same two for the
                magnetisation). Ask for exactly what the question asked about:
                everything is computed either way, and this decides what the
                reader is shown and what you are told about.
            coupling: The interaction strength J between neighbours. The sweep is
                in h/J, so this only sets the units.
            boundary: "periodic" for a ring, "open" for a line with two ends. The
                closed form needs a ring.
            points: How many field values to sample between zero and ratio_max.
            ratio_max: The largest h/J to sweep to. The transition sits at 1, so
                the default of 2 shows both phases with the interesting part in
                the middle.
            levels: How many energy levels a "spectrum" request draws.
            geometry: The shape the magnets sit on -- "chain" for a line, or
                "square" or "triangular" for a two-dimensional lattice. **Pass the
                shape the question named.** A lattice's curve and a line's are
                different curves: at sixteen sites the energies differ by nearly a
                factor of two per magnet, because a lattice has more neighbours.
                Leaving this at "chain" for a lattice question draws the wrong
                problem's curve under the right problem's title.
            rows: How many rows of magnets, for a two-dimensional shape. The
                columns follow from n_sites, so a 3x3 square is n_sites=9 with
                rows=3. Leave at 1 for a line.

        Returns:
            The chain, which methods ran, whether the two independent solvers
            agreed and by how much, and a table of the requested curves with the
            features named by number: where the magnetisation rises fastest, where
            the energy's curvature dips deepest, what the gap is at the critical
            field, and how far this finite chain sits from the infinite one.
        """
        if n_sites > MAX_TOOL_SITES:
            return {
                "error": (
                    f"n_sites={n_sites} is above the tool limit of {MAX_TOOL_SITES}. "
                    "Sweep a shorter chain, and say in your answer that the shape of "
                    "these curves stops changing well before that length."
                )
            }
        answered = bench(
            n_sites=n_sites,
            coupling=coupling,
            boundary=boundary,
            curves=list(curves),
            points=max(3, min(int(points), MAX_SWEEP_POINTS)),
            ratio_max=ratio_max,
            levels=levels,
            geometry=geometry,
            rows=rows,
        )
        # The numeric arrays are for the figure the interface draws, not for the
        # answer: a narrator handed four hundred floats writes about floats. It
        # re-runs this same call to plot, so nothing is lost by leaving them out
        # here -- and what stays is what the answer has to be able to say, which
        # is why the two short lists are kept and only the long ones dropped.
        return {key: value for key, value in answered.items() if key not in SWEEP_ARRAYS}

    # `@tool` already returns a BaseTool; mypy knows it, so no cast is needed here.
    return exact_field_sweep


TOOLS: tuple[BaseTool, ...] = (
    describe_problem,
    list_methods,
    estimate_measurement_cost,
    assess_device_fit,
    run_classical_baseline,
    search_arxiv,
    estimate_token_cost,
    solve_symbolic_maths,
)
"""Every tool the agent may call, in the order it should usually reach for them.

An explicit tuple rather than a decorator registry, for the same reason the
method catalogue is: a tool that nobody listed does not silently gain the power
to run. The order is a hint and not a constraint -- look at the problem, see what
could solve it, cost it, then spend -- and it is the order a reader should meet
them in too.

**Five about the problem, three about the answer.** The first five ask what the
chain is, what could solve it, what a measurement would cost, whether a machine
could hold the circuit, and what an ordinary computer gets. The last three serve
the *answer* rather than the physics, and each one closes a route by which a
fluent reply could be wrong:

- :func:`search_arxiv` means a request for papers is answered from an index
  rather than from the model's memory, which is where fabricated citations come
  from.
- :func:`estimate_token_cost` prices the language-model half of a run, so a
  recommendation to use a stronger model arrives with its bill. It is the twin
  of :func:`estimate_measurement_cost`: one costs the quantum experiment in
  measurements, the other costs the agent in dollars.
- :func:`solve_symbolic_maths` does the algebra exactly, so a derivative or a
  critical point in an explanation is computed rather than asserted.
"""


def tool_names() -> tuple[str, ...]:
    """Return the name of every registered tool, in registration order."""
    return tuple(instrument.name for instrument in TOOLS)


def get_tool(name: str) -> BaseTool:
    """Look up one tool by the name the model calls it by.

    Args:
        name: A tool name, as returned by :func:`tool_names`.

    Returns:
        The tool.

    Raises:
        KeyError: If no tool is registered under that name. The message lists the
            valid names, because this is the failure mode when a model invents a
            tool that sounds like it ought to exist.
    """
    for instrument in TOOLS:
        if instrument.name == name:
            return instrument
    raise KeyError(f"unknown tool {name!r}; registered tools are {tool_names()}")


def catalogue_methods_named_in_tools() -> tuple[str, ...]:
    """Return the method names a tool can mention, for cross-checking prose.

    Used by the tests to assert that no tool description names a solver that the
    catalogue does not carry, which is how a stale description starts sending the
    model after something that was renamed two weeks ago.

    Returns:
        Every catalogued method name.
    """
    return method_names()
