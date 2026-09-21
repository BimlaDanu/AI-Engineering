r"""What a feasibility campaign knows, and what it has spent finding out.

One campaign answers one question: *is a quantum computer worth using for this
problem?* Answering it honestly means holding several things at once -- the
question exactly as it was asked, the Hamiltonian it was turned into, the budget
still unspent, every configuration already tried and why each was abandoned, the
classical number the quantum arm has to beat, and finally a verdict with the
condition under which it would change. This module is that bundle.

There are two shapes here and each has its reason. :class:`CampaignState` is a
``TypedDict`` because it is the schema the graph merges into: a node returns only
the keys it changed and the graph combines that with what was already there, and a
``TypedDict`` is the only shape in which "a mapping holding just two of these twelve
keys" is a type the checker can verify. Everything inside the state is a frozen
dataclass instead. Those are records rather than partial updates, and once a run has
happened nothing should be able to edit what it cost.

Budgets refuse rather than warn. :class:`ShotLedger` and :class:`CoherenceBudget`
both answer with a refusal string instead of a number the caller may ignore. A budget
written as an instruction in a prompt is a suggestion; one that returns "this costs
4,000,000 shots and you have 120,000" is a wall. The failure being guarded against is
an agent that spends whatever it likes and then reports success.

The question is stored verbatim. :attr:`Request.text` is never paraphrased, tidied or
stripped of pressure before a later node sees it. Whether a verdict moves when the
same problem is put enthusiastically or doubtfully can only be measured if the voice
survives the trip, and a helpful normalisation step early in the graph would launder
away the thing being measured.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict, TypeVar

from src.agent.diagnosis import Diagnosis, DiagnosisRecord
from src.agent.intent import Reading
from src.agent.llm import ToolCall
from src.physics.lattice import Geometry, Lattice

# Imported at module level rather than behind `TYPE_CHECKING`, and that is forced
# rather than chosen: LangGraph builds its channels by calling `get_type_hints` on
# `CampaignState`, which evaluates every annotation for real. A name that exists only
# for the type checker fails there with a bare `NameError` at graph-construction time.
from src.physics.quantum.method_race import MethodRace

ItemT = TypeVar("ItemT")
"""Element type of an accumulating state field. See :func:`extend`."""

Framing = Literal["neutral", "vendor", "skeptical"]
"""The voice a question was asked in.

The same problem put three ways. ``"neutral"`` states it plainly, ``"vendor"``
arrives with a claim of advantage already attached, and ``"skeptical"`` arrives
with the assumption that it will not work. A sound agent returns the same verdict
for all three; the gap between them is a measurement of how much the answer
depends on who is asking.
"""

Call = Literal["go", "no", "conditional"]
"""The three answers a feasibility question can honestly receive.

``"conditional"`` is not a hedge and not a way of avoiding a decision. It is the
correct answer whenever the obstruction is a stated, checkable quantity -- a
device error rate, a chain length, a coherence time -- and it is only usable if
the condition is written down alongside it.
"""

NOMINAL_TWO_QUBIT_GATE_NS = 300.0
"""Duration of one two-qubit gate, in nanoseconds.

A round figure for a superconducting device. It is the placeholder that lets a
campaign do real coherence arithmetic before a device model exists, and it is
stated as a constant rather than hidden in a default argument so that the day a
measured value replaces it, there is exactly one line to change.
"""

NOMINAL_COHERENCE_NS = 100_000.0
"""Coherence time :math:`T_2`, in nanoseconds -- 100 microseconds.

The same kind of placeholder as :data:`NOMINAL_TWO_QUBIT_GATE_NS`, and the same
caveat: a real device publishes a different :math:`T_2` for every qubit, and the
one that matters is the worst one on the path the circuit actually uses.
"""

USABLE_COHERENCE_FRACTION = 0.1
"""How much of :math:`T_2` a circuit may occupy before the answer is noise.

A circuit that runs for the full :math:`T_2` has, by the definition of :math:`T_2`,
lost most of its phase information -- the state at the end is not the state the
algorithm designed. A tenth is the conventional working figure: long enough to be
useful, short enough that the result still means something.
"""


@dataclass(frozen=True, slots=True)
class Request:
    """The question as it arrived, with nothing removed.

    Attributes:
        text: The user's words, unedited. Downstream nodes see this string and not
            a summary of it.
        framing: Which voice the question was asked in.
        screening_note: What the prompt-injection screen said about the text, kept
            beside it so a report can state that the input was checked. ``None``
            means the text was not screened, which is different from screened and
            found clean.
        in_scope: Whether the question is about something this project holds notes
            on. ``False`` stops the campaign before a Hamiltonian is invented for a
            question that named no physics -- which is the difference between an
            agent and a machine that answers everything.
        clarified: A tidied restatement of the question, when one was worth making.
            Stored *beside* the original and never in place of it, for the reason in
            the class docstring: a helpful normalisation that replaced the text would
            launder away the framing this project exists to measure. It is used to
            build a search query, where tone is noise and precision helps, and it is
            shown next to the original so a reader can see what was understood.
            Empty when the question needed no help or no model was available.
    """

    text: str
    framing: Framing = "neutral"
    screening_note: str | None = None
    clarified: str = ""
    in_scope: bool = True

    @property
    def blocked(self) -> bool:
        """Whether the input screen rejected this question.

        A property rather than a string comparison repeated at each call site.
        Two places need the answer -- the node that refuses to formalise a blocked
        request, and the one that refuses to suggest follow-ups for it -- and two
        spellings of the same test is how one of them ends up not being applied.
        """
        return self.screening_note is not None and self.screening_note.startswith("blocked")

    @property
    def for_search(self) -> str:
        """The wording to search the corpus with.

        The clarified restatement when there is one, and the original otherwise. This
        is the *only* place the restatement is allowed to win, and it is safe here
        precisely because a retrieval query has no verdict in it: which passages come
        back should not depend on whether the asker sounded hopeful.
        """
        return self.clarified or self.text


@dataclass(frozen=True, slots=True)
class FormalModel:
    """The Hamiltonian the question was turned into, and the leap that took.

    Turning a sentence into a model is the step where a feasibility study most
    often goes quietly wrong, because the model is then treated as the problem for
    the rest of the study. So the assumptions are stored next to the specification
    rather than left in the reasoning that produced it: a verdict that depends on
    "assumed nearest-neighbour coupling only" is a different verdict once that
    assumption is written down where a reader can reject it.

    Attributes:
        n_sites: Number of spins in the chain.
        coupling: The Ising coupling :math:`J`.
        transverse_field: The transverse field :math:`h`.
        longitudinal_field: The field :math:`g` along the coupling axis. Zero
            leaves the chain exactly solvable, which matters to the verdict: an
            exactly solvable problem has no quantum advantage to offer, however
            well the circuit performs.
        boundary: ``"open"`` for a segment, ``"periodic"`` for a ring.
        geometry: The shape the sites sit on. A chain unless the question named
            something else. It is a field rather than an inference from
            ``n_sites``, because sixteen sites is a chain, a four-by-four square
            and a two-by-eight rectangle, and guessing between them would put a
            shape into the report that nobody asked for.
        rows: Rows of sites for a two-dimensional lattice, and one for a chain.
            The columns follow from ``n_sites``, so the two cannot disagree.
        assumptions: Each modelling choice that was made rather than given, in
            plain words a non-specialist can disagree with.
        length_was_given: Whether the chain length came from the question or from
            the conversation before it, rather than from a default. It decides
            whether a *verdict* may be reached: a feasibility call is a statement
            about one specific problem, so reaching one about a chain nobody
            mentioned is a confident answer to a question nobody asked. Kept as a
            field rather than re-derived, because by the time a router wants it the
            conversation that supplied the length is no longer in reach.
    """

    n_sites: int
    coupling: float = 1.0
    transverse_field: float = 1.0
    longitudinal_field: float = 0.0
    boundary: Literal["periodic", "open"] = "open"
    geometry: Geometry = "chain"
    rows: int = 1
    assumptions: tuple[str, ...] = ()
    length_was_given: bool = True

    @property
    def lattice(self) -> Lattice:
        """The shape as the physics layer wants it, built from the fields above.

        Derived rather than stored so that ``n_sites`` stays the single authority on
        how big the problem is: every consumer in the project already reads it, and a
        second stored size is a second thing to keep in step.

        Returns:
            The lattice. Its columns are ``n_sites // rows``.

        Raises:
            ValueError: If the rows do not divide the site count, which would
                describe a ragged lattice that nothing can solve.
        """
        if self.n_sites % self.rows != 0:
            raise ValueError(
                f"{self.rows} rows do not divide {self.n_sites} sites into a rectangle"
            )
        return Lattice(
            geometry=self.geometry,
            rows=self.rows,
            cols=self.n_sites // self.rows,
            boundary=self.boundary,
        )

    def label(self) -> str:
        """A short identifier for logs, figure legends and the composed answer.

        ``g`` appears only when it is non-zero. Naming a parameter in order to report
        that it is switched off is how a two-parameter problem comes to look like a
        three-parameter one.

        Returns:
            The label. Matches :meth:`src.physics.model.TFIMSpec.label` term for term at
            ``g = 0``, which is what lets a run and its grading be lined up by name.
        """
        shape = "" if self.geometry == "chain" else f" {self.rows}x{self.n_sites // self.rows}"
        parts = [
            f"TFIM{shape} L={self.n_sites}",
            f"J={self.coupling:g}",
            f"h={self.transverse_field:g}",
        ]
        if self.longitudinal_field != 0.0:
            parts.append(f"g={self.longitudinal_field:g}")
        return f"{' '.join(parts)} ({self.boundary[:4]})"

    @property
    def is_exactly_solvable(self) -> bool:
        r"""Whether this model has a closed-form answer and so cannot need a quantum computer.

        At :math:`g = 0` **on a chain** the model maps to free fermions and its
        ground-state energy is a sum of :math:`L` cosines -- microseconds of
        arithmetic at any length. Every honest feasibility verdict for such a
        problem is "no", regardless of how well the circuit runs, and an agent that
        fails to say so has missed the entire question. A non-zero :math:`g` breaks
        the mapping and is what makes the question worth asking.

        Both conditions matter, and the geometry is the easier one to forget. The
        Jordan-Wigner transformation turns the chain's :math:`\hat\sigma^z
        \hat\sigma^z` into a quadratic fermion term because each site has exactly
        one forward neighbour and the string between them is empty. On a square or
        triangular lattice it does not: the string runs through every site the
        row-major order puts in between, the coupling becomes a many-body operator,
        and no closed form survives. A two-dimensional Ising model in a transverse
        field has no known exact ground-state energy at all.

        Reading only :math:`g` here would have the verdict layer state, confidently
        and in fluent prose, that a 4x4 lattice reduces to independent particles --
        the same failure this screen prevents, with the sign reversed.
        """
        return self.longitudinal_field == 0.0 and self.geometry == "chain"


@dataclass(frozen=True, slots=True)
class ShotLedger:
    """A measurement budget that is enforced by arithmetic rather than by asking.

    A "shot" is one preparation of the circuit followed by one measurement. Every
    energy a real device reports is an average over many of them, so shots are the
    currency a quantum run is actually priced in, and a plan that ignores the price
    is not a plan.

    Attributes:
        budget: Total shots the campaign may spend.
        spent: Shots consumed so far.
    """

    budget: int
    spent: int = 0

    def __post_init__(self) -> None:
        """Reject a ledger that could not describe a real budget.

        Raises:
            ValueError: If the budget is negative, or more has been spent than exists.
        """
        if self.budget < 0:
            raise ValueError(f"a shot budget cannot be negative, got {self.budget}")
        if self.spent < 0:
            raise ValueError(f"shots spent cannot be negative, got {self.spent}")
        if self.spent > self.budget:
            raise ValueError(f"spent {self.spent} shots against a budget of {self.budget}")

    @property
    def remaining(self) -> int:
        """Shots still available to spend."""
        return self.budget - self.spent

    def refusal(self, shots: int) -> str | None:
        """Say why a purchase cannot go ahead, or nothing if it can.

        A string rather than a boolean because the refusal is read by whatever
        chooses the next configuration, and "you are short by 300,000 shots" tells
        it how much cheaper the next attempt has to be. A bare ``False`` would
        force it to guess.

        Args:
            shots: What the next run would cost.

        Returns:
            A sentence naming the shortfall, or ``None`` when the run is affordable.
        """
        if shots < 0:
            return f"a run cannot cost a negative number of shots, got {shots}"
        if shots > self.remaining:
            return (
                f"this run needs {shots:,} shots and only {self.remaining:,} remain "
                f"of the {self.budget:,} budgeted"
            )
        return None

    def debit(self, shots: int) -> ShotLedger:
        """Spend shots and return the ledger that results.

        Args:
            shots: How many to spend.

        Returns:
            A new ledger. The original is unchanged, so a rejected branch of the
            graph cannot leave the budget quietly reduced.

        Raises:
            ValueError: If the spend is not affordable. Callers are expected to ask
                :meth:`refusal` first; reaching this exception means a run was
                started without pricing it, which is the failure this class exists
                to prevent and so is loud rather than silent.
        """
        reason = self.refusal(shots)
        if reason is not None:
            raise ValueError(f"shot budget exceeded: {reason}")
        return ShotLedger(budget=self.budget, spent=self.spent + shots)


@dataclass(frozen=True, slots=True)
class CoherenceBudget:
    """How deep a circuit may be before the qubits forget what they were doing.

    Qubits hold their phase relationships for a limited time, :math:`T_2`. A
    circuit whose duration approaches that window returns a state the algorithm did
    not design, so depth is not free and cannot be traded for accuracy without
    limit. Two-qubit gates dominate the duration -- they are roughly an order of
    magnitude slower than single-qubit rotations -- so two-qubit depth is what
    :meth:`refusal` counts, from nothing but the layer structure.

    That is a floor on the duration and not the duration. Single-qubit layers and
    the readout at the end are real time the qubits spend dephasing, and on a
    superconducting machine readout alone is a sixth of the window. Once a circuit
    has been compiled its whole schedule is known, and :meth:`duration_refusal`
    charges that instead -- which is what a caller holding a compiled circuit
    should use.

    The point of checking this *before* a run is that it costs nothing. A plan can
    be rejected on arithmetic alone, with no shots spent and no simulation done,
    which is exactly what a practitioner does before booking device time.

    Attributes:
        two_qubit_gate_ns: Duration of one two-qubit gate.
        coherence_ns: The coherence time :math:`T_2`.
        usable_fraction: How much of :math:`T_2` a circuit may occupy.
    """

    two_qubit_gate_ns: float = NOMINAL_TWO_QUBIT_GATE_NS
    coherence_ns: float = NOMINAL_COHERENCE_NS
    usable_fraction: float = USABLE_COHERENCE_FRACTION

    def __post_init__(self) -> None:
        """Reject a budget whose numbers could not describe a device.

        Raises:
            ValueError: If any duration is not positive, or the fraction is not
                between zero and one.
        """
        if self.two_qubit_gate_ns <= 0.0:
            raise ValueError(f"gate duration must be positive, got {self.two_qubit_gate_ns}")
        if self.coherence_ns <= 0.0:
            raise ValueError(f"coherence time must be positive, got {self.coherence_ns}")
        if not 0.0 < self.usable_fraction <= 1.0:
            raise ValueError(f"usable fraction must be in (0, 1], got {self.usable_fraction}")

    @property
    def usable_ns(self) -> float:
        """The slice of :math:`T_2` a circuit may occupy."""
        return self.usable_fraction * self.coherence_ns

    @property
    def max_two_qubit_depth(self) -> int:
        """The deepest circuit whose entangling layers alone fit inside the window."""
        return math.floor(self.usable_ns / self.two_qubit_gate_ns)

    def duration_ns(self, two_qubit_depth: int) -> float:
        """How long a circuit of this two-qubit depth would take to run.

        Args:
            two_qubit_depth: Layers of two-qubit gates, counted after scheduling.

        Returns:
            The duration in nanoseconds.
        """
        return two_qubit_depth * self.two_qubit_gate_ns

    def duration_refusal(self, duration_ns: float) -> str | None:
        """Say why a compiled circuit runs too long, or nothing if it fits.

        The same window as :meth:`refusal` against a schedule that is known rather
        than estimated, so it is the stricter of the two and the one to prefer
        wherever the circuit has already been compiled.

        Args:
            duration_ns: How long the whole schedule occupies the machine.

        Returns:
            A sentence comparing the duration against the window, or ``None`` when
            it fits.
        """
        if duration_ns < 0.0:
            return f"a circuit cannot run for a negative time, got {duration_ns}"
        if duration_ns > self.usable_ns:
            return (
                f"this circuit runs for {duration_ns / 1000:.1f} microseconds, past the "
                f"{self.usable_fraction:.0%} of the {self.coherence_ns / 1000:.0f}-microsecond "
                f"coherence time this device can use -- {self.usable_ns / 1000:.1f} "
                f"microseconds is all there is"
            )
        return None

    def refusal(self, two_qubit_depth: int) -> str | None:
        """Say why a circuit is too deep to run, or nothing if it fits.

        Args:
            two_qubit_depth: Layers of two-qubit gates the circuit needs.

        Returns:
            A sentence comparing the circuit's duration against the window, or
            ``None`` when it fits.
        """
        if two_qubit_depth < 0:
            return f"two-qubit depth cannot be negative, got {two_qubit_depth}"
        if two_qubit_depth > self.max_two_qubit_depth:
            return (
                f"a circuit of two-qubit depth {two_qubit_depth} runs for "
                f"{self.duration_ns(two_qubit_depth) / 1000:.1f} microseconds, past the "
                f"{self.usable_fraction:.0%} of the {self.coherence_ns / 1000:.0f}-microsecond "
                f"coherence time this device can use -- the deepest that fits is "
                f"{self.max_two_qubit_depth}"
            )
        return None


@dataclass(frozen=True, slots=True)
class PlannedRun:
    """A configuration that has been costed but not yet run.

    Priced before it is run, and priced by arithmetic rather than by building the
    circuit: the layer structure fixes the gate counts, so twenty candidate depths
    cost twenty multiplications to compare. That is what makes it reasonable to
    reject a plan before spending anything on it.

    Attributes:
        label: Short identifier, carried through to the :class:`RunRecord` if the
            run goes ahead and to the :class:`RuledOut` entry if it does not.
        family: Which circuit family to run.
        depth: Number of ansatz layers proposed.
        two_qubit_depth: What that costs in two-qubit layers, after the even/odd
            scheduling that lets a whole chain be covered in a fixed number of
            rounds regardless of its length.
        shots: What the run would cost from the measurement budget.
        reason: Why this configuration is the one worth trying next. Recorded
            because a plan without a reason cannot be argued with afterwards.
    """

    label: str
    family: str
    depth: int
    two_qubit_depth: int
    shots: int
    reason: str


@dataclass(frozen=True, slots=True)
class RunRecord:
    """One configuration that was actually tried, and what it produced.

    Kept even when the run went badly, and especially then. A campaign that only
    records its successes cannot tell the difference between "the best of six
    honest attempts" and "the first thing that worked", and those two claims
    deserve very different amounts of trust.

    Attributes:
        label: Short identifier for the configuration, unique within a campaign.
        family: Which circuit family was run.
        depth: Number of ansatz layers.
        two_qubit_depth: Two-qubit depth after the even/odd scheduling.
        energy: The lowest energy the run reached. An upper bound on the true
            ground state, never a lower one.
        energy_per_site: The same energy divided by the chain length, which is what
            makes two different chain lengths comparable.
        shots_spent: What the run cost from the ledger, priced before it ran.
        energy_evaluations: Energies the optimiser actually asked for. The budget
            above was priced from :data:`~src.agent.graph.EVALUATIONS_PER_LAYER`
            before anything ran, and this is the count that estimate is answerable
            to -- a run that needed far more readings than budgeted is a finding
            about the landscape rather than a bookkeeping error.
        shots_at_true_variance: What the same readings would have cost had the
            variance of the prepared state been known in advance. The budget is
            priced from a worst case that charges every term the largest variance
            its spectrum allows, so this is always the smaller number, and the two
            side by side say how loose the price the run was approved on was. Zero
            when nothing measured it.
        diagnosis: What the run's own evidence says about it.
        parameters: The angles that reached this energy. Carried for two reasons:
            the next depth up starts from them rather than from scratch, which is
            what makes a sweep cost about as much as its deepest run alone; and a
            result nobody can reproduce the settings for is not a result.
        energy_history: The energy after each optimiser step, oldest first -- the
            run's loss curve. Kept because the final number alone cannot tell a
            search that descended steadily from one that stalled on its first step
            and one that fell off a cliff and recovered, and those are three
            different things to report. It is also what lets the interface draw the
            descent for whatever chain the question named, rather than for a stock
            example: the curve belongs to this campaign's own J, h and g.
    """

    label: str
    family: str
    depth: int
    two_qubit_depth: int
    energy: float
    energy_per_site: float
    shots_spent: int
    diagnosis: Diagnosis
    energy_evaluations: int = 0
    shots_at_true_variance: int = 0
    parameters: tuple[float, ...] = ()
    energy_history: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class BaselineResult:
    """The classical number the quantum arm has to beat.

    Running this is not optional and not a step the planner may skip when the
    quantum result looks good. Reporting a quantum energy with no classical
    comparison is the characteristic failure of the field, and it is a failure of
    method rather than of physics: without the comparison there is no claim, only a
    number.

    Attributes:
        method: Which classical method produced it.
        energy_per_site: The energy it reached, per spin. An upper bound.
        energy_error: One standard deviation of statistical uncertainty. A
            difference between two energies smaller than this is not a difference.
        n_measurements: What the classical run cost, in its own currency.
        confidence: How far this number may be leaned on. ``high`` is the
            calibrated regime; ``low`` means the method ran but something the
            comparison depends on is unreliable -- in practice the error bar,
            which is the quantity :data:`src.agent.verdict.ADVANTAGE_MARGIN`
            divides by. Defaults to ``high`` so that existing callers keep the
            behaviour they had.
        caveat: What could not be checked, in words a reader with no physics can
            act on. Empty when the baseline is sound. Carried here rather than
            recomputed downstream because the verdict and the report both need
            it and neither may guess at it.
    """

    method: str
    energy_per_site: float
    energy_error: float
    n_measurements: int
    confidence: Literal["low", "medium", "high"] = "high"
    caveat: str = ""

    @property
    def uncertainty_is_usable(self) -> bool:
        """Whether :attr:`energy_error` is a number a comparison can divide by.

        It is not always. ``_bin_error`` in the classical baseline returns ``nan``
        for a single bin **on purpose** -- one bin carries no information about its
        own spread, and returning ``0.0`` would let a caller divide by it and call
        the result significant. The verdict then has to honour that choice, and it
        did not: every comparison against ``nan`` is false, so a signed margin
        checked against ``> nan`` and ``< -nan`` fell through both branches into
        the tie case, and the report said *"the gap of 0.100000 is inside the
        classical error bar"* beside a printed ``± nan``. Both halves of that
        sentence were false and the reader had no way to tell.

        A non-positive bar is refused for the same reason: a lead measured in units
        of zero is infinite, so every quantum result would win.
        """
        return math.isfinite(self.energy_error) and self.energy_error > 0.0

    @property
    def is_trustworthy(self) -> bool:
        """Whether a verdict may lean on this number without qualifying it.

        A property rather than a comparison written out at each call site: the
        verdict screen and the report both ask this question, and two spellings
        of one rule is one spelling that will eventually be wrong.
        """
        return self.confidence == "high" and self.uncertainty_is_usable


@dataclass(frozen=True, slots=True)
class RuledOut:
    """A configuration that will not be tried, and the reason it was dropped.

    This is what stops a campaign going in circles. An agent that rejects a depth
    for exceeding the coherence budget and then proposes it again three steps later
    has not reasoned about it -- it has forgotten -- and the difference is visible
    only if the rejection was written down.

    Attributes:
        label: The configuration, named the same way a :class:`RunRecord` would name it.
        reason: Why it was ruled out, in words that would let someone re-try it
            deliberately if they disagreed.
    """

    label: str
    reason: str


@dataclass(frozen=True, slots=True)
class Belief:
    """Something the campaign now holds to be true, and what makes it true.

    The support is required rather than optional. A belief with no support is an
    assertion, and an agent accumulating unsupported assertions builds confidence
    without building evidence -- which is precisely the behaviour this project sets
    out to measure.

    Attributes:
        claim: One sentence, stated plainly enough for a non-specialist to disagree with.
        support: The observations that back it, each pointing at a run, a baseline
            number, or a retrieved source.
    """

    claim: str
    support: tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject a belief with nothing behind it.

        Raises:
            ValueError: If no support was given.
        """
        if not self.support:
            raise ValueError(f"a belief needs support: {self.claim!r} has none")


@dataclass(frozen=True, slots=True)
class Citation:
    """One retrieved source, reduced to what a report has to be able to print.

    Deliberately not the retrieval library's own passage object. The state is
    logged, checkpointed and rendered, and a record that carries an embedding
    vector and a store handle around with it is expensive in all three. What a
    reader needs is where the claim came from and enough text to check it.

    Attributes:
        identifier: A citable handle for the source.
        title: Its title, for the reference list.
        snippet: The passage that was actually used.
        shelf: Which knowledge base it came off, empty when it did not come from
            one. Recorded because a hardware measurement and a closed-form result
            are different kinds of claim, and a reference list that flattens them
            invites a reader to weigh them equally.
        reviewed: Whether a person wrote and checked this source. ``False`` marks
            a note fetched from an external index during the run: real, screened,
            and nobody has read it. A report that cites unreviewed material and
            does not say so is overstating what it stands on.
    """

    identifier: str
    title: str
    snippet: str
    shelf: str = ""
    reviewed: bool = True

    def provenance(self) -> str:
        """One phrase describing where this source came from.

        Returns:
            Text for a reference list, naming the knowledge base when there is
            one and flagging anything nobody has reviewed.
        """
        where = f"from {self.shelf}" if self.shelf else "from an external search"
        return where if self.reviewed else f"{where}, unreviewed"


@dataclass(frozen=True, slots=True)
class SearchRound:
    """One search the retrieval loop actually ran, kept for the trace.

    Retrieval here is a loop rather than a lookup: it searches, grades what came
    back, and -- when a round kept nothing -- rewrites the query and searches
    again. Only the surviving passages reach the answer, so without this record
    the two most interesting facts about a thin answer are unrecoverable: *what
    was actually searched for*, and *who decided to search for that*. A reader
    who can see that the first query kept nothing and a rewritten one found the
    note has learned something about the corpus; a reader shown only the final
    citations has been handed a result with its working erased.

    Mirrors :class:`src.rag.retrieve.Attempt` rather than holding one. The
    retrieval package is imported lazily -- it pulls in a vector store -- and this
    module is imported by everything, so a record here that forced that import at
    module load would make the state schema expensive for every caller that never
    searches.

    Attributes:
        query: The text that was sent to the index, verbatim.
        kind: ``"question"`` for the round built from the question as asked,
            ``"rewrite"`` for a round that ran on a reformulation.
        found: How many candidates the index returned.
        kept: How many survived grading. ``0`` with a non-zero ``found`` is the
            interesting case -- the index answered and the grader threw it all
            out -- and it is the one that triggers a rewrite.
        reason: The grader's sentence for this round, which is what a refusal is
            built from when every round comes back empty.
        shelves: Which knowledge bases this round preferred. Empty means no shelf
            was chosen, which is also what every round after the first looks like.
        rewritten_by: Who wrote this round's query -- ``"grader"`` when the model
            that read the failed passages proposed one, ``"model"`` when a
            rewriter was asked for nothing else, ``"heuristic"`` when the
            deterministic floor supplied it. Empty on the first round. A rewrite
            is a decision, and a decision nobody can attribute cannot be reviewed.
    """

    query: str
    kind: Literal["question", "rewrite"]
    found: int
    kept: int
    reason: str = ""
    shelves: tuple[str, ...] = ()
    rewritten_by: Literal["", "grader", "model", "heuristic"] = ""

    def where(self) -> str:
        """Name the knowledge bases this round searched, for the trace.

        Returns:
            The shelf names joined for reading, or ``"the whole library"`` when
            the round was unrestricted.
        """
        return ", ".join(self.shelves) if self.shelves else "the whole library"

    def author(self) -> str:
        """Say who wrote this round's query, in words a reader can act on.

        Returns:
            A short phrase naming the author. The opening round is named as the
            opening round rather than left blank, because "nobody wrote this" and
            "nobody rewrote this" look identical in a table and mean opposite
            things.

            It is deliberately not called "the question". On the explanation and
            code branches the opening query *is* the question; on the feasibility
            branch it is a deliberately broader one the retrieve step builds, and
            a label claiming otherwise would be wrong on the branch where a reader
            is most likely to check.
        """
        if self.kind == "question":
            return "the opening query, written before anything had been graded"
        return {
            "grader": "the grader, having read what the last round returned",
            "model": "a model asked for a better query",
            "heuristic": "the deterministic fallback, with no model involved",
        }.get(self.rewritten_by, "unattributed")


Origin = Literal["model", "deterministic", "none"]
"""Who wrote the follow-up suggestions.

Recorded and shown, for the same reason every other decision in this project names
its author: a feature that quietly degrades to a canned list when the credential is
missing is a feature whose behaviour in a demonstration cannot be compared with its
behaviour in a test.
"""


@dataclass(frozen=True, slots=True)
class Suggestion:
    """One question the agent proposes asking next.

    Attributes:
        question: The question, phrased as somebody would type it. It is put on a
            button and submitted verbatim, so it is treated as untrusted input on
            the way in -- see :func:`src.agent.followups.admissible`. Text this
            application produced is not text this application trusts.
        why: One short phrase on what asking it would add, shown as a hint.
    """

    question: str
    why: str


@dataclass(frozen=True, slots=True)
class Followups:
    """What to ask next, and who proposed it.

    An answer is rarely the end of a question, and the useful next question is
    rarely obvious from an empty chat box. What makes this worth a field on the
    state rather than a flourish in the interface is that the suggestions are
    derived from *what this run established* -- the chain that was solved, the
    configurations that were refused, the shelves that went unread -- so they are
    part of the campaign's output and are recorded with it.

    Attributes:
        suggestions: What to offer, best first. Empty is a normal outcome, not a
            failure: a question the guard blocked gets nothing at all.
        proposed_by: See :data:`Origin`. ``"none"`` means the node never ran.
        rejected: How many candidates were dropped before the rest were shown.
            Counted rather than discarded quietly, because a model whose
            suggestions are routinely rejected is worth noticing.
    """

    suggestions: tuple[Suggestion, ...] = ()
    proposed_by: Origin = "none"
    rejected: int = 0

    @property
    def any_offered(self) -> bool:
        """Whether there is anything to show."""
        return bool(self.suggestions)

    def explain(self) -> str:
        """Say what happened, in one line, for the campaign trail.

        Returns:
            The count and its provenance.
        """
        if self.proposed_by == "none":
            return "no follow-ups were proposed"
        dropped = f", {self.rejected} rejected" if self.rejected else ""
        return f"proposed {len(self.suggestions)} follow-up questions ({self.proposed_by}{dropped})"


@dataclass(frozen=True, slots=True)
class Verdict:
    """The answer, with the condition that would change it.

    The crossover condition is what separates a feasibility study from an opinion.
    "No" on its own expires silently as hardware improves; "no, and it stays no
    until two-qubit error rates fall below :math:`10^{-4}`" can be checked against
    next year's devices by someone who never read the study.

    Attributes:
        call: Go, no, or conditional.
        summary: The verdict in plain words, with no domain vocabulary left
            unexplained.
        crossover_condition: What would have to become true for the answer to
            change. Required even for a "go", where it names what would take the
            answer away again.
        confidence: How much weight to put on this, in words rather than a
            fabricated percentage.
    """

    call: Call
    summary: str
    crossover_condition: str
    confidence: Literal["low", "medium", "high"] = "medium"


@dataclass(frozen=True, slots=True)
class Draft:
    """Code written for one question, on the ``implement`` branch.

    The record lives here with the other state records while the writing lives in
    :mod:`src.agent.drafting`, the same split :class:`Followups` uses. It keeps
    :class:`CampaignState` importable without pulling in a prompt.

    Attributes:
        code: The program. Empty means none was written, which every caller treats
            as "the answer is composed without one" rather than as an error.
        language: The language tag, lowercased, for the interface's highlighting.
        preamble: At most a sentence or two before the code, usually naming the
            method or a package. Dropped by the interface when empty.
        mathematics: The equations the program implements, in markdown, composed by
            :mod:`src.physics.quantum.circuit_algebra` from the circuit the question
            described -- never written by the model. It sits *above* the code because
            a reader who has already scrolled past eighty lines of Python has stopped
            asking what the program is for, and because it is the only part of a code
            answer this project can stand behind: the code is unrun text, the
            equations are composed from integers. Empty when the question described
            no circuit precisely enough to write equations about.
        note: Why there is no code, when there is none. Empty on success. A reader
            told only "no code was written" assumes the feature is broken, when the
            commonest reason is that no model was reachable.
    """

    code: str = ""
    language: str = "python"
    preamble: str = ""
    mathematics: str = ""
    note: str = ""

    @property
    def written(self) -> bool:
        """Whether there is code to show."""
        return bool(self.code.strip())

    def explain(self) -> str:
        """One line for the campaign trail."""
        if self.written:
            lines = len(self.code.splitlines())
            algebra = " under composed equations" if self.mathematics else ""
            return f"wrote {lines} lines of {self.language} for this chain{algebra}"
        return f"no code was written: {self.note}"


@dataclass(frozen=True, slots=True)
class Answer:
    """A prose answer to one question, on the ``explain`` branch.

    Attributes:
        text: The explanation, in markdown. Never empty: with no model and no
            passages it says plainly that the notes do not cover the question, which
            is an answer and is a more useful one than silence.
        rests_on: What a reader would check the answer against, in one sentence,
            when the writer offered one.
        written_by: ``"model"``, ``"notes"``, or ``"refused"``. Recorded rather than
            inferred from whether a key was set, because a failed call and a missing
            key produce the same document and should not look the same in the trace.
            ``"refused"`` is the case where the notes were searched and held nothing:
            the branch computes nothing, so an answer would have stood on a language
            model's memory alone -- see :func:`src.agent.explaining.grounded`.
        cited: How many passages the answer leans on.
    """

    text: str = ""
    rests_on: str = ""
    written_by: str = "notes"
    cited: int = 0

    @property
    def written(self) -> bool:
        """Whether there is prose to show."""
        return bool(self.text.strip())

    @property
    def refused(self) -> bool:
        """Whether this is a refusal rather than an answer.

        Read off the field rather than sniffed out of the prose, so the interface
        cannot draw a refusal in the shape of an answer -- which is the one
        presentation mistake that would undo the honesty the refusal exists for.
        """
        return self.written_by == "refused"

    def explain(self) -> str:
        """One line for the campaign trail."""
        if self.refused:
            return "refused to answer in prose: nothing retrieved and nothing computed"
        source = "a model wrote it" if self.written_by == "model" else "composed from the notes"
        return f"answered in prose from {self.cited} passages ({source})"


def extend(existing: tuple[ItemT, ...], incoming: tuple[ItemT, ...]) -> tuple[ItemT, ...]:
    """Append to an accumulating field instead of overwriting it.

    The graph runs the classical baseline and the quantum campaign at the same
    time, and both can add to the record of what was tried. Without a rule for
    combining them, whichever finished second would silently erase the other's
    entries. This is that rule, and it is attached to the field itself so that no
    node has to remember to merge.

    Args:
        existing: What the state already holds.
        incoming: What a node has just returned.

    Returns:
        The two in order, oldest first.
    """
    return (*existing, *incoming)


class CampaignState(TypedDict):
    """Everything one campaign knows, as the graph passes it between nodes.

    A node reads whatever it needs and returns only the keys it changed. Fields
    marked with :func:`extend` accumulate across nodes; the rest are replaced by
    the last node to write them.

    Attributes:
        request: The question, verbatim.
        intent: What kind of answer the question was read as wanting, and how that
            was decided. Set before any work is done, because it chooses which
            branch does the work.
        model: The Hamiltonian it was turned into. ``None`` until it has been.
        citations: Sources retrieved as background, accumulating.
        searches: Every search the retrieval loop ran, in order, accumulating.
            Kept beside the citations rather than folded into them because a
            round that found nothing leaves no citation and is exactly the round
            worth seeing.
        shots: The measurement budget and what is left of it.
        coherence: How deep a circuit this device can run.
        runs: Every configuration tried, accumulating.
        ruled_out: Every configuration rejected and why, accumulating.
        pending: The configuration priced and waiting to run, if there is one.
            Cleared as soon as the run happens, so a stale plan can never be run twice.
        classical: The number the quantum arm must beat. ``None`` until it is run.
        beliefs: What the campaign has concluded, accumulating.
        verdict: The answer. ``None`` until the campaign reaches one.
        report: The written document, whichever branch composed it. ``None``
            until it is.
        answer: The prose answer, on the ``explain`` branch only. ``None``
            everywhere else, which is how the interface knows not to look for one.
        draft: The code, on the ``implement`` branch only. ``None`` elsewhere.
        followups: What to ask next, proposed from what this run established.
        tool_calls: Every tool the model asked for during this run, and what came
            back, accumulating. Kept beside the citations rather than folded into
            them because a tool call is not a source: a search that returned nothing
            and a derivative that came back exact both belong in the trail, and
            neither is something to cite.
        race: Three methods run against each other on this campaign's own chain,
            with their loss curves. ``None`` unless the question asked for the
            comparison -- racing three methods costs real seconds, so it is done
            when it was asked for and not on every question as a matter of course.
        notes: A running log of what each node did, for the run view and for
            anyone reading a campaign back afterwards, accumulating.
    """

    request: Request
    intent: Reading
    model: FormalModel | None
    citations: Annotated[tuple[Citation, ...], extend]
    searches: Annotated[tuple[SearchRound, ...], extend]
    shots: ShotLedger
    coherence: CoherenceBudget
    runs: Annotated[tuple[RunRecord, ...], extend]
    ruled_out: Annotated[tuple[RuledOut, ...], extend]
    pending: PlannedRun | None
    classical: BaselineResult | None
    beliefs: Annotated[tuple[Belief, ...], extend]
    verdict: Verdict | None
    report: str | None
    answer: Answer | None
    draft: Draft | None
    followups: Followups
    tool_calls: Annotated[tuple[ToolCall, ...], extend]
    race: MethodRace | None
    notes: Annotated[tuple[str, ...], extend]


class RunSummary(TypedDict):
    """One row of the run table, flattened for a log line or a report.

    Attributes:
        label: The configuration's identifier.
        family: Which circuit family was run.
        depth: Number of ansatz layers.
        two_qubit_depth: Two-qubit depth after scheduling.
        energy: The energy reached.
        energy_per_site: That energy per spin.
        shots_spent: What it cost.
        diagnosis: The verdict on the run itself.
    """

    label: str
    family: str
    depth: int
    two_qubit_depth: int
    energy: float
    energy_per_site: float
    shots_spent: int
    diagnosis: DiagnosisRecord


class CampaignSnapshot(TypedDict):
    """The whole campaign as primitives, for a JSON log or the run view.

    Separate from :class:`CampaignState` because they answer different questions.
    The state is what the graph passes around and holds live objects; the snapshot
    is what survives being written to a file and read back tomorrow by something
    that does not import this package.

    Attributes:
        request: The question as asked.
        framing: The voice it was asked in.
        model: The Hamiltonian's label, or ``None``.
        exactly_solvable: Whether the model has a closed-form answer.
        shots_budget: Shots allowed.
        shots_spent: Shots used.
        max_two_qubit_depth: The deepest circuit the coherence budget allows.
        runs: Every configuration tried.
        ruled_out: Configurations rejected, as label-and-reason pairs.
        classical_energy_per_site: The classical baseline, or ``None``.
        beliefs: The claims reached.
        verdict: Go, no, or conditional -- or ``None`` if the campaign stopped early.
        crossover_condition: What would change the verdict.
        citations: Identifiers of the sources used.
        searches: Every query the retrieval loop ran, in order, each rendered as
            ``kind: query`` so that a rewritten round is legible in a log line
            without the reader having to hold a second table in their head.
        race: The method comparison and its loss curves, or ``None`` when the
            question did not ask for one. Included in full rather than summarised:
            a race whose curves were dropped is a comparison nobody can re-plot,
            and the curve is the deliverable here rather than a diagnostic.
        notes: What each node did, in order.
    """

    request: str
    framing: Framing
    model: str | None
    exactly_solvable: bool | None
    shots_budget: int
    shots_spent: int
    max_two_qubit_depth: int
    runs: list[RunSummary]
    ruled_out: list[dict[str, str]]
    classical_energy_per_site: float | None
    beliefs: list[str]
    verdict: Call | None
    crossover_condition: str | None
    citations: list[str]
    searches: list[str]
    race: dict[str, Any] | None
    notes: list[str]


def new_campaign(
    request: Request,
    shot_budget: int,
    coherence: CoherenceBudget | None = None,
) -> CampaignState:
    """Open a campaign with nothing known and the budget untouched.

    Every optional field starts at ``None`` and every accumulating field at an
    empty tuple, so that "not yet established" and "established to be nothing" stay
    distinguishable all the way to the report. A campaign that ran no quantum
    configurations and a campaign whose configurations were all rejected are
    different stories, and collapsing both to an empty list would tell neither.

    Args:
        request: The question, as asked.
        shot_budget: Total measurements the campaign may spend.
        coherence: The device's depth limit. Defaults to the nominal figures.

    Returns:
        The opening state.
    """
    return CampaignState(
        request=request,
        intent=Reading(),
        model=None,
        citations=(),
        searches=(),
        shots=ShotLedger(budget=shot_budget),
        coherence=CoherenceBudget() if coherence is None else coherence,
        runs=(),
        ruled_out=(),
        pending=None,
        classical=None,
        beliefs=(),
        verdict=None,
        report=None,
        answer=None,
        draft=None,
        followups=Followups(),
        tool_calls=(),
        race=None,
        notes=(),
    )


def best_run(state: CampaignState) -> RunRecord | None:
    """The quantum run that reached the lowest energy, if any did.

    Runs whose diagnosis says the energy is impossible are excluded. Letting one
    win on energy would be the worst possible outcome of the bug it reports: the
    lower an erroneous energy is, the more likely it is to be selected and quoted
    as the headline result.

    Args:
        state: The campaign so far.

    Returns:
        The best trustworthy run, or ``None`` if there is not one.
    """
    trustworthy = [
        run for run in state["runs"] if run.diagnosis.signal != "below_variational_bound"
    ]
    if not trustworthy:
        return None
    return min(trustworthy, key=lambda run: run.energy)


def snapshot(state: CampaignState) -> CampaignSnapshot:
    """Flatten the campaign into primitives that survive a JSON round trip.

    Args:
        state: The campaign to render.

    Returns:
        A mapping of strings, numbers and lists only -- nothing that needs this
        package imported in order to be read.
    """
    model = state["model"]
    verdict = state["verdict"]
    classical = state["classical"]
    return CampaignSnapshot(
        request=state["request"].text,
        framing=state["request"].framing,
        model=None if model is None else model.label(),
        exactly_solvable=None if model is None else model.is_exactly_solvable,
        shots_budget=state["shots"].budget,
        shots_spent=state["shots"].spent,
        max_two_qubit_depth=state["coherence"].max_two_qubit_depth,
        runs=[_summarise(run) for run in state["runs"]],
        ruled_out=[{"label": item.label, "reason": item.reason} for item in state["ruled_out"]],
        classical_energy_per_site=None if classical is None else classical.energy_per_site,
        beliefs=[belief.claim for belief in state["beliefs"]],
        verdict=None if verdict is None else verdict.call,
        crossover_condition=None if verdict is None else verdict.crossover_condition,
        citations=[citation.identifier for citation in state["citations"]],
        searches=[f"{round_.kind}: {round_.query}" for round_ in state["searches"]],
        race=None if state["race"] is None else state["race"].describe(),
        notes=list(state["notes"]),
    )


def _summarise(run: RunRecord) -> RunSummary:
    """Flatten one run into its table row.

    Args:
        run: The run to render.

    Returns:
        The row, with energies rounded to a precision a report can print without
        implying more accuracy than a variational method has.
    """
    return RunSummary(
        label=run.label,
        family=run.family,
        depth=run.depth,
        two_qubit_depth=run.two_qubit_depth,
        energy=round(run.energy, 9),
        energy_per_site=round(run.energy_per_site, 9),
        shots_spent=run.shots_spent,
        diagnosis=run.diagnosis.describe(),
    )
