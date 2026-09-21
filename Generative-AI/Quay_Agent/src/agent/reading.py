"""Reading a chain out of a sentence, without asking a language model.

The agent's first job is to turn "a ring of twelve spins in a strong sideways
field" into numbers. A language model does that well, and if it were the only
thing that did, a campaign with no credential would fall back on a fixed default
and answer a question nobody asked -- so the free offline demonstration would be
about the wrong problem, and the one path with no model in it would have no
reading step to test.

This reads what it can by pattern, and is used twice: as the floor, formalising
the question when no model is available, and as a check on the model, since
anything this finds and the model missed is worth surfacing. The two disagreeing
about the chain length is the kind of quiet error a feasibility study carries all
the way to its verdict.

It never guesses. A sentence with no number in it yields no number, and the
caller records that a default was assumed rather than receiving a confident
invention: a reader that always returns something is a reader whose output cannot
be distinguished from a shrug.

It does not read intent, tone or pressure. "Our vendor swears this will win" and
"I doubt this will work" describe the same chain, and anything here that
responded to the difference would be laundering the framing the project exists to
measure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Literal, cast

from src.physics.lattice import MIN_SIDE, Geometry

WORD_NUMBERS: dict[str, int] = {
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "sixteen": 16,
    "twenty": 20,
}
"""Spelled-out counts people actually write. Beyond twenty they use digits."""

MIN_SITES = 2
MAX_SITES = 64
"""Chain lengths this will believe.

Below two there is no bond. Above sixty-four the number is far likelier to be a
temperature, a budget or a year that happened to sit next to the word "spins" than a
chain somebody wants simulated, and inventing a sixty-five-thousand-site chain from
a stray figure is worse than reading nothing.
"""

_SITE_UNITS = r"(?:spins?|sites?|qubits?|magnets?|elements?|atoms?|particles?)"

_JOINING_WORDS = (
    "of with on in and or for at to per from by that which the a an "
    "hundred thousand million billion"
).split()
_MODIFIER = r"(?:(?!(?:" + "|".join(_JOINING_WORDS) + r")[\s-])(?![\d])[A-Za-z]+[\s-]+){0,2}"
"""Adjectives a count may be separated from its unit by, as a regex fragment.

*Six interacting elements* and *ten coupled spins* name a length as plainly as *six
spins* does, and requiring the two to be adjacent read no length out of either. Two
words rather than any number of them, and the joining words are excluded, because the
gap has to stay too narrow to jump between two separate quantities: without that
exclusion *3 devices with 8 qubits* reads three.
"""
# Six digits, not three. At three, "1000 spins" matched nothing at all and "a chain
# of 1000 spins" matched the *first three digits* -- so a question about a thousand
# magnets was either answered about an invented six-element default under the printed
# assumption "the request named no chain", or refused with "it names a chain of 100"
# about a sentence that said 1000. Both are false statements about the reader's own
# words, and `Reading.too_long` exists specifically to prevent the first one. The
# ceiling is a length nothing here can price either way, so reading it correctly
# costs nothing and lets the refusal name the number that was actually asked for.
_COUNT = r"(\d{1,6}|" + "|".join(WORD_NUMBERS) + r")"

MIN_DEPTH = 1
MAX_DEPTH = 32
"""Layer counts this will believe.

Zero layers is a legal circuit -- it is the bare initial state -- but nobody writes a
sentence asking for it, so a zero here is far likelier to be a typo than a request.
Above thirty-two the number is much more often a shot count, a year or a temperature
that happened to sit next to the word "layers".
"""

_LAYER_UNITS = r"(?:layers?|rounds?|repetitions?|reps?)"

SITE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "10-spin", "12 spins", "twenty qubits", "six interacting elements"
    re.compile(_COUNT + r"[\s-]+" + _MODIFIER + _SITE_UNITS, re.IGNORECASE),
    # "chain of 12", "ring of twenty", "a row of 8"
    re.compile(r"(?:chain|ring|row|line|loop|lattice)\s+of\s+" + _COUNT, re.IGNORECASE),
    # "L = 8", "N=12"
    re.compile(r"\b[LN]\s*=\s*(\d{1,3})\b"),
)
"""How a chain length gets written, in the order the patterns are tried.

Ordered by how unambiguous each one is. A number attached to a unit is a length
almost every time; a bare ``L = 8`` is certain but rare in ordinary prose.
"""

DEPTH_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "depth 3", "depth of three", "depth-3"
    re.compile(r"depth[\s-]+(?:of[\s-]+)?" + _COUNT, re.IGNORECASE),
    # "3 layers", "three-layer", "two rounds"
    re.compile(_COUNT + r"[\s-]+" + _LAYER_UNITS, re.IGNORECASE),
    # "p = 3", which is what the QAOA literature calls it
    re.compile(r"\bp\s*=\s*(\d{1,2})\b"),
)
"""How a circuit depth gets written, in the order the patterns are tried."""

CURVE_WORDS = (
    "plot",
    "curve",
    "chart",
    "converge",
    "convergence",
    "loss",
    "epoch",
    "iteration",
    "learning",
    "training run",
)
"""Words that mean the asker wants to watch a descent, not just be told where it ended.

*Plot the energy against epoch*, *which converges fastest*, *how many iterations does
this take* -- all three are asking for the same picture, and none of them is answered
by a single final number. A question with none of these words is asking about the
answer rather than about the route to it, and racing three methods for it would spend
several seconds of compute on something nobody asked to see.
"""

METHOD_WORDS: dict[str, tuple[str, ...]] = {
    "VQE": ("vqe", "variational quantum eigensolver", "variational eigensolver"),
    "QAOA": ("qaoa", "approximate optimisation", "approximate optimization"),
    "VarQITE": (
        "varqite",
        "imaginary time",
        "imaginary-time",
        "qite",
        "cooling",
    ),
}
"""How each of the three raceable methods gets written, in the words people use.

Spelled out rather than taken from :data:`src.physics.quantum.method_race.METHOD_NAMES`
because those are the labels a *legend* wants and these are the strings a *sentence*
contains. "Imaginary time" and "VarQITE" name one method; matching only the acronym
would miss the question this project's own starter asks.
"""


def methods_named_in(text: str) -> tuple[str, ...]:
    """Find which of the three raceable methods a question mentions, in legend order.

    Args:
        text: The question as it was asked.

    Returns:
        The method names, deduplicated and ordered as
        :data:`src.physics.quantum.method_race.METHOD_NAMES` orders them.

    Examples:
        >>> methods_named_in("VQE, QAOA or imaginary time for a 10-spin chain?")
        ('VQE', 'QAOA', 'VarQITE')
        >>> methods_named_in("how deep should the circuit be?")
        ()
    """
    lowered = text.lower()
    return tuple(
        name for name, spellings in METHOD_WORDS.items() if any(w in lowered for w in spellings)
    )


TEACHING_PHRASES: tuple[str, ...] = (
    "teach me",
    "teach us",
    "could you teach",
    "can you teach",
    "explain to me",
    "walk me through",
    "show me how",
    "help me understand",
    "i want to learn",
    "i'd like to learn",
    "introduce me",
    "an introduction to",
    "give me an overview",
    "tell me about",
    "what are the",
    "how do .* work",
)
"""Phrases that mean the asker wants to be taught rather than told a result.

A *phrase* list rather than a word list, and deliberately so: the single words that
mark a tutorial request -- *learn*, *overview*, *about* -- are far too common to carry
the meaning on their own, and "learning" is already in :data:`CURVE_WORDS` where it
means an optimiser's learning curve. Only "teach me about the loss" is a request for a
lesson; "the loss went up" is not.

The last entry is a regular expression fragment rather than a literal, because *how do
VQE and QAOA work* puts the subject in the middle of the phrase. Matching is done with
:func:`re.search` for that reason, which costs nothing here -- the list is short and the
input is one sentence.
"""


def asks_to_be_taught(text: str) -> bool:
    """Say whether the question asks for a lesson rather than for a number.

    This is a **different question from the intent**, and it has to be asked
    separately. The intent reader already routes this to ``explain``; what it cannot
    express is that *naming three methods should not start a race*. That rule --
    :func:`asks_to_compare_methods`, two methods named means compare them -- is right
    for *VQE, QAOA or imaginary time for a 10-spin chain?* and wrong for *could you
    teach me about VQE, QAOA and VarQITE*, and the two sentences are indistinguishable
    by method count. Both name three.

    What went wrong without it is worth recording, because it is the failure mode this
    whole project is about. The teaching question ran the race, and the race table --
    three methods, three identical energies to six decimal places on a six-spin chain
    nobody asked about -- *became the answer*. The reader who asked what QAOA is was
    handed a number and no explanation, and the number was the same one every other
    button on the page returns. A measurement presented where an explanation was asked
    for is not a bonus; it displaces the answer.

    It also decides how the explanation is *written*. A lesson wants headed sections,
    each algorithm's purpose stated before its mathematics, and its equations set on
    their own lines -- see :data:`src.agent.explaining.TEACHING_SHAPE`. A question with
    a measurement attached wants the measurement and two lines saying what it means.
    The same prompt cannot be tuned for both, so it is told which it is writing.

    Args:
        text: The question as it was asked.

    Returns:
        Whether any teaching phrase appears in it.

    Examples:
        >>> asks_to_be_taught("Could you teach me about VQE, QAOA and VarQITE?")
        True
        >>> asks_to_be_taught("Of VQE, QAOA and VarQITE, which converges fastest?")
        False
        >>> asks_to_be_taught("How do VQE and QAOA work on this chain?")
        True
    """
    lowered = text.lower()
    return any(re.search(phrase, lowered) for phrase in TEACHING_PHRASES)


def asks_to_compare_methods(text: str) -> bool:
    """Say whether the question puts two or more methods against each other.

    The companion to :func:`asks_for_a_curve`, and it exists because that function was
    not enough. *VQE, QAOA or imaginary time for a 10-spin critical chain?* contains no
    plotting word at all, so it raced nothing -- and the campaign answered the question
    it could answer by rule, "is a quantum computer worth it here?", without ever saying
    which of the three named methods it would be. A reader who lists three options and
    is handed a verdict that mentions none of them is being answered past.

    Two is the bar rather than one. A question naming a single method is asking *about*
    that method, and racing it against two it never mentioned would be answering a
    question nobody asked -- the same fault in the other direction.

    Args:
        text: The question as it was asked.

    A lesson is not a comparison, however many methods it names. *Could you teach me
    about VQE, QAOA and VarQITE* names three and asks for none of them to be run;
    raced, all three run and the table displaces the explanation entirely. The screen
    is :func:`asks_to_be_taught`, checked here rather than at the call site so that
    the two places deciding whether to race cannot disagree.

    An explicit request for the picture still wins: *teach me how VQE and QAOA
    converge, and plot it* is caught by :func:`asks_for_a_curve`, which is checked
    independently of this and is not overridden by the teaching screen.

    Args:
        text: The question as it was asked.

    Returns:
        Whether at least two distinct methods are named and a lesson is not what was
        asked for.

    Examples:
        >>> asks_to_compare_methods("VQE, QAOA or imaginary time for a 10-spin chain?")
        True
        >>> asks_to_compare_methods("Does VQE pay off on molecules?")
        False
        >>> asks_to_compare_methods("Could you teach me about VQE, QAOA and VarQITE?")
        False
    """
    if asks_to_be_taught(text):
        return False
    return len(methods_named_in(text)) >= 2


MAX_EPOCH = 100_000
"""Largest epoch number this will believe a reader wants to see.

A bound rather than a formality: ``[0, 2024]`` in a sentence about a paper is a year,
and an axis silently stretched to two thousand epochs would show three flat lines.
"""

RANGE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "[0, 20]" and "(0,20)" -- an interval written as one
    re.compile(r"[\[(]\s*(\d{1,6})\s*[,;]\s*(\d{1,6})\s*[\])]"),
    # "between 0 and 20"
    re.compile(r"\bbetween\s+(\d{1,6})\s+and\s+(\d{1,6})\b", re.IGNORECASE),
    # "from 0 to 20", "in the range 0 to 20", "range of 0-20"
    re.compile(
        r"\b(?:from|range\s+of|range)\s+(\d{1,6})\s*(?:to|-|\u2013)\s*(\d{1,6})\b", re.IGNORECASE
    ),
)
"""How somebody asks for part of an axis, in the order the patterns are tried.

Ordered by how unambiguous each is. Brackets around two numbers are an interval and
nothing else; a bare ``0-20`` is a range only when a word like *range* or *from* says
so, because otherwise it is as likely to be a date or a subtraction.
"""

TRIANGULAR_WORDS: tuple[str, ...] = ("triangular", "triangle lattice", "frustrated lattice")
"""Words that name the non-bipartite lattice.

Checked before the square words, because *triangular lattice* contains *lattice* and
a question that says both -- "a triangular lattice, not a square one" -- means the
first. Frustration is included as a name for it: on this project's one model a
frustrated geometry **is** the triangular one, so a reader who asks for frustration
by name has named a shape.
"""

SQUARE_WORDS: tuple[str, ...] = (
    "square lattice",
    "square grid",
    "2d lattice",
    "2-d lattice",
    "two-dimensional lattice",
    "two dimensional lattice",
    "grid of spins",
    "plaquette",
)
"""Words that name the two-dimensional bipartite lattice.

Whole phrases rather than the bare word *square*, and the reason is a defect this
would otherwise reintroduce: *lattice* on its own does not mean two dimensions -- a
chain is a one-dimensional lattice and the notes call it one -- so matching it would
turn every question that used the word into a question about a grid.
"""

_DIMENSIONS = re.compile(
    r"\b(\d{1,3})\s*(?:x|\u00d7|by)\s*(\d{1,3})\b",
    re.IGNORECASE,
)
"""``4x4``, ``4 x 4``, ``4 by 4`` and ``4\u00d74``, which is how a size is written."""

_SIZED_GRID = re.compile(
    r"\b\d{1,3}\s*(?:x|\u00d7|by)\s*\d{1,3}[\s-]+(?:grid|lattice|array|arrangement|block)\b",
    re.IGNORECASE,
)
"""A size written against a shape noun, as in *a 3 by 3 grid*.

Two sides and the word *grid* name a square lattice without the word *square*
appearing anywhere, and :data:`SQUARE_WORDS` requires whole phrases -- so *9 magnets
in a 3 by 3 grid* was read as a chain of nine, which has a closed-form energy the
lattice does not. The shape noun is required: *3 by 3 metres* is not a lattice.
"""

PERIODIC_WORDS = ("ring", "periodic", "closed loop", "closed chain", "wraps around", "cyclic")
OPEN_WORDS = ("open", "line", "row", "segment", "two ends", "not periodic")
"""Words that settle the boundary. Checked in that order -- see :func:`read_boundary`."""

CRITICAL_WORDS = ("critical", "criticality", "critical point", "phase transition")
r"""Words that fix :math:`h = J` without either being written down.

"At criticality" is a complete specification of the field to anybody in the field,
and a reader that missed it would ask a question about the easy regime while the
asker was asking about the hard one.
"""


@dataclass(frozen=True, slots=True)
class Reading:
    """What could be read out of a sentence, and what could not.

    Every field is optional, and that is the design. A caller can tell the difference
    between "the question said ten spins" and "the question said nothing about
    length", which is what lets it record an assumption honestly instead of
    presenting a default as a finding.

    Attributes:
        n_sites: Number of spins, if the sentence named one -- for a lattice this is
            the product of its sides.
        geometry: The shape the spins sit on, if the sentence named one. ``None``
            means the sentence did not say, which is *not* the same as saying
            "chain": the difference is what lets the formaliser record "a chain was
            assumed" as an assumption a reader can reject.
        rows: Rows of a two-dimensional lattice, if the sentence gave a size like
            ``4x4``. ``None`` for a chain, and also for a lattice named without a
            size -- "a square lattice of 16 spins" says how many but not how they
            are arranged, and 16 could be 4x4 or 2x8.
        boundary: Whether it is a ring or a line, if the sentence said.
        coupling: The Ising coupling :math:`J`, if given.
        field: The transverse field :math:`h`, if given.
        longitudinal: The field :math:`g` along the coupling axis, if given.
        critical: Whether the sentence asked for the critical point, which fixes
            :math:`h = J` whatever else was said about either.
        depth: How many circuit layers the sentence asked for, if it named a number.
            Not part of the Hamiltonian -- it is a choice about the *program* run
            against the chain -- so it is deliberately left out of
            :attr:`found_anything` and :meth:`restate`, both of which answer "did
            this sentence name a physical system?". A question that says only
            "depth 3" has named a circuit, not a chain.
        epoch_range: The stretch of a convergence curve the sentence asked to see,
            as ``(first, last)``. Like :attr:`depth` it describes the *picture* and
            not the chain, so it is left out of :attr:`found_anything` and
            :meth:`restate` for the same reason. See :func:`read_epoch_range` for
            why it is read as epochs rather than as energies.
        strengths_from_ratio: Whether :attr:`coupling` and :attr:`field` came from a
            stated *ratio* between them rather than from two numbers in the
            sentence. When they did, the pair is on the literature's scale
            (:math:`J = 1`) and only the ratio was ever asked about -- so a reader
            downstream must keep the pair together and not rescale one of them.
        too_long: A chain length the sentence *did* name, which is past what
            anything here can price. ``None`` when the sentence named a usable
            length or named none at all.

            This field exists because those last two cases were indistinguishable
            and must not be. *Is a quantum computer worth using for a chain of 200
            magnets?* set ``n_sites=None``, exactly as a question naming no chain
            does, so the campaign invented a six-element default and printed the
            assumption "the request named no chain" -- a false statement about the
            reader's own sentence, on the reader's own screen, under a verdict about
            a chain thirty times shorter than the one they asked about. Recording the
            number lets the answer say what was asked for and why it cannot be
            served.
    """

    n_sites: int | None = None
    geometry: Geometry | None = None
    rows: int | None = None
    boundary: Literal["open", "periodic"] | None = None
    coupling: float | None = None
    field: float | None = None
    longitudinal: float | None = None
    critical: bool = False
    depth: int | None = None
    epoch_range: tuple[int, int] | None = None
    strengths_from_ratio: bool = False
    too_long: int | None = None

    @property
    def found_anything(self) -> bool:
        """Whether the sentence yielded a single usable fact.

        A length that is out of range does **not** count as a usable fact -- nothing
        downstream can build a chain from it -- but it is still a fact about the
        sentence, which is why :attr:`too_long` is carried separately and read by
        :attr:`named_too_long`.
        """
        return (
            any(
                value is not None
                for value in (
                    self.n_sites,
                    self.geometry,
                    self.boundary,
                    self.coupling,
                    self.field,
                )
            )
            or self.critical
        )

    @property
    def names_the_whole_chain(self) -> bool:
        """Whether the sentence stated every term of the Hamiltonian itself.

        This was written as a latency control, and it would be the largest one in
        the project. Reading the problem out of a sentence is one language-model call
        and the second most expensive step in a question: 5.96 s of a 32.8 s answer,
        on a reasoning model, every time. That is money well spent on *"each magnet
        pulling on its neighbours about as strongly as a sideways influence pushes on
        all of them"*, where there is no number to match. It buys nothing on *"a ring
        of 12 spins with J = 1 and h = 1"*, where the regexes in this module have
        already read the same four values the model would return and
        :func:`~src.agent.graph._clamp_model` prefers the sentence in any case.

        Nothing reads it, and that is a result rather than an oversight. Of the
        fifteen starter questions the interface offers, none names all four terms.
        The bar is all four plus the boundary -- a sentence naming three and leaving
        the field open is where the model earns its six seconds -- and real questions
        do not clear it. Wiring it in would add a branch to the hot path for no
        saving, so it stays unwired and says so.

        The bar includes the coupling, so a question naming the critical point but
        no :math:`J` does **not** clear it. That is arguable: :attr:`critical` fixes
        the *ratio* :math:`h/J`, and :math:`J` alone is only an energy scale, so a
        model would contribute a convention rather than a fact. The stricter reading
        is kept because this property's one purpose is deciding whether a call is
        skippable, and "the sentence named it" is a claim that should not quietly
        include "we assumed the usual value".

        Returns:
            True when nothing a model could add is missing.

        Examples:
            >>> read("a ring of 12 spins with J = 1 and h = 1").names_the_whole_chain
            True

            The critical point fixes ``h/J`` but names no ``J``, so this is False --
            see the paragraph above, which is the one debatable case:

            >>> read("6 magnets in a line at the critical point").names_the_whole_chain
            False
            >>> read("twelve magnets pulling hard on each other").names_the_whole_chain
            False
        """
        if self.n_sites is None or self.boundary is None or self.coupling is None:
            return False
        return self.critical or self.field is not None

    @property
    def named_too_long(self) -> bool:
        """Whether the sentence named a chain longer than anything here can price.

        Read by the graph in place of ``n_sites is None``, which conflated a question
        that named nothing with one that named too much.
        """
        return self.too_long is not None

    def with_sites(self, n_sites: int) -> Reading:
        """This reading with its site count replaced.

        Used where a language model's proposal has already been clamped to what can
        be simulated: the shape still comes from the sentence, but the size that the
        arrangement is derived from must be the clamped one, or a request for a 6x6
        lattice would be shortened to sixteen sites and then arranged as a 6x6.

        Args:
            n_sites: The count to use.

        Returns:
            A copy. Frozen, so a copy rather than a mutation.
        """
        return replace(self, n_sites=n_sites)

    def without_rows(self) -> Reading:
        """This reading with its row count dropped, so an arrangement is re-derived.

        The companion to :meth:`with_sites` and the reason that method's warning is
        now enforced rather than only written down. A sentence can name both a shape
        and a size that agree -- ``6x6`` is thirty-six spins in six rows -- and the
        size can then be clamped to what a state vector can carry while the row
        count, which came from the sentence, stays as it was. Six rows over sixteen
        spins is not a rectangle, and every consumer downstream is entitled to
        assume it is one.

        Dropping the rows rather than the sites is deliberate: the site count has
        already been lowered for a reason that is reported to the reader, and
        adjusting it a second time to fit an arrangement would assess a size nobody
        chose or explained.

        Returns:
            A copy with ``rows`` unset. Frozen, so a copy rather than a mutation.
        """
        return replace(self, rows=None)

    def under(self, earlier: Reading) -> Reading:
        """Fill this reading's gaps from an earlier one.

        What makes a follow-up work without a language model. "And with periodic
        boundary conditions?" names a boundary and nothing else; laid over the
        reading of the question before it, it becomes the same chain with the
        boundary changed, which is what the asker meant and what any person would
        have understood.

        This one always wins where it says anything. An earlier chain length is a
        default to fall back on, never an override -- otherwise asking about six
        spins and then about twenty would keep answering about six.

        Args:
            earlier: The reading of what was asked before.

        Returns:
            This reading with its empty fields taken from ``earlier``.
        """
        return Reading(
            n_sites=self.n_sites if self.n_sites is not None else earlier.n_sites,
            # Inherited together, or a follow-up would land on a shape from one
            # question and a size from another. "And with a tilt of 0.4?" after a
            # question about a 4x4 square must stay a 4x4 square; keeping the
            # geometry but dropping the rows would silently re-derive the
            # arrangement and could answer about a 2x8 strip instead.
            geometry=self.geometry if self.geometry is not None else earlier.geometry,
            rows=(
                self.rows
                if self.geometry is not None
                else (self.rows if self.rows is not None else earlier.rows)
            ),
            boundary=self.boundary if self.boundary is not None else earlier.boundary,
            coupling=self.coupling if self.coupling is not None else earlier.coupling,
            field=self.field if self.field is not None else earlier.field,
            # True only when the pair being carried forward is the one the ratio
            # set. A follow-up that names a number of its own has anchored the
            # scale, and the flag must not outlive that.
            strengths_from_ratio=(
                self.strengths_from_ratio
                if (self.coupling is not None or self.field is not None)
                else earlier.strengths_from_ratio
            ),
            longitudinal=(
                self.longitudinal if self.longitudinal is not None else earlier.longitudinal
            ),
            critical=self.critical or earlier.critical,
            depth=self.depth if self.depth is not None else earlier.depth,
            epoch_range=(self.epoch_range if self.epoch_range is not None else earlier.epoch_range),
            # Not inherited. An oversized length is a fact about *this* sentence, and
            # carrying it forward would make a follow-up about a reasonable chain
            # keep refusing because the question before it was too big.
            too_long=self.too_long,
        )

    def restate(self) -> str:
        """Write what was read as one plain sentence.

        The restatement an offline campaign uses. A model does this better when there
        is one; with none, this is what stands between a reader and no view at all of
        what the agent understood -- and seeing that *before* the conclusion is most of
        what makes a conclusion checkable.

        It says only what the sentence said. Nothing here fills a gap with a plausible
        value, so a restatement that omits the field is telling the reader the question
        omitted it.

        Returns:
            The sentence, or the empty string when the question named nothing.
        """
        if not self.found_anything:
            return ""
        parts: list[str] = []
        length = f"{self.n_sites} spins" if self.n_sites is not None else "a chain"
        shape = {"periodic": "joined into a ring", "open": "in a line"}.get(self.boundary or "", "")
        parts.append(" ".join(filter(None, (length, shape))))
        if self.coupling is not None:
            parts.append(f"coupling J = {self.coupling:g}")
        if self.field is not None:
            reason = " (the critical point)" if self.critical else ""
            parts.append(f"transverse field h = {self.field:g}{reason}")
        if self.longitudinal:
            parts.append(f"longitudinal field g = {self.longitudinal:g}")
        return f"The ground state of {', '.join(parts)}."

    def describe(self) -> tuple[str, ...]:
        """List what was read, in plain words, for a campaign's notes.

        Returns:
            One phrase per fact found, in reading order. Empty when nothing was.
        """
        found: list[str] = []
        if self.n_sites is not None:
            found.append(f"{self.n_sites} spins")
        if self.too_long is not None:
            found.append(
                f"{self.too_long} spins, which is longer than the {MAX_SITES} this can price"
            )
        if self.boundary is not None:
            found.append("a ring" if self.boundary == "periodic" else "a line with two ends")
        if self.critical:
            found.append("at the critical point, so the field equals the coupling")
        if self.coupling is not None:
            found.append(f"coupling J = {self.coupling:g}")
        if self.field is not None:
            found.append(f"transverse field h = {self.field:g}")
        if self.longitudinal is not None:
            found.append(f"longitudinal field g = {self.longitudinal:g}")
        if self.depth is not None:
            found.append(f"a circuit {self.depth} layers deep")
        if self.epoch_range is not None:
            found.append(
                f"the curve between epochs {self.epoch_range[0]} and {self.epoch_range[1]}"
            )
        return tuple(found)


def read_count(text: str) -> int | None:
    """Find the chain length, if the sentence names one.

    Args:
        text: The question as it was asked.

    Returns:
        The number of spins, or ``None`` when the sentence names none or names
        something outside what this will believe.
    """
    return _first_count(text, SITE_PATTERNS, MIN_SITES, MAX_SITES)


def read_depth(text: str) -> int | None:
    """Find how many circuit layers the sentence asks for, if it names a number.

    Read separately from the chain because it describes the *program*, not the
    physical system: two questions about the same twelve magnets can ask for
    circuits of different depths. What it buys is a picture -- with a length and a
    depth the interface can draw the exact circuit the question described, from this
    project's own specification, next to whatever prose or code answered it.

    Args:
        text: The question as it was asked.

    Returns:
        The number of layers, or ``None`` when the sentence names none or names
        something outside what this will believe.
    """
    return _first_count(text, DEPTH_PATTERNS, MIN_DEPTH, MAX_DEPTH)


def read_epoch_range(text: str) -> tuple[int, int] | None:
    """Find the stretch of a convergence curve the sentence asks to see.

    *Can we plot the loss curve in the range [0, 20]?* is a request about the picture
    already on screen, and answering it means redrawing that picture rather than
    reasoning about a chain. Without this the sentence names no physics at all and is
    refused, which is the worst possible reply to a question about the figure directly
    above it.

    The pair is read as epochs rather than energies, and the figure's caption says so
    rather than leaving it to be inferred. A pair of numbers in a sentence about a
    curve could be either axis; two things settle it here. The energies this project
    plots are negative, so a range written with non-negative bounds cannot be one of
    them; and the horizontal axis is the one a reader can usefully crop, since cropping
    the vertical axis of a descent hides the descent. A negative bound is refused
    outright rather than guessed at, and whatever is drawn is named in the figure's
    caption -- see :func:`src.ui.panels.convergence_asked_for`.

    Args:
        text: The question as it was asked.

    Returns:
        ``(first, last)`` epochs, or ``None`` when the sentence names no range, names
        one backwards, or names one outside what this will believe.

    Examples:
        >>> read_epoch_range("can we plot the loss curve in the range of [0,20]")
        (0, 20)
        >>> read_epoch_range("show it between 5 and 40 steps")
        (5, 40)
        >>> read_epoch_range("plot the loss curve") is None
        True
    """
    for pattern in RANGE_PATTERNS:
        found = pattern.search(text)
        if found is None:
            continue
        first, last = int(found.group(1)), int(found.group(2))
        if first < last <= MAX_EPOCH:
            return first, last
    return None


def _first_count(
    text: str,
    patterns: tuple[re.Pattern[str], ...],
    lowest: int,
    highest: int,
) -> int | None:
    """Take the first count a list of patterns finds, and believe it only in range.

    Shared by :func:`read_count` and :func:`read_depth`, which differ in nothing but
    their patterns and their bounds. A second copy of this loop is a second place
    for "a match outside the bounds means keep looking, not give up" to be got
    wrong.

    Args:
        text: The question as it was asked.
        patterns: The patterns to try, in order, each with the count as group one.
        lowest: Smallest value worth believing.
        highest: Largest value worth believing.

    Returns:
        The count, or ``None`` when nothing matched inside the bounds.
    """
    for pattern in patterns:
        found = pattern.search(text)
        if found is None:
            continue
        token = found.group(1).lower()
        value = WORD_NUMBERS.get(token, int(token) if token.isdigit() else 0)
        if lowest <= value <= highest:
            return value
    return None


def asks_for_a_curve(text: str) -> bool:
    """Say whether the question is about how a method gets there, not only where it ends.

    Used twice, and it has to be the same answer in both places. The graph asks it to
    decide whether to spend seconds racing three methods against each other, and the
    interface asks it to decide whether to draw the result -- so a question that ran the
    race and got no picture, or drew a picture from a race that never ran, would be this
    function disagreeing with itself.

    Args:
        text: The question as it was asked.

    Returns:
        Whether any of :data:`CURVE_WORDS` appears in it.

    Examples:
        >>> asks_for_a_curve("plot the energy against optimisation epoch")
        True
        >>> asks_for_a_curve("how deep should the circuit be?")
        False
    """
    lowered = text.lower()
    return any(word in lowered for word in CURVE_WORDS)


FIELD_SWEEP_SUBJECTS: tuple[str, ...] = (
    "spectrum",
    "spectra",
    "energy level",
    "energy levels",
    "low lying",
    "low-lying",
    "excitation",
    "excitations",
    "gap",
    "magnetisation",
    "magnetization",
    "susceptibility",
    "order parameter",
    "ground state energy",
    "ground-state energy",
    "energy density",
    "derivative",
    "derivatives",
    "curvature",
    "free fermion",
    "free-fermion",
    "jordan-wigner",
    "jordan wigner",
    "phase transition",
    "phase diagram",
    "critical point",
    "criticality",
)
"""Quantities that live in a *curve against the field*, not in a single run.

Deliberately excludes the bare word "energy". *Plot the energy against epoch* is a
question about an optimiser's descent and belongs to the method race; *plot the
ground-state energy against h/J* is a question about the model and belongs to the
exact solver. One word separates them and it is not "energy".
"""

FIELD_SWEEP_AXES: tuple[str, ...] = (
    "as a function of",
    "as the field",
    "against the field",
    "against h",
    "versus the field",
    "versus h",
    "vs the field",
    "vs h",
    "vs. h",
    "h/j",
    "h / j",
    "with the field",
    "with the transverse field",
    "with the external field",
    "with the magnetic field",
    "as the transverse field",
    "as the external field",
    "as the magnetic field",
    "vary with",
    "varies with",
)
"""Ways of naming the field as the horizontal axis.

Enough on their own to make a question a sweep when a swept quantity is also named,
because *how does the magnetisation behave as a function of h/J* asks for a curve
without ever using the word "plot".
"""

OPTIMISER_AXES: tuple[str, ...] = (
    "epoch",
    "epochs",
    "iteration",
    "iterations",
    "step count",
    "loss",
    "converge",
    "converges",
    "convergence",
    "learning",
    "training run",
    "optimisation step",
    "optimization step",
)
"""Words that put an *optimiser's progress* on the horizontal axis instead.

The screen that keeps the two kinds of curve apart. Both are answered with a figure
and they are answered by completely different machinery -- one by racing three
variational methods on one chain, one by solving many chains exactly -- so a
sentence that names an optimiser's axis is not a field sweep however many physical
quantities it also mentions.
"""


def asks_for_a_field_sweep(text: str) -> bool:
    r"""Say whether the question asks for an exact quantity traced against the field.

    Without this screen, *can you plot the low-lying spectrum of the quantum Ising
    chain as a function of the external field using the free-fermion approach?* goes
    to :func:`asks_for_a_curve` on the word "plot", and the only curve the graph can
    draw is a race between VQE, QAOA and imaginary time. A reader who asked for the
    exact spectrum gets three variational energies agreeing to six decimal places on
    a chain nobody named: the right answer to a different question, with the asked-for
    one nowhere on the page. The same goes for *plot the ground-state energy and its
    first and second derivatives*, *plot the magnetisation against h/J*, and *plot the
    magnetisation and the energy for an L-site chain*.

    The rule is not to collect more plotting words. A curve has an axis, and which
    axis it is decides what answers it: an optimiser's epoch means race the methods,
    the field means solve the model exactly at every point. So this looks for a swept
    quantity and a swept axis together, and refuses when the axis named belongs to an
    optimiser -- see :data:`OPTIMISER_AXES`.

    Dashes are normalised before matching, because *Jordan-Wigner* is written with an
    en dash as often as with a hyphen and a reader typing the correct character should
    not get the worse answer.

    Args:
        text: The question as it was asked.

    Returns:
        Whether an exact field sweep is what would answer it.

    Examples:
        >>> asks_for_a_field_sweep("plot the low-lying spectrum against the field")
        True
        >>> asks_for_a_field_sweep(
        ...     "plot the ground state energy and its derivatives as a function of h/J"
        ... )
        True
        >>> asks_for_a_field_sweep(
        ...     "plot the energy of a 3x3 square lattice as the field is turned up"
        ... )
        True
        >>> asks_for_a_field_sweep("plot the energy against optimisation epoch for VQE")
        False
        >>> asks_for_a_field_sweep("what is the energy of a 6-spin chain?")
        False
        >>> asks_for_a_field_sweep("how does the Jordan-Wigner transformation work?")
        False
    """
    lowered = text.lower().replace("\u2013", "-").replace("\u2014", "-")
    if any(word in lowered for word in OPTIMISER_AXES):
        return False
    swept_against_the_field = any(axis in lowered for axis in FIELD_SWEEP_AXES)
    asks_to_be_shown = any(word in lowered for word in CURVE_WORDS)
    if not any(subject in lowered for subject in FIELD_SWEEP_SUBJECTS):
        # Bare "energy" is not on the subject list, and should not be: it turns up
        # in "energy cost", "lowest-energy arrangement" and a dozen phrases that
        # are not about a curve. It counts only when the field is named
        # relationally, as in "as the field is turned up".
        return swept_against_the_field and "energy" in lowered
    # A bare field name is not enough here. Every question this project answers
    # describes a transverse field, so accepting the name alone sent the
    # application's own central question -- a chain, a field, and is a quantum
    # computer worth it -- down the prose branch, where it can reach no verdict.
    return asks_to_be_shown or swept_against_the_field


SPECTRUM_WORDS: tuple[str, ...] = (
    "spectrum",
    "spectra",
    "energy level",
    "energy levels",
    "low lying",
    "low-lying",
    "excitation",
    "excitations",
    "gap",
)
"""Ways of asking for the levels rather than for the ground state alone."""

MAGNETISATION_WORDS: tuple[str, ...] = (
    "magnetisation",
    "magnetization",
    "order parameter",
    "susceptibility",
    "polarisation",
    "polarization",
)
"""Ways of naming the alignment with the field."""

ENERGY_WORDS: tuple[str, ...] = (
    "ground state energy",
    "ground-state energy",
    "energy density",
    "energy per site",
    "energy per spin",
    "ground state enegy",
)
"""Ways of naming the ground-state energy, including one common misspelling.

*Ground state enegy* is in the list because it was in a real question, and a reader
who mistypes one letter should get the curve rather than the default. The project's
spelling layer repairs prose, not the words a router matches on.
"""

DERIVATIVE_WORDS: tuple[str, ...] = (
    "derivative",
    "derivatives",
    "curvature",
    "susceptibility",
    "response",
    "slope",
)
"""Words asking for the rate of change rather than for the quantity."""


def curves_asked_for(text: str) -> tuple[str, ...]:
    r"""Name the exact curves a sweep question asked to be shown.

    The deterministic reading of *which* curve, used on the path where no model is
    available to choose for itself. Online the agent picks the curves as tool
    arguments, which is the point of giving it a tool; this is what the offline
    demonstration uses, and it exists so that the branch has a specified behaviour
    with no key rather than an empty panel.

    Derivatives are attached to **the quantity they were asked about**, which is the
    one subtlety here and it comes from a real defect. By the Hellmann-Feynman
    theorem the first derivative of the energy density *is* minus the magnetisation,
    so a question about the magnetisation's derivatives answered with the energy's
    has been shown the right numbers under the wrong name -- and a question about the
    energy's derivatives answered with a magnetisation curve has been shown the one
    panel with no feature in it.

    Args:
        text: The question as it was asked.

    Returns:
        Curve names from ``src.physics.reference.field_sweep.Curve``, in that type's
        own order. Falls back to the ground-state pair when nothing recognisable was
        named, since "what does the ground state do as the field rises" is the
        question behind most of the others.

    Examples:
        >>> curves_asked_for("plot the low-lying spectrum against the field")
        ('spectrum',)
        >>> curves_asked_for(
        ...     "plot the ground state energy and its first and second derivatives"
        ... )
        ('energy', 'energy_derivatives')
        >>> curves_asked_for("plot the magnetisation and its derivatives vs h/J")
        ('magnetisation', 'magnetisation_derivatives')
        >>> curves_asked_for("plot magnetization and ground state energy vs h/J")
        ('energy', 'magnetisation')
        >>> curves_asked_for("what happens to the gap and the magnetisation?")
        ('spectrum', 'magnetisation')
    """
    lowered = text.lower().replace("\u2013", "-").replace("\u2014", "-")
    wants_spectrum = any(word in lowered for word in SPECTRUM_WORDS)
    wants_magnetisation = any(word in lowered for word in MAGNETISATION_WORDS)
    wants_energy = any(word in lowered for word in ENERGY_WORDS)
    wants_derivatives = any(word in lowered for word in DERIVATIVE_WORDS)
    chosen: list[str] = []
    if wants_spectrum:
        chosen.append("spectrum")
    if wants_energy:
        chosen.append("energy")
    if wants_magnetisation:
        chosen.append("magnetisation")
    if wants_derivatives:
        # Neither quantity named alongside a request for derivatives means the
        # energy's, because that is what "its first and second derivatives" refers
        # to in every question this project has been asked that omits the noun.
        if wants_magnetisation:
            chosen.append("magnetisation_derivatives")
        if wants_energy or not wants_magnetisation:
            chosen.append("energy_derivatives")
    if not chosen:
        return ("energy", "magnetisation")
    order = (
        "spectrum",
        "energy",
        "magnetisation",
        "energy_derivatives",
        "magnetisation_derivatives",
    )
    return tuple(name for name in order if name in set(chosen))


LONGITUDINAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "longitudinal field of 0.4", "longitudinal field 0.4", "longitudinal field = 0.4"
    re.compile(r"longitudinal\s+field\s*(?:of|=|:)?\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE),
    # "a field of 0.4 along the coupling direction" -- the number comes first
    re.compile(
        r"field\s*(?:of|=|:)?\s*(-?\d+(?:\.\d+)?)[^.?!]{0,40}?along\s+the\s+coupling",
        re.IGNORECASE,
    ),
    # "along the coupling direction, 0.4" -- and the direction comes first
    re.compile(
        r"along\s+the\s+coupling[^.?!]{0,40}?(-?\d+(?:\.\d+)?)",
        re.IGNORECASE,
    ),
)
"""Ways of naming the longitudinal field that do not use the letter :math:`g`.

Until these existed the reader understood exactly one spelling, ``g = 0.4``, which is
the one spelling the interface is not allowed to show. The chat page states a two-term
Hamiltonian, so a reader who has never been shown :math:`g` cannot be expected to type
it -- and *longitudinal field of 0.4*, the phrase they would reach for instead, set
nothing at all and was not reported as having set nothing.

That matters more than a missing convenience. At :math:`g = 0` the model is integrable
and the honest verdict is always "no advantage", so a reader who cannot switch this knob
on cannot reach the only regime in which the feasibility question has a real answer.

The bounded ``[^.?!]{0,40}`` keeps a match inside one clause, so that "a field of 0.4"
in one sentence and "along the coupling direction" in the next are not welded into a
reading neither sentence supports.
"""


def read_longitudinal(text: str) -> float | None:
    """Find the field along the coupling axis, however the sentence spells it.

    Tries the symbol first and the words after it, because ``g = 0.4`` is exact while
    the phrasings are inferences from wording, and a sentence carrying both should be
    read as its own notation.

    Args:
        text: The question as it was asked.

    Returns:
        The field strength, or ``None`` when the sentence does not give one.

    Examples:
        >>> read_longitudinal("a 10-spin chain with g = 0.4")
        0.4
        >>> read_longitudinal("a 10-spin chain with a longitudinal field of 0.4")
        0.4
        >>> read_longitudinal("add a field of 0.4 along the coupling direction")
        0.4
        >>> read_longitudinal("a 10-spin chain at criticality") is None
        True
    """
    symbol = read_symbol(text, "g")
    if symbol is not None:
        return symbol
    for pattern in LONGITUDINAL_PATTERNS:
        found = pattern.search(text)
        if found:
            return float(found.group(1))
    return None


def read_geometry(text: str) -> tuple[Geometry, int | None, int | None] | None:
    """Read the shape and, if it is given, the size of a two-dimensional lattice.

    Geometry gets a reader of its own because it is the one property of this model
    that changes what is knowable rather than what is expensive. A chain with no field
    along the coupling direction has a closed-form ground-state energy and needs no
    computer at all; the same spins on a square have none, because the Jordan-Wigner
    string that makes the chain solvable runs through every site the row-major order
    puts between two neighbours. A campaign that read "16 spins" and dropped "4x4"
    would answer the easy question confidently in front of a reader who asked the
    hard one.

    Args:
        text: The question as it was asked.

    Returns:
        ``(geometry, rows, cols)`` when a lattice is named, with the sides ``None``
        if the sentence gave a shape but no dimensions. ``None`` when nothing in the
        sentence names a two-dimensional lattice, which leaves the choice to the
        formaliser to make and to record as an assumption.

    Examples:
        >>> read_geometry("is a 4x4 square lattice worth it?")
        ('square', 4, 4)
        >>> read_geometry("what about a triangular lattice of 9 spins?")
        ('triangular', None, None)
        >>> read_geometry("a 3 by 4 grid of spins")
        ('square', 3, 4)
        >>> read_geometry("9 magnets in a 3 by 3 grid")
        ('square', 3, 3)
        >>> read_geometry("a chain of 12 spins") is None
        True
        >>> read_geometry("the one-dimensional lattice in your notes") is None
        True
    """
    lowered = text.lower()
    if any(word in lowered for word in TRIANGULAR_WORDS):
        geometry: Geometry = "triangular"
    elif any(word in lowered for word in SQUARE_WORDS) or _SIZED_GRID.search(text):
        geometry = "square"
    else:
        return None
    found = _DIMENSIONS.search(text)
    if found is None:
        return geometry, None, None
    rows, cols = int(found.group(1)), int(found.group(2))
    # A side of one is a chain written in a lattice's notation, and calling it a
    # lattice in the report would describe a problem nobody asked about.
    if rows < MIN_SIDE or cols < MIN_SIDE or rows * cols > MAX_SITES:
        return geometry, None, None
    return geometry, rows, cols


def read_boundary(text: str) -> Literal["open", "periodic"] | None:
    """Find whether the chain is a ring or a line.

    Periodic words are checked first, because they are the marked case: a ring is
    always described as one, while a line is usually described by saying nothing.
    "Not periodic" is handled by the open list, which is why it is checked after.

    Args:
        text: The question as it was asked.

    Returns:
        The boundary, or ``None`` when the sentence does not say.
    """
    lowered = text.lower()
    if "not periodic" in lowered or "non-periodic" in lowered:
        return "open"
    if any(word in lowered for word in PERIODIC_WORDS):
        return "periodic"
    if any(word in lowered for word in OPEN_WORDS):
        return "open"
    return None


def read_symbol(text: str, symbol: str) -> float | None:
    r"""Find a named coupling or field, written as an equation.

    A symbol on either side of a slash is *not* read here, because it is one half of
    a ratio rather than a value: ``h/J = 0.5`` sets neither :math:`h` nor :math:`J`
    to 0.5. See :func:`read_field_ratio`, which reads that form. Without the
    exclusion ``\bJ`` matched the ``J`` after the slash and returned the ratio as the
    coupling -- and since a missing field is then taken equal to the coupling, every
    question written in the standard ``h/J`` notation came back about a chain at
    exactly the critical point, whatever ratio was asked for.

    Args:
        text: The question as it was asked.
        symbol: The symbol to look for -- ``"J"``, ``"h"`` or ``"g"``. Matched
            case-sensitively, because ``h`` and ``H`` are the field and the
            Hamiltonian and confusing them would be a silent error.

    Returns:
        The value, or ``None`` when the sentence does not give one.

    Examples:
        >>> read_symbol("a chain with J = 2 and h = 1", "J")
        2.0
        >>> read_symbol("a 6-spin chain at h/J = 0.5", "J") is None
        True
        >>> read_symbol("a 6-spin chain at J/h = 2", "h") is None
        True
    """
    found = re.search(
        rf"(?<![A-Za-z0-9])(?<!/){symbol}(?![A-Za-z0-9])(?!\s*/)\s*[=:]\s*(-?\d+(?:\.\d+)?)",
        text,
    )
    return float(found.group(1)) if found else None


_FIELD_RATIO = re.compile(r"(?<![A-Za-z0-9])(h\s*/\s*J|J\s*/\s*h)\s*[=:]\s*(\d+(?:\.\d+)?)")
"""``h/J = 0.5`` and ``J/h = 2``, the way this subject is normally written.

Case-sensitive, like :func:`read_symbol` and for the same reason: ``H`` is the
Hamiltonian and ``h`` is the field.
"""


def read_field_ratio(text: str) -> tuple[float, float] | None:
    r"""Read :math:`h/J` written as a ratio, on the conventional scale.

    The notation the whole subject uses, and the one the README is written in. A
    ratio fixes the physics and no unit, so the pair comes back with :math:`J = 1` --
    the same convention :func:`read_relative_strength` uses for the same reason.

    Args:
        text: The question as it was asked.

    Returns:
        :math:`(J, h)`, or ``None`` when the sentence names no such ratio. A ratio
        of zero is refused: it turns off the term that makes the problem quantum.

    Examples:
        >>> read_field_ratio("a 6-spin chain at h/J = 0.5")
        (1.0, 0.5)
        >>> read_field_ratio("a 6-spin ring with J/h = 2")
        (1.0, 0.5)
        >>> read_field_ratio("a chain with J = 2 and h = 1") is None
        True
    """
    found = _FIELD_RATIO.search(text)
    if found is None:
        return None
    value = float(found.group(2))
    if value <= 0.0:
        return None
    inverted = found.group(1).replace(" ", "").startswith("J")
    return (1.0, 1.0 / value) if inverted else (1.0, value)


def read_count_too_long(text: str) -> int | None:
    """Find a chain length the sentence named that is past what this can price.

    The counterpart to :func:`read_count`, and it exists because that function
    returns ``None`` for both of the things it cannot use: a sentence that named no
    length, and a sentence that named one beyond :data:`MAX_SITES`. Those two are
    not the same event and must not be reported as one -- see :attr:`Reading.too_long`
    for what the conflation actually did to an answer.

    Args:
        text: The question as it was asked.

    Returns:
        The first length found above :data:`MAX_SITES`, or ``None``. A length below
        :data:`MIN_SITES` is *not* returned: one spin is far more often a typo or a
        stray digit than a request, and refusing on it would turn a slip into a
        stopped run.

    Examples:
        >>> read_count_too_long("Is a quantum computer worth it for a chain of 200 magnets?")
        200
        >>> read_count_too_long("Is a 12-spin chain worth running?") is None
        True
        >>> read_count_too_long("Is quantum hardware worth it here?") is None
        True
    """
    for pattern in SITE_PATTERNS:
        found = pattern.search(text)
        if found is None:
            continue
        token = found.group(1).lower()
        value = WORD_NUMBERS.get(token, int(token) if token.isdigit() else 0)
        if value > MAX_SITES:
            return value
        if MIN_SITES <= value <= MAX_SITES:
            # A usable length was found first, so nothing here is out of range. The
            # early return matters: "a 12-spin chain measured over 200 shots" must
            # not be refused because a later number is large.
            return None
    return None


def read(text: str) -> Reading:
    """Read whatever a sentence says about the chain.

    Args:
        text: The question as it was asked, unedited.

    Returns:
        Everything that could be read, with ``None`` wherever the sentence was
        silent. A silent field is never filled in with a plausible value -- see the
        module docstring.
    """
    critical = any(word in text.lower() for word in CRITICAL_WORDS)
    coupling = read_symbol(text, "J")
    field = read_symbol(text, "h")
    # A comparison between the two strengths, where at least one of them was not
    # written down and the question did not name the critical point -- which says
    # more than a ratio does. A stated strength is kept and the other is derived
    # from the ratio against it. See :func:`read_relative_strength`.
    from_ratio = False
    # "h/J = 0.5" first, because it is the notation this subject actually uses and it
    # states both strengths at once. Only when neither was written on its own: a
    # sentence giving J and h outright has fixed the scale and this must not move it.
    if not critical and coupling is None and field is None:
        ratio_pair = read_field_ratio(text)
        if ratio_pair is not None:
            coupling, field = ratio_pair
            from_ratio = True
    if not critical and (coupling is None or field is None):
        relative = read_relative_strength(text)
        if relative is not None:
            ratio = relative[1] / relative[0]
            if coupling is None and field is None:
                # Neither was written down, so the sentence fixed the ratio and
                # nothing else. Flagged, because an energy is not a pure number
                # until the unit is: J=1,h=0.2 and J=5,h=1 are the same physics
                # and different energies, and only one of them is the convention.
                coupling, field = relative
                from_ratio = True
            elif field is None and coupling is not None:
                field = coupling * ratio
            elif coupling is None and field is not None and ratio > 0.0:
                coupling = field / ratio
    if critical and field is None:
        field = coupling if coupling is not None else 1.0
    shape = read_geometry(text)
    geometry, rows, cols = shape if shape is not None else (None, None, None)
    # A size written as ``4x4`` states the site count as surely as "16 spins" does,
    # and more precisely, so it wins. Without this the count reader finds nothing in
    # "a 4x4 square lattice" and the campaign would assume its default length while
    # reporting a lattice -- a shape and a size that describe different problems.
    counted = read_count(text)
    n_sites = rows * cols if rows is not None and cols is not None else counted
    return Reading(
        n_sites=n_sites,
        geometry=geometry,
        rows=rows,
        boundary=read_boundary(text),
        coupling=coupling,
        field=field,
        longitudinal=read_longitudinal(text),
        critical=critical,
        depth=read_depth(text),
        epoch_range=read_epoch_range(text),
        strengths_from_ratio=from_ratio,
        too_long=read_count_too_long(text),
    )


COUPLING_NAMES: tuple[str, ...] = (
    "neighbour",
    "neighbours",
    "coupling",
    "interaction",
    "exchange",
    "bond",
    "pull",
)
FIELD_NAMES: tuple[str, ...] = ("sideways", "transverse", "field", "influence")
"""The two strengths, as ordinary language names them.

Read by :func:`read_relative_strength`, which has to decide which of the two a
comparison is about. Both lists are checked against a clause rather than against the
whole question, so a sentence naming both is not ambiguous -- what matters is which
one sits on which side of the comparison.
"""

_NAMED_FRACTIONS: dict[str, float] = {
    "half": 0.5,
    "twice": 2.0,
    "double": 2.0,
    "a third": 1.0 / 3.0,
    "a quarter": 0.25,
    "a fifth": 0.2,
    "a tenth": 0.1,
}

_FACTOR = re.compile(
    r"\b(?:" + "|".join(_NAMED_FRACTIONS).replace(" ", r"\s+") + r"|" + _COUNT + r"\s+times)\b",
    re.IGNORECASE,
)
_COMPARISON = re.compile(
    r"\b(?:as\s+(?:strong|strongly|large|big|great)\s+as|(?:stronger|larger|bigger)\s+than)\b",
    re.IGNORECASE,
)
_CLAUSE_END = re.compile(r"[.;?!]")


def _factor_value(phrase: str) -> float | None:
    """Turn a written multiplier into a number.

    Args:
        phrase: The multiplier exactly as matched -- ``"twice"``, ``"five times"``.

    Returns:
        The multiplier, or ``None`` when it is out of the range worth believing. A
        ratio outside a factor of a hundred is far likelier to be a shot count that
        happened to sit beside the word *times*.
    """
    lowered = " ".join(phrase.lower().split())
    if lowered in _NAMED_FRACTIONS:
        return _NAMED_FRACTIONS[lowered]
    token = lowered.removesuffix("times").strip()
    value = float(WORD_NUMBERS.get(token, float(token) if token.isdigit() else 0.0))
    return value if 0.0 < value <= 100.0 else None


def _strength_named(clause: str, *, last: bool) -> Literal["coupling", "field"] | None:
    """Say which of the two strengths a clause is about.

    Args:
        clause: One side of a comparison.
        last: Take the name nearest the comparison. True for the clause before it,
            false for the clause after.

    Returns:
        Which strength, or ``None`` when the clause names neither.
    """
    lowered = clause.lower()
    places = [
        (lowered.rfind(name) if last else lowered.find(name), kind)
        for kind, names in (("coupling", COUPLING_NAMES), ("field", FIELD_NAMES))
        for name in names
        if name in lowered
    ]
    if not places:
        return None
    _, kind = max(places) if last else min(places)
    return cast('Literal["coupling", "field"]', kind)


def read_relative_strength(text: str) -> tuple[float, float] | None:
    r"""Read a coupling and a field out of a comparison between them.

    *The neighbour interaction is twice as strong as the sideways field* fixes the
    physics completely and names no number, which is how people who are not
    physicists state this problem. Without this the sentence read as silent and the
    campaign assessed :math:`h = J`, which is a different chain -- and on the ordered
    side of the transition, a different answer.

    The coupling is set to one. Only the ratio is stated, and the energy of a chain
    is not a pure number until something fixes the unit; :math:`J = 1` is the
    convention the literature uses and the one every other reading here already
    assumes.

    Args:
        text: The question as it was asked.

    Returns:
        :math:`(J, h)`, or ``None`` when the sentence draws no such comparison.

    Examples:
        >>> read_relative_strength("The neighbour interaction is twice as strong as the field.")
        (1.0, 0.5)
        >>> read_relative_strength("the sideways field is four times the neighbour pull")
        (1.0, 4.0)
        >>> read_relative_strength("each pulls on its neighbours as strongly as the field does")
        (1.0, 1.0)
        >>> read_relative_strength("a chain of 8 spins") is None
        True
    """
    factor_match = _FACTOR.search(text)
    if factor_match is None:
        equal = _COMPARISON.search(text)
        if equal is None:
            return None
        factor, start, end = 1.0, equal.start(), equal.end()
    else:
        value = _factor_value(factor_match.group(0))
        if value is None:
            return None
        factor, start, end = value, factor_match.start(), factor_match.end()

    before = _before(text, start)
    after = _after(text, end)
    subject = _strength_named(before, last=True)
    against = _strength_named(after, last=False)
    if subject is None or against is None or subject == against:
        return None
    # The subject is `factor` times the object, and the coupling is the unit.
    return (1.0, 1.0 / factor) if subject == "coupling" else (1.0, factor)


def _before(text: str, position: int) -> str:
    """The clause running up to a position, back to the previous sentence end.

    Args:
        text: The question.
        position: Where the clause ends.

    Returns:
        The clause.
    """
    boundaries = [found.end() for found in _CLAUSE_END.finditer(text[:position])]
    return text[boundaries[-1] if boundaries else 0 : position]


def _after(text: str, position: int) -> str:
    """The clause running from a position to the next sentence end.

    Args:
        text: The question.
        position: Where the clause starts.

    Returns:
        The clause.
    """
    found = _CLAUSE_END.search(text, position)
    return text[position : found.start() if found else len(text)]
