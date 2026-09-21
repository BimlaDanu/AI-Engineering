"""Reading a chain out of a sentence, with no language model involved.

This module is what an offline campaign formalises with, so its failures are not
subtle: get it wrong and the whole assessment is about a chain nobody asked about,
confidently and in detail. Before it existed that is exactly what happened -- every
offline run answered about a fixed six-site default whatever the question said.

The rule it is held to throughout is *never guess*. A sentence with no number in it
must yield no number, so that the caller records a default as an assumption rather
than presenting one as a finding.
"""

from __future__ import annotations

import pytest

from src.agent.reading import (
    MAX_DEPTH,
    MAX_SITES,
    Reading,
    asks_for_a_curve,
    asks_for_a_field_sweep,
    asks_to_be_taught,
    asks_to_compare_methods,
    curves_asked_for,
    methods_named_in,
    read,
    read_boundary,
    read_count,
    read_depth,
    read_epoch_range,
    read_symbol,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a 10-spin chain", 10),
        ("12 spins in a row", 12),
        ("twenty qubits", 20),
        ("a chain of 8", 8),
        ("a ring of twelve", 12),
        ("six magnetic elements", 6),
        ("L = 14", 14),
        ("N=4 sites", 4),
    ],
)
def test_a_chain_length_is_read_however_it_is_written(text: str, expected: int) -> None:
    assert read_count(text) == expected


@pytest.mark.parametrize(
    "text",
    ["a chain of magnets", "is quantum computing worth it?", "", "a 1-spin chain"],
)
def test_a_sentence_with_no_usable_length_yields_none(text: str) -> None:
    # Never guess. A caller that receives None records that it assumed a default; one
    # that receives a confident invention cannot tell the two apart, and neither can
    # the reader of its report.
    assert read_count(text) is None


def test_an_implausible_count_is_refused_rather_than_believed() -> None:
    # A three-digit number beside the word "spins" is far likelier to be a budget, a
    # temperature or a year than a chain somebody wants simulated.
    assert read_count(f"{MAX_SITES + 1} spins") is None
    assert read_count("in 2024 we ran 8 spins") == 8


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a closed ring of 12", "periodic"),
        ("with periodic boundary conditions", "periodic"),
        ("a chain that wraps around", "periodic"),
        ("a line with two ends", "open"),
        ("an open chain", "open"),
        ("a row of eight", "open"),
        ("not periodic", "open"),
        ("eight spins", None),
    ],
)
def test_the_boundary_is_read_when_the_sentence_says(text: str, expected: str | None) -> None:
    assert read_boundary(text) == expected


def test_a_negated_boundary_is_not_read_as_the_thing_it_negates() -> None:
    # "not periodic" contains "periodic", and a naive substring check gets it exactly
    # backwards -- which would silently turn a segment into a ring.
    assert read_boundary("the chain is not periodic") == "open"


@pytest.mark.parametrize(
    ("symbol", "text", "expected"),
    [
        ("J", "J = 1.5", 1.5),
        ("h", "h=0.5", 0.5),
        ("g", "g: 0.4", 0.4),
        ("J", "coupling strength two", None),
    ],
)
def test_a_named_coupling_is_read_from_an_equation(
    symbol: str, text: str, expected: float | None
) -> None:
    assert read_symbol(text, symbol) == expected


def test_the_field_and_the_hamiltonian_are_not_confused() -> None:
    # `h` is the transverse field and `H` is the Hamiltonian. Matching case-blind
    # would read the operator's name as a field strength.
    assert read_symbol("H = -J sum sigma^z sigma^z", "h") is None


def test_criticality_fixes_the_field_without_either_being_written() -> None:
    # "At criticality" is a complete specification of the field to anybody in the
    # field, and a reader that missed it would answer about the easy regime while the
    # asker was asking about the hard one.
    found = read("a 10-spin chain at criticality")
    assert found.critical
    assert found.field == 1.0
    assert read("a 10-spin chain with J = 2 at the critical point").field == 2.0


def test_an_explicit_field_beats_the_critical_shorthand() -> None:
    assert read("a critical chain with h = 0.3").field == 0.3


def test_a_question_about_nothing_reads_as_nothing() -> None:
    found = read("What is the capital of France?")
    assert not found.found_anything
    assert found.describe() == ()


def test_everything_read_is_reported_in_plain_words() -> None:
    # The caller puts this straight into a campaign's notes, so a reader can see what
    # was understood before seeing what was concluded.
    described = read("12 spins in a closed ring, J = 1, h = 0.5, g = 0.2").describe()
    assert "12 spins" in described
    assert "a ring" in described
    assert any("g = 0.2" in phrase for phrase in described)


# --------------------------------------------------------------------------
# Circuit depth
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("the QAOA circuit for 8 spins at depth 3", 3),
        ("a depth-2 ansatz", 2),
        ("a circuit of depth of four", 4),
        ("show me a three-layer ansatz", 3),
        ("6 layers", 6),
        ("two rounds of gates", 2),
        ("QAOA with p = 5", 5),
    ],
)
def test_a_circuit_depth_is_read_however_it_is_written(text: str, expected: int) -> None:
    assert read_depth(text) == expected


@pytest.mark.parametrize(
    "text",
    ["a 12-spin chain", "how deep should the circuit be?", "", "depth 0"],
)
def test_a_sentence_that_names_no_depth_yields_none(text: str) -> None:
    # Same rule as the chain length: never guess. The interface fills a missing depth
    # from the settings knob and says in the caption that it did, which it can only do
    # if it can tell "the question said three" from "the question said nothing".
    assert read_depth(text) is None


def test_an_implausible_depth_is_refused_rather_than_believed() -> None:
    assert read_depth(f"depth {MAX_DEPTH + 1}") is None


def test_a_depth_and_a_length_do_not_read_as_each_other() -> None:
    # The one collision worth guarding: both readers look for a small number beside a
    # word, and "8 spins at depth 3" has two of them.
    found = read("Can you write the QAOA circuit for 8 spins at depth 3?")
    assert (found.n_sites, found.depth) == (8, 3)


def test_a_depth_alone_does_not_amount_to_a_named_chain() -> None:
    # A depth describes the program, not the physical system. If it counted as having
    # read a chain, a question naming only a depth would be restated as a question
    # about "a chain" that nobody asked about.
    found = read("would three layers be enough?")
    assert found.depth == 3
    assert not found.found_anything
    assert found.restate() == ""


def test_a_follow_up_inherits_the_depth_it_does_not_name() -> None:
    earlier = read("a 10-spin chain at depth 4")
    assert read("and as a ring?").under(earlier).depth == 4


# --------------------------------------------------------------------------
# Follow-ups
# --------------------------------------------------------------------------


def test_a_follow_up_inherits_what_it_does_not_name() -> None:
    # The offline half of the memory layer. "And with periodic boundary conditions?"
    # names a boundary and nothing else; laid over the question before it, it becomes
    # the same chain with the boundary changed, which is what anybody would understand.
    earlier = read("a 10-spin chain with J = 2 and h = 1")
    later = read("and with periodic boundary conditions?").under(earlier)
    assert later.n_sites == 10
    assert later.coupling == 2.0
    assert later.boundary == "periodic"


def test_a_follow_up_overrides_what_it_does_name() -> None:
    # Otherwise asking about six spins and then about twenty would keep answering
    # about six, which is a memory that has stopped being useful and started lying.
    earlier = read("a 6-spin chain")
    assert read("what about 20 spins?").under(earlier).n_sites == 20


def test_laying_a_reading_over_nothing_changes_nothing() -> None:
    found = read("a 10-spin ring")
    assert found.under(Reading()) == found


# --------------------------------------------------------------------------
# Asking to watch, rather than to be told
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "plot the energy against optimisation epoch",
        "can you chart the loss for all three?",
        "which of them converges fastest?",
        "show me the convergence",
        "how many iterations does this take?",
        "is there a learning curve for this?",
    ],
)
def test_a_question_about_the_route_rather_than_the_answer_is_recognised(text: str) -> None:
    # This decides whether the graph spends seconds racing three methods, and whether
    # the interface draws the result. Both ask the same function, so a question that
    # raced and got no picture is this disagreeing with itself.
    assert asks_for_a_curve(text)


@pytest.mark.parametrize(
    "text",
    [
        "how deep should the circuit be?",
        "is a 12-spin chain worth running on hardware?",
        "why is the critical point the hardest case?",
        "can you write the QAOA circuit for 8 spins?",
    ],
)
def test_a_question_about_the_answer_alone_asks_for_no_curve(text: str) -> None:
    assert not asks_for_a_curve(text)


# --------------------------------------------------------------------------
# The stretch of a curve somebody asked to see
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("can we plot the loss curve in the range of [0,20]", (0, 20)),
        ("plot it over (5, 40)", (5, 40)),
        ("show it between 5 and 40 steps", (5, 40)),
        ("plot from 0 to 30", (0, 30)),
        ("just the range 10-25 please", (10, 25)),
    ],
)
def test_a_range_is_read_however_it_is_written(text: str, expected: tuple[int, int]) -> None:
    assert read_epoch_range(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "plot the loss curve",
        "the range [20,5]",
        "the 2024 paper says otherwise",
        "a 12-spin chain at depth 3",
    ],
)
def test_a_sentence_naming_no_usable_range_yields_none(text: str) -> None:
    # A backwards range and a year are both refused rather than repaired into
    # something plausible: an axis silently stretched to two thousand epochs would
    # draw three flat lines and look like a working figure.
    assert read_epoch_range(text) is None


def test_a_range_describes_the_picture_and_not_the_chain() -> None:
    # Same rule as the depth: it says nothing about what the physical system is, so a
    # question carrying only a range has still named no problem to assess.
    found = read("can we plot it in the range [0,20]?")
    assert found.epoch_range == (0, 20)
    assert not found.found_anything
    assert found.restate() == ""


def test_a_follow_up_inherits_a_range_it_does_not_name() -> None:
    earlier = read("plot the loss curve between 0 and 20")
    assert read("and for a ring?").under(earlier).epoch_range == (0, 20)


# --------------------------------------------------------------------------
# Naming methods, and asking which
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("VQE, QAOA or imaginary time for a 10-spin critical chain?", ("VQE", "QAOA", "VarQITE")),
        ("Of VQE, QAOA and VarQITE, which converges fastest?", ("VQE", "QAOA", "VarQITE")),
        ("Does VQE pay off on molecules but not on this chain?", ("VQE",)),
        ("How deep should the circuit be before noise wins?", ()),
        ("How can a circuit run imaginary time, which is not unitary?", ("VarQITE",)),
    ],
)
def test_the_methods_a_question_names_are_found_in_legend_order(
    question: str, expected: tuple[str, ...]
) -> None:
    assert methods_named_in(question) == expected


def test_a_question_putting_two_methods_against_each_other_asks_for_a_comparison() -> None:
    # The failure this exists for. The question below contains no plotting word at all,
    # so `asks_for_a_curve` is false and nothing was raced -- and the campaign answered
    # a question about a chain to a reader who had asked a question about methods.
    question = "VQE, QAOA or imaginary time for a 10-spin critical chain?"
    assert not asks_for_a_curve(question)
    assert asks_to_compare_methods(question)


def test_a_question_naming_one_method_is_not_a_comparison() -> None:
    # Racing a method against two the reader never mentioned is the same fault in the
    # other direction: an answer to a question nobody asked.
    assert not asks_to_compare_methods("Does VQE pay off on molecules?")


def test_a_request_to_be_taught_about_three_methods_does_not_race_them() -> None:
    # The defect this exists for, and it is the sharpest one this project has produced.
    # "Could you teach me about VQE, QAOA and VarQITE" names three methods, so the
    # two-methods-means-compare rule fired and raced all three. The race table -- three
    # identical energies to six decimal places, on a six-spin chain nobody had asked
    # about -- then *became the answer*: a reader who asked what QAOA is was handed a
    # number, and it was the same number every other button on the page returns.
    #
    # A measurement shown where an explanation was asked for does not supplement the
    # answer. It displaces it.
    question = (
        "Could you teach me about the VQE, QAOA, and VarQITE quantum computing "
        "algorithms applied to the above model?"
    )
    assert asks_to_be_taught(question)
    assert len(methods_named_in(question)) == 3
    assert not asks_to_compare_methods(question)
    assert not asks_for_a_curve(question)


def test_asking_to_be_taught_does_not_suppress_an_explicit_request_for_the_picture() -> None:
    # The teaching screen sits in front of the comparison rule and nowhere else. A
    # reader who says the word "plot" has asked for the plot, and a lesson that
    # silently withheld it because of how the sentence opened would be the previous
    # defect inverted -- the screen deciding what the reader is allowed to have.
    question = "Teach me how VQE and QAOA converge on this chain, and plot the loss."
    assert asks_to_be_taught(question)
    assert asks_for_a_curve(question)


@pytest.mark.parametrize(
    "question",
    [
        "Of VQE, QAOA and VarQITE, which converges fastest and gets closest?",
        "VQE, QAOA or imaginary time for a 10-spin critical chain?",
        "How deep should the circuit be before noise wins?",
    ],
)
def test_a_question_asking_for_a_result_is_not_a_request_to_be_taught(question: str) -> None:
    # The screen has to be narrow. Every one of these is a request for an answer, and
    # a lesson in reply to any of them is padding around the number -- which is the
    # fault `EXPLAIN_SYSTEM` rule 8 exists to prevent, arrived at from the other side.
    assert not asks_to_be_taught(question)


def test_a_method_written_out_in_full_counts_the_same_as_its_acronym() -> None:
    # "Imaginary time" and "VarQITE" name one method, and this project's own starter
    # writes the first. Matching only acronyms would miss it.
    assert asks_to_compare_methods(
        "the variational quantum eigensolver against imaginary time evolution"
    )


# --------------------------------------------------------------------------
# A curve in the field is not a curve in an optimiser's epoch
# --------------------------------------------------------------------------

# The five questions a reader actually asked, verbatim, typos included. Every one of
# them contains a plotting word, so every one of them used to start a race between
# three variational methods on a chain nobody had named -- and the answer to each is
# an exact curve this project can compute.
ASKED_FOR_A_SWEEP = [
    "Can you plot low lying spectrum of quantum Ising as a function of an external "
    "filed using free fermion approach?",
    "Using free fermion approach can you plot the ground state energy and its first "
    "and second derivatives of the transverse field Ising model as a function of the "
    "magnetic field h/J",
    "Using free fermion approach can you plot the ground state magnetization and its "
    "first and second derivatives of the transverse field Ising model as a function "
    "of the magnetic field h/J",
    "Plot magnetization and ground state enegy as a function of h/J for L site chain",
    "How does the magnetisation behave as a function of the transverse field?",
]

NOT_A_SWEEP = [
    # An optimiser's axis. This is the race, and it must stay the race.
    "Plot the energy against optimisation epoch for VQE, QAOA and imaginary time",
    "Which of these converges fastest?",
    "can we plot the loss curve in the range of [0,20]",
    # A derivation with no curve in it at all.
    "How does the Jordan-Wigner transformation map the spin chain to free fermions?",
    "detail mathematical detail of quantum to classical mapping",
    # A feasibility question, which must keep its verdict.
    "Is a quantum computer worth it for a 12-spin critical chain?",
]


@pytest.mark.parametrize("text", ASKED_FOR_A_SWEEP)
def test_a_quantity_traced_against_the_field_is_read_as_a_sweep(text: str) -> None:
    assert asks_for_a_field_sweep(text)


@pytest.mark.parametrize("text", NOT_A_SWEEP)
def test_a_curve_in_an_optimisers_epoch_is_not_a_sweep(text: str) -> None:
    assert not asks_for_a_field_sweep(text)


def test_an_optimisers_axis_wins_over_a_physical_quantity() -> None:
    # Both readings are available in this sentence and only one of them can answer
    # it: the descent is what was asked about, and racing the methods is what draws
    # a descent. The screen is what keeps the two kinds of curve apart.
    both = "plot the energy and the gap against epoch for VQE and QAOA"
    assert asks_for_a_curve(both)
    assert not asks_for_a_field_sweep(both)


def test_an_en_dash_reads_the_same_as_a_hyphen() -> None:
    # A reader who types the correct character should not get the worse answer.
    typed = "plot the Jordan\u2013Wigner spectrum vs h/J"
    assert curves_asked_for(typed) == ("spectrum",)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("plot the low-lying spectrum against the field", ("spectrum",)),
        (
            "plot the ground state energy and its first and second derivatives",
            ("energy", "energy_derivatives"),
        ),
        (
            "plot the magnetisation and its derivatives vs h/J",
            ("magnetisation", "magnetisation_derivatives"),
        ),
        (
            "plot magnetization and ground state enegy as a function of h/J",
            ("energy", "magnetisation"),
        ),
        ("what happens to the gap and the magnetisation?", ("spectrum", "magnetisation")),
        ("what does the field do to this chain?", ("energy", "magnetisation")),
    ],
)
def test_the_curves_a_question_asked_for_are_read_out_of_it(
    text: str, expected: tuple[str, ...]
) -> None:
    assert curves_asked_for(text) == expected


def test_derivatives_are_attached_to_the_quantity_they_were_asked_about() -> None:
    # By Hellmann-Feynman the energy's first derivative *is* minus the magnetisation,
    # so answering a question about one with the other is showing the right numbers
    # under the wrong name. The two requests must not collapse into one.
    energy = curves_asked_for("the energy and its second derivative against h/J")
    alignment = curves_asked_for("the magnetisation and its second derivative")
    assert "energy_derivatives" in energy and "magnetisation_derivatives" not in energy
    assert "magnetisation_derivatives" in alignment and "energy_derivatives" not in alignment


# --------------------------------------------------------------------------
# Geometry: the one property that changes what is knowable, not just what is hard
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "geometry", "rows", "n_sites"),
    [
        ("is a 4x4 square lattice worth it?", "square", 4, 16),
        ("a 3 by 4 grid of spins", "square", 3, 12),
        ("what about a triangular lattice of 9 spins?", "triangular", None, 9),
        ("a 3x3 frustrated lattice", "triangular", 3, 9),
        ("a two-dimensional lattice", "square", None, None),
    ],
)
def test_a_lattice_is_read_out_of_the_sentence(
    question: str, geometry: str, rows: int | None, n_sites: int | None
) -> None:
    found = read(question)
    assert found.geometry == geometry
    assert found.rows == rows
    assert found.n_sites == n_sites


@pytest.mark.parametrize(
    "question",
    [
        "a chain of 12 spins",
        "the one-dimensional lattice in your notes",
        "how deep must the ansatz be?",
        "a ring of 8 spins",
    ],
)
def test_a_chain_is_not_mistaken_for_a_lattice(question: str) -> None:
    # `lattice` on its own must not open this door: a chain *is* a one-dimensional
    # lattice and the notes call it one, so matching the bare word would turn every
    # question that used it into a question about a grid.
    assert read(question).geometry is None


def test_a_written_size_beats_a_counted_one() -> None:
    # "a 4x4 square lattice" contains no site count for `read_count` to find, so
    # without this the campaign would assume its default length while reporting a
    # lattice -- a shape and a size describing two different problems.
    assert read("is a 4x4 square lattice worth it?").n_sites == 16


def test_a_lattice_is_inherited_by_a_follow_up_that_names_only_a_field() -> None:
    # What makes the offered follow-ups work without a language model. "And with a
    # tilt?" after a question about a 4x4 square has to stay a 4x4 square, and the
    # shape and the arrangement must be inherited together or the answer could come
    # back about a 2x8 strip.
    earlier = read("is a 4x4 square lattice worth it?")
    later = read("and with a field of 0.4 along the coupling direction?").under(earlier)
    assert (later.geometry, later.rows, later.n_sites) == ("square", 4, 16)
    assert later.longitudinal == 0.4


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Six interacting elements arranged in a closed loop", 6),
        ("Nine interacting elements on a triangular grid", 9),
        ("ten coupled quantum spins", 10),
        # The gap is too narrow to jump between two separate quantities: without the
        # joining words excluded, this reads three.
        ("3 devices with 8 qubits", 8),
        ("a budget of 500 million shots on a 6 spin chain", 6),
        ("20 shots per spin", None),
        ("six hundred spins", None),
    ],
)
def test_a_count_reaches_its_unit_past_an_adjective(question: str, expected: int | None) -> None:
    # "Six interacting elements" names a length as plainly as "six elements" does, and
    # requiring the two to be adjacent read no length out of either -- so the campaign
    # answered in prose about a chain it had in fact understood.
    assert read_count(question) == expected


def test_a_size_against_a_shape_noun_is_a_lattice_without_the_word_square() -> None:
    # "9 magnets in a 3 by 3 grid" was read as a chain of nine, which has a
    # closed-form energy the lattice does not -- so the easy question was answered
    # confidently in front of a reader who had asked the hard one.
    assert read("9 magnets arranged in a 3 by 3 grid, edges free").geometry == "square"
    assert read("9 magnets arranged in a 3 by 3 grid, edges free").rows == 3
    # The shape noun is required.
    assert read("a room 3 by 3 metres").geometry is None


@pytest.mark.parametrize(
    ("question", "coupling", "field"),
    [
        ("The neighbour interaction is twice as strong as the sideways field.", 1.0, 0.5),
        ("the sideways influence is twice as strong as the pull between neighbours", 1.0, 2.0),
        ("the pull between neighbours is five times the sideways influence", 1.0, 0.2),
        ("each pulls on its neighbours as strongly as the field does", 1.0, 1.0),
        # One strength stated: the other is derived against it rather than dropped.
        ("The field is twice the coupling, with J=3", 3.0, 6.0),
        ("a chain of 8 spins", None, None),
    ],
)
def test_a_comparison_between_the_two_strengths_is_a_reading(
    question: str, coupling: float | None, field: float | None
) -> None:
    # People who are not physicists state this problem as a ratio and no number. The
    # sentence read as silent and the campaign assessed h = J, which on the ordered
    # side of the transition is a different chain and a different answer. The coupling
    # is set to one because only the ratio is stated and an energy is not a pure
    # number until something fixes the unit.
    found = read(question)
    assert (found.coupling, found.field) == (coupling, field)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("a 6-spin chain at h/J = 0.5", (1.0, 0.5)),
        ("a 6-spin chain with h/J = 2", (1.0, 2.0)),
        ("a 6-spin ring with J/h = 2", (1.0, 0.5)),
        ("a chain at h/J=1", (1.0, 1.0)),
        # Two numbers of its own, so the scale is the sentence's and nothing moves it.
        ("a chain with J = 2 and h = 1", (2.0, 1.0)),
    ],
)
def test_a_ratio_written_as_h_over_j_is_read_as_a_ratio(
    question: str, expected: tuple[float, float]
) -> None:
    # The notation the whole subject uses, and it was read as a coupling. `\bJ` matched
    # the J after the slash, so "h/J = 0.5" set J to 0.5 and left h unset -- and an
    # unset field is taken equal to the coupling, which put the chain at h = J. Every
    # question written the standard way was therefore answered at exactly the critical
    # point, the hardest case, whatever ratio had been asked for.
    found = read(question)

    assert (found.coupling, found.field) == expected


def test_a_bare_ratio_symbol_with_no_number_reads_nothing() -> None:
    # "plot the energy against h/J" names an axis, not a chain. Reading a value out of
    # it would invent one, and this module's rule is that a sentence with no number in
    # it yields no number.
    found = read("Can you plot the ground-state energy against h/J?")

    assert found.coupling is None
    assert found.field is None
