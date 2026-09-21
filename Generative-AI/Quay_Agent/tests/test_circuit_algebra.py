"""The equations printed above a drafted program.

These are the one part of a code answer this project can stand behind: the program
itself is text a language model wrote and nothing here ran, while every string in
``circuit_algebra`` is composed from integers. So the claims worth testing are the ones
a reader would be misled by if they were wrong -- the order the factors act in, the
count of angles, the worked shot arithmetic -- and each is held against something that
shares no code with it: a closed-form norm written out by hand here, and the
specification's own properties.

The rendering claims matter as much. Streamlit draws display maths only when ``$$`` sits
alone on its own line, and an equation that renders as literal dollar signs teaches a
reader nothing while looking like a bug in the physics.
"""

from __future__ import annotations

import re

import pytest

from src.physics.quantum.ansatz import AnsatzSpec
from src.physics.quantum.circuit_algebra import (
    MAX_LAYERS_WRITTEN_OUT,
    circuit_mathematics,
    coefficient_l1,
    halves_latex,
    state_preparation_latex,
)


def spec(n_qubits: int = 8, depth: int = 3, **kwargs: object) -> AnsatzSpec:
    return AnsatzSpec(n_qubits=n_qubits, depth=depth, **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# The state the circuit builds
# --------------------------------------------------------------------------


def test_the_last_factor_written_is_the_first_layer_applied() -> None:
    # The single most common thing a reader new to circuits gets backwards, and the
    # reason the product is written out instead of being folded into a product sign.
    written = state_preparation_latex(spec(depth=3))
    assert written.index(r"\beta_{3}") < written.index(r"\beta_{1}")
    assert written.index(r"\beta_{1}") < written.index(r"\lvert +\rangle")


def test_every_layer_is_named_while_the_line_is_still_readable() -> None:
    written = state_preparation_latex(spec(depth=MAX_LAYERS_WRITTEN_OUT))
    for layer in range(1, MAX_LAYERS_WRITTEN_OUT + 1):
        assert rf"\gamma_{{{layer}}}" in written


def test_a_deep_circuit_elides_its_middle_rather_than_running_off_the_line() -> None:
    written = state_preparation_latex(spec(depth=MAX_LAYERS_WRITTEN_OUT + 3))
    assert r"\cdots" in written
    assert rf"\beta_{{{MAX_LAYERS_WRITTEN_OUT + 3}}}" in written
    assert r"\beta_{1}" in written
    assert r"\beta_{2}" not in written


def test_the_register_the_state_is_written_over_is_the_one_that_was_asked_for() -> None:
    assert r"\otimes 12" in state_preparation_latex(spec(n_qubits=12))


def test_a_depthless_specification_still_describes_one_layer() -> None:
    # AnsatzSpec allows depth zero -- the cost arithmetic uses it. An equation with no
    # factors at all would say the circuit does nothing, which is not what a reader
    # asking about a circuit needs to be told.
    assert r"\gamma_{1}" in state_preparation_latex(spec(depth=0))


# --------------------------------------------------------------------------
# The field along the coupling direction, named only when it is switched on
# --------------------------------------------------------------------------


LONE_G = re.compile(r"(?<![\\a-zA-Z])g(?![a-zA-Z])")
"""The letter *g* standing on its own, past every ``\\sigma`` and ``\\gamma``.

Searching for the bare character finds the *g* in every Greek command in the string,
which is how a check like this passes while the symbol is still on the page.
"""


def test_the_third_term_is_absent_from_the_split_when_it_is_switched_off() -> None:
    assert not LONE_G.search(halves_latex())


def test_the_third_term_joins_the_diagonal_half_when_it_is_switched_on() -> None:
    # It costs one single-qubit rotation per site and no two-qubit depth at all,
    # which is the whole reason it is the knob this project holds in reserve.
    split = halves_latex(longitudinal=True)
    assert r"g\sum_i \hat\sigma^z_i" in split
    assert split.index(r"g\sum") < split.index(r"\hat H_\text{diag}")


@pytest.mark.parametrize("longitudinal", [0.0, 0.4])
def test_the_composed_block_names_the_longitudinal_field_only_when_there_is_one(
    longitudinal: float,
) -> None:
    written = circuit_mathematics(spec(), 1.0, 1.0, longitudinal)
    assert bool(LONE_G.search(written)) is bool(longitudinal)


# --------------------------------------------------------------------------
# The arithmetic, held against the same numbers worked out by hand
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_qubits", [4, 9])
@pytest.mark.parametrize("longitudinal", [0.0, 0.4])
def test_the_norm_agrees_with_the_closed_form_for_a_segment(
    n_qubits: int, longitudinal: float
) -> None:
    # `coefficient_l1` sums the coefficients of an assembled Pauli operator. This is
    # the same quantity counted from the lattice instead: one bond coefficient per
    # bond, one field coefficient per site. Two routes sharing no algebra.
    by_hand = 1.5 * (n_qubits - 1) + 0.75 * n_qubits + longitudinal * n_qubits
    assert coefficient_l1(n_qubits, 1.5, 0.75, longitudinal) == pytest.approx(by_hand)


def test_the_angle_count_in_the_prose_is_the_specification_s_own() -> None:
    described = spec(depth=4)
    assert f"**{described.n_parameters} angles**" in circuit_mathematics(described, 1.0, 1.0)


def test_the_round_count_in_the_prose_is_the_specification_s_own() -> None:
    # An odd ring cannot be two-coloured, so it costs three rounds rather than two.
    # A sentence that said "two rounds however long the chain is" beside a picture
    # showing three would be the page arguing with itself.
    ring = spec(n_qubits=7, boundary="periodic")
    assert f"**{ring.two_qubit_rounds_per_layer} rounds**" in circuit_mathematics(ring, 1.0, 1.0)


def test_the_worked_shot_count_is_the_shot_equation_with_this_chain_in_it() -> None:
    # The number a reader is most likely to quote, so it is checked against the
    # arithmetic spelled out beside it rather than against a stored value.
    norm = coefficient_l1(8, 1.0, 1.0, 0.0)
    expected = int((norm / 0.01) ** 2)
    assert f"**{expected:,} shots**" in circuit_mathematics(spec(), 1.0, 1.0, target_error=0.01)


def test_a_chain_nobody_named_gets_the_equations_and_no_invented_shot_count() -> None:
    # A measurement count is the number most likely to be quoted out of the page, and
    # one computed from a chain the reader never mentioned would be quoted wrongly.
    written = circuit_mathematics(spec())
    assert "shots**" not in written
    assert r"\lVert c \rVert_1" in written


# --------------------------------------------------------------------------
# Whether any of it will render
# --------------------------------------------------------------------------


@pytest.mark.parametrize("longitudinal", [0.0, 0.4])
def test_every_display_delimiter_sits_alone_on_its_own_line(longitudinal: float) -> None:
    lines = circuit_mathematics(spec(), 1.0, 1.0, longitudinal).splitlines()
    opened = 0
    for line in lines:
        if "$$" in line:
            assert line.strip() == "$$", f"display maths shares a line with prose: {line!r}"
            opened += 1
    assert opened and opened % 2 == 0, "an unclosed display block swallows the rest of the page"


def test_the_three_questions_are_answered_in_the_order_a_circuit_runs_in() -> None:
    written = circuit_mathematics(spec(), 1.0, 1.0)
    build = written.index("**What the circuit builds.**")
    measure = written.index("**What is measured")
    cost = written.index("**What that costs.**")
    assert build < measure < cost
