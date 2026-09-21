r"""Tests for the guarantee that an answer's mathematics renders.

Two failures are covered here, and they came from opposite directions. The first is
a model writing correct LaTeX in delimiters the page does not render -- ``\(x\)``
instead of ``$x$`` -- which shows the reader backslashes. The second is a model
writing correct LaTeX into a JSON string field without doubling the backslash, so
``\frac`` reaches this module as a form feed and the reader sees ``rac``.

The first is fixable here and is fixed here. The second is only *partly* fixable
here, which is why the composed answer no longer travels through JSON at all --
see :func:`src.agent.llm.ask_prose`. These tests pin both halves, including the
part that cannot be recovered, so nobody later mistakes the safety net for a
solution.
"""

from __future__ import annotations

from src.agent.mathmarkup import EATEN_ESCAPES, balance_inline, to_dollar_math
from src.physics.model import (
    HAMILTONIAN_DISPLAY,
    HAMILTONIAN_INLINE,
    HAMILTONIAN_LATEX,
)

# --------------------------------------------------------------------------
# Delimiters the page does not render
# --------------------------------------------------------------------------


def test_inline_parentheses_become_dollars() -> None:
    assert to_dollar_math(r"the energy is \(E_0 = -1.27\) per site") == (
        "the energy is $E_0 = -1.27$ per site"
    )


def test_bracket_display_becomes_a_centred_line() -> None:
    result = to_dollar_math(r"so \[\hat H = -J \sum_i \hat\sigma^z_i\] follows")
    assert r"$$" in result
    assert r"\hat H = -J \sum_i \hat\sigma^z_i" in result
    assert r"\[" not in result and r"\]" not in result


def test_a_display_environment_is_wrapped() -> None:
    result = to_dollar_math(r"\begin{align} E &= 2 \\ F &= 3 \end{align}")
    assert result.startswith("$$")
    assert result.endswith("$$")


def test_a_list_environment_is_left_alone() -> None:
    # `itemize` is not mathematics, and wrapping it would turn a list into a
    # rendering error. Only the environments named in DISPLAY_ENVIRONMENTS are
    # touched.
    text = r"\begin{itemize}\item one\end{itemize}"
    assert to_dollar_math(text) == text


def test_text_that_already_uses_dollars_is_unchanged() -> None:
    text = "the gap closes at $g = 1$, where $\\hat H$ is critical"
    assert to_dollar_math(text) == text


def test_running_twice_is_the_same_as_running_once() -> None:
    once = to_dollar_math(r"an equation \[E = mc^2\] in prose")
    assert to_dollar_math(once) == once


def test_prose_with_no_mathematics_gains_no_delimiters() -> None:
    # The deliberate limit: guessing that a bare word is mathematics would render
    # prose as symbols, which is worse than the fault being fixed.
    text = "The chain orders below the critical field."
    assert to_dollar_math(text) == text


def test_empty_text_survives() -> None:
    assert to_dollar_math("") == ""


# --------------------------------------------------------------------------
# Backslashes a JSON parser ate
# --------------------------------------------------------------------------


def test_a_form_feed_is_restored_as_a_backslash() -> None:
    # What `{"answer": "\frac{1}{2}"}` decodes to when the model forgets to
    # double the backslash: form feed, then `rac`.
    assert to_dollar_math("$\x0crac{1}{2}$") == r"$\frac{1}{2}$"


def test_every_recoverable_escape_comes_back_as_a_command() -> None:
    for character, command in EATEN_ESCAPES.items():
        assert to_dollar_math(f"x{character}y") == f"x{command}y"


def test_tabs_and_newlines_are_not_second_guessed() -> None:
    r"""``\times`` is lost the same way and is deliberately not recovered.

    A tab is a legitimate character and an eaten ``\times`` is the same byte, so
    restoring it would corrupt text that was never broken. This test exists to
    record that the limitation is a choice: the real fix is upstream, where the
    answer is asked for as prose instead of as JSON.
    """
    assert to_dollar_math("a\tb") == "a\tb"
    assert to_dollar_math("a\nb") == "a\nb"


# --------------------------------------------------------------------------
# The one Hamiltonian
# --------------------------------------------------------------------------


def test_the_hamiltonian_is_written_with_hats() -> None:
    assert r"\hat H" in HAMILTONIAN_LATEX
    assert r"\hat\sigma^z_i" in HAMILTONIAN_LATEX
    assert r"\hat\sigma^x_i" in HAMILTONIAN_LATEX
    # The letter form reads as a gate rather than an operator, and mixing the two
    # notations across the pages is what made this a single constant.
    assert "Z_i" not in HAMILTONIAN_LATEX
    assert "X_i" not in HAMILTONIAN_LATEX


def test_the_delimited_forms_wrap_the_same_string() -> None:
    assert HAMILTONIAN_INLINE == f"${HAMILTONIAN_LATEX}$"
    assert HAMILTONIAN_LATEX in HAMILTONIAN_DISPLAY
    assert HAMILTONIAN_DISPLAY.startswith("$$")


def test_the_hamiltonian_needs_no_rewriting_to_render() -> None:
    # It is already in the delimiters the page renders, so the guard is a no-op on
    # it. A constant that had to be repaired on the way out would be the wrong
    # constant.
    assert to_dollar_math(HAMILTONIAN_INLINE) == HAMILTONIAN_INLINE


# --------------------------------------------------------------------------
# An inline delimiter with no partner
# --------------------------------------------------------------------------
#
# The fault that reached a reader. Quoting `data/corpus/physics-notes/
# pfeuty-exact-solution.md`, whose own dollars are balanced, the model dropped the
# one after `2|J - h|`. Five delimiters instead of six shifts every pairing after
# it by one, so the span opened on the equation and closed on the *next* one: the
# page showed `\epsilon_0 = 2|J-h|.Thegapthereforecloses` as a single equation --
# spaces stripped, because that is what math mode does to prose -- then `h = J` as
# text, then a bare `$`.


QUOTED_WITH_A_LOST_DELIMITER = (
    "(quote from [1]: “The dispersion is minimised at $k = 0$, where "
    "$\\epsilon_0 = 2|J - h|. The gap therefore closes linearly in the distance "
    "from $h = J$ and vanishes there.”)"
)


def test_a_sentence_is_not_swallowed_into_an_unclosed_equation() -> None:
    repaired = to_dollar_math(QUOTED_WITH_A_LOST_DELIMITER)
    # The equation is closed where it ends, so the sentence after it is prose again.
    assert "$\\epsilon_0 = 2|J - h|$. The gap therefore closes" in repaired
    # And the spans that the shifted pairing had broken are spans again.
    assert "$k = 0$" in repaired
    assert "$h = J$" in repaired


def test_no_bare_dollar_is_left_on_the_page() -> None:
    # The visible symptom: the leftover delimiter printed as a dollar sign.
    assert to_dollar_math(QUOTED_WITH_A_LOST_DELIMITER).count("$") % 2 == 0


def test_a_delimiter_that_opened_on_prose_is_dropped_not_closed() -> None:
    # There is no equation to close here, so putting a partner back would set a
    # phrase in mathematics -- the worse failure of the two. It is dropped because it
    # would otherwise pair with the delimiter that follows and swallow the clause.
    assert balance_inline("$ the gap closes linearly and $h = J$") == (
        " the gap closes linearly and $h = J$"
    )


def test_a_dollar_with_nothing_to_pair_with_is_left_where_it_is() -> None:
    # It cannot swallow anything -- the renderer just prints it, which is how this
    # whole fault was noticed -- and removing it would eat the money.
    assert balance_inline("the run cost $0.25 in tokens") == "the run cost $0.25 in tokens"


def test_a_dollar_inside_a_code_span_is_not_a_delimiter() -> None:
    # Markdown decides `...` is code before any maths is looked for, so the `$` in a
    # shell variable must not be paired with a real delimiter -- doing so put the
    # clause between them inside an equation.
    text = "set `EVAL_WORKERS=$N` and read $h = J$ after"
    assert balance_inline(text) == text


def test_text_whose_delimiters_already_pair_is_untouched() -> None:
    # The guarantee that this repair cannot regress an answer that renders today.
    for balanced in (
        "the field $h$ and the coupling $J$ meet at $h = J$",
        "$\\hat\\sigma^z_i \\hat\\sigma^z_{i+1}$ and $\\hat\\sigma^x_i$",
        "the exponent is $z = 1$ because the gap closes linearly in $|h - J|$",
    ):
        assert balance_inline(balanced) == balanced


def test_greek_letters_are_read_as_mathematics_not_prose() -> None:
    # Three lowercase runs in a row is the prose signal, and `\alpha \beta \gamma`
    # is three lowercase runs. A rule that cannot tell them apart would break every
    # span in this project that names more than two symbols.
    assert balance_inline("$\\alpha \\beta \\gamma$ and $x$") == "$\\alpha \\beta \\gamma$ and $x$"


def test_words_inside_a_text_command_are_still_mathematics() -> None:
    # `\text{...}` is where English is legitimate inside an equation, and "ground
    # state energy" is exactly the three-word run the prose signal looks for. Cutting
    # the span there would break a working equation, so the contents are masked
    # before the signals run -- including when the surrounding text is unbalanced and
    # the repair is actually looking for somewhere to cut.
    span = "the value $E_{\\text{ground state energy}} = 2$ per site"
    assert balance_inline(span) == span
    assert balance_inline(f"{span} and $h") == f"{span} and $h"


def test_a_printed_dollar_sign_is_not_a_delimiter() -> None:
    # `\$` is a dollar on the page. Counting it as a delimiter would make a balanced
    # text look odd and invite a repair that is not needed.
    priced = "the run cost \\$0.25 and the field is $h = J$"
    assert balance_inline(priced) == priced


def test_the_repair_leaves_display_blocks_alone() -> None:
    # A `$$` block contains no inline delimiters, and its own are balanced by
    # `balance_display` before this runs.
    text = "the energy is $$\\epsilon_0 = 2|J-h|$$ so it vanishes at $h = J$"
    assert to_dollar_math(text).count("$") == text.count("$")
