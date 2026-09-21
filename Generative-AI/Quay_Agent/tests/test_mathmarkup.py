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

from src.agent.mathmarkup import EATEN_ESCAPES, balance_inline, clip, to_dollar_math
from src.physics.model import (
    HAMILTONIAN_CHAIN_LATEX,
    HAMILTONIAN_CHAIN_WITH_LONGITUDINAL_LATEX,
    HAMILTONIAN_DISPLAY,
    HAMILTONIAN_INLINE,
    HAMILTONIAN_LATEX,
    HAMILTONIAN_WITH_LONGITUDINAL_LATEX,
    hamiltonian_display,
    hamiltonian_latex,
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


def test_the_displayed_hamiltonian_carries_the_two_terms_a_reader_can_change() -> None:
    # Two terms, not three. The equation on a page says what problem is being
    # solved, and this project solves a problem with one dimensionless knob, h/J.
    # A `g` that is zero everywhere by default, moves no number on any screen, and
    # is reachable only from an advanced control reads as a third free parameter --
    # which is what a reader counts to decide how hard the problem is.
    assert r"-J \sum_{\langle ij \rangle} \hat\sigma^z_i \hat\sigma^z_j" in HAMILTONIAN_LATEX
    assert r"- h \sum_i \hat\sigma^x_i" in HAMILTONIAN_LATEX
    # Matched as the term rather than as the letter: "g" also lives inside \sigma.
    assert r"- g \sum" not in HAMILTONIAN_LATEX
    # The diagonal term comes first: it commutes with itself across sites, and the
    # transverse term is the one that does not.
    index = HAMILTONIAN_LATEX.index
    assert index(r"-J \sum") < index(r"- h \sum")


def test_the_neighbour_sum_covers_every_shape_and_a_line_is_written_out() -> None:
    # The application solves lines, square lattices and triangular lattices, so the
    # equation a reader meets sums over neighbouring pairs rather than over i and
    # i+1. Writing the one-dimensional sum as the general one would make every 2D
    # answer describe a problem nobody asked about.
    assert r"\langle ij \rangle" in HAMILTONIAN_LATEX
    assert r"\sigma^z_{i+1}" not in HAMILTONIAN_LATEX

    # One dimension keeps its own form, because there the neighbour sum can be
    # written out with limits -- and because it is the only shape with a
    # closed-form answer, so it is the case a reader is walked through.
    assert r"-J \sum_{i=1}^{L} \hat\sigma^z_i \hat\sigma^z_{i+1}" in HAMILTONIAN_CHAIN_LATEX
    assert hamiltonian_latex(geometry="chain") == HAMILTONIAN_CHAIN_LATEX

    # A shape with no closed form gets the general form, not the line's.
    assert hamiltonian_latex(geometry="square") == HAMILTONIAN_LATEX
    assert hamiltonian_latex(geometry="triangular") == HAMILTONIAN_LATEX
    # And so does "no shape named", which is how the model itself is described.
    assert hamiltonian_latex() == HAMILTONIAN_LATEX

    # All four corners of (shape, longitudinal field) are reachable and distinct.
    forms = {
        hamiltonian_latex(0.0, None),
        hamiltonian_latex(0.0, "chain"),
        hamiltonian_latex(0.4, None),
        hamiltonian_latex(0.4, "chain"),
    }
    assert len(forms) == 4


def test_the_three_term_form_is_kept_and_is_what_a_non_zero_g_renders() -> None:
    # The argument for showing the longitudinal term is still true and is still
    # served: at g = 0 this chain is integrable, so every honest verdict about it is
    # "no advantage", and g != 0 is what makes the feasibility question open. So the
    # term is *reachable* rather than removed, and the surfaces switch to it the
    # moment it is switched on.
    assert r"- g \sum_i \hat\sigma^z_i" in HAMILTONIAN_WITH_LONGITUDINAL_LATEX
    assert r"- h \sum_i \hat\sigma^x_i" in HAMILTONIAN_WITH_LONGITUDINAL_LATEX
    # And the line's own version of it, for a one-dimensional run with g switched on.
    assert r"- g \sum_{i=1}^{L} \hat\sigma^z_i" in HAMILTONIAN_CHAIN_WITH_LONGITUDINAL_LATEX
    assert hamiltonian_latex(0.4, "chain") == HAMILTONIAN_CHAIN_WITH_LONGITUDINAL_LATEX
    # Both classical terms before the transverse one, for the reason above.
    index = HAMILTONIAN_WITH_LONGITUDINAL_LATEX.index
    assert index(r"- g \sum") < index(r"- h \sum")

    assert hamiltonian_latex(0.0) == HAMILTONIAN_LATEX
    assert hamiltonian_latex(0.4) == HAMILTONIAN_WITH_LONGITUDINAL_LATEX
    assert hamiltonian_display(0.4).startswith("$$")
    assert HAMILTONIAN_WITH_LONGITUDINAL_LATEX in hamiltonian_display(0.4)


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


# --------------------------------------------------------------------------
# Clipping a passage without breaking it
# --------------------------------------------------------------------------
#
# Citations were cut with `text[:400]`. That ends mid-word, which reads as a broken
# file rather than a quotation, and it ends inside `$...$` often enough to matter:
# an unmatched dollar makes a markdown renderer swallow everything up to the next
# one, so a stray delimiter in the third citation can hide the paragraph above it.


def test_a_passage_that_already_fits_is_returned_untouched() -> None:
    assert clip("short enough", 40) == "short enough"


def test_the_cut_lands_on_a_word_boundary() -> None:
    cut = clip("the transverse field drives the transition", 20)
    assert cut.endswith("…")
    assert not cut[:-1].endswith(" ")
    assert cut[:-1] in "the transverse field drives the transition"


def test_the_cut_never_leaves_half_an_equation() -> None:
    # The failure worth having a function for. Every prefix length is tried, and
    # none of them may leave an odd number of inline delimiters behind.
    passage = "the energy is $E_0 = -1.27$ per site and the gap is $\\Delta = 2|J - h|$ exactly"
    for limit in range(1, len(passage) + 5):
        assert clip(passage, limit).count("$") % 2 == 0, limit


def test_a_clipped_equation_is_dropped_rather_than_closed() -> None:
    # Retreating out of the equation, not inventing a closing delimiter: half an
    # equation is not one, and a fabricated close would render arithmetic nobody
    # wrote and attribute it to a cited source.
    assert "E_0" not in clip("the energy is $E_0 = -1.27$ per site", 22)


def test_clipping_is_stable_under_being_run_twice() -> None:
    once = clip("a passage worth quoting at some length", 20)
    assert clip(once, 20) == once


def test_no_prose_prompt_writes_the_longitudinal_term_into_the_hamiltonian() -> None:
    r"""The prompts must show the same two-term chain the pages do.

    This leaked three times before it was caught, each time somewhere different --
    ``FormalModel.label()``, the method race's caption, and then here, in the opening
    line of the two prompts that write prose. The symptom was always the reader being
    shown a term the interface does not have: an answer opening with
    ``H = -J ... - g sum_i sigma^z_i - h ...`` beneath a page whose Hamiltonian has two
    terms, or a closing paragraph offering to "add a field along the coupling
    direction" as though the reader had one to add.

    A prompt is the hardest of the three to notice, because nothing renders it -- it is
    only ever visible in what the model chooses to echo back, which is intermittently.
    Hence a test rather than a convention.
    """
    from src.agent.drafting import DRAFT_SYSTEM
    from src.agent.explaining import EXPLAIN_SYSTEM

    for name, prompt in (("EXPLAIN_SYSTEM", EXPLAIN_SYSTEM), ("DRAFT_SYSTEM", DRAFT_SYSTEM)):
        assert "- g sum_i sigma^z_i" not in prompt, f"{name} writes the longitudinal term"
        assert r"- g \sum_i" not in prompt, f"{name} writes the longitudinal term in LaTeX"
        # Present as a fact about the specification, which is the honest version: the
        # term exists and is off. What must not happen is it being written into the
        # Hamiltonian the reader is shown.
        assert "-g sum_i sigma^z_i" in prompt, f"{name} no longer explains why g is absent"


def test_no_label_names_the_longitudinal_field_when_it_is_switched_off() -> None:
    """Every short chain identifier omits ``g`` at zero, and names it otherwise.

    Two of these print into captions and composed answers, and both got this wrong
    independently -- ``FormalModel.label`` put ``g=0`` into every sentence naming the
    chain, and ``MethodRace.label`` put it under every race figure. The rule is one
    sentence long and was still worth a test, because it has to hold in two classes
    that share no base and no module.
    """
    from src.agent.state import FormalModel
    from src.physics.quantum.method_race import MethodRace

    off = FormalModel(n_sites=8, coupling=1.0, transverse_field=1.0, longitudinal_field=0.0)
    on = FormalModel(n_sites=8, coupling=1.0, transverse_field=1.0, longitudinal_field=0.4)
    assert "g=" not in off.label()
    assert "g=0.4" in on.label()

    race_off = MethodRace(
        n_sites=8,
        depth=2,
        coupling=1.0,
        transverse_field=1.0,
        longitudinal_field=0.0,
        boundary="open",
        runs=(),
    )
    race_on = MethodRace(
        n_sites=8,
        depth=2,
        coupling=1.0,
        transverse_field=1.0,
        longitudinal_field=0.4,
        boundary="open",
        runs=(),
    )
    assert "g=" not in race_off.label()
    assert "g=0.4" in race_on.label()


def test_the_reserved_knob_can_be_asked_for_without_the_letter() -> None:
    """The follow-up that offers it must set what its own words say.

    It used to read *a longitudinal field of g = 0.4* and was reworded to drop the
    symbol, which the chat page does not show. That rewording is only safe because
    :func:`src.agent.reading.read_longitudinal` understands the words as well as the
    symbol -- without it the button would have set nothing, said nothing about setting
    nothing, and returned the same exactly-solvable verdict it was offered to escape.

    This test is the reason the rewording is not a silent regression.
    """
    from src.agent.reading import read_longitudinal

    assert read_longitudinal("a 10-spin chain with g = 0.4") == 0.4
    assert (
        read_longitudinal(
            "What about a 10-spin chain with a field of 0.4 along the coupling direction?"
        )
        == 0.4
    )
    assert read_longitudinal("a 10-spin chain with a longitudinal field of 0.4") == 0.4
    assert read_longitudinal("an 8-spin chain at criticality") is None
