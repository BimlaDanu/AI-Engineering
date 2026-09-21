r"""Make the mathematics in an answer render instead of appearing as source.

Streamlit's markdown renders exactly two math delimiters -- ``$...$`` inline and
``$$...$$`` displayed -- and shows everything else verbatim. A model that writes
``\(E_0 = -1.27\)``, which is perfectly ordinary LaTeX, therefore puts backslashes and
parentheses on the page, and an answer about a physics equation is judged on that
line before it is read.

The prompt asks for dollar delimiters (see ``COMPOSE_SYSTEM``). This module is what
happens when the model does not comply: one pure function, run over the text on its
way out, that rewrites the delimiters a renderer ignores into the ones it honours.
An instruction is a request; a rewrite is a guarantee, and the two together are why
no answer shows its own markup.

Deliberately narrow. It moves delimiters and nothing else: it does not add
delimiters around text that has none, because deciding that a bare word is
mathematics is a guess, and a wrong guess renders prose as symbols -- a worse
failure than the one being fixed.
"""

from __future__ import annotations

import re

DISPLAY_ENVIRONMENTS: tuple[str, ...] = ("equation", "equation*", "align", "align*", "gather")
"""LaTeX environments that stand as their own displayed equation.

Written out rather than matched with a wildcard because ``\\begin{itemize}`` is not
mathematics, and wrapping it in ``$$`` would turn a list into a rendering error.
"""

EATEN_ESCAPES: dict[str, str] = {"\x07": r"\a", "\x08": r"\b", "\x0b": r"\v", "\x0c": r"\f"}
r"""Control characters that can only be a LaTeX command a JSON parser swallowed.

``\bigl``, ``\frac`` and ``\vec`` inside a JSON string field are the JSON escapes
``\b``, ``\f`` and ``\v`` unless the model doubles the backslash, and a model
writing mathematics regularly forgets. The reader is then shown ``igl``, ``rac``
and ``ec``.

Only these four are reversible. ``\t`` and ``\n`` are the same bug -- ``\times``
and ``\nabla`` vanish exactly this way -- but a tab is also a legitimate character,
so restoring them would corrupt text that was never broken. That asymmetry is why
the composed answer does not travel through JSON at all (see
:func:`src.agent.llm.ask_prose`); this map is the safety net for the short
model replies that still do.
"""

_BRACKET_DISPLAY = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
_PAREN_INLINE = re.compile(r"\\\((.+?)\\\)", re.DOTALL)
_DOLLAR_DISPLAY = re.compile(r"(?<!\$)\$\$(?!\$)(.+?)(?<!\$)\$\$(?!\$)", re.DOTALL)
_ENVIRONMENT = re.compile(
    r"(\\begin\{(" + "|".join(re.escape(name) for name in DISPLAY_ENVIRONMENTS) + r")\}.+?"
    r"\\end\{\2\})",
    re.DOTALL,
)


def _already_wrapped(text: str, start: int, end: int) -> bool:
    """Whether the span at ``start``:``end`` is inside ``$$`` already.

    Args:
        text: The whole answer.
        start: Where the span begins.
        end: Where the span ends.

    Returns:
        ``True`` when dollars sit immediately outside the span, ignoring
        whitespace. Wrapping twice yields ``$$$$``, which renders as literal
        dollars -- the exact fault this module exists to remove.
    """
    return text[:start].rstrip().endswith("$$") and text[end:].lstrip().startswith("$$")


def balance_display(text: str) -> str:
    r"""Put displayed equations on their own lines and drop a ``$$`` with no partner.

    Two faults with one cause, both of which the reader sees as a broken page rather
    than as a broken equation.

    A model writing a derivation tends to run the delimiters into the sentence --
    ``the mapping is $$A = B$$ and expanding gives $$C = D$$``. **That is the fault
    that reddens a whole answer**, and the delimiters are not what is wrong with it:
    a span sitting mid-paragraph is inline markdown first and mathematics second, so
    the ``_`` in ``\hat\sigma^z_i`` pairs with the next one as emphasis and KaTeX is
    handed a subscript that has become an ``<em>``. Every answer here writes
    subscripts, which is why one loose habit of the narrator's could redden all of
    them at once. On its own lines the span is a block, and it reaches the renderer
    as the model wrote it.

    The second is the one that matters. **An odd number of ``$$`` turns the rest of
    the answer red.** KaTeX colours a parse failure, and an unclosed display span
    fails at the *end of the text*, so one forgotten delimiter halfway through a
    derivation takes every paragraph after it down -- the mathematics, the prose and
    the citations. Dropping the unmatched delimiter costs one equation its rendering
    and leaves the rest of the answer readable, which is the better trade by a wide
    margin. It is also the conservative repair: the alternative, closing the span at
    the end, guesses where the equation stopped, and a guess that is wrong renders
    the prose that followed it as symbols.

    Pairing runs left to right, so the delimiter left over is the last one. That is a
    statement about the pairing and not about the model: a text whose *first*
    delimiter is the missing one pairs off by one, and one sentence renders as an
    equation. Still an improvement on the alternative -- one sentence set in
    mathematics rather than every remaining paragraph in red.

    Args:
        text: The answer, after the other delimiters have been rewritten.

    Returns:
        The same text with each ``$$...$$`` on its own lines and any unpartnered
        ``$$`` removed.

    Examples:
        >>> balance_display("so $$A = B$$ and then")
        'so\n$$\nA = B\n$$\nand then'
        >>> balance_display("the mapping is $$A = B$$ but $$C = D")
        'the mapping is\n$$\nA = B\n$$\nbut C = D'
        >>> balance_display("nothing to do here")
        'nothing to do here'
    """
    if "$$" not in text:
        return text

    def on_its_own_lines(match: re.Match[str]) -> str:
        return f"\n$$\n{match.group(1).strip()}\n$$\n"

    rewritten = _DOLLAR_DISPLAY.sub(on_its_own_lines, text)
    # Every pair was consumed left to right, so an odd count leaves exactly one
    # delimiter over and it is the last one in the text.
    if rewritten.count("$$") % 2:
        head, _, tail = rewritten.rpartition("$$")
        rewritten = f"{head.rstrip()} {tail.lstrip()}"
    # The moved delimiters strand the spaces that used to sit beside them. A trailing
    # one is markdown for a line break, so it would put one where the paragraph did
    # not ask for it; a leading one indents the line that follows the equation.
    return re.sub(r"[ \t]*\n[ \t]*", "\n", rewritten).strip()


_BLANK_LINE = re.compile(r"\n[ \t]*\n")
"""A paragraph break. TeX forbids one inside inline math, so a span containing one
is not a span: its opening delimiter lost its partner."""

_SENTENCE_BREAK = re.compile(r"[.!?][”\'\")]*\s+[A-Z“\"(]|[.!?]\s*\n")
r"""A sentence ending inside a candidate span -- the same evidence, less absolute.

``$\epsilon_0 = 2|J-h|. The gap therefore closes`` is one sentence too many for one
equation. Not a TeX rule, but this project has never written an inline span that
spans a full stop, and the alternative reading renders the sentence as symbols.
"""

_ENGLISH_RUN = re.compile(
    r"(?<![\\\w])[a-z]{2,}\s+(?<![\\\w])[a-z]{2,}\s+(?<![\\\w])[a-z]{2,}(?!\w)"
)
r"""Three plain English words in a row.

The strongest signal that a span is prose, and it has to exclude LaTeX commands to be
usable: ``\alpha \beta \gamma`` is three lowercase runs and is mathematics, which is
what the ``(?<![\\\w])`` guards are for. Real inline mathematics that wants words uses
``\text{...}``; three bare ones mean the delimiter pairing is wrong.
"""


_WORDS_IN_MATHS = re.compile(r"(\\(?:text|textrm|mathrm|mbox|operatorname)\{)([^{}]*)(\})")
r"""The one place inside an equation where English words are legitimate.

``$E_{\text{ground state energy}}$`` is three plain words and is mathematics. The
prose signals cannot see the difference, so the contents are masked out before they
run -- with a filler of the same length, because :func:`_math_ends` reports an offset
into the body and a shifted one would cut the equation in the wrong place.
"""


def _without_words(body: str) -> str:
    r"""The body with the contents of ``\text{...}`` masked, length preserved.

    Args:
        body: The text between two inline delimiters.

    Returns:
        The same string with the inside of every word-carrying LaTeX command replaced
        by ``x`` of equal length, so the prose signals cannot mistake a legitimate
        ``\text{the ground state}`` for a sentence that escaped its delimiter.
    """
    if "\\" not in body:
        return body
    return _WORDS_IN_MATHS.sub(lambda m: m[1] + "x" * len(m[2]) + m[3], body)


def _is_prose(body: str) -> bool:
    """Whether a candidate ``$...$`` body is sentence text rather than mathematics.

    Args:
        body: The text between two inline delimiters, exclusive.

    Returns:
        ``True`` when the span cannot be an equation. Three independent signals, any
        of which is enough, because a false negative renders a sentence as symbols
        and a false positive costs one equation its rendering.
    """
    masked = _without_words(body)
    return bool(
        _BLANK_LINE.search(masked) or _SENTENCE_BREAK.search(masked) or _ENGLISH_RUN.search(masked)
    )


def _inline_dollars(text: str) -> list[int]:
    r"""Where the inline ``$`` delimiters are.

    Args:
        text: The answer, after display math has been balanced.

    Returns:
        The index of every ``$`` that is an inline delimiter. Three things are not
        one, and each would otherwise be paired with a real delimiter and drag the
        prose between them into an equation: the contents of a ``$$`` display block,
        ``\$``, which is a printed dollar sign, and anything inside a ``\`` code
        span -- ``\`EVAL_WORKERS=$N\``` names a shell variable, and markdown has
        already decided that span is code before any maths is looked for.
    """
    positions: list[int] = []
    index, in_display = 0, False
    while index < len(text):
        if text[index] == "\\" and index + 1 < len(text):
            index += 2  # an escape: whatever follows is not a delimiter
            continue
        if text[index] == "`":
            closing = text.find("`", index + 1)
            index = len(text) if closing == -1 else closing + 1
            continue
        if text.startswith("$$", index):
            in_display = not in_display
            index += 2
            continue
        if text[index] == "$" and not in_display:
            positions.append(index)
        index += 1
    return positions


def _math_ends(body: str) -> int | None:
    r"""Where the mathematics stops inside a body that runs on into prose.

    Args:
        body: The text between two inline delimiters, known to contain prose.

    Returns:
        The offset just past the last mathematical character, or ``None`` when the
        body is prose from its first character and there is no equation to close.
        Taken as the earliest of the three prose signals, so the cut lands at the
        full stop in ``\epsilon_0 = 2|J - h|. The gap therefore closes``.
    """
    masked = _without_words(body)
    found = [
        match.start()
        for match in (
            _BLANK_LINE.search(masked),
            _SENTENCE_BREAK.search(masked),
            _ENGLISH_RUN.search(masked),
        )
        if match is not None
    ]
    if not found:
        return None
    mathematics = body[: min(found)].rstrip()
    return len(mathematics) or None


def balance_inline(text: str) -> str:
    r"""Close the inline ``$`` that lost its partner, before it swallows a sentence.

    :func:`balance_display` makes this argument for ``$$`` at length. The same fault
    with one dollar is worse, and it is the one that reaches the reader: an odd number
    of inline delimiters does not fail loudly, it **shifts every pairing after it by
    one**, so each span opens on an equation and closes on the *next* one, and the
    sentence in between is set in mathematics -- spaces stripped, letters italicised --
    with the leftover delimiter printed on the page as a dollar sign.

    Observed, quoting a note whose own source is balanced::

        “The dispersion is minimised at $k = 0$, where $\epsilon_0 = 2|J - h|.
        The gap therefore closes linearly in the distance from $h = J$ and vanishes.”

    Five delimiters. The reader got ``\epsilon_0 = 2|J-h|.Thegapthereforecloses`` set
    as one equation, ``h = J`` as prose, and a bare ``$`` before "and vanishes".

    The repair pairs left to right and tests each candidate body for prose
    (:func:`_is_prose`). A body that is part equation and part sentence gets the
    **missing delimiter put back** where the mathematics stops -- at the full stop
    above -- which returns the equation, the sentence and the two spans after it all
    at once. Only when the body is prose from its first character, with no equation to
    close, is the delimiter dropped instead.

    Closing the span is a guess about where the equation ended, and it is a safe one
    in a way that :func:`balance_display`'s alternative is not: cutting too early
    leaves mathematics showing as source, which is the failure this module already
    tolerates, whereas cutting too late is impossible -- the cut is bounded by the
    delimiter that follows. It can never put prose inside an equation.

    **Nothing happens when the count is even.** A text that pairs off cannot be
    mispaired by this function, so an answer that renders today renders identically
    tomorrow. That is the whole guarantee: this is a repair, not a policy.

    Args:
        text: The answer, after display math has been balanced.

    Returns:
        The same text with each unpartnered inline delimiter closed where its
        equation ends, or dropped when it opened on prose. Unchanged when the
        delimiters already pair off.

    Examples:
        >>> balance_inline("at $k = 0$, where $E = 2|J - h|. The gap closes at $h = J$")
        'at $k = 0$, where $E = 2|J - h|$. The gap closes at $h = J$'
        >>> balance_inline("the field $h$ and the coupling $J$")
        'the field $h$ and the coupling $J$'
        >>> balance_inline("the run cost $0.25 in tokens")
        'the run cost $0.25 in tokens'
    """
    dollars = _inline_dollars(text)
    if len(dollars) % 2 == 0:
        return text

    strays: set[int] = set()
    closings: set[int] = set()
    cursor = 0
    while cursor < len(dollars):
        opening = dollars[cursor]
        if cursor + 1 == len(dollars):
            # Left alone, deliberately. A delimiter with nothing to pair with cannot
            # swallow anything -- the renderer prints it, which is how this fault was
            # spotted in the first place -- and removing it would eat the dollar sign
            # in "the run cost $0.25", where there was never any mathematics at all.
            break
        body = text[opening + 1 : dollars[cursor + 1]]
        if _is_prose(body):
            ends = _math_ends(body)
            if ends is None:
                strays.add(opening)  # opened on prose; there is nothing to close
            else:
                closings.add(opening + 1 + ends)
            cursor += 1
        else:
            cursor += 2  # a plausible span; both delimiters stay as they are
        continue

    if not strays and not closings:
        return text

    repaired: list[str] = []
    for at, character in enumerate(text):
        if at in closings:
            repaired.append("$")
        if at not in strays:
            repaired.append(character)
    if len(text) in closings:
        repaired.append("$")
    return "".join(repaired)


def to_dollar_math(text: str) -> str:
    r"""Rewrite LaTeX delimiters into the ones a markdown renderer honours.

    Args:
        text: The answer as the model wrote it.

    Returns:
        The same text with ``\[...\]`` and display environments as ``$$...$$`` on
        their own lines, and ``\(...\)`` inline as ``$...$``. Text that already uses
        dollars is returned unchanged, so running this twice is the same as running
        it once.

    Examples:
        >>> to_dollar_math(r"the energy is \(E_0\)")
        'the energy is $E_0$'
        >>> to_dollar_math("a \x0crac{1}{2} chain")
        'a \\frac{1}{2} chain'
    """
    if not text:
        return text

    for character, command in EATEN_ESCAPES.items():
        if character in text:
            text = text.replace(character, command)

    def display(match: re.Match[str]) -> str:
        return f"\n$$\n{match.group(1).strip()}\n$$\n"

    def inline(match: re.Match[str]) -> str:
        return f"${match.group(1).strip()}$"

    def environment(match: re.Match[str]) -> str:
        body = match.group(1)
        if _already_wrapped(text, match.start(), match.end()):
            return body
        return f"\n$$\n{body}\n$$\n"

    rewritten = _BRACKET_DISPLAY.sub(display, text)
    rewritten = _PAREN_INLINE.sub(inline, rewritten)
    rewritten = _ENVIRONMENT.sub(environment, rewritten)
    # Last, so that it also tidies the blocks the rewrites above have just created.
    rewritten = balance_display(rewritten)
    # After the display pass, so that a $$ it moved or dropped cannot be counted as
    # an inline delimiter here.
    rewritten = balance_inline(rewritten)
    # Collapse the blank lines the wrapping can leave behind, so a displayed
    # equation is separated from its paragraph by one break rather than three.
    return re.sub(r"\n{3,}", "\n\n", rewritten).strip()
