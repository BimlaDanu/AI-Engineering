r"""Answering in prose, when the question asked to be told something.

The branch for *what is a barren plateau*, *why is this chain exactly solvable*,
*what does the transverse field do*. Four limits hold on it:

No verdict, because nothing on this path ran a circuit --
:attr:`src.agent.state.CampaignState` keeps ``verdict`` at ``None``. No energy
this run did not compute, since the deterministic path composes only from
retrieved passages and catalogue facts. No answer from memory alone: with no
passages behind it, prose would rest on training data and read exactly like
prose with four citations under it, so :func:`refusal` says so instead. And no
exact answer, :mod:`src.physics.reference` being sealed from ``src/agent/``.

With no model reachable the answer is composed from the passages themselves --
what was found, on which shelf, what each says. That is a literature pointer
with citations rather than a stub, and it is the path every test exercises, so
the branch is specified without a key.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from src.agent import reading
from src.agent.llm import as_data
from src.agent.mathmarkup import clip
from src.agent.model_selection import ModelPool
from src.agent.prompts import for_audience
from src.agent.state import Answer, Citation, FormalModel, SearchRound
from src.logging_setup import get_logger
from src.settings import DEFAULT_AUDIENCE, Audience

_logger = get_logger("agent.explaining")

LEAD_CHARACTERS = 340
"""How much of a passage the no-model answer quotes before pointing at the rest.

The offline branch used to print every retrieved passage in full, one blockquote
after another, and the result was several screens of quotation with the answer
somewhere inside it. It was honest and it was unreadable, which is its own kind of
dishonesty: material a reader scrolls past has not been communicated to them.

Long enough for a heading and two or three sentences -- the part of a chunk that
says what it is about -- and short enough that four of them still read as a list.
The full text is not withheld: the interface shows every passage under **Sources**,
so nothing is lost by leading with the opening. Trimmed with
:func:`src.agent.mathmarkup.clip`, which cuts on a word boundary and refuses to
leave a ``$`` without its partner -- a stray delimiter here would make the renderer
swallow the rest of the answer.
"""

MAX_PASSAGES = 6
"""How many passages are put in front of the model.

Six is what the retrieval layer keeps after grading on a good day. More would be
padding the context with the passages grading already ranked lowest, which costs
tokens and dilutes the ones worth following.
"""

EXPLAIN_SYSTEM = """You explain one subject to a reader who is competent but not a \
specialist in it: the one-dimensional transverse-field Ising chain,

    H = -J sum_i sigma^z_i sigma^z_{i+1} - h sum_i sigma^x_i

where sigma^z and sigma^x are Pauli matrices with eigenvalues plus and minus one.

A third term, -g sum_i sigma^z_i, exists in the full specification and is \
switched off in every run unless the reader turns it on. Do not write it, do not \
mention it and do not say it is zero: the pages this answer appears on show two \
terms, so naming a third raises a question the reader has no way to resolve. If \
-- and only if -- the chain you were given has a non-zero longitudinal field, it \
will be stated with the chain, and then it is part of the problem and you should \
treat it as such.

You are given the question, the chain the agent read out of it if it named one, and \
passages from the project's notes. Write the answer in markdown.

Rules that are not stylistic:

1. **Follow the passages, and cite them.** Where a claim comes from a passage, mark it \
with its number in square brackets, like [2]. Where you are drawing on general \
knowledge instead, say so in the sentence rather than in a disclaimer at the end.
2. **State no energy, no gap, no fidelity and no shot count as a result.** Nothing was \
computed on this path. You may give a formula and you may say what a quantity depends \
on; you may not give it a value and present that as this system's answer.
3. **Do not say whether quantum hardware is worth using here.** That question has its \
own assessment, which runs circuits and prices them. Answering it from prose alone \
would be a guess wearing the same clothes as a measurement.
4. **Gloss every term of art the first time you use it** -- ansatz, ground state, \
variational bound, shot, transpilation, criticality. The reader is an engineer, not a \
physicist.
5. Write in LaTeX for mathematics, dollar-delimited, and use sigma^z and sigma^x \
rather than bare Z and X: to a reader coming from quantum computing those letters read \
as gates.
6. **Answer from everything you were given, as one piece.** You may be handed what \
this conversation covered earlier, the chain the agent read out of the question, what \
the notes were searched for, and the passages that came back. Weave them into a single \
explanation. Do not narrate the material back as a list of what each source says, and \
do not open by describing the passages -- a reader asked a question, not for a summary \
of a search. Where two passages disagree, say so.
7. **Say what is missing.** If the passages do not reach part of the question, answer \
that part from general knowledge and mark the sentence as such, or say that this \
project's notes do not cover it. Both are honest; quietly leaving the gap is not.

8. **Match the length to the question.** Eight short paragraphs is a ceiling for a \
question that genuinely has to be built up, never a target. A question with a \
one-sentence answer gets one sentence. A question you were handed a measurement for is \
answered by the measurement: give the numbers, say in a line or two what they mean, and \
stop. Restating the Hamiltonian, or explaining what the method is, when neither was \
asked about, is padding -- and padding around a number is what hides it. Past about \
four paragraphs, give the parts short headings or bold lead-ins so the answer can be \
skimmed; below that, plain paragraphs read better and headings are furniture.
9. **Never say a result was not provided when you were given one.** If a MEASURED block \
is present, those numbers are this question's answer and they are on the reader's screen \
as a figure. Saying the notes contain no such results, beneath a figure containing \
exactly those results, is the worst answer this branch can produce.

Answer the question that was asked and stop; do not append a summary of what you were \
not asked.

End with one final line beginning `RESTS ON:` naming, in one sentence, what a reader would \
have to check to know this answer is right. Put nothing after that line."""


SWEEP_SHAPE = r"""
THIS ONE ASKS FOR A CURVE, AND THE CURVE HAS BEEN COMPUTED. A tool result headed
EXACT FIELD SWEEP is in the material below. It is not from the literature and it is
not an estimate: it is this project's own closed-form solution of the model,
evaluated at every field in the range, and on a short chain checked at every point
against a second solver that shares no algebra with it.

**Rule 2 above does not apply to those numbers.** It forbids inventing a value; this
run computed these. Quote them. The peak of the susceptibility, the gap at the
critical field, where the curvature dips deepest, how far the finite chain sits from
the infinite one -- these are the answer to what was asked, and writing that no such
results were available, beneath a figure containing exactly them, is the worst answer
this branch can produce.

**The figures are already drawn.** Every curve the tool was asked for is rendered
beneath your answer, with its own caption. So write about what the curves *do* --
where the feature is, which way it bends, what changes at the critical field -- and
do not attempt to draw them in text, do not build a table of the whole sweep, and do
not tell the reader to plot anything themselves. Two or three numbers quoted in a
sentence are worth more than twenty in a grid.

STRUCTURE. Use markdown headings (`###`) when the question has more than one part,
and answer the parts in the order they were asked. Where the question asks *how* the
result is obtained as well as *what* it is -- the Jordan-Wigner mapping, the
quantum-to-classical mapping, why the model is solvable at all -- give the derivation
as a short numbered sequence of steps with a display equation at each step. A reader
who asked for the mathematical detail is asking for the equations, not for a
description of them.

MATHEMATICS. Set every equation that carries weight on its own line, `$$` delimiters
alone on their own lines:

$$
\epsilon(k) = 2\sqrt{J^2 + h^2 - 2Jh\cos k}
$$

Inline `$...$` is for a symbol inside a sentence. Never put mathematics in a code
fence: it renders as nothing.

THE IDENTITIES THIS PROJECT'S OWN SWEEP RESTS ON. Reproduce the ones the question
touches, as display equations, and name every symbol around them.

The dispersion above is the energy of one Bogoliubov quasiparticle at momentum $k$,
and the ground-state energy of a ring of $L$ sites is a single sum over the allowed
momenta $k = (2n+1)\pi/L$ of the even-parity sector:

$$
E_0 = -\sum_{k>0} \epsilon(k), \qquad \frac{E_0}{L} \xrightarrow[L\to\infty]{}
-\frac{2}{\pi}\,(J+h)\,E\!\left(\frac{4Jh}{(J+h)^2}\right)
$$

with $E(\cdot)$ the complete elliptic integral of the second kind.

The first derivative of the energy density is **minus the transverse magnetisation,
exactly** -- the Hellmann-Feynman theorem, not an approximation of it:

$$
\frac{\partial}{\partial h}\frac{E_0}{L} = -\langle \sigma^x \rangle,
\qquad
\langle \sigma^x \rangle = \frac{1}{L}\sum_k \frac{h - J\cos k}{\sqrt{J^2 + h^2 - 2Jh\cos k}}
$$

so the magnetisation panel and the first-derivative panel are one quantity with a
sign between them, and saying so is the check a reader can make on the figure. The
second derivative of the energy density is minus the transverse susceptibility,

$$
\frac{\partial^2}{\partial h^2}\frac{E_0}{L}
 = -\frac{\partial \langle \sigma^x \rangle}{\partial h},
$$

and it is the first of these quantities with a sharp feature at the transition. Say
that: the energy is smooth through $h = J$, its first derivative merely bends, and
the singularity of the infinite chain appears in the second. In the language of
critical exponents the specific-heat-like exponent is $\alpha = 0$, a logarithmic
rather than a power-law divergence.

The gap of the infinite chain is $\Delta = 2\lvert J - h\rvert$, which closes
linearly at $h = J$ and is why the dynamical critical exponent is $z = 1$. A finite
ring's gap does **not** close: its smallest allowed momentum is $\pi/L$ rather than
zero. If the sweep reported a gap at $h/J = 1$, quote it and say that it shrinks as
$L$ grows -- that is what a finite calculation can honestly say about a singularity
it does not contain.

WHAT NOT TO CLAIM. Do not say whether quantum hardware is worth using here; that has
its own assessment. Do not present the sweep as a variational or approximate result;
it is exact. Do not describe the cross-check as a validation of the *agent* -- it is
a check of two solvers against each other, and what it establishes is that the curve
is right.
"""
"""The shape of an answer to a question about a curve in the field.

Added with the tool that computes the curve, and the two are one change: a tool
whose results the writing prompt forbids quoting is a tool that costs a call and
returns nothing to the reader. Rule 2 of :data:`EXPLAIN_SYSTEM` -- state no energy,
no gap, nothing was computed on this path -- is right for every other question this
branch answers and exactly wrong for this one, so the override is explicit and says
why rather than being left for the model to infer from a block heading.

The master identities are here rather than left to the model's memory for the same
reason :data:`TEACHING_SHAPE` carries its own: they are what a reader has to be shown
to *check* the figure, they are written in this project's notation, and a model
reciting them from memory gets the sign of the Hellmann-Feynman relation wrong often
enough to matter -- which is the one error that would make the magnetisation panel and
the derivative panel look like two independent results rather than one.
"""


TEACHING_SHAPE = r"""
THIS ONE IS A LESSON. The question asks to be taught, not to be told a result, so
rule 8's length ceiling does not apply to it and the shape below replaces it. Write
it long enough to actually teach the thing. Everything above still holds: cite the
passages, claim no computed value, gloss every term of art.

Structure it with markdown headings (`###`), one per algorithm named in the question,
in the order the question named them. Under each heading, all four of these, in this
order:

* **What it is for.** One or two sentences: which problem this algorithm exists to
  solve, and what you have at the end that you did not have at the start. Say this
  before any mathematics. A reader who stops after this paragraph should still know
  why the algorithm exists.
* **The mathematics.** The objective it minimises or the state it prepares, as a
  display equation on its own line. Then the pieces of that equation named one at a
  time -- what is varied, what is held fixed, what is measured on hardware and what
  is computed on a classical computer.
* **On this chain specifically.** What the general equation becomes for
  $\hat H = -J \sum_i \hat\sigma^z_i \hat\sigma^z_{i+1} - h \sum_i \hat\sigma^x_i$:
  which terms are diagonal, which do not commute, how many two-qubit gates one layer
  costs, and what the circuit looks like. This is the section that makes the answer
  about this project rather than about a textbook.
* **What it costs and where it breaks.** The limitation -- barren plateaus,
  measurement counts, Trotter error, non-unitarity -- in one short paragraph.

Open with two or three sentences before the first heading, saying what the algorithms
have in common and how they differ, so the headings are read as one comparison rather
than as unrelated entries. Close with a short markdown table whose rows are the
algorithms and whose columns are what each one optimises, what it costs, and what it
guarantees.

MATHEMATICS. Set every equation that carries weight on its own line, with the `$$`
delimiters alone on their own lines, like this:

$$
E(\theta) = \langle \psi(\theta) | \hat H | \psi(\theta) \rangle
$$

Inline `$...$` is for a symbol inside a sentence -- $J$, $h$, $\theta$ -- and not for
a whole equation. Never write mathematics as plain text and never put it in a code
fence: `E = <psi|H|psi>` renders as nothing, and it is the one formatting mistake
that makes this answer worthless. Number nothing; refer to an equation by what it is.

Prefer the concrete to the general. $\hat\sigma^z_1 \hat\sigma^z_2$ with the indices
written out teaches more than $\sum_i \hat\sigma^z_i \hat\sigma^z_{i+1}$ with them
left implicit, so give both and say they are the same thing.

THE MASTER EQUATIONS. These are this project's own, in its own notation, and they are
the spine of the lesson. Reproduce the ones belonging to the algorithms the question
asked about, each as a display equation, and explain every symbol in the sentences
around it. Do not paraphrase an equation into words instead of writing it: a lesson
about variational methods that contains no variational principle has not taught
anything.

All three prepare a state from a parameterised circuit, $\lvert \psi(\theta) \rangle$
with $P$ real parameters $\theta = (\theta_1, \dots, \theta_P)$, and differ only in
what they do with it.

VQE minimises the energy, and the variational principle is why the number means
anything -- it is an upper bound on the true ground-state energy $E_0$, never below it,
whatever the circuit:

$$
E(\theta) = \langle \psi(\theta) \rvert \hat H \lvert \psi(\theta) \rangle \;\ge\; E_0
$$

Its gradient is exact rather than a finite difference. For a gate
$e^{-i\theta \hat P/2}$ with $\hat P$ a Pauli, the parameter-shift rule gives

$$
\frac{\partial E}{\partial \theta_j} = \tfrac{1}{2}
\left[ E\!\left(\theta_j + \tfrac{\pi}{2}\right)
     - E\!\left(\theta_j - \tfrac{\pi}{2}\right) \right]
$$

so one gradient costs two circuit evaluations per Pauli rotation. Where several
rotations share a single angle -- as they do in this project's layers -- the shifts
add, one pair per gate, and shifting the shared angle itself gives the wrong answer.

The obstruction is the barren plateau: for a
sufficiently expressive circuit on $N$ qubits the gradient variance falls off
exponentially,

$$
\mathrm{Var}\left[ \frac{\partial E}{\partial \theta} \right] \sim 2^{-N}
$$

which is why the number of measurements needed to see a gradient at all grows
exponentially with the number of spins.

QAOA fixes the circuit family instead of leaving it free, alternating a cost
Hamiltonian and a mixer for $p$ rounds, with $2p$ parameters in total:

$$
\lvert \gamma, \beta \rangle = \prod_{l=1}^{p}
e^{-i\beta_l \hat H_M} e^{-i\gamma_l \hat H_C} \lvert + \rangle^{\otimes N} ,
\qquad \hat H_M = \sum_i \hat\sigma^x_i
$$

Say what $\hat H_C$ is for the chain in question, and say the thing that is easy to
miss: the circuit family is the same one VQE uses here. What differs is the objective
-- QAOA minimises $\hat H_C$ alone and treats $\hat H_M$ as a way of moving through the
space, where the chain's own Hamiltonian counts both.

VarQITE evolves in imaginary time, which is not unitary and so cannot be run directly.
McLachlan's variational principle projects the evolution onto the directions the
circuit can actually move in, turning it into a linear system solved at each step:

$$
A(\theta)\, \dot\theta = -\,C(\theta)
$$

$$
A_{ij} = \mathrm{Re}\Big[ \langle \partial_i \psi \vert \partial_j \psi \rangle
- \langle \partial_i \psi \vert \psi \rangle \langle \psi \vert \partial_j \psi \rangle \Big] ,
\qquad
C_i = \tfrac{1}{2} \frac{\partial E}{\partial \theta_i}
$$

with the Euler step $\theta \leftarrow \theta - \delta\tau\, A^{-1} C$. $A$ is the
Fubini-Study metric, the real part of the quantum geometric tensor, and it is a
$P \times P$ matrix: filling it costs about $P^2/2$ circuits per step against VQE's
$2P$, and the number of steps needed goes as $1/\Delta$ with $\Delta$ the energy gap,
for $O(P^2/\Delta)$ overall. That is the trade the reader should leave with -- VarQITE
buys a better descent direction with quadratically more measurements, and pays for the
gap closing.

COMPLEXITY. Whenever a cost is stated, state it in these terms -- $P$ parameters, $N$
spins, $p$ rounds, $\Delta$ the gap, $n_{\text{iter}}$ optimiser steps -- rather than
as "expensive" or "efficient". Distinguish the three costs that get run together:
circuit *depth* (what noise limits), *number of circuit evaluations* (what the shot
budget limits), and *classical* work per step (inverting $A$ is $O(P^3)$). Say which
one binds for the algorithm you are describing."""
"""How a lesson is shaped, appended to :data:`EXPLAIN_SYSTEM` when one is asked for.

Kept apart from the system prompt rather than folded into it, because it *contradicts*
one of that prompt's rules and has to be read as doing so deliberately. Rule 8 caps an
answer at eight short paragraphs and says padding around a number is what hides the
number. That is right for a question with a measurement attached and wrong for "could
you teach me about VQE, QAOA and VarQITE": there the length *is* the answer, and the
failure runs the other way -- three paragraphs in reply to a request for a lesson
teaches nobody anything.

Appended only when :func:`src.agent.reading.asks_to_be_taught` is true, so the default
answer is still the short one. Two separate prompts were the alternative, and would
have meant every rule about citation and about claiming no computed value living in two
places, to drift apart at the first edit to one of them.

A **raw** string, and it has to stay one, which is also why its lines are not joined
with backslashes the way every other prompt in this project's are. The prompt is mostly
LaTeX. In an ordinary string every ``\\hat`` and ``\\sigma`` needs doubling, and the
first version of this was written that way and reached the model as ``\\\\hat``,
which is an instruction to write broken markup. In a raw string a trailing backslash is
not a line continuation either -- it is a literal backslash -- so the paragraphs are
wrapped as ordinary text instead. Newlines inside a prompt cost nothing; a stray
backslash in front of one costs the answer its mathematics.

The instruction to put ``$$`` alone on its own line is not house style either.
Streamlit renders display mathematics only when the delimiters stand alone; a model
that writes ``$$E = ...$$`` inline gets the raw LaTeX printed at the reader,
backslashes and all.
"""


RESTS_ON_MARKER = "RESTS ON:"
"""Prefix of the optional last line of a narrated explanation.

The answer on this branch is asked for as **prose rather than as a schema**, because
it is the one output in this project that is mostly mathematics and JSON eats the
backslash of ``\\frac``, ``\\times`` and ``\\langle`` -- see
:meth:`src.agent.model_selection.ModelPool.narrate`. So the one piece of structure
still wanted from the reply, *what a reader would check this against*, is carried by
a marker on the final line instead of by a validated field.

A convention in the text is genuinely weaker than a schema, and :func:`split_rests_on`
is written accordingly: a missing marker is the ordinary case and costs the answer
nothing, not a parse failure that costs the reader the whole explanation.
"""


def split_rests_on(reply: str) -> tuple[str, str]:
    r"""Separate the answer from its closing "what to check against" line.

    Args:
        reply: The model's reply, verbatim.

    Returns:
        The explanation and the one-sentence check, in that order. The check is
        empty when the model did not write the marker, which is not an error --
        the answer is worth showing either way, and an explanation withheld
        because its footnote is missing is a strictly worse outcome than one shown
        without it.

    Examples:
        >>> split_rests_on("The gap closes at $h = J$.\nRESTS ON: Pfeuty's solution.")
        ('The gap closes at $h = J$.', "Pfeuty's solution.")
        >>> split_rests_on("No marker here.")
        ('No marker here.', '')
    """
    lines = reply.strip().splitlines()
    for position in range(len(lines) - 1, -1, -1):
        stripped = lines[position].strip()
        if stripped.upper().startswith(RESTS_ON_MARKER):
            check = stripped[len(RESTS_ON_MARKER) :].strip()
            return "\n".join(lines[:position]).strip(), check
    return reply.strip(), ""


def _material(citations: tuple[Citation, ...]) -> str:
    """Number the passages so the answer can point at them.

    Args:
        citations: What retrieval kept, best first.

    Returns:
        One numbered block per passage. The numbering is what the ``[n]`` markers in
        the answer refer to, so it has to match the order the interface renders them
        in -- which is why both read the same tuple rather than each sorting it.
    """
    return "\n\n".join(
        f"[{index}] {citation.title} ({citation.identifier}, shelf: {citation.shelf or 'unfiled'})"
        f"\n{citation.snippet}"
        for index, citation in enumerate(citations[:MAX_PASSAGES], start=1)
    )


_HEADING = re.compile(r"^#{1,6}(?:\s+(.*))?$")
"""A markdown heading line inside an indexed passage.

Requires a space after the hashes, so ``#hashtag`` in quoted prose is not mistaken
for one; a line of bare hashes matches too and contributes nothing, because an
empty heading inside a blockquote is a grey stripe with a gap in it.

Demoted to bold rather than kept. A ``>`` prefix does not survive a heading:
markdown ends the blockquote and renders the heading at page scale in the theme's
heading colour, in the middle of somebody's chat. A quotation that shouts is not a
quotation -- and dropping the heading text entirely is worse, because it is usually
the line that says what the passage is about.
"""

_BULLET = re.compile(r"^([-*+]|\d+[.)])\s+(.*)$")
"""A list item. Kept as a list item, on its own line, marker and all."""


def as_quotation(passage: str) -> str:
    """Render an indexed passage as a blockquote that stays one.

    Three things go wrong when a corpus chunk is dropped into a chat unaltered.

    Only the first line carries the ``>``. Every line after it leaves the
    quotation, so a ``##`` the chunk happened to contain renders as a page heading
    inside the reply, in the heading colour, and the prose under it stops being
    marked as somebody else's words. This is the one a reader notices immediately.

    The wrapping is wrong for the width. The corpus is hand-wrapped at about
    eighty columns, which is right for a file and ragged in a chat bubble half that
    wide. Wrapped lines are rejoined; blank lines, which are real paragraph breaks,
    are kept.

    Structure is not punctuation. Rejoining everything indiscriminately runs a
    heading straight into the sentence beneath it and turns a list into one long
    line of dashes. So headings and list items end a block rather than being
    swallowed into one.

    Args:
        passage: The passage as it was indexed.

    Returns:
        The passage as markdown blockquote lines: headings bold, list items kept as
        items, wrapped prose rejoined into paragraphs. Empty input gives an empty
        string rather than a lone ``>``, which renders as a grey stripe with nothing
        beside it.
    """
    blocks: list[str] = []
    running: list[str] = []

    def flush() -> None:
        if running:
            blocks.append(" ".join(running))
            running.clear()

    for raw in passage.strip().splitlines():
        line = raw.strip()
        if not line:
            flush()
            continue
        heading = _HEADING.match(line)
        if heading:
            flush()
            text = (heading.group(1) or "").strip()
            if text:
                blocks.append(f"**{text}**")
            continue
        bullet = _BULLET.match(line)
        if bullet:
            flush()
            # Started rather than finished, so the wrapped remainder of the item --
            # which in a hand-wrapped file is most items -- rejoins it instead of
            # becoming an orphan paragraph beneath its own bullet.
            running.append(f"{bullet.group(1)} {bullet.group(2).strip()}")
            continue
        running.append(line)
    flush()

    if not blocks:
        return ""
    return "\n>\n".join(f"> {block}" for block in blocks)


def grounded(
    citations: tuple[Citation, ...],
    searches: tuple[SearchRound, ...],
    measured: str = "",
    consulted: str = "",
) -> bool:
    """Whether there is anything behind an explanation except the model's memory.

    The feasibility branch cannot fail this: it runs circuits, prices them and runs
    the classical baseline, so its report always stands on something a reader can
    re-derive. The explain branch has no such floor: it computes nothing by
    design, so with no passages it stands on the model's training data alone -- and a
    fluent, well-organised, entirely unsourced paragraph is the single most dangerous
    thing this project can put on a screen. It is indistinguishable, to the reader it
    is written for, from the ones with four citations under them.

    So the test is not "did anything come back", it is **"was the corpus asked"**.
    Those differ in the case that matters:

    * A search ran and kept nothing. The notes are silent on the subject, and
      answering anyway substitutes an unrecorded source for a recorded one.
      :func:`refusal` says so.
    * No search ran at all -- the reader switched it off, or no chain was formalised
      to search for. Refusing here would blame the corpus for the reader's own dial,
      so the answer is written from general knowledge and *says* it is.

    * Something was **computed** for this question. The branch is described above as
      computing nothing, and that stopped being true when the graph gained a node that
      races the methods and hands the result here: an answer standing on a curve this
      project produced is better sourced than one standing on a quoted passage, because
      the reader can re-run it. Refusing on top of a measurement would be the corpus's
      silence overruling the project's own arithmetic.

    * A **tool returned something**. Same argument as a measurement, one step
      further out: an abstract fetched from arXiv during this run is a source the
      reader can open, and an exactly computed derivative is better evidence than a
      quoted sentence about one. Refusing while holding either would be the corpus's
      silence overruling material the run actually obtained.

    Args:
        citations: What retrieval kept.
        searches: Every round the retrieval loop ran.
        measured: What the campaign computed for this question, already rendered.
            Empty when it computed nothing.
        consulted: What the tools returned, already rendered. Empty when none was
            called or none of them answered.

    Returns:
        ``True`` when the answer may be written. ``False`` only for the first case
        above.

    Examples:
        >>> grounded((), ())
        True
        >>> empty = (SearchRound(query="barren plateau", kind="question", found=0, kept=0),)
        >>> grounded((), empty)
        False

        Unless the campaign computed something for the question itself:

        >>> grounded((), empty, "VQE reached -9.8 in 47 steps")
        True

        Or a tool answered it:

        >>> grounded((), empty, "", "search_arxiv returned 2408.01234")
        True
    """
    return bool(citations) or bool(measured) or bool(consulted) or not searches


def refusal(question: str, searches: tuple[SearchRound, ...]) -> Answer:
    """Say that the question cannot be answered from what this project holds.

    A refusal rather than a hedge, and the difference is the point. "I could not
    find much on this, but generally..." is the same unsourced paragraph with an
    apology in front of it: the reader still comes away with the claim, and now also
    with the impression that it was checked. Saying no is legible; a soft answer is
    not.

    What it must not be is a dead end. The trail says exactly which queries ran and
    what each kept, so a reader can see whether the corpus is silent or the query was
    wrong -- and those call for different second questions.

    Args:
        question: What was asked, repeated back so the reader can see what was
            searched for on their behalf.
        searches: Every round the retrieval loop ran.

    Returns:
        The refusal, marked ``written_by="refused"`` so the interface and the trace
        can tell it from an answer rather than having to read the prose.
    """
    tried = "\n".join(
        f"- {round_.query!r} ({round_.kind}) -- {round_.found} found, {round_.kept} kept "
        f"after grading"
        for round_ in searches
    )
    return Answer(
        text=(
            "**I could not answer that from what I have.**\n\n"
            "This project's notes were searched and hold nothing on it. I would "
            "rather say so than answer from a language model's memory, which is not "
            "a source you can check -- and on this branch nothing is computed either, "
            "so there would be no measurement behind the words.\n\n"
            f"The question was read as: *{question}*\n\n"
            f"**What was searched for:**\n{tried}\n\n"
            "The shelves hold four literatures -- the physics of the "
            "transverse-field Ising chain, the quantum circuits it is run on, how "
            "those methods compare, and what the model is used for outside physics. "
            "A question outside those, "
            "or one phrased in vocabulary none of them use, comes back empty. "
            "Rephrasing in the notes' own terms is often all it takes; asking for a "
            "**feasibility assessment** instead reaches the half of this application "
            "that computes rather than retrieves."
        ),
        written_by="refused",
    )


def _from_notes(question: str, citations: tuple[Citation, ...], measured: str = "") -> Answer:
    """Compose an answer out of the passages and whatever was computed.

    Args:
        question: What was asked, repeated back so the reader can see what the
            passages are being offered as an answer to.
        citations: What retrieval kept.
        measured: What the campaign computed for this question, already rendered.
            Put **first** when there is any, because it is the answer and the passages
            are background to it -- a reader who asked to see three methods race is
            owed the result of the race before four quotations about circuit depth.

    Returns:
        An answer, always. With nothing retrieved and nothing computed it says the
        notes do not cover the question and names what they do cover -- a reader who is
        told what is on the shelves can ask a better second question, and one who is
        told only "no results" cannot.
    """
    if measured and not citations:
        return Answer(
            text=(
                f"{measured}\n\n"
                "**No language model was reachable and this project's notes returned "
                "nothing**, so the paragraph above is the measurement itself rather "
                "than prose about it. The figure below is the same run, drawn."
            ),
            written_by="notes",
        )
    if not citations:
        return Answer(
            text=(
                "**The notes do not cover this, and no language model was reachable "
                "to answer it from general knowledge.**\n\n"
                f"The question was read as: *{question}*\n\n"
                "This project's notes hold four literatures -- the physics of the "
                "transverse-field Ising chain, the quantum circuits people run it on, "
                "how those methods compare, and what the model is applied to outside "
                "physics. A question outside "
                "those, or one phrased in vocabulary none of them use, returns nothing "
                "and is reported as returning nothing rather than being answered from "
                "somewhere else."
            ),
            written_by="notes",
        )
    kept = [
        (index, as_quotation(_without_title(clip(citation.snippet, LEAD_CHARACTERS), citation)))
        for index, citation in enumerate(citations[:MAX_PASSAGES], start=1)
    ]
    quoted = [(index, block) for index, block in kept if block]
    lines = [
        f"**No language model was reachable**, so this is what the notes returned for "
        f"your question -- {_counted(len(quoted))}, quoted rather than paraphrased and "
        f"trimmed to their opening lines. Each is numbered; **Sources**, below, says "
        f"where each one came from and holds the untrimmed text.\n",
    ]
    if measured:
        # Above the quotations, for the reason given in the docstring: what this
        # project computed for the question is the answer, and the passages are
        # background to it.
        lines.insert(0, f"{measured}\n")
    # The marker and nothing else. The title, the shelf and the link belong to the
    # source list, and printing them here as well put every title on the page twice
    # and every passage twice over -- which reads as a bug in the page rather than
    # as a citation. One number ties the two halves together, which is all a
    # numbered reference system has ever needed.
    lines += [f"**[{index}]**\n\n{block}\n" for index, block in quoted]
    return Answer(
        text="\n".join(lines),
        written_by="notes",
        cited=len(quoted),
    )


def _without_title(snippet: str, citation: Citation) -> str:
    """Drop a chunk's opening heading when it only repeats the title above it.

    Chunking keeps the note's own title at the head of its first chunk, so a
    passage quoted underneath a source list that already names the document
    printed that title twice within a couple of lines -- the second time inside a
    blockquote, which makes it look like the passage's own subject rather than the
    document's.

    Only the *first* heading, and only when it matches. A heading further down is
    the passage's structure and belongs in the quotation; a first heading that says
    something the title does not is information, not repetition.

    Args:
        snippet: The passage, already trimmed to the lead.
        citation: The source it belongs to, for the title to compare against.

    Returns:
        The passage, with a duplicated opening title removed.
    """
    lines = snippet.splitlines()
    if not lines:
        return snippet
    head = lines[0].lstrip("#").strip()
    if head and head.casefold() == citation.title.strip().casefold():
        return "\n".join(lines[1:]).lstrip("\n")
    return snippet


def _counted(passages: int) -> str:
    """Say how many passages came back, in words rather than as a bare digit.

    Args:
        passages: How many are being shown.

    Returns:
        A short phrase. Singular is spelled out because "1 passages" in the first
        sentence of an answer is the kind of thing a reader notices and nothing
        else on the page recovers from.
    """
    return "one passage" if passages == 1 else f"{passages} passages"


def _chain_block(model: FormalModel) -> str:
    """Describe the chain in full, not as a label.

    ``FormalModel.label()`` names the chain but hides which parts of it the question
    actually specified. The assumptions are the interesting half: an answer resting
    on "took no mention of boundary conditions to mean an open chain" is one a reader
    can push back on.

    Args:
        model: The chain the question named.

    Returns:
        The label, then each assumption on its own line.
    """
    lines = [f"THE CHAIN THE AGENT READ OUT OF THE QUESTION: {model.label()}"]
    if model.assumptions:
        lines.append(
            "Filled in rather than stated by the reader -- say so if the answer turns "
            "on one of these:"
        )
        lines += [f"- {assumption}" for assumption in model.assumptions]
    return "\n".join(lines)


def _searches_block(searches: tuple[SearchRound, ...]) -> str:
    """Say what the notes were actually searched for.

    Retrieval is a loop -- search, grade, rewrite, search again -- and the surviving
    passages alone cannot distinguish the two situations a thin answer comes from: a
    corpus silent on the subject, and a query that never named it. Told which queries
    ran, the narrator can say "this project's notes were searched for X and hold
    nothing on it", which is a fact about the corpus rather than an apology.

    Args:
        searches: Every round the retrieval loop ran, in order.

    Returns:
        One line per round, or the empty string when no search ran at all.
    """
    if not searches:
        return ""
    lines = ["THE NOTES WERE SEARCHED FOR:"]
    lines += [
        f"- {round_.query!r} ({round_.kind}) -- {round_.found} found, {round_.kept} kept"
        for round_ in searches
    ]
    return "\n".join(lines)


def compose_material(
    question: str,
    model: FormalModel | None,
    citations: tuple[Citation, ...],
    recalled: str = "",
    searches: tuple[SearchRound, ...] = (),
    measured: str = "",
    consulted: str = "",
) -> str:
    """Lay out everything the narrator is allowed to draw on, in one block.

    The branch aims at a single woven answer rather than a tour of the shelves, and
    a narrator can only weave what it was given -- so it receives the recalled
    conversation and the queries that ran, not just the question and the passages.

    The material is ordered least trustworthy first: the recall is a paraphrase and
    the passages are quoted text, so the strongest provenance is the last thing read
    before the answer is written. Anything the campaign computed goes last of all,
    being the one item the reader can re-derive rather than take on trust.

    Args:
        question: What was asked, verbatim.
        model: The chain the question named, if it named one.
        citations: What retrieval kept, best first.
        recalled: What the agent remembers of this conversation, already rendered.
        searches: Every round the retrieval loop ran.
        measured: What the campaign computed for this question, already rendered.
            Not wrapped as data: it came from this project's own physics layer and
            never left it.
        consulted: What the tools returned, already rendered. Wrapped as data, unlike
            ``measured``, because a tool result can be an abstract fetched from an
            index nobody here controls.

    Returns:
        The material, as labelled blocks. Untrusted text -- the question, the recall
        and the passages -- is wrapped by :func:`src.agent.llm.as_data` as it enters,
        so an instruction sitting in a corpus chunk arrives marked as content.
    """
    blocks = [f"QUESTION:\n{as_data(question)}"]
    if recalled.strip():
        blocks.append(f"EARLIER IN THIS CONVERSATION:\n{as_data(recalled.strip())}")
    if model is not None:
        blocks.append(_chain_block(model))
    else:
        blocks.append(
            "THE QUESTION NAMED NO SPECIFIC CHAIN. Answer in general terms and do not "
            "invent a length, a coupling or a field to answer about."
        )
    searched = _searches_block(searches)
    if searched:
        blocks.append(searched)
    material = _material(citations)
    if material:
        blocks.append(f"PASSAGES FROM THIS PROJECT'S NOTES:\n{as_data(material)}")
    else:
        blocks.append(
            "NO PASSAGES WERE RETRIEVED, so cite nothing. Answer from general "
            "knowledge, say in the text that you are doing so, and say that this "
            "project's notes do not cover it."
        )
    if measured.strip():
        blocks.append(
            "MEASURED BY THIS PROJECT FOR THIS QUESTION. These numbers were computed "
            "here, are shown to the reader as a figure directly beneath your answer, "
            "and are the answer to what was asked. Lead with them, quote them exactly, "
            "do not describe them as reported in the literature, and never write that "
            "no such results were available. Keep the whole answer to two or three "
            "short paragraphs: the reader asked to see a comparison, not to be taught "
            "the model it was run on.\n"
            f"{measured.strip()}"
        )
    if consulted.strip():
        blocks.append(
            "TOOLS CALLED DURING THIS RUN, AND WHAT THEY RETURNED. This is the "
            "freshest material you have and the reader can check every item of it. "
            "Use it, quote identifiers exactly, and attribute anything fetched from "
            "arXiv to arXiv rather than to this project's notes. If a tool returned "
            "nothing, say that nothing was found -- never fill the gap from memory, "
            "because a citation invented here is indistinguishable to the reader "
            "from one that was fetched.\n"
            f"{as_data(consulted.strip())}"
        )
    return "\n\n".join(blocks)


def explain(
    question: str,
    model: FormalModel | None,
    citations: tuple[Citation, ...],
    pool: ModelPool | None = None,
    audience: Audience = DEFAULT_AUDIENCE,
    recalled: str = "",
    searches: tuple[SearchRound, ...] = (),
    measured: str = "",
    consulted: str = "",
    sink: Callable[[str], None] | None = None,
) -> Answer:
    """Answer the question in prose, from the notes and a model if one is reachable.

    Args:
        question: What was asked. Wrapped as data: an explanation request is a
            natural place for an injected instruction to sit, and this branch's
            output goes straight to the reader with no arithmetic in between to
            make a tampered answer look wrong.
        model: The chain the question named, if it named one. Passed so an answer
            about "this chain" is about the same chain the rest of the session is.
        citations: What retrieval kept, best first.
        pool: The campaign's models. Omitted builds one from configuration.
        audience: Who the answer is written for. Changes the wording and nothing
            else -- the rules about what may be claimed and what must be cited are
            in the prompt this extends and are identical at every level.
        recalled: What the agent remembers of this conversation, already rendered.
            An explanation that ignores the turn before it makes the reader repeat
            themselves, which is the most common way a chat interface feels stupid.
        searches: Every round the retrieval loop ran. Handed over so the answer can
            say what the notes were searched for and came back empty on, rather than
            leaving a gap the reader has to guess the cause of.
        measured: What the campaign computed for this question, already rendered.
            A question asking to watch three methods converge is answered by the race
            the graph ran, not by four passages about circuit depth -- so this both
            grounds the answer and leads it.
        consulted: What the tools the model called returned, already rendered. This
            is how a request for recent papers gets recent papers: the corpus is a
            fixed set of abstracts and cannot answer it, and a model answering it
            from memory invents the citation.
        sink: Given, the model's reply is streamed and each piece is handed to this
            as it arrives, so a reader can start reading the longest call in the
            graph while it is still being written. It sees the reply *raw*: the
            trailing "rests on" section is still attached and the mathematics has
            not been through :func:`src.agent.mathmarkup.to_dollar_math` yet, so a display
            using it should treat it as a preview and redraw the returned answer
            when this function comes back. Every path that does not reach a model --
            a refusal, an offline pool, an empty reply -- never calls it, so a
            display must not assume it will be called at all.

    Returns:
        An answer, always. Every failure -- offline, over the call ceiling, a refused
        call -- lands on the passages, because the reader's situation is the same in
        all of them and it is a situation the notes can still partly serve.
    """
    # Before the model, and before the notes. A refusal is not a fallback that a
    # reachable model overrides -- an unsourced answer written by a *better* model is
    # a worse outcome, not a better one, because it is more convincing and no more
    # checkable. It also saves the call.
    if not grounded(citations, searches, measured, consulted):
        _logger.info("campaign_refused", extra={"rounds": len(searches)})
        return refusal(question, searches)

    models = pool if pool is not None else ModelPool()
    if models.offline:
        return _from_notes(question, citations, measured)

    # Prose rather than a schema. This answer is mostly LaTeX, and a JSON string
    # field eats the backslash of `\frac` and `\langle` whenever the model forgets
    # to double it -- which is intermittent, so it reads as a mystery rather than as
    # a bug. See `ModelPool.narrate`.
    # A lesson is a different shape of answer, not a longer one, and the prompt is
    # told which it is writing. See `TEACHING_SHAPE` for why this cannot be one prompt
    # tuned to sit between the two.
    system = EXPLAIN_SYSTEM
    if reading.asks_to_be_taught(question):
        system += TEACHING_SHAPE
    # Both can apply -- *teach me the mapping and plot the spectrum* is one question
    # with two halves -- and the sweep shape goes last so that its override of rule 2
    # is the final instruction read. A lesson that refused to quote the numbers it was
    # handed would be the teaching shape winning an argument it should not be in.
    if reading.asks_for_a_field_sweep(question):
        system += SWEEP_SHAPE
    reply = models.narrate(
        "explanation",
        for_audience(system, audience),
        compose_material(question, model, citations, recalled, searches, measured, consulted),
        sink,
    )
    if reply is None or not reply.strip():
        return _from_notes(question, citations, measured)

    text, rests_on = split_rests_on(reply)
    if not text:
        return _from_notes(question, citations, measured)

    written = Answer(
        text=text,
        rests_on=rests_on,
        written_by="model",
        cited=min(len(citations), MAX_PASSAGES),
    )
    _logger.info(
        "campaign_explain",
        extra={"characters": len(written.text), "cited": written.cited},
    )
    return written
