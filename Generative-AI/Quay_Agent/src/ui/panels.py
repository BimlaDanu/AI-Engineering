r"""The pieces every page draws with, and the words they draw.

Four pages had four versions of the same header, four ways of writing "present"
and "absent", and no shared idea of what a reader is assumed to know. This module
is the correction: the *content* -- the glossary, the page index, the health facts,
the vocabulary of ticks and marks -- is plain data and pure functions, and the
drawing on top of it is deliberately thin.

That split is the same one :mod:`src.ui.status` makes and it is made for the same
reason. A dashboard that composes its own prose inline cannot be tested, and the
particular failure being guarded against is a monitor that keeps rendering
confidently after the thing it monitors has stopped working. Everything below the
``Drawing`` heading is a handful of Streamlit calls over a value computed above it,
so the part that can be wrong is the part that is checked.

Written for a reader who does not do physics
--------------------------------------------

This interface assumes no physics. Two consequences run through the whole module.
Every domain word is glossed the first time it appears on a page, from
:data:`GLOSSARY`, in ordinary English with no second term smuggled into the
explanation of the first. And every page says where its numbers came from, because
"the interface said so" is not a provenance anybody can check, and a monitor that
cannot be checked is decoration.

The sidebar is the same on every page for the same reason a form is: somebody
arriving on page three should be able to tell what the project is, how far it has
got, and what to press, without having read page one.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, cast, get_args
from uuid import uuid4

import numpy as np
import pandas as pd
import streamlit as st

from src.agent import reading
from src.agent.explaining import as_quotation
from src.agent.graph import circuit_the_question_named
from src.agent.llm import ToolCall
from src.agent.mathmarkup import demote_headings, to_dollar_math
from src.agent.memory import Memory
from src.agent.model_selection import DEFAULT_TIER_SLUGS, featured_slugs, selectable_slugs
from src.agent.state import CampaignState, RunRecord, Verdict
from src.figure_export import FIGURE_FORMAT, figure_path
from src.hardware.devices import device_names
from src.logging_setup import configure_logging, get_logger, log_path_from_environment
from src.physics.model import hamiltonian_display
from src.physics.quantum.ansatz import AnsatzSpec
from src.physics.quantum.circuit_algebra import state_preparation_latex
from src.physics.quantum.method_race import METHOD_BLURBS
from src.rag.ingest import CURATED, SHELVES, get_shelf, load_library
from src.settings import DEFAULT_AUDIENCE, Audience, get_settings
from src.ui import access, figures, session_log
from src.ui import setting as knob
from src.ui.status import (
    PROJECT_ROOT,
    build_progress,
    corpus_size,
    missing_dependencies,
)

LOG = get_logger("ui.panels")
"""Where a corpus that cannot be read is reported.

The interface says "no notes" either way; only the log can say whether that
means an empty directory or an unreadable one.
"""

APP_NAME = "Quay"
"""What the application is called.

``QU`` and ``AI``, which is what it is: quantum information met by an ordinary
language-model agent. A quay is also the place where two domains meet and cargo is
inspected before anybody accepts it, which is a fair description of a feasibility
study and an unusually literal one for a pun.

It is pronounced "key", and roughly half of any audience will read it as "kway" --
so wherever it is a heading it carries :data:`APP_DESCRIPTOR` and
:data:`APP_PRONUNCIATION` beneath it. A short name earns its keep by being
remembered; the descriptor is what makes it understood the first time, and somebody
with five seconds to spare needs both.
"""

APP_PRONUNCIATION = 'QU + AI, said like "key"'
"""How to say the name, and where the name came from, in one line.

A named constant rather than a literal at the point of use, because it moved once:
it was in the sidebar, on all nine pages, and is now on the one page that is the
product. A string that travels between call sites is a string that should have had
a name before it travelled.
"""

APP_DESCRIPTOR = "Quantum Feasibility Agent"
"""What the name means, in words that need no explaining.

Always rendered directly under :data:`APP_NAME`. Everything the application claims
to be is in the three words: it reasons about quantum computers, it answers a
question of feasibility rather than of physics, and it is an agent.
"""

ICON = str(Path(__file__).parent / "assets" / "quantum-circuit.png")
"""A two-wire circuit with a rotation gate and a CNOT: the tab icon and sidebar mark.

Drawn rather than borrowed, so that it says what the project is at a glance: a
parameterised single-qubit rotation and an entangling gate are exactly the two
ingredients every circuit here is built from.

A PNG because a browser tab is a raster target. It is generated by
the icon script, which also writes the PDF copy that can be opened
outside a browser -- every *figure* in this project is a PDF, for that reason.
"""

CURRENT_TIER = 5
"""How deep into the dependency order the project has reached.

It decides what counts as a missing dependency. A package that only a later tier
needs is not a problem yet, and a monitor that reports it as one will have been
learned to ignore by the time something is actually wrong.
"""

PRESENT = "✅"
ABSENT = "⬜"
WARNING = "⚠️"
"""One vocabulary of marks, so that a tick means the same thing on every page.

Three states and no more. A tick is *this is here*, an empty box is *this is not
here yet*, and a warning is *this is here and something about it is wrong*. The
third is the one worth having: a dependency that is installed at the wrong version
is a different problem from one that is missing, and a monitor with only two marks
has to file it under one of them.
"""


@dataclass(frozen=True, slots=True)
class Term:
    """One piece of vocabulary, explained without using another one.

    Attributes:
        word: The term as it appears in the interface and in the reports.
        gloss: What it means, in ordinary English. The rule this file is written
            under is that the gloss may not contain another glossary term, because
            an explanation that needs its own explanation has not been given.
        why: Why a reader should care -- what changes in the project if this
            quantity moves. A definition tells somebody what a word means; this
            tells them why it is on the page.
    """

    word: str
    gloss: str
    why: str


GLOSSARY: tuple[Term, ...] = (
    Term(
        "lattice",
        "Which magnets count as neighbours of which. A line gives each magnet two "
        "neighbours, a square grid four, a triangular grid six.",
        "It is the one thing that differs between the three problems here, and it "
        "changes everything downstream: more neighbours means more gates per "
        "circuit layer, and a line is the only shape whose answer can be written "
        "down as a formula.",
    ),
    Term(
        "transverse-field Ising model",
        "Small magnets on a lattice. Each one prefers to point the same way as its "
        "neighbours, and a sideways push tries to knock all of them over.",
        "It is the whole problem. Everything here designs, prices or scores an "
        "attempt to find its lowest-energy arrangement.",
    ),
    Term(
        "ground state",
        "The arrangement with the least energy -- the one the system settles into "
        "when nothing is disturbing it.",
        "It is the answer being looked for, and the number every method is scored against.",
    ),
    Term(
        "superposition",
        "The system does not settle into one arrangement of its magnets. Its "
        "lowest-energy state is several arrangements at once, each carrying a share.",
        "It is the only reason a quantum computer is in this conversation at all. "
        "An answer that was one arrangement could be written down on paper, and the "
        "Lab counts how many arrangements the state is at once rather than "
        "asserting it.",
    ),
    Term(
        "entanglement",
        "Two magnets can be more definite about each other than either is about "
        "itself -- you can know they disagree without knowing which way either "
        "points.",
        "It is what an ordinary computer has to store explicitly and pays for, and "
        "what a quantum one carries for free. How much of it a problem has is most "
        "of what decides whether the quantum route is worth it.",
    ),
    Term(
        "energy per magnet",
        "The total energy divided by how many magnets there are.",
        "Every number in this project is quoted this way so that problems of "
        "different sizes can be compared. A total energy always grows with the "
        "number of magnets, which tells a reader nothing about whether the answer "
        "got better.",
    ),
    Term(
        "critical point",
        "The one setting of the sideways push where the system stops being one "
        "thing and starts being the other, with no sharp switch in between.",
        "It is where every approximate method has its hardest time, so it is where "
        "the test cases are concentrated. A method scored away from it would report "
        "a high mark for machinery that fails exactly where it matters.",
    ),
    Term(
        "reference answer",
        "On a line, this problem can be solved with pencil and paper, so the true "
        "lowest energy is known in closed form. On a square or triangular lattice "
        "there is no such formula, and the reference has to be computed instead -- "
        "which costs exponentially and is why lattice sizes here are small.",
        "It is what makes this project unusual: there is a right answer to score "
        "against. Most language-model applications are scored by another language "
        "model.",
    ),
    Term(
        "circuit",
        "The list of operations a quantum computer performs, in order.",
        "Longer circuits can represent better answers and are more likely to be "
        "ruined by noise. Choosing the length is the central decision.",
    ),
    Term(
        "ansatz",
        "A circuit whose shape is fixed in advance but which has adjustable dials. "
        "A search turns the dials to make the energy as low as it will go.",
        "It is the guess about what the answer looks like. A guess that cannot "
        "express the true answer will never reach it, however well the dials are "
        "tuned.",
    ),
    Term(
        "barren plateau",
        "The search surface goes flat. Wherever the dials are moved, the energy "
        "barely changes, so nothing tells the optimiser which way to go.",
        "It is the standard way this family of algorithms fails, and it gets worse "
        "as the problem grows -- which is the opposite of what a scalable method "
        "needs. The circuit used here is built to resist it, and the Lab says how.",
    ),
    Term(
        "variational bound",
        "Any answer this method produces is guaranteed to be at or above the true "
        "lowest energy, never below it.",
        "It is why a result can be trusted without knowing the answer. A number "
        "that comes back *below* the reference is a bug, not a discovery.",
    ),
    Term(
        "depth",
        "How many operations have to happen one after another, rather than at the "
        "same time. It is the circuit's running time, not its size.",
        "Qubits forget what they were doing after a fixed period, so depth is the "
        "budget that actually binds.",
    ),
    Term(
        "shot",
        "One run of the circuit, producing one random outcome. An energy is the "
        "average over many of them.",
        "Accuracy improves as the square root of the count, so ten times the "
        "precision costs a hundred times as much. This is where budgets are spent.",
    ),
    Term(
        "transpilation",
        "Rewriting a circuit into the operations a particular machine can actually "
        "perform, on the qubits it actually has.",
        "A machine's qubits are wired to some of their neighbours and not others. "
        "Interactions the wiring does not provide have to be shuffled into place, "
        "and that is paid for in depth.",
    ),
    Term(
        "coherence time",
        "How long a qubit holds on to what it is doing before it dissolves into noise.",
        "A circuit that runs longer than a fraction of it returns something the "
        "algorithm did not design.",
    ),
    Term(
        "fidelity",
        "The fraction of the result that is still the intended one rather than noise.",
        "Recovering a fixed accuracy from a damped signal costs the inverse square "
        "in extra measurements, so this converts directly into money.",
    ),
    Term(
        "baseline",
        "The best available answer from an ordinary computer, computed here rather than quoted.",
        "It is what a quantum result has to beat. A feasibility study without one "
        "is an advertisement.",
    ),
    Term(
        "imaginary time",
        "A bookkeeping trick, not a physical process. Running the chain forwards "
        "through it makes every state except the lowest-energy one fade away.",
        "It is how the reference answer on the Lab page is produced, and it is also "
        "what the classical competitor is doing. No hardware can run it, which is "
        "exactly why it makes a clean thing for the circuit to be measured against.",
    ),
    Term(
        "sign problem",
        "Some quantum problems turn, when rewritten for an ordinary computer, into "
        "sums where the terms cancel almost exactly. Those are the ones an ordinary "
        "computer cannot do.",
        "This chain has no sign problem, so the classical competitor here is not "
        "handicapped. That is most of the reason the honest verdict on it is "
        "usually 'no', and saying so is what separates a feasibility study from an "
        "advertisement.",
    ),
    Term(
        "verdict",
        "Go, no, or conditional -- decided by a rule applied to the numbers, not by "
        "asking a language model what it thinks.",
        "'Conditional' is not a hedge. It is the right answer when the obstruction "
        "is a stated, checkable quantity, and it is only usable with that quantity "
        "written beside it.",
    ),
)
"""Every domain term the interface uses, glossed for a reader with no physics.

Ordered as a reader meets them: what the problem is, what an answer is, what a
circuit is, what it costs, and what is concluded. Not alphabetical -- an
alphabetical glossary is a reference for somebody who already knows the subject.
"""


@dataclass(frozen=True, slots=True)
class Page:
    """One page of the interface, and the question a visitor arrives holding.

    Attributes:
        module: Path to the page file, relative to this package's directory.
        label: How the page is named in the navigation.
        icon: One emoji, taken from the page's own subject. Emoji rather than
            monochrome glyphs for one reason: they carry colour, and a column of
            outline icons reads as decoration while six distinct symbols are
            scannable -- a reader learns "the chip one is the machines" and stops
            reading the labels.
        group: Which heading it sits under in the navigation. Grouped by what a
            visitor came for, never by how the code is organised.
        question: What the page answers. Phrased as a question because an index of
            nouns tells somebody what exists and not which one they want.
    """

    module: str
    label: str
    icon: str
    group: str
    question: str


PAGES: tuple[Page, ...] = (
    Page(
        "pages/chat.py",
        # **"Quay -> Ask", not "Chat -> Chat".** The navigation prints the group
        # heading above the page label, so a group called Chat holding a page called
        # Chat printed the same word twice, two lines apart, at the very top of the
        # sidebar. The group now names the product once and the label is a verb
        # saying what to do with it.
        #
        # Not "QuayGPT", which was the other candidate. It borrows another company's
        # suffix, and worse, it points at exactly what this application is not: the
        # whole claim is that nothing here is generated-and-hoped-for. Putting
        # "plausible text generator" in the first word on screen would undercut the
        # argument the rest of the project spends itself defending.
        "Ask",
        "💬",
        "Quay",
        "Put a question to the agent and read the verdict it reaches.",
    ),
    Page(
        "pages/lab.py",
        "Quay Lab",
        # An atom rather than a microscope. A microscope says "science happens
        # here", which is true of half the pages; this one is about spins, a
        # circuit and a quantum state, and the atom is the only glyph in the set
        # that says which science.
        "⚛️",
        "Explore",
        "The circuit, the optimisation and the time evolution, with no model in the loop.",
    ),
    Page(
        "pages/knowledge.py",
        "Knowledge base",
        "📚",
        "Explore",
        "What has the agent actually read, and what is outside its notes?",
    ),
    Page(
        "pages/physics_and_hardware.py",
        # Was two pages, *Cross-check* and *Machines*. They are one because the
        # second is worthless without the first -- a shot budget priced against a
        # Hamiltonian with a flipped sign is an exact answer about a problem nobody
        # asked about -- and because the property they shared was the one thing
        # least obvious about either: both are driven only by the settings knob,
        # with no model in the loop. Apart, that was a caption repeated twice.
        # Together it is what the page is.
        #
        # Not "Model and machine", which was the other candidate. "Model" is the
        # most overloaded word in this application -- the physics one and the
        # language one -- and this is the page that insists no language model
        # touches it, so spending its label on that ambiguity would undercut the
        # sentence directly beneath it.
        "Physics and hardware",
        # The cryostat, which is what a photograph of a quantum computer is a
        # photograph of: a gold chandelier hanging inside a fridge at fifteen
        # millikelvin. It earns the slot on this page because the hardware half's
        # central quantity *is* that cold -- coherence time is how long the machine
        # stays quantum, and every circuit this page refuses is one that outlived it.
        #
        # The alternatives all misled. Scales, which were here, weighed nothing a
        # reader could see. A desktop computer says *ordinary machine*, which is the
        # thing this page prices the alternative to. The atom would be right and is
        # already the Lab's. `:material/memory:` is a literal microchip and was
        # tried first -- `test_every_page_has_exactly_one_icon` rejected it, and
        # correctly: one monochrome glyph among nine coloured ones reads as a
        # rendering fault, and colour is what lets a reader navigate by icon at all.
        "🧊",
        "Explore",
        "Is the number right, and what would it cost to run?",
    ),
    Page(
        "pages/pipeline.py",
        "Pipeline trace",
        "🧭",
        "Under the hood",
        "How did it decide, and which nodes actually ran?",
    ),
    Page(
        "pages/evaluations.py",
        "Evaluations",
        "✅",
        "Under the hood",
        "How well does it answer, on questions it was not tuned against?",
    ),
    Page(
        "pages/analytics.py",
        "Analytics",
        "📊",
        "Under the hood",
        "What did it cost — money for model calls, and measurements on hardware?",
    ),
)
"""Every page, in navigation order.

**The labels are the ones this project settled on and they are not to be churned.**
*Ask*, *Quay Lab*, *Knowledge base*, *Physics and hardware*, *Analytics*,
*Pipeline trace*, *Evaluations*. They were argued over once and a rename costs a
reader who has learned the menu more than it gains anybody.

*Physics and hardware* is the one exception and it was not a rename: *Machines* and
*Cross-check* were merged, which is a different decision from renaming a page --
there is one fewer thing in the menu rather than the same thing under a new word.

**The sectioning is the one the previous project arrived at**, because it was
arrived at by watching people use it rather than by reasoning about it. *Quay* is
the product and *Ask* is the one thing to do with it, so that group holds one page
and is the default. *Explore* is everything a visitor can poke at:
the circuit and the optimisation behind an answer, the notes the agent is allowed
to read, the machines it was priced against, and the cross-check that says the model
underneath it is right. *Under the hood* is evidence about the **agent** rather than
about the problem -- which nodes ran, how it scores on questions it was not tuned
against, and what the whole thing costs.

*Analytics* sits **under the hood** rather than in Explore, which is where the
previous project put it and where it does not belong here. Four of its five tabs
are evidence about the *language-model* half of the system -- what this session's
calls cost, which arm of the graph served each question and which nodes ran, a
head-to-head bake-off of one model against another, and the developer section. Only
the cost-surface tab is about the physics, and one tab does not move a page.

**What changed is what was removed.** *Runs*, *Environment*, *Build* and *About*
were pages and are not. The first three were instrumentation for whoever was
building this project rather than features for anybody using it, and *Build* in
particular put a progress bar announcing incompleteness on the screen of the person
deciding whether to trust the thing. *About* was the only entry that was not a
question about an answer: its ethics statement is now a tab on Evaluations and its
developer section a tab on Analytics, each beside the thing that raises the
question.

So one group holds a single page -- the product a visitor arrives for, which is an
anchor rather than a category. Every other heading covers more than one.

Data rather than a hardcoded call to :func:`streamlit.navigation`, so that a test
can hold every entry against a file that exists. A navigation list maintained by
hand goes stale silently, and a menu offering a page that is not there reads as a
fault in the reader's browser rather than in the index.
"""


def pages_in(group: str) -> tuple[Page, ...]:
    """List the pages under one navigation heading.

    Args:
        group: The heading.

    Returns:
        Its pages, in navigation order. Empty for a heading nothing uses.
    """
    return tuple(page for page in PAGES if page.group == group)


def groups() -> tuple[str, ...]:
    """List the navigation headings, in order and without repeats.

    Returns:
        The headings.
    """
    seen: list[str] = []
    for page in PAGES:
        if page.group not in seen:
            seen.append(page.group)
    return tuple(seen)


def tick(present: bool) -> str:
    """Render a yes-or-no state in the shared vocabulary.

    Args:
        present: Whether the thing is there.

    Returns:
        The tick or the empty box.
    """
    return PRESENT if present else ABSENT


def presence(present: bool, present_word: str = "set", absent_word: str = "not set") -> str:
    """Render a yes-or-no state with a word beside the mark.

    A mark alone is ambiguous the first time somebody sees it, and a legend at the
    bottom of the page is a legend nobody scrolls to.

    Args:
        present: Whether the thing is there.
        present_word: What to say when it is.
        absent_word: What to say when it is not.

    Returns:
        The mark and the word.
    """
    return f"{PRESENT} {present_word}" if present else f"{ABSENT} {absent_word}"


def humanise_bytes(size: int) -> str:
    """Render a file size in units somebody reads rather than counts.

    Args:
        size: The size in bytes. Negative sizes are treated as zero, since a file
            listing is not the place to discover a bug in a stat call.

    Returns:
        The size with a unit, to at most one decimal place.
    """
    remaining = float(max(size, 0))
    for unit in ("B", "kB", "MB"):
        if remaining < 1024.0 or unit == "MB":
            return f"{remaining:.0f} {unit}" if unit == "B" else f"{remaining:.1f} {unit}"
        remaining /= 1024.0
    return f"{remaining:.1f} MB"


@dataclass(frozen=True, slots=True)
class Health:
    """The three facts the sidebar reports on every page.

    Gathered into a record rather than read three times, so that the sidebar and a
    test are looking at the same snapshot rather than at three reads that could
    disagree if a file appeared between them.

    Attributes:
        built: Modules present in the working tree.
        planned: Modules the build order lists.
        missing: Names of packages needed by now and not installed.
        notes: How many documents the retrieval corpus holds.
    """

    built: int
    planned: int
    missing: tuple[str, ...]
    notes: int

    @property
    def ready(self) -> bool:
        """Whether the next piece of work can run here."""
        return not self.missing

    @property
    def progress(self) -> float:
        """How much of the build order exists, between zero and one."""
        return self.built / self.planned if self.planned else 0.0


def health(tier: int = CURRENT_TIER) -> Health:
    """Take one snapshot of where the project stands.

    Args:
        tier: How deep into the dependency order to check against. A package needed
            only by later work is not missing yet.

    Returns:
        The snapshot.
    """
    built, planned = build_progress()
    return Health(
        built=built,
        planned=planned,
        missing=tuple(dependency.distribution for dependency in missing_dependencies(tier)),
        notes=corpus_size(),
    )


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------


_LOGGING_STARTED = False


def start_logging() -> None:
    """Install this project's log handler in the process serving the interface.

    **The interface was the one process that never did.** Every script, the eval
    harness, the MCP server and the campaign runner call
    :func:`~src.logging_setup.configure_logging`; ``streamlit run`` did not. So under
    the interface the project's logger had no handler of its own, and three things
    followed, each worse than the last.

    Records propagated to the root logger, which Streamlit configures to print the
    message and nothing else -- so every field passed through ``extra=`` was
    discarded. ``structured_call_unusable`` carries *which model, which schema, and
    which of the four ways of failing it was*; all a person saw was the two words.
    The root logger also sits at ``WARNING``, so every successful call vanished and
    only the failures printed, which makes a run that failed eight calls out of forty
    look like a run that did nothing but fail. And :class:`~src.logging_setup.JsonFormatter`
    is where the credential is masked, so any line that did print was unmasked.

    A diagnosis published to a terminal with its cause removed is worse than silence:
    it tells somebody something is wrong and denies them every means of finding out
    what. That is what this function exists to prevent.

    Idempotent, and called from both places a page can be drawn from -- the entry
    point and, for a page opened on its own, :func:`current_setting`. The guard is
    for the settings read rather than for the handler: ``configure_logging`` replaces
    handlers rather than adding to them and is safe on every rerun, but a rerun is
    cheap and reading configuration on each of them is not.

    **The interface is the one caller that sends the stream to a file.** Everything
    else here runs to completion and hands its terminal back -- a campaign from the
    command line, an eval sweep, an ingest -- and for those the terminal *is* the
    right sink, because somebody is reading the run as it happens and nothing else
    is competing for the space. The interface is different: it is long-lived, it
    reruns on every click, and one question emits between forty and a hundred
    records while a person waits for an answer they will read in the browser rather
    than in the terminal. So the terminal keeps the warnings and errors, which are
    the only lines anybody has to act on, and
    :data:`~src.logging_setup.DEFAULT_LOG_PATH` keeps the whole trace. Nothing is
    discarded, and nothing the interface itself needs comes from this stream -- the
    Pipeline trace page reads the campaign it recorded in session state, and the
    cost meter reads :func:`~src.logging_setup.observing_calls`, so both are
    unaffected by where the text goes.
    """
    global _LOGGING_STARTED
    if _LOGGING_STARTED:
        return
    _LOGGING_STARTED = True
    # The credential is read for one purpose: to give the formatter the literal
    # string to mask. Nothing here sends it anywhere, and a checkout with no `.env`
    # takes the second branch and logs just as well -- which keeps the entry point's
    # promise that drawing a page needs no credential.
    secrets: list[str] = []
    try:
        secrets.append(get_settings().openrouter_api_key.get_secret_value())
    except Exception:
        pass
    # Through the environment rather than straight from the constant, so the file
    # can be moved or switched off by whoever runs the process -- which the test
    # suite has to do from outside, because importing a page module runs the page and
    # therefore configures logging during collection, before any fixture exists.
    configure_logging(
        secrets=[secret for secret in secrets if secret],
        log_file=log_path_from_environment(),
    )


TOP_BAR_CSS = f"""<style>
/* Lift the account controls out of the page body and onto the header strip, to the
   left of Streamlit's own Deploy button.

   They belong beside Deploy rather than in the page: right-aligned inside a wide
   content column, they landed a long way in from the window's edge -- which reads as
   the middle of the screen rather than as a corner, and a control in the middle of a
   page is a control competing with the answer.

   Fixed rather than floated, because the header strip is not this element's parent
   and no amount of margin will move it there. Out of the flow, so the content it used
   to sit above closes up behind it.

   The right offset has to clear everything Streamlit puts in that strip, and the
   list is longer than Deploy and the menu: while a script is running a **Stop**
   button appears to their left. At 8.5rem this bar sat on top of it, which is the
   one overlap that is not cosmetic -- a reader watching a long campaign could no
   longer stop it. 12rem cleared Stop and then covered the navigation control that
   sits beside it, so the number went up again: 16rem, about 256px.

   It is a measured guess rather than a computed value, because the header is
   Streamlit's and its contents are not ours to interrogate. If a future version
   widens that toolbar or adds to it, this is the one number to change -- and the
   thing to check is the whole strip at a wide window, not Deploy alone, since every
   overlap so far has been with something that appears next to it rather than with
   Deploy itself.

   It stood at 19.5rem, and part of that was reserved for a notice that is not there
   any more: with a file watcher installed, a source change is picked up instead of
   announced, so the **File change. / Rerun / Always rerun** strip no longer parks
   itself between Deploy and this bar. That left about twelve rem of empty header
   between the last control and Deploy, which reads as the bar having drifted inward
   rather than sitting in the corner.

   17rem takes most of that space back and leaves the rest as a margin. 15.5rem was
   tried first and read as crowding Deploy: the two groups belong to different
   applications, and with too little between them they scan as one toolbar with an
   arbitrary break in it. Whatever the number, it must clear Stop, which appears to
   Deploy's left while a script runs and is the one neighbour that must never be
   covered -- a reader watching a long campaign has to be able to stop it.

   Below a narrow window the header has no room to share, so the rule is dropped and
   the controls go back into the page where they still fit. */
@media (min-width: 900px) {{
    .st-key-{access.TOP_BAR_NAME} {{
        position: fixed;
        top: 0.6rem;
        right: 17rem;
        z-index: 999991;
        width: auto;
        margin: 0;
    }}
    /* And with the bar up there, close the gap it left behind. The page body is
       padded clear of the header strip by default, which is right when the first
       thing in the body is content -- but here the strip is where this application's
       own controls now live, so a page opened at this width began with about eighty
       pixels of nothing above its first line. Every page draws the bar, so the trim
       is right on every page. This is the same media query on purpose: below it the
       bar drops back into the body, and the padding is the only thing keeping it out
       from under Streamlit's own header. */
    [data-testid="stMainBlockContainer"] {{
        padding-top: 1.1rem;
    }}
}}
</style>"""
"""Where the account controls sit, and why it takes a stylesheet to put them there.

Here rather than beside the controls it moves, so that the one call to
:func:`st.set_page_config` and the stylesheets are configured in the same place --
all of them are properties of the window rather than of a page. See
:data:`SURFACE_CSS` for the other one.
"""

SURFACE_CSS = """<style>
/* Surfaces. Colour and shape live in `.streamlit/config.toml`; this is the part a
   theme cannot express -- depth, spacing and what a thing does when pointed at.

   Written against `data-testid` attributes, which are the only selectors Streamlit
   treats as an interface. Anything keyed on a generated class name would survive
   exactly one upgrade. Every rule is additive and cosmetic: if a future version
   renames a testid the rule stops applying and the page renders plainly, which is
   the correct way for decoration to fail.

   Nothing here moves an element, changes a label or hides a control. A reader who
   turns the stylesheet off sees the same application in a plainer coat. */

/* Headings: Streamlit sets them loose and evenly spaced, so a section reads as a
   list of equals. Tightening the top margin and the tracking puts a heading with
   the text it introduces rather than midway between two blocks. */
h1, h2, h3 { letter-spacing: -0.015em; }
h2 { margin-top: 2.2rem; }
h3 { margin-top: 1.6rem; }

/* Buttons. The starters are the first thing anybody touches and they were flat
   grey rectangles indistinguishable from a form control. The left bar marks them
   as a *suggestion* rather than an action, and it grows on hover, which is the
   cheapest way to say "clickable" without animating a whole element. */
[data-testid="stBaseButton-secondary"] {
    border-left: 3px solid var(--secondary-background-color, #F4F6FB);
    text-align: left;
    transition: border-color 120ms ease, box-shadow 120ms ease, transform 120ms ease;
}
[data-testid="stBaseButton-secondary"]:hover {
    border-left-color: var(--primary-color, #4c6ef5);
    box-shadow: 0 2px 10px rgba(31, 41, 51, 0.09);
    transform: translateY(-1px);
}

/* Metrics as cards. A row of bare numbers on a white ground has no edges, so the
   eye cannot tell where one figure stops and the next begins -- which matters here,
   because the numbers next to each other are usually in different units. */
[data-testid="stMetric"] {
    background: var(--secondary-background-color, #F4F6FB);
    border: 1px solid var(--border-color, #E3E8F2);
    border-radius: 0.6rem;
    padding: 0.75rem 0.9rem;
}
[data-testid="stMetricLabel"] { opacity: 0.72; }

/* Expanders hold the arithmetic behind almost every claim on these pages, so they
   are opened often and should look like a surface rather than a seam. */
[data-testid="stExpander"] details {
    border-radius: 0.6rem;
    overflow: hidden;
}
[data-testid="stExpander"] summary:hover { color: var(--primary-color, #4c6ef5); }

/* A figure is evidence here, not decoration, so it gets the same card treatment as
   a number and stops floating unbounded in the column. */
[data-testid="stImage"], [data-testid="stPlotlyChart"], .stPyplot {
    border-radius: 0.5rem;
    overflow: hidden;
}

/* Tabs: a heavier active label, because the tab bars on the Lab page carry eight
   of them and the default weight difference is nearly invisible. */
[data-testid="stTabs"] button[aria-selected="true"] { font-weight: 650; }

/* The sidebar is settings and the page is the answer; a rule between them says so
   more quietly than a shadow. */
[data-testid="stSidebar"] { border-right: 1px solid var(--border-color, #E3E8F2); }

/* Dataframes and code, rounded to match everything else. */
[data-testid="stDataFrame"], pre { border-radius: 0.5rem; }
</style>"""
"""Depth, spacing and hover -- the part of a look a Streamlit theme cannot state.

Separate from :data:`TOP_BAR_CSS` because that one *moves* a control and this one
only paints. A rule that repositions something can break a page; a rule that rounds
a corner cannot, and keeping them apart means the risky stylesheet stays four rules
long and reviewable.
"""


def open_page() -> None:
    """Set the tab, the icon and the layout, once, before anything is drawn.

    Called from the entry point alone. Streamlit permits one such call per run and
    the navigation model makes that natural: the pages are run *by* the entry point
    rather than reached directly, so a page that configured itself would be
    configuring somebody else's window.
    """
    start_logging()
    st.set_page_config(
        page_title=f"{APP_NAME} — {APP_DESCRIPTOR}",
        page_icon=ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(TOP_BAR_CSS, unsafe_allow_html=True)
    st.markdown(SURFACE_CSS, unsafe_allow_html=True)


def navigation() -> Any:
    """Build the grouped navigation from :data:`PAGES`.

    Returns:
        The Streamlit navigation object, ready to run. The first page is the
        default, which is why the product is listed first.
    """
    return st.navigation(
        {
            group: [
                st.Page(page.module, title=page.label, icon=page.icon, default=(index == 0))
                for index, page in enumerate(PAGES)
                if page.group == group
            ]
            for group in groups()
        }
    )


def masthead() -> None:
    """Draw the product's own name at the top of the page that *is* the product.

    Not a page title. Every other page opens with :func:`header`, which names the
    page and the module its figures came from, because those pages report on the
    project. The chat page is the project, and what belongs at the top of it is what
    the thing is called and what it does. It is the only place either is written: the
    sidebar carries no identity caption, so a page that lost this one would be a page
    that cannot be screenshotted.

    What is deliberately *not* here is a page title. The navigation two centimetres
    to the left already names the page, and a heading that repeats the menu item
    above it spends the most valuable line on the page saying nothing.
    """
    # The name, and directly under it what it means and how to say it, all on one
    # line: they are one thing -- a name, what it means, how to say it -- and as a
    # heading over two caption lines they read as three claims stacked above the box
    # somebody came to type in. "Quay" is heard as "kway" by roughly half of any
    # audience, and the line that fixes that belongs under the name where somebody
    # is already looking rather than in a sidebar column that collapses.
    #
    # Raw `<h4>` rather than `####`, because everything inside a markdown heading
    # renders at heading size and the descriptor would then carry the same weight as
    # the name. What lifts this line up against the header strip is not a margin here
    # but the page padding `TOP_BAR_CSS` trims, so the circuit under it starts high
    # and the space goes to the middle of the page instead.
    st.markdown(
        f'<h4 style="margin:0 0 0.35rem;">{APP_NAME}'
        f'&nbsp;&nbsp;<span style="font-size:0.60rem; font-weight:400; opacity:0.62;">'
        f"{APP_DESCRIPTOR}&nbsp; · &nbsp;<em>{APP_PRONUNCIATION}</em></span></h4>",
        unsafe_allow_html=True,
    )
    # Under the name, not above it. `TOP_BAR_CSS` lifts these onto the header strip
    # on a window wide enough to share it; on a narrower one they stay in the page,
    # and a page whose first line is somebody else's account controls has given the
    # most valuable line away.
    access.account_bar(lead=new_chat_button)


def new_chat_button() -> None:
    """Offer to start a fresh conversation, beside the one it would clear.

    On the page rather than in the sidebar, and next to the thread it acts on. In
    the sidebar it was directly below the navigation's ``💬 Chat`` link, so the
    column carried two controls that read as the same affordance -- and it was the
    only sidebar item that acted on one page instead of reporting something every
    page needs.

    Nothing is deleted. The thread key is dropped, so the next question starts a new
    conversation and the old one is still listed under **Past chats** with whatever
    the agent remembered of it.
    """
    if st.button("🆕 New chat"):
        for key in ("thread", "history"):
            st.session_state.pop(key, None)
        st.rerun()


def header(title: str, lede: str, source: str) -> None:
    """Draw a page's title, its one-line purpose, and where its numbers come from.

    The third argument is required rather than optional. A monitor is only worth
    reading if a reader can go and check it, and naming the module every figure was
    read from is the cheapest possible way to make that true.

    Args:
        title: The page title.
        lede: One sentence on what the page is for.
        source: Where the page's numbers come from, as a module path or a command.
    """
    st.markdown(f"#### {title}")
    st.caption(lede)
    st.caption(f"Every number on this page is read from `{source}`.")
    # The same account row the chat page's masthead draws, on every other page. It
    # was reachable from Ask alone, so a signed-in reader who had navigated anywhere
    # else had no way to sign out and a guest no way to offer their own key.
    #
    # After the block rather than before it, so a narrow window opens a page with
    # its own name rather than with somebody's account controls -- the rule the
    # masthead already follows -- and because a title and the two captions that
    # qualify it are one unit that a row of buttons should not split. On a window
    # wide enough to share the header strip, `TOP_BAR_CSS` lifts the row out of the
    # page and this position stops mattering.
    access.account_bar()


def hamiltonian(note: str | None = None, longitudinal_field: float = 0.0) -> None:
    """Draw the model this whole project is about.

    **Two terms unless the longitudinal field is on.** The equation on a page is
    there to tell a reader what problem is being solved, and this project solves a
    problem with one dimensionless knob, $h/J$. A third symbol that is zero
    everywhere by default and moves no number on any screen reads as a third free
    parameter, and it was the single most confusing thing on the chat page. It
    reappears, in the same place and with no other change, the moment somebody sets
    ``g`` to something -- see :func:`src.physics.model.hamiltonian_display`.

    Args:
        note: An optional sentence beneath it, for a page that wants to say which
            part of the expression it is varying.
        longitudinal_field: The ``g`` of the setting being described. Non-zero adds
            the third term, because at that point the reader *is* solving a
            three-parameter problem and needs to see it.
    """
    st.markdown(hamiltonian_display(longitudinal_field))
    if note:
        st.caption(note)


LETTERHEAD_WIRES = 4
"""Wires the circuit strip draws before it truncates the register.

Four, because the strip runs the whole width of the page above the question box and
a taller picture pushes the box itself off the first screen. Four is also enough to
show the structure -- two bonds firing together, then the one between them.
"""

LETTERHEAD_LAYERS = 4
"""Layers it draws before it truncates the depth, which is where its length comes from."""

LETTERHEAD_ASPECT = 5.2
"""Least width the strip may have per unit of height.

It fills the column it sits in, so its own proportions decide how tall that makes
it. A short circuit is spread across the full width rather than drawn compact and
then stretched into a band deep enough to push the question box off the screen.
"""

LETTERHEAD_LONGEST = 8.0
"""Most width per unit of height, which is what caps how many layers get drawn.

The other end of the same constraint: at a fixed page width a longer picture is a
shorter one, and past this the gates are too small to read. A circuit that would
overrun it loses a layer and says so with dots.
"""

_LETTERHEAD_ROW = 28.0
"""Drawing units between one wire and the next."""

_LETTERHEAD_EDGE = 22.0
"""Units of clear space above the first wire and below the last."""

_LETTERHEAD_LEAD = 64.0
"""Units from the left edge to the first column, leaving room for the state labels."""

_LETTERHEAD_STEP = 92.0
"""Units between columns once the circuit is long enough to set its own width."""

_LETTERHEAD_BOND = "#5bb4e5"
"""Sky blue for the coupling.

The two operators keep one colour each throughout the application -- one couples
neighbours, the other knocks a single magnet sideways -- and a masthead is chrome
that is on screen before anybody has asked anything, so the pair is a light sky and
its warm complement. Saturated ink at the top of every visit competes with the
question box, which is what the page is for.
"""

_LETTERHEAD_FIELD = "#efa677"
"""Warm sand for the transverse field, the sky's complement at the same lightness."""

_LETTERHEAD_WIRE = "#9dbdd6"
"""Paler sky for the Hadamard boxes, which are structure rather than a learned angle."""

_LETTERHEAD_SHADOW = "#1b2430"
"""What a chip is lifted off the wire with.

A fixed near-black rather than the theme's own text colour, which on a dark
background would ring every gate with a halo instead of a shadow.
"""

_SIGMA_Z_LABEL = '&#963;<tspan dy="-4" font-size="8">z</tspan>'
_SIGMA_X_LABEL = '&#963;<tspan dy="-4" font-size="8">x</tspan>'
"""What the single-qubit boxes say, as a raised letter rather than a bare Z or X.

A box labelled "X" reads as the Pauli gate, which is not what these are: they are
rotations generated by that operator through an angle the optimiser learns. The
letter is raised by a ``tspan`` offset rather than typed as a superscript
character, because a font missing that glyph would silently draw nothing, and the
sigma is written as an entity for the same reason.
"""


_LETTERHEAD_DEFS = "".join(
    f'<linearGradient id="quay-{colour.lstrip("#")}" x1="0" y1="0" x2="0" y2="1">'
    f'<stop offset="0" stop-color="{colour}" stop-opacity="0.34"/>'
    f'<stop offset="1" stop-color="{colour}" stop-opacity="0.10"/>'
    "</linearGradient>"
    for colour in (_LETTERHEAD_BOND, _LETTERHEAD_FIELD, _LETTERHEAD_WIRE)
)
"""One gradient fill per colour, defined once and referenced by id.

A gate drawn as a flat tint reads as a coloured rectangle; the same chip lit from
the top reads as a component sitting on a wire, which is what it is. Each fill's id
comes from its own colour, so a colour brings its own.

Gradients and nothing else -- no filter. This markup is parsed as HTML before it
reaches the page, and only the element names the specification lists get their
capitals back, so a drop shadow is drawn below as a rectangle instead.
"""


def _letterhead_gate(x: float, y: float, label: str, colour: str) -> str:
    """One boxed single-qubit gate, as a lit chip on the wire.

    Args:
        x: Centre of the column.
        y: Centre of the wire.
        label: What the box says, as SVG text content.
        colour: Stroke and text colour; the fill is its gradient.

    Returns:
        The markup for one gate.
    """
    tint = colour.lstrip("#")
    return (
        '<g class="gate">'
        f'<rect x="{x - 28:.1f}" y="{y - 9.4:.1f}" width="58" height="22" rx="8" '
        f'fill="{_LETTERHEAD_SHADOW}" fill-opacity="0.1"/>'
        f'<rect x="{x - 29:.1f}" y="{y - 11:.1f}" width="58" height="22" rx="8" '
        f'fill="url(#quay-{tint})" stroke="{colour}" stroke-width="1.2"/>'
        f'<text x="{x:.1f}" y="{y:.1f}" fill="{colour}" font-size="13" '
        f'text-anchor="middle" dominant-baseline="central">{label}</text>'
        "</g>"
    )


def _letterhead_bond(x: float, upper: float, lower: float) -> str:
    r"""One :math:`\hat\sigma^z\hat\sigma^z` rotation, as a connector between two wires.

    Args:
        x: Centre of the column.
        upper: Wire the connector starts on.
        lower: Wire it ends on.

    Returns:
        The markup for one bond rotation.
    """
    ends = "".join(
        f'<circle cx="{x:.1f}" cy="{end:.1f}" r="5" fill="{_LETTERHEAD_BOND}"/>'
        f'<circle cx="{x - 1.5:.1f}" cy="{end - 1.5:.1f}" r="1.7" fill="#ffffff" '
        f'fill-opacity="0.45"/>'
        for end in (upper, lower)
    )
    return (
        '<g class="bond">'
        f'<line x1="{x:.1f}" y1="{upper:.1f}" x2="{x:.1f}" y2="{lower:.1f}" '
        f'stroke="{_LETTERHEAD_BOND}" stroke-width="3" stroke-linecap="round"/>' + ends + "</g>"
    )


def _letterhead_continues(x: float, y: float, across: bool) -> str:
    """Three faint dots saying the circuit goes on past what the strip drew.

    Drawn rather than written, because the strip carries no words: a chain longer or
    a circuit deeper than there is room for is a fact about the picture and belongs
    in it.

    Args:
        x: Centre of the group.
        y: Centre of the group.
        across: Lay the dots along a wire, for depth; otherwise down the register.

    Returns:
        The markup for one ellipsis.
    """
    dots = "".join(
        f'<circle cx="{x + offset if across else x:.1f}" '
        f'cy="{y if across else y + offset:.1f}" r="1.8" fill="currentColor" '
        f'fill-opacity="0.42"/>'
        for offset in (-6.5, 0.0, 6.5)
    )
    return f'<g class="continues">{dots}</g>'


def ansatz_svg(spec: AnsatzSpec) -> str:
    r"""Draw the circuit a setting describes, as a strip for the top of a page.

    Every gate comes from ``spec``: the bonds are grouped by
    :attr:`~src.physics.quantum.ansatz.AnsatzSpec.rounds`, which is the schedule the
    depth arithmetic prices, and the order -- Hadamards, then the diagonal half,
    then the field half, once per layer -- is the order
    :func:`~src.physics.quantum.statevector.evolve` applies them in. A letterhead
    showing a plausible circuit instead of this one would be the only picture in the
    application that does not describe what runs.

    Inline SVG rather than a raster figure, because this sits at masthead height on
    every visit: it stays sharp at any zoom, costs no plotting, and the ground, the
    wires and the layer bands are drawn in ``currentColor``, so the chrome follows
    whichever theme the reader is in while the two operators keep the colours they
    have everywhere else.

    Args:
        spec: The circuit to draw, truncated to :data:`LETTERHEAD_WIRES` and
            :data:`LETTERHEAD_LAYERS`. A bond that wraps a ring is left out, since it
            would draw straight through every gate between its ends.

    Returns:
        Markup for :func:`streamlit.markdown` with ``unsafe_allow_html``.
    """
    wires = min(spec.n_qubits, LETTERHEAD_WIRES)
    layers = min(spec.depth, LETTERHEAD_LAYERS)
    wire_y = [_LETTERHEAD_EDGE + _LETTERHEAD_ROW * wire for wire in range(wires)]
    height = wire_y[-1] + _LETTERHEAD_EDGE
    # Only the rounds with a bond inside the truncated register take a column.
    rounds = [
        group
        for group in spec.rounds
        if any(max(one, other) < wires and abs(one - other) == 1 for one, other in group)
    ]
    per_layer = len(rounds) + (2 if spec.longitudinal else 1)
    # Drop a layer rather than draw one too small to read, and again if need be.
    while True:
        tail = _LETTERHEAD_STEP if spec.depth > layers else 2 * _LETTERHEAD_EDGE
        least = _LETTERHEAD_LEAD + _LETTERHEAD_STEP * layers * per_layer + tail
        if layers <= 1 or least <= height * LETTERHEAD_LONGEST:
            break
        layers -= 1
    columns = 1 + layers * per_layer
    width = max(least, height * LETTERHEAD_ASPECT)
    # The columns share whatever the width is, so a short circuit spreads across the
    # page instead of leaving the right half of it as bare wire.
    step = (width - _LETTERHEAD_LEAD - tail) / (columns - 1) if columns > 1 else 0.0

    marks = [_letterhead_gate(_LETTERHEAD_LEAD, y, "H", _LETTERHEAD_WIRE) for y in wire_y]
    bands: list[str] = []
    column = _LETTERHEAD_LEAD
    for _ in range(layers):
        opening = column + step / 2
        for group in rounds:
            column += step
            for one, other in group:
                if max(one, other) >= wires or abs(one - other) != 1:
                    continue
                marks.append(_letterhead_bond(column, wire_y[one], wire_y[other]))
        if spec.longitudinal:
            column += step
            marks += [_letterhead_gate(column, y, _SIGMA_Z_LABEL, _LETTERHEAD_BOND) for y in wire_y]
        column += step
        marks += [_letterhead_gate(column, y, _SIGMA_X_LABEL, _LETTERHEAD_FIELD) for y in wire_y]
        # One quiet band per layer, so the repeating block reads as a block without
        # a word of caption to say so.
        bands.append(
            f'<rect x="{opening:.1f}" y="7" width="{column + step / 2 - opening:.1f}" '
            f'height="{height - 14:.1f}" rx="13" fill="currentColor" fill-opacity="0.032"/>'
        )
    if spec.depth > layers:
        marks += [_letterhead_continues(column + step / 2, y, across=True) for y in wire_y]
    if spec.n_qubits > wires:
        marks.append(
            _letterhead_continues(
                _LETTERHEAD_LEAD - 49, wire_y[-1] + _LETTERHEAD_EDGE * 0.62, across=False
            )
        )

    first, last = _LETTERHEAD_LEAD - 34, width - 16
    wire_marks = [
        f'<line x1="{first:.1f}" y1="{y:.1f}" x2="{last:.1f}" y2="{y:.1f}" '
        f'stroke="url(#quay-wire)" stroke-width="1.1"/>'
        f'<text x="{_LETTERHEAD_LEAD - 39:.1f}" y="{y:.1f}" fill="currentColor" '
        f'fill-opacity="0.5" font-size="11" text-anchor="end" '
        f'dominant-baseline="central">|0&#10217;</text>'
        for y in wire_y
    ]
    # The wires fade out at both ends rather than stopping, and the ground is a wash
    # of the same two colours: it is a masthead, and an edge that ends in a hard line
    # reads as a figure somebody cropped.
    defs = (
        f"<defs>{_LETTERHEAD_DEFS}"
        f'<linearGradient id="quay-wire" gradientUnits="userSpaceOnUse" '
        f'x1="{first:.1f}" y1="0" x2="{last:.1f}" y2="0">'
        f'<stop offset="0" stop-color="currentColor" stop-opacity="0.05"/>'
        f'<stop offset="0.07" stop-color="currentColor" stop-opacity="0.3"/>'
        f'<stop offset="0.93" stop-color="currentColor" stop-opacity="0.3"/>'
        f'<stop offset="1" stop-color="currentColor" stop-opacity="0.05"/>'
        f"</linearGradient>"
        f'<linearGradient id="quay-ground" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0" stop-color="{_LETTERHEAD_BOND}" stop-opacity="0.09"/>'
        f'<stop offset="0.6" stop-color="{_LETTERHEAD_BOND}" stop-opacity="0.02"/>'
        f'<stop offset="1" stop-color="{_LETTERHEAD_FIELD}" stop-opacity="0.07"/>'
        f"</linearGradient></defs>"
    )
    ground = (
        f'<rect x="0.6" y="0.6" width="{width - 1.2:.1f}" height="{height - 1.2:.1f}" '
        f'rx="17" fill="url(#quay-ground)" stroke="currentColor" stroke-opacity="0.07"/>'
    )
    carries_on = spec.depth > layers or spec.n_qubits > wires
    alt = (
        f"A {wires}-wire quantum circuit. Each wire starts in the state written zero "
        f"and passes through a Hadamard gate, then {layers} repeated "
        f"{'layer' if layers == 1 else 'layers'} of sky-blue connectors coupling "
        f"neighbouring wires and sand-coloured boxes turning each wire on its own"
        f"{', with dots where it carries on past the drawing' if carries_on else ''}."
    )
    return (
        f'<svg viewBox="0 0 {width:.1f} {height:.1f}" role="img" aria-label="{alt}" '
        f'xmlns="http://www.w3.org/2000/svg" font-family="inherit" '
        f'style="width:100%;height:auto;display:block;margin:-0.1rem 0 0;">'
        + defs
        + ground
        + "".join(bands)
        + "".join(wire_marks)
        + "".join(marks)
        + "</svg>"
    )


def ansatz_letterhead(physics: knob.Physics) -> None:
    """Open a page with the circuit the setting describes rather than its equation.

    An equation suits a reader who already knows what the terms are, and not
    somebody deciding whether to type a question. The picture carries the same two
    competing terms and is also the thing the agent prices, so it says more and asks
    less. The equation sits on Quay Lab, under the drawing of the chain.

    No caption. It is a masthead above the box the page is for, and a paragraph
    explaining a picture that spans the window is text somebody has to get past to
    reach the question. What the strip cannot draw it says with dots.

    Args:
        physics: The knob's position. The circuit tracks it, so the chain length,
            the depth and the longitudinal field all move the picture.
    """
    # The gap below the strip is padding on a wrapper rather than a margin on the
    # drawing: the drawing is a flex item's only child, so a bottom margin there
    # collapses away and moves nothing. The gap is the point -- the strip and the name
    # above it are chrome, and the controls under it are the page, so the two want
    # daylight between them rather than a stack of evenly spaced blocks.
    # And a negative top margin, for the same reason in the other direction: the name
    # above is one short line of chrome, and the daylight belongs under the strip
    # rather than split evenly above and below it. The drawing carries -0.2rem of its
    # own, so this is the rest of the lift.
    #
    # It is this large because the space above is two of Streamlit's block gaps
    # rather than one: the account row between the name and this drawing is lifted
    # onto the header strip and takes no height on a wide window, but the container
    # it left behind is still an item in the column and still gets a gap on each
    # side. So the drawing sat about two rem below a heading with nothing between
    # them. This pulls one of those gaps back.
    st.markdown(
        '<div style="margin-top:-1.2rem; padding-bottom:1.35rem;">'
        + ansatz_svg(
            AnsatzSpec(
                n_qubits=physics.n_sites,
                depth=physics.depth,
                boundary=physics.boundary,
                longitudinal=physics.longitudinal > 0.0,
            )
        )
        + "</div>",
        unsafe_allow_html=True,
    )


def conversation() -> Memory:
    """This browser session's memory, falling back to none if it cannot be had.

    Memory lives in a file whose location comes from configuration that may not be
    present on a fresh checkout. Rendering must never depend on it, so a failure
    here degrades to a memory that stores nothing rather than to a page that will
    not draw -- and :attr:`~src.agent.memory.Memory.enabled` is what the panel reads
    to say so on screen, because a feature that fails silently is worse than one
    that is switched off.

    Lives here rather than on the chat page because the sidebar draws the memory
    panel on every page and the chat page runs on one.

    Returns:
        The memory for this session's thread.
    """
    if "thread" not in st.session_state:
        st.session_state["thread"] = f"ui-{uuid4().hex[:8]}"
    try:
        return Memory(user="visitor", thread=st.session_state["thread"])
    except Exception:
        return Memory.disabled()


def open_conversation(name: str) -> None:
    """Switch the session to an earlier conversation.

    The on-screen history is cleared rather than reloaded: what could be redrawn
    from a stored thread is a summary of each turn, not the campaign behind it, and
    a scrollback that had silently lost its evidence would be worse than an empty
    one. What comes back is the recall -- the next question is answered in the light
    of that conversation, which is the part that was actually kept.

    Args:
        name: The conversation's key.
    """
    st.session_state["thread"] = name
    st.session_state.pop("history", None)
    st.rerun()


def conversation_controls(store: Memory, on_open: Callable[[str], None]) -> None:
    """Draw the thread controls: start a new one, reopen an old one, read the memory.

    Directly below the settings knob and on **every** page, which is a change from
    where these used to live. They were drawn by the chat page, so clicking away from it
    made the whole conversation disappear from the sidebar -- and with it the only
    place a reader can see what the agent is holding about them and clear it. A
    memory that is inspectable on one page out of seven is not meaningfully
    inspectable, and this is the one control in the application with a privacy claim
    attached to it.

    Args:
        store: This session's memory.
        on_open: Called with a conversation's key when its button is pressed.
    """
    with st.sidebar:
        # No rule between the knob and these. One was tried, on the argument that the
        # two answer different questions -- what this run is set to, and what this
        # conversation is carrying -- and it bought a separation nobody needed at the
        # price of pushing both expanders down the column. Three collapsed expanders
        # are already three lines; a reader scans them faster in one block.
        # "🕘 Past chats" and "🧠 Memory" -- named for the thing rather than for the
        # mechanism, and glyphed so the column is scannable without reading it.
        #
        # "🆕 New chat" was here and is now on the chat page, beside the heading.
        # In a sidebar it sat directly under the navigation's own "💬 Chat" link,
        # which is two chat-shaped controls in a column two centimetres apart --
        # and it is the only one of these three that acts on *this page's* thread
        # rather than reporting something every page needs. Past chats and Memory
        # stay, because they carry the privacy claim: a memory inspectable on one
        # page out of seven is not meaningfully inspectable.
        past_chats(store, on_open)
        memory_panel(store)
        # Under the memory panel, on every page, for the reason in its own docstring.
        session_total_line()


def memory_panel(store: Memory) -> None:
    """Show what the agent is holding from this conversation, and offer to clear it.

    An agent that reads a follow-up in the light of an earlier question has to be
    able to say what it is carrying, or the adjustment is indistinguishable from the
    model having a bad day. Forgetting is offered in the same place and for the same
    reason: a memory a reader cannot inspect and cannot clear is a memory they did
    not agree to.

    What is shown is what is actually stored, read back off disk rather than
    reconstructed from this session -- so a claim here that something was remembered
    is a claim about the file, which is the only version that matters after the tab
    is closed.

    Args:
        store: This conversation's memory.
    """
    recalled = store.recall()
    with st.expander(f"🧠 Memory ({len(recalled)})", expanded=False):
        if not store.enabled:
            st.caption(
                f"{ABSENT} Switched off -- no memory file is configured, so every "
                "question starts from nothing. Follow-ups will not know what came "
                "before them."
            )
            return
        if not recalled:
            st.caption(
                "Nothing yet. After the first answer, a follow-up is read in the "
                'light of it -- ask about six magnets, then ask "what about '
                'twenty?" and the second question is answered even though it '
                "carries no chain length of its own."
            )
        else:
            st.caption("Carried into the next question, oldest first:")
            for entry in recalled:
                st.markdown(f"<sub>{entry.summarise()}</sub>", unsafe_allow_html=True)
        held = store.describe()
        taught = learned_audience(store)
        if taught is not None:
            st.caption(
                f"**Learned from your ratings:** write for a reader who "
                f"{AUDIENCE_LABELS.get(taught, taught)}. This follows you into the "
                "next conversation — a reading level is a fact about the reader "
                "rather than about one thread — and the settings knob overrides it."
            )
        st.caption(
            f"{held['turns']} turn{'' if held['turns'] == 1 else 's'} and "
            f"{held['ratings']} rating{'' if held['ratings'] == 1 else 's'} stored for "
            f"`{held['user']}`. Questions the injection screen refused are never "
            "written: memory is replayed into a later prompt, so storing one would "
            "turn a single attempt into a standing one."
        )
        _forget_everything_button(store)


def _forget_everything_button(store: Memory) -> None:
    """Offer to erase this user's whole memory, in two presses.

    The label used to read *Forget this conversation* while calling
    :meth:`~src.agent.memory.Memory.forget`, which erases every thread this user
    has -- so the one control in the application that destroys the most said it
    destroyed the least, and it did it on a single press while deleting one chat
    asked twice. Both halves of that are fixed here.

    Args:
        store: This user's memory.
    """
    if st.session_state.get("memory_forget_armed"):
        if st.button(
            "✓ Yes, forget all of it",
            width="stretch",
            type="primary",
            help="Erases every conversation stored for you, not just this one.",
        ):
            removed = store.forget()
            st.session_state.pop("memory_forget_armed", None)
            st.session_state.pop("history", None)
            st.toast(f"Forgot {removed} entr{'y' if removed == 1 else 'ies'}.")
            st.rerun()
        if st.button("Cancel", width="stretch"):
            st.session_state.pop("memory_forget_armed", None)
            st.rerun()
        return
    if st.button(
        "Forget everything",
        width="stretch",
        help="Erases every conversation stored for you. Asks once more before it does.",
    ):
        st.session_state["memory_forget_armed"] = True
        st.rerun()


VECTOR_SHARE_LABEL = r"Vector search weight $\alpha$"
"""How the retrieval weighting dial is labelled, on both surfaces that draw it.

One constant so the sidebar and the chat page cannot drift apart. Alpha is what the
retrieval literature calls the vector half's weight, and writing it as LaTeX keeps
this file ASCII -- a literal Greek alpha trips ruff's RUF001 -- and matches the
``Sites $L$`` dials.

One name for one dial, wherever it is drawn. A control that is called two things in
two places is a control a reader has to learn twice.
"""

SHELF_CHOICE_LABELS: dict[str, str] = {
    "decide for me": "Let the router choose",
    "physics-notes": "The model itself (physics-notes)",
    "quantum-computing": "Quantum computing (quantum-computing)",
    "method-comparison": "Method comparison (method-comparison)",
    "applications": "Business, ML and R&D uses (applications)",
}
"""How each knowledge base reads on screen.

The default has to say what it *does* rather than what it is called, because a
reader meeting "decide for me" in a dropdown cannot tell a policy from a
placeholder.
"""


SLOW_ON_CHEAP_WORK = 15
"""Seconds a strong model adds to one question's cheap calls, measured.

From a real session: nine calls, all of them served by `claude-haiku-4.5` because
one dropdown had set every tier at once. Four of those nine are *fast-tier* work --
rewriting a search query, choosing a shelf, expanding a query into paraphrases, and
permuting four identifiers -- and they took 22.3 s between them. The same four on
`gemini-2.5-flash-lite`, measured by ``make timing``, take 6.9 s. Fifteen seconds of
a fifty-two-second answer, spent on judgements about *wording* by a model chosen for
its physics.

Rounded down and stated as a whole number, because the point is the order of
magnitude and a figure with a decimal point invites somebody to trust it further
than one measurement deserves.
"""


def one_model_everywhere(tiers: tuple[tuple[str, str], ...]) -> None:
    """Say what it costs when all three tiers are pointed at the same model.

    **A configuration that hides its own consequence.** Setting one model everywhere
    is exactly what the bake-off needs -- the project's central claim is that the
    numbers do not move when the model does, and checking that means running one
    model across the whole ladder. It is also how somebody reaches for the best model
    available, puts it on the six cheap calls a question makes before it reaches the
    report, and then quite reasonably reports that the application is slow.

    So the cost is stated beside the three dials that produce it, with the number
    that was measured rather than an adjective. Not a warning and not a block:
    flattening the ladder is sometimes the right thing to do, and a control that nags
    is one people learn to dismiss.

    Silent when the three differ, which is the normal case, and silent when they are
    all the *cheap* model -- that costs nothing and is what a guest already gets.

    Args:
        tiers: The three ``(tier, slug)`` pairs as the selectors now stand.
    """
    slugs = {slug for _, slug in tiers}
    if len(slugs) != 1:
        return
    slug = slugs.pop()
    fast_default = DEFAULT_TIER_SLUGS["fast"]
    if slug == fast_default:
        return
    st.caption(
        f":grey[All three tiers are **{slug.split('/')[-1]}**, so it will serve the "
        f"cheap calls too — the search rewrite, the shelf choice, the query "
        f"expansion, the passage ordering. Measured, that adds about "
        f"**{SLOW_ON_CHEAP_WORK} s** per question against "
        f"`{fast_default.split('/')[-1]}` on those, and changes no number. Worth it "
        f"to compare models on equal footing; not worth it otherwise.]"
    )


def search_note(vector_share: float) -> str:
    """Say what this weighting gives up, in one sentence.

    Both ends of the slider lose something real, and a control that advertises only
    what it improves invites a reader to drag it to an end and then wonder why the
    answers got worse.

    Args:
        vector_share: The slider's position, as the vector half's weight.

    Returns:
        The sentence. Never empty: the middle of the range has a claim to make too.

    Examples:
        >>> "keyword" in search_note(1.0)
        True
        >>> "paraphrase" in search_note(0.0)
        True
    """
    if vector_share >= 1.0:
        return (
            "Vector search alone — keyword matching is off, so a question naming an "
            "author or an arXiv id no longer matches, because that text sits in a "
            "note's frontmatter and was never embedded."
        )
    if vector_share <= 0.0:
        return (
            "Keyword search alone — paraphrase stops working, and “why does the gap "
            "close” no longer finds a passage that says the dispersion vanishes."
        )
    return "A passage both halves rank outranks one that only either liked."


def chat_settings(base: knob.Setting) -> knob.Setting:
    r"""Draw the chat page's own knobs and return what to ask with.

    A second control, deliberately. The sidebar configures the session; this is where
    the dials worth reaching for *mid conversation* live, without scrolling past a
    request timeout to get to them.

    **Two columns, and the split is the project's central claim laid out as a
    layout.** Everything in the left column changes how an answer is *written* and
    what it is *grounded in*; everything in the right column changes the problem being
    *solved*. Drag anything on the left and no number below moves. That is a claim a
    sceptic can check in one gesture, and it was unavailable while the same dials were
    spread across four tabs that had to be opened one at a time.

    The four dials at the top of the left column are the ones worth changing between
    two questions -- which model answers, who it is written for, how much of the notes
    it may draw on, and how the notes are searched. Everything that only matters while
    something is being tested is behind **Developer knobs**, so it is not competing
    for attention with them.

    Every dial is named for what it changes rather than for the parameter behind it:
    *Model*, *Explain it for*, *Passages to use*, *Vector search weight*
    :math:`\alpha`, *Searches per question*, *Knowledge base*, *Sites* :math:`L`,
    *Coupling* :math:`J`, *Field* :math:`h`, *Boundary*. The same dial carries the
    same name here and in the sidebar, from one constant where the wording is long
    enough to drift -- see :data:`VECTOR_SHARE_LABEL`.

    Precedence is explicit rather than implied. Left alone it follows the sidebar
    exactly and the caption says so; tick the box and this page wins for questions
    asked from it. The alternative -- two sets of live widgets bound to one value --
    has a rule nobody can see, and whichever loses silently discards a dial somebody
    has just moved.

    Args:
        base: The sidebar knob, which this may override.

    Returns:
        What to ask with: ``base`` itself, or a copy with the overridden fields
        replaced.
    """
    # The same gear as the sidebar header, because it is the same idea in a second
    # place. A different glyph here would suggest a different kind of control.
    with st.expander("⚙️ Chat settings"):
        override = st.checkbox(
            "Set them here for this chat",
            value=False,
            key="chat_override",
            help="Unticked, this page follows the sidebar.",
        )
        if not override:
            level = AUDIENCE_LABELS.get(base.model.audience, base.model.audience)
            st.caption(
                f"Following the sidebar: **{base.model.audience}** ({level}), "
                f"{base.physics.n_sites} sites, "
                f"J = {base.physics.coupling:g}, h = {base.physics.field:g}, "
                f"{base.physics.depth} circuit layers on **{base.physics.device}**, "
                f"{base.physics.shots:,} shots, "
                f"{base.search.passages} passages, "
                rf"$\alpha$ = {base.search.vector_share:.1f}."
            )
            return base

        model_column, physics_column = st.columns(2)

        with model_column:
            st.markdown("**Model and retrieval**")
            st.caption(
                "Nothing here changes a computed number. Turn the temperature up, "
                "switch the model, and the energy, the depth and the verdict come "
                "back identical."
            )
            # **This sets the model that writes the answer, and only that.** It used
            # to be labelled "Model" and to write its value to all *three* tiers,
            # which broke twice over. Its default was `index=0` of a list beginning
            # with the **fast** tier's slug -- `DEFAULT_TIERS` is ordered fast,
            # standard, strong -- so it displayed `gemini-2.5-flash-lite` while the
            # strong tier was really `claude-haiku-4.5`, and ticking the override
            # above without touching it silently demoted the written report to the
            # cheap model. Used the other way, picking the strongest model put it on
            # the six cheap calls a question makes before it reaches the report:
            # measured at about fifteen extra seconds per question, for no change in
            # any number.
            #
            # Naming the tier it actually sets fixes both. The default is the strong
            # tier's own value, so the control opens on the model that is genuinely
            # about to write the answer; the cheap calls stay on the fast tier where
            # they belong; and the sidebar remains the place to set all three -- which
            # is what the bake-off needs and what `one_model_everywhere` prices.
            answering = dict(base.model.tiers).get("strong", DEFAULT_TIER_SLUGS["strong"])
            offered = list(dict.fromkeys([answering, *featured_slugs()]))
            answering_model = st.selectbox(
                "Answering model",
                options=offered,
                index=0,
                key="chat_model",
                disabled=not knob.credential_available() or not access.visitor().may_choose_models,
                help=(
                    "The model that writes the explanation and the report — the "
                    "strong tier. The cheap calls that come before it stay on the "
                    "fast tier, because a strong model rewriting a search query "
                    "costs seconds and changes nothing. Set all three separately in "
                    "the sidebar."
                ),
            )
            audience = _as_audience(
                st.select_slider(
                    "Explain it for",
                    options=list(AUDIENCE_LABELS),
                    value=base.model.audience,
                    format_func=lambda value: f"{value} — {AUDIENCE_LABELS[value]}",
                    key="chat_audience",
                    help=(
                        "The numbers are computed and cross-checked before any of this "
                        "is consulted, so they are identical at every level."
                    ),
                )
            )
            passages = st.slider(
                "Passages to use",
                min_value=1,
                max_value=knob.MAX_PASSAGES,
                value=base.search.passages,
                key="chat_passages",
                help=(
                    "The most one answer may cite from the notes. The search always "
                    "looks at more than this and the grader throws the rest away, so "
                    "raising it admits weaker passages rather than finding better ones."
                ),
            )
            vector_share = st.slider(
                VECTOR_SHARE_LABEL,
                min_value=0.0,
                max_value=1.0,
                value=base.search.vector_share,
                step=0.1,
                key="chat_vector_share",
                help=(
                    "The vector half's weight: 1.0 is vector search alone; 0.0 is "
                    "keyword search alone. Worth moving mid-conversation, because "
                    "which half answers depends on what you just typed."
                ),
            )
            # The number is the honest label: a "50 / 50" caption under a slider
            # reading 0.7 is the kind of drift that makes a control untrustworthy.
            st.caption(
                rf"$\alpha$ = {vector_share:.1f}, so ranking is "
                rf"{vector_share:.0%} vector search and "
                rf"{1 - vector_share:.0%} keyword search. " + search_note(vector_share)
            )
            rounds = st.slider(
                "Searches per question",
                min_value=1,
                max_value=knob.MAX_SEARCH_ROUNDS,
                value=base.search.rounds,
                key="chat_rounds",
                help=(
                    "Two lets a search that found nothing be rewritten from the "
                    "grader's own verdict and tried again. One switches that "
                    "corrective loop off, which is how you see what it was doing."
                ),
            )
            shelf = st.selectbox(
                "Knowledge base",
                options=list(knob.SHELF_CHOICES),
                index=list(knob.SHELF_CHOICES).index(base.search.shelf),
                format_func=lambda name: SHELF_CHOICE_LABELS.get(name, name),
                key="chat_shelf",
                help=(
                    "Left to the router, the shelf is chosen from the question's own "
                    "vocabulary and raises that knowledge base to the top rather than "
                    "removing the others, so a wrong guess costs the first few results."
                ),
            )
            corpus = st.checkbox(
                "Search the project's notes",
                value=base.search.corpus,
                key="chat_corpus",
                help=(
                    "The keyword half needs no model, so this returns citations even "
                    "with the language model off."
                ),
            )
            external = st.checkbox(
                "Let it search arXiv",
                value=base.search.external,
                key="chat_external",
                help=(
                    "A network call. Always reached when the question asks for papers "
                    "— a fixed corpus cannot say what is recent — and otherwise only "
                    "when the notes come back with nothing usable. Anything fetched "
                    "this way is cited as unreviewed."
                ),
            )
            followups = st.checkbox(
                "Suggest what to ask next",
                value=base.search.followups,
                key="chat_followups",
                help=(
                    "Each suggestion is screened as if you had typed it. Off here "
                    "because a suggestion is a model call, and a campaign being "
                    "metered should be readable without one."
                ),
            )

            with st.expander("Developer knobs"):
                st.caption(
                    "How the gateway is called, rather than what is asked of it. "
                    "These change latency, cost and how a flaky connection is "
                    "handled — never a computed number."
                )
                offline = st.toggle(
                    "Run without a language model",
                    value=base.model.offline,
                    key="chat_offline",
                    # Nothing to switch on without a credential, and a live-looking
                    # toggle that cannot change the answer is the dead knob this
                    # panel's own audit was written to remove.
                    disabled=not knob.credential_available(),
                    help=(
                        "Every branch has a path that needs no model, except the one "
                        "that writes code — which says so rather than inventing one."
                    ),
                )
                temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=2.0,
                    value=base.model.temperature,
                    step=0.05,
                    key="chat_temperature",
                    help=(
                        "How much the model may vary its wording. Reachable precisely "
                        "so a sceptic can turn it up, ask the same question again, and "
                        "watch every number come back identical."
                    ),
                )
                cap_length = st.checkbox(
                    "Cap the reply length",
                    value=base.model.max_output_tokens is not None,
                    key="chat_cap_length",
                    help=(
                        "Unticked leaves the provider's own default in place, which on "
                        "some models is very long and is billed by the token."
                    ),
                )
                max_output_tokens = (
                    st.slider(
                        "Max output tokens",
                        min_value=knob.MAX_OUTPUT_TOKEN_BOUNDS[0],
                        max_value=knob.MAX_OUTPUT_TOKEN_BOUNDS[1],
                        value=base.model.max_output_tokens or knob.DEFAULT_MAX_OUTPUT_TOKENS,
                        step=64,
                        key="chat_max_output_tokens",
                    )
                    if cap_length
                    else None
                )
                max_calls = st.select_slider(
                    "Model calls per question",
                    options=list(knob.MAX_CALLS_CHOICES),
                    value=base.model.max_calls,
                    key="chat_max_calls",
                    help=(
                        "The ceiling that turns a runaway tool-calling loop into a "
                        "stopped run. A wall rather than a warning: a budget written "
                        "into a prompt is a suggestion; one that refuses the call is a "
                        "budget."
                    ),
                )
                max_retries = st.slider(
                    "Retries on transient failure",
                    min_value=0,
                    max_value=knob.MAX_RETRY_BOUND,
                    value=base.model.max_retries,
                    key="chat_max_retries",
                    help=(
                        "Only transient failures are retried — a rate limit or a "
                        "dropped connection. A refusal or a bad request is permanent "
                        "and is never tried again, however high this goes."
                    ),
                )
                requests_per_second = st.slider(
                    "Requests per second",
                    min_value=0.0,
                    max_value=knob.MAX_REQUESTS_PER_SECOND,
                    value=base.model.requests_per_second,
                    step=0.5,
                    key="chat_requests_per_second",
                    help=(
                        "Spacing between calls, and zero switches it off. An answer "
                        "makes about seven, so this is a direct multiplier on how long "
                        "one takes."
                    ),
                )
                request_timeout = st.slider(
                    "Request timeout (s)",
                    min_value=knob.REQUEST_TIMEOUT_BOUNDS[0],
                    max_value=knob.REQUEST_TIMEOUT_BOUNDS[1],
                    value=base.model.request_timeout_s,
                    step=5.0,
                    key="chat_request_timeout",
                    help=(
                        "How long one call may hang before it is abandoned and, if "
                        "retries are left, tried again."
                    ),
                )
                parallel_calls = st.checkbox(
                    "Start the opening calls together",
                    value=base.model.parallel_calls,
                    key="chat_parallel_calls",
                    help=(
                        "Three calls happen before any visible work: restating your "
                        "question, working out what kind of answer it wants, and "
                        "reading the chain's numbers out of it. None reads another's "
                        "answer, so they go out at once and the wait is the slowest "
                        "rather than the sum. Turn it off to put them back in order — "
                        "every number stays where it was."
                    ),
                )

        with physics_column:
            st.markdown("**Physics**")
            st.caption("The only group here that changes a computed number.")
            n_sites = st.slider(
                "Sites $L$",
                min_value=knob.MIN_SITES,
                max_value=knob.MAX_SITES,
                value=base.physics.n_sites,
                key="chat_n_sites",
                help="One magnet, and one qubit, each.",
            )
            coupling = st.slider(
                "Coupling $J$",
                min_value=0.1,
                max_value=2.0,
                value=base.physics.coupling,
                step=0.1,
                key="chat_coupling",
                help=("How strongly each magnet prefers to point the same way as its neighbours."),
            )
            field = st.slider(
                "Field $h$",
                min_value=0.0,
                max_value=3.0,
                value=base.physics.field,
                step=0.05,
                key="chat_field",
                help=(
                    "The sideways push that tries to knock every magnet over. Only its "
                    "ratio to the coupling matters, and at $h/J = 1$ the chain is "
                    "critical, which is where every approximate method has its hardest "
                    "time."
                ),
            )
            boundary = st.radio(
                "Boundary",
                options=("open", "periodic"),
                index=0 if base.physics.boundary == "open" else 1,
                format_func=lambda value: (
                    "open — a line with two ends" if value == "open" else "periodic — a ring"
                ),
                horizontal=True,
                key="chat_boundary",
                help=(
                    "A ring needs its two ends to interact, and no machine here wires "
                    "them together. It is the cheapest way to see what connectivity "
                    "costs."
                ),
            )
            st.divider()
            depth = st.slider(
                "Circuit layers",
                min_value=1,
                max_value=knob.MAX_CIRCUIT_DEPTH,
                value=max(base.physics.depth, 1),
                key="chat_depth",
                help=(
                    "Each layer adds entangling gates, so it buys accuracy and spends "
                    "coherence time."
                ),
            )
            device = st.selectbox(
                "Machine",
                options=list(device_names()),
                index=list(device_names()).index(base.physics.device),
                key="chat_device",
                help=(
                    "Circuits are placed on this device's wiring and checked against "
                    "its coherence time before anything is run."
                ),
            )
            shots = st.select_slider(
                "Measurement budget",
                options=list(knob.SHOT_CHOICES),
                value=base.physics.shots,
                format_func=lambda value: f"{value:,}",
                key="chat_shots",
                help=(
                    "A configuration costing more than what is left is refused on "
                    "arithmetic, before it spends anything."
                ),
            )
            precision = st.select_slider(
                "Target accuracy per site",
                options=list(knob.PRECISION_CHOICES),
                value=base.physics.precision,
                format_func=lambda value: f"{value:.0e}",
                key="chat_precision",
                help=(
                    "Measurements scale as the inverse square of this, so each step "
                    "down the list costs a hundred times as much."
                ),
            )
            # `g` is not offered here, and the equation above this panel shows two
            # terms because of it. It is zero by default, moves no number on this
            # page, and reads as a third free parameter in a problem that has one --
            # h/J. The sidebar's advanced control still reaches it, the corpus still
            # answers questions about it, and the moment it is non-zero the equation
            # grows its third term. See `panels.hamiltonian`.
            longitudinal = base.physics.longitudinal
            if longitudinal:
                st.caption(
                    f"{WARNING} A longitudinal field $g = {longitudinal:g}$ is set from "
                    "the sidebar. It stays set for this chat, and there is no exact "
                    "answer to grade against while it is."
                )

        st.caption(
            f"{WARNING} This chat now ignores the sidebar for everything above. "
            "Nothing else changes, and the sidebar comes back the moment the box at "
            "the top is unticked."
        )
        return (
            base.with_physics(
                n_sites=n_sites,
                boundary="periodic" if boundary == "periodic" else "open",
                coupling=coupling,
                field=field,
                longitudinal=longitudinal,
                precision=precision,
                depth=depth,
                device=device,
                shots=shots,
            )
            .with_model(
                offline=offline,
                audience=audience,
                temperature=temperature,
                # Only the strong tier is replaced. The other two are carried
                # through exactly as the sidebar set them, so this panel cannot
                # flatten the ladder by accident -- which is what it used to do.
                tiers=tuple(
                    (tier, answering_model if tier == "strong" else slug)
                    for tier, slug in base.model.tiers
                ),
                max_calls=max_calls,
                max_output_tokens=max_output_tokens,
                max_retries=max_retries,
                requests_per_second=requests_per_second,
                request_timeout_s=request_timeout,
                parallel_calls=parallel_calls,
            )
            .with_search(
                corpus=corpus,
                external=external,
                shelf=shelf,
                passages=passages,
                vector_share=vector_share,
                rounds=rounds,
                followups=followups,
            )
        )


PAST_CHATS_SHOWN = 10
"""How many earlier conversations the sidebar lists at once.

The panel drew every conversation the file held, and the file grows by one on every
browser session that asks a question. On this machine that reached 148 threads --
296 widgets rebuilt in the sidebar on every rerun of every page, which is what made
the whole interface feel like it was ignoring clicks. Ten is roughly a screen, and
:meth:`~src.agent.memory.Memory.threads` still returns the full list: the cap is on
what is drawn, not on what is kept, so nothing is silently discarded.
"""


def past_chats(store: Memory, on_open: Callable[[str], None]) -> None:
    """List this user's most recent conversations, and offer to return to one.

    Beside the memory panel because the two read the same store at the same scope,
    and because a conversation belongs to the session rather than to any one page.
    Starting a new chat does not delete the old one -- it stops being recalled, which
    is a different thing -- and this is where somebody who has just cleared the
    screen goes looking for what was on it.

    Args:
        store: This user's memory.
        on_open: Called with a conversation's key when its button is pressed.
    """
    threads = store.threads()
    with st.expander(f"🕘 Past chats ({len(threads)})", expanded=False):
        if not store.enabled:
            st.caption(
                f"{ABSENT} Nothing is being kept: no memory file is configured, so a "
                "conversation lasts as long as the tab."
            )
            return
        if not threads:
            st.caption(
                "None yet. Starting a new chat will not delete this one -- it is kept "
                "here and only stops being recalled."
            )
            return
        shown = threads[:PAST_CHATS_SHOWN]
        st.caption(
            "Newest first. Opening one makes it the conversation being recalled; "
            "deleting one removes it from the file for good, and **Delete all "
            "chats** at the foot of the list clears every one of them."
        )
        if len(threads) > len(shown):
            st.caption(
                f"{ABSENT} Showing the {len(shown)} most recent of {len(threads)}. "
                "Nothing has been deleted -- the older ones are still in the file, and "
                "**Delete all chats** reaches them as well. They are not listed because "
                "a sidebar of a hundred rows is not a list anybody reads."
            )
        for thread in shown:
            open_column, delete_column = st.columns([5, 1])
            if open_column.button(
                thread.label(),
                key=f"thread-{thread.name}",
                width="stretch",
                disabled=thread.name == store.thread,
            ):
                on_open(thread.name)
            _delete_chat_button(store, thread.name, delete_column)
        _delete_all_chats_button(store, len(threads))


def _delete_chat_button(store: Memory, thread: str, column: Any) -> None:
    """Offer to delete one past conversation, in two presses.

    Two presses rather than one, and this is the one place in the application where
    a confirmation is worth its friction: every other control here is reversible by
    moving it back, and this one rewrites a file. It is also a control whose reward
    for a misclick is losing the thing the reader came to the panel to find.

    The pending state is keyed by thread name rather than held as a single flag, so
    arming one delete does not arm the one below it -- which is what a shared flag
    does on a rerun, and it would arm the row a reader is most likely to press next.

    Args:
        store: This user's memory.
        thread: The conversation the button belongs to.
        column: Where to draw it, so the caller keeps control of the layout.
    """
    armed: set[str] = st.session_state.setdefault("chat_delete_armed", set())
    if thread in armed:
        if column.button(
            "✓",
            key=f"confirm-delete-{thread}",
            help="Press to delete this conversation permanently.",
            width="stretch",
        ):
            removed = store.forget_thread(thread)
            armed.discard(thread)
            st.toast(f"Deleted {removed} entr{'y' if removed == 1 else 'ies'}.")
            st.rerun()
        return
    if column.button(
        "🗑",
        key=f"delete-{thread}",
        help="Delete this conversation. Asks once more before it does.",
        width="stretch",
    ):
        armed.add(thread)
        st.rerun()


def _delete_all_chats_button(store: Memory, total: int) -> None:
    """Offer to clear the whole listing at once, in two presses.

    Below the rows rather than above them, because a control that empties the list
    should not sit where a reader aims for the newest conversation in it. It counts
    every conversation stored, not the ten drawn, so the count on the button is the
    number that will actually go.

    Ratings are left alone. A reading level the reader taught the agent is a fact
    about the reader rather than about any conversation, so clearing the list does
    not unteach it -- **Forget everything** in the memory panel is the control that
    does.

    Args:
        store: This user's memory.
        total: How many conversations are stored, listed or not.
    """
    if total < 2:
        return
    if st.session_state.get("chat_delete_all_armed"):
        if st.button(
            f"✓ Yes, delete all {total}",
            key="confirm-delete-all-chats",
            type="primary",
            width="stretch",
            help="Deletes every conversation, including the one on screen.",
        ):
            removed = store.forget_all_threads()
            st.session_state.pop("chat_delete_all_armed", None)
            st.session_state.pop("chat_delete_armed", None)
            # The open conversation was one of them, so the transcript on screen
            # now describes records that are gone. Same keys `new_chat_button`
            # drops, for the same reason.
            for key in ("thread", "history"):
                st.session_state.pop(key, None)
            st.toast(f"Deleted {removed} entr{'y' if removed == 1 else 'ies'}.")
            st.rerun()
        if st.button("Cancel", key="cancel-delete-all-chats", width="stretch"):
            st.session_state.pop("chat_delete_all_armed", None)
            st.rerun()
        return
    if st.button(
        "🗑 Delete all chats",
        key="delete-all-chats",
        width="stretch",
        help="Deletes every conversation stored for you. Asks once more before it does.",
    ):
        st.session_state["chat_delete_all_armed"] = True
        st.rerun()


def session_total_line() -> None:
    """State what this session has spent, in one line, in the sidebar.

    Below the memory panel because that is where a reader is already looking at what
    the session is carrying, and a running cost belongs where it is unavoidable
    rather than on a page somebody has to choose to open.

    Deliberately the **only** copy of the total. The breakdown -- per model, per
    node, per question -- is on the Analytics page, and this line and that page read
    the same :func:`src.ui.session_log.totals`. One number written in two places is
    one number that will eventually disagree with itself, and the disagreement is
    invisible until somebody checks, at which point neither figure can be trusted.
    """
    total = session_log.totals(asked_this_session())
    st.caption(f"**This session:** {total.summary()}")
    if total.calls_made:
        st.caption("Full breakdown on the 📊 Analytics page.")


def glossary_expander(title: str = "What the words mean") -> None:
    """Draw the glossary, collapsed.

    Collapsed rather than absent, and on every page rather than on one. A reader
    who does not need it loses one line; a reader who does would otherwise have to
    know that a glossary page exists in order to look for it.

    Args:
        title: The expander's label.
    """
    with st.expander(title):
        st.caption(
            "This project is graded by a reader who works on software rather than on "
            "physics. Every term below is explained without using another one."
        )
        for term in GLOSSARY:
            st.markdown(f"**{term.word}** — {term.gloss}  \n<sub>{term.why}</sub>", True)


def metrics(values: dict[str, Any]) -> None:
    """Draw a row of headline numbers.

    Args:
        values: Label to value, in the order they should appear. An empty mapping
            draws nothing rather than an empty row.
    """
    if not values:
        return
    for column, (label, value) in zip(st.columns(len(values)), values.items(), strict=True):
        column.metric(label, value)


def empty_state(headline: str, body: str) -> None:
    """Say that there is nothing to show, and what would put something here.

    An empty chart and a chart of nothing look the same, and the first is
    indistinguishable from a working system. Every page that can have no data says
    so in a sentence and names the command that would change that.

    Args:
        headline: What is absent.
        body: What to run, in Markdown.
    """
    st.info(headline)
    st.markdown(body)


# --------------------------------------------------------------------------
# Drawing a finished campaign
# --------------------------------------------------------------------------

VERDICT_BADGE: dict[str, str] = {
    "go": "green",
    "conditional": "orange",
    "no": "blue",
}
"""What colour each verdict is drawn in.

**A "no" is not red.** It was, and it made every correct answer look like a crash --
which is worse than cosmetic, because a reader who sees a red alert box stops reading
the reasoning and starts looking for the bug. A "no" is usually the right answer here:
the chain this project studies has a closed-form solution, so a quantum computer has
nothing to offer it, and a campaign that says so has done its job perfectly.

Green means go, amber means it depends on a stated condition, and blue means no -- a
finding, delivered calmly. Red is kept for the one thing that actually is a failure:
a verdict that moved when only the wording did.
"""


def prose(text: str) -> None:
    r"""Write one block of an answer to the page, with its mathematics rendered.

    Every route to the reader goes through here. Streamlit's markdown honours
    exactly two math delimiters -- ``$...$`` and ``$$...$$`` -- and prints everything
    else verbatim, so an answer containing ``\(E_0\)``, which is perfectly ordinary
    LaTeX, puts backslashes and brackets on the page and is judged on that line
    before it is read. :func:`src.agent.mathmarkup.to_dollar_math` rewrites the
    delimiters a renderer ignores into the ones it honours, and drops a delimiter
    that has lost its partner rather than letting it swallow the rest of the
    paragraph.

    That module was written for exactly this and had no caller: the instruction to
    use dollars lived in the prompt and nothing enforced it, which makes it a
    request rather than a guarantee. This is the enforcement, and it is at the last
    possible moment -- the text is repaired on its way to the screen, so no branch
    can bypass it by composing its answer somewhere new.

    It is idempotent, so text that was already correct passes through untouched.

    Headings are demoted here for the same reason and at the same moment. A model
    asked to explain three algorithms writes ``# Teaching VQE, QAOA and VarQITE``,
    which Streamlit renders at page-title size -- larger than the page's own title
    and louder than the question it is answering. An answer is a reply *inside* a
    page, so its shallowest heading belongs at the weight of a lead-in, and
    :func:`~src.agent.mathmarkup.demote_headings` shifts the whole block to keep
    whatever structure the writer meant.

    Args:
        text: The answer, as whatever wrote it left it.
    """
    st.markdown(demote_headings(to_dollar_math(text)))


def _runs_badge(runs: Sequence[RunRecord]) -> str:
    """Say how many circuits were run, in words that agree with the number.

    Args:
        runs: Every configuration the campaign actually executed.

    Returns:
        ``"1 run"`` or ``"5 runs"``. The badge read "5 run" for a year because the
        word was a literal beside a count.
    """
    return f"{len(runs)} run{'' if len(runs) == 1 else 's'}"


def verdict_card(verdict: Verdict | None, state: CampaignState) -> None:
    """Draw the answer as an answer: a badge, then prose, then the numbers behind it.

    Not in a bordered box. A box says "notification" -- something the application is
    telling you about itself -- and this is the reply to a question, which should read
    like one. The badge carries the call so it can be seen at a glance without the
    colour having to shout, and the reasoning is a paragraph because that is what a
    reader is here for.

    Args:
        verdict: The campaign's conclusion, or ``None`` if it reached none.
        state: The finished campaign, for the numbers shown beneath.
    """
    runs = state["runs"]
    if verdict is None:
        # A withheld verdict is not an empty campaign, and this is the mirror of the
        # defect recorded in `answer_card`: there a refusal drawn as a verdict
        # asserted a conclusion nothing had reached; here a full depth ladder drawn
        # as "the campaign did not reach one" threw away every rung of it. *How deep
        # should the circuit be before noise wins?* climbs five depths, spends 774
        # million measurements and is answered by exactly that sentence.
        if not runs:
            st.markdown(":grey-badge[NO VERDICT] &nbsp; :grey-badge[nothing run]")
            prose(
                "The campaign did not reach one, and nothing was run. The "
                "**Pipeline trace** page says how far it got, which is a more "
                "useful thing to show than a guess."
            )
            return
        best = min(runs, key=lambda run: run.energy_per_site)
        model = state["model"]
        st.markdown(
            ":grey-badge[MEASURED, NO VERDICT] &nbsp; "
            f":grey-badge[{_runs_badge(runs)}] &nbsp; "
            f":grey-badge[{len(state['ruled_out'])} refused] &nbsp; "
            f":grey-badge[{state['shots'].spent:,} measurements]"
        )
        prose(
            "**The runs stand; the verdict does not.** A feasibility call is a "
            "statement about one specific chain, and this question named none — so "
            + (
                f"the ladder was climbed on the {model.label()} the assumptions "
                "state, and calling that a yes or a no would answer a question "
                "nobody asked."
                if model is not None
                else "no verdict is offered in place of one."
            )
        )
        prose(
            f"**What the ladder reached.** {_runs_badge(runs)}, deepest "
            f"{max(run.depth for run in runs)} layers, best energy "
            f"**{best.energy_per_site:.6f} per spin** at depth {best.depth}. "
            "The rung that stopped it, and the arithmetic that refused it, are on "
            "the **Pipeline trace** page."
        )
        return
    colour = VERDICT_BADGE.get(verdict.call, "grey")
    st.markdown(
        f":{colour}-badge[{verdict.call.upper()}] &nbsp; "
        f":grey-badge[confidence: {verdict.confidence}] &nbsp; "
        f":grey-badge[{_runs_badge(runs)}] &nbsp; "
        f":grey-badge[{len(state['ruled_out'])} refused] &nbsp; "
        f":grey-badge[{state['shots'].spent:,} measurements]"
        + (f" &nbsp; :grey-badge[{len(state['citations'])} sources]" if state["citations"] else "")
    )
    prose(verdict.summary + _citation_markers(state))
    prose(f"**What would change this answer.** {verdict.crossover_condition}")


def answer_card(state: CampaignState) -> None:
    """Draw the reply in the shape the question asked for.

    Three questions used to get one document. *Write me a variational eigensolver*
    came back as a verdict badge reading **NO** over a table of energies; so did
    *what is a barren plateau*. The graph now takes a different branch for each --
    see :mod:`src.agent.intent` -- and this is the interface half of that: it reads
    the branch off the finished campaign rather than being told, so a reply can never
    be drawn in a shape the agent did not actually produce.

    The badge line differs on purpose. A feasibility reply leads with its call, its
    run count and what it spent, because those are what make it worth anything. A
    prose reply leads with the fact that **nothing was run**, because a reader
    arriving from a feasibility answer will otherwise assume the same machinery
    stands behind this one.

    Args:
        state: The finished campaign.
    """
    request = state["request"]
    # Before the intent branch, because a request turned away at the door never
    # reached one. It used to fall through to `verdict_card`, and the shape that
    # produced was the worst in the application: *detail the mathematics of the
    # quantum-to-classical mapping* came back as a blue **NO** badge reading
    # "confidence: high, 0 run, 0 refused, 0 measurements". Every one of those
    # numbers was accurate and the sentence they formed was not -- "no" here means
    # *a quantum computer is not worth it for this problem*, which is a conclusion
    # about hardware, and no hardware had been considered. A refusal drawn as a
    # verdict asserts the verdict.
    if request.blocked or not request.in_scope:
        reason = (
            "The input screen rejected it."
            if request.blocked
            else (
                "It does not name a lattice of interacting spins, which is the only "
                "thing this assessment covers."
            )
        )
        st.markdown(
            ":grey-badge[NOT ASSESSED] &nbsp; :grey-badge[nothing run] &nbsp; "
            ":grey-badge[no verdict reached]"
        )
        prose(
            f"**This question was not assessed.** {reason} Nothing was run and no "
            "conclusion is offered in place of one."
        )
        return

    intent = state["intent"].intent

    if intent == "explain":
        written = state["answer"]
        # A refusal gets its own badge line and its own colour. Drawn in the shape of
        # an answer it would read as a thin answer, which is the one presentation
        # mistake that undoes the honesty it exists for: the whole value of saying
        # "I could not answer that from what I have" is that a reader can tell it
        # apart from an answer at a glance.
        if written is not None and written.refused:
            st.markdown(
                ":orange-badge[NOT ANSWERED] &nbsp; :grey-badge[nothing retrieved] "
                f"&nbsp; :grey-badge[nothing computed] &nbsp; "
                f":grey-badge[{state['intent'].decided_by}-routed]"
            )
            prose(written.text)
            return
        st.markdown(
            ":blue-badge[EXPLANATION] &nbsp; :grey-badge[no circuits run] &nbsp; "
            f":grey-badge[{state['intent'].decided_by}-routed]"
            + (f" &nbsp; :grey-badge[{written.cited} sources]" if written else "")
        )
        prose(written.text if written and written.written else _nothing_written())
        if written and written.rests_on:
            st.caption(f"How to check this: {written.rests_on}")
        return

    if intent == "implement":
        drafted = state["draft"]
        st.markdown(
            ":violet-badge[CODE] &nbsp; :orange-badge[not run by this app] &nbsp; "
            f":grey-badge[{state['intent'].decided_by}-routed]"
        )
        if drafted is None or not drafted.written:
            reason = drafted.note if drafted is not None else "the drafting step did not run"
            st.warning(
                f"{WARNING} **No code was written.** {reason.capitalize()}.\n\n"
                "Nothing is offered in its place: a feasibility report is not an "
                "answer to a request for a file."
            )
            return
        if drafted.preamble:
            prose(drafted.preamble)
        # Above the code, and composed rather than generated. The program below it is
        # a language model's text that nothing here ran; these equations came out of
        # the same layer arithmetic that prices every circuit in the project, so a
        # reader who checks the file against them is checking it against something.
        if drafted.mathematics:
            prose(drafted.mathematics)
        st.code(drafted.code, language=drafted.language)
        st.caption(
            "Written by a language model and not executed here. Every other number "
            "in this application is one two methods sharing no algebra agreed on; "
            "this has none of that backing, so it ends by checking itself against a "
            "known result -- which is the check for you to run."
        )
        return

    verdict_card(state["verdict"], state)


def circuit_asked_for(state: CampaignState, fallback_depth: int, position: int = 0) -> None:
    """Draw the circuit the question described, whichever branch answered it.

    Drawn here rather than on the code branch alone, because the request that
    prompted it -- *can you write the QAOA circuit for 8 spins at depth 3?* -- is one
    shape of a family. *Show me a three-layer ansatz for a ring of 10 qubits* is an
    explanation, *is a 14-spin chain at depth 4 worth running* is a feasibility
    study, and all three described a circuit precisely enough to draw. So the
    condition is what the question said, not which node answered it.

    **The picture is computed, and the code beside it is not.** A drafted program is
    a language model's text that this application does not execute -- see
    :mod:`src.agent.drafting` -- so on that branch the reader is looking at one
    unverified thing. The diagram is built by
    :class:`~src.physics.quantum.ansatz.AnsatzSpec` from the length and depth read out
    of the question, by the same code that prices circuits everywhere else in the
    project. The caption says which is which, because the diagram is only worth
    drawing as a *check* on the code, and a reader who thinks it was parsed out of the
    code has lost exactly that.

    Args:
        state: The finished campaign.
        fallback_depth: How many layers to draw when the question named none. The
            page passes the setting it asked with, rather than this reading the knob
            back: reading it back draws a second copy of the sidebar on any page
            opened on its own, which is how a test reaches one.
        position: Which turn of the conversation this is, used to key the download
            button. Every panel a chat thread can draw more than once needs one --
            see :func:`figure`.
    """
    spec = circuit_the_question_named(state, fallback_depth)
    if spec is None:
        return

    st.markdown("##### The circuit your question describes")
    caption = [
        f"**{spec.n_qubits} magnets, {spec.depth} "
        f"{'layer' if spec.depth == 1 else 'layers'}.** "
        + (
            f"The picture shows the first {min(spec.n_qubits, figures.MAX_DRAWN_QUBITS)} "
            f"and the first {min(spec.depth, figures.MAX_DRAWN_LAYERS)}, because the "
            "pattern repeats and it stops being legible past that. "
            if spec.n_qubits > figures.MAX_DRAWN_QUBITS or spec.depth > figures.MAX_DRAWN_LAYERS
            else ""
        )
        + "Each blue connector ties two "
        "neighbouring magnets together; each orange box nudges one magnet on its own. "
        "The connectors in a column share no wire, so they all happen at the same "
        "moment -- which is why a chain of any length needs only "
        f"**{spec.two_qubit_rounds_per_layer} such columns per layer**."
    ]
    if reading.read(state["request"].text).depth is None:
        caption.append(
            f"The question named no depth, so this is the **{spec.depth}** currently "
            "set on the settings knob in the sidebar. Move it and the picture moves."
        )
    if state["intent"].intent == "implement":
        caption.append(
            "Drawn by this application from the chain and depth in your question, "
            "**not** read out of the program above -- which nothing here ran. If the "
            "two disagree, that disagreement is the thing to check first."
        )
    figure(
        figures.circuit_figure(spec.n_qubits, spec.depth, spec.bonds, spec.rounds),
        " ".join(caption),
        download="circuit-asked-for",
        # Keyed by turn. The same latent crash `convergence_asked_for` hit: the chat
        # thread redraws every past answer, so two questions that both drew a circuit
        # put two download buttons under one key, which Streamlit raises on. Found by
        # auditing every `download=` in this module after the first one fell over,
        # rather than by waiting for a second person to meet it.
        key=f"circuit-asked-for-{position}",
    )

    with st.expander("The algebra this picture is a drawing of"):
        st.markdown(
            "**Every gate in the diagram comes from one of two halves of the "
            "Hamiltonian, and the split is the whole design.** Write it as"
        )
        st.markdown(
            r"$$\hat H \;=\; \underbrace{-J\sum_i \hat\sigma^z_i\hat\sigma^z_{i+1}"
            r"}_{\hat H_\text{diag}} \;+\; "
            r"\underbrace{-\,h\sum_i \hat\sigma^x_i}_{\hat H_\text{field}}$$"
        )
        st.markdown(
            "The first half is diagonal -- it only ever multiplies an arrangement of "
            "magnets by a number -- and the second is what makes the problem quantum. "
            f"One layer of the circuit is one **Trotter step**: the two halves applied "
            f"one after the other, each for its own learned angle. {spec.depth} layers is"
        )
        # The same string the code branch prints above its program, from
        # `circuit_algebra`. Written out layer by layer rather than as a product sign,
        # because the product notation hides the one thing a reader has to take from
        # it: the factor written last is the one that acts first.
        st.markdown(f"$$\n{state_preparation_latex(spec)}\n$$")
        st.caption(
            "Read right to left: the rightmost factor acts first, so layer 1 is what "
            "happens to the register first."
        )
        st.markdown(
            "**Why it is an approximation, and why that is fine here.** The two halves "
            "do not commute, so applying them in turn is not the same as applying "
            r"their sum: $e^{A}e^{B} \neq e^{A+B}$ unless $AB = BA$, and the error per "
            r"layer is of order $\gamma_k\beta_k$. A *Trotterised* circuit fixes the "
            "angles from a time step and wears that error. This one does not -- the "
            f"**{spec.n_parameters} angles are free and are optimised**, so the "
            "circuit is only ever asked to be the best state of this shape, and the "
            "Trotter error stops being an error and becomes the shape itself. That is "
            "the difference between simulating time evolution and running a "
            "variational method, and it is the same circuit either way.\n\n"
            r"Each $e^{-i\gamma_k \hat H_\text{diag}}$ is the blue column: "
            r"$\hat H_\text{diag}$ is a sum of commuting two-magnet terms, so its "
            "exponential factors into one gate per bond and the bonds can be run in "
            f"**{spec.two_qubit_rounds_per_layer} rounds** of disjoint pairs. Each "
            r"$e^{-i\beta_k \hat H_\text{field}}$ is the orange row: "
            r"$\hat H_\text{field}$ is a sum of single-magnet terms, all commuting, "
            "so it is one rotation per wire and costs no two-qubit depth at all."
        )

    described = spec.describe()
    with st.expander("What this circuit costs, counted rather than guessed"):
        metrics(
            {
                "Dials the optimiser turns": described["n_parameters"],
                "Two-qubit gates": described["two_qubit_gates"],
                "Two-qubit depth": described["two_qubit_depth"],
                "One after another instead": described["naive_two_qubit_depth"],
            }
        )
        st.caption(
            f"Counted from the {described['n_bonds']} bonds of this "
            f"{'ring' if spec.boundary == 'periodic' else 'chain'} by "
            "`src.physics.quantum.ansatz` -- the same code the Quay Lab draws with and "
            "the agent prices with. The last two numbers are one circuit scheduled two "
            f"ways, and the ratio between them, **{described['depth_saving_factor']}x** "
            "here, is bought by noticing that the bonds commute rather than by better "
            "hardware. QAOA and the Hamiltonian variational ansatz are two names for "
            "this identical layer pattern; what differs is the objective they are "
            "usually pointed at, which changes no gate in the picture."
        )


@st.cache_data(show_spinner=False)
def true_energy(
    n_sites: int,
    coupling: float,
    field: float,
    boundary: str,
    longitudinal_field: float = 0.0,
    geometry: str = "chain",
    rows: int = 1,
) -> tuple[float | None, str | None]:
    r"""Get the exact ground-state energy, or the reason there is not one.

    Every panel that grades a variational answer needs this and every panel needs to
    survive not having it, so the refusal comes back beside the value and no caller has
    to know which cap it fell foul of.

    This function is on the grader's side of the import wall and could not live anywhere
    under :mod:`src.agent` or :mod:`src.physics.quantum` -- see the project's working
    agreement. It is what makes the interface able to draw the truth as a flat line
    beneath curves that were produced with no knowledge of it.

    Args:
        n_sites: Chain length.
        coupling: The Ising coupling :math:`J`.
        field: The transverse field :math:`h`.
        boundary: Ring or segment.
        longitudinal_field: The field :math:`g` along the coupling axis. Anything
            non-zero is refused rather than silently ignored: the closed-form route maps
            the chain to free fermions and that mapping is exactly what a non-zero
            :math:`g` breaks, so an answer computed without it would be the right number
            for a different problem.
        geometry: The shape the spins sit on. **Not cosmetic, and defaulting it was a
            defect:** the same site count on a different shape is a different problem
            with a different answer, because a square lattice has more bonds per spin
            than a line does. Sixteen spins on a ring reach -20.40 and on a 4x4
            square -34.01. Grading a lattice's variational curves against the line's
            number drew the exact answer *above* honest curves, so they appeared to
            fall through it -- which reads as a broken variational bound rather than
            as the mismatched comparison it was.
        rows: Rows of the lattice, ``1`` for a line.

    Returns:
        The energy and ``None``, or ``None`` and the reason there is none.
    """
    from src.physics.model import TFIMSpec
    from src.physics.reference import exact_diagonalisation

    if longitudinal_field != 0.0:
        return None, (
            f"this chain has a longitudinal field g = {longitudinal_field:g}, which breaks "
            "the mapping every exact route here relies on -- and that is the regime the "
            "whole project is about, not a gap in the page"
        )
    spec = TFIMSpec(
        n_sites,
        coupling,
        field,
        cast("Any", boundary),
        geometry=cast("Any", geometry),
        rows=rows,
    )
    reason = exact_diagonalisation.unsupported_reason(spec)
    if reason is not None:
        return None, reason
    return exact_diagonalisation.ground_state_energy(spec), None


def convergence_asked_for(state: CampaignState, position: int = 0) -> None:
    r"""Draw the three methods descending, when the campaign raced them.

    The picture for *which of these converges fastest, and how close does it get* --
    a question a paragraph answers badly and one axis answers completely.

    **Nothing here is decided by this function.** Whether the race happened at all is
    the graph's decision, taken in :func:`src.agent.graph.converge_methods` on the
    chain the campaign formalised, and this draws whatever it finds in
    :attr:`~src.agent.state.CampaignState.race` or nothing. That is deliberate: a panel
    that ran its own comparison would be showing a reader a second, unrecorded
    experiment that the report, the trace and the follow-ups all know nothing about.

    **The one thing the picture cannot show is stated under it.** Every energy in it is
    exact -- a simulated state vector with no shot noise and no gate error -- so it
    compares the methods on a perfect device. A reader who takes it as a statement
    about robustness to noise has taken the opposite of what it says, which is why that
    sentence is in the caption and not in an expander.

    Args:
        state: The finished campaign.
        position: Which turn of the conversation this is, used to key the download
            button. **Not optional in practice**, and its absence took the page down:
            the chat thread redraws every past answer's panels, so a second question
            that also raced the methods drew a second figure under the same download
            slug, and two Streamlit buttons sharing one key raise rather than
            degrade -- the whole page, mid-conversation, with the answer already on
            screen. :func:`field_sweep_asked_for` had taken a position for this
            reason since it was written; this function was the one that did not, and
            the ``key`` argument on :func:`figure` documents the exact failure it
            then hit.
    """
    race = state["race"]
    if race is None:
        return

    exact, refusal = true_energy(
        race.n_sites,
        race.coupling,
        race.transverse_field,
        race.boundary,
        race.longitudinal_field,
        race.geometry,
        race.rows,
    )
    st.markdown("##### How each method got there")

    where = (
        "right at the critical point, where the methods have their hardest time"
        if race.at_criticality
        else f"at a field-to-coupling ratio of {race.transverse_field / race.coupling:.3g}"
    )
    # A range asked for in *this* question, not inherited: "can we plot it between 0 and
    # 20?" is a request about the picture on screen, and it is read as epochs rather
    # than energies for the reason given on `reading.read_epoch_range`. Which axis was
    # cropped is said out loud, because a reader who meant the other one has to be able
    # to tell that from the caption rather than from the numbers.
    window = reading.read_epoch_range(state["request"].text)
    longest = max((len(history) for _, history in race.curves()), default=1) - 1

    caption = [
        f"**{race.n_sites} magnets, {race.depth} "
        f"{'layer' if race.depth == 1 else 'layers'}, {where}.** Lower is better.",
    ]
    if exact is not None:
        # The endpoint annotations sit at each curve's *own* last step, so a narrowed
        # view usually leaves them off the right-hand edge. Promising a number that is
        # not on the picture is worse than not mentioning it.
        caption.append(
            "The dashed line is the true answer, and the number at the end of "
            "each curve is how far short that method stopped."
            if window is None
            else "The dashed line is the true answer; how far short each method "
            "stopped is in the table below, since the curves run past this view."
        )
    caption.append(
        "**No noise anywhere in this figure.** Every energy is computed exactly, so this "
        "compares the methods on a *perfect* device and says nothing about how any of "
        "them stands up to measurement noise or gate error."
    )

    if window is not None:
        caption.append(
            f"**Showing epochs {window[0]} to {window[1]}, as you asked.** The curves "
            f"themselves are unchanged and the longest still runs to epoch {longest}; "
            "only the view is narrowed, so nothing here ends where it was cut off."
        )
    figure(
        figures.method_race_figure(
            race.curves(),
            exact,
            xlabel="optimisation epoch (one accepted step)",
            xlimit=window,
        ),
        " ".join(caption),
        download="methods-raced",
        key=f"methods-raced-{position}",
    )
    if exact is None and refusal is not None:
        st.caption(f"No true answer is drawn: {refusal}")

    # Folded away rather than printed under the figure. The picture is the answer to
    # the question; the table is for a reader who has looked at the picture and wants
    # the numbers behind it, and printing both at once pushed the reply out of sight.
    with st.expander("What each method cost, counted rather than guessed"):
        graded = "" if exact is None else " short by |"
        st.markdown(
            "\n".join(
                [
                    "| method | what moves the dials | energy reached |"
                    + graded
                    + " epochs | energy evaluations | stopped because |",
                    "|---|---|---|---|---|---|" + ("---|" if exact is not None else ""),
                    *(
                        f"| **{run.method}** | {METHOD_BLURBS[run.method]} | "
                        f"{run.energy:.6f} |"
                        + ("" if exact is None else f" {run.energy - exact:.2e} |")
                        + f" {run.n_steps} | {run.n_energy_evaluations} | "
                        f"{run.stop_reason.replace('_', ' ')} |"
                        for run in race.runs
                    ),
                ]
            )
        )
        if exact is not None:
            st.caption(
                f"**Graded against exact diagonalisation, {exact:.6f}.** That column is "
                "the one thing on this page the agent could not have written: the exact "
                "solvers are sealed off from everything under `src/agent/`, so nothing "
                "that produced these curves can evaluate the true answer, and this "
                "interface -- which sits on the grader's side of that wall -- supplies "
                "it. An agent that could read the answer it is being marked against "
                "would make the whole comparison worthless."
            )
        st.caption(
            "**Read the last two columns together.** An epoch is one accepted step and "
            "the curves are drawn against it, but a step does not cost the same in all "
            "three: the optimiser's includes a line search, and the imaginary-time "
            "method's includes a whole matrix of overlaps that a real device would pay "
            "for in extra circuits. Steps are the *shape* of the descent; evaluations "
            "are the bill."
        )


def _nothing_written() -> str:
    """The line shown when a branch produced nothing at all.

    Returns:
        A sentence that names it as a fault rather than a finding. An empty reply
        that reads like a considered answer is the worst of both.
    """
    return (
        "**Nothing was written.** The question was read as asking for an "
        "explanation and none was composed. That is a fault, not a finding -- the "
        "**Pipeline trace** page says where it stopped."
    )


def _citation_markers(state: CampaignState) -> str:
    """Render the numbered markers that tie an answer to its sources.

    Placed at the end of the paragraph rather than inside it, and that is a decision
    rather than a limitation. Dropping a marker mid-sentence would claim that *this
    clause* came from *that source*, and on the deterministic path nothing knows
    which clause came from where -- the verdict is computed from measurements and the
    sources are background that was read alongside. A marker placed where the claim
    cannot be checked is worse than one placed honestly at the end.

    Args:
        state: The finished campaign.

    Returns:
        The markers, or the empty string when nothing was cited.
    """
    citations = state["citations"]
    if not citations:
        return ""
    return " " + " ".join(f"[{index}]" for index in range(1, len(citations) + 1))


def what_it_assumed(state: CampaignState) -> None:
    """Show the modelling choices the verdict rests on, above everything else.

    Turning a sentence into a Hamiltonian is where a feasibility study most often
    goes quietly wrong, because whatever comes out of that step is treated as the
    problem for the rest of the study. So the choices that were *made* rather than
    *given* are shown where the reader will meet them before the conclusion, not
    three sections into a report and out of sight -- a reader who disagrees with an assumption
    can stop reading, and one who never sees it cannot.

    Args:
        state: The finished campaign.
    """
    model = state["model"]
    if model is None or not model.assumptions:
        return
    with st.expander(f"What it assumed ({len(model.assumptions)})", expanded=False):
        st.caption(
            "Chosen rather than given. The verdict depends on every one of them, and "
            "any of them is a fair thing to reject."
        )
        for assumption in model.assumptions:
            st.markdown(f"- {assumption}")


_ARXIV_NEW = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
"""An arXiv identifier in the post-2007 form, ``2111.05176`` or ``2111.05176v2``."""

_ARXIV_OLD = re.compile(r"^[a-z-]+(\.[A-Z]{2})?/\d{7}(v\d+)?$")
"""An arXiv identifier in the pre-2007 form, ``cond-mat/9804280``."""


def source_link(identifier: str) -> str:
    """Turn a citation's identifier into something a reader can open, when it is one.

    A citation nobody can follow is a citation a reader has to take on trust, which
    is the opposite of the reason it was retrieved. Most of this corpus is arXiv, and
    an arXiv identifier is a URL with three characters missing -- so the interface
    supplies them rather than printing a bare number and leaving the reader to paste
    it somewhere.

    Both identifier forms are matched, and anything else is returned as plain text
    rather than guessed at: a link built from an identifier that is not one goes to a
    404, and a broken link is worse than no link because it looks checkable.

    Args:
        identifier: The citation's handle, as retrieval recorded it.

    Returns:
        Markdown -- a link for an arXiv identifier, backticked plain text otherwise.
    """
    if _ARXIV_NEW.match(identifier) or _ARXIV_OLD.match(identifier):
        return f"[arXiv:{identifier}](https://arxiv.org/abs/{identifier})"
    return f"`{identifier}`"


SWEEP_TOOL = "exact_field_sweep"
"""The tool whose arguments this page redraws as figures.

Named here rather than imported from :mod:`src.agent.tools`, where the tool is
built rather than declared -- it exists only when a host lends the solver behind it,
so there is no module-level object to take the name off.
"""

SWEEP_ARGUMENTS: tuple[str, ...] = (
    "n_sites",
    "coupling",
    "boundary",
    "curves",
    "points",
    "ratio_max",
    "levels",
    "geometry",
    "rows",
)
"""Which recorded arguments are passed back to the solver when redrawing.

A filter rather than a splat, because the arguments were written by a *model*: an
extra key it invented would be a ``TypeError`` in the middle of a page that was
otherwise about to render correctly.

**A filter is also how the guarantee above can be lost, and it was.** The whole
point of re-running the solver from the recorded call is that the figure and the
prose cannot disagree, because both came from one set of arguments. An argument
dropped here breaks exactly that: ``geometry`` and ``rows`` were missing, so a
question about a ``3 x 3`` square lattice was answered in prose about the lattice
and *drawn* as a line of nine -- the two halves of one answer describing different
problems, with nothing on the page to say so. Anything the tool accepts belongs in
this tuple, and a test holds the two together.
"""

SWEEP_CAPTIONS: dict[str, str] = {
    "spectrum": (
        "**The lowest few energies of the whole chain, measured from the ground "
        "state.** Below $h/J = 1$ the first excited state sits on the axis: it is the "
        "ground state's mirror image — all spins flipped — and on a finite ring the "
        "two are split by an amount that shrinks exponentially as the chain grows. "
        "Above $h/J = 1$ that pair separates and the cheapest excitation becomes a "
        "genuine disturbance travelling along the chain. The dashed line is the "
        "infinite chain's gap, $2|J-h|$, which reaches **zero** at $h/J = 1$; this "
        "chain's does not, and cannot — that difference is the whole of finite-size "
        "physics in one picture."
    ),
    "energy": (
        "**The ground-state energy per magnet as the sideways push is turned up.** "
        "Note what it does *not* do: there is no kink, no jump and no visible feature "
        "at $h/J = 1$, even though a phase transition happens there. The energy is "
        "smooth through it, which is exactly why the derivatives are the panel the "
        "transition shows up in. The dashed line is the same quantity for an "
        "*infinitely* long chain — how far this one sits from it is the whole of what "
        '"finite-size effect" means, and it is smaller than most readers expect.'
    ),
    "magnetisation": (
        "**How far the magnets have been knocked over, from 0 (none) to 1 (fully "
        "over).** This is $\\langle \\sigma^x \\rangle$, the average alignment with "
        "the push. It rises fastest near $h/J = 1$ and the rise sharpens as the chain "
        "gets longer; in an infinite chain it becomes a singular point."
    ),
    "energy_derivatives": (
        "**The energy per magnet, then its first and second derivative with respect "
        "to the field.** Read them top to bottom: featureless, bending, and then a "
        "sharp dip. That progression is the signature of this transition — the "
        "quantity itself is smooth, and the sharpness lives two derivatives down. "
        "The middle panel is minus the magnetisation exactly, by the "
        "Hellmann\u2013Feynman theorem, so it is the same curve as the magnetisation "
        "panel with a sign on it rather than a second measurement."
    ),
    "magnetisation_derivatives": (
        "**The alignment with the field, then its first and second derivative.** The "
        "middle panel is the *susceptibility*: how much more aligned the chain gets "
        "for a little more push. Its peak is this finite chain's own estimate of "
        "where the transition is. The bottom panel crosses zero at that peak, which "
        "is what makes the location something to solve for rather than to eyeball."
    ),
}
"""What a reader should take from each figure, in words that assume no physics.

One caption per curve, held here rather than beside each ``st.pyplot`` call so that
a reader comparing two of them is reading prose written to the same standard. Every
one of them states the *claim* -- what the picture shows and why it is the picture
worth showing -- because a curve a non-specialist cannot read a conclusion off is
decoration.
"""


def _sweep_call(state: CampaignState) -> ToolCall | None:
    """Find the exact field sweep this campaign fetched, if it fetched one.

    Args:
        state: The finished campaign.

    Returns:
        The last successful sweep call, or ``None``. The last rather than the first,
        because a model that swept, read the result and swept again with a longer
        chain meant the second one -- and a page showing the first would be showing
        a reader the argument the agent rejected.
    """
    found = [call for call in state["tool_calls"] if call.name == SWEEP_TOOL and not call.failed]
    return found[-1] if found else None


def field_sweep_asked_for(state: CampaignState, position: int = 0) -> None:
    r"""Draw the exact curves the agent asked the sealed solver for.

    The picture for *plot the low-lying spectrum against the field*, *plot the
    ground-state energy and its first and second derivatives against* $h/J$, *how
    does the magnetisation switch on* -- questions with right answers that used to be
    answered from a shelf of abstracts, because the only curve this application could
    draw was a race between three variational methods and the word "plot" was enough
    to start one.

    **Nothing here is decided by this function.** Which curves are drawn, for which
    chain, over which range and with how many levels are all read off the arguments
    the *agent* passed to ``exact_field_sweep`` -- see
    :func:`src.agent.tools.field_sweep_tool`. The solver is re-run with those same
    arguments rather than the numbers being carried through the model's context: the
    curve is several hundred floats, a context window is the wrong place to keep
    them, and re-running a deterministic closed form is cheaper than serialising it.
    What that buys is the guarantee that matters -- the figure and the prose above it
    cannot disagree, because they came from one call with one set of arguments.

    **Where the numbers come from, and why the agent cannot read them.** The exact
    solvers are sealed from everything under ``src/agent/``; this page sits on the
    grader's side of that wall. The agent reaches the sweep through a callable the
    host lends to one tool, and only on the branch that writes prose -- the branch
    that runs circuits and is graded against an exact answer cannot call it at all.
    See :func:`src.physics.registry.field_sweep_bench`.

    Args:
        state: The finished campaign.
        position: Where this answer sits in the thread. Only used to key the
            download buttons, which a chat page draws once per past answer as well
            as for the newest one.
    """
    call = _sweep_call(state)
    if call is None:
        _say_why_there_is_no_curve(state)
        return
    from src.physics.registry import field_sweep_bench

    arguments = {key: value for key, value in call.arguments.items() if key in SWEEP_ARGUMENTS}
    try:
        payload = field_sweep_bench()(**arguments)
    except Exception as error:  # a figure must never take the answer down with it
        st.caption(
            f"The curves could not be redrawn here ({type(error).__name__}); the "
            f"numbers the answer quotes are in the 🛠 Tools it called panel."
        )
        return
    if "error" in payload:
        st.caption(f"No curves are drawn: {payload['error']}")
        return

    ratios = np.array(payload["ratio"], dtype=float)
    asked = tuple(payload.get("curves_requested") or ())
    st.markdown("##### What the exact solution does as the field is turned up")
    st.caption(_sweep_provenance(payload, call, state["request"].text))
    for curve in asked:
        drawn = _sweep_figure(curve, ratios, payload)
        if drawn is None:
            continue
        figure(
            drawn,
            SWEEP_CAPTIONS[curve],
            download=f"field-sweep-{curve.replace('_', '-')}",
            key=f"field-sweep-{curve}-{position}",
        )
    with st.expander("The numbers behind the curves, and the check on them"):
        st.code(payload["table"], language="text")


def _say_why_there_is_no_curve(state: CampaignState) -> None:
    """Explain a refused sweep, when the question asked for one and got none.

    Silence is the wrong answer here. A reader who asked to see the magnetisation of
    an *open* eight-spin chain against the field asked for something the closed form
    cannot give -- the free-fermion solution needs a ring -- and the sweep says so in
    one sentence. Printing nothing would leave them to conclude the application
    ignored them, which is the same failure as answering the wrong question with
    confidence, one step quieter.

    Nothing is drawn when no sweep was attempted at all, which is the ordinary case
    for every question that did not ask for a curve.

    Args:
        state: The finished campaign.
    """
    refused = [call for call in state["tool_calls"] if call.name == SWEEP_TOOL and call.failed]
    if not refused:
        return
    st.caption(
        "**No curve is drawn, and this is why.** The exact solution was asked for and "
        f"declined the chain: {refused[-1].result[:400]}"
    )


def _sweep_provenance(payload: dict[str, Any], call: ToolCall, question: str) -> str:
    """Say where the curves came from, in one caption a sceptic can act on.

    Deliberately does not claim *who* chose the arguments. Online a model chose them
    and offline they were read out of the sentence by rule, and this function cannot
    tell the two apart from what it is handed -- so it points at the record that can
    rather than asserting the more flattering of the two. The campaign trail says
    which it was.

    What it does say is where the numbers a reader might otherwise take for their own
    came from. A length nobody typed must never look like one somebody did, so a
    chain length the question did not name is called out as this project's default.

    Args:
        payload: What the solver returned.
        call: The recorded tool call, kept for the arguments it was given.
        question: The question as asked, to tell a named length from a supplied one.

    Returns:
        The caption. It names the chain, says which numbers came out of the question
        and which did not, and reports the disagreement between the two independent
        solvers -- the number that decides whether the picture is worth believing.
    """
    worst = payload.get("worst_disagreement")
    agreement = (
        f"Every point was computed **twice**, by the closed-form free-fermion "
        f"solution and by exact diagonalisation, which share no algebra; the worst "
        f"disagreement anywhere on the curve is {worst:.1e}."
        if isinstance(worst, float)
        else (
            "Computed by the closed-form free-fermion solution alone: solving every "
            "point a second time by diagonalisation is limited to short chains, and "
            "this one is past that limit."
        )
    )
    ring = (
        " A **ring** — the closed form exists for a chain joined end to end, so that "
        "is what is swept unless the question asks for a line with two ends."
        if payload.get("boundary") == "periodic"
        else ""
    )
    supplied = (
        ""
        if reading.read_count(question) is not None
        else (
            f" The question named no length, so this is the project's default of "
            f"{payload['n_sites']} magnets — ask for another and the curves are "
            f"redrawn for it."
        )
    )
    # `.get`, with the old key as the fallback and the label as the last resort:
    # a provenance line is the one caption that must not be the thing that takes
    # the page down, and this reads a payload the solver builds.
    named = payload.get("problem") or payload.get("chain") or "the problem"
    # Analytic derivatives are a property of the closed form, which only a line
    # has. On a lattice the sweep supplies the energy's first derivative exactly
    # and declines the rest, so claiming "every derivative is analytic" there
    # would be claiming something about curves that were not drawn.
    analytic = (
        " Nothing here is fitted, sampled or approximated, and every derivative is "
        "analytic rather than a finite difference."
        if payload.get("geometry", "chain") == "chain"
        else " Nothing here is fitted, sampled or approximated. On a lattice there "
        "is no closed form, so every point is diagonalised and the higher "
        "derivatives are declined rather than finite-differenced."
    )
    return (
        f"**{named}.** Drawn from the arguments passed to `{SWEEP_TOOL}`, "
        f"which are recorded under 🛠 **Tools it called** below.{ring}{supplied} "
        f"{agreement}{analytic}"
    )


def _sweep_figure(curve: str, ratios: Any, payload: dict[str, Any]) -> Any:
    """Build the one figure a named curve calls for.

    Args:
        curve: Which curve to draw.
        ratios: The swept :math:`h/J` values.
        payload: What the solver returned.

    Returns:
        A matplotlib figure, or ``None`` when the payload does not carry what that
        curve needs -- which is how a spectrum request that produced no levels
        leaves the panel out instead of drawing an empty pair of axes.
    """
    if curve == "spectrum":
        excitations = tuple(tuple(row) for row in payload.get("excitations") or ())
        if min((len(row) for row in excitations), default=0) < 2:
            return None
        # The levels are in energy units, so the infinite chain's gap 2|J - h| has to
        # carry the coupling: at J = 2 a reference drawn as 2|1 - h/J| would be half
        # the curve it claims to be, and it would look plausible.
        coupling = float(payload.get("coupling") or 1.0)
        return figures.spectrum_sweep_figure(
            ratios,
            excitations,
            np.array(
                payload.get("thermodynamic_gap")
                or [2.0 * coupling * abs(1.0 - ratio) for ratio in ratios],
                dtype=float,
            ),
        )
    if curve == "energy":
        reference = payload.get("thermodynamic_energy_density")
        return figures.observable_sweep_figure(
            ratios,
            np.array(payload["energy_density"], dtype=float),
            r"$E_0/L$",
            None if reference is None else np.array(reference, dtype=float),
        )
    if curve == "magnetisation":
        return figures.observable_sweep_figure(
            ratios,
            np.array(payload["magnetisation"], dtype=float),
            r"$\langle \sigma^x \rangle$",
        )
    if curve == "energy_derivatives":
        return figures.derivative_sweep_figure(
            ratios,
            np.array(payload["energy_density"], dtype=float),
            np.array(payload["energy_slope"], dtype=float),
            np.array(payload["energy_curvature"], dtype=float),
            (
                r"$E_0/L$",
                r"$\partial (E_0/L)/\partial h$",
                r"$\partial^2 (E_0/L)/\partial h^2$",
            ),
        )
    if curve == "magnetisation_derivatives":
        return figures.derivative_sweep_figure(
            ratios,
            np.array(payload["magnetisation"], dtype=float),
            np.array(payload["magnetisation_slope"], dtype=float),
            np.array(payload["magnetisation_curvature"], dtype=float),
            (
                r"$\langle \sigma^x \rangle$",
                r"$\partial \langle \sigma^x \rangle/\partial h$",
                r"$\partial^2 \langle \sigma^x \rangle/\partial h^2$",
            ),
        )
    return None


def tools_called(state: CampaignState) -> None:
    """Show which tools the model asked for, with the arguments and what came back.

    The observability half of letting a model choose its own actions, and the reason
    the consultation is a node rather than a few lines inside the answering branch. A
    reader looking at an answer that cites an arXiv id needs to be able to see that a
    search was made, what was searched for, and what the index returned -- otherwise
    the citation is indistinguishable from one the model remembered, which is the
    failure this whole route exists to close.

    Failed calls are shown rather than hidden. A model that asked for a tool with a
    bad argument and recovered is a model working correctly, and a panel that showed
    only successes would make a four-call consultation look like a one-call one.

    Nothing is drawn when no tool was called, which is the ordinary case for a
    feasibility question -- that branch computes everything it claims without a model
    choosing anything -- and a permanently empty expander is a control that teaches a
    reader to ignore the panel.

    Args:
        state: The finished campaign.
    """
    calls = state["tool_calls"]
    if not calls:
        return
    worked = sum(1 for call in calls if not call.failed)
    with st.expander(f"🛠 Tools it called ({worked}/{len(calls)} returned something)"):
        st.caption(
            "The model chose these itself, from the same list the MCP server "
            "publishes. It may make at most four rounds of calls, and each round is "
            "logged and billed separately — the count is on the 📊 Analytics page."
        )
        for index, call in enumerate(calls, start=1):
            given = ", ".join(
                f"`{key}` = {value!r}" for key, value in sorted(call.arguments.items())
            )
            marker = WARNING if call.failed else PRESENT
            arguments = given or "no arguments"
            st.markdown(f"{marker} **{index}. `{call.name}`** — {arguments}")
            st.code(call.result, language="text")
        if worked < len(calls):
            st.caption(
                "A call that returned an error is handed back to the model rather "
                "than raised, so it can correct its own argument and try again. That "
                "is why a failure here is not a failed answer."
            )


def sources_used(state: CampaignState) -> None:
    """List what the answer stands on, numbered to match the markers inside it.

    A reference list rather than a stack of collapsed panels. The numbers here are
    the numbers in the prose -- a reader who meets ``[2]`` mid-sentence should be
    able to find out what ``[2]`` is by glancing down, and four grey expander bars
    make them click four times to learn four titles. So the titles, the shelf and
    the link are all on the page, and the passages themselves -- which are long, and
    wanted only by a reader actually checking a claim -- are behind one expander at
    the end rather than one each.

    Reviewed and unreviewed sources are marked differently and never mixed silently.
    A note somebody wrote and checked, and a passage pulled from an outside index
    mid-run, are different kinds of evidence; a list that flattens them invites a
    reader to weigh them equally.

    A repeated identifier is labelled as a repeat. Retrieval works in chunks, so a
    note that answers the question well comes back several times over, and a list
    printing the same title three times reads as three sources agreeing -- which is
    the single most misleading thing a reference list can do.

    Args:
        state: The finished campaign.
    """
    citations = state["citations"]
    if not citations:
        st.caption(
            f"{ABSENT} No sources were found, so this answer rests entirely on the "
            "measurements above. That is honest, and weaker than it needed to be."
        )
        return

    st.markdown("**Sources** &nbsp; :grey-badge[numbered as the answer refers to them]")
    seen: set[str] = set()
    for index, citation in enumerate(citations, start=1):
        again = " · another passage from the same note" if citation.identifier in seen else ""
        seen.add(citation.identifier)
        st.markdown(f"**[{index}]** {citation.title}")
        marks = [citation.shelf or "corpus", source_link(citation.identifier)]
        if not citation.reviewed:
            marks.append(f"{WARNING} fetched during this run, nobody has reviewed it")
        st.caption(" · ".join(marks) + again)

    # One collapsed bar, at the end, and the only place the untrimmed text lives.
    # The offline answer quotes a *lead* of each passage rather than all of it, so
    # suppressing this would silently throw the remainder away -- which is a worse
    # fault than the duplication it was suppressing. The duplication that had to go
    # was the uncollapsed kind: every title as a heading and every passage in full,
    # printed once by the answer and again by this list.
    with st.expander(f"Read the passages themselves, untrimmed ({len(citations)})"):
        st.caption(
            "Shown as they were indexed rather than summarised. A summary is where a "
            "qualifier goes missing, and the point of quoting is that the claim can "
            "be checked against the words it came from."
        )
        for index, citation in enumerate(citations, start=1):
            st.markdown(f"**[{index}] {citation.title}**")
            prose(as_quotation(citation.snippet))


QUERY_AUTHORS: dict[str, str] = {
    "grader": "the relevance grader, which had just read the passages it threw out",
    "model": "a model asked for nothing except a better query",
    "heuristic": "the deterministic fallback -- keep the subject, add the model's "
    "name -- which runs with no language model at all",
}
"""How to name whoever wrote a reformulated query, keyed by ``SearchRound.rewritten_by``.

Named rather than counted. "The query was rewritten" and "the query was rewritten by
the fallback because no model was reachable" are different facts about a run, and the
second is the one to know *before* reading the query itself: a reader who sees a
mechanical-looking query and does not know a model was unavailable will conclude the
rewriter is poor rather than absent.
"""


def query_trace(state: CampaignState) -> None:
    """Show the question as it was typed beside the query that actually went to the index.

    Both halves are always drawn, and that is the whole design. Showing the searched
    query *only* when a rewrite happened means the common case -- the first round
    worked -- displays a question and then silence, and a reader cannot tell from the
    screen whether the query was left alone or merely not reported. "Nothing was
    rewritten" is a claim a trace has to make out loud.

    The two sides differ on the feasibility branch even when nothing is rewritten,
    because the opening query there is built rather than quoted: it deliberately asks
    about the classical methods too, so that the reading list behind a verdict is not
    one in which quantum always looks promising. That is a decision worth seeing, and
    before this panel existed there was nowhere in the interface it could be seen.

    Args:
        state: The finished campaign.
    """
    rounds = state["searches"]
    st.markdown("##### What was asked, and what was searched for")
    asked, searched = st.columns(2)
    with asked:
        st.caption("The question, as you typed it")
        st.code(state["request"].text, language=None, wrap_lines=True)
    rewrites = [item for item in rounds if item.kind == "rewrite"]
    last = rounds[-1] if rounds else None
    with searched:
        st.caption("Rewritten search query" if rewrites else "The query that went to the index")
        st.code(
            last.query if last is not None else "(nothing was searched)",
            language=None,
            wrap_lines=True,
        )

    if not rounds:
        st.caption(
            f"{ABSENT} No search was recorded -- either the question needed none, or "
            "the notes could not be opened. The question above was never sent anywhere."
        )
        return

    if rewrites:
        rewrite = rewrites[-1]
        st.caption(
            f"The first search kept nothing, so the query was rewritten by "
            f"{QUERY_AUTHORS.get(rewrite.rewritten_by, rewrite.rewritten_by)}. The "
            f"rewritten one found {rewrite.found} and kept {rewrite.kept}."
        )
    else:
        opening = rounds[0]
        st.caption(
            f"Searched once, with no reformulation: the first round found "
            f"{opening.found} and kept {opening.kept}, so there was nothing for a "
            "rewriter to improve on. On a feasibility question the query above is "
            "built rather than quoted -- it asks about the classical methods as well, "
            "so that the background behind a verdict is not a reading list in which "
            "quantum always wins."
        )

    with st.expander(f"Every round, in order ({len(rounds)})"):
        st.caption(
            "One row per search. The row that matters is a round with a high **found** "
            "and a **kept** of zero: the index answered and the grader threw all of it "
            "out, which is what triggers a rewrite and what a bare citation list can "
            "never show."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "round": number,
                        "query": item.query,
                        "searched": item.where(),
                        "found": item.found,
                        "kept": item.kept,
                        "written by": item.author(),
                    }
                    for number, item in enumerate(rounds, start=1)
                ]
            ),
            hide_index=True,
            width="stretch",
        )


WHO_PROPOSED: dict[str, str] = {
    "model": "by a language model",
    "deterministic": "by rule, with no language model involved",
    "none": "by nothing -- the step did not run",
}
"""How to say who wrote the follow-up questions, keyed by ``Followups.proposed_by``.

The bare literal was being dropped into the sentence, which produced "by the
deterministic" on every offline run -- the interface's own text ungrammatical on the
path somebody with no credential meets first. Naming them properly also makes the
distinction usable: suggestions written by a model and suggestions composed from the
run's own facts are different things, and which one a reader is looking at changes
how much weight the suggestion deserves.
"""


def followups(state: CampaignState, position: int) -> None:
    """Offer what to ask next, as buttons that actually ask it.

    Buttons rather than a printed list, for one reason: a suggestion the reader has
    to retype is a suggestion most readers will not take. They set the same
    ``pending`` key the starter questions use, so a click and a typed question reach
    the agent by exactly the same path -- including the input screen, which is what
    makes it safe to put model-written text on a button.

    Args:
        state: The finished campaign.
        position: Index in the session's history, so two answers on screen cannot
            collide on a widget key.
    """
    proposed = state["followups"]
    if not proposed.any_offered:
        return
    st.markdown("**What to ask next**")
    for index, suggestion in enumerate(proposed.suggestions):
        st.button(
            suggestion.question,
            key=f"followup-{position}-{index}",
            help=suggestion.why,
            on_click=lambda text=suggestion.question: st.session_state.__setitem__("pending", text),
        )
    st.caption(
        f"Proposed from what this run established, {WHO_PROPOSED[proposed.proposed_by]}. "
        "Each one was screened on the way out, exactly as a question you typed is "
        "screened on the way in."
    )


def learned_audience(store: Memory) -> Audience | None:
    """The reading level this reader's earlier ratings settled on, if any.

    Args:
        store: This session's memory.

    Returns:
        The level, or ``None`` when nothing has been rated -- which is the first
        visit and is also what a memory that cannot be written looks like. Both
        correctly learn nothing.
    """
    if not store.enabled:
        return None
    recorded = store.describe().get("ratings")
    if not recorded:
        return None
    return store.preferred_audience()


def _shift(level: Audience, steps: int) -> Audience:
    """Move a reading level along the scale, stopping at each end.

    :data:`AUDIENCE_LABELS` is ordered by how much of *this domain* is assumed, not
    by seniority, which is what makes "one step simpler" a meaningful instruction
    rather than a slight.

    Args:
        level: Where it is now.
        steps: How far to move, negative towards less assumed knowledge.

    Returns:
        The new level, clamped to the ends of the scale.
    """
    scale = list(AUDIENCE_LABELS)
    place = scale.index(level) if level in scale else scale.index(DEFAULT_AUDIENCE)
    return _as_audience(scale[max(0, min(len(scale) - 1, place + steps))])


def answer_feedback(
    store: Memory, level: Audience, position: int, *, offline: bool = False
) -> None:
    """Offer the two buttons the agent learns a reading level from.

    Two buttons and no free-text box, because a rating has to be cheap enough that
    somebody actually gives one. What it buys is concrete and is said on the spot,
    which is the difference between feedback and a suggestion box: the level is
    written to memory, it survives into the next conversation, the settings knob
    opens there next time, and dragging the knob still wins.

    The down button has a direction rather than a mood. "This was wrong for me" is
    not actionable and a rating nothing can act on is a rating that should not have
    been collected; "too much assumed" is, and it is the failure this project
    expects, since the interface assumes no physics of its reader.
    Somebody who wanted the *opposite* has the slider, which is one click away and
    is named in the caption.

    Args:
        store: This session's memory. A disabled one draws nothing at all rather
            than buttons that quietly discard what they were told.
        level: The reading level this answer was written at.
        position: Index in the session's history, so two answers on screen cannot
            collide on a widget key.
        offline: Whether this answer was produced with no language model. The
            buttons still work -- the level is stored and applies the moment a model
            is switched on -- but the prompt above them says so, because with no
            model the answer was assembled from templates and the reading level
            changed nothing about the text the reader just read. Asking somebody to
            rate the pitch of something whose pitch was not chosen is the kind of
            small dishonesty that makes every other control on the page suspect.
    """
    if not store.enabled:
        return
    given: dict[int, str] = st.session_state.setdefault("ratings", {})
    if position in given:
        st.caption(
            f"Thanks — noted. The next answer is written for **{given[position]}**, "
            "here and in your next conversation. The settings knob overrides it."
        )
        return

    def record(chosen: Audience) -> None:
        store.remember_rating(chosen)
        given[position] = chosen

    st.caption(
        "Was this pitched right? It sets how much the agent explains, and nothing else."
        if not offline
        else "How much should it explain? No model wrote this one — the answer was "
        "assembled from templates — so this sets the level for when one is switched on."
    )
    up, down, _ = st.columns([1, 1, 8])
    up.button(
        "👍",
        key=f"rate-up-{position}",
        help=f"About right. Keep writing for a reader who {AUDIENCE_LABELS[level]}.",
        on_click=record,
        args=(level,),
    )
    down.button(
        "👎",
        key=f"rate-down-{position}",
        help=(
            "Too much assumed. The next answer is written for a reader who "
            f"{AUDIENCE_LABELS[_shift(level, -1)]}."
        ),
        on_click=record,
        args=(_shift(level, -1),),
    )


def campaign_evidence(state: CampaignState) -> None:
    """Draw everything behind a verdict, in four tabs.

    Four rather than one long page, and in this order: what it did, what it
    concluded, what it ran, what it refused. A reader who trusts the verdict reads
    none of them; a reader who does not starts at the first, which is the trail
    written as the campaign went rather than reconstructed afterwards.

    Drawn on the **Pipeline trace** page and deliberately not under the reply on the
    Chat page. Four tabs of machinery beneath every answer served neither reader: it
    pushed the conclusion out of sight for somebody who came to ask a question, and it scattered
    the trail across the thread for somebody auditing the agent, who wants it in one
    place and beside the graph it ran through. Nothing is hidden by the move -- this
    is the same panel, one page over, next to the route it describes.

    Args:
        state: The finished campaign.
    """
    trail, report, ran, refused = st.tabs(
        ["What it did", "The report", "What it ran", "What it refused"]
    )

    with trail:
        st.caption(
            "One line per step, written as the campaign went. It is what decides "
            "whether an answer was earned or merely arrived at."
        )
        for index, note in enumerate(state["notes"], start=1):
            st.markdown(f"`{index:2d}` &nbsp; {note}", unsafe_allow_html=True)

    with report:
        if state["report"]:
            st.markdown(state["report"])
        else:
            st.info("No report was composed.")

    with ran:
        _runs_table(state)

    with refused:
        _refusals_table(state)


def _runs_table(state: CampaignState) -> None:
    """Draw every configuration that was actually tried.

    Args:
        state: The finished campaign.
    """
    if not state["runs"]:
        st.info("Nothing was run: everything proposed was refused before it cost anything.")
        return
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "configuration": record.label,
                    "layers": record.depth,
                    "depth on the machine": record.two_qubit_depth,
                    "energy per magnet": round(record.energy_per_site, 6),
                    "measurements": f"{record.shots_spent:,}",
                    "what happened": record.diagnosis.evidence,
                    "what to change": record.diagnosis.repair,
                }
                for record in state["runs"]
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    baseline = state["classical"]
    if baseline is not None:
        st.caption(
            f"The ordinary computer reached **{baseline.energy_per_site:.6f}** per magnet "
            "for comparison. Every energy here is an upper bound on the true lowest one, "
            "so lower is better -- and a number below the reference is a bug rather than "
            "a discovery."
        )


def _refusals_table(state: CampaignState) -> None:
    """Draw every configuration that was rejected before it cost anything.

    Args:
        state: The finished campaign.
    """
    if not state["ruled_out"]:
        st.info("Nothing was refused.")
        return
    st.caption(
        "Refused on arithmetic before anything was spent, which is what a practitioner "
        "does before booking machine time. A campaign that recorded only what worked "
        "could not be told apart from one that got lucky on its first try."
    )
    st.dataframe(
        pd.DataFrame(
            [
                {"configuration": ruled.label, "why it was refused": ruled.reason}
                for ruled in state["ruled_out"]
            ]
        ),
        hide_index=True,
        width="stretch",
    )


# --------------------------------------------------------------------------
# The two knobs
# --------------------------------------------------------------------------

AUDIENCE_LABELS: dict[str, str] = {
    "beginner": "no background at all",
    "entrepreneur": "wants the decision and the cost",
    "software, no physics": "fluent in code, new to the physics",
    "practitioner": "works with this material",
    "quantum computing engineer": "knows circuits and hardware",
    "researcher": "wants the derivation",
}
"""How each reader is described, in words rather than as a bare slug.

Ordered by how much of *this domain* is assumed rather than by seniority, which is
what lets it be a slider at all. ``software, no physics`` sits third not because it
assumes less skill than ``practitioner`` but because it assumes less about
*magnets*, and the two are independent: somebody can write compilers and still want
the word "Hamiltonian" unpacked. Each name describes a background rather than a job
title, because a title predicts neither.
"""

AUDIENCE_VALUES: tuple[Audience, ...] = get_args(Audience)
"""The reading levels, read off the type so the two cannot drift apart."""

SETTING_KEY = "setting"
"""Where the knob's position lives between reruns."""


def read_settings_knob() -> knob.Setting:
    """Draw both knobs in the sidebar and return where they are pointing.

    Two tabs, and the separation is the point rather than the layout. **Physics**
    changes what is computed; **Language model** changes only how it is described.
    A visitor who suspects the numbers are coming out of a language model can open
    the second tab, switch the model off, and watch every number on every page stay
    exactly where it was. One undifferentiated panel would make that experiment
    impossible to perform and the claim impossible to believe.

    Returns:
        The knob's position, validated.
    """
    previous = st.session_state.get(SETTING_KEY)
    moved = previous.changes_from_defaults() if isinstance(previous, knob.Setting) else ()
    label = "⚙️ Settings knob" + (f" — {len(moved)} changed" if moved else "")

    with st.sidebar:
        # No "⚙️ Settings" heading above this. It printed the same word and the same
        # glyph as the expander directly beneath it, two lines apart -- the failure
        # the navigation already fixed once when a group called Chat held a page
        # called Chat. The caption says the one thing the heading did not: that the
        # knob can be left shut.
        st.caption("Everything adjustable is in here. The defaults answer a sensible question.")
    with st.sidebar, st.expander(label, expanded=False):
        if moved:
            st.caption("Changed from default: " + "; ".join(moved) + ".")
        physics_tab, model_tab, search_tab = st.tabs(
            ["Physics", "Language model", "Knowledge search"]
        )

        with physics_tab:
            st.caption("These change the computed number.")
            n_sites = st.slider(
                "Sites $L$",
                min_value=knob.MIN_SITES,
                max_value=knob.MAX_SITES,
                value=knob.DEFAULT_SITES,
                help="One qubit each.",
            )
            shape = st.radio(
                "Boundary",
                options=("open", "periodic"),
                format_func=lambda value: (
                    "a line with two ends" if value == "open" else "a closed ring"
                ),
                horizontal=True,
                help="A ring needs its two ends to interact, and no machine here "
                "wires them together. It is the cheapest way to see what "
                "connectivity costs.",
            )
            coupling = st.slider("Coupling $J$", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
            field = st.slider(
                "Field $h$",
                min_value=0.0,
                max_value=3.0,
                value=1.0,
                step=0.05,
                help="Only the ratio to the coupling matters. At h/J = 1 the chain "
                "is critical, which is where every approximate method has its "
                "hardest time.",
            )
            # Behind an expander, and the only control in the application that is.
            # Everything a reader meets describes a two-parameter problem, because
            # that is the problem this project solves and grades: one ratio, h/J,
            # with a closed-form answer to be marked against. `g` is real, it is
            # implemented, and it is what would make the feasibility question
            # genuinely open -- but it is zero by default, it moves no number on any
            # page until it is touched, and there is no exact answer to grade a run
            # against once it is. A dial like that belongs one click away, with the
            # consequence stated, rather than in front of every reader as a third
            # free parameter they have to account for.
            with st.expander("Advanced — a third field"):
                st.caption(
                    "Everything above describes a chain with two terms, $J$ and $h$, "
                    "and that is the problem this project can grade: it has a "
                    "closed-form answer that the agent cannot see and is marked "
                    "against. Adding a field along the coupling direction breaks that "
                    "— there is then no exact answer to mark against, and the honest "
                    "verdict stops being automatic. That is exactly why it is "
                    "interesting, and exactly why it is not the default."
                )
                longitudinal = st.slider(
                    "Longitudinal field $g$",
                    min_value=0.0,
                    max_value=knob.MAX_LONGITUDINAL,
                    value=0.0,
                    step=0.05,
                    help="A field along the coupling direction. Zero leaves the chain "
                    "exactly solvable, which is what makes a closed-form check "
                    "possible. Non-zero is what makes the feasibility question "
                    "non-trivial, and it costs no two-qubit depth. The displayed "
                    "Hamiltonian grows a third term as soon as this moves.",
                )
            precision = st.select_slider(
                "Target accuracy per site",
                options=list(knob.PRECISION_CHOICES),
                value=knob.DEFAULT_PRECISION,
                format_func=lambda value: f"{value:.0e}",
                help="Measurements scale as the inverse square of this, so each "
                "step down the list costs a hundred times as much.",
            )
            depth = st.slider(
                "Circuit layers",
                min_value=1,
                max_value=knob.MAX_CIRCUIT_DEPTH,
                value=knob.DEFAULT_DEPTH,
            )
            device = st.selectbox(
                "Machine",
                options=list(device_names()),
                index=list(device_names()).index(knob.DEFAULT_DEVICE),
            )
            shots = st.select_slider(
                "Measurement budget",
                options=list(knob.SHOT_CHOICES),
                value=knob.DEFAULT_SHOTS,
                format_func=lambda value: f"{value:,}",
            )

        with model_tab:
            st.caption(
                "These change only the words. Move anything here and every number "
                "in this application stays exactly where it was -- which is the "
                "fastest way to check that no model computed one."
            )
            # Opens where the process actually runs, like every transport dial
            # below it: on a checkout with a credential the model is *on*, and on one
            # without it there is nothing to switch on. It was hard-coded to `True`,
            # which meant a fully configured application opened with its model off
            # and answered everything on the deterministic path -- correct numbers,
            # templated prose, and nothing on screen saying why.
            has_key = knob.credential_available()
            offline = st.toggle(
                "Run without a language model",
                value=knob.default_offline(),
                disabled=not has_key,
                help="On, the agent still designs circuits, prices them, runs the "
                "baseline and reaches a verdict by rule. The question is read by "
                "pattern and the report is assembled from templates. Costs nothing "
                "and needs no key.",
            )
            if not has_key:
                st.caption(
                    f"{ABSENT} No `OPENROUTER_API_KEY` is configured, so there is no "
                    "model to switch on. Everything below still runs -- the circuits, "
                    "the baseline and the verdict are arithmetic."
                )
            # Opens where the ratings left it. A 👍 or 👎 under an answer writes a
            # level to memory, and a slider that ignored it would make the buttons
            # a suggestion box: the reader would rate an answer, watch the knob
            # still say "practitioner", and reasonably conclude nothing happened.
            taught = learned_audience(conversation())
            audience = st.select_slider(
                "Explain it for",
                options=list(AUDIENCE_LABELS),
                value=taught or knob.DEFAULT_AUDIENCE,
                format_func=lambda value: f"{value} — {AUDIENCE_LABELS[value]}",
                help="Changes who the prose is aimed at and nothing else. The "
                "energy, the depth, the shot count and the verdict are identical "
                "at every position on this slider.",
            )
            if taught is not None:
                st.caption(
                    f"Set to **{taught}** by your ratings under earlier answers — "
                    "drag it anywhere and your choice wins."
                )
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=knob.DEFAULT_TEMPERATURE,
                step=0.1,
                help="How much the model may vary its wording. Reachable precisely "
                "so a sceptic can turn it up and watch the numbers not move.",
            )
            st.divider()
            st.caption(
                "**Which model serves which call.** The tiers are not three sizes "
                "of the same thing -- the cheap one serves everything a person is "
                "waiting on, and the strong one serves the written report and "
                "nothing else. These start on the models the agent actually falls "
                "back to, named rather than hidden behind the word 'default'."
            )
            offered = list(selectable_slugs())
            # A guest is pinned to the free endpoint, so these are shown disabled
            # and said out loud rather than left live and quietly ignored. A dial
            # that moves while the process does something else is the worst of the
            # three states this panel can be in.
            who = access.visitor()
            if not who.may_choose_models:
                st.caption(
                    "Sign in, or use your own key, to choose these. Guests run on "
                    "the free endpoint -- which changes the prose and none of the "
                    "numbers."
                )
            chosen_tiers = tuple(
                (
                    tier,
                    st.selectbox(
                        f"{tier} tier",
                        options=offered,
                        index=offered.index(default) if default in offered else 0,
                        key=f"tier_{tier}",
                        disabled=not who.may_choose_models,
                    ),
                )
                for tier, default in knob.DEFAULT_TIERS
            )
            one_model_everywhere(chosen_tiers)
            max_calls = st.select_slider(
                "Model calls per question",
                options=list(knob.MAX_CALLS_CHOICES),
                value=knob.DEFAULT_MAX_CALLS,
                help="A wall rather than a warning. A budget written into a prompt "
                "is a suggestion; one that refuses the call is a budget.",
            )
            st.divider()
            st.caption(
                "**How the gateway is called.** Latency, cost and what happens when "
                "a call fails. Every one of these opens where the process actually "
                "runs, read off the configuration rather than retyped here — a "
                "slider that opened at a stale literal would not merely mislabel "
                "itself, it would quietly reconfigure the run to that literal."
            )
            cap_length = st.checkbox(
                "Cap the reply length",
                value=knob.DEFAULT_MAX_OUTPUT_TOKENS is not None,
                help="Unticked leaves the provider's own default in place, which on "
                "some models is very long and is billed by the token.",
            )
            max_output_tokens = (
                st.slider(
                    "Max output tokens",
                    min_value=knob.MAX_OUTPUT_TOKEN_BOUNDS[0],
                    max_value=knob.MAX_OUTPUT_TOKEN_BOUNDS[1],
                    value=knob.DEFAULT_MAX_OUTPUT_TOKENS,
                    step=64,
                )
                if cap_length
                else None
            )
            max_retries = st.slider(
                "Retries on transient failure",
                min_value=0,
                max_value=knob.MAX_RETRY_BOUND,
                value=knob.DEFAULT_MAX_RETRIES,
                help="Only transient failures are retried — a rate limit or a "
                "dropped connection. A refusal or a malformed request is permanent "
                "and is never tried again, however high this goes.",
            )
            requests_per_second = st.slider(
                "Requests per second",
                min_value=0.0,
                max_value=knob.MAX_REQUESTS_PER_SECOND,
                value=knob.DEFAULT_REQUESTS_PER_SECOND,
                step=0.5,
                help="Spacing between outbound calls; zero switches the throttle "
                "off. A campaign's calls are sequential, so this widens the gaps "
                "rather than slowing the work — it is what keeps an evaluation "
                "sweep under the gateway's rate limit.",
            )
            request_timeout = st.slider(
                "Request timeout (s)",
                min_value=knob.REQUEST_TIMEOUT_BOUNDS[0],
                max_value=knob.REQUEST_TIMEOUT_BOUNDS[1],
                value=knob.DEFAULT_REQUEST_TIMEOUT_S,
                step=5.0,
                help="How long one call may hang before it is abandoned and, if any "
                "retries are left, tried again.",
            )
            if offline:
                st.caption(
                    f"{ABSENT} No model is being called, so nothing on this tab is "
                    "doing anything. Everything on the Physics tab still is."
                )

        with search_tab:
            st.caption(
                "What the agent may look up. These change what the answer stands "
                "on, which is why they belong beside the dials that change what it "
                "computes rather than hidden inside the retrieval layer."
            )
            corpus = st.checkbox(
                "Search the project's notes",
                value=True,
                help="The keyword half of the search needs no language model, so "
                "this returns citations even with the model off. Turn it off once "
                "to see what an answer rests on without them.",
            )
            external = st.checkbox(
                "Let it search arXiv",
                value=False,
                help="A network call, reached only after a local search found "
                "nothing -- so a question the notes already answer never pays for "
                "it. Anything fetched this way is cited as unreviewed, because it "
                "is: the text is real and nobody has read it.",
            )
            shelf = st.selectbox(
                "Knowledge base",
                options=list(knob.SHELF_CHOICES),
                index=0,
                help="Three separate literatures. A hardware measurement, a "
                "closed-form result and an industrial pilot are different kinds of "
                "claim, and a search that mixes them will answer a question about "
                "circuit depth out of a note on portfolio optimisation.",
            )
            passages = st.slider(
                "Passages to use",
                min_value=1,
                max_value=12,
                value=knob.DEFAULT_PASSAGES,
                help="More material to draw on, and more of it ranked low enough "
                "that the grader nearly dropped it.",
            )
            vector_share = st.slider(
                VECTOR_SHARE_LABEL,
                min_value=0.0,
                max_value=1.0,
                value=knob.DEFAULT_VECTOR_SHARE,
                step=0.1,
                help="The rest is keyword-matching, which is what finds a passage "
                "naming an author or an arXiv number. Taking this to one is how a "
                "search stops finding those.",
            )
            rounds = st.slider(
                "Searches per question",
                min_value=1,
                max_value=knob.MAX_SEARCH_ROUNDS,
                value=knob.MAX_SEARCH_ROUNDS,
                help="Two lets a search that found nothing be rewritten from the "
                "grader's own verdict and tried again. One switches that corrective "
                "loop off, which is how you see what it was doing.",
            )
            followups = st.checkbox(
                "Suggest what to ask next",
                value=True,
                help="Each suggestion is screened as if you had typed it. Worth "
                "turning off while metering a campaign, since a suggestion is a "
                "model call like any other.",
            )

    return knob.Setting(
        physics=knob.Physics(
            n_sites=n_sites,
            boundary="periodic" if shape == "periodic" else "open",
            coupling=coupling,
            field=field,
            longitudinal=longitudinal,
            precision=precision,
            depth=depth,
            device=device,
            shots=shots,
        ),
        model=knob.Model(
            offline=offline,
            audience=_as_audience(audience),
            temperature=temperature,
            tiers=chosen_tiers,
            max_calls=max_calls,
            max_output_tokens=max_output_tokens,
            max_retries=max_retries,
            requests_per_second=requests_per_second,
            request_timeout_s=request_timeout,
        ),
        search=knob.Search(
            corpus=corpus,
            external=external,
            passages=passages,
            vector_share=vector_share,
            shelf=shelf,
            rounds=rounds,
            followups=followups,
        ),
    )


def _as_audience(value: str) -> Audience:
    """Narrow a widget's string back to the reading levels the project defines.

    By comparison rather than by a cast, so that renaming a level is a type error
    here rather than a setting that silently falls back to the default.

    Args:
        value: What the widget returned.

    Returns:
        The matching level, or the project default if it matches none.
    """
    for candidate in AUDIENCE_VALUES:
        if candidate == value:
            return candidate
    return DEFAULT_AUDIENCE


def current_setting() -> knob.Setting:
    """The knob's position, for a page that did not draw it.

    The entry point draws the knob once per run and every page reads it here, so the
    sidebar is identical on all of them. A page opened on its own -- which is how a
    test reaches one -- draws its own rather than failing, so no page depends on
    having been arrived at through the navigation.

    Returns:
        The position.
    """
    # Every page but About passes through here, including a page opened on its own,
    # which is the path `open_page` does not cover.
    start_logging()
    existing = st.session_state.get(SETTING_KEY)
    return existing if isinstance(existing, knob.Setting) else read_settings_knob()


def model_has_no_say(what: str) -> None:
    """State, on a page that computes, that no language model was involved.

    Args:
        what: What the page computed, named in a few words.
    """
    st.caption(
        f"{PRESENT} No language model was called to produce {what}. The **Language "
        "model** tab of the settings knob has no effect on this page -- move it and "
        "check."
    )


def asked_this_session() -> tuple[session_log.Asked, ...]:
    """Every question this session has answered, oldest first.

    The one place session state is read for the analytics. Pages go through here
    rather than reaching for the key themselves, so that how the conversation is
    stored stays a detail of this module -- and so that a page cannot accidentally
    read a shape the chat never wrote.

    Returns:
        The rows, or an empty tuple before anything has been asked.
    """
    return tuple(st.session_state.get("asked", []))


def session_cost() -> None:
    """Report what this session's questions have cost in model calls and money.

    Kept in one place. A running total shown in two places is a total that will
    eventually disagree with itself, and the disagreement is invisible until somebody
    checks -- at which point neither figure can be trusted.

    Zero is a result, not an empty panel. "No model was called" is a fact about how
    the answer was reached and one of the more surprising things about this
    application, so it is stated rather than left as a blank.
    """
    spent = asked_this_session()
    total = session_log.totals(spent)
    calls = total.calls_made
    metrics(
        {
            "Questions asked": len(spent),
            "Model calls": calls,
            "Tokens": f"{total.total_tokens:,}",
            "Estimated spend (USD)": total.cost_label(),
        }
    )
    if not spent:
        st.caption("Nothing asked yet in this session.")
        return
    if calls == 0:
        st.caption(
            f"{PRESENT} **No language model was called.** Every one of those answers "
            "was reached on the deterministic path -- the question read by pattern, "
            "the circuits designed and priced by arithmetic, the verdict decided by "
            "rule. What a model would have added is the prose."
        )
        return
    basis = total.cost_basis()
    st.caption(
        f"Across {len(spent)} question(s). "
        + ("Every call was priced from the catalogue." if not basis else f"{WARNING} {basis}")
    )
    by_model = session_log.calls_by_model(spent)
    st.dataframe(
        pd.DataFrame([{"model": name, "calls": count} for name, count in by_model.items()]),
        hide_index=True,
        width="stretch",
    )


def footer(source: str) -> None:
    """Close a page with the path a reader can go and read instead.

    Args:
        source: Repository-relative path of the module behind the page.
    """
    st.divider()
    st.caption(
        f"Drawn from `{source}`, in `{PROJECT_ROOT.name}`. "
        "The interface computes nothing of its own -- if a number here is wrong, it "
        "is wrong in the module, where a test can catch it."
    )


def figure(
    drawn: Any,
    caption: str = "",
    download: str | None = None,
    key: str | None = None,
) -> None:
    """Draw a figure, with the option of taking it away as a PDF.

    Every figure this project writes is a PDF -- see :mod:`src.figure_export` for
    why -- and a figure a reader can only screenshot is not one. The button hands
    over the same vector file a script would have written, built in memory so that
    looking at a page never writes into ``reports/figures/``.

    Args:
        drawn: A matplotlib ``Figure``.
        caption: What the reader should take from it, in plain words. Drawn under
            the figure rather than as a title, because a caption that states the
            claim is the part a non-specialist actually reads.
        download: A bare slug for the file, or ``None`` for no button. Slugs are
            checked by :func:`src.figure_export.figure_path`, so a name with a
            directory in it fails here rather than writing somewhere unexpected.
        key: What to key the button on, when the slug is not unique on the page.
            A chat thread draws every past answer's figures as well as the newest,
            so a panel that can appear more than once needs the position in the
            key -- two buttons sharing one key is a Streamlit error that takes the
            whole page down, not a cosmetic clash. Defaults to the slug, which is
            right for a figure that appears once.
    """
    # Rasterising parses the same maths the builder did, so it takes the same lock:
    # `st.pyplot` and the download both run matplotlib's shared text machinery, and
    # this call happens on whichever thread Streamlit gave the script run.
    with figures.DRAWING:
        st.pyplot(drawn, width="stretch")
    if caption:
        st.caption(caption)
    if download is None:
        return
    buffer = BytesIO()
    with figures.DRAWING:
        drawn.savefig(buffer, format=FIGURE_FORMAT)
    st.download_button(
        f"Download as {FIGURE_FORMAT.upper()}",
        data=buffer.getvalue(),
        file_name=figure_path(download).name,
        mime="application/pdf",
        key=f"download-{key or download}",
    )


def corpus_notes() -> tuple[dict[str, str], ...]:
    """Read the committed corpus so the Knowledge base page can publish it.

    Reads the *files*, not the index, and the difference is the point: a note
    edited since the last ``make ingest`` appears here and is not yet retrievable,
    which is a fact a reader is owed rather than one to be hidden. The page says so.

    Returns:
        One row per note -- slug, shelf, title, citation, declared topics, length
        in words, and whether a person wrote it or a search turned it up. Empty
        when the corpus cannot be read at all, which the page reports as a
        sentence rather than a traceback: a missing corpus is a deployment
        mistake, not a bug. It is logged either way, because "no notes" from an
        empty directory and "no notes" from an unreadable one are different
        problems and only the log can tell them apart.
    """
    try:
        notes = load_library()
    except Exception as error:
        LOG.warning(
            "corpus_unreadable",
            extra={
                "error_type": type(error).__name__,
                "detail": "knowledge-base page has nothing to show",
            },
        )
        return ()
    return tuple(
        {
            "note": note.slug,
            "shelf": note.shelf,
            "title": note.title,
            "source": note.source,
            "topics": ", ".join(note.topics),
            "words": str(len(note.body.split())),
            "written by": "a person" if note.provenance == CURATED else "a search",
        }
        for note in notes
    )


def render_knowledge() -> None:
    """Publish what the agent has read, shelf by shelf.

    The commonest way a retrieval application misleads somebody is by refusing a
    perfectly reasonable question: that reads as *this cannot be answered* when it
    means *this is not in my notes*. Showing the whole corpus turns the refusal
    into something a visitor can check for themselves, and it is the only way the
    claim "there is no hidden corpus" can be believed by anyone who did not write
    the code.

    Grouped by shelf because *which* shelf to search is a decision the agent makes
    per question, and a reader cannot judge that decision without seeing what is on
    each one.
    """
    rows = corpus_notes()
    if not rows:
        st.error(
            "The corpus could not be read. It lives in `data/corpus/`, one directory "
            "per knowledge base, and is committed to the repository. Each note is "
            "Markdown carrying `title`, `source` and `topics` in its frontmatter."
        )
        return

    topics = sorted({topic for row in rows for topic in row["topics"].split(", ") if topic})
    metrics(
        {
            "Knowledge bases": len({row["shelf"] for row in rows}),
            "Notes": len(rows),
            "Topics": len(topics),
            "Words": f"{sum(int(row['words']) for row in rows):,}",
        }
    )
    st.caption(
        "**Note** — one Markdown file the agent may quote from. **Shelf** — one "
        "knowledge base; the agent picks which to search per question, and widens to "
        "the others when the first comes back empty. **Topic** — a label declared in "
        "a note's own header, used to steer the search towards it."
    )

    for shelf in SHELVES:
        on_shelf = [row for row in rows if row["shelf"] == shelf.name]
        if not on_shelf:
            continue
        st.markdown(
            f"**{shelf.title}** &nbsp; `{shelf.name}` &nbsp; · &nbsp; {len(on_shelf)} notes"
        )
        st.caption(f"Covers {shelf.covers}.")
        st.dataframe(
            [{key: value for key, value in row.items() if key != "shelf"} for row in on_shelf],
            hide_index=True,
            width="stretch",
        )

    unshelved = [row for row in rows if get_shelf(row["shelf"]) is None]
    if unshelved:
        st.warning(
            f"{len(unshelved)} notes sit in a directory no shelf declares, so nothing "
            "searches them. The declaration is in `src/rag/ingest.py`."
        )

    st.markdown("**Topics the search can be steered by**")
    st.markdown(" ".join(f"`{topic}`" for topic in topics))

    st.caption(
        "Every note carries a citation precise enough to check the claim against the "
        "original, and **no note contains a computed number** -- a number sitting in "
        "the corpus would be retrievable, and indistinguishable from one this project "
        "worked out and verified. Numbers are computed or they are not shown."
    )

    with st.expander("What happens to a note before it can be retrieved"):
        st.markdown(
            "1. **Parsed.** A note without a title, a citation and a topic is refused "
            "at ingest rather than indexed uncitable.\n"
            "2. **Split on its own headings**, so a retrieved passage is a section with "
            "a heading trail and can be attributed to a place rather than to a file.\n"
            "3. **Embedded** and written to the vector store under an id derived from "
            "the note and the position, so a second ingest *replaces* a chunk instead "
            "of adding a copy of it. Duplicates crowd out the passages that would have "
            "filled those slots, and a vector store degrades silently.\n"
            "4. **Reconciled.** Chunks the corpus no longer produces are deleted -- "
            "otherwise a shortened note keeps answering from its old tail forever.\n\n"
            "The table above reads the **files**; the agent searches the **index**. "
            "Run `make ingest` after changing anything in `data/corpus/`, or the "
            "difference between the two is invisible."
        )
