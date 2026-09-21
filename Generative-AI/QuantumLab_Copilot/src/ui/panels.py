"""Streamlit front end for QuantumLab Copilot.

Deliberately thin: every decision it renders is made in :mod:`src.verification`
and :mod:`src.physics`, because logic living inside a Streamlit script cannot be
unit-tested. This module chooses widgets, picks colours and formats numbers.

**Every dial lives behind one collapsed knob.** The main surface shows the
answer and the evidence for it; nothing that could be adjusted competes with
them for attention. Two audiences are served by one page that way -- a visitor
reads a verified number without meeting a single parameter, and anyone who wants
the machinery opens
the knob and finds all of them in one place, grouped by whether they change the
number or merely the prose. The bounds on those widgets are not written here:
they come from :mod:`src.agent.setting`, where they can be tested.

**Rendering the page calls no model and reads no credential**, so ``make run``
works on a checkout with no ``.env`` at all -- a tested invariant. A model is
reached only when somebody actually asks a question, and even then
:func:`src.agent.graph.ask` answers without one if there is none to be had.

The answer is shown in three layers, because two audiences share one page.
First the reply and what it understood the question to be. Then the evidence:
both methods' numbers and the citations. Last, behind an expander, the machinery
-- which route was taken, which methods ran, what the corpus search did. That
split is only possible because :class:`src.agent.graph.Answer` arrives as fields
rather than as prose; a paragraph cannot be folded.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib
import streamlit as st

matplotlib.use("Agg")  # headless: never try to open a window from a server process

from matplotlib.figure import Figure
from pydantic import SecretStr

from src.agent import graph, memory
from src.agent.deciding import MAX_STEPS
from src.agent.drafting import Draft
from src.agent.setting import (
    DEFAULT_SITES,
    DEFAULT_SWEEP_POINTS,
    FIELD_STEP,
    MAX_PASSAGES,
    MAX_SWEEP_POINTS,
    MIN_SWEEP_POINTS,
    MODEL_CHOICES,
    ModelSetting,
    PhysicsSetting,
    RetrievalSetting,
    Setting,
    ToolSetting,
)
from src.agent.usage import Usage
from src.evals import run as evals_run
from src.logging_setup import get_logger
from src.physics import exact, quantumness
from src.physics.model import HAMILTONIAN_INLINE, MAX_SITES_STATEVECTOR, TFIMSpec
from src.physics.quantumness import Configuration, Quantumness, Superposition
from src.rag.ingest import SHELVES, get_shelf, load_corpus, shelf_names
from src.rag.retrieve import (
    DEFAULT_TOP_K,
    DEFAULT_VECTOR_SHARE,
    MAX_ROUNDS,
    Passage,
    Retrieval,
    content_words,
)
from src.settings import (
    DEFAULT_AUDIENCE,
    DEFAULT_MAX_OUTPUT_TOKENS,
    Audience,
    Settings,
    get_settings,
)
from src.tools import mcp, websearch
from src.tools.papers import MAX_PAPERS
from src.tools.sweeps import (
    MAX_COSTLY_SITES,
    MAX_POINTS,
    MAX_RATIO,
    FieldSweep,
    Observable,
    sweep_field,
)
from src.ui import identity
from src.verification.cross_check import CrossCheck, cross_check

LOG = get_logger("ui.panels")
"""Where this page reports things a visitor cannot see: an unreadable corpus,
for instance, which shows on screen only as an empty table."""

# Written as comments, not as attribute docstrings: Streamlit's "magic" renders
# a bare string expression at module level, so a docstring here would appear on
# the page as content. This module is the one place in the project where that
# is true, which is why it is the one place that documents constants this way.

# Even chain lengths used in the finite-size panel. Even, because only an even
# ring admits the closed-form solution, which is what makes those points exact
# rather than merely computed.
SCALING_SITES = (4, 6, 8, 10, 12)

# What each audience level is called on screen. The slugs are the contract with
# src.settings.Audience; these are only labels.
AUDIENCE_LABELS: dict[str, str] = {
    "beginner": "Beginner",
    "practitioner": "Practitioner",
    "researcher": "Researcher",
}

ACCENT = "#4c6ef5"
HIGHLIGHT = "#e8590c"
MUTED = "#888888"

# Heading for each outcome of a run, with the colour of its badge. The three ways
# of not answering are shown as three different things, because they ask different
# things of the user: read why, answer a question, or grant permission. One shared
# "sorry" would hide which.
STATUS_LABELS: dict[str, tuple[str, str]] = {
    "answered": ("Answered", "#2f9e44"),
    "refused": ("Declined", "#c92a2a"),
    "clarification_needed": ("One more detail needed", "#f08c00"),
    "approval_needed": ("Waiting for your go-ahead", "#1971c2"),
}

STYLE = """
    <style>
      .block-container { padding-top: 2.5rem; max-width: 1100px; }
      h1 { font-size: 1.9rem !important; font-weight: 650 !important; letter-spacing: -0.01em; }
      [data-testid="stMetric"] {
          border: 1px solid rgba(128,128,128,0.22);
          border-radius: 0.5rem;
          padding: 0.8rem 1rem;
      }
      [data-testid="stMetricValue"] {
          font-variant-numeric: tabular-nums;
          font-size: 1.45rem;
      }
      [data-testid="stMetricLabel"] { opacity: 0.7; }
      .badge {
          display: inline-block;
          padding: 0.12rem 0.6rem;
          border-radius: 1rem;
          font-size: 0.74rem;
          font-weight: 600;
          letter-spacing: 0.04em;
          text-transform: uppercase;
      }
    </style>
"""
# The page's whole stylesheet, applied once by :func:`open_page`. Streamlit gives
# no styling API, so this is the only place raw CSS is allowed, and it is limited
# to spacing, weight and one badge class -- nothing that could hide a number.


TOP_BAR_CSS = f"""
    <style>
      /* Lift the top bar out of the page body and onto Streamlit's own header
         strip, to the left of the Deploy button.

         It belongs up there rather than in the page. Right-aligned inside a
         1100px content column it landed a long way in from the window's edge,
         which reads as the middle of the screen -- and a row of account controls
         in the middle of a page is a row of controls competing with the answer.

         `position: fixed` rather than a float, because the header strip is not
         this element's parent and no amount of margin will move it there. Out of
         the flow, so the caption it used to sit under closes up behind it.

         The right offset has to clear everything Streamlit puts in that strip,
         and the list is longer than Deploy and the menu: while a script is
         running -- which here means while a question is being answered, for tens
         of seconds -- a **Stop** button appears to their left. Covering that is
         the one overlap that is not merely cosmetic, because a reader watching a
         long run has to be able to stop it. 17rem clears Stop, Deploy and the
         menu with a margin left over.

         It is a measured guess rather than a computed value: the header is
         Streamlit's and its contents are not ours to interrogate. If a future
         version widens that toolbar, this is the one number to change, and the
         thing to check is the whole strip during a run rather than Deploy alone.

         Below 900px the header has no room to share, so the rule is dropped and
         the controls fall back into the page, where `horizontal_alignment` puts
         them against the right edge of the content column and they still fit. */
      @media (min-width: 900px) {{
          .st-key-{identity.TOP_BAR_NAME} {{
              position: fixed;
              top: 0.6rem;
              right: 17rem;
              z-index: 999991;
              width: auto;
              margin: 0;
          }}
      }}
    </style>
"""
# Kept apart from :data:`STYLE` because that one only paints and this one *moves*
# something. A rule that repositions an element can put a control somewhere
# unreachable; a rule that rounds a corner cannot. Keeping them separate means the
# risky stylesheet stays one rule long and reviewable.


def open_page(title: str) -> None:
    """Configure the page and apply the shared styling.

    Called at the top of the entry script, before anything is drawn. Streamlit
    requires ``set_page_config`` to come first, which is why it lives here rather
    than in each page.

    Args:
        title: What the browser tab should say.
    """
    st.set_page_config(page_title=title, page_icon="🔬", layout="wide")
    st.markdown(STYLE, unsafe_allow_html=True)
    st.markdown(TOP_BAR_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Computation — cached so that dragging a slider does not re-diagonalise
# --------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def solve(n_sites: int, coupling: float, field: float, boundary: str) -> CrossCheck:
    """Solve one specification with every applicable method and compare.

    Takes primitives rather than a :class:`~src.physics.model.TFIMSpec` because
    Streamlit keys its cache on the arguments, and primitives keep that key
    stable across reruns.

    Args:
        n_sites: Chain length ``L``.
        coupling: The Ising coupling ``J``.
        field: The transverse field ``h``.
        boundary: ``"periodic"`` or ``"open"``.

    Returns:
        Every method's answer, every refusal, and the verdict.
    """
    spec = TFIMSpec(
        n_sites=n_sites,
        coupling=coupling,
        field=field,
        boundary=boundary,  # type: ignore[arg-type]
    )
    return cross_check(spec)


@st.cache_data(show_spinner="Sweeping the field…")
def field_sweep(
    n_sites: int,
    coupling: float,
    boundary: str,
    points: int,
    observable: Observable = "ground_state",
    max_costly_sites: int = MAX_COSTLY_SITES,
) -> FieldSweep:
    """Trace the magnetisation and the energy across the field range.

    The resolution arrives as an argument rather than being read from the knob
    here, because Streamlit keys its cache on the arguments: a resolution the
    function looked up itself would not appear in the key, and lowering the
    slider would silently return the previous curve. ``observable`` is an argument
    for the same reason, and it is how this page draws the spectrum with exactly the
    figure the chat draws -- one definition of "the low-lying spectrum", rendered in
    two places, rather than two definitions that can drift apart.

    Args:
        n_sites: Chain length ``L``.
        coupling: The Ising coupling ``J``, held fixed across the sweep.
        boundary: ``"periodic"`` or ``"open"``.
        points: How many field values to sample.
        observable: Which curve is wanted. See
            :data:`~src.tools.sweeps.Observable`.
        max_costly_sites: Longest chain to sweep by diagonalisation. An argument
            for the same cache-key reason as the two above.

    Returns:
        The curve, with every point computed by both methods where both apply --
        or a declined sweep carrying the reason.
    """
    base = TFIMSpec(
        n_sites=n_sites,
        coupling=coupling,
        field=coupling,
        boundary=boundary,  # type: ignore[arg-type]
    )
    return sweep_field(base, points, observable, max_costly_sites)


@st.cache_data(show_spinner=False)
def momentum_spectrum(
    n_sites: int, coupling: float, points: int
) -> tuple[list[float], list[list[float]]]:
    r"""Quasiparticle energies at every allowed momentum, across the field range.

    The free-fermion mapping's own answer to "what does the low-lying spectrum look
    like as the field is turned up": one curve per allowed momentum, each the cost
    of exciting that mode. Closed form and ``O(L)`` per field value, so this is
    cheap at any chain length the closed form covers -- no diagonalisation and no
    size cap.

    Args:
        n_sites: Chain length ``L``. Must be even for the closed form to apply.
        coupling: The Ising coupling ``J``, held fixed.
        points: How many field values to sample.

    Returns:
        The ratios ``g = h/J``, and one list per momentum holding
        :math:`\epsilon(k)` along the sweep. Empty lists when the closed form does
        not cover this chain, which the caller shows as an explanation rather than
        as an empty plot.
    """
    ratios = [index * MAX_RATIO / (points - 1) for index in range(points)]
    columns: list[list[float]] = []
    for ratio in ratios:
        spec = TFIMSpec(n_sites=n_sites, coupling=coupling, field=ratio * coupling)
        if exact.unsupported_reason(spec) is not None:
            return [], []
        energies = exact.excitation_energies(spec)
        if not columns:
            columns = [[] for _ in energies]
        for column, energy in zip(columns, energies, strict=True):
            column.append(float(energy))
    return ratios, columns


@st.cache_data(show_spinner=False)
def energy_response(
    n_sites: int, coupling: float, points: int
) -> tuple[list[float], list[float], list[float], list[float]]:
    r"""The energy density and its first two field derivatives, in closed form.

    Why this plot earns its space: the energy itself is featureless, its first
    derivative rises through the transition, and its second dips sharply at it. The
    transition is invisible in the quantity and unmistakable in its curvature, which
    is the whole reason a phase transition is defined through derivatives of a free
    energy rather than through the energy.

    Every column is analytic. The first derivative is
    :func:`src.physics.exact.transverse_magnetisation`, which *is*
    :math:`-\partial (E_0/L)/\partial h` by Hellmann-Feynman, and the second is
    :func:`src.physics.exact.energy_density_curvature`. Nothing here is a finite
    difference, which matters because numerical differentiation is where a plot
    acquires wiggles the physics does not have.

    Args:
        n_sites: Chain length ``L``. Must be even for the closed form to apply.
        coupling: The Ising coupling ``J``, held fixed.
        points: How many field values to sample.

    Returns:
        The ratios, then ``E_0/L``, then :math:`\partial (E_0/L)/\partial h`, then
        :math:`\partial^2 (E_0/L)/\partial h^2`. All empty when the closed form does
        not cover this chain.
    """
    ratios = [index * MAX_RATIO / (points - 1) for index in range(points)]
    density: list[float] = []
    slope: list[float] = []
    curvature: list[float] = []
    for ratio in ratios:
        spec = TFIMSpec(n_sites=n_sites, coupling=coupling, field=ratio * coupling)
        if exact.unsupported_reason(spec) is not None:
            return [], [], [], []
        density.append(exact.energy_density(spec))
        # Minus the magnetisation, and that identity is the check on this curve: the
        # same number is plotted as an observable elsewhere on this page.
        slope.append(-exact.transverse_magnetisation(spec))
        curvature.append(exact.energy_density_curvature(spec))
    return ratios, density, slope, curvature


@st.cache_data(show_spinner=False)
def scaling(coupling: float, field: float) -> tuple[list[float], list[float], float]:
    """Energy density against inverse chain length, with the exact limit.

    Args:
        coupling: The Ising coupling ``J``.
        field: The transverse field ``h``.

    Returns:
        The inverse lengths, the energy densities, and the exact infinite-chain
        value the sequence should be converging to. The finite points come from
        the closed-form solution, which the panel above shows agreeing with exact
        diagonalisation to floating-point noise at the selected length.
    """
    densities = [
        exact.energy_density(TFIMSpec(n_sites=sites, coupling=coupling, field=field))
        for sites in SCALING_SITES
    ]
    return (
        [1.0 / sites for sites in SCALING_SITES],
        densities,
        exact.energy_density_thermodynamic(coupling, field),
    )


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def phase_label(ratio: float) -> str:
    """Name the side of the critical point a ratio sits on.

    Args:
        ratio: The control parameter ``g = h / J``.

    Returns:
        A short label. Hedged deliberately: a finite chain has a smooth
        crossover, not a phase transition, and only the infinite chain has a
        genuine critical point at ``g = 1``.
    """
    if math.isclose(ratio, 1.0, abs_tol=1e-9):
        return "critical"
    return "ordered side" if ratio < 1.0 else "disordered side"


def render_verdict(check: CrossCheck) -> None:
    """Show whether independent methods corroborated the answer.

    Args:
        check: The completed cross-check.
    """
    spread = check.max_disagreement
    if check.is_corroborated and spread is not None:
        st.success(
            f"**Verified.** {len(check.results)} methods that share no algebra agree "
            f"to **{spread:.1e}**.",
            icon="✅",
        )
    elif spread is None:
        st.warning(
            "**Unverified.** Only one method applies to this chain, so nothing "
            "independent confirms the number. It is not necessarily wrong — it is "
            "unchecked.",
            icon="⚠️",
        )
    else:
        st.error(
            f"**Contradiction.** The methods disagree by **{spread:.1e}**, far beyond "
            "floating-point error. Do not use this number.",
            icon="🚨",
        )


def found_how(passage: Passage) -> str:
    """Say how a passage was found, and how strongly.

    The vector score is reported as a percentage and the keyword score is not,
    because only one of them is a proportion. A cosine relevance is already in
    ``[0, 1]`` and has been measured against this index -- squarely on-topic is
    0.53 to 0.66, an off-topic question tops out at 0.11 -- so a percentage means
    something to a reader. A BM25 score is unbounded and corpus-relative, so a
    percentage of it would be a number with no scale behind it.

    Args:
        passage: A kept passage.

    Returns:
        One phrase naming which half of the search found it. A passage the
        keyword half found alone has no vector similarity to report, and saying
        "0%" would read as a broken score rather than as a different route in.
    """
    if passage.lexical_score and passage.score:
        return f"{passage.score:.0%} similarity, and a keyword match"
    if passage.lexical_score:
        return "found by keyword match, not by similarity"
    return f"{passage.score:.0%} similarity"


def render_sources(answer: graph.Answer) -> None:
    """List the notes an answer rests on, with the shelf each came from.

    The shelf is on screen because it is a decision, not a detail: the router
    chose which knowledge base to search, and an answer drawn from the quantum
    computing notes is a different kind of claim from one drawn from the physics
    notes. Reading the citations alone, that difference is invisible.

    Args:
        answer: The finished run, already known to carry citations.
    """
    retrieval = answer.retrieval
    if retrieval is None or not retrieval.passages:
        # A citation with no passage behind it: nothing to qualify, so it is shown
        # as it always was rather than annotated with zeroes.
        for citation in answer.citations:
            st.markdown(f"- {citation}")
        return
    for passage in retrieval.passages:
        st.markdown(f"- {passage.citation}")
        shelf = get_shelf(passage.shelf)
        where = shelf.title if shelf is not None else (passage.shelf or "unknown shelf")
        st.caption(f"{where} · {found_how(passage)}")


def render_code(written: Draft) -> None:
    """Show a drafted program, as code rather than as prose.

    The one part of an answer the reader is meant to take away and run, so it gets
    Streamlit's code block -- syntax highlighting, no LaTeX interpretation, and a
    copy button. Folding it into :attr:`~src.agent.graph.Answer.text` would put a
    fenced block inside markdown that is also carrying ``$..$`` maths, and the two
    do not survive each other.

    Nothing here says the code was not executed. That belongs with the other
    limitations, in the caveats, where a reader looking for what to distrust will
    find all of them together -- see :func:`src.agent.graph.caveats_for`.

    Args:
        written: The draft. Callers check :attr:`~src.agent.drafting.Draft.written`
            first, so an empty one draws an empty heading rather than nothing.
    """
    st.markdown("**Code for this chain** — written for you, not run here")
    if written.preamble:
        st.markdown(written.preamble)
    st.code(written.code, language=written.language)


def render_answer(answer: graph.Answer) -> None:
    """Show one answer in three layers.

    Args:
        answer: What the agent returned.
    """
    label, colour = STATUS_LABELS[answer.status]
    st.markdown(
        f'<span class="badge" style="background:{colour}22;color:{colour}">{label}</span>',
        unsafe_allow_html=True,
    )
    st.markdown(answer.text)

    if answer.sweep is not None:
        render_curves(answer.sweep)

    if answer.code.written:
        render_code(answer.code)

    for caveat in answer.caveats:
        st.warning(caveat)

    if answer.verification is not None:
        render_verdict(answer.verification)
    if answer.citations:
        st.markdown("**Sources used**")
        render_sources(answer)
    if answer.papers:
        st.markdown("**Further reading from arXiv** — fetched live, not checked")
        for paper in answer.papers:
            link = f"[{paper.title}]({paper.url})" if paper.url else paper.title
            st.markdown(f"- {link} — {', '.join(paper.authors)}")

    # Grouped under one heading and kept below the corpus citations, because the
    # order on the page is the order of how much each source is worth.
    background = [run for run in answer.tools if run.wiki is not None and run.wiki.found]
    web = [run for run in answer.tools if run.web is not None and run.web.found]
    remote = [run for run in answer.tools if run.remote is not None and run.remote.ok]
    if background or web or remote:
        st.markdown("**Outside sources** — background only, none of it verified here")
        for run in background:
            article = run.wiki.article if run.wiki is not None else None
            if article is not None:
                link = f"[{article.title}]({article.url})" if article.url else article.title
                depth = " (whole article)" if article.depth == "deep" else ""
                st.markdown(f"- Wikipedia: {link}{depth}")
        for run in web:
            for hit in run.web.results if run.web is not None else ():
                st.markdown(f"- Web: [{hit.title}]({hit.url})")
        for run in remote:
            call = run.remote
            if call is not None:
                st.markdown(f"- MCP server: `{call.tool}`")

    render_trajectory(answer)

    with st.expander("How this answer was produced"):
        st.caption(
            "Quoted, not composed: every line was fixed before the wording was "
            "written, so it cannot be a rationalisation of a different run."
        )
        st.code(answer.justification(), language="text")

    render_cost(answer.usage)


# What each action did, in the words the loop uses. Read from the trajectory
# rather than from the node list, so an action that was chosen and then produced
# nothing still appears -- "it looked and found nothing" is the interesting case.
ACTION_LABELS = {
    "compute": "Solved the chain and cross-checked it",
    "retrieve": "Searched the knowledge base",
    "consult": "Reached for the outside sources",
    "implement": "Wrote code for the chain",
    "finish": "Answered from what it had",
}


def render_trajectory(answer: graph.Answer) -> None:
    """Show the loop's decisions, in order.

    The difference between an agent and a pipeline is that the steps were chosen,
    and a reader cannot take that on trust from a diagram in which every box is
    always drawn. So each action is shown with the reason it was chosen and who
    chose it -- the model, or the deterministic policy that runs without a key.

    Args:
        answer: The finished run. A question that needed no work done has an empty
            trajectory, and nothing is drawn for it rather than an empty heading.
    """
    if not answer.steps:
        return
    taken = [step for step in answer.steps if step.action != "finish"]
    with st.expander(f"Plan → act → observe · {len(taken)} of {MAX_STEPS} steps used"):
        st.caption(
            "Each step was decided from what the run had established at that "
            "moment, never from the words in the question. The knowledge base is "
            "always searched before the outside sources are."
        )
        for number, step in enumerate(answer.steps, start=1):
            label = ACTION_LABELS.get(step.action, step.action)
            # A second search is only worth making if it looks for something the
            # first one missed, so the trajectory names what it went after: that is
            # the visible difference between a retry and an observation.
            aim = f" for “{step.focus}”" if step.focus else ""
            st.markdown(f"**{number}. {label}{aim}** — {step.reason} _({step.decided_by})_")


def render_followups(answer: graph.Answer, position: int) -> None:
    """Offer what to ask next, as buttons that actually ask it.

    Buttons rather than a printed list, for one reason: a suggestion the reader has
    to retype is a suggestion most readers will not take. They reuse the same
    ``starter`` key the sample questions use, so a click and a typed question reach
    the agent by the same path -- including the guard, which is what makes it safe
    to put model-written text on a button.

    Args:
        answer: The answer the suggestions belong to.
        position: Index in the session history, so two answers on screen cannot
            collide on a widget key.
    """
    followups = answer.followups
    if not followups.any:
        return
    st.markdown("**What to ask next**")
    for index, suggestion in enumerate(followups.suggestions):
        st.button(
            suggestion.question,
            key=f"followup-{position}-{index}",
            help=suggestion.why,
            on_click=lambda text=suggestion.question: st.session_state.__setitem__("starter", text),
        )
    st.caption(
        f"Proposed from what this run established, by the {followups.proposed_by}. "
        "Each one was screened as if you had typed it."
    )


class SessionMemory:
    """A memory that lives in Streamlit's session state and dies with the tab.

    The store an unidentified visitor gets. It satisfies
    :class:`src.agent.memory.Memory` in full, so the graph cannot tell the
    difference -- follow-up questions, ratings and the learned register all work
    exactly as they do for a signed-in visitor, for as long as the session lasts.

    Kept here rather than in :mod:`src.agent.memory` because a *session* is a
    Streamlit idea. The agent knows about threads and stores; which of those is
    backed by a file is the interface's business, and putting ``st.session_state``
    inside the agent would make the agent untestable without a browser.
    """

    KEY = "session_memory"

    def _records(self) -> list[memory.Record]:
        """The list backing this store, created on first use."""
        held: list[memory.Record] = st.session_state.setdefault(self.KEY, [])
        return held

    def write(self, record: memory.Record) -> None:
        """Append one record to the session."""
        self._records().append(record)

    def read(self, thread: str | None = None) -> tuple[memory.Record, ...]:
        """Return the session's records, oldest first, optionally for one thread."""
        return tuple(
            record for record in self._records() if thread is None or record.thread == thread
        )

    def forget(self, thread: str) -> int:
        """Drop every record for one thread, returning how many went."""
        return self.forget_threads((thread,))

    def forget_threads(self, threads: Iterable[str]) -> int:
        """Drop every record for each of several threads, returning how many went.

        The session's answer to :meth:`src.agent.memory.FileMemory.forget_threads`.
        There is no file here and so no atomicity to preserve, but the method is
        part of the protocol and a store that made the caller loop would make
        "delete all my chats" mean something different depending on which store
        the visitor happened to be given.
        """
        wanted = set(threads)
        if not wanted:
            return 0
        held = self._records()
        keeping = [record for record in held if record.thread not in wanted]
        removed = len(held) - len(keeping)
        st.session_state[self.KEY] = keeping
        return removed


CHAT_NUMBER = "chat_number"


def current_thread() -> str:
    """Which conversation this session is filed under.

    Returns:
        ``"<user>/<conversation>"``: the key from
        :func:`src.ui.identity.session_thread` -- a digest of a signed-in name, or
        a token unique to this browser session -- and which chat of theirs this is.
        Never empty and never shared, which is what makes it safe to key a memory
        on.

        The conversation number is what :func:`start_new_chat` moves. Recall follows
        the whole key and the learned register follows the user half, so a new chat
        starts the agent's memory of the conversation over without discarding what
        the ratings taught it.
    """
    number = st.session_state.get(CHAT_NUMBER, 1)
    return f"{identity.session_thread()}{memory.THREAD_SEPARATOR}{number}"


def next_chat_number() -> int:
    """The number a conversation started now should be filed under.

    Returns:
        One past the highest number this user has used, counting both the chat on
        screen and every conversation still in the store.

        Not simply *this one plus one*, because :func:`reopen_chat` moves the
        number backwards and that arithmetic then walks forwards over conversations
        that already exist: reopen chat 2 of four, press *New chat*, and the next
        question is filed under chat 3 -- a conversation with its own history,
        which the agent recalls into a page the button had just promised was empty
        and then writes the answer into. Two conversations merge and neither can be
        told apart afterwards.

        The store is consulted rather than a high-water mark in session state,
        because the store is what :func:`past_chats` lists from and the two have to
        agree about which numbers are taken.
    """
    used = {
        chat.number
        for chat in memory.conversations(memory.base_of(current_thread()), session_store())
    }
    return max({int(st.session_state.get(CHAT_NUMBER, 1)), *used}) + 1


def start_new_chat() -> None:
    """Clear the thread on screen and start a fresh conversation.

    Three things move together, and leaving any of them behind produces a "new
    chat" that is not one: the answers being rendered, the record of which of them
    have been rated, and the conversation half of the memory key. Without the last,
    the agent would open a blank page and then answer the first question as though
    it had already had four.

    The previous conversation is not deleted. It is simply no longer recalled --
    which is what ending a conversation means, and it leaves the earlier turns
    available to :meth:`src.agent.memory.Memory.forget` if the user does want them
    gone.
    """
    st.session_state["history"] = []
    st.session_state["ratings"] = {}
    st.session_state[CHAT_NUMBER] = next_chat_number()


def render_new_chat() -> None:
    """Offer to start a fresh conversation.

    On the chat page rather than in the sidebar, next to the thread it clears. The
    reason it exists is length: every answer stays on the page, which is what makes
    two answers comparable, and after a dozen questions that is a very long page.
    """
    number = int(st.session_state.get(CHAT_NUMBER, 1))
    # Never disabled, though an empty thread has nothing to clear. This is drawn
    # above the panel that answers the question, so on the run where a question is
    # answered the history it could inspect is still one short -- a button greyed out
    # by that ordering would refuse the click that most obviously should work.
    st.button(
        "🆕 New chat",
        on_click=start_new_chat,
        help=(
            f"Chat {number} of this session. Starting a new one clears the page and "
            "stops the agent recalling these turns; what your ratings taught it is kept, "
            "and this chat stays listed under **Past chats** in the sidebar."
        ),
    )


# Most conversations listed, and turns listed within one. The sidebar is a column,
# not a page: a listing that grew without bound would push the memory and cost
# panels off the bottom of it. Newest first, so what is cut is the oldest.
MAX_LISTED_CHATS = 8
MAX_LISTED_TURNS = 4


def past_chats() -> tuple[memory.Conversation, ...]:
    """The conversations this session can go back to, newest first.

    Returns:
        Every conversation of this user's *except* the one on screen. Excluded
        because the current chat is not something to return to -- and because
        including it would make the list change under a reader mid-conversation,
        as the chat they are in climbs to the top of a list of chats they are not.
    """
    thread = current_thread()
    return tuple(
        chat
        for chat in memory.conversations(memory.base_of(thread), session_store())
        if chat.thread != thread
    )


def reopen_chat(number: int) -> None:
    """Go back to an earlier conversation.

    The inverse of :func:`start_new_chat`, and it moves the same three things: the
    answers on screen, the record of which were rated, and the conversation half of
    the memory key.

    What comes back is the agent's memory of the conversation, not the page. The
    rendered answers were never stored -- they hold solver functions, passages and a
    verification record, none of which is serialisable -- so the turns return as the
    recap :func:`src.agent.memory.recall` reads, and the next question is answered
    in their context. Redrawing the old replies would mean either storing prose the
    agent would then be quoting unverified, or answering every past question again.

    Args:
        number: Which conversation of this user's to return to.
    """
    st.session_state["history"] = []
    st.session_state["ratings"] = {}
    st.session_state[CHAT_NUMBER] = number


DELETE_ARMED = "chat_delete_armed"
"""Session-state key holding the chat numbers whose delete has been armed.

A set keyed by conversation rather than one shared flag, so arming the delete on
one row does not arm the row below it -- which is what a single flag does on the
rerun that follows the press, and it would arm whichever row the reader is most
likely to aim at next.
"""

DELETE_ALL_ARMED = "chat_delete_all_armed"
"""Session-state key saying the *delete all* button is waiting to be confirmed."""


def render_past_chats() -> None:
    """List the earlier conversations, and offer to reopen or delete one.

    In the sidebar beside 🧠 Memory because the two read the same store at the same
    scope -- everything this user has -- and because a conversation is not the
    property of any one page. The button that ends a conversation is in the top
    bar; the list of the ones that ended is here, which is where a reader who has
    just cleared the screen goes looking for it.

    **Ending a chat and deleting one are different things, and the panel has to say
    so.** *New chat* stops a conversation being recalled and keeps every word of it
    here; the 🗑 on a row erases it from the store for good. A panel that offered
    only the first left a reader with no way to remove a question they wished they
    had not asked, which for a store that outlives the session is not a missing
    convenience but a missing promise.

    The conversation on screen is never in this list and so is never deleted from
    it -- **Forget this conversation** in the 🧠 Memory panel is the control for
    that one, and keeping them apart is what lets *Delete all past chats* be
    pressed without a reader losing the thread they are in the middle of.
    """
    chats = past_chats()
    with st.expander(f"🕘 Past chats ({len(chats)})", expanded=False):
        if not chats:
            st.caption(
                "None yet. Starting a new chat does not delete this one — it is kept "
                "here, and only stops being recalled."
            )
            return
        st.caption(retention_caption())
        st.caption(
            "**Reopen** recalls one again. 🗑 deletes it from the store for good, and "
            "asks once more first. Neither touches the conversation on screen — "
            "**Forget this conversation** under 🧠 Memory is the control for that one."
        )
        for chat in chats[:MAX_LISTED_CHATS]:
            asked = len(chat.turns)
            day = chat.when[:10]
            st.markdown(f"**Chat {chat.number}** — {asked} question{'s' if asked != 1 else ''}")
            if day:
                st.caption(day)
            for turn in chat.turns[:MAX_LISTED_TURNS]:
                # The question, not the reply: it is what the reader typed and so
                # what they recognise the conversation by, and the stored reply is
                # clipped to a length that reads as a broken sentence.
                st.caption(f"· {turn.question}")
            if asked > MAX_LISTED_TURNS:
                st.caption(f"· … {asked - MAX_LISTED_TURNS} more")
            # Wide beside narrow: the label carries the meaning and the bin is a
            # glyph, so giving them equal columns would put a destructive control
            # under half the reader's pointer.
            reopen_column, delete_column = st.columns([4, 1])
            reopen_column.button(
                "Reopen",
                key=f"reopen-{chat.number}",
                width="stretch",
                help=(
                    "Recall this conversation again. The page starts empty — the "
                    "answers themselves are not stored — but the agent knows what "
                    "was asked here and answers your next question in that context."
                ),
                on_click=reopen_chat,
                args=(chat.number,),
            )
            _delete_chat_button(chat, delete_column)
        if len(chats) > MAX_LISTED_CHATS:
            st.caption(
                f"{len(chats) - MAX_LISTED_CHATS} older chat(s) not listed — nothing "
                "has been deleted, and **Delete all past chats** reaches them too."
            )
        _delete_all_past_chats_button(chats)


def _delete_chat_button(chat: memory.Conversation, column: Any) -> None:
    """Offer to delete one past conversation, in two presses.

    Two presses rather than one, and this is the one control in the sidebar worth
    that friction: every other one here is undone by moving it back, and this one
    rewrites the store. It is also the control whose reward for a misclick is
    losing the thing the reader opened the panel to find.

    Args:
        chat: The conversation the button belongs to.
        column: Where to draw it, so the caller keeps control of the layout.

    The rerun after each press is not optional. A button returns ``True`` during
    the rerun its own click caused, by which point this row has already drawn
    itself in the old state -- so without asking for another pass, arming a delete
    would leave 🗑 on screen and deleting one would leave the row listed.
    """
    armed: set[int] = st.session_state.setdefault(DELETE_ARMED, set())
    if chat.number in armed:
        if column.button(
            "✓",
            key=f"confirm-delete-{chat.number}",
            width="stretch",
            help="Press to delete this conversation permanently.",
        ):
            removed = session_store().forget(chat.thread)
            armed.discard(chat.number)
            st.toast(f"Chat {chat.number} deleted — {removed} record(s) removed.")
            st.rerun()
        return
    if column.button(
        "🗑",
        key=f"delete-{chat.number}",
        width="stretch",
        help="Delete this conversation. Asks once more before it does.",
    ):
        armed.add(chat.number)
        st.rerun()


def _delete_all_past_chats_button(chats: tuple[memory.Conversation, ...]) -> None:
    """Offer to clear every past conversation at once, in two presses.

    Below the rows rather than above them: a control that empties the list should
    not sit where a reader aims for the newest conversation in it.

    It counts and deletes **every** past conversation, including any beyond
    :data:`MAX_LISTED_CHATS` that the panel did not draw, because a button that
    quietly spared the ones off-screen would leave a reader believing they had
    cleared a history they had not. The conversation on screen is not among them --
    it is not a past chat, it is the current one.

    Args:
        chats: Every past conversation, listed or not.

    Withheld below two, where it would duplicate the single row's own 🗑 and give
    a destructive action two buttons instead of one.
    """
    total = len(chats)
    if total < 2:
        return
    if st.session_state.get(DELETE_ALL_ARMED):
        if st.button(
            f"✓ Yes, delete all {total}",
            key="confirm-delete-all-chats",
            type="primary",
            width="stretch",
            help="The conversation on screen is not one of them and is left alone.",
        ):
            # One call rather than a loop, so a file store rewrites once and this
            # either happened or did not. See `FileMemory.forget_threads`.
            removed = session_store().forget_threads(chat.thread for chat in chats)
            st.session_state.pop(DELETE_ALL_ARMED, None)
            st.session_state.pop(DELETE_ARMED, None)
            st.toast(f"{total} chat(s) deleted — {removed} record(s) removed.")
            st.rerun()
        if st.button("Cancel", key="cancel-delete-all-chats", width="stretch"):
            st.session_state.pop(DELETE_ALL_ARMED, None)
            st.rerun()
        return
    if st.button(
        "🗑 Delete all past chats",
        key="delete-all-chats",
        width="stretch",
        help=(
            "Deletes every past conversation, including any not listed above. "
            "Asks once more before it does."
        ),
    ):
        st.session_state[DELETE_ALL_ARMED] = True
        st.rerun()


def signed_in() -> bool:
    """Whether this session belongs to a verified name.

    One reader for a fact two panels state out loud -- whether what is on screen
    outlives the tab. Said differently in two places, it would eventually be said
    wrongly in one of them.
    """
    who = st.session_state.get("identity")
    return isinstance(who, identity.Identity) and who.verified


def retention_caption() -> str:
    """How long what is on screen will last, in one sentence.

    Returns:
        The line 🕘 Past chats and 🧠 Memory both print. It was written out in
        both, which is one fact in two places and therefore two places for it to
        drift -- and this particular fact is a promise about somebody's data, so
        the two panels disagreeing about it would not be a cosmetic bug.

        It reads :func:`signed_in` rather than taking an argument because the
        caller has no business deciding the answer; there is one session and it is
        either signed in or it is not.
    """
    if signed_in():
        return "Kept across sessions, because you are signed in."
    return "Kept for this session only — nothing is written to disk."


def session_settings() -> Settings | None:
    """The configuration this session's questions should be answered with.

    The one reader of :func:`src.ui.identity.own_key`, and deliberately the only
    one: a credential that several modules reach for directly is a credential that
    ends up somewhere it was promised not to go. Everything else asks here and gets
    settings back, never a key.

    Returns:
        ``None`` when the host's own credential should be used, which is the
        ordinary case and is exactly what :func:`src.agent.graph.ask` reads as
        "resolve the process settings yourself". Otherwise the process settings
        with the visitor's key substituted -- so the model, the throttle, the
        corpus and every other dial stay as the deployment configured them and the
        only thing that changes is who is billed.

        The fallback matters more than it looks. :func:`src.settings.get_settings`
        raises when no credential is configured at all, which is the case on a
        checkout with no ``.env`` -- and that is precisely the session most likely
        to paste a key. Building the settings from the pasted key instead means the
        key *works* there, rather than being accepted and then dropped by the
        exception raised while trying to copy the configuration it was meant to
        replace.
    """
    key = identity.own_key()
    if key is None:
        return None
    secret = SecretStr(key)
    try:
        return get_settings().model_copy(update={"openrouter_api_key": secret})
    except Exception:
        return Settings(openrouter_api_key=secret)


def session_store() -> memory.Memory:
    """The memory this session should use.

    Returns:
        The configured file store for a signed-in visitor, whose history is meant
        to outlive the session, and a :class:`SessionMemory` for everyone else.

        The pairing is the privacy rule, expressed as a choice of object rather
        than as a promise in a docstring: without a login there is no key that
        will exist tomorrow, so writing to disk would leave records nobody can
        ever reach, inspect or delete. Memory that dies with the tab is the honest
        implementation of "kept for this session only".
    """
    if signed_in():
        return memory.open_memory()
    return SessionMemory()


def remembered_turn(answer: graph.Answer) -> memory.Turn | None:
    """Find the stored record for an answer, if there is one.

    Rating goes through the store rather than through the answer on screen,
    because only the store knows the register the answer was actually written
    at -- and because it makes the rule visible: you can rate what was
    remembered, and nothing else.

    Args:
        answer: The answer being shown.

    Returns:
        The turn, or ``None`` when the run was one that is never stored -- a
        blocked question, or a question asked before this session had a thread.

    The store is read once per rendered answer. That is a session list or a read of
    a length-capped file, measured in fractions of a millisecond; the alternative --
    caching it in session state -- would show a stale profile immediately after a
    rating, which is the one moment the user is watching.
    """
    thread = current_thread()
    wanted = memory.turn_id(thread, answer.question)
    for record in reversed(session_store().read(thread)):
        if isinstance(record, memory.Turn) and record.id == wanted:
            return record
    return None


def render_feedback(answer: graph.Answer, position: int) -> None:
    """Offer the two buttons the agent learns from.

    Two buttons and no free-text box in the thread, because a rating has to be
    cheap enough to give. What it buys is concrete and stated on the spot: enough
    agreeing ratings and the register changes, which the user can see in the
    sidebar and override in the knob.

    Args:
        answer: The answer being rated.
        position: Where it sits in the thread, used only to keep widget keys
            unique -- two identical questions in one session would otherwise
            collide on the same key and Streamlit would refuse to draw the
            second.
    """
    turn = remembered_turn(answer)
    if turn is None:
        return
    given: dict[str, bool] = st.session_state.setdefault("ratings", {})
    key = f"{turn.id}-{position}"
    if key in given:
        st.caption("Thanks — noted. It takes a couple of agreeing ratings to change anything.")
        return

    def record(liked: bool) -> None:
        memory.rate(turn=turn, liked=liked, memory=session_store())
        given[key] = liked

    helpful, unhelpful, _ = st.columns([1, 1, 6])
    helpful.button(
        "👍",
        key=f"up-{key}",
        help="Written at about the right level.",
        on_click=record,
        args=(True,),
    )
    unhelpful.button(
        "👎", key=f"down-{key}", help="Wrong level for me.", on_click=record, args=(False,)
    )


def remembered_preference() -> memory.Preference:
    """Read what this thread has taught the agent.

    Returns:
        The profile. An empty one on the first turn of a conversation, which is
        also what an untouched store looks like -- and both correctly learn
        nothing.
    """
    return memory.recall(current_thread(), session_store()).preference


def render_memory(preference: memory.Preference) -> None:
    """Show what the agent has remembered, and offer to forget it.

    An agent that adjusts to feedback has to be able to say what it adjusted and
    why, or the adjustment is indistinguishable from the model having a bad day.
    :meth:`src.agent.memory.Preference.explain` is that sentence, and it is
    rendered here rather than logged.

    Forgetting is offered in the same place for the same reason: a memory a user
    cannot inspect and cannot clear is a memory they did not agree to.

    **Forget this conversation** is scoped to the one on screen, and that is the
    division of labour with :func:`render_past_chats`: this button is the only way
    to erase the current thread, and the 🗑 controls there are the only way to
    erase the earlier ones. Neither reaches into the other's half, so a reader
    clearing their history cannot lose the conversation they are in the middle of
    by accident, and a reader deleting the thread on screen is not also silently
    deleting last week's.

    Args:
        preference: What this thread has taught the agent.
    """
    with st.expander("🧠 Memory", expanded=False):
        st.caption(preference.explain())
        if preference.chains:
            st.caption("Chains solved here: " + "; ".join(preference.chains))
        st.caption(retention_caption())
        if st.button("Forget this conversation", key="forget"):
            removed = session_store().forget(current_thread())
            st.session_state["ratings"] = {}
            st.success(f"Forgotten — {removed} record{'s' if removed != 1 else ''} removed.")


# What each node is doing, in the words the reader needs while waiting for it.
# Keyed by node name, so a node with no entry shows its own name rather than
# nothing -- a new step in the graph appears here as itself and can then be named.
NODE_LABELS = {
    "screen": "Screening the question",
    "recall": "Recalling this conversation",
    "route": "Deciding what kind of question it is",
    "decide": "Choosing the next action",
    "plan": "Choosing a method",
    "solve": "Solving and cross-checking",
    "search": "Searching the notes",
    "consult": "Offering the tools",
    "draft": "Writing code",
    "compose": "Writing the answer",
    "suggest": "Proposing what to ask next",
    "remember": "Storing the turn",
}


def ask_agent(question: str, setting: Setting, *, approved: bool) -> graph.Answer:
    """Run the agent and remember the answer for the rest of the session.

    The steps are shown as they happen, and that is a fix for a real complaint
    rather than decoration. An answer is about seven sequential calls to a provider
    -- measured at 2.3s to screen, 3.6s to route, 3.0s to grade the passages and
    5.3s to compose -- so twenty-odd seconds is the honest cost of the pipeline and
    not something the interface can remove. What it can remove is the blankness: a
    spinner reading "screening, recalling, routing, solving, checking" is a promise
    that all five will happen, which is exactly the claim this project spends a page
    denying. The live list is the trajectory instead, and a run that stalls now says
    where.

    Args:
        question: What the user asked.
        setting: The knob the question was asked with.
        approved: Whether an expensive run has already been authorised.

    Returns:
        The answer, which is also appended to the session history.
    """
    # Timed from here, and the elapsed seconds are shown per step rather than
    # totalled at the end. Almost the whole wait is a queue of sequential calls to
    # a provider -- seven of them for a typical question -- so the honest thing the
    # interface can do about it is say which one is being waited on and for how
    # long. A number moving on screen is also the difference between "slow" and
    # "hung", and the two were indistinguishable while nothing counted.
    started = time.monotonic()

    with st.status("Working on it…", expanded=True) as box:

        def report(node: str) -> None:
            """Name the step that just finished and how long the run has taken."""
            label = NODE_LABELS.get(node, node)
            elapsed = time.monotonic() - started
            box.update(label=f"{label}… ({elapsed:.0f}s)")
            st.markdown(f"- {label} · {elapsed:.1f}s")

        answer = graph.ask(
            question,
            setting=setting,
            approved=approved,
            thread=current_thread(),
            # The store is passed explicitly, and it must be: without it the graph
            # opens the *configured* one while every panel here reads
            # :func:`session_store`. For a signed-in visitor those are the same file
            # and the omission was invisible; for everyone else they are not, and the
            # consequences were both halves of the same bug. Nothing appeared to be
            # remembered -- no rating buttons, no learned register, "Past chats (0)"
            # after a dozen questions -- while the turns were in fact being written to
            # disk under a session token, which is exactly what this application
            # promises not to do without a login.
            memory=session_store(),
            # None unless this visitor pasted their own key, in which case it is
            # the deployment's configuration with that key substituted. Passed
            # here rather than resolved inside the graph for the same reason the
            # store is: the graph would otherwise reach for the *process*
            # credential while the top bar says the answer is running on theirs.
            settings=session_settings(),
            progress=report,
        )
        took = time.monotonic() - started
        st.caption(
            f"{took:.0f} seconds, nearly all of it spent waiting on the model — the "
            "steps above run one after another because each one reads the last one's "
            "result. **Searches per question** and **Suggest what to ask next** in the "
            "settings knob are each worth a call."
        )
        box.update(
            label=f"{STATUS_LABELS[answer.status][0]} · {took:.0f}s",
            state="complete",
            expanded=False,
        )
    history: list[graph.Answer] = st.session_state.setdefault("history", [])
    history.append(answer)
    return answer


def ask_panel(setting: Setting, greeting: str = "") -> None:
    """Render the conversation, the box to add to it, and the cost gate.

    A thread rather than a single reply, because the answers are meant to be
    compared: a reader who asks about six spins and then about ten wants both on
    screen. Nothing is re-answered on a rerun -- Streamlit re-executes this script
    whenever any widget moves, and an agent that answered again each time would
    charge for a slider drag -- so the history is replayed from what was already
    computed.

    Args:
        setting: The knob the question will be asked with.
        greeting: Who to address in the opening message, or ``""`` for nobody.
    """
    # Called first so the box is pinned to the bottom of the viewport. What it
    # returns is only read, not acted on, until the thread below has been drawn.
    typed = st.chat_input("Ask about the transverse-field Ising model")

    # The starter and approval buttons set a key and let Streamlit rerun. Which of
    # the three arrived is decided here; the asking happens further down.
    approved_question = st.session_state.pop("approved_question", None)
    starter = st.session_state.pop("starter", None)
    if approved_question is not None:
        pending, approved = str(approved_question), True
    elif starter is not None:
        pending, approved = str(starter), False
    elif typed and typed.strip():
        pending, approved = typed.strip(), False
    else:
        pending, approved = None, False

    history: list[graph.Answer] = st.session_state.get("history", [])
    if approved and history and history[-1].status == "approval_needed":
        # The gate's own turn is replaced, not added to. It carries this exact
        # question and the reply "shall I?", so leaving it in place printed the
        # question twice with a dead end between the copies -- which reads as the
        # interface having lost track rather than as a cost having been approved.
        # The approved answer says in its own trajectory what it went on to do.
        history.pop()

    if not history and pending is None:
        opening = f"Hello {greeting}. " if greeting else ""
        with st.chat_message("assistant"):
            st.markdown(
                f"{opening}Ask in plain language. I decide for myself whether to compute, "
                "to read the notes, or to say I cannot answer — and I tell you which."
            )
            st.caption(
                "Every number I show has been computed twice by methods that share no "
                "algebra. If they disagree, you see that instead of the number."
            )
        return

    # Replayed before the new question is asked, and that order is the whole point.
    # The agent used to run up here, which put its live progress box above every
    # turn already on screen: by the third question the only sign that anything was
    # happening sat several screens above a reader who had just typed at the bottom,
    # and a pipeline that honestly takes twenty seconds looked like a frozen page.
    # Drawing the thread first costs nothing -- it is replayed from answers already
    # computed -- and puts the steps under the question that caused them.
    for position, past in enumerate(tuple(history)):
        with st.chat_message("user"):
            st.markdown(past.question)
        with st.chat_message("assistant"):
            render_answer(past)
            render_feedback(past, position)

    if pending is not None:
        # Echoed before the run rather than after it, so the question is on screen
        # for the whole wait instead of appearing with the answer. Read now, because
        # ask_agent appends to this same list.
        position = len(history)
        with st.chat_message("user"):
            st.markdown(pending)
        with st.chat_message("assistant"):
            answer = ask_agent(pending, setting, approved=approved)
            render_answer(answer)
            render_feedback(answer, position)

    # Re-read rather than reused: on the very first question of a session
    # ``get`` above returned a fresh empty list that was never stored, while
    # ask_agent created and appended to the session's own. The two are not the
    # same object, and the local one is still empty here.
    history = st.session_state.get("history", [])
    if not history:
        return

    latest = history[-1]
    if latest.status == "approval_needed":
        st.button(
            "Yes, go ahead",
            type="primary",
            on_click=lambda: st.session_state.__setitem__("approved_question", latest.question),
        )
    else:
        # Only under the latest answer. Repeating them beneath every past turn
        # would fill a long conversation with stale buttons, and the useful one is
        # always the one at the bottom of the page.
        render_followups(latest, len(history) - 1)


def style_axes(axes: Any) -> None:
    """Apply the plot styling shared by every figure on the page.

    Args:
        axes: The axes to style.
    """
    axes.set_facecolor("none")
    axes.tick_params(labelsize=8, colors=MUTED)
    axes.grid(color=MUTED, alpha=0.15, linewidth=0.6)
    for spine in axes.spines.values():
        spine.set_color(MUTED)
        spine.set_alpha(0.3)


def curves_figure(curve: FieldSweep, marker: float | None = None) -> Figure:
    r"""Draw the magnetisation and the energy against the field, one above the other.

    **Ground-state properties only.** Both quantities here are expectation values in
    the ground state, computed by two independent methods and agreeing to
    floating-point noise. The excitation spectrum is a different class of quantity
    with a different guarantee behind it, so it gets its own figure -- see
    :func:`spectrum_figure`. Putting them on one axis would attach one caption, and
    therefore one claim of verification, to two things that are not verified the
    same way.

    The two panels share an axis on purpose. The energy density is smooth and
    featureless -- it hides the transition -- while :math:`\langle \sigma^x \rangle` rises
    steeply through it. Reading them together is what shows that the interesting
    structure is in the *derivative* of the energy rather than in the energy itself,
    which is the whole reason a transition is hard to see in a finite system.

    Args:
        curve: The swept data.
        marker: A ratio ``g`` to mark on both panels, or ``None``.

    Returns:
        The figure, ready for :func:`streamlit.pyplot`.
    """
    ratios = [point.ratio for point in curve.points]
    figure = Figure(figsize=(7.4, 4.6), dpi=200)
    figure.patch.set_alpha(0.0)
    panels = figure.subplots(2, 1, sharex=True)
    top, bottom = panels[0], panels[1]
    turning = curve.turning_point

    top.plot(
        ratios,
        [point.magnetisation for point in curve.points],
        color=ACCENT,
        linewidth=1.8,
        zorder=2,
    )
    top.set_ylabel(r"$\langle \hat\sigma^x \rangle$", fontsize=9, color=MUTED)
    top.set_ylim(-0.03, 1.03)

    bottom.plot(
        ratios,
        [point.energy_density for point in curve.points],
        color=ACCENT,
        linewidth=1.8,
        zorder=2,
    )
    bottom.set_ylabel("$E_0 / L$", fontsize=9, color=MUTED)
    bottom.set_xlabel("$g = h / J$", fontsize=9, color=MUTED)
    bottom.set_xlim(0.0, MAX_RATIO)

    for axes in panels:
        style_axes(axes)
        axes.axvline(1.0, color=MUTED, linestyle="--", linewidth=1.0, zorder=1)
        if turning is not None:
            axes.axvline(turning, color=HIGHLIGHT, linestyle=":", linewidth=1.2, zorder=1)
        if marker is not None:
            axes.axvline(marker, color=HIGHLIGHT, alpha=0.25, linewidth=6.0, zorder=0)

    top.annotate(
        "$g = 1$: critical point\nof the infinite chain",
        xy=(1.0, 0.12),
        xytext=(8, 0),
        textcoords="offset points",
        fontsize=7,
        color=MUTED,
    )
    figure.tight_layout()
    return figure


def spectrum_figure(curve: FieldSweep, marker: float | None = None) -> Figure:
    r"""Draw the low-lying excitation energies against the field.

    A separate figure from :func:`curves_figure`, because this is a separate kind of
    physics. The magnetisation and the energy density are ground-state expectation
    values; these are **excitations** -- the energies of the states above the ground
    state -- and no amount of ground-state information contains them. They also come
    with a weaker guarantee: the ground state at each field is computed twice and
    cross-checked, while the levels above it come from diagonalisation alone.

    Each line is :math:`E_n - E_0`, so the ground state is the flat zero and the
    lowest line is the gap. What the figure shows is the finite chain's version of
    the transition: the lowest two levels are all but degenerate while the coupling
    dominates, separate as the field grows, and the gap at :math:`g = 1` is small but
    never zero -- it reaches zero only in the infinite chain.

    Args:
        curve: The swept data. Must carry a spectrum; see
            :attr:`~src.tools.sweeps.FieldSweep.levels_drawn`.
        marker: A ratio ``g`` to mark, or ``None``.

    Returns:
        The figure, ready for :func:`streamlit.pyplot`.
    """
    ratios = [point.ratio for point in curve.points]
    count = curve.levels_drawn
    figure = Figure(figsize=(7.4, 3.2), dpi=200)
    figure.patch.set_alpha(0.0)
    axes = figure.subplots()

    # Highest level palest, and drawn first so the gap sits on top of the rest. The
    # upper levels are context for the gap rather than the subject, and a plot where
    # six lines are equally emphatic is a plot with no subject at all.
    for level in reversed(range(1, count)):
        axes.plot(
            ratios,
            [point.excitations[level] for point in curve.points],
            color=ACCENT,
            linewidth=1.6 if level == 1 else 1.1,
            alpha=0.25 + 0.6 * (1.0 - (level - 1) / max(1, count - 1)),
            zorder=2,
        )
    axes.axhline(0.0, color=MUTED, linewidth=1.0, alpha=0.5, zorder=1)
    axes.set_ylabel("$E_n - E_0$", fontsize=9, color=MUTED)
    axes.set_xlabel("$g = h / J$", fontsize=9, color=MUTED)
    axes.set_xlim(0.0, MAX_RATIO)
    style_axes(axes)
    axes.axvline(1.0, color=MUTED, linestyle="--", linewidth=1.0, zorder=1)
    if marker is not None:
        axes.axvline(marker, color=HIGHLIGHT, alpha=0.25, linewidth=6.0, zorder=0)

    critical = curve.critical_gap
    if critical is not None:
        axes.annotate(
            f"gap at $g = {critical[0]:.2f}$: {critical[1]:.3f}",
            xy=(critical[0], critical[1]),
            xytext=(8, 10),
            textcoords="offset points",
            fontsize=7,
            color=MUTED,
        )
    figure.tight_layout()
    return figure


def render_spectrum(curve: FieldSweep, marker: float | None = None) -> None:
    """Show the excitation spectrum, saying plainly what stands behind it.

    Drawn only when the levels exist, which needs diagonalisation and therefore a
    short chain. A sweep that could only be done in closed form shows the
    ground-state curves and no spectrum, which is the honest rendering of a method
    that was not available rather than something to apologise for.

    Args:
        curve: The swept data.
        marker: A ratio ``g`` to mark, or ``None``.
    """
    if not curve.wants_spectrum:
        return
    st.pyplot(spectrum_figure(curve, marker), width="stretch")
    critical = curve.critical_gap
    # Two claims, deliberately in this order: what these numbers are not (twice
    # computed), then what they show. A reader who stops after the first sentence
    # has still been told the weaker guarantee.
    # No "unlike the curves above": on the Lab page this figure has its own tab and
    # there is nothing above it, so the comparison sent a reader looking for
    # something that was not on screen. The claim stands on its own instead.
    note = (
        f"**Excitation spectrum.** The {curve.levels_drawn} lowest levels come from "
        "exact diagonalisation alone: no second method checks them, though the ground "
        "state each one is measured from was cross-checked."
    )
    if critical is not None:
        note += (
            f" At **g = {critical[0]:.2f}** the gap is **{critical[1]:.3f}**, not zero: "
            "it closes exactly only in the infinite chain, and shrinks towards zero as "
            "the chain grows."
        )
    st.caption(note)


def momentum_spectrum_figure(
    ratios: list[float],
    columns: list[list[float]],
    n_sites: int,
    marker: float | None = None,
) -> Figure:
    r"""Draw every quasiparticle branch against the field, in one panel.

    One plot rather than a grid, because the point is how the branches move
    *relative to each other*: the lowest one dips towards zero near the critical
    field while the rest stay high, and separate axes would hide exactly that. The
    lowest branch is drawn solid and labelled, since it is the gap.

    Args:
        ratios: The swept ``g = h/J``.
        columns: One list of :math:`\epsilon(k)` per allowed momentum.
        n_sites: Chain length, for the momentum labels.
        marker: A ratio to mark, or ``None``.

    Returns:
        The figure, ready for :func:`streamlit.pyplot`.
    """
    figure = Figure(figsize=(7.4, 3.6), dpi=200)
    figure.patch.set_alpha(0.0)
    axes = figure.subplots()
    count = len(columns)
    for index, column in reversed(list(enumerate(columns))):
        # k = (2n+1)pi/L, the antiperiodic momenta the even sector is built from.
        momentum = (2 * index + 1) / n_sites
        axes.plot(
            ratios,
            column,
            color=ACCENT,
            linewidth=1.7 if index == 0 else 1.1,
            alpha=1.0 if index == 0 else 0.3 + 0.3 * (1.0 - index / max(1, count - 1)),
            label=rf"$k = {momentum:.3g}\pi$",
            zorder=2,
        )
    axes.set_ylabel(r"$\epsilon(k)$", fontsize=9, color=MUTED)
    axes.set_xlabel("$g = h / J$", fontsize=9, color=MUTED)
    axes.set_xlim(0.0, MAX_RATIO)
    axes.set_ylim(bottom=0.0)
    style_axes(axes)
    axes.axvline(1.0, color=MUTED, linestyle="--", linewidth=1.0, zorder=1)
    if marker is not None:
        axes.axvline(marker, color=HIGHLIGHT, alpha=0.25, linewidth=6.0, zorder=0)
    legend = axes.legend(fontsize=7, frameon=False, loc="upper left", ncol=2)
    for text in legend.get_texts():
        text.set_color(MUTED)
    figure.tight_layout()
    return figure


def render_momentum_spectrum(spec: TFIMSpec, points: int) -> None:
    r"""Show the free-fermion excitation spectrum, and say what it is not.

    Args:
        spec: The chain in the knob.
        points: Field values to sample.
    """
    ratios, columns = momentum_spectrum(spec.n_sites, spec.coupling, points)
    if not columns:
        st.info(
            "The free-fermion spectrum is derived for a periodic ring of even length, "
            f"and this chain is {graph.BOUNDARY_NAMES[spec.boundary]} of {spec.n_sites} spins. "
            "The mapping does not apply, so there is no dispersion to plot."
        )
        return
    st.pyplot(momentum_spectrum_figure(ratios, columns, spec.n_sites, spec.ratio), width="stretch")
    lowest = min(min(column) for column in columns)
    st.caption(
        rf"**Closed form, exact at every point.** The mapping turns the chain into "
        rf"{len(columns)} independent modes at momenta $k = (2n+1)\pi/L$, and "
        rf"$\epsilon(k) = 2\sqrt{{J^2 + h^2 - 2Jh\cos k}}$ is the cost of exciting one. "
        rf"These are **excitations**, not ground-state properties. The lowest branch "
        rf"comes down to ${lowest:.3f}$ and no further: the smallest momentum on a ring "
        rf"of {spec.n_sites} spins is $\pi/{spec.n_sites}$, not zero, which is precisely "
        "why a finite chain has no closing gap and therefore no sharp transition."
    )


def energy_response_figure(
    ratios: list[float],
    density: list[float],
    slope: list[float],
    curvature: list[float],
    marker: float | None = None,
) -> Figure:
    r"""Draw the energy density and its first two field derivatives, stacked.

    Args:
        ratios: The swept ``g = h/J``.
        density: ``E_0 / L``.
        slope: :math:`\partial (E_0/L)/\partial h`.
        curvature: :math:`\partial^2 (E_0/L)/\partial h^2`.
        marker: A ratio to mark on all three, or ``None``.

    Returns:
        The figure, ready for :func:`streamlit.pyplot`.
    """
    figure = Figure(figsize=(7.4, 6.2), dpi=200)
    figure.patch.set_alpha(0.0)
    panels = figure.subplots(3, 1, sharex=True)
    labels = (
        "$E_0 / L$",
        r"$\partial (E_0/L) / \partial h$",
        r"$\partial^2 (E_0/L) / \partial h^2$",
    )
    for axes, values, label in zip(panels, (density, slope, curvature), labels, strict=True):
        axes.plot(ratios, values, color=ACCENT, linewidth=1.8, zorder=2)
        axes.set_ylabel(label, fontsize=9, color=MUTED)
        style_axes(axes)
        axes.axvline(1.0, color=MUTED, linestyle="--", linewidth=1.0, zorder=1)
        if marker is not None:
            axes.axvline(marker, color=HIGHLIGHT, alpha=0.25, linewidth=6.0, zorder=0)

    sharpest = ratios[curvature.index(min(curvature))]
    panels[-1].annotate(
        f"sharpest at $g = {sharpest:.2f}$",
        xy=(sharpest, min(curvature)),
        xytext=(8, 12),
        textcoords="offset points",
        fontsize=7,
        color=MUTED,
    )
    panels[-1].set_xlabel("$g = h / J$", fontsize=9, color=MUTED)
    panels[-1].set_xlim(0.0, MAX_RATIO)
    figure.tight_layout()
    return figure


def render_energy_response(spec: TFIMSpec, points: int) -> None:
    r"""Show the energy and its derivatives, with the identity that checks them.

    Args:
        spec: The chain in the knob.
        points: Field values to sample.
    """
    ratios, density, slope, curvature = energy_response(spec.n_sites, spec.coupling, points)
    if not density:
        st.info(
            "The closed form is derived for a periodic ring of even length, and this "
            f"chain is {graph.BOUNDARY_NAMES[spec.boundary]} of {spec.n_sites} spins. Its "
            "derivatives are not available in closed form here."
        )
        return
    figure = energy_response_figure(ratios, density, slope, curvature, spec.ratio)
    st.pyplot(figure, width="stretch")
    sharpest = ratios[curvature.index(min(curvature))]
    st.caption(
        "**Where the transition actually shows up.** The energy is featureless, its "
        "first derivative rises, and its curvature dips sharpest at "
        f"**g = {sharpest:.2f}** — which is why a phase transition is defined through "
        "derivatives of a free energy and not through the energy. Every column is "
        "analytic, no finite differences: the first derivative is minus the transverse "
        "magnetisation by Hellmann-Feynman, so it is the same number this page plots "
        "as an observable elsewhere, and the two agreeing is what stands behind it."
    )


def magnetisation_response_figure(
    ratios: list[float],
    magnetisation: list[float],
    slope: list[float],
    curvature: list[float],
    marker: float | None = None,
) -> Figure:
    r"""Draw the transverse magnetisation and its first two field derivatives.

    The companion to :func:`energy_response_figure`, and it exists because the two
    are asked for separately. A question about how the *magnetisation* responds to
    the field is not answered by the energy's panels plus a note that one of them is
    the same curve negated: the reader asked for a susceptibility and its rate of
    change, and only the middle panel here is a quantity the energy figure carries.

    Args:
        ratios: The swept ``g = h/J``.
        magnetisation: :math:`\langle \sigma^x \rangle`.
        slope: :math:`\partial \langle \sigma^x \rangle/\partial h`.
        curvature: :math:`\partial^2 \langle \sigma^x \rangle/\partial h^2`.
        marker: A ratio to mark on all three, or ``None``.

    Returns:
        The figure, ready for :func:`streamlit.pyplot`.
    """
    figure = Figure(figsize=(7.4, 6.2), dpi=200)
    figure.patch.set_alpha(0.0)
    panels = figure.subplots(3, 1, sharex=True)
    labels = (
        r"$\langle \sigma^x \rangle$",
        r"$\partial \langle \sigma^x \rangle / \partial h$",
        r"$\partial^2 \langle \sigma^x \rangle / \partial h^2$",
    )
    series = (magnetisation, slope, curvature)
    for axes, values, label in zip(panels, series, labels, strict=True):
        axes.plot(ratios, values, color=ACCENT, linewidth=1.8, zorder=2)
        axes.set_ylabel(label, fontsize=9, color=MUTED)
        style_axes(axes)
        axes.axvline(1.0, color=MUTED, linestyle="--", linewidth=1.0, zorder=1)
        if marker is not None:
            axes.axvline(marker, color=HIGHLIGHT, alpha=0.25, linewidth=6.0, zorder=0)
    panels[1].annotate(
        f"steepest rise at $g = {ratios[slope.index(max(slope))]:.2f}$",
        xy=(ratios[slope.index(max(slope))], max(slope)),
        xytext=(8, -14),
        textcoords="offset points",
        fontsize=7,
        color=MUTED,
    )
    panels[-1].set_xlabel("$g = h / J$", fontsize=9, color=MUTED)
    panels[-1].set_xlim(0.0, MAX_RATIO)
    figure.tight_layout()
    return figure


def render_sweep_derivatives(curve: FieldSweep, marker: float | None = None) -> None:
    r"""Show the energy and its two field derivatives from an agent's own sweep.

    The same three panels the Lab page draws, from the sweep the model asked for
    rather than from the settings knob -- so a chat question about
    :math:`\partial^2 (E_0/L)/\partial h^2` is answered with that quantity and not
    with the magnetisation, which is the first derivative wearing a different name
    and carries nothing about the second.

    Args:
        curve: The swept data, which must carry derivatives.
        marker: A ratio ``g`` to mark, or ``None``.
    """
    if not curve.wants_derivatives:
        return
    ratios = [point.ratio for point in curve.points]
    density = [point.energy_density for point in curve.points]
    # Both are present at every point: `wants_derivatives` is all-or-nothing, so
    # this is a narrowing for the type checker rather than a fallback.
    slope = [point.slope for point in curve.points if point.slope is not None]
    curvature = [point.curvature for point in curve.points if point.curvature is not None]
    st.pyplot(energy_response_figure(ratios, density, slope, curvature, marker), width="stretch")
    sharpest = curve.sharpest_curvature
    where = (
        f" Its curvature dips sharpest at **g = {sharpest[0]:.2f}**, reaching "
        f"**{sharpest[1]:.3f}**."
        if sharpest is not None
        else ""
    )
    st.caption(
        "**Energy and its field derivatives.** Both derivatives are analytic, not "
        "finite differences: the first is minus the transverse magnetisation by "
        "Hellmann-Feynman and the second is differentiated in closed form, so neither "
        f"depends on how finely the {len(curve.points)} points were spaced.{where} The "
        "energy itself is featureless through the transition, which is why a phase "
        "transition is defined through derivatives of a free energy rather than "
        "through the energy."
    )
    mag = [point.magnetisation for point in curve.points]
    mag_slope = [
        point.magnetisation_slope for point in curve.points if point.magnetisation_slope is not None
    ]
    mag_curvature = [
        point.magnetisation_curvature
        for point in curve.points
        if point.magnetisation_curvature is not None
    ]
    # Drawn only when the sweep carries them at every point, which is the same
    # all-or-nothing rule the energy panels follow: a curve with holes is not a
    # curve. They come from the same closed form, so in practice this holds
    # whenever `wants_derivatives` does.
    if len(mag_slope) == len(curve.points) and len(mag_curvature) == len(curve.points):
        st.pyplot(
            magnetisation_response_figure(ratios, mag, mag_slope, mag_curvature, marker),
            width="stretch",
        )
        steepest = ratios[mag_slope.index(max(mag_slope))]
        st.caption(
            "**The same response, read as the magnetisation.** The first panel is what "
            "the chain does; the second is how hard it responds to a nudge, peaking at "
            f"**g = {steepest:.2f}**; the third is where that responsiveness itself "
            "turns over. All three are closed-form, and the middle one is minus the "
            "energy's curvature above — the same number arrived at two ways, which is "
            "the check that stands behind both figures."
        )


def render_curves(curve: FieldSweep, marker: float | None = None) -> None:
    """Show whichever curve was asked for, each with its own claim behind it.

    Which figures appear is the agent's decision, not this function's: the sweep
    carries the observable the model chose when it called the tool. A question about
    excitations gets the spectrum, a question about the derivatives gets the energy
    and its derivatives, a question about the ground state gets the ground-state
    curves, and only a question that asked for both gets both. See
    :data:`~src.tools.sweeps.Observable`.

    Args:
        curve: The swept data.
        marker: A ratio ``g`` to mark, or ``None``.
    """
    if not curve.ok:
        st.info(curve.detail)
        return
    if curve.observable == "derivatives":
        if curve.wants_derivatives:
            render_sweep_derivatives(curve, marker)
        else:
            st.info(
                "The derivatives are computed in closed form, which is derived for a "
                "periodic ring of even length; this chain is not one, so there is "
                "nothing to differentiate here."
            )
        return
    if not curve.wants_ground_state:
        render_spectrum(curve, marker)
        return
    st.pyplot(curves_figure(curve, marker), width="stretch")
    worst = curve.max_disagreement
    verified = (
        f"Every one of the {len(curve.points)} points was computed twice, by "
        f"{' and '.join(curve.methods)}, agreeing to {worst:.1e}."
        if worst is not None
        else f"Computed by {curve.methods[0]} alone: no independent check applies to this chain."
    )
    turning = curve.turning_point
    where = (
        f" The magnetisation rises fastest at **g = {turning:.2f}**, this chain's own estimate "
        "of a critical point that only exists exactly in the infinite chain."
        if turning is not None
        else ""
    )
    st.caption(verified + where)
    # Below the caption, not inside it. The sentence above claims two independent
    # methods agreed; the spectrum has no such claim behind it, and one caption
    # covering both would extend the guarantee to something that does not have it.
    render_spectrum(curve, marker)


# Where the mini-cartoons are sampled: deep in the ordered phase, on the way up,
# at the critical ratio of the infinite chain, and past it. Four columns, because
# the story is "two arrangements -> all of them" and four steps tell it without
# the strip becoming a wall.
CARTOON_RATIOS = (0.2, 0.7, 1.0, 2.0)

# Rows drawn in each mini-cartoon. Fewer than the main one: these are thumbnails,
# and their point is how the weight spreads rather than what each row is.
CARTOON_ROWS = 4

# Points in the gap curve beside the phase diagram. Twenty-five over a range of
# 0..2, so the step is 1/12 of a coupling and g = 1 is sampled exactly -- which is
# what keeps the cusp at the critical field sharp instead of rounded off. Nothing
# here diagonalises anything: both curves are closed-form and O(L) per point. The
# expensive sweeps on this page are the ones behind the points slider, not this.
PHASE_POINTS = 25

# Smallest window the phase diagram draws, in units of the coupling. Two, so the
# critical line is the diagonal of a square plot and both phases get equal room;
# making one bigger would look like a claim that one is bigger. The knob allows
# J up to 10 and h up to 20, so the window grows when it has to -- a diagram that
# does not contain the chain it is marking is worse than no diagram.
PHASE_SPAN = 2.0


@st.cache_data(show_spinner=False)
def ground_state_picture(
    n_sites: int, coupling: float, field: float, boundary: str
) -> Superposition:
    """The knob's ground state read back as spin configurations.

    Args:
        n_sites: Chain length ``L``.
        coupling: The Ising coupling ``J``.
        field: The transverse field ``h``.
        boundary: ``"periodic"`` or ``"open"``.

    Returns:
        The heaviest arrangements and the numbers around them.
    """
    return quantumness.superposition(
        TFIMSpec(n_sites=n_sites, coupling=coupling, field=field, boundary=boundary)  # type: ignore[arg-type]
    )


@st.cache_data(show_spinner=False)
def superposition_across_field(
    n_sites: int, coupling: float, boundary: str, ratios: tuple[float, ...]
) -> list[Superposition]:
    """The same ground state at several fields, for the strip of cartoons.

    Args:
        n_sites: Chain length ``L``.
        coupling: The Ising coupling ``J``.
        boundary: ``"periodic"`` or ``"open"``.
        ratios: The values of ``g = h / J`` to look at.

    Returns:
        One picture per ratio, in the order given.
    """
    return [ground_state_picture(n_sites, coupling, ratio * coupling, boundary) for ratio in ratios]


@st.cache_data(show_spinner=False)
def gap_curves(
    n_sites: int, coupling: float, boundary: str
) -> tuple[list[float], list[float], list[float]]:
    """The gap against the field, for the infinite chain and for this one.

    Two curves rather than one, because they say different things and only one of
    them is about the chain on screen. The infinite chain's gap reaches zero, and
    where it does is the critical field; a finite ring's does not, and the distance
    between the two curves is the whole of what "finite" costs.

    Args:
        n_sites: Chain length ``L``.
        coupling: The Ising coupling ``J``.
        boundary: ``"periodic"`` or ``"open"``.

    Returns:
        The ratios ``g = h / J``, the infinite chain's gap at each, and this
        chain's own gap at each -- the last empty when the closed form does not
        cover this chain, since there is then nothing exact to draw.
    """
    ratios = [MAX_RATIO * step / (PHASE_POINTS - 1) for step in range(PHASE_POINTS)]
    thermodynamic = [exact.gap_thermodynamic(coupling, ratio * coupling) for ratio in ratios]
    finite: list[float] = []
    probe = TFIMSpec(n_sites=n_sites, coupling=coupling, field=coupling, boundary=boundary)  # type: ignore[arg-type]
    if exact.unsupported_reason(probe) is None:
        finite = [
            exact.gap(
                TFIMSpec(
                    n_sites=n_sites,
                    coupling=coupling,
                    field=ratio * coupling,
                    boundary=boundary,  # type: ignore[arg-type]
                )
            )
            for ratio in ratios
        ]
    return ratios, thermodynamic, finite


def draw_spin_rows(
    axes: Any, rows: tuple[Configuration, ...], *, periodic: bool, label_rows: bool = True
) -> None:
    """Draw spin configurations as rows of up and down arrows.

    The one drawing primitive behind every cartoon here, so the big figure and the
    thumbnails cannot drift apart in convention: up is up, and the two directions
    are the two colours this page uses everywhere else.

    Args:
        axes: Where to draw.
        rows: The configurations, most probable first, drawn top to bottom.
        periodic: Whether the chain wraps, which the wall count depends on.
        label_rows: Whether to label each row with its domain-wall count.
    """
    sites = len(rows[0].spins)
    horizontal: list[int] = []
    vertical: list[int] = []
    lengths: list[float] = []
    colours: list[str] = []
    for row, configuration in enumerate(rows):
        for site, spin in enumerate(configuration.spins):
            horizontal.append(site)
            vertical.append(-row)
            lengths.append(0.62 * spin)
            colours.append(ACCENT if spin > 0 else HIGHLIGHT)

    axes.quiver(
        horizontal,
        vertical,
        [0.0] * len(horizontal),
        lengths,
        color=colours,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.006,
        headwidth=3.2,
        headlength=4.2,
        headaxislength=3.6,
        pivot="mid",
    )
    axes.set_xlim(-0.8, sites - 0.2)
    axes.set_ylim(-len(rows) + 0.45, 0.55)
    axes.set_xticks([])
    if label_rows:
        axes.set_yticks([-row for row in range(len(rows))])
        axes.set_yticklabels(
            [
                f"{configuration.domain_walls(periodic=periodic)} wall"
                + ("s" if configuration.domain_walls(periodic=periodic) != 1 else "")
                for configuration in rows
            ],
            fontsize=7,
            color=MUTED,
        )
    else:
        axes.set_yticks([])


def superposition_figure(picture: Superposition) -> Figure:
    """Draw the ground state as the arrangements it is made of.

    Args:
        picture: The measured superposition.

    Returns:
        The figure: one row of arrows per arrangement, and beside it the share of
        the state that arrangement carries.
    """
    rows = picture.configurations
    figure = Figure(figsize=(7.4, 0.42 * len(rows) + 1.0), dpi=200)
    figure.patch.set_alpha(0.0)
    panels = figure.subplots(1, 2, gridspec_kw={"width_ratios": [3.0, 1.0]})
    spins, shares = panels[0], panels[1]

    style_axes(spins)
    spins.grid(False)
    draw_spin_rows(spins, rows, periodic=picture.spec.boundary == "periodic")
    spins.set_title(
        f"the {len(rows)} heaviest of {picture.full_count} arrangements",
        fontsize=8,
        color=MUTED,
    )

    style_axes(shares)
    positions = [-row for row in range(len(rows))]
    shares.barh(
        positions,
        [configuration.probability for configuration in rows],
        height=0.5,
        color=ACCENT,
        alpha=0.75,
    )
    for position, configuration in zip(positions, rows, strict=True):
        shares.annotate(
            f"{configuration.probability:.1%}",
            xy=(configuration.probability, position),
            xytext=(4, -2),
            textcoords="offset points",
            fontsize=7,
            color=MUTED,
        )
    widest = max(configuration.probability for configuration in rows)
    shares.set_xlim(0.0, widest * 1.45)
    shares.set_ylim(-len(rows) + 0.45, 0.55)
    shares.set_xticks([])
    shares.set_yticks([])
    shares.set_title("share of the state", fontsize=8, color=MUTED)

    figure.tight_layout()
    return figure


def spreading_figure(pictures: list[Superposition], ratios: tuple[float, ...]) -> Figure:
    """Draw the same ground state at four fields, side by side.

    One panel per field, each showing its heaviest arrangements. Read left to
    right it is the transition itself: the weight starts on the two aligned
    arrangements, spreads through the ones with a domain wall in them, and ends
    shared equally among all :math:`2^L`.

    Args:
        pictures: One measured superposition per ratio.
        ratios: The ratios they were measured at, in the same order.

    Returns:
        The figure.
    """
    figure = Figure(figsize=(7.4, 2.5), dpi=200)
    figure.patch.set_alpha(0.0)
    panels = figure.subplots(1, len(pictures))
    for axes, picture, ratio in zip(panels, pictures, ratios, strict=True):
        style_axes(axes)
        axes.grid(False)
        rows = picture.configurations[:CARTOON_ROWS]
        draw_spin_rows(axes, rows, periodic=picture.spec.boundary == "periodic", label_rows=False)
        # Weight, drawn as opacity, would fight the arrow colours. A bar under each
        # row is unambiguous and stays readable when the weights are nearly equal --
        # which is exactly what the rightmost panel has to show.
        for row, configuration in enumerate(rows):
            axes.barh(
                -row - 0.36,
                configuration.probability / max(rows[0].probability, 1e-12) * (len(rows) - 1),
                height=0.1,
                left=-0.5,
                color=MUTED,
                alpha=0.55,
            )
        axes.set_title(
            f"$g = {ratio:g}$\n{picture.effective_count:.1f} of {picture.full_count}",
            fontsize=8,
            color=MUTED,
        )
    figure.tight_layout()
    return figure


ORDERED_FILL = "#4c6ef5"
DISORDERED_FILL = "#2f9e44"


def phase_diagram_figure(spec: TFIMSpec) -> Figure:
    r"""Draw the phase diagram beside the gap that decides it.

    Two panels, because the diagram alone answers "which phase" and not "why
    there". On the left the plane is filled flat -- one colour per phase, each
    labelled with what the spins are doing in it -- and the boundary between the
    fills *is* the critical line, so the two phases and their border are visible
    before anything is read. On the right the gap along this chain's own coupling,
    which is what makes that border a border: the infinite chain's gap reaches zero
    at :math:`h = J` and nowhere else, and this ring's does not reach zero at all.

    An earlier version shaded the plane continuously by the gap. It was prettier and
    it was harder to read: a smooth gradient has no boundary in it, so the phases
    and the critical field both had to be found by reading the labels.

    Args:
        spec: The chain in the knob.

    Returns:
        The figure.
    """
    # Wide enough to contain the chain being marked. The knob allows J up to 10 and
    # h up to 20, and a star drawn outside its own axes is a bug a reader cannot see.
    span = max(PHASE_SPAN, 1.18 * max(spec.coupling, spec.field))
    figure = Figure(figsize=(7.4, 3.5), dpi=200)
    figure.patch.set_alpha(0.0)
    panels = figure.subplots(1, 2, gridspec_kw={"width_ratios": [1.0, 1.15]})
    plane, curve = panels[0], panels[1]

    # ---- left: the plane, one flat colour per phase ----
    style_axes(plane)
    plane.grid(False)
    plane.fill_between([0.0, span], [0.0, span], color=ORDERED_FILL, alpha=0.30)
    plane.fill_between([0.0, span], [0.0, span], [span, span], color=DISORDERED_FILL, alpha=0.30)
    plane.plot([0.0, span], [0.0, span], color=HIGHLIGHT, linewidth=2.2, zorder=3)
    plane.annotate(
        "ORDERED\nferromagnet\n↑↑↑↑↑↑  or  ↓↓↓↓↓↓",
        xy=(0.70 * span, 0.20 * span),
        fontsize=8,
        color=ORDERED_FILL,
        ha="center",
        va="center",
        fontweight="bold",
    )
    plane.annotate(
        "DISORDERED\nparamagnet\n→→→→→→",
        xy=(0.26 * span, 0.80 * span),
        fontsize=8,
        color=DISORDERED_FILL,
        ha="center",
        va="center",
        fontweight="bold",
    )
    plane.annotate(
        "critical line  $h = J$",
        xy=(0.52 * span, 0.52 * span),
        fontsize=7.5,
        color=HIGHLIGHT,
        rotation=45,
        rotation_mode="anchor",
        ha="center",
        va="bottom",
    )
    # The dotted line is the slider: moving the field walks this chain up and down
    # it, and the crossing is the field at which it changes phase.
    plane.axvline(spec.coupling, color=MUTED, linestyle=":", linewidth=1.0, zorder=2)
    plane.scatter(
        [spec.coupling],
        [spec.field],
        s=170,
        marker="*",
        color=HIGHLIGHT,
        edgecolor="white",
        linewidth=0.7,
        zorder=5,
    )
    plane.annotate(
        f"this chain\n$g = {spec.ratio:.2f}$",
        xy=(spec.coupling, spec.field),
        xytext=(8, -16),
        textcoords="offset points",
        fontsize=7.5,
        color=HIGHLIGHT,
    )
    plane.set_xlabel("coupling $J$", fontsize=9, color=MUTED)
    plane.set_ylabel("field $h$", fontsize=9, color=MUTED)
    plane.set_xlim(0.0, span)
    plane.set_ylim(0.0, span)
    plane.set_aspect("equal")

    # ---- right: the gap, which is what makes the line a line ----
    ratios, thermodynamic, finite = gap_curves(spec.n_sites, spec.coupling, spec.boundary)
    style_axes(curve)
    curve.fill_between(
        [0.0, 1.0], 0.0, 1.0, transform=curve.get_xaxis_transform(), color=ORDERED_FILL, alpha=0.10
    )
    curve.fill_between(
        [1.0, MAX_RATIO],
        0.0,
        1.0,
        transform=curve.get_xaxis_transform(),
        color=DISORDERED_FILL,
        alpha=0.10,
    )
    curve.plot(ratios, thermodynamic, color=MUTED, linewidth=1.6, label="infinite chain")
    if finite:
        curve.plot(
            ratios, finite, color=ACCENT, linewidth=1.8, label=f"this ring, $L = {spec.n_sites}$"
        )
    curve.axvline(1.0, color=HIGHLIGHT, linewidth=2.2, zorder=3)
    curve.annotate(
        f"critical field\n$h_c = J = {spec.coupling:g}$",
        xy=(1.0, 0.94),
        xycoords=curve.get_xaxis_transform(),
        xytext=(7, 0),
        textcoords="offset points",
        fontsize=7.5,
        color=HIGHLIGHT,
        va="top",
    )
    curve.axvline(spec.ratio, color=HIGHLIGHT, alpha=0.25, linewidth=6.0, zorder=0)
    curve.set_xlabel("$g = h / J$", fontsize=9, color=MUTED)
    curve.set_ylabel(r"gap  $\Delta$", fontsize=9, color=MUTED)
    curve.set_xlim(0.0, MAX_RATIO)
    curve.set_ylim(bottom=0.0)
    legend = curve.legend(fontsize=7, frameon=False, loc="upper left")
    for text in legend.get_texts():
        text.set_color(MUTED)

    figure.tight_layout()
    return figure


@st.cache_data(show_spinner=False)
def quantum_facts(n_sites: int, coupling: float, field: float, boundary: str) -> Quantumness:
    """Measure the algebraic identities for the chain in the knob.

    Args:
        n_sites: Chain length ``L``.
        coupling: The Ising coupling ``J``.
        field: The transverse field ``h``.
        boundary: ``"periodic"`` or ``"open"``.

    Returns:
        The measured statements. Cached on primitives rather than on the spec,
        like every other panel here, so a slider drag that returns to a previous
        position costs nothing.
    """
    return quantumness.measure(
        TFIMSpec(n_sites=n_sites, coupling=coupling, field=field, boundary=boundary)  # type: ignore[arg-type]
    )


def render_quantumness(spec: TFIMSpec) -> None:
    r"""Show why the model is *quantum*: the algebra, the phase diagram, the state.

    This used to open with :math:`\|[\hat H_{ZZ}, \hat H_X]\|` as a headline number,
    and that was a mistake worth recording. A Frobenius norm is not a physical
    quantity -- it scales with the size of the matrix, so ``55.426`` says only "not
    zero, on a chain of this length", which is the one thing a reader would have
    believed without being shown. It is still evaluated, in the table where a check
    belongs, and no longer presented as a finding.

    Three sections, in the order the argument runs. **The algebra** is the premise:
    the two terms do not commute, so no arrangement of spins can be an eigenstate.
    **The phase diagram** is what that competition produces, and says where this
    chain sits in it. **The state itself** is the premise cashed out -- the actual
    arrangements the eigenvector is made of, and how that mixture spreads as the
    field turns up. Every arrow is read off the eigenvector; nothing here is an
    artist's impression.

    Args:
        spec: The chain in the knob.
    """
    if spec.n_sites > quantumness.MAX_SITES:
        st.info(
            f"These pictures are drawn from the whole state vector, and the identities "
            f"under them are matrix-wide, so both are built densely and capped at "
            f"L = {quantumness.MAX_SITES}. This chain has {spec.n_sites} spins."
        )
        return
    found = quantum_facts(spec.n_sites, spec.coupling, spec.field, spec.boundary)
    picture = ground_state_picture(spec.n_sites, spec.coupling, spec.field, spec.boundary)

    st.markdown(
        "**The two terms pull in different directions, and cannot both be satisfied.** "
        f"In {HAMILTONIAN_INLINE} the couplings want the spins lined up along $z$ and "
        "the field wants each of them along $x$. Because the two pieces do not commute, "
        "no arrangement of spins is an eigenstate of the whole thing — which is the only "
        "reason anything below is interesting."
    )
    render_quantum_identities(found)

    st.divider()
    st.markdown(
        "**What that competition produces.** Below the line the couplings win and the "
        "spins order; above it the field wins and each spin points along $x$ on its own."
    )
    st.pyplot(phase_diagram_figure(spec), width="stretch")
    ring_gap = (
        f" This ring's own smallest excitation is **{exact.gap(spec):.3f}** — it never "
        "reaches zero, which is why a finite chain crosses over instead of transitioning."
        if exact.unsupported_reason(spec) is None
        else ""
    )
    st.caption(
        f"The star is this chain, on the {phase_label(spec.ratio)}. The dark valley is not "
        f"drawn on: it is where the shaded gap actually vanishes.{ring_gap}"
    )

    st.divider()
    st.markdown(
        "**So what is the ground state, then?** Several arrangements of spins at the same "
        "time. These are the ones it is mostly made of, labelled by how many bonds in "
        "them are broken — a *domain wall* is a pair of neighbours pointing opposite ways, "
        "and each one costs $2J$."
    )
    st.pyplot(superposition_figure(picture), width="stretch")
    st.caption(
        f"Those rows hold **{picture.shown_weight:.0%}** of the state. Counted over all "
        f"{picture.full_count} arrangements it is **{picture.effective_count:.1f} of them "
        f"at once**. A single spin's own arrow is only **{picture.arrow_length:.2f}** long "
        "out of 1 — the rest of it is not missing, it is stored in this spin's "
        "correlations with its neighbours."
    )

    st.markdown(
        "**Turn the field up and that superposition spreads.** Left to right is the "
        "transition itself: the weight starts on the two aligned arrangements, moves "
        "through the ones with a domain wall in them, and ends shared out evenly."
    )
    st.pyplot(
        spreading_figure(
            superposition_across_field(spec.n_sites, spec.coupling, spec.boundary, CARTOON_RATIOS),
            CARTOON_RATIOS,
        ),
        width="stretch",
    )
    st.caption(
        "The number under each panel is how many arrangements that state is at once, out "
        f"of {picture.full_count}. Blue is up and orange is down; the grey bar under each "
        "row is that row's share of the state."
    )


def render_quantum_identities(found: Quantumness) -> None:
    r"""Show the operator identities everything else on the tab rests on.

    A table rather than a paragraph, and its own function rather than fifty lines
    inside :func:`render_quantumness`. Each row carries the number the matrices
    actually gave -- an operator that should vanish shown vanishing, and the one
    that must not vanish shown not vanishing -- so the premise of the pictures
    below it is checked on screen rather than asserted.

    Args:
        found: The measured identities.
    """
    # A Markdown table rather than a dataframe, because these cells are operator
    # algebra and a dataframe renders them as literal text: `$\hat\sigma^x$` would
    # appear with its dollars showing. Markdown gives them to KaTeX.
    tick = "✅"
    cross = "❌"
    rows = (
        (
            r"$[\hat H_{ZZ}, \hat H_X] \neq 0$ — the two terms compete",
            f"{found.terms_commutator:.6f}",
            "non-zero when $J, h > 0$",
            cross if (found.terms_commutator > 1e-12) == found.classical else tick,
        ),
        (
            r"$\{\hat\sigma^x_0, \hat\sigma^z_0\} = 0$ — one spin, two axes: anticommute",
            f"{found.onsite_anticommutator:.2e}",
            "$0$",
            tick if found.onsite_anticommutator < 1e-12 else cross,
        ),
        (
            r"$[\hat\sigma^x_0, \hat\sigma^z_1] = 0$ — different spins are independent",
            f"{found.offsite_commutator:.2e}",
            "$0$",
            tick if found.offsite_commutator < 1e-12 else cross,
        ),
        (
            r"$[\hat H, \hat P] = 0$, with $\hat P = \prod_i \hat\sigma^x_i$ — the "
            r"$\mathbb{Z}_2$ symmetry",
            f"{found.parity_commutator:.2e}",
            "$0$",
            tick if found.parity_commutator < 1e-9 else cross,
        ),
        (
            r"$\langle \hat P \rangle$ in the ground state",
            f"{found.parity_expectation:+.6f}",
            "$+1$ (even sector)",
            tick if found.parity_expectation > 0.999 else "—",
        ),
    )
    st.markdown("**Every identity below is evaluated on the matrices, not quoted**")
    st.markdown(
        "\n".join(
            [
                "| identity | measured | expected | |",
                "|---|---|---|---|",
                *(
                    f"| {identity} | `{value}` | {want} | {held} |"
                    for identity, value, want, held in rows
                ),
            ]
        )
    )
    if found.parity_expectation <= 0.999:
        st.caption(
            r"**$\langle \hat P \rangle$ is not $\pm 1$ here, and that is the physics "
            r"rather than a failure.** At $h = 0$ the two ground states are exactly "
            r"degenerate, so every combination of them is also a ground state and the "
            r"solver returns an arbitrary one. Parity is only pinned once the field "
            r"separates the sectors — which is the degeneracy the ordered phase breaks."
        )

    # One row, not three metrics and a paragraph. The inequality is worth showing
    # holding; how many decimal places of slack it holds by is not the point, and a
    # sentence converting hbar into joule-seconds was three lines nobody read.
    st.markdown(
        r"**And the uncertainty relation holds.** With "
        r"$\hat S^a = \tfrac{\hbar}{2}\hat\sigma^a$ and $\hbar = 1$, "
        r"$\Delta \hat S^y \Delta \hat S^z \ge \tfrac{\hbar}{2}"
        r"|\langle \hat S^x \rangle|$ reads "
        rf"${found.product:.4f} \ge {found.bound:.4f}$ here, and the gap between the two "
        r"sides closes as the field grows — deep in the disordered phase the ground "
        r"state *is* an eigenstate of $\hat\sigma^x$ and the inequality becomes an "
        r"equality."
    )

    st.caption(
        f"These are evaluated on operators built here from scratch, not on the solver's "
        f"matrices — so a shared convention cannot hide a shared mistake. This module's "
        f"own ground-state energy is {found.energy:.9f}"
        + (
            f", which is {found.energy_disagreement:.1e} from the free-fermion closed form."
            if found.energy_disagreement is not None
            else ", and the closed form does not apply to this chain, so nothing checks it."
        )
    )


def render_scaling(spec: TFIMSpec) -> None:
    """Plot the energy density against ``1/L`` with the exact infinite-chain value.

    This is the panel that says what a small exact calculation is *for*. A single
    chain length is a number with no context; the sequence shows the number
    converging, and the horizontal line shows what it is converging to -- computed
    in closed form for the infinite chain rather than extrapolated from the points
    above it.

    Args:
        spec: The chain in the knob, whose coupling and field are used.
    """
    inverse, densities, limit = scaling(spec.coupling, spec.field)
    figure = Figure(figsize=(7.4, 3.2), dpi=200)
    figure.patch.set_alpha(0.0)
    axes = figure.subplots()
    style_axes(axes)

    axes.axhline(limit, color=MUTED, linestyle="--", linewidth=1.0, zorder=1)
    axes.plot(inverse, densities, color=ACCENT, marker="o", markersize=4, linewidth=1.4, zorder=2)
    for reciprocal, density, sites in zip(inverse, densities, SCALING_SITES, strict=True):
        if sites == spec.n_sites:
            axes.scatter([reciprocal], [density], color=HIGHLIGHT, s=55, zorder=3)
        axes.annotate(
            f"L={sites}",
            xy=(reciprocal, density),
            xytext=(0, 7),
            textcoords="offset points",
            fontsize=7,
            color=MUTED,
            ha="center",
        )
    axes.annotate(
        f"infinite chain: {limit:.6f}",
        xy=(0.0, limit),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=7,
        color=MUTED,
    )
    axes.set_xlabel("$1 / L$", fontsize=9, color=MUTED)
    axes.set_ylabel("$E_0 / L$", fontsize=9, color=MUTED)
    axes.set_xlim(-0.01, max(inverse) * 1.15)

    st.pyplot(figure, width="stretch")
    here = SCALING_SITES.index(spec.n_sites) if spec.n_sites in SCALING_SITES else None
    gap = abs(densities[here] - limit) if here is not None else None
    closing = (
        f" At L = {spec.n_sites} the energy per site is still **{gap:.4f}** away from the "
        "infinite-chain value, which is the size of the error every finite-chain answer carries."
        if gap is not None
        else ""
    )
    st.caption(
        "Points from the closed-form solution; the dashed line is the exact "
        "infinite-chain energy density, an elliptic integral rather than an "
        "extrapolation of these points." + closing
    )


# --------------------------------------------------------------------------
# The settings knob, and the pages that read it
# --------------------------------------------------------------------------


# Both surfaces label the weighting dial with this, so the sidebar and the chat
# page cannot drift apart. Alpha is what the retrieval literature calls the vector
# half's weight, and writing it as LaTeX keeps the source ASCII (a literal Greek
# alpha is RUF001) and matches the "Sites $L$" dials.
VECTOR_SHARE_LABEL = r"Vector search weight $\alpha$"

# How the shelf choice reads on screen. The empty string is the default and has to
# say what it does, because "" in a dropdown looks like a bug rather than a policy.
SHELF_CHOICE_LABELS = {
    "": "Let the router choose",
    "physics-notes": "The model itself (physics-notes)",
    "quantum-computing": "Quantum computing (quantum-computing)",
    "applications": "Business, ML and R&D uses (applications)",
}


def search_note(vector_share: float) -> str:
    """Say what this weighting costs, in one sentence.

    Args:
        vector_share: The slider's position, as the vector half's weight.

    Returns:
        What the setting gives up. Both ends of the slider lose something real,
        and a control that only advertises what it improves invites a user to
        drag it to an end and wonder why the answers got worse.
    """
    if vector_share >= 1.0:
        return (
            "Vector search alone — a question naming an author or an arXiv id no longer "
            "matches, because that text sits in a note's frontmatter and was never "
            "embedded."
        )
    if vector_share <= 0.0:
        return (
            "Keyword search alone — paraphrase stops working, and “why does the gap "
            "close” no longer finds a passage that says the dispersion vanishes."
        )
    return "A passage both halves rank outranks one that only either liked."


def read_settings_knob(learned: Audience | None = None) -> Setting:
    """Render the settings knob in the sidebar and return what it reads.

    Collapsed by default, and grouped into the two things a parameter can
    change: **Physics** moves the number, **Language model** moves only the
    prose describing it. Keeping the groups visibly apart is how the page makes
    its central claim legible -- turn the temperature to 2.0 and the energy
    below does not move by one digit, because no model computed it.

    Every bound comes from :mod:`src.agent.setting` rather than being typed into
    the widget calls, so the limit a user meets in the interface is the same
    limit the tests assert.

    Args:
        learned: The register this thread's ratings point to, if any. It becomes
            the slider's starting position and says so on screen.

            This is the whole of "the agent adjusts to feedback", and it is
            applied *here* rather than inside the graph on purpose. A preference
            that moved the answer without moving the control would be an invisible
            override -- the user would read a beginner's answer with the slider
            showing "practitioner" and have no way to work out why. On the slider
            it is visible, attributable and draggable back in one gesture.

    Returns:
        The knob's current position, validated.
    """
    # A shut knob hides a moved dial, and a user who left the temperature at 1.4 and
    # forgot needs to be told before they read a strangely worded answer. That used
    # to be a loose sentence under the knob, which read as a status code nobody
    # asked for; it belongs on the knob's own label, where it is a marker rather
    # than a message. Read from the previous rerun's knob, which is the only
    # position available before this run's widgets have drawn themselves -- and the
    # same position, because widget state persists across a rerun.
    previous = st.session_state.get("setting")
    already_moved = previous.changes_from_defaults() if isinstance(previous, Setting) else ()
    label = "⚙️ Settings knob"
    if already_moved:
        label = f"{label} — {len(already_moved)} changed"

    with st.sidebar:
        st.header("⚙️ Settings")
        st.caption(
            "Everything adjustable lives in the knob below. Defaults answer a "
            "sensible question, so it can be left shut."
        )

        with st.expander(label, expanded=False):
            if already_moved:
                st.caption("Changed from default: " + "; ".join(already_moved) + ".")
            physics_tab, model_tab, search_tab, tools_tab = st.tabs(
                ["Physics", "Language model", "Knowledge search", "Tool playground"]
            )

            with physics_tab:
                st.caption("These change the computed number.")
                n_sites = st.slider(
                    "Sites $L$",
                    min_value=2,
                    max_value=MAX_SITES_STATEVECTOR,
                    value=DEFAULT_SITES,
                    help=(
                        "Every added site doubles the state vector, so the wait and the "
                        "memory both grow with this and nothing else on the page."
                    ),
                )
                coupling = st.slider(
                    "Coupling $J$", min_value=0.1, max_value=2.0, value=1.0, step=0.1
                )
                field = st.slider(
                    "Field $h$",
                    min_value=0.0,
                    max_value=2.0,
                    value=1.0,
                    step=FIELD_STEP,
                    help=(
                        "Coarse on purpose: each position is a full re-solve, and the "
                        "values that matter — zero, the critical point at h = J, deep in "
                        "either phase — are all on this grid."
                    ),
                )
                boundary = st.radio("Boundary", options=["periodic", "open"], horizontal=True)
                st.divider()
                sweep_points = st.slider(
                    "Sweep resolution",
                    min_value=MIN_SWEEP_POINTS,
                    max_value=MAX_SWEEP_POINTS,
                    value=DEFAULT_SWEEP_POINTS,
                    help=(
                        "Points sampled across the field range — both in the sweep plots "
                        "below and in the curves the agent draws when it is asked for one. "
                        "Each point is a diagonalisation, so a high setting is a slow page. "
                        "A question that asks for a rougher sketch can still get one; this "
                        "setting is the ceiling."
                    ),
                )
                max_sites_costly = st.slider(
                    "Sweep by diagonalisation up to $L$",
                    min_value=2,
                    max_value=MAX_SITES_STATEVECTOR,
                    value=8,
                    help=(
                        "Above this length a sweep needs one diagonalisation per point, "
                        "so it is declined rather than run slowly."
                    ),
                )

            with model_tab:
                st.caption(
                    "These change only the wording of the explanation, never a "
                    "computed number. Turn the temperature up and the energy below "
                    "does not move by one digit."
                )
                # First in this tab, because it is the one dial here a reader who
                # does not care about sampling temperature still wants -- and it
                # belongs *in* this group rather than above the knob: it changes how
                # an answer is written and never what it says, which is exactly what
                # this tab means.
                audience = st.select_slider(
                    "Explain it for",
                    options=list(AUDIENCE_LABELS),
                    value=learned or DEFAULT_AUDIENCE,
                    format_func=lambda level: AUDIENCE_LABELS[level],
                    help=(
                        "The number is computed and cross-checked before any of this is "
                        "consulted, so it is identical at every level."
                    ),
                )
                if learned is not None:
                    st.caption(
                        f"Set to **{AUDIENCE_LABELS[learned]}** from your earlier ratings — "
                        "drag it anywhere and your choice wins."
                    )
                st.divider()
                chat_model = st.selectbox("Chat model", options=MODEL_CHOICES, index=0)
                custom_model = st.text_input(
                    "…or a slug typed by hand",
                    value="",
                    placeholder="vendor/model-name",
                    help=(
                        "The escape hatch: a slug that is wrong here, or a model "
                        "released after this list was written, can still be reached."
                    ),
                )
                # Every dial below opens where the process actually runs, read off
                # ModelSetting rather than repeated as a literal. A slider that
                # opens at a number the configuration no longer uses does not
                # merely mislabel itself: the sidebar is what the run is made
                # from, so drawing it is what sets the value.
                default_model = ModelSetting()
                temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=2.0,
                    value=default_model.temperature,
                    step=0.05,
                )
                cap_length = st.checkbox(
                    "Cap the reply length", value=default_model.max_output_tokens is not None
                )
                max_output_tokens = (
                    st.slider(
                        "Max output tokens",
                        min_value=64,
                        max_value=8192,
                        value=default_model.max_output_tokens or DEFAULT_MAX_OUTPUT_TOKENS,
                        step=64,
                    )
                    if cap_length
                    else None
                )
                st.divider()
                max_model_calls = st.slider(
                    "Model calls per question",
                    min_value=1,
                    max_value=100,
                    value=12,
                    help="The ceiling that turns a runaway tool-calling loop into a stopped run.",
                )
                max_retries = st.slider(
                    "Retries on transient failure", 0, 10, value=default_model.max_retries
                )
                requests_per_second = st.slider(
                    "Requests per second",
                    min_value=0.0,
                    max_value=10.0,
                    value=default_model.requests_per_second,
                    step=0.5,
                    help=(
                        "Spacing between calls. An answer makes about seven, so this "
                        "is a direct multiplier on how long one takes."
                    ),
                )
                request_timeout = st.slider(
                    "Request timeout (s)", min_value=5.0, max_value=300.0, value=60.0, step=5.0
                )

            with search_tab:
                st.caption(
                    "How the notes are searched before an answer is written. These "
                    "change which passages are cited, never a computed number — the "
                    "solver never reads a passage."
                )
                passages = st.slider(
                    "Passages to use",
                    min_value=1,
                    max_value=MAX_PASSAGES,
                    value=DEFAULT_TOP_K,
                    help=(
                        "The most an answer may cite. The search always looks at more "
                        "than this and the grader throws the rest away, so raising it "
                        "admits weaker passages rather than finding better ones."
                    ),
                )
                vector_share = st.slider(
                    VECTOR_SHARE_LABEL,
                    min_value=0.0,
                    max_value=1.0,
                    value=DEFAULT_VECTOR_SHARE,
                    step=0.1,
                    help=(
                        "Alpha is the weight the vector half of the search carries, and "
                        "what is left over is the keyword half's. 1.0 is vector search alone; "
                        "0.0 is keyword search alone. A half weighted at zero is not "
                        "searched at all."
                    ),
                )
                # The number is the honest label: a "50 / 50" caption under a slider
                # showing 0.7 is the kind of drift that makes a control untrustworthy.
                st.caption(
                    rf"$\alpha$ = {vector_share:.1f}, so ranking is "
                    rf"{vector_share:.0%} vector search and "
                    rf"{1 - vector_share:.0%} keyword search. " + search_note(vector_share)
                )
                rounds = st.slider(
                    "Searches per question",
                    min_value=1,
                    max_value=MAX_ROUNDS,
                    value=MAX_ROUNDS,
                    key="knob_rounds",
                    help=(
                        "Two lets a failed search be reformulated from the grader's "
                        "verdict and tried again. One turns the corrective loop off, "
                        "which is how you see what it was doing."
                    ),
                )
                shelf = st.selectbox(
                    "Knowledge base",
                    options=["", *shelf_names()],
                    index=0,
                    format_func=lambda name: SHELF_CHOICE_LABELS.get(name, name),
                    key="knob_shelf",
                    help=(
                        "Normally the router picks, and an empty shelf widens to the "
                        "whole library. Forcing the wrong one is how you check the "
                        "filter does anything."
                    ),
                )

            with tools_tab:
                st.caption(
                    "The agent decides for itself whether extra work would help. This "
                    "is what it is allowed to reach for — switch one off and it is not "
                    "even described to the model, which is stronger than declining the "
                    "call: an invisible tool cannot be talked into being wanted."
                )
                st.markdown("**Verified — the physics, run again**")
                st.caption(
                    "Always available and never switched off: solving a second chain "
                    "length, and sweeping the field. Both go through the same "
                    "cross-check, so neither can produce an unchecked number."
                )

                st.markdown("**After the answer**")
                suggest_followups = st.checkbox(
                    "Suggest what to ask next",
                    value=True,
                    help=(
                        "One short model call per answer, proposing follow-up questions "
                        "from what the run established. Each suggestion is screened as if "
                        "you had typed it. Switched off, the agent falls back to a "
                        "composed list rather than showing nothing."
                    ),
                )

                st.markdown("**Unverified — outside this machine**")
                use_arxiv = st.checkbox(
                    "arXiv — papers, by keyword, title or author",
                    value=True,
                    help=(
                        "Searches the physics categories only. An author's name is "
                        "matched in every spelling arXiv indexes, so “Kadowaki”, "
                        "“Tadashi Kadowaki” and “Kadowaki, Tadashi” find the same person."
                    ),
                )
                max_papers = st.slider(
                    "Papers per search",
                    min_value=1,
                    max_value=MAX_PAPERS,
                    value=3,
                    disabled=not use_arxiv,
                )
                use_wikipedia = st.checkbox(
                    "Wikipedia — background, shallow or deep",
                    value=True,
                    help=(
                        "The opening definition by default; the whole article and its "
                        "section list when the agent asks for a deep read. Background, "
                        "never evidence for a number."
                    ),
                )
                web_ready = websearch.is_configured()
                use_web = st.checkbox(
                    "Open web search",
                    value=False,
                    disabled=not web_ready,
                    help=(
                        "The least trustworthy source here: an arbitrary page, no review "
                        f"of any kind. Needs {websearch.ENDPOINT_VARIABLE} and "
                        f"{websearch.KEY_VARIABLE} in the environment."
                    ),
                )
                mcp_ready = mcp.is_configured()
                use_mcp = st.checkbox(
                    "Tools on a connected MCP server",
                    value=False,
                    disabled=not mcp_ready,
                    help=(
                        "Borrows whatever tools the server advertises, over MCP's "
                        f"JSON-RPC transport. Needs {mcp.ENDPOINT_VARIABLE} in the "
                        "environment. Output is a third party's and is not verified."
                    ),
                )
                if not web_ready or not mcp_ready:
                    missing = [
                        name
                        for name, ready in (("web search", web_ready), ("MCP", mcp_ready))
                        if not ready
                    ]
                    st.caption(
                        f"Unavailable here: {', '.join(missing)} — nothing is configured to "
                        "reach, so the tool is withheld rather than offered and failed."
                    )

        knob = Setting(
            physics=PhysicsSetting(
                n_sites=n_sites,
                coupling=coupling,
                field=field,
                boundary=boundary,  # type: ignore[arg-type]
                sweep_points=sweep_points,
                max_sites_for_costly_sweep=max_sites_costly,
            ),
            model=ModelSetting(
                # The slider's options are exactly the Audience literals, which a
                # widget return type cannot express.
                audience=audience,  # type: ignore[arg-type]
                chat_model=custom_model.strip() or chat_model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                max_model_calls_per_run=max_model_calls,
                max_retries=max_retries,
                request_timeout_s=request_timeout,
                requests_per_second=requests_per_second,
            ),
            retrieval=RetrievalSetting(
                passages=passages,
                vector_share=vector_share,
                rounds=rounds,
                shelf=shelf,
            ),
            tools=ToolSetting(
                suggest_followups=suggest_followups,
                use_arxiv=use_arxiv,
                use_wikipedia=use_wikipedia,
                use_web=use_web,
                use_mcp=use_mcp,
                max_papers=max_papers,
            ),
        )

    # What moved is now shown on the expander's label and inside it, drawn above --
    # nothing is left to say under the knob.
    #
    # The explanation of the chain-length cap used to sit here, under the knob. It
    # answers a question about the *physics*, which nobody is asking while adjusting
    # a slider, so it moved to the Lab page where the chain is the subject. See
    # `render_chain`.
    return knob


def current_setting() -> Setting:
    """The knob for this page, drawing the sidebar once per run.

    The entry script draws the knob and every page reads it from here, so the
    sidebar is the same on all of them. A page opened on its own -- which is how
    the tests exercise them -- draws its own knob instead of failing, so no page
    depends on having been reached through the navigation.

    Returns:
        The knob's position.
    """
    # Read, never written: the entry script rewrites this key on every rerun, and
    # a page that cached its own copy would keep answering with the position the
    # knob was in when it first loaded.
    existing = st.session_state.get("setting")
    return existing if isinstance(existing, Setting) else read_settings_knob()


def chat_settings(base: Setting) -> Setting:
    """Draw the chat page's own settings and return what to ask with.

    A second settings control, deliberately: the sidebar knob is where a session
    is configured, and this is where the two or three dials worth changing *mid
    conversation* live, without scrolling past sampling temperature to reach them.

    Precedence is explicit rather than implied. Left alone it follows the sidebar
    exactly, and the caption says so; tick the box and this page wins for the
    questions asked from it. The alternative -- two sets of live widgets for the
    same value -- has a rule nobody can see, and the loser silently discards a
    dial the user just moved.

    Args:
        base: The sidebar knob, which this may override.

    Returns:
        The setting to ask with: ``base`` itself, or a copy with the overridden
        fields replaced.
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
                f"Following the sidebar: **{level}**, {base.physics.n_sites} sites, "
                f"J = {base.physics.coupling:g}, h = {base.physics.field:g}, "
                rf"{base.retrieval.passages} passages, "
                rf"$\alpha$ = {base.retrieval.vector_share:.1f}."
            )
            return base

        # Two columns, and the split is the project's central claim laid out as a
        # layout: everything on the left changes how the answer is *written* and
        # what it is grounded in, everything on the right changes the problem being
        # *solved*. Drag anything on the left and the number below does not move.
        # The developer knobs sit behind an expander, so the dials that shape this
        # answer are not competing for attention with the ones that only matter
        # while something is being tested.
        ai_column, physics_column = st.columns(2)

        with ai_column:
            st.markdown("**Model and retrieval**")
            # Worth having here rather than only in the sidebar: while a feature is
            # being tested, every question is a cost, and the cheap model answers
            # "does the wiring work" just as well as the expensive one. The list is
            # the sidebar's, plus whatever the sidebar is already set to -- a slug
            # typed by hand there must not vanish when this page is opened.
            model_options = list(dict.fromkeys([base.model.chat_model, *MODEL_CHOICES]))
            chat_model = st.selectbox(
                "Model",
                options=model_options,
                index=0,
                key="chat_model",
                help="Switch to a cheaper model while testing; the sidebar's choice is first.",
            )
            audience = st.select_slider(
                "Explain it for",
                options=list(AUDIENCE_LABELS),
                value=base.model.audience,
                format_func=lambda level: AUDIENCE_LABELS[level],
                key="chat_audience",
            )
            passages = st.slider(
                "Passages to use",
                min_value=1,
                max_value=MAX_PASSAGES,
                value=base.retrieval.passages,
                key="chat_passages",
                help="The most one answer may cite from the notes.",
            )
            vector_share = st.slider(
                VECTOR_SHARE_LABEL,
                min_value=0.0,
                max_value=1.0,
                value=base.retrieval.vector_share,
                step=0.1,
                key="chat_vector_share",
                help=(
                    "The vector half's weight: 1.0 is vector search alone; 0.0 is keyword "
                    "search alone. Worth moving mid-conversation, because which half "
                    "answers depends on what you just typed."
                ),
            )
            st.caption(search_note(vector_share))
            rounds = st.slider(
                "Searches per question",
                min_value=1,
                max_value=MAX_ROUNDS,
                value=base.retrieval.rounds,
                key="chat_rounds",
                help="Two lets a failed search be reformulated and tried again.",
            )
            shelf = st.selectbox(
                "Knowledge base",
                options=list(dict.fromkeys([base.retrieval.shelf, "", *shelf_names()])),
                index=0,
                format_func=lambda name: SHELF_CHOICE_LABELS.get(name, name),
                key="chat_shelf",
                help="Empty leaves the choice to the router, which is usually right.",
            )
            use_arxiv = st.checkbox(
                "Let it search arXiv",
                value=base.tools.use_arxiv,
                key="chat_arxiv",
                help="Only reached when the notes come back with nothing usable.",
            )

            with st.expander("Advanced: model tuning"):
                # Every bound matches the knob's, so a limit met here is the limit
                # Settings would enforce -- see read_settings_knob.
                temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=2.0,
                    value=base.model.temperature,
                    step=0.1,
                    key="chat_temperature",
                    help="0 is repeatable, which is what a demonstration wants.",
                )
                # The tick box is here for the same reason it is in the sidebar,
                # and leaving it out was a silent trap. The sidebar can set no cap
                # at all, which arrives here as ``None``; a slider handed ``None``
                # opens at its minimum, so turning the cap *off* upstairs and then
                # ticking "set them here" turned it back on at 256 tokens -- the
                # tightest setting on the dial -- and the answer came back cut off
                # mid-sentence with nothing on screen to explain it.
                cap_length = st.checkbox(
                    "Cap the reply length",
                    value=base.model.max_output_tokens is not None,
                    key="chat_cap_length",
                    help="Unticked, a reply runs to the model's own limit.",
                )
                max_output_tokens = (
                    st.slider(
                        "Max output tokens",
                        min_value=256,
                        max_value=8192,
                        value=base.model.max_output_tokens or DEFAULT_MAX_OUTPUT_TOKENS,
                        step=256,
                        key="chat_max_tokens",
                    )
                    if cap_length
                    else None
                )
                max_model_calls = st.slider(
                    "Model calls per question",
                    min_value=1,
                    max_value=100,
                    value=base.model.max_model_calls_per_run,
                    key="chat_max_calls",
                    help="The ceiling that turns a runaway loop into a stopped run.",
                )
                max_retries = st.slider(
                    "Retries on transient failure",
                    min_value=0,
                    max_value=10,
                    value=base.model.max_retries,
                    key="chat_max_retries",
                )

        with physics_column:
            st.markdown("**Physics**")
            n_sites = st.slider(
                "Sites $L$",
                min_value=2,
                max_value=MAX_SITES_STATEVECTOR,
                value=base.physics.n_sites,
                key="chat_sites",
                help="Every added site doubles the state vector the page diagonalises.",
            )
            coupling = st.slider(
                "Coupling $J$",
                min_value=0.1,
                max_value=2.0,
                value=base.physics.coupling,
                step=0.1,
                key="chat_coupling",
            )
            field = st.slider(
                "Field $h$",
                min_value=0.0,
                max_value=2.0,
                value=base.physics.field,
                step=FIELD_STEP,
                key="chat_field",
            )
            st.caption(
                f"$h/J$ = {base.physics.field / base.physics.coupling:.2f} at the sidebar's "
                "setting; the critical point of the infinite chain sits at 1."
            )

    # Constructed rather than copied with an update, so every value is validated
    # on the way in -- the same reason src.tools.chains builds a fresh spec.
    physics = base.physics.model_dump() | {
        "n_sites": n_sites,
        "coupling": coupling,
        "field": field,
    }
    model = base.model.model_dump() | {
        "audience": audience,
        "chat_model": chat_model,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "max_model_calls_per_run": max_model_calls,
        "max_retries": max_retries,
    }
    retrieval = base.retrieval.model_dump() | {
        "passages": passages,
        "vector_share": vector_share,
        "rounds": rounds,
        "shelf": shelf,
    }
    tools = base.tools.model_dump() | {"use_arxiv": use_arxiv}
    return Setting(
        physics=PhysicsSetting(**physics),
        model=ModelSetting(**model),
        retrieval=RetrievalSetting(**retrieval),
        tools=ToolSetting(**tools),
    )


def render_cost(usage: Usage) -> None:
    """Show what one answer spent on model calls.

    Displayed with every answer rather than on a page of its own, because the
    number that changes behaviour is the one attached to the thing that cost it.

    Args:
        usage: What the run spent.
    """
    st.caption(usage.summary())


def session_usage() -> Usage:
    """What this whole conversation has spent so far.

    Returns:
        The merged usage of every answer in the thread. Computed from the history
        rather than accumulated in a counter, so it cannot drift out of step with
        what is on screen.
    """
    history: list[graph.Answer] = st.session_state.get("history", [])
    total = Usage()
    for past in history:
        total = total.merge(past.usage)
    return total


def render_session_cost() -> None:
    """Show the conversation's running total in one line.

    The line stays in the sidebar because a running cost is only useful where it is
    unavoidable; the breakdown, the charts and the per-question table live on the
    Analytics page. One number in two places is one number that will disagree with
    itself, so this is deliberately the *only* copy of the total.
    """
    total = session_usage()
    st.caption(f"**This session:** {total.summary()}")
    if total.calls:
        st.caption("Full breakdown on the 📊 Analytics page.")


def bars_figure(labels: list[str], values: list[float], *, xlabel: str) -> Figure:
    """Draw one horizontal bar chart in the page's own style.

    Horizontal rather than vertical, for one reason that matters here: the
    categories are words -- ``compute_and_retrieve``, ``screen``, ``compose`` -- and
    vertical bars would either rotate them or truncate them.

    Args:
        labels: Category names, drawn top to bottom in the order given.
        values: One value per label.
        xlabel: What the axis measures.

    Returns:
        The figure, ready for :func:`streamlit.pyplot`.
    """
    figure = Figure(figsize=(6.0, 0.45 * max(len(labels), 1) + 1.1), dpi=160)
    axes = figure.add_subplot(111)
    positions = range(len(labels))
    axes.barh(list(positions), values, color=ACCENT, height=0.6)
    axes.set_yticks(list(positions))
    axes.set_yticklabels(labels, fontsize=9)
    axes.invert_yaxis()  # first label at the top, where a reader starts
    axes.set_xlabel(xlabel, fontsize=9, color=MUTED)
    style_axes(axes)
    for position, value in zip(positions, values, strict=True):
        axes.text(
            value,
            position,
            f"  {value:,.0f}" if value >= 1 else f"  {value:.6f}",
            va="center",
            fontsize=8,
            color=MUTED,
        )
    figure.tight_layout()
    return figure


def route_counts() -> dict[str, int]:
    """Count how this session's questions were routed.

    Returns:
        Route name to how many questions took it, commonest first. This is the
        session's own evidence that the pipeline is a decision rather than a fixed
        sequence: a conversation whose questions all took different routes ran
        different nodes each time.
    """
    history: list[graph.Answer] = st.session_state.get("history", [])
    counts: dict[str, int] = {}
    for past in history:
        name = past.routing.route if past.routing is not None else "blocked by the guard"
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items(), key=lambda pair: pair[1], reverse=True))


def node_counts() -> dict[str, int]:
    """Count how often each graph node ran this session.

    Returns:
        Node name to how many runs visited it, in the graph's own order. Read off
        each answer's fields by :func:`src.agent.graph.executed_nodes`, so the
        chart cannot claim a node the run did not reach.
    """
    history: list[graph.Answer] = st.session_state.get("history", [])
    # Read off the compiled graph rather than listed here, for the reason the
    # diagram on the pipeline page is generated rather than drawn: a hand-kept list
    # goes stale silently. This one had, and the symptom was subtle -- an unlisted
    # node was still counted, because the loop below adds keys it has not seen, but
    # it arrived at the end of the dict and so was charted out of order.
    order = list(graph.pipeline_nodes())
    counts = dict.fromkeys(order, 0)
    for past in history:
        for node in graph.executed_nodes(past):
            counts[node] = counts.get(node, 0) + 1
    return {node: count for node, count in counts.items() if count}


def render_analytics() -> None:
    """Draw the Analytics page: what the session did, and what it cost.

    Four questions, in the order a reader asks them: how much was spent, on which
    step, which routes the questions took, and which nodes ran. The last two are
    the ones worth the page -- they are the session's own evidence that the graph
    branches, and they come from the answers themselves rather than from a counter
    that could drift out of step with them.
    """
    history: list[graph.Answer] = st.session_state.get("history", [])
    total = session_usage()
    if not history:
        st.info("Ask something on the Chat page and this fills in.")
        return

    verified = sum(1 for past in history if past.is_verified)
    refused = sum(1 for past in history if past.status == "refused")
    columns = st.columns(4)
    columns[0].metric("Questions", len(history))
    columns[1].metric("Verified answers", verified)
    columns[2].metric("Declined", refused)
    columns[3].metric("Tokens", f"{total.total_tokens:,}")

    st.caption(
        f"**Estimated spend:** {total.summary()} — prices are the gateway's published "
        "rates, copied by hand and rounded. An estimate, not an invoice."
    )
    st.divider()

    steps = total.by_purpose()
    if steps:
        st.markdown("**Tokens by pipeline step**")
        st.pyplot(
            bars_figure(
                [step.purpose for step in steps],
                [float(step.total_tokens) for step in steps],
                xlabel="tokens",
            ),
            width="stretch",
        )
        st.caption(
            "Per step rather than per call, because the question a reader has is not "
            "*how many calls* but *what was the money spent on*."
        )

    left, right = st.columns(2)
    with left:
        routes = route_counts()
        st.markdown("**How the questions were routed**")
        st.pyplot(
            bars_figure(
                list(routes),
                [float(count) for count in routes.values()],
                xlabel="questions",
            ),
            width="stretch",
        )
    with right:
        nodes = node_counts()
        st.markdown("**Which nodes actually ran**")
        st.pyplot(
            bars_figure(
                list(nodes),
                [float(count) for count in nodes.values()],
                xlabel="runs that reached it",
            ),
            width="stretch",
        )
    st.caption(
        "Unequal bars on the right are the point: `plan` and `solve` run only when a "
        "question needs a number, and `search` only when it needs the notes. A fixed "
        "pipeline would draw these all the same height."
    )

    st.divider()
    st.markdown("**Question by question**")
    st.dataframe(
        [
            {
                "question": past.question,
                "status": past.status,
                "route": past.routing.route if past.routing is not None else "—",
                "verified": past.is_verified,
                "tokens": past.usage.total_tokens,
                "cost (USD)": round(past.usage.cost_usd, 6),
            }
            for past in history
        ],
        hide_index=True,
        width="stretch",
    )
    if steps:
        st.dataframe(
            [
                {
                    "step": step.purpose,
                    "calls": step.calls,
                    "tokens": step.total_tokens,
                    "cost (USD)": round(step.cost_usd, 6),
                }
                for step in steps
            ],
            hide_index=True,
            width="stretch",
        )


def render_evaluation() -> None:
    """Draw the Evaluation page from the last scorecard written to disk.

    Read from :data:`src.evals.run.SUMMARY_PATH` rather than by running the suite:
    a page that ran an evaluation on render would fire twenty agent runs at whoever
    clicked the tab, and would report a different number every time somebody moved a
    slider. The suite is a command (``make evals``); this is its published result.

    The page says when it was measured and what answered the questions, because a
    scorecard whose provenance is unstated is a number a reader has to take on
    trust -- which is the thing this whole project is arguing against.
    """
    summary = Path(evals_run.SUMMARY_PATH)
    if not summary.exists():
        st.info(
            "No scorecard yet. Run `make evals` to score the held-out suite and write "
            "one — it answers every case with the real agent, so it makes live model "
            "calls and needs the gateway credential."
        )
        return
    try:
        report = json.loads(summary.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        st.error("The scorecard on disk could not be read. Run `make evals` to rewrite it.")
        return

    passed = int(report.get("passed", 0))
    total = int(report.get("total", 0))
    rate = float(report.get("rate", 0.0))
    columns = st.columns(3)
    columns[0].metric("Cases passed", f"{passed}/{total}")
    columns[1].metric("Pass rate", f"{rate:.0%}")
    columns[2].metric("Model calls", report.get("model_calls", 0))
    st.caption(str(report.get("provenance", "")))
    for note in report.get("notes", []):
        st.caption(f"— {note}")

    families = report.get("families", [])
    if families:
        st.markdown("**What each family of cases defends**")
        st.pyplot(
            bars_figure(
                [str(entry["family"]) for entry in families],
                [
                    100.0 * int(entry["passed"]) / int(entry["total"]) if entry["total"] else 0.0
                    for entry in families
                ],
                xlabel="% of cases passed",
            ),
            width="stretch",
        )

    st.dataframe(
        [
            {
                "case": entry.get("name"),
                "defends": entry.get("family"),
                "result": "pass" if entry.get("passed") else "FAIL",
                "ended": entry.get("status"),
                "path": " → ".join(entry.get("nodes", [])),
            }
            for entry in report.get("cases", [])
        ],
        hide_index=True,
        width="stretch",
    )
    failures = [entry for entry in report.get("cases", []) if not entry.get("passed")]
    for entry in failures:
        st.warning(f"**{entry.get('name')}** — " + "; ".join(entry.get("failures", [])))

    written = Path(evals_run.REPORT_PATH)
    if written.exists():
        with st.expander("The full report, as `make evals` wrote it"):
            st.markdown(written.read_text(encoding="utf-8"))


def last_answer() -> graph.Answer | None:
    """The most recent answer in this session, or ``None``.

    Returns:
        The last entry in the chat thread. Pages other than the chat read the
        conversation through here rather than reaching into session state, so how
        the thread is stored stays one module's business.
    """
    history: list[graph.Answer] = st.session_state.get("history", [])
    return history[-1] if history else None


def render_chain(setting: Setting) -> None:
    """Show everything deterministic about the chain in the knob.

    No model and no credential are involved anywhere in here, which is why this
    is a page of its own: it is the part of the application that cannot be wrong
    because of a language model, and it stays useful when there is none.

    Args:
        setting: The knob position to solve.
    """
    spec = setting.physics.spec()
    check = solve(spec.n_sites, spec.coupling, spec.field, spec.boundary)
    ratio = spec.ratio

    if check.energy is None:
        st.error("No available method can solve this specification.")

    # The four metrics that used to sit here -- E0, E0/L, g and the regime -- are all
    # in the verdict line below and in the Report tab, and repeating them cost the
    # page a screenful before the first curve. The verdict is what a reader cannot
    # get anywhere else, so it goes first.
    render_verdict(check)
    # Moved here from under the settings knob: it answers a question about the
    # physics, and this is the page where the chain itself is the subject.
    st.caption(
        f"**Why L ≤ {MAX_SITES_STATEVECTOR}?** Exact diagonalisation stores a "
        r"$2^L$ state vector; every extra site quadruples the memory. The cap is a "
        "refusal, not a clamp — ask for more and the method declines and says why."
    )

    # Ordered by what the reader is looking at, not by what is cheap to compute:
    # the algebra that says why any of this is quantum at all, then the ground-state
    # properties, the excitations, the derivatives that show the transition, the
    # finite-size story, and last the machinery.
    (
        quantum_tab,
        curves_tab,
        spectrum_tab,
        response_tab,
        scaling_tab,
        methods_tab,
        report_tab,
    ) = st.tabs(
        [
            "What makes it quantum",
            "Magnetisation and energy",
            "Excitation spectrum",
            "Energy and its derivatives",
            "Finite-size scaling",
            "Methods",
            "Report",
        ]
    )
    # Sampled finely whatever the sweep slider says, and the slider's limit is not
    # the relevant cost here. Measured on this machine at 121 points: the spectrum
    # takes 0.9 ms and the derivatives 3.3 ms, because both are closed-form and O(L)
    # per point. The sweep in the first tab takes 113 ms at 21 points, since every
    # one of those needs a diagonalisation -- which is what the slider exists to
    # limit. Coarsening these two would save about three milliseconds and cost a
    # visibly jagged curve, and the curvature panel is exactly where a coarse grid
    # invents structure that is not in the physics.
    fine = MAX_POINTS

    # The user's own ceiling, read once and used for both halves of the decision:
    # whether diagonalisation joins the sweep at all, and -- because that is what
    # makes each point expensive -- whether the resolution has to be clipped. These
    # were two different numbers until now. The knob set one of them and the module
    # default set the other, so the slider moved and nothing on the page did.
    ceiling = setting.physics.max_sites_for_costly_sweep

    with quantum_tab:
        render_quantumness(spec)

    with curves_tab:
        render_curves(
            field_sweep(
                spec.n_sites,
                spec.coupling,
                spec.boundary,
                setting.physics.points_for(costly=spec.n_sites <= ceiling),
                max_costly_sites=ceiling,
            ),
            marker=ratio,
        )

    with spectrum_tab:
        # The many-body levels first, and measured from the ground state. This is the
        # figure a lecture note draws for a short chain, and it is the same function
        # the chat calls -- the Lab page used to show only the dispersion here, which
        # is a different quantity: eps(k) is the cost of one quasiparticle, while
        # E_n - E_0 is the energy of a state, and states of the even sector are built
        # from *pairs* of quasiparticles. At zero field that is 4J against 2J, so the
        # two plots disagree by a factor of two and neither is wrong.
        render_spectrum(
            field_sweep(
                spec.n_sites,
                spec.coupling,
                spec.boundary,
                setting.physics.points_for(costly=spec.n_sites <= ceiling),
                "spectrum",
                max_costly_sites=ceiling,
            ),
            marker=ratio,
        )
        st.divider()
        st.markdown("**Where those levels come from**")
        render_momentum_spectrum(spec, fine)

    with response_tab:
        render_energy_response(spec, fine)

    with scaling_tab:
        if spec.boundary == "periodic" and spec.n_sites % 2 == 0:
            render_scaling(spec)
        else:
            st.info(
                "Finite-size scaling is shown for even rings, the case the closed-form "
                "solution covers exactly. This chain is not one, so the sequence would "
                "mix methods of different standing."
            )

    with methods_tab:
        if check.results:
            st.dataframe(
                {
                    "method": [result.method for result in check.results],
                    "E₀": [f"{result.energy:.12f}" for result in check.results],
                },
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "The free-fermion solution does $O(L)$ arithmetic over momenta; exact "
                "diagonalisation builds a $2^L$ matrix and iterates. They share no code, "
                "so agreement between them is evidence rather than a tautology."
            )
        else:
            st.write("No method accepted this specification.")

        if check.rejected:
            st.markdown("**Unavailable here**")
            for item in check.rejected:
                st.markdown(f"- `{item.method.name}` — {item.reason}")

    with report_tab:
        st.code(check.summary(), language="text")
        st.caption("The same text the agent logs and quotes when it justifies an answer.")


@st.cache_data(show_spinner=False)
def corpus_notes() -> list[dict[str, str]]:
    """Read the committed corpus for the knowledge-base page.

    Returns:
        One row per note: title, source, topics, and its size in words. Empty
        when the corpus cannot be read at all, which the page reports rather than
        raising -- a missing corpus is a deployment mistake worth a sentence, not
        a traceback. It is also worth a log line: the page says "no notes", and
        whether that means an empty directory or an unreadable one is a question
        only the logs can answer.
    """
    try:
        notes = load_corpus()
    except Exception as error:
        LOG.warning(
            "corpus_unreadable",
            extra={"error_type": type(error).__name__, "detail": "knowledge-base page is empty"},
        )
        return []
    return [
        {
            "note": note.slug,
            "shelf": note.shelf,
            "title": note.title,
            "source": note.source,
            "topics": ", ".join(note.topics),
            "words": str(len(note.body.split())),
        }
        for note in notes
    ]


def render_knowledge() -> None:
    """Show what the agent has read, and what it has not.

    The most common way a retrieval application misleads someone is by refusing a
    perfectly reasonable question, which reads as "this cannot be answered" when
    it means "this is not in my notes". Publishing the corpus turns that into a
    checkable statement: a visitor can see every note, and a refusal about a
    subject that is not there stops being mysterious.

    Grouped by shelf, because the shelf is a decision the agent makes per question
    and a reader cannot judge that decision without seeing what is on each one.
    """
    rows = corpus_notes()
    if not rows:
        st.error(
            "The corpus could not be read. It lives in `data/corpus/`, one directory "
            "per knowledge base, and is committed to the repository -- see "
            "`data/README.md`."
        )
        return

    topics = sorted({topic for row in rows for topic in row["topics"].split(", ") if topic})
    first, second, third, fourth = st.columns(4)
    first.metric("Knowledge bases", len({row["shelf"] for row in rows}))
    second.metric("Notes", len(rows))
    third.metric("Topics", len(topics))
    fourth.metric("Words", sum(int(row["words"]) for row in rows))

    for shelf in SHELVES:
        on_shelf = [row for row in rows if row["shelf"] == shelf.name]
        if not on_shelf:
            continue
        st.markdown(f"**{shelf.title}** — `{shelf.name}`")
        st.caption(f"Covers {shelf.covers}.")
        st.dataframe(
            [{key: value for key, value in row.items() if key != "shelf"} for row in on_shelf],
            hide_index=True,
            width="stretch",
        )

    unshelved = [row for row in rows if not get_shelf(row["shelf"])]
    if unshelved:
        st.warning(
            f"{len(unshelved)} notes are in a directory no Shelf declares, so nothing "
            "searches them. See `src/rag/ingest.py`."
        )

    st.caption(
        "Every note carries a citation precise enough to check the claim against the "
        "original, and no note contains a computed number -- a number sitting in the "
        "corpus would be retrievable and indistinguishable from a verified one. The "
        "agent chooses which base to search per question, and widens to both if the "
        "first comes back empty."
    )

    st.markdown("**Topics the agent can be steered by**")
    st.markdown(" ".join(f"`{topic}`" for topic in topics))

    with st.expander("What happens to a note before it can be retrieved"):
        st.markdown(
            "1. **Parsed** -- a note without a title, a source and a topic is refused at "
            "ingest rather than indexed unciteable.\n"
            "2. **Split on headings**, so a retrieved passage is a section with a heading "
            "trail and can be attributed to a place, not just to a file.\n"
            "3. **Embedded** and written to Chroma under an id derived from the note and "
            "the position, so re-ingesting updates a chunk instead of duplicating it.\n"
            "4. **Reconciled** -- chunks the corpus no longer produces are deleted, "
            "because a shortened note would otherwise keep answering from its old tail.\n\n"
            "Run `make ingest` after changing anything here. Until then the index is what "
            "the agent searches, not the files above."
        )


LEGEND_HEIGHT = 56
"""Vertical room the colour key needs inside the diagram's frame.

Two lines' worth: the entries wrap on a narrow window, and a legend clipped by
the frame it explains is worse than none.
"""

LEGEND_LABELS = (
    ("visited", "ran for this question"),
    ("default", "in the graph, not run this time"),
    ("last", "where the run ended"),
)
"""Mermaid class names and what each colour means, in reading order.

Paired with :func:`src.agent.graph.mermaid_fills`, which supplies the colours
themselves. Only the meanings are written here -- a legend that also named the
shades would be a copy of the diagram's palette, and the two would part company
without anything failing.
"""


def legend_html(fills: dict[str, str]) -> str:
    """Build the diagram's colour and line key.

    Two colours had been on screen with nothing saying which was which, and a
    reader cannot tell "did not run" from "ran" by looking. The line styles need
    the same treatment for a stronger reason: the dashed edges are the branch
    points, so they are where the routing decisions actually are.

    Args:
        fills: Class name to colour, from :func:`src.agent.graph.mermaid_fills`.

    Returns:
        An HTML fragment. Classes the diagram did not declare are skipped rather
        than shown in a default colour -- with nothing highlighted there is no
        ``visited`` shade on screen, and inventing one would explain a colour the
        reader cannot see.
    """
    swatch = (
        "display:inline-block;width:14px;height:14px;border-radius:3px;"
        "border:1px solid #adb5bd;background:{colour}"
    )
    entry = (
        '<span style="display:inline-flex;align-items:center;gap:7px;'
        'margin:0 18px 6px 0;white-space:nowrap">{mark}{label}</span>'
    )
    parts = [
        entry.format(mark=f'<span style="{swatch.format(colour=fills[name])}"></span>', label=label)
        for name, label in LEGEND_LABELS
        if name in fills
    ]
    # Solid and dashed edges are Mermaid's own, so they are drawn rather than
    # coloured from the source: what matters is that a dashed edge is a decision.
    lines = (
        ("6,0", "always taken"),
        ("5,4", "a decision picks the branch"),
    )
    parts += [
        entry.format(
            mark=(
                '<svg width="26" height="10" style="flex:none">'
                f'<line x1="0" y1="5" x2="26" y2="5" stroke="#495057" stroke-width="2" '
                f'stroke-dasharray="{dash}"/></svg>'
            ),
            label=label,
        )
        for dash, label in lines
    ]
    return (
        '<div style="font:13px/1.5 -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;'
        'color:#343a40;margin:4px 0 10px">' + "".join(parts) + "</div>"
    )


def render_pipeline(answer: graph.Answer | None) -> None:
    """Draw the agent's graph, lighting up the path the last question took.

    Args:
        answer: The most recent answer, or ``None`` if nothing has been asked.
    """
    visited = graph.executed_nodes(answer) if answer is not None else ()
    mermaid = graph.pipeline_mermaid(visited)
    st.markdown(
        "**The pipeline, drawn by LangGraph from the compiled graph.** "
        "It is generated rather than maintained, so it cannot describe a pipeline "
        "other than the one that runs."
    )
    if answer is not None:
        st.caption(
            f"Highlighted: the {len(visited)} nodes your last question actually went "
            "through. Which nodes run is decided per question -- that is what makes this "
            "an agent rather than a fixed pipeline."
        )
    else:
        st.caption("Ask something on the chat page and the path it took will be marked here.")

    html = (
        legend_html(graph.mermaid_fills(mermaid)) + '<div class="mermaid" id="pipeline"></div>'
        '<script type="module">'
        "import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';"
        f"const source = {json.dumps(mermaid)};"
        "const element = document.getElementById('pipeline');"
        "element.textContent = source;"
        "mermaid.initialize({ startOnLoad: false, securityLevel: 'loose' });"
        "mermaid.run({ nodes: [element] }).catch(() => "
        "{ element.textContent = 'The diagram did not render; the source is below.'; });"
        "</script>"
    )
    # Mermaid is fetched by the browser from a CDN, not by this process. With no
    # network the diagram degrades to its source, which is why the source is
    # always offered rather than hidden behind a rendering that might not happen.
    #
    # The height follows the graph rather than being a constant. A fixed frame was
    # right for seven nodes and clipped the diagram the moment recall and remember
    # were added -- and a clipped diagram reads as a missing one, which is the worst
    # outcome for the page whose whole job is to show the path.
    # Counted from the source rather than hard-coded, so the frame grows with the
    # graph on its own: a node declaration is a line with a shape on it and no arrow.
    nodes = sum(
        1
        for line in mermaid.splitlines()
        if "(" in line and "-->" not in line and "-.->" not in line
    )
    # The legend sits inside the frame, so the frame has to be taller by its height.
    st.iframe(html, height=max(520, 120 + 52 * nodes) + LEGEND_HEIGHT)
    with st.expander("Diagram source (renders on any Mermaid viewer, readable as text)"):
        st.code(mermaid, language="text")

    if answer is None:
        return
    render_retrieval_trace(answer)
    render_call_trace(answer)


def retrieval_rows(retrieval: Retrieval) -> list[dict[str, str]]:
    """Tabulate the passages a search kept, with the scores behind them.

    Every score the two halves produced, side by side, because a reader comparing
    them learns how the retriever actually behaves.
    :func:`src.rag.retrieve.heuristic_grade` keeps a passage that scores highly
    *or* one that shares enough vocabulary with the question, so a row with a low
    score and high overlap was kept lexically, and a row with the reverse was kept
    by the embedding -- which is the paraphrase the embedding exists to catch.

    The three numbers are not on one scale and are deliberately not blended into
    one: *vector* is a cosine relevance in ``[0, 1]``, *keyword* is an unbounded
    BM25 score, and *fused* is the reciprocal-rank score the ordering used. A
    zero in either of the first two means that half did not return the passage at
    all, which is the most informative cell in the table.

    Args:
        retrieval: What retrieval produced.

    Returns:
        One row per kept passage, best first.
    """
    asked = content_words(retrieval.question)
    rows: list[dict[str, str]] = []
    for passage in retrieval.passages:
        shared = sorted(asked & content_words(passage.citable_text))
        rows.append(
            {
                "note": passage.document,
                "shelf": passage.shelf or "(pre-shelf index)",
                "section": passage.section,
                "vector score": f"{passage.score:.4f}",
                "keyword score": f"{passage.lexical_score:.2f}",
                "fused": f"{passage.fused:.4f}",
                "shared words": str(len(shared)),
                "matched on": ", ".join(shared[:6]),
            }
        )
    return rows


AUTHOR_NAMES = {
    "grader": "the relevance grader, which had just read the rejected passages",
    "model": "the rewriter, from the question and the passages that failed",
    "heuristic": "the deterministic rewrite — keep the subject, add the model's name",
}
"""How to name whoever wrote a reformulated query, keyed by ``Attempt.rewritten_by``.

Named rather than numbered because "written by the model" and "written by the
fallback because no model was available" are different facts about a run, and the
second one is the one to know before reading the query itself.
"""


def render_query_trace(retrieval: Retrieval) -> None:
    """Show the question as asked beside the query that was actually searched.

    Both sides are always drawn, and that is the point. An earlier version showed
    the searched query only when a rewrite had happened, which meant the most
    common run -- the one where the first round succeeded -- displayed the
    question and then a sentence about a rewrite that did not occur. A reader
    could not tell from the screen whether the query had been left alone or
    merely not reported, and "the query was not altered" is a claim a trace has
    to make out loud rather than by omission.

    So the right-hand side names what went to the index in every case, and the
    caption underneath says who wrote it: the asker, a grader, a rewriter, or the
    deterministic floor.

    Args:
        retrieval: The run's retrieval, already known to have been attempted.
    """
    asked, searched = st.columns(2)
    with asked:
        st.markdown("**Original question**")
        st.code(retrieval.question, language=None)
    rewrite = retrieval.reformulation
    # The last round is what the answer rests on, and with no rewrite it is the
    # question itself. No attempts at all means the index could not be opened.
    last = retrieval.attempts[-1] if retrieval.attempts else None
    with searched:
        st.markdown("**Rewritten search query**" if rewrite else "**Query actually searched**")
        st.code(
            rewrite.query if rewrite else (last.query if last else retrieval.question),
            language=None,
        )
    if rewrite is not None:
        author = AUTHOR_NAMES.get(rewrite.rewritten_by, rewrite.rewritten_by)
        st.caption(
            f"The first search kept nothing, so the query was rewritten by {author}. "
            f"It found {rewrite.found} and kept {rewrite.kept}."
        )
    elif last is not None:
        st.caption(
            "Searched as asked — the first round found enough, so no reformulation was "
            f"needed, and the query above is the question verbatim. It found {last.found} "
            f"and kept {last.kept}."
        )
    else:
        st.caption(
            "No search was recorded: the index could not be opened, so the question "
            "above was never sent anywhere."
        )


def render_retrieval_trace(answer: graph.Answer) -> None:
    """Show what the corpus search looked at and what it kept.

    Args:
        answer: The finished run.
    """
    retrieval = answer.retrieval
    if retrieval is None or retrieval.outcome == "not_needed":
        return
    if retrieval.outcome == "store_unavailable":
        # Instead of the tables rather than above them: there are no rounds and no
        # scores to show, and the empty-state line below ("nothing was kept") would
        # describe a corpus that had nothing to say, which is not what happened.
        st.warning(
            "The knowledge base could not be opened, so the notes were never searched. "
            "Nothing here reflects what the corpus contains. Build the index with "
            "`make ingest`, then ask again."
        )
        return
    render_query_trace(retrieval)
    st.markdown("**Retrieved passages, and the scores that admitted them**")
    rows = retrieval_rows(retrieval)
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
    else:
        st.info("Nothing was kept, so the answer does not rest on the notes.")
    st.dataframe(
        [
            {
                "round": attempt.kind,
                "query": attempt.query,
                "written by": attempt.rewritten_by or "the user",
                "searched": attempt.where(),
                "found": str(attempt.found),
                "kept": str(attempt.kept),
                "grader said": attempt.reason,
            }
            for attempt in retrieval.attempts
        ],
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Only kept passages are listed above; the round table is where a rejection "
        "shows up. A restricted first round and an unrestricted second means the "
        "chosen knowledge base came back empty and the search widened."
    )


def render_call_trace(answer: graph.Answer) -> None:
    """Break the run's model calls down by step, with tokens, cost and latency.

    The summary under an answer gives one number. This gives the diagnosis: which
    step spent it, how long each call took, and whether any call failed -- a
    retried call is two calls and the provider charges for both.

    Args:
        answer: The finished run.
    """
    usage = answer.usage
    st.markdown("**Model calls this question made**")
    if not usage.calls:
        st.info(
            "No model was called. The guard, the router, the grader and the reply all "
            "have deterministic paths, so this is a complete answer rather than a "
            "degraded one."
        )
        return
    st.dataframe(
        [
            {
                "step": call.purpose,
                "model": call.model,
                "in": f"{call.prompt_tokens:,}",
                "out": f"{call.completion_tokens:,}",
                "cost": "unpriced" if call.cost_usd is None else f"${call.cost_usd:.6f}",
                "latency": "-" if call.latency_ms is None else f"{call.latency_ms / 1000:.2f}s",
                "outcome": call.outcome,
            }
            for call in usage.calls
        ],
        hide_index=True,
        width="stretch",
    )
    st.dataframe(
        [
            {
                "step": total.purpose,
                "calls": str(total.calls),
                "tokens": f"{total.total_tokens:,}",
                "cost": f"${total.cost_usd:.6f}",
            }
            for total in usage.by_purpose()
        ],
        hide_index=True,
        width="stretch",
    )
    st.caption(f"{usage.summary()} — prices are the gateway's published rates, copied by hand.")
