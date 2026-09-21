"""The words and marks four pages share, and the promises they make about them.

`src/ui/panels.py` is split so that its *content* -- the glossary, the page index,
the health snapshot, the vocabulary of ticks -- is data and pure functions, and the
Streamlit drawing on top is thin enough not to need testing. These tests hold the
content, which is the half that can be quietly wrong.

Two of them are unusual and are the reason the file exists. One holds every gloss
against every other term, because a glossary written for a reader with no physics
is worthless the moment one entry needs another entry to be understood. The other
holds the page index against the files on disk, because a navigation list
maintained by hand goes stale silently and a monitor that offers a page that is not
there is worse than one that offers nothing.
"""

from __future__ import annotations

import pytest

from src.agent import graph
from src.agent.state import (
    CampaignState,
    Diagnosis,
    FormalModel,
    Request,
    RunRecord,
    Verdict,
    new_campaign,
)
from src.physics.quantum.ansatz import AnsatzSpec
from src.ui import panels
from src.ui.status import PROJECT_ROOT

PAGE_DIRECTORY = PROJECT_ROOT / "src" / "ui" / "pages"


# --------------------------------------------------------------------------
# The glossary
# --------------------------------------------------------------------------


def test_the_glossary_is_not_empty_and_has_no_duplicates() -> None:
    words = [term.word for term in panels.GLOSSARY]
    assert words
    assert len(set(words)) == len(words)


@pytest.mark.parametrize("term", panels.GLOSSARY, ids=lambda t: t.word)
def test_every_term_carries_both_a_meaning_and_a_reason(term: panels.Term) -> None:
    # A definition tells somebody what a word means. The second field tells them why
    # it is on the page, which is the half that makes a glossary worth reading.
    assert term.gloss.strip()
    assert term.why.strip()
    assert term.gloss[0].isupper(), "a gloss is a sentence"


def test_nothing_is_explained_with_a_word_that_has_not_been_explained_yet() -> None:
    # The rule the glossary is written under, and the reason it is not alphabetical.
    # Leaning on an *earlier* entry is how a glossary should work -- "shot" is allowed
    # to say "circuit" because a reader reading downwards has already met one. Leaning
    # on a later entry is the failure: the reader hits a word they have been given no
    # way to understand, and an alphabetical list does that constantly.
    for position, term in enumerate(panels.GLOSSARY):
        later = {other.word for other in panels.GLOSSARY[position + 1 :]}
        borrowed = sorted(word for word in later if word in term.gloss.lower())
        assert not borrowed, (
            f"the gloss for {term.word!r} uses {borrowed}, which the reader has not "
            "reached yet -- move it earlier or rewrite the gloss"
        )


def test_the_glossary_covers_the_words_the_interface_actually_uses() -> None:
    # The terms a reader cannot avoid meeting. If one of these is dropped, a page is
    # using vocabulary it never introduced.
    words = {term.word for term in panels.GLOSSARY}
    for required in ("ground state", "shot", "depth", "fidelity", "baseline", "verdict"):
        assert required in words


# --------------------------------------------------------------------------
# The page index
# --------------------------------------------------------------------------


def test_every_indexed_page_exists_on_disk() -> None:
    # A navigation list maintained by hand goes stale silently, and a menu offering a
    # page that is not there reads as a fault in the reader's browser rather than in
    # the index.
    for page in panels.PAGES:
        assert (PROJECT_ROOT / "src" / "ui" / page.module).is_file(), page.module


def test_every_page_on_disk_is_in_the_index() -> None:
    # The other direction, which is the one that goes wrong quietly: a page added and
    # never listed is a page nobody can reach.
    indexed = {page.module.split("/")[-1] for page in panels.PAGES}
    on_disk = {path.name for path in PAGE_DIRECTORY.glob("*.py") if path.name != "__init__.py"}
    assert on_disk == indexed


def test_the_entry_point_is_not_a_page() -> None:
    # `app.py` runs the navigation rather than appearing in it, so it must not be
    # listed as a destination.
    assert "app.py" not in {page.module.split("/")[-1] for page in panels.PAGES}
    assert (PROJECT_ROOT / "src" / "ui" / "app.py").is_file()


def test_the_product_comes_first_and_nothing_about_the_project_is_a_page() -> None:
    # Somebody opening this should be able to use the thing before being shown how it
    # is assembled. The ordering was the other way round for most of this project's
    # life, which was honest about the work and wrong about the visitor.
    #
    # The About page went further than reordering: it is gone. It was the only entry
    # in the navigation that was not a question about an answer, so it competed with
    # the product for a slot while being the page least likely to be opened. Its
    # ethics statement is a tab on Evaluations and its developer section a tab on
    # Analytics -- both beside the thing that raises the question -- and its "what
    # this is" tab was deleted, because that is the README's job.
    assert panels.PAGES[0].module == "pages/chat.py"
    assert "pages/about.py" not in {page.module for page in panels.PAGES}


def test_no_developer_instrumentation_is_offered_as_a_page() -> None:
    # Runs, Environment and Build were a directory listing, a dependency table and a
    # module checklist. Not features -- instrumentation for whoever was building this,
    # and Build in particular put "modules built: 47 of 52" in front of the person
    # deciding whether to trust the thing. What a reader genuinely wants from them
    # is the About page's third tab, which says on opening that nothing in it is
    # needed to use the application.
    modules = {page.module for page in panels.PAGES}
    assert not modules & {"pages/runs.py", "pages/environment.py", "pages/build.py"}


def test_no_middle_group_is_a_heading_over_a_single_page() -> None:
    # A heading above a single page is a category nobody needed -- in the middle of a
    # menu. The first group is the one exception, and it is an exception on principle
    # rather than by accident: it is an anchor rather than a category, and it shares
    # its page's name because a product that arrives first should not need a label
    # above it to be found. Every other heading covers more than one page.
    for group in panels.groups()[1:]:
        assert len(panels.pages_in(group)) > 1, f"{group!r} is a heading over one page"


def test_the_one_anchor_group_holds_exactly_one_page() -> None:
    # The other half of the rule above. There used to be two single-page groups --
    # the product at the top and About at the bottom -- and the rule was written for
    # both. About is gone: it was the only entry in the navigation that was not a
    # question about an answer, so its ethics statement moved to a tab on
    # Evaluations and its developer section to a tab on Analytics. One anchor is
    # left, and if it grows a sibling it stops being one.
    first = panels.groups()[0]
    assert len(panels.pages_in(first)) == 1
    assert len(panels.pages_in(panels.groups()[-1])) > 1


@pytest.mark.parametrize("page", panels.PAGES, ids=lambda p: p.label)
def test_every_page_has_exactly_one_icon(page: panels.Page) -> None:
    # Emoji rather than monochrome glyphs because they carry colour: a reader learns
    # "the microscope one is the physics" and stops reading the labels.
    assert page.icon
    assert len(page.icon) <= 3, "one emoji, possibly with a variation selector"


@pytest.mark.parametrize("page", panels.PAGES, ids=lambda p: p.label)
def test_every_page_is_indexed_by_a_question(page: panels.Page) -> None:
    # A page index listing nouns tells somebody what exists. It does not tell them
    # which one they want, which is the only thing they came to the index for.
    assert page.question.endswith("?") or page.question.endswith(".")
    assert page.question.strip()


# --------------------------------------------------------------------------
# The vocabulary of marks
# --------------------------------------------------------------------------


def test_the_three_marks_are_distinct() -> None:
    assert len({panels.PRESENT, panels.ABSENT, panels.WARNING}) == 3


def test_a_tick_means_the_same_thing_everywhere() -> None:
    assert panels.tick(True) == panels.PRESENT
    assert panels.tick(False) == panels.ABSENT


def test_a_mark_is_shown_with_a_word_beside_it() -> None:
    # A mark alone is ambiguous the first time somebody sees it, and a legend at the
    # bottom of the page is a legend nobody scrolls to.
    assert panels.presence(True) == f"{panels.PRESENT} set"
    assert (
        panels.presence(False, "present", "not created yet") == f"{panels.ABSENT} not created yet"
    )


@pytest.mark.parametrize(
    ("size", "expected"),
    [(0, "0 B"), (512, "512 B"), (2048, "2.0 kB"), (5_242_880, "5.0 MB")],
)
def test_file_sizes_are_rendered_in_units_somebody_reads(size: int, expected: str) -> None:
    assert panels.humanise_bytes(size) == expected


def test_a_negative_size_is_shown_as_nothing_rather_than_as_a_bug() -> None:
    # A file listing is not the place to discover a bug in a stat call.
    assert panels.humanise_bytes(-1) == "0 B"


# --------------------------------------------------------------------------
# The health snapshot
# --------------------------------------------------------------------------


def test_the_snapshot_reports_the_same_build_progress_the_checklist_does() -> None:
    snapshot = panels.health()
    assert 0 < snapshot.built <= snapshot.planned
    assert 0.0 < snapshot.progress <= 1.0


def test_a_snapshot_is_taken_once_rather_than_read_three_times() -> None:
    # So that the sidebar and a test are looking at the same moment, rather than at
    # three reads that could disagree if a file appeared between them.
    first, second = panels.health(), panels.health()
    assert first == second


def test_readiness_is_exactly_whether_anything_is_missing() -> None:
    snapshot = panels.health()
    assert snapshot.ready == (not snapshot.missing)


def test_a_later_tier_can_want_more_than_an_earlier_one() -> None:
    # The point of tiering the check: a package needed only by later work is not a
    # problem yet, and a monitor that says it is will have been learned to ignore.
    assert len(panels.health(tier=1).missing) <= len(panels.health(tier=9).missing)


# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------


def test_the_name_never_travels_without_the_words_that_explain_it() -> None:
    # "Quay" is read as "kway" by roughly half of any audience, and somebody meeting
    # the application has about five seconds. The descriptor is what buys them back.
    assert panels.APP_NAME
    assert panels.APP_DESCRIPTOR
    assert panels.APP_NAME.lower() not in panels.APP_DESCRIPTOR.lower()


def test_the_icon_that_the_tab_shows_actually_exists() -> None:
    from pathlib import Path

    assert Path(panels.ICON).is_file()


# --------------------------------------------------------------------------
# The knowledge base
# --------------------------------------------------------------------------


def test_explore_is_about_the_problem_and_under_the_hood_about_the_agent() -> None:
    # The line between the two groups is *what the page is evidence about*. Explore is
    # the problem: the circuit, the notes, the machines, the cross-check. Under the
    # hood is the language-model half: which nodes ran, how it scores, what the calls
    # cost and which model is best value. Analytics is on that side of the line
    # despite being about money -- three of its four tabs are LLM instrumentation.
    by_module = {page.module: page for page in panels.PAGES}
    assert by_module["pages/lab.py"].group == "Explore"
    assert by_module["pages/knowledge.py"].group == "Explore"
    assert by_module["pages/physics_and_hardware.py"].group == "Explore"
    assert by_module["pages/pipeline.py"].group == "Under the hood"
    assert by_module["pages/evaluations.py"].group == "Under the hood"
    assert by_module["pages/analytics.py"].group == "Under the hood"


def test_the_corpus_is_published_whole() -> None:
    # The claim the Knowledge base page makes is "there is no hidden corpus". It is
    # only checkable if the page shows every note there is, so this holds the rows
    # against the files rather than against a number written down anywhere.
    rows = panels.corpus_notes()
    on_disk = sum(1 for _ in (PROJECT_ROOT / "data" / "corpus").rglob("*.md"))
    assert len(rows) == on_disk
    assert on_disk > 0, "the corpus is committed; an empty one is a packaging fault"


def test_every_published_note_can_be_checked_against_its_original() -> None:
    # A note listed without a citation is a note a reader has to take on trust, which
    # is exactly what publishing the corpus was supposed to remove.
    for row in panels.corpus_notes():
        assert row["title"].strip(), row["note"]
        assert row["source"].strip(), row["note"]
        assert row["topics"].strip(), row["note"]
        assert int(row["words"]) > 0, row["note"]


def test_every_note_is_on_a_shelf_the_agent_can_choose() -> None:
    # The shelf is a decision the router makes per question, from a fixed vocabulary.
    # A note filed under a name that vocabulary does not contain is unreachable: it is
    # in the repository, it is in the index, and nothing will ever search it.
    from src.rag.ingest import shelf_names

    declared = set(shelf_names())
    for row in panels.corpus_notes():
        assert row["shelf"] in declared, f"{row['note']} is on an undeclared shelf"


def test_a_reader_is_told_which_notes_a_person_wrote() -> None:
    # Curated and promoted notes are not equally trustworthy -- one was read by a
    # person, the other arrived from a search nobody reviewed -- and a listing that
    # hides the difference overstates the weight of half its own rows.
    provenance = {row["written by"] for row in panels.corpus_notes()}
    assert provenance
    assert provenance <= {"a person", "a search"}


def test_an_unreadable_corpus_is_a_sentence_rather_than_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A missing corpus is a deployment mistake. The page has to say so; a stack trace
    # in the middle of the interface says it to the wrong audience.
    def explode(*_args: object, **_kwargs: object) -> tuple[object, ...]:
        raise RuntimeError("no corpus here")

    monkeypatch.setattr(panels, "load_library", explode)
    assert panels.corpus_notes() == ()


# --------------------------------------------------------------------------
# The circuit a question describes
# --------------------------------------------------------------------------

FALLBACK_DEPTH = 4
"""Stands in for the settings knob, which no test has a session to read."""


def asked(question: str, model: FormalModel | None = None) -> CampaignState:
    """Build a finished-looking campaign for one question.

    Only the fields the drawing decision reads are filled. Everything else stays at
    its opening value, which is the point: the decision is made from the question's
    own words, so it must not need a campaign to have run.

    Args:
        question: What was typed.
        model: The formalised chain, when the test is about one.

    Returns:
        The state.
    """
    state = new_campaign(Request(text=question), shot_budget=1000)
    state["model"] = model
    return state


@pytest.mark.parametrize(
    ("question", "n_qubits", "depth"),
    [
        ("Can you write the QAOA circuit for 8 spins at depth 3?", 8, 3),
        ("Show me a three-layer ansatz for 10 qubits", 10, 3),
        ("Is a 14-spin chain at depth 2 worth running on real gates?", 14, 2),
        ("What does the variational circuit for 6 spins look like?", 6, FALLBACK_DEPTH),
    ],
)
def test_a_question_that_describes_a_circuit_gets_one_drawn(
    question: str, n_qubits: int, depth: int
) -> None:
    # The point of the panel: this is not one starter's special case. Every question
    # that names a chain and a circuit is answered with the diagram it described,
    # whichever branch of the graph wrote the words above it.
    spec = graph.circuit_the_question_named(asked(question), FALLBACK_DEPTH)
    assert spec is not None
    assert (spec.n_qubits, spec.depth) == (n_qubits, depth)


@pytest.mark.parametrize(
    "question",
    [
        "What is a barren plateau?",
        "Is a 12-spin chain hard for an ordinary computer?",
        "Which real quantum computers could run these methods?",
    ],
)
def test_nothing_is_drawn_for_a_question_that_described_no_circuit(question: str) -> None:
    # A chain with no circuit, a circuit with no chain, and neither: all three would
    # have to be drawn from an invented number, and an invented picture is worse than
    # no picture because it looks exactly as computed as a real one.
    assert graph.circuit_the_question_named(asked(question), FALLBACK_DEPTH) is None


def test_a_refused_question_gets_no_diagram_under_its_refusal() -> None:
    # A competent-looking circuit beneath "I could not answer that" reads as an answer,
    # which undoes the refusal the branch exists to make.
    state = asked("Write the QAOA circuit for 8 spins at depth 3")
    state["request"] = Request(text=state["request"].text, in_scope=False)
    assert graph.circuit_the_question_named(state, FALLBACK_DEPTH) is None


def test_the_drawn_circuit_uses_the_chain_the_question_named() -> None:
    # A ring has one more bond than a line of the same length, and on an odd ring it
    # also costs a third round of gates. Reading the boundary out of the question is
    # what keeps the picture from being a plausible circuit for a different problem.
    spec = graph.circuit_the_question_named(
        asked("the QAOA circuit for a ring of 9 qubits at depth 2"), FALLBACK_DEPTH
    )
    assert spec is not None
    assert spec.boundary == "periodic"
    assert spec.two_qubit_rounds_per_layer == 3


def test_a_longitudinal_field_reaches_the_drawn_circuit() -> None:
    # It costs one single-qubit rotation per site per layer and no two-qubit depth at
    # all, so a picture that ignored it would understate the gate count of the one
    # regime this project holds in reserve.
    chain = FormalModel(n_sites=8, longitudinal_field=0.3)
    spec = graph.circuit_the_question_named(
        asked("the QAOA circuit for 8 spins at depth 3", chain), FALLBACK_DEPTH
    )
    assert spec is not None
    assert spec.longitudinal
    assert spec.single_qubit_gates > 8 * 3 + 8


# --------------------------------------------------------------------------
# The circuit strip above the question box
# --------------------------------------------------------------------------


def bonds_drawn(spec: AnsatzSpec, wires: int) -> int:
    """How many bond rotations of one layer fit inside the truncated strip.

    Args:
        spec: The circuit being drawn.
        wires: How many wires the strip has room for.

    Returns:
        The count the drawing should contain per layer.
    """
    return sum(1 for one, other in spec.bonds if max(one, other) < wires and abs(one - other) == 1)


def test_the_strip_draws_the_schedule_the_depth_arithmetic_prices() -> None:
    # Every connector and every box comes off the spec: one connector per bond
    # rotation per layer, one field box per wire per layer, one Hadamard per wire. A
    # strip drawn from anything else would be the one picture in the application that
    # does not describe what runs.
    spec = AnsatzSpec(n_qubits=10, depth=2)
    wires = panels.LETTERHEAD_WIRES
    svg = panels.ansatz_svg(spec)
    assert svg.count('class="bond"') == bonds_drawn(spec, wires) * spec.depth
    assert svg.count(">x</tspan>") == wires * spec.depth
    assert svg.count(">H<") == wires
    assert ">z</tspan>" not in svg


def test_a_bond_that_wraps_a_ring_is_left_out_rather_than_drawn_through_the_circuit() -> None:
    # It joins the top wire to the bottom one, so drawing it would put a connector
    # straight through every gate between them.
    ring = panels.ansatz_svg(AnsatzSpec(n_qubits=6, depth=1, boundary="periodic"))
    line = panels.ansatz_svg(AnsatzSpec(n_qubits=6, depth=1))
    assert ring.count('class="bond"') == line.count('class="bond"')


def test_the_third_field_reaches_the_strip_the_moment_it_is_switched_on() -> None:
    # It costs one single-qubit rotation per site per layer, so a strip that ignored
    # it would draw a cheaper circuit than the one that would run.
    svg = panels.ansatz_svg(AnsatzSpec(n_qubits=4, depth=1, longitudinal=True))
    assert svg.count(">z</tspan>") == 4
    assert svg.count(">x</tspan>") == 4


def test_a_circuit_with_no_layers_draws_no_gates_it_does_not_have() -> None:
    # Depth zero is legal and means the bare initial state.
    svg = panels.ansatz_svg(AnsatzSpec(n_qubits=4, depth=0))
    assert 'class="bond"' not in svg
    assert ">x</tspan>" not in svg
    assert svg.count(">H<") == 4


def test_what_the_strip_has_no_room_for_is_marked_rather_than_dropped_silently() -> None:
    # The strip carries no caption, so a chain longer and a circuit deeper than it
    # draws are said with dots -- one group per wire for the layers it stopped short
    # of, one under the register for the magnets. Without them the masthead would
    # quietly claim to be the whole circuit.
    spec = AnsatzSpec(n_qubits=16, depth=12)
    svg = panels.ansatz_svg(spec)
    drawn = svg.count('class="bond"')
    assert 0 < drawn < bonds_drawn(spec, panels.LETTERHEAD_WIRES) * spec.depth
    assert svg.count('class="continues"') == panels.LETTERHEAD_WIRES + 1
    assert 'class="continues"' not in panels.ansatz_svg(AnsatzSpec(n_qubits=4, depth=2))


@pytest.mark.parametrize(
    "spec",
    [
        AnsatzSpec(n_qubits=10, depth=2),
        AnsatzSpec(n_qubits=16, depth=12),
        AnsatzSpec(n_qubits=6, depth=3, longitudinal=True),
        AnsatzSpec(n_qubits=2, depth=1),
        AnsatzSpec(n_qubits=10, depth=0),
    ],
    ids=["default", "widest", "three-term", "smallest", "no-layers"],
)
def test_the_strip_stays_a_strip_at_every_knob_position(spec: AnsatzSpec) -> None:
    # The constraint the drawing exists under. It fills the width of the page above
    # the question box, so its own proportions are what decide how tall that makes
    # it: too square and the box the page is for goes off the first screen, too long
    # and the gates are too small to read. Both ends are held by truncation and by
    # spreading the columns, so both are asserted at the extremes of the knob.
    width, height = (float(value) for value in _view_box(panels.ansatz_svg(spec)))
    # The tenth of a unit is the rounding the view box is written with.
    assert width + 0.1 >= height * panels.LETTERHEAD_ASPECT
    assert width - 0.1 <= height * panels.LETTERHEAD_LONGEST


def _view_box(svg: str) -> tuple[str, str]:
    """The drawing's width and height, read out of its own markup.

    Args:
        svg: The markup.

    Returns:
        Width and height, as they were written.
    """
    box = svg.split('viewBox="0 0 ', 1)[1].split('"', 1)[0].split()
    return box[0], box[1]


# --------------------------------------------------------------------------
# The true answer the interface grades curves against
# --------------------------------------------------------------------------


def test_the_true_answer_is_available_for_a_chain_small_enough_to_solve() -> None:
    energy, refusal = panels.true_energy(6, 1.0, 1.0, "open")
    assert refusal is None
    assert energy is not None
    assert energy < 0.0


def test_a_tilted_chain_is_refused_rather_than_answered_for_a_different_problem() -> None:
    # A non-zero g breaks the mapping every exact route here relies on. Ignoring it and
    # returning the g = 0 energy would be the right number for the wrong Hamiltonian,
    # drawn as a dashed line labelled "the true answer".
    energy, refusal = panels.true_energy(6, 1.0, 1.0, "open", 0.5)
    assert energy is None
    assert refusal is not None
    assert "g = 0.5" in refusal


def test_a_chain_too_large_to_solve_exactly_says_so_rather_than_hanging() -> None:
    energy, refusal = panels.true_energy(400, 1.0, 1.0, "open")
    assert energy is None
    assert refusal


# --------------------------------------------------------------------------
# The exact curves, drawn from what the agent asked for
# --------------------------------------------------------------------------


def with_sweep_call(arguments: dict[str, object]) -> CampaignState:
    """A campaign carrying one recorded sweep call and nothing else.

    Args:
        arguments: What the agent passed to the tool.

    Returns:
        The state, as the page receives it.
    """
    from src.agent.llm import ToolCall

    state = new_campaign(Request(text="plot the spectrum against the field"), shot_budget=1000)
    state["tool_calls"] = (
        ToolCall(name=panels.SWEEP_TOOL, arguments=arguments, result="{}", failed=False),
    )
    return state


def test_the_curves_are_found_from_the_last_successful_call() -> None:
    from src.agent.llm import ToolCall

    state = with_sweep_call({"n_sites": 8, "curves": ["energy"]})
    state["tool_calls"] = (
        *state["tool_calls"],
        # A call the agent made and then superseded, and one that failed. The page
        # must draw the argument the agent settled on, not the one it abandoned.
        ToolCall(name=panels.SWEEP_TOOL, arguments={"n_sites": 12}, result="{}", failed=False),
        ToolCall(name="search_arxiv", arguments={}, result="{}", failed=False),
    )
    found = panels._sweep_call(state)
    assert found is not None
    assert found.arguments["n_sites"] == 12


def test_a_failed_call_draws_nothing() -> None:
    from src.agent.llm import ToolCall

    state = new_campaign(Request(text="plot the spectrum"), shot_budget=1000)
    state["tool_calls"] = (
        ToolCall(name=panels.SWEEP_TOOL, arguments={}, result="error", failed=True),
    )
    assert panels._sweep_call(state) is None


def test_a_campaign_that_called_nothing_draws_nothing() -> None:
    assert panels._sweep_call(new_campaign(Request(text="anything"), shot_budget=1000)) is None


def test_only_the_solvers_own_arguments_are_passed_back_to_it() -> None:
    # The arguments were written by a model, so an invented key would otherwise be a
    # TypeError in the middle of a page that was about to render correctly.
    from src.physics.registry import field_sweep_bench

    arguments = {"n_sites": 8, "curves": ["energy"], "invented": True, "points": 5}
    kept = {key: value for key, value in arguments.items() if key in panels.SWEEP_ARGUMENTS}
    assert "invented" not in kept
    assert field_sweep_bench()(**kept)["n_sites"] == 8


@pytest.mark.parametrize("curve", sorted(panels.SWEEP_CAPTIONS))
def test_every_curve_the_solver_offers_has_a_caption_and_a_figure(curve: str) -> None:
    from typing import get_args

    from src.physics.reference.field_sweep import Curve
    from src.physics.registry import field_sweep_bench

    assert curve in get_args(Curve)
    caption = panels.SWEEP_CAPTIONS[curve]
    # A caption states the claim in plain words: it is the part a reader without
    # physics actually reads, so an empty or jargon-only one is a real defect.
    assert caption.startswith("**") and len(caption) > 120
    payload = field_sweep_bench()(n_sites=8, curves=[curve], points=9)
    drawn = panels._sweep_figure(curve, payload["ratio"], payload)
    assert drawn is not None


def test_a_curve_the_solver_does_not_carry_is_left_out_rather_than_drawn_empty() -> None:
    from src.physics.registry import field_sweep_bench

    # A spectrum was not asked for, so no levels came back. Drawing an empty pair of
    # axes would read as a broken figure rather than as a question not asked.
    payload = field_sweep_bench()(n_sites=8, curves=["energy"], points=5)
    assert panels._sweep_figure("spectrum", payload["ratio"], payload) is None


def test_the_provenance_line_names_the_chain_and_the_disagreement() -> None:
    from src.agent.llm import ToolCall
    from src.physics.registry import field_sweep_bench

    payload = field_sweep_bench()(n_sites=8, curves=["energy"], points=5)
    said = panels._sweep_provenance(
        payload,
        ToolCall(name=panels.SWEEP_TOOL, arguments={"curves": ["energy"]}, result=""),
        "plot the energy of an 8-spin ring against h/J",
    )
    assert "L=8" in said
    assert "twice" in said
    # A length the question named is not called out as a default.
    assert "named no length" not in said


def test_the_provenance_line_says_when_the_curve_was_not_cross_checked() -> None:
    from src.agent.llm import ToolCall
    from src.physics.registry import field_sweep_bench

    # Past the length a sweep will diagonalise at every point, the closed form runs
    # alone. A curve that quietly stopped being checked is the worse failure.
    payload = field_sweep_bench()(n_sites=40, curves=["energy"], points=5)
    said = panels._sweep_provenance(
        payload, ToolCall(name=panels.SWEEP_TOOL, arguments={}, result=""), "plot the energy"
    )
    assert "alone" in said
    # And a length nobody typed is said out loud rather than left to look chosen.
    assert "named no length" in said


def test_the_infinite_chains_gap_is_drawn_in_energy_units_not_in_units_of_j() -> None:
    # The levels are absolute energies, so the reference 2|J - h| carries the
    # coupling. Drawn as 2|1 - h/J| it would be half the curve it claims to be at
    # J = 2 -- and it would look entirely plausible.
    from src.physics.registry import field_sweep_bench

    payload = field_sweep_bench()(n_sites=8, coupling=2.0, curves=["spectrum"], points=5)
    assert payload["thermodynamic_gap"][0] == pytest.approx(4.0)
    assert panels._sweep_figure("spectrum", payload["ratio"], payload) is not None


def test_the_energy_panel_draws_the_infinite_chain_beneath_the_finite_one() -> None:
    from src.physics.registry import field_sweep_bench

    payload = field_sweep_bench()(n_sites=8, curves=["energy"], points=9)
    drawn = panels._sweep_figure("energy", payload["ratio"], payload)
    assert drawn is not None
    # Two curves and the critical-field marker: the finite chain, the infinite one,
    # and the line the question is about.
    assert len(drawn.axes[0].lines) == 3


def test_the_page_and_the_graph_agree_on_the_tools_name() -> None:
    from src.agent.graph import SWEEP_TOOL_NAME

    assert panels.SWEEP_TOOL == SWEEP_TOOL_NAME


def test_a_refused_curve_is_explained_rather_than_left_silent() -> None:
    # A reader who asked for the magnetisation of an *open* chain asked for something
    # the closed form cannot give. Printing nothing would leave them to conclude they
    # were ignored, which is the same failure as a confident wrong answer, quieter.
    from src.agent.llm import ToolCall

    state = new_campaign(Request(text="plot the magnetisation"), shot_budget=1000)
    state["tool_calls"] = (
        ToolCall(
            name=panels.SWEEP_TOOL,
            arguments={"n_sites": 7},
            result="{'error': 'the closed-form solution does not apply'}",
            failed=True,
        ),
    )
    assert panels._sweep_call(state) is None
    # It draws a caption rather than returning early in silence; the assertion is that
    # it finds the refusal to explain.
    refused = [c for c in state["tool_calls"] if c.name == panels.SWEEP_TOOL and c.failed]
    assert refused
    panels._say_why_there_is_no_curve(state)


def test_a_question_that_asked_for_no_curve_stays_silent() -> None:
    panels._say_why_there_is_no_curve(new_campaign(Request(text="hello"), shot_budget=1000))


def test_a_declined_request_is_never_drawn_as_a_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A question turned away at the door must not come back as a hardware call.

    The worst shape this application produced: *detail the mathematics of the
    quantum-to-classical mapping* was declined by the scope gate, fell through to
    the verdict card, and was drawn as a blue **NO** with "confidence: high, 0 run,
    0 refused, 0 measurements". The badges were accurate; the sentence they formed
    was not. "No" here means *a quantum computer is not worth it for this problem*,
    and no hardware had been considered.

    The gate no longer refuses that particular question -- see
    ``test_the_scope_gate_admits_the_projects_own_hyphenated_subject`` -- but
    something will always be out of scope, so the presentation is pinned too.
    """
    written: list[str] = []
    monkeypatch.setattr(panels.st, "markdown", lambda text, **_: written.append(str(text)))

    state = asked("What is the best pizza in Vilnius?")
    state["request"] = Request(text=state["request"].text, in_scope=False)
    state["verdict"] = Verdict(
        call="no",
        summary="The request could not be turned into a specific physics problem.",
        crossover_condition="restate the problem",
        confidence="high",
    )

    panels.answer_card(state)

    drawn = " ".join(written)
    assert "NOT ASSESSED" in drawn
    assert "NO]" not in drawn, f"a refusal was drawn as a verdict: {drawn!r}"
    assert "confidence" not in drawn, f"a refusal claimed a confidence: {drawn!r}"


def _rung(depth: int, energy_per_site: float) -> RunRecord:
    """One rung of the depth ladder, filled only where the drawing reads it.

    Args:
        depth: How many ansatz layers this rung ran.
        energy_per_site: The energy it reached, per spin.

    Returns:
        The record.
    """
    return RunRecord(
        label=f"hva-p{depth}",
        family="hva",
        depth=depth,
        two_qubit_depth=4 * depth,
        energy=energy_per_site * 6,
        energy_per_site=energy_per_site,
        shots_spent=48_400_000 * depth,
        diagnosis=Diagnosis(signal="healthy", evidence="converged", repair="", is_actionable=False),
    )


def test_a_withheld_verdict_still_reports_the_ladder_it_climbed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runs happened. Saying "the campaign did not reach one" throws them away.

    *How deep should the circuit be before noise wins?* is advertised in the README.
    It climbs five rungs, spends 774 million measurements and is refused a sixth on
    budget -- and then names no chain, so `converge` correctly withholds the call.
    The card used to answer all of that with one sentence saying nothing was
    reached, and pointed at a trail that lives on another page.
    """
    written: list[str] = []
    monkeypatch.setattr(panels.st, "markdown", lambda text, **_: written.append(str(text)))
    monkeypatch.setattr(panels, "prose", lambda text, **_: written.append(str(text)))

    state = asked("How deep should the circuit be before noise wins?", FormalModel(n_sites=6))
    state["runs"] = tuple(_rung(depth, -1.1 - depth / 100) for depth in (1, 2, 3, 4, 6))

    panels.verdict_card(None, state)

    drawn = " ".join(written)
    assert "5 runs" in drawn, f"the ladder was not reported: {drawn!r}"
    assert "-1.160000" in drawn, f"the energy it reached was not reported: {drawn!r}"
    assert "did not reach one" not in drawn, f"five runs drawn as none: {drawn!r}"


def test_a_campaign_that_ran_nothing_still_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    """The other half. A withheld verdict over no runs must not imply there were any."""
    written: list[str] = []
    monkeypatch.setattr(panels.st, "markdown", lambda text, **_: written.append(str(text)))
    monkeypatch.setattr(panels, "prose", lambda text, **_: written.append(str(text)))

    panels.verdict_card(None, asked("how deep before noise wins?"))

    drawn = " ".join(written)
    assert "nothing run" in drawn, drawn
    assert "runs" not in drawn.replace("nothing run", ""), f"runs implied where none ran: {drawn!r}"


@pytest.mark.parametrize(("count", "expected"), [(0, "0 runs"), (1, "1 run"), (5, "5 runs")])
def test_the_run_badge_counts_and_the_word_agree(count: int, expected: str) -> None:
    """The badge read "1 circuits" and "5 run" in two different places.

    A number beside a literal word is the commonest wrong sentence this interface
    produces, and it is the one a reader notices first.
    """
    assert panels._runs_badge(tuple(_rung(d, -1.2) for d in range(1, count + 1))) == expected
