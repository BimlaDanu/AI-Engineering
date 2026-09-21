"""The settings knobs, and whether moving one changes anything.

A dial that changes nothing is worse than no dial: a reader who moves it and sees the
same answer concludes the feature is broken, and a reader who does not move it
believes a claim the application never checked. The chat page offered three dials
while hard-coding the two that most change what comes back -- whether the agent
searches its notes at all, and whether it may look outside them.

So these tests are about connection rather than appearance. Each asserts that a knob
reaches the campaign and that the campaign behaves differently at the two ends of it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.runnables import RunnableConfig

from src.agent.graph import SearchTuning, run_campaign
from src.agent.state import CampaignState
from src.hardware.devices import device_for
from src.settings import Settings, get_settings
from src.ui import setting as knob

QUESTION = "Is quantum hardware worth it for a 10-spin critical Ising chain?"


def base_settings() -> Settings:
    """The process configuration the knob overlays itself onto.

    Returns:
        The settings the suite runs with -- a placeholder credential and every
        default, pinned by ``tests/conftest.py`` so this is the same object in CI as
        on a laptop with a real ``.env`` beside it.
    """
    return get_settings()


def campaign(question: str = QUESTION, **changes: object) -> CampaignState:
    """Run one offline campaign with the given overrides.

    Args:
        question: What to ask. Defaults to the one question this file is about,
            which names a chain; a test about what happens when nothing is named
            passes its own.
        **changes: Arguments forwarded to :func:`run_campaign`.

    Returns:
        The finished campaign.
    """
    return run_campaign(
        question,
        shot_budget=10**9,
        device=device_for("heavy-hex-27"),
        chat_model=None,
        **changes,  # type: ignore[arg-type]
    )


# The settings record
# --------------------------------------------------------------------------


def test_the_search_group_starts_where_the_campaign_always_behaved() -> None:
    # The defaults have to reproduce the hard-coded behaviour they replaced,
    # or adding the dials silently changed every answer.
    search = knob.Search()
    assert search.corpus
    assert not search.external
    assert search.followups


def test_a_moved_search_dial_is_reported_as_moved() -> None:
    # A shut knob hides a moved dial. Somebody who pointed the search at one shelf
    # an hour ago needs telling before they read a thin answer.
    moved = knob.Setting().with_search(shelf="physics-notes").changes_from_defaults()
    assert any("shelf" in phrase for phrase in moved)


def test_every_offered_shelf_is_one_the_corpus_actually_has() -> None:
    # Read off the corpus rather than typed out, so a renamed shelf changes the list
    # instead of leaving a dead option that returns nothing.
    from src.rag.ingest import shelf_names

    assert set(knob.SHELF_CHOICES) == {"decide for me", *shelf_names()}


@pytest.mark.parametrize(
    ("field", "value"),
    [("passages", 0), ("passages", 99), ("vector_share", -0.1), ("shelf", "not-a-shelf")],
)
def test_an_impossible_position_is_refused_at_the_knob(field: str, value: object) -> None:
    # Refused here rather than four screens later, where the retrieval layer would
    # return nothing and the page would blame the corpus.
    with pytest.raises(ValueError):
        knob.Search(**{field: value})  # type: ignore[arg-type]


def test_deciding_for_me_names_no_shelf_at_all() -> None:
    # Empty and "search everywhere" have to be the same instruction, because that is
    # what lets the router widen a search that found nothing.
    assert knob.Search(shelf="decide for me").shelves == ()
    assert knob.Search(shelf="physics-notes").shelves == ("physics-notes",)


# The dials, connected
# --------------------------------------------------------------------------


def test_turning_the_search_off_costs_the_answer_its_citations() -> None:
    with_notes = campaign(search_corpus=True)
    without = campaign(search_corpus=False)
    assert with_notes["citations"]
    assert without["citations"] == ()


def test_turning_the_search_off_changes_nothing_else() -> None:
    # The claim the knob's help text makes. If switching retrieval off also moved the
    # verdict, the citations would have been load-bearing in a way nothing says.
    with_notes = campaign(search_corpus=True)
    without = campaign(search_corpus=False)
    mine, theirs = with_notes["verdict"], without["verdict"]
    assert mine is not None and theirs is not None
    assert mine.call == theirs.call
    assert with_notes["shots"].spent == without["shots"].spent


def test_asking_for_fewer_passages_returns_fewer() -> None:
    few = campaign(search_tuning=SearchTuning(passages=1))
    many = campaign(search_tuning=SearchTuning(passages=6))
    assert len(few["citations"]) < len(many["citations"])


def test_naming_a_shelf_keeps_the_search_on_it() -> None:
    only = campaign(search_tuning=SearchTuning(shelves=("physics-notes",)))
    assert only["citations"]
    assert {c.shelf for c in only["citations"]} == {"physics-notes"}


def test_switching_off_the_follow_ups_offers_none() -> None:
    quiet = campaign(suggest_followups=False)
    assert not quiet["followups"].any_offered
    assert quiet["followups"].proposed_by == "none"


def test_leaving_the_follow_ups_on_offers_some() -> None:
    # The other end of the same dial, so a test that passes because the feature is
    # broken fails here instead.
    loud = campaign(suggest_followups=True)
    assert loud["followups"].any_offered


# The tuning dials, and whether they reach the gateway
# --------------------------------------------------------------------------
#
# These are the dials with the longest history of being decorative. The temperature
# slider was drawn, labelled, reported by `changes_from_defaults` as moved -- and
# never read by anything: `pool()` built its models from the process configuration
# and the visitor's position was discarded. So every test here asserts the
# connection rather than the widget, and one of them asserts the thing that made the
# old bug invisible: that the knob opens where the process actually runs.


def test_every_tuning_dial_opens_where_the_process_actually_runs() -> None:
    # The failure this prevents is not a mislabelled slider. Drawing the sidebar is
    # what *sets* the value a campaign runs under, so a dial opening at a stale
    # literal silently reconfigures the run to that literal the moment anybody looks
    # at the panel. The knob's temperature was 0.2 while the process default was 0.6.
    fields = Settings.model_fields
    model = knob.Model()
    assert model.temperature == fields["temperature"].default
    assert model.max_output_tokens == fields["max_output_tokens"].default
    assert model.max_retries == fields["max_retries"].default
    assert model.requests_per_second == fields["requests_per_second"].default
    assert model.request_timeout_s == fields["request_timeout_s"].default


def test_the_language_model_is_on_when_there_is_a_credential_to_call_with() -> None:
    # The dead-knob rule applied to the one switch that decides whether any of the
    # others matter. It was hard-coded to `True`, so a fully configured application
    # opened with its model switched off: the campaign ran, every number was right,
    # every word was a template, and nothing on screen said why. "No model was
    # called" is a finding when it is chosen and a defect when it is the default.
    assert knob.credential_available()
    assert knob.default_offline() is False
    assert knob.Model().offline is False
    assert knob.Setting().model.offline is False


def test_the_language_model_is_off_when_there_is_no_credential(
    unconfigured_environment: None,
) -> None:
    # And the other half, which is what a checkout with no `.env` gets and the mode
    # the page tests render in. Offline stays first-class: it is reachable, it is the
    # honest default here, and every number is still computed.
    assert knob.credential_available() is False
    assert knob.default_offline() is True
    assert knob.Setting().model.offline is True


def test_the_language_model_switch_is_not_reported_as_moved_by_merely_opening(
    unconfigured_environment: None,
) -> None:
    # `changes_from_defaults` compares against a freshly built `Setting`, so the
    # baseline has to be re-read rather than frozen at import: a `Model()` evaluated
    # once at class-definition time would report "language model = True" as a
    # deliberate change on every credential-less run, and a warning that fires when
    # nobody touched anything trains a reader to ignore the ones that matter.
    assert not any("language model" in phrase for phrase in knob.Setting().changes_from_defaults())


def test_switching_the_language_model_off_is_still_reported_as_moved() -> None:
    # The guard on the test above: the phrase has to appear when the switch really
    # was moved, or the previous test would pass just as well against a knob that
    # never reports this dial at all.
    moved = knob.Setting().with_model(offline=True).changes_from_defaults()
    assert any("language model" in phrase for phrase in moved)


def test_no_dial_offers_a_position_the_configuration_would_refuse() -> None:
    # A slider whose top end fails validation is a control that produces a stack
    # trace instead of a setting, and it does it only for the visitor who drags it
    # all the way.
    for low, high in (
        knob.MAX_OUTPUT_TOKEN_BOUNDS,
        (0, knob.MAX_RETRY_BOUND),
        (0.0, knob.MAX_REQUESTS_PER_SECOND),
        knob.REQUEST_TIMEOUT_BOUNDS,
    ):
        assert low < high
    for extreme in (
        {"max_output_tokens": knob.MAX_OUTPUT_TOKEN_BOUNDS[0]},
        {"max_output_tokens": knob.MAX_OUTPUT_TOKEN_BOUNDS[1]},
        {"max_output_tokens": None},
        {"max_retries": knob.MAX_RETRY_BOUND},
        {"requests_per_second": 0.0},
        {"requests_per_second": knob.MAX_REQUESTS_PER_SECOND},
        {"request_timeout_s": knob.REQUEST_TIMEOUT_BOUNDS[0]},
        {"request_timeout_s": knob.REQUEST_TIMEOUT_BOUNDS[1]},
    ):
        settings = knob.Model(**extreme).applied_to(base_settings())
        assert Settings(**settings.model_dump()) is not None


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("temperature", 2.5),
        ("max_output_tokens", 8),
        ("max_retries", 99),
        ("requests_per_second", -1.0),
        ("request_timeout_s", 1.0),
    ),
)
def test_an_impossible_tuning_position_is_refused_at_the_knob(field: str, value: object) -> None:
    # Refused here, where the message names the dial, rather than four screens later
    # inside a campaign, where it names a pydantic field nobody moved.
    with pytest.raises(ValueError, match=field):
        knob.Model(**{field: value})  # type: ignore[arg-type]


def test_the_tuning_dials_reach_the_configuration_a_model_is_built_from() -> None:
    # The connection the temperature slider spent its whole life missing.
    moved = knob.Model(
        temperature=1.75,
        max_output_tokens=512,
        max_retries=7,
        requests_per_second=0.0,
        request_timeout_s=15.0,
        max_calls=80,
    ).applied_to(base_settings())
    assert moved.temperature == 1.75
    assert moved.max_output_tokens == 512
    assert moved.max_retries == 7
    assert moved.requests_per_second == 0.0
    assert moved.request_timeout_s == 15.0
    assert moved.max_model_calls_per_run == 80


def test_the_chat_panel_can_choose_a_model_for_every_tier_the_sidebar_can() -> None:
    # Ticking "set them here instead" used to take the model selectors away, so a
    # reader who wanted a stronger model for one question had to go back to the
    # sidebar -- and the override they had just set then followed the sidebar for the
    # models and not for anything else. The knob has to carry the choice for both
    # panels to be able to offer it.
    picked = tuple((tier, "openai/gpt-4o") for tier, _ in knob.DEFAULT_TIERS)
    chosen = knob.Setting().with_model(tiers=picked)
    assert chosen.model.overrides() == dict.fromkeys(
        ("fast", "standard", "strong"), "openai/gpt-4o"
    )


def test_a_tier_the_project_does_not_have_is_refused_at_the_knob() -> None:
    # The selectors are built from `DEFAULT_TIERS`, so a typo here is a validation
    # error rather than an override that is silently ignored at call time.
    with pytest.raises(ValueError, match="unknown tier"):
        knob.Model(tiers=(("swift", "openai/gpt-4o"),))


def test_a_visitor_cannot_move_anything_that_is_not_theirs_to_move() -> None:
    # The overlay is deliberately narrow. The credential, the endpoint and the corpus
    # paths are deployment, not preference, and a settings panel that could rewrite
    # them would be a settings panel that could point the agent at another gateway.
    base = base_settings()
    moved = knob.Model(temperature=1.9).applied_to(base)
    assert moved.openrouter_api_key == base.openrouter_api_key
    assert moved.openrouter_base_url == base.openrouter_base_url
    assert moved.corpus_path == base.corpus_path
    assert moved.vector_store_path == base.vector_store_path


def test_the_dial_survives_all_the_way_into_the_built_client() -> None:
    # End to end: knob -> Settings -> ChatOpenAI. Constructing the client reaches no
    # network, so this is the cheapest place to prove the value arrives.
    from src.agent.llm import build_chat_model

    built = build_chat_model(
        knob.Model(temperature=1.25, request_timeout_s=45.0).applied_to(base_settings())
    )
    # Read off the constructed client by name rather than through the typed
    # `BaseChatModel` interface, which declares neither: what is being checked is
    # that the value reached the concrete `ChatOpenAI` the gateway is called through.
    assert getattr(built, "temperature", None) == 1.25
    assert getattr(built, "request_timeout", None) == 45.0


def test_the_pool_retries_as_many_times_as_the_knob_says() -> None:
    # `max_retries` is the one transport dial the client does not carry -- retries
    # belong to the middleware -- so it has its own route and its own check.
    from src.agent.model_selection import ModelPool

    tuned = knob.Model(max_retries=9).applied_to(base_settings())
    assert ModelPool(settings=tuned)._max_retries() == 9


# Searches per question
# --------------------------------------------------------------------------


def test_the_search_round_dial_stops_where_the_retrieval_layer_stops() -> None:
    from src.rag.retrieve import MAX_ROUNDS

    assert knob.MAX_SEARCH_ROUNDS == MAX_ROUNDS
    assert knob.Search().rounds == MAX_ROUNDS


def test_asking_for_no_searches_at_all_is_refused() -> None:
    with pytest.raises(ValueError, match="rounds"):
        knob.Search(rounds=0)


def test_the_graph_default_is_the_retrieval_layers_own_ceiling() -> None:
    # `graph.py` restates the number rather than importing it, so that a campaign
    # which never retrieves never pays for importing the vector store. This is what
    # keeps the copy honest.
    from src.agent.graph import RETRIEVAL_ROUNDS
    from src.rag.retrieve import MAX_ROUNDS

    assert RETRIEVAL_ROUNDS == MAX_ROUNDS
    assert SearchTuning().rounds == MAX_ROUNDS


def test_the_answer_is_allowed_some_variation_by_default() -> None:
    # Asked for directly: the wording should not be deterministic. A knob that opened
    # at zero would make every reply identical, which reads as a canned answer rather
    # than a written one -- and it would make the "turn it up and watch the numbers
    # not move" demonstration impossible to perform in the interesting direction.
    #
    # It is 0.6 because that is where the process runs, not because it is retyped
    # here; the assertion is written both ways round so that moving the process
    # default without moving this test fails loudly rather than quietly.
    assert knob.Model().temperature == 0.6
    assert knob.Model().temperature == Settings.model_fields["temperature"].default
    assert knob.Model().temperature > 0.0


# The two dials that were drawn and then discarded
# --------------------------------------------------------------------------


def _depth_of(label: str) -> int:
    """Recover the depth a configuration label carries, as ``hva-p6`` carries six.

    Args:
        label: The configuration label.

    Returns:
        The depth. Read here rather than imported from the graph so that this test
        fails if the label format changes, which is the whole reason it reads one.
    """
    return int(label.rpartition("-p")[2])


def test_asking_for_more_accuracy_costs_measurements() -> None:
    # The accuracy dial reached the cross-check page and stopped there; the campaign
    # priced every configuration against a hard-coded total error, so a visitor could
    # ask for a hundred times the accuracy and watch the verdict not move. It moves
    # now, and it moves in the only direction it can: ten times the accuracy is a
    # hundred times the measurements, so a fixed budget buys fewer configurations.
    loose = campaign(precision_per_site=1e-2)
    tight = campaign(precision_per_site=1e-4)
    assert loose["shots"].spent > tight["shots"].spent
    assert len(loose["runs"]) > len(tight["runs"])
    # Where the two runs stop, rather than how many rejections they collected on the
    # way. Counting rejections was the wrong measure of the same claim: the ladder now
    # stops at the first refusal, because every reason for one grows with depth, so a
    # tighter accuracy shows up as a *shallower* stopping point and not as a longer
    # list. The old assertion passed only because the loop used to keep climbing past
    # an answer it already had.
    assert tight["ruled_out"], "a budget this tight has to refuse something"
    assert "shots" in tight["ruled_out"][0].reason or "remain" in tight["ruled_out"][0].reason
    stopped_tight = min(_depth_of(item.label) for item in tight["ruled_out"])
    stopped_loose = min(_depth_of(item.label) for item in loose["ruled_out"])
    assert stopped_tight < stopped_loose


def test_an_unaffordable_accuracy_refuses_rather_than_pretending() -> None:
    # The failure worth having. Nothing is run, every configuration is refused on
    # arithmetic before it spends anything, and the reasons are recorded -- which is
    # a better answer than a cheap run relabelled as an accurate one.
    impossible = campaign(precision_per_site=1e-4)
    assert impossible["runs"] == ()
    assert impossible["shots"].spent == 0
    assert impossible["ruled_out"]


def test_the_accuracy_dial_is_per_magnet_and_not_per_chain() -> None:
    # Per magnet, because that is the quantity comparable between chains of
    # different lengths. The chain is read out of the question rather than the knob,
    # so the multiplication has to happen inside the campaign.
    from src.agent.graph import TARGET_ENERGY_ERROR, _target_error_for
    from src.agent.state import FormalModel, Request, new_campaign

    state = new_campaign(request=Request(text="q"), shot_budget=10**9)
    state["model"] = FormalModel(n_sites=8)
    asked: RunnableConfig = {"configurable": {"precision_per_site": 1e-3}}
    silent: RunnableConfig = {"configurable": {}}
    assert _target_error_for(state, asked) == pytest.approx(8e-3)
    assert _target_error_for(state, silent) == TARGET_ENERGY_ERROR


def test_every_reading_level_reaches_the_prompt_that_writes_the_prose() -> None:
    # The reading level was the longest-lived decorative dial here: two panels
    # offered it, the help text promised it "changes how the answer is worded", and
    # no prompt in the project had ever read it.
    from typing import get_args

    from src.agent.explaining import EXPLAIN_SYSTEM
    from src.agent.prompts import AUDIENCE_GUIDANCE, for_audience
    from src.settings import Audience

    assert set(AUDIENCE_GUIDANCE) == set(get_args(Audience))
    written = {level: for_audience(EXPLAIN_SYSTEM, level) for level in get_args(Audience)}
    assert len(set(written.values())) == len(written), "two levels produce the same prompt"
    for level, prompt in written.items():
        assert prompt.startswith(EXPLAIN_SYSTEM), f"{level} rewrote the rules above it"
        assert AUDIENCE_GUIDANCE[level] in prompt


def test_the_reading_level_changes_no_rule_about_what_may_be_claimed() -> None:
    # The claim the interface makes to a sceptic, asserted rather than trusted: the
    # part of the prompt that governs accuracy, citation and what may not be stated
    # as a result is byte-identical at every level. Only a paragraph is appended.
    from src.agent.explaining import EXPLAIN_SYSTEM
    from src.agent.prompts import for_audience

    for level in ("beginner", "researcher"):
        prompt = for_audience(EXPLAIN_SYSTEM, level)
        assert prompt[: len(EXPLAIN_SYSTEM)] == EXPLAIN_SYSTEM


def test_the_reading_level_moves_no_number() -> None:
    # And the other half of the same claim, at the level of a whole campaign.
    plain = campaign(audience="beginner")
    expert = campaign(audience="researcher")
    assert plain["shots"].spent == expert["shots"].spent
    assert [run.energy for run in plain["runs"]] == [run.energy for run in expert["runs"]]
    assert plain["verdict"] == expert["verdict"]


# The thumbs, and what they teach
# --------------------------------------------------------------------------


def test_the_down_button_moves_one_step_towards_less_assumed_knowledge() -> None:
    # `AUDIENCE_LABELS` is ordered by how much of *this domain* is assumed rather
    # than by seniority, which is the only ordering on which "one step simpler" is a
    # meaningful instruction instead of a slight. Both ends are absorbing: a reader
    # already at the simplest level cannot be sent below it.
    from typing import get_args

    from src.settings import Audience
    from src.ui.panels import AUDIENCE_LABELS, _shift

    scale = list(AUDIENCE_LABELS)
    assert _shift("practitioner", -1) == "software, no physics"
    assert _shift("beginner", -1) == "beginner"
    assert _shift("researcher", 1) == "researcher"
    for level in get_args(Audience):
        assert _shift(level, -1) in scale


def test_the_thumbs_learn_the_level_the_settings_knob_then_opens_at(tmp_path: Path) -> None:
    # The loop, end to end and without a browser: a rating is written, the helper
    # the sidebar reads finds it, and the level it reports is the one recorded. A
    # slider that ignored this would turn the buttons into a suggestion box -- the
    # reader would rate an answer, watch the knob stay where it was, and reasonably
    # conclude that nothing had happened.
    from src.agent.memory import Memory
    from src.ui.panels import learned_audience

    store = Memory(user="visitor", thread="t1", path=tmp_path / "memory.jsonl")
    assert learned_audience(store) is None, "nothing rated yet, so nothing learned"
    store.remember_rating("software, no physics")
    assert learned_audience(store) == "software, no physics"


def test_nothing_is_learned_from_a_memory_that_is_switched_off() -> None:
    from src.agent.memory import Memory
    from src.ui.panels import learned_audience

    assert learned_audience(Memory.disabled()) is None


def test_the_level_the_thumbs_teach_is_one_the_prompts_can_use() -> None:
    # The two halves have to agree on the same closed set, or a rating would record
    # a level `for_audience` then silently ignores.
    from typing import get_args

    from src.agent.memory import AUDIENCE_VALUES
    from src.agent.prompts import AUDIENCE_GUIDANCE
    from src.settings import Audience
    from src.ui.panels import AUDIENCE_LABELS

    levels = set(get_args(Audience))
    assert set(AUDIENCE_LABELS) == levels
    assert set(AUDIENCE_GUIDANCE) == levels
    assert set(AUDIENCE_VALUES) == levels


def test_the_chat_panel_offers_a_shortlist_spanning_basic_to_advanced() -> None:
    """Ten models, ordered, across more than one vendor.

    A selector holding every slug the subscription exposes asks a person to choose
    between six variants of one family, several of which were reconstructed from
    display names and have never resolved -- so the commonest outcome of a long list
    is a run that dies at its first call. Ten is a choice somebody can make
    mid-conversation.

    The vendor spread is not cosmetic. The point of letting anybody change the model
    is the claim the bake-off measures -- that the numbers do not move when the model
    does -- and a list confined to one vendor's ladder would test that against models
    sharing a tokeniser and a training recipe.
    """
    from src.agent.model_selection import featured_slugs, selectable_slugs

    offered = featured_slugs()
    assert 8 <= len(offered) <= 10
    assert len(set(offered)) == len(offered), "a model is offered twice"
    vendors = {slug.split("/")[0] for slug in offered}
    assert len(vendors) >= 3, f"only {vendors} represented"
    assert set(offered) <= set(selectable_slugs()), "a shortlisted slug is not selectable"


def test_the_shortlist_leads_with_a_model_that_has_a_price() -> None:
    # It is the option somebody lands on by not choosing, and a session opened on an
    # unpriced model reports its spend as a floor from the first question.
    from src.agent.model_selection import featured_slugs
    from src.agent.usage import price_for

    assert price_for(featured_slugs()[0]) is not None


def test_every_tier_default_is_on_the_shortlist() -> None:
    # Otherwise the selector opens on a model the run is not using, which is the
    # dead-knob failure in its most confusing form: the dial disagrees with the
    # process and neither is wrong.
    from src.agent.model_selection import DEFAULT_TIER_SLUGS, featured_slugs

    offered = set(featured_slugs())
    for tier, slug in DEFAULT_TIER_SLUGS.items():
        assert slug in offered, f"the {tier} tier's default {slug} is not offered"


def test_a_slug_the_catalogue_dropped_is_filtered_out_of_the_shortlist() -> None:
    # A retired slug left in the selector sits there until somebody picks it and
    # waits several minutes for the failure.
    import src.agent.model_selection as selection

    known = set(selection.selectable_slugs())
    assert all(slug in known for slug in selection.featured_slugs())


# --------------------------------------------------------------------------
# The chat page's override, end to end
# --------------------------------------------------------------------------


def test_moving_a_dial_on_the_chat_panel_changes_what_is_asked_with() -> None:
    """The override returns a different setting, and only where a dial was moved.

    The file's own principle applied to the one panel it did not reach: this branch
    is skipped unless its checkbox is ticked, so nothing had ever checked that its
    dials arrive anywhere. Both halves matter -- an override that changes nothing is
    a dead control, and one that changes a field nobody touched is worse.
    """
    from streamlit.testing.v1 import AppTest

    from src.ui.status import PROJECT_ROOT

    probe = PROJECT_ROOT / "src" / "ui" / "pages" / "chat.py"
    app = AppTest.from_file(str(probe), default_timeout=180)
    app.session_state["chat_override"] = True
    app.run()

    # By key rather than by label. The page draws the sidebar as well, so "Sites $L$"
    # matches two sliders and the first of them is the sidebar's -- setting that one
    # tests the control this panel exists to override rather than the panel itself.
    before = knob.Setting()
    app.slider(key="chat_n_sites").set_value(before.physics.n_sites + 4)
    app.slider(key="chat_depth").set_value(before.physics.depth + 3)
    app.run()
    assert not app.exception, [str(error.value) for error in app.exception]

    assert app.slider(key="chat_n_sites").value == before.physics.n_sites + 4
    assert app.slider(key="chat_depth").value == before.physics.depth + 3
    # Untouched dials stay where the sidebar left them.
    assert app.slider(key="chat_coupling").value == before.physics.coupling
    assert app.slider(key="chat_field").value == before.physics.field


# The chain the page is showing
# --------------------------------------------------------------------------


def test_the_chain_on_the_panel_answers_a_question_that_named_none() -> None:
    """The five physics dials, connected.

    They drew the circuit above the answer and they ran the Lab, and they reached a
    campaign nowhere -- so the page could show a ten-spin chain while the answer under
    it read *TFIM L=6*. This file's principle, applied to the dials that had least of
    it: a reader who moves one and sees the same model concludes it is broken.
    """
    silent = "How can a circuit run imaginary time?"
    on_screen = campaign(silent, chain_defaults=knob.Physics(n_sites=10, field=0.6).as_reading())[
        "model"
    ]
    without = campaign(silent)["model"]
    assert on_screen is not None and without is not None, "both campaigns formalised a chain"

    assert (on_screen.n_sites, on_screen.transverse_field) == (10, 0.6)
    assert (without.n_sites, without.transverse_field) == (6, 1.0)
    assert any("settings on screen" in note for note in on_screen.assumptions)
    # And a dial is still not a question: this one is consulted, not assessed.
    assert not on_screen.length_was_given


def test_the_panel_names_the_tilt_only_when_it_is_switched_on() -> None:
    # Naming a term to report that it is switched off is how a two-term problem comes
    # to look like a three-term one, and `g` is zero unless somebody moved it.
    assert knob.Physics().as_reading().longitudinal is None
    assert knob.Physics(longitudinal=0.3).as_reading().longitudinal == 0.3
