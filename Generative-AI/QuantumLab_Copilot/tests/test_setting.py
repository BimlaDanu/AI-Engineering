"""Tests for the settings knob.

Two properties matter here and neither is about widgets. First, **a ceiling
cannot be widened from the knob** -- the whole value of a principled refusal
disappears if a slider can raise the limit that produced it. Second, **applying
the knob leaves the process settings alone and carries the credential across
untouched**, because the frozen-settings guarantee is what makes a run
reproducible from its own trace.
"""

from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from src.agent.setting import (
    DEFAULT_SITES,
    MAX_PASSAGES,
    MODEL_CHOICES,
    SWEEP_POINTS_COSTLY_CEILING,
    ModelSetting,
    PhysicsSetting,
    RetrievalSetting,
    Setting,
)
from src.physics.model import MAX_SITES_STATEVECTOR
from src.rag.ingest import shelf_names
from src.rag.retrieve import DEFAULT_TOP_K, DEFAULT_VECTOR_SHARE, MAX_ROUNDS
from src.settings import DEFAULT_CHAT_MODEL, DEFAULT_TEMPERATURE, Settings


def base_settings(**overrides: object) -> Settings:
    """Build settings without touching the environment or a ``.env`` file.

    Args:
        **overrides: Fields to set explicitly.

    Returns:
        A valid :class:`~src.settings.Settings` carrying a dummy credential.
    """
    values: dict[str, object] = {"openrouter_api_key": SecretStr("test-key")}
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


# --- defaults -------------------------------------------------------------


def test_the_default_knob_describes_the_showcase_question() -> None:
    # An even ring admits both methods, so the default position is one where the
    # verification claim can actually be demonstrated -- and it is the shortest such
    # chain the finite-size panel still has something to say about, because every
    # added site doubles the state vector the page recomputes on each drag.
    knob = Setting()
    spec = knob.physics.spec()
    assert spec.n_sites == DEFAULT_SITES
    assert spec.n_sites % 2 == 0
    assert spec.boundary == "periodic"
    assert spec.ratio == pytest.approx(1.0)  # h = J, the critical point


def test_a_default_knob_reports_nothing_changed() -> None:
    assert Setting().changes_from_defaults() == ()


def test_the_default_model_matches_the_settings_default() -> None:
    # Two copies of a model slug are two things to forget to update together.
    assert ModelSetting().chat_model == DEFAULT_CHAT_MODEL
    assert base_settings().chat_model == DEFAULT_CHAT_MODEL


def test_every_offered_model_is_a_vendor_qualified_slug() -> None:
    # OpenRouter requires "vendor/model"; a bare name is a 400, and a 400 is not
    # retried, so a malformed entry here fails every question instantly.
    assert MODEL_CHOICES
    for slug in MODEL_CHOICES:
        assert slug.count("/") == 1, slug
        vendor, model = slug.split("/")
        assert vendor and model, slug


# --- ceilings cannot be widened ------------------------------------------


def test_the_chain_length_cap_cannot_be_exceeded() -> None:
    with pytest.raises(ValidationError):
        PhysicsSetting(n_sites=MAX_SITES_STATEVECTOR + 1)


def test_the_chain_length_cap_itself_is_reachable() -> None:
    # The cap is a limit, not a no-go area: L = 12 must still be selectable.
    assert PhysicsSetting(n_sites=MAX_SITES_STATEVECTOR).n_sites == MAX_SITES_STATEVECTOR


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("n_sites", 1),
        ("coupling", 0.0),
        ("coupling", -1.0),
        ("field", -0.1),
        ("sweep_points", 2),
        ("sweep_points", 10_000),
    ],
)
def test_a_physically_meaningless_dial_is_refused(field_name: str, value: object) -> None:
    with pytest.raises(ValidationError):
        PhysicsSetting(**{field_name: value})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("temperature", -0.1),
        ("temperature", 2.1),
        ("max_model_calls_per_run", 0),
        ("max_model_calls_per_run", 101),
        ("max_retries", -1),
        ("max_retries", 11),
        ("request_timeout_s", 0.0),
        ("max_output_tokens", 8),
        ("chat_model", ""),
    ],
)
def test_an_out_of_range_model_dial_is_refused(field_name: str, value: object) -> None:
    with pytest.raises(ValidationError):
        ModelSetting(**{field_name: value})  # type: ignore[arg-type]


def test_the_knob_bounds_match_the_settings_bounds() -> None:
    # The knob must not be able to reach a value Settings would then reject:
    # validation would fail at apply time, which is far from the slider that
    # caused it.
    knob = ModelSetting(
        temperature=2.0,
        max_model_calls_per_run=100,
        max_retries=10,
        requests_per_second=100.0,
        max_output_tokens=8192,
    )
    applied = Setting(model=knob).applied_to(base_settings())
    assert applied.temperature == 2.0
    assert applied.max_model_calls_per_run == 100


# --- an expensive sweep stays bounded ------------------------------------


def test_a_costly_sweep_is_clipped_however_high_the_resolution_goes() -> None:
    physics = PhysicsSetting(sweep_points=241)
    assert physics.points_for(costly=False) == 241
    assert physics.points_for(costly=True) == SWEEP_POINTS_COSTLY_CEILING


def test_a_low_resolution_is_not_raised_to_the_costly_ceiling() -> None:
    # The clip is a maximum, not a target: asking for 11 points must give 11.
    physics = PhysicsSetting(sweep_points=11)
    assert physics.points_for(costly=True) == 11


# --- applying the knob ----------------------------------------------------


def test_applying_the_knob_overrides_only_the_fields_it_owns() -> None:
    base = base_settings(embedding_model="openai/text-embedding-3-small")
    knob = Setting(model=ModelSetting(chat_model="openai/gpt-4o-mini", temperature=0.7))

    applied = knob.applied_to(base)

    assert applied.chat_model == "openai/gpt-4o-mini"
    assert applied.temperature == pytest.approx(0.7)
    # Untouched: the knob owns no embedding or tracing field.
    assert applied.embedding_model == base.embedding_model
    assert applied.langsmith_project == base.langsmith_project


def test_applying_the_knob_leaves_the_original_settings_unchanged() -> None:
    base = base_settings()
    Setting(model=ModelSetting(temperature=1.5)).applied_to(base)
    assert base.temperature == pytest.approx(DEFAULT_TEMPERATURE)


def test_the_slider_starts_where_the_process_starts() -> None:
    # Two defaults for one dial is how a knob comes to lie: the page reads 0.6
    # on the slider and the run is at something else. So the knob's default is the
    # process default, by construction rather than by coincidence.
    assert ModelSetting().temperature == pytest.approx(base_settings().temperature)
    assert ModelSetting().max_output_tokens == base_settings().max_output_tokens


def test_a_knob_left_alone_reports_no_change() -> None:
    # The corollary, and the one a reader sees: an untouched panel must not claim
    # the temperature was moved just because the default stopped being zero.
    assert Setting().changes_from_defaults() == ()


def test_applying_the_knob_preserves_the_credential() -> None:
    base = base_settings()
    applied = Setting().applied_to(base)
    assert applied.openrouter_api_key.get_secret_value() == "test-key"


def test_the_applied_settings_are_still_frozen() -> None:
    # A reproducible trace depends on configuration that cannot drift mid-run.
    applied = Setting().applied_to(base_settings())
    with pytest.raises(ValidationError):
        applied.temperature = 1.0


# --- reporting what moved -------------------------------------------------


def test_a_moved_dial_is_reported_in_plain_language() -> None:
    knob = Setting(model=ModelSetting(temperature=1.4))
    moved = knob.changes_from_defaults()
    assert len(moved) == 1
    assert "temperature" in moved[0]
    assert "1.4" in moved[0]


def test_physics_changes_are_listed_before_model_changes() -> None:
    # The number-changing dials are the ones a reader most needs to see first.
    knob = Setting(
        physics=PhysicsSetting(n_sites=DEFAULT_SITES + 2),
        model=ModelSetting(temperature=0.5),
    )
    moved = knob.changes_from_defaults()
    assert len(moved) == 2
    assert "n sites" in moved[0]
    assert "temperature" in moved[1]


def test_no_model_dial_can_change_the_problem_being_solved() -> None:
    # The claim the whole project rests on: the physics is computed, and the
    # language model only narrates it. Turning every model dial to an extreme
    # must leave the specification handed to the solver byte-identical.
    plain = Setting()
    extreme = Setting(
        model=ModelSetting(
            chat_model="openai/gpt-4o-mini",
            temperature=2.0,
            max_output_tokens=64,
            max_model_calls_per_run=100,
            max_retries=10,
        )
    )
    assert extreme.physics.spec() == plain.physics.spec()


def test_the_knob_is_frozen() -> None:
    # Settings shown next to an answer must be the settings that produced it.
    knob = Setting()
    with pytest.raises(ValidationError):
        knob.physics = PhysicsSetting(n_sites=4)


# --- the retrieval dials --------------------------------------------------


def test_the_retrieval_defaults_match_the_librarys_own() -> None:
    # Two copies of a default are two things to forget to update together: an
    # untouched interface must answer exactly as a direct call to retrieve does.
    knob = RetrievalSetting()
    assert knob.passages == DEFAULT_TOP_K
    assert knob.vector_share == pytest.approx(DEFAULT_VECTOR_SHARE)


def test_the_passage_count_cannot_be_raised_past_the_ceiling() -> None:
    with pytest.raises(ValidationError):
        RetrievalSetting(passages=MAX_PASSAGES + 1)


def test_an_answer_must_be_built_from_at_least_one_passage() -> None:
    # Zero passages is not a stricter search, it is a guaranteed refusal.
    with pytest.raises(ValidationError):
        RetrievalSetting(passages=0)


@pytest.mark.parametrize("share", [-0.1, 1.1])
def test_the_search_weighting_stays_a_share(share: float) -> None:
    # Outside [0, 1] one half would be weighted negatively, which is not "ignore
    # it" but "rank it upside down".
    with pytest.raises(ValidationError):
        RetrievalSetting(vector_share=share)


@pytest.mark.parametrize("share", [0.0, 0.5, 1.0])
def test_either_half_of_the_search_may_be_switched_off(share: float) -> None:
    # Both ends are legitimate positions, not out-of-range values: pure similarity
    # search is what this project had before the keyword half existed, and pure
    # keyword search is what still works with no credential.
    assert RetrievalSetting(vector_share=share).vector_share == pytest.approx(share)


def test_no_retrieval_dial_can_change_the_problem_being_solved() -> None:
    # The same claim as for the model dials. Retrieval changes what an answer
    # cites; the solver never reads a passage.
    extreme = Setting(retrieval=RetrievalSetting(passages=MAX_PASSAGES, vector_share=0.0))
    assert extreme.physics.spec() == Setting().physics.spec()


def test_the_corrective_loop_can_be_switched_off_but_not_extended() -> None:
    # One round is a legitimate position -- it is how a reader sees what the retry
    # was doing -- while a fourth round is a ceiling nobody may raise from a widget.
    assert RetrievalSetting(rounds=1).rounds == 1
    assert RetrievalSetting(rounds=MAX_ROUNDS).rounds == MAX_ROUNDS
    for refused in (0, MAX_ROUNDS + 1):
        with pytest.raises(ValidationError):
            RetrievalSetting(rounds=refused)


def test_the_router_keeps_the_shelf_choice_by_default() -> None:
    assert RetrievalSetting().shelf == ""


@pytest.mark.parametrize("name", list(shelf_names()))
def test_every_registered_shelf_can_be_forced(name: str) -> None:
    assert RetrievalSetting(shelf=name).shelf == name


def test_a_shelf_that_does_not_exist_is_refused() -> None:
    # A typo would become a filter matching nothing, and a search returning nothing
    # is indistinguishable from a corpus that does not cover the question.
    with pytest.raises(ValidationError):
        RetrievalSetting(shelf="physics")


def test_a_moved_retrieval_dial_is_reported_like_any_other() -> None:
    # The knob is collapsed by default, so a search weighted all the way to one end
    # has to be visible from outside it.
    moved = Setting(retrieval=RetrievalSetting(vector_share=1.0)).changes_from_defaults()
    assert len(moved) == 1
    assert "vector share" in moved[0]
