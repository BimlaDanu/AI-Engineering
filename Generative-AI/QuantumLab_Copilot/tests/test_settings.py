"""Tests for runtime configuration.

None of these read the developer's real ``.env``. Every case goes through
:func:`from_env` or :func:`build`, both of which pass ``_env_file=None``, and
the ``clean_env`` fixture clears the relevant variables first. So the suite
behaves identically on a laptop with credentials and in CI without them -- and
a passing test can never depend on a secret.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest
from pydantic import SecretStr, ValidationError

from src.settings import (
    CHROMA_MAX_COLLECTION_NAME,
    DEFAULT_BASE_URL,
    DEFAULT_CORPUS_PATH,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_VECTOR_STORE_PATH,
    Settings,
    get_settings,
)

VARIABLES = [
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL",
    "CHAT_MODEL",
    "EMBEDDING_MODEL",
    "TEMPERATURE",
    "REQUEST_TIMEOUT_S",
    "MAX_RETRIES",
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
    "LANGSMITH_TRACING",
    "CORPUS_PATH",
    "VECTOR_STORE_PATH",
    "CHROMA_COLLECTION",
]


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Remove every setting from the environment for the duration of a test."""
    for name in VARIABLES:
        monkeypatch.delenv(name, raising=False)
    yield


def from_env() -> Settings:
    """Construct settings from the process environment, ignoring any ``.env``."""
    return Settings(_env_file=None)  # type: ignore[call-arg]


def build(**overrides: object) -> Settings:
    """Construct settings from explicit values, ignoring any ``.env``."""
    values: dict[str, object] = {"openrouter_api_key": "test-key"}
    values.update(overrides)
    # Both ignores are unavoidable: pydantic-settings does not declare
    # `_env_file` in __init__, and a **dict cannot be matched to typed fields.
    return Settings(_env_file=None, **values)  # type: ignore[call-arg, arg-type]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def test_the_credential_is_read_from_the_environment(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-the-environment")
    assert from_env().openrouter_api_key.get_secret_value() == "from-the-environment"


def test_a_missing_credential_fails_at_startup(clean_env: None) -> None:
    # The whole point of validating here: no LLM call should ever be the thing
    # that discovers the key is absent.
    with pytest.raises(ValidationError, match="openrouter_api_key"):
        from_env()


def test_an_empty_credential_is_rejected_by_name(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "   ")
    with pytest.raises(ValidationError, match="OPENROUTER_API_KEY is set but empty"):
        from_env()


def test_variable_names_are_case_insensitive(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("openrouter_api_key", "lower-case")
    assert from_env().openrouter_api_key.get_secret_value() == "lower-case"


def test_unrelated_variables_are_ignored(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    # A shared .env holds keys for other projects; they must not break startup.
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("SOME_UNRELATED_TOOL_TOKEN", "irrelevant")
    assert from_env().openrouter_api_key.get_secret_value() == "k"


# --------------------------------------------------------------------------
# Secrecy
# --------------------------------------------------------------------------


def test_the_credential_is_a_secret_not_a_string(clean_env: None) -> None:
    assert isinstance(build().openrouter_api_key, SecretStr)


@pytest.mark.parametrize("render", [repr, str])
def test_rendering_the_settings_never_reveals_the_credential(
    clean_env: None, render: Callable[[object], str]
) -> None:
    # Settings objects reach tracebacks, logs and traces. Masking is the
    # difference between an incident and a stack trace.
    settings = build(openrouter_api_key="super-secret-value", langsmith_api_key="also-secret")
    text = render(settings)
    assert "super-secret-value" not in text
    assert "also-secret" not in text


# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------


def test_the_defaults_are_the_documented_ones(clean_env: None) -> None:
    settings = build()
    assert settings.openrouter_base_url == DEFAULT_BASE_URL
    assert settings.chat_model == "openai/gpt-5-mini"
    assert settings.embedding_model == "openai/text-embedding-3-small"


def test_sampling_sits_mid_scale_rather_than_at_an_end(clean_env: None) -> None:
    # This used to assert 0.0, on the argument that variation is a liability when
    # answers are checked against exact physics. Half of that was never true: no
    # number in an answer is sampled at all -- src.physics computes them and
    # src.verification.cross_check agrees on them before a model is called -- so
    # the temperature buys wording and cannot reach a result. Greedy decoding did
    # cost something, though: at zero, a routing or grading call that starts down
    # the wrong path takes the same wrong path every time.
    assert build().temperature == pytest.approx(DEFAULT_TEMPERATURE)
    assert 0.0 < build().temperature < 2.0


def test_a_reply_has_a_ceiling_and_it_is_not_a_tight_one(clean_env: None) -> None:
    # Unset invited a runaway generation; set low truncates a verified explanation
    # mid-sentence, which is the worse of the two failures.
    ceiling = build().max_output_tokens
    assert ceiling == DEFAULT_MAX_OUTPUT_TOKENS
    assert ceiling is not None and ceiling >= 2048


def test_the_reasoning_effort_stays_at_the_bottom_of_its_scale(clean_env: None) -> None:
    # The deliberate exception to the two above, and the reason is measured rather
    # than argued: at the middle of this scale the routing call spent 14.6 seconds
    # and 669 output tokens filling four short fields, against 3.3 seconds to fill
    # them identically at minimal -- about seven calls to an answer, so a
    # 23-second reply or a 99-second one. See src.settings.ReasoningEffort.
    assert build().reasoning_effort == "minimal"


def test_tracing_is_off_until_it_is_switched_on(clean_env: None) -> None:
    # A test run must never post to an external service by accident.
    assert build().langsmith_tracing is False
    assert build().tracing_enabled is False


def test_tracing_needs_both_the_switch_and_a_credential(clean_env: None) -> None:
    assert not build(langsmith_tracing=True).tracing_enabled
    assert not build(langsmith_api_key="k").tracing_enabled
    assert build(langsmith_tracing=True, langsmith_api_key="k").tracing_enabled


def test_the_environment_overrides_a_default(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("CHAT_MODEL", "openai/gpt-4.1-mini")
    assert from_env().chat_model == "openai/gpt-4.1-mini"


# --------------------------------------------------------------------------
# Validation of ranges
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", -0.1),
        ("temperature", 2.1),
        ("request_timeout_s", 0.0),
        ("request_timeout_s", -1.0),
        ("max_retries", -1),
        ("max_retries", 11),
    ],
)
def test_out_of_range_values_are_rejected(clean_env: None, field: str, value: float) -> None:
    with pytest.raises(ValidationError, match=field):
        build(**{field: value})


def test_settings_cannot_be_mutated_after_construction(clean_env: None) -> None:
    # Configuration read at startup must not drift mid-run, or a trace stops
    # describing the run that produced it.
    with pytest.raises(ValidationError):
        build().chat_model = "something/else"


# --------------------------------------------------------------------------
# The knowledge base
# --------------------------------------------------------------------------


def test_the_knowledge_base_paths_point_into_the_data_folder(clean_env: None) -> None:
    settings = build()
    assert settings.corpus_path == DEFAULT_CORPUS_PATH
    assert settings.vector_store_path == DEFAULT_VECTOR_STORE_PATH


def test_an_evaluation_run_can_redirect_the_index(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An eval build must be able to write a throwaway index without disturbing
    # the one the running app is serving from.
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("VECTOR_STORE_PATH", "data/eval-index")
    assert from_env().vector_store_path == "data/eval-index"


def test_the_collection_name_records_the_embedding_model(clean_env: None) -> None:
    # Vectors from two embedding models are coordinates in unrelated spaces.
    # Comparing them returns nonsense quietly, so the name keeps them apart.
    name = build(embedding_model="openai/text-embedding-3-small").collection_name
    assert name == "tfim-literature-openai-text-embedding-3-small"


def test_changing_the_embedding_model_changes_the_collection(clean_env: None) -> None:
    small = build(embedding_model="openai/text-embedding-3-small").collection_name
    large = build(embedding_model="openai/text-embedding-3-large").collection_name
    assert small != large


@pytest.mark.parametrize(
    "model",
    [
        "openai/text-embedding-3-small",
        "some.vendor/model_v2",
        "a/" + "very-long-model-name" * 6,
    ],
)
def test_every_collection_name_is_one_chroma_will_accept(clean_env: None, model: str) -> None:
    # Chroma validates the name itself and raises. Failing that check deep inside
    # ingestion, on a name the caller never typed, is a bad way to learn this.
    name = build(embedding_model=model).collection_name
    assert 3 <= len(name) <= CHROMA_MAX_COLLECTION_NAME
    assert name[0].isalnum() and name[-1].isalnum()
    assert all(char.isalnum() or char in "-_" for char in name)


# --------------------------------------------------------------------------
# The cached accessor
# --------------------------------------------------------------------------


def test_get_settings_parses_the_environment_once(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    get_settings.cache_clear()
    try:
        assert get_settings() is get_settings()
    finally:
        get_settings.cache_clear()
