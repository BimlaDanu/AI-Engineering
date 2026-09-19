"""Offline tests for the chat-model registry and its derived views.

The multi-model switcher and cost accounting are driven by a single source of truth —
:data:`src.config.MODELS` (a tuple of :class:`~src.config.ModelSpec`). The legacy shapes the
rest of the app imports (``MODEL_PRICES``: id → (in, out); ``OPENROUTER_MODELS``: the
OpenRouter-routable subset) are *derived* from it. These tests pin that contract so the derived
views can never silently drift from the registry, and guard the invariants callers rely on:
prices are non-negative, ids are unique, and ``estimate_cost`` agrees with the registry.
"""

from __future__ import annotations

import pytest

from src import config
from src.config import MODEL_BY_ID, MODEL_PRICES, MODELS, OPENROUTER_MODELS
from src.utils import estimate_cost


def test_model_ids_are_unique():
    ids = [m.id for m in MODELS]
    assert len(ids) == len(set(ids)), f"duplicate model ids: {ids}"


def test_model_prices_is_derived_from_registry():
    # MODEL_PRICES must be exactly the id -> (price_in, price_out) projection of MODELS.
    assert {m.id: (m.price_in, m.price_out) for m in MODELS} == MODEL_PRICES


def test_model_by_id_covers_every_spec():
    assert {m.id: m for m in MODELS} == MODEL_BY_ID
    assert len(MODEL_BY_ID) == len(MODELS)


def test_the_native_gemini_option_calls_the_model_it_advertises():
    # The one entry whose id is a sentinel rather than a provider slug: the registry carries
    # its label and price, and src.llm decides what is actually called. Nothing else ties the
    # two together, and they had already come apart — the picker said "Gemini 2.5 Flash
    # (native)" and priced it while the call asked for 2.0 Flash, so the cost line described
    # a different model from the one that wrote the answer.
    from src.llm import GEMINI_NATIVE_MODEL

    spec = MODEL_BY_ID["gemini-native"]
    version = GEMINI_NATIVE_MODEL.removeprefix("gemini-").split("-")[0]  # "2.5"
    assert version in spec.label, (
        f"{spec.label!r} does not name the model actually called ({GEMINI_NATIVE_MODEL!r})"
    )


def test_openrouter_models_excludes_native_and_preserves_order():
    # gemini-native is the only non-OpenRouter entry; everything else stays in declared order.
    assert "gemini-native" not in OPENROUTER_MODELS
    assert [m.id for m in MODELS if m.id != "gemini-native"] == OPENROUTER_MODELS
    assert OPENROUTER_MODELS, "there must be at least one OpenRouter-routable model"


def test_prices_are_non_negative_and_have_labels():
    for m in MODELS:
        assert m.price_in >= 0 and m.price_out >= 0, f"{m.id} has a negative price"
        assert m.label.strip(), f"{m.id} has an empty label"


def test_estimate_cost_agrees_with_registry_for_every_model():
    # A 1M-input / 1M-output call should cost exactly (price_in + price_out) USD.
    for m in MODELS:
        expected = m.price_in + m.price_out
        assert estimate_cost(m.id, 1_000_000, 1_000_000) == pytest.approx(expected)


# --- Per-visitor credential override ("Use a key") ---------------------------------------------
# A visitor may paste their own OpenRouter key so their questions are billed to them. The
# override has to be reachable from `src/core`, which must never import Streamlit — so it is a
# thread-local in `src.config` rather than a session-state read. These pin the two properties
# that makes it safe: it wins over the environment for the run that set it, and it is invisible
# to every other thread, which on a deployment means every other visitor.


def test_a_pasted_key_wins_over_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-host-key")
    config.set_runtime_key("OPENROUTER_API_KEY", "sk-or-visitor-key")
    try:
        assert config.openrouter_api_key() == "sk-or-visitor-key"
    finally:
        config.set_runtime_key("OPENROUTER_API_KEY", None)


def test_clearing_the_override_falls_back_to_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-host-key")
    config.set_runtime_key("OPENROUTER_API_KEY", "sk-or-visitor-key")
    config.set_runtime_key("OPENROUTER_API_KEY", None)
    assert config.openrouter_api_key() == "sk-or-host-key"


def test_an_empty_override_is_treated_as_no_override(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-host-key")
    config.set_runtime_key("OPENROUTER_API_KEY", "")
    assert config.openrouter_api_key() == "sk-or-host-key"


def test_one_visitors_key_is_invisible_to_another_thread(monkeypatch) -> None:
    # The property that makes this safe to deploy. Streamlit runs each session's script in its
    # own thread; a plain module global here would let one visitor's pasted key pay for
    # another visitor's question.
    import threading

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-host-key")
    config.set_runtime_key("OPENROUTER_API_KEY", "sk-or-visitor-key")
    seen: list[str | None] = []

    def _other_session() -> None:
        seen.append(config.openrouter_api_key())

    thread = threading.Thread(target=_other_session)
    thread.start()
    thread.join()
    try:
        assert seen == ["sk-or-host-key"], "a pasted key leaked into another session"
    finally:
        config.set_runtime_key("OPENROUTER_API_KEY", None)


def test_the_google_key_honours_the_same_override(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "host-gemini-key")
    config.set_runtime_key("GOOGLE_API_KEY", "visitor-gemini-key")
    try:
        assert config.google_api_key() == "visitor-gemini-key"
    finally:
        config.set_runtime_key("GOOGLE_API_KEY", None)


def test_a_run_does_not_inherit_the_previous_runs_credentials(monkeypatch) -> None:
    # `apply_runtime_keys` is called once per script run on a thread the server reuses, so
    # whatever the previous run left in the override table is what this one starts with. It
    # used to re-state only OPENROUTER_API_KEY, which meant the "no run inherits the last
    # one's credential" guarantee covered one key of the two the app can read.
    from src import auth

    monkeypatch.setenv("GOOGLE_API_KEY", "host-gemini-key")
    monkeypatch.setattr(auth, "own_key", lambda: "")
    config.set_runtime_key("GOOGLE_API_KEY", "a-previous-visitors-gemini-key")

    auth.apply_runtime_keys()

    assert config.google_api_key() == "host-gemini-key", (
        "a key from an earlier run survived into this one"
    )


def test_applying_keys_still_publishes_the_visitors_own_key(monkeypatch) -> None:
    # The clearing must not take the current run's choice with it.
    from src import auth

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-host-key")
    monkeypatch.setattr(auth, "own_key", lambda: "sk-or-visitor-key")
    try:
        auth.apply_runtime_keys()
        assert config.openrouter_api_key() == "sk-or-visitor-key"
    finally:
        config.clear_runtime_keys()
