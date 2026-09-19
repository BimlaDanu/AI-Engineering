"""The shared model picker: one selectbox, rendered wherever the active model is chosen.

Kept in one place so ⚙️ Settings and the 💬 Chat answer-settings popover offer the *same*
list, the same labels and the same price hint without duplicating logic — the pattern
:mod:`src.ui.exporters` already follows for the download buttons.

Both call sites bind the widget to ``st.session_state["model"]``, which is the single key
:mod:`src.app` reads when it builds the per-run ``ctx``. Sharing the key rather than keeping a
per-page copy is what makes the two pickers one setting seen from two places: change the model
beside the question you are about to ask, and ⚙️ Settings already agrees. Streamlit renders
one page per run, so the two widgets never collide.
"""

from __future__ import annotations

import streamlit as st

from src.config import MODEL_BY_ID, OPENROUTER_MODELS, google_api_key
from src.utils import estimate_cost

# Representative answer size used only to preview a model's relative cost in the picker: a
# retrieval-augmented prompt (query + a few cited chunks) plus a medium completion. It anchors
# the "≈ $/answer" readout so models can be compared at a glance — not a billing figure.
TYPICAL_INPUT_TOKENS = 2000
TYPICAL_OUTPUT_TOKENS = 500


def available_models() -> list[str]:
    """Every model this deployment can actually reach, in registry (cheapest-first) order.

    ``gemini-native`` goes straight to Google rather than through OpenRouter, so it is only
    offered when a ``GOOGLE_API_KEY`` is set — listing it without one would be a choice that
    fails at the first question.
    """
    return OPENROUTER_MODELS + (["gemini-native"] if google_api_key() else [])


def model_option(model_id: str) -> str:
    """Format a model id for the picker as ``Label · $in/$out per 1M`` (falls back to the id)."""
    spec = MODEL_BY_ID.get(model_id)
    if spec is None:
        return model_id
    return f"{spec.label} · ${spec.price_in:g}/${spec.price_out:g} per 1M"


def model_label(model_id: str) -> str:
    """The model's short display name, for status lines that have no room for the price."""
    spec = MODEL_BY_ID.get(model_id)
    return spec.label if spec else model_id


def model_picker(label: str = "Active model", *, help: str | None = None) -> str:
    """Draw the model selectbox bound to ``st.session_state["model"]`` and return the choice.

    Args:
        label: The widget label. Callers differ: ⚙️ Settings heads a whole section with it,
            the Chat popover sits among four other answer controls.
        help: Tooltip override; the default explains the price suffix in the option labels.

    The persisted choice is clamped to what is reachable *now* — a Gemini key that disappears
    between runs would otherwise leave the session pointing at a model it cannot call.
    """
    models = available_models()
    if st.session_state.get("model") not in models:
        st.session_state.model = models[0]
    st.selectbox(
        label,
        models,
        key="model",
        format_func=model_option,
        help=help or "Each option shows its rough price per 1M tokens (in / out).",
    )
    return st.session_state.model


def model_cost_hint(model_id: str) -> None:
    """Show the selected model's positioning note and an approximate per-answer cost."""
    spec = MODEL_BY_ID.get(model_id)
    if spec is None:
        return
    per_answer = estimate_cost(model_id, TYPICAL_INPUT_TOKENS, TYPICAL_OUTPUT_TOKENS)
    st.caption(
        f"{spec.note}  \n"
        f"≈ **${per_answer:.4f}** per typical answer "
        f"(~{TYPICAL_INPUT_TOKENS // 1000}k tokens in / {TYPICAL_OUTPUT_TOKENS} out). "
        "Prices are approximate — actual session cost is tracked in **📈 Analytics**."
    )
