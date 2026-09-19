"""Creates and configures the LLM. Uses OpenRouter by default, with optional support for Gemini."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from src.config import OPENROUTER_BASE_URL, google_api_key, openrouter_api_key

# What ``"gemini-native"`` actually calls on Google's own API. It is spelled out here rather
# than inline because it has to agree with the registry entry in :data:`src.config.MODELS`,
# which carries that option's label ("Gemini 2.5 Flash (native)") and its per-1M prices — and
# for a while it did not: the call asked for 2.0 Flash while the picker named 2.5 and priced
# it, so the cost readout was quoting one model for the answers of another.
GEMINI_NATIVE_MODEL = "gemini-2.5-flash"


def get_llm(
    model: str,
    temperature: float = 0.2,
    *,
    top_p: float = 1.0,
    max_tokens: int = 0,
) -> BaseChatModel:
    """Build a chat model: OpenRouter by default, native Gemini as an alternative.

    Args:
        model: Registry model id, or ``"gemini-native"`` for the Google backend.
        temperature: Sampling temperature (higher = more creative).
        top_p: Nucleus-sampling mass; ``1.0`` leaves it disabled.
        max_tokens: Hard cap on the completion length; ``0`` lets the model decide.
    """
    # Only forward the optional knobs when set away from their defaults, so a stray
    # value never overrides a provider default the user did not intend to touch.
    limit = max_tokens if max_tokens > 0 else None
    if model == "gemini-native":
        from langchain_google_genai import ChatGoogleGenerativeAI

        key = google_api_key()
        if not key:
            raise RuntimeError("GOOGLE_API_KEY is not set — add it to .env")
        return ChatGoogleGenerativeAI(
            model=GEMINI_NATIVE_MODEL,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=limit,
            google_api_key=key,
        )
    from langchain_openai import ChatOpenAI

    key = openrouter_api_key()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set — add it to .env")
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=limit,
        api_key=key,
        base_url=OPENROUTER_BASE_URL,
    )
