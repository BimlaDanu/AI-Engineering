"""Token-count and cost-estimation tool."""

from __future__ import annotations

from langchain_core.tools import tool

from src.config import MODEL_PRICES
from src.utils import estimate_cost, estimate_tokens


@tool
def estimate_tokens_and_cost(text: str, model: str = "openai/gpt-4o-mini") -> str:
    """Estimate the token count of a text and its input cost for a given model.

    Args:
        text: The prompt or text to measure.
        model: Model id from the app's price table, e.g. 'openai/gpt-4o-mini'.
    """
    tokens = estimate_tokens(text)
    cost = estimate_cost(model, tokens, 0)
    note = "" if model in MODEL_PRICES else " (model not in price table — cost shown as $0)"
    return (
        f"≈ **{tokens} tokens** (heuristic estimate); "
        f"input cost for `{model}`: **${cost:.6f}**{note}"
    )
