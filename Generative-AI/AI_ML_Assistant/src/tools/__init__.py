"""Tool-calling functions exposed to the LLM and runnable standalone from the UI.

Each tool lives in its own module and is a LangChain ``@tool``. To add a new tool,
create a module here and append its function to :data:`ALL_TOOLS` below — the chain and
the Tools page pick it up automatically.
"""

from __future__ import annotations

from src.tools.arxiv import search_arxiv
from src.tools.calculator import math_calculator
from src.tools.tokens import estimate_tokens_and_cost

# Order defines how tools are advertised to the model and listed in the UI.
ALL_TOOLS = [search_arxiv, estimate_tokens_and_cost, math_calculator]

__all__ = ["ALL_TOOLS", "estimate_tokens_and_cost", "math_calculator", "search_arxiv"]
