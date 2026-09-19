"""Offline tests for the Synapse MCP server (:mod:`src.mcp_server`).

These pin the two properties that matter for B4:

* **Coverage + schema** — the server advertises exactly the app's built-in tools, by the
  same names and descriptions the in-app LLM sees (single source of truth).
* **Execution parity** — calling a tool *through* the MCP server returns the same result as
  calling the tool's underlying function directly, proving the server merely re-exports the
  existing logic rather than a divergent copy.

No transport is started and no network is touched: we build the server in-process and drive
its async ``list_tools`` / ``call_tool`` handlers directly. Only offline, deterministic tools
(the SymPy calculator and the token estimator) are exercised end to end — arXiv, which hits
the network, is checked for *registration* only.
"""

from __future__ import annotations

import asyncio

from src.mcp_server import SERVER_NAME, build_server
from src.tools import ALL_TOOLS
from src.tools.calculator import math_calculator
from src.tools.tokens import estimate_tokens_and_cost


def test_server_has_expected_name() -> None:
    assert build_server().name == SERVER_NAME


def test_server_exposes_every_builtin_tool_by_name() -> None:
    server = build_server()
    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert names == {t.name for t in ALL_TOOLS}


def test_advertised_descriptions_match_the_source_tools() -> None:
    # The MCP description must be the tool's own docstring-derived description, not a copy
    # that can drift — so a client sees exactly what the in-app LLM sees.
    server = build_server()
    tools = {t.name: t for t in asyncio.run(server.list_tools())}
    for source_tool in ALL_TOOLS:
        assert tools[source_tool.name].description == source_tool.description


def test_every_tool_advertises_an_input_schema() -> None:
    # FastMCP derives the JSON schema from each function's typed signature; every tool takes
    # at least one argument, so the schema must declare properties.
    server = build_server()
    for tool in asyncio.run(server.list_tools()):
        assert tool.inputSchema.get("properties")


def _call(server, name: str, arguments: dict) -> str:
    """Invoke a tool through the server and return its structured string result."""
    _content, structured = asyncio.run(server.call_tool(name, arguments))
    return structured["result"]


def test_calculator_through_server_matches_direct_call() -> None:
    server = build_server()
    args = {"expression": "integrate(2*x, (x, 0, 3))"}
    through_server = _call(server, "math_calculator", args)
    direct = math_calculator.func(**args)
    assert through_server == direct
    assert "9" in through_server  # ∫₀³ 2x dx = 9 — sanity check the real tool ran


def test_token_estimator_through_server_matches_direct_call() -> None:
    server = build_server()
    args = {"text": "the quick brown fox", "model": "openai/gpt-4o-mini"}
    assert _call(server, "estimate_tokens_and_cost", args) == estimate_tokens_and_cost.func(**args)
