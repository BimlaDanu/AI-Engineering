"""Offline tests for the remote MCP client wrapper (no dependency, no network required).

These pin the fail-soft contract of :mod:`src.core.mcp_client`: it must return ``[]`` — never
raise — whenever MCP is disabled, unconfigured, its optional dependency is missing, or the
server is unreachable. That contract is what lets the service call
``ALL_TOOLS + load_mcp_tools(settings)`` unconditionally. The sync bridge is exercised too.
"""

from __future__ import annotations

from src.config import RagSettings
from src.core import mcp_client
from src.core.mcp_client import _transport_for, load_mcp_tools, run_async


def test_run_async_executes_a_coroutine() -> None:
    """run_async drives an awaitable to completion from synchronous test code."""

    async def _double(x: int) -> int:
        return x * 2

    assert run_async(_double(21)) == 42


def test_disabled_returns_empty_without_touching_the_network() -> None:
    """MCP off -> no tools, no import, no connection attempt."""
    assert load_mcp_tools(RagSettings(enable_mcp=False)) == []


def test_enabled_but_no_url_returns_empty() -> None:
    """Enabled with a blank URL is a no-op, not an error."""
    assert load_mcp_tools(RagSettings(enable_mcp=True, mcp_server_url="")) == []


def test_missing_dependency_or_unreachable_is_graceful() -> None:
    """A bad server (or a missing adapter package) degrades to [] rather than raising."""
    settings = RagSettings(enable_mcp=True, mcp_server_url="https://mcp.invalid.example/mcp")
    mcp_client.clear_cache()
    assert load_mcp_tools(settings) == []


def test_transport_is_inferred_from_the_url() -> None:
    """/sse endpoints use SSE transport; everything else uses streamable HTTP."""
    assert _transport_for("https://host/sse") == "sse"
    assert _transport_for("https://host/sse/") == "sse"
    assert _transport_for("https://mcp.deepwiki.com/mcp") == "streamable_http"
