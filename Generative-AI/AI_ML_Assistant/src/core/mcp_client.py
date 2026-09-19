"""
Remote MCP (Model Context Protocol) client — remote tools for the answer loop.

The Model Context Protocol (MCP) lets the assistant call remote tools served over
HTTP, in addition to the local ``@tool`` functions in :mod:`src.tools`. This
module connects to a configured MCP server, converts its advertised tools into
LangChain tools using ``langchain-mcp-adapters``, and provides them to
:func:`src.generation.answer_question`, where they are used like the built-in
arXiv, calculator, and token tools.

Design constraints:
* **No hard dependency.** The adapter is imported only when needed. If
  ``langchain-mcp-adapters`` is unavailable, :func:`load_mcp_tools` returns
  ``[]`` and the application runs normally.
* **Opt-in.** A connection is made only when ``settings.enable_mcp`` is enabled
  and an MCP server URL is configured.
* **Fail-soft.** Import, connection, or protocol errors return ``[]`` instead
  of raising an exception, so an unavailable MCP server simply means no extra
  tools.

MCP tools are asynchronous. :func:`run_async` bridges them into Streamlit's
synchronous request flow and also allows
:func:`src.generation.answer_question` to execute async MCP tools when the
model selects them.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.config import RagSettings

# Remote tool *definitions* are stable for a given server, so cache them by URL to avoid a
# network round-trip on every question (and every Streamlit rerun). Cleared via clear_cache.
_TOOLS_CACHE: dict[str, list[Any]] = {}


def run_async(coro: Any) -> Any:
    """Run an awaitable to completion from synchronous code (Streamlit, the tool loop).

    Uses :func:`asyncio.run` when no event loop is active. If a loop is already running in
    this thread (rare under Streamlit, but possible), it runs the coroutine on a dedicated
    worker thread so we never call ``asyncio.run`` inside a live loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(coro)).result()


def _transport_for(url: str) -> str:
    """Guess the MCP transport from the URL (``/sse`` → SSE, otherwise streamable HTTP)."""
    return "sse" if url.rstrip("/").endswith("sse") else "streamable_http"


def load_mcp_tools(settings: RagSettings) -> list[Any]:
    """Return LangChain tools from the configured remote MCP server (``[]`` on any failure).

    Returns ``[]`` — never raises — when MCP is disabled, no URL is set, the adapter package
    is not installed, or the server cannot be reached, so callers can always do
    ``ALL_TOOLS + load_mcp_tools(settings)`` unconditionally.
    """
    if not getattr(settings, "enable_mcp", False):
        return []
    url = (getattr(settings, "mcp_server_url", "") or "").strip()
    if not url:
        return []
    if url in _TOOLS_CACHE:
        return _TOOLS_CACHE[url]
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except Exception:
        return []  # dependency not installed — feature stays dormant
    try:
        client = MultiServerMCPClient({"remote": {"url": url, "transport": _transport_for(url)}})
        tools = run_async(client.get_tools())
    except Exception:
        return []  # unreachable server / protocol error — degrade to no extra tools
    _TOOLS_CACHE[url] = tools
    return tools


def clear_cache() -> None:
    """Forget cached remote tool definitions (e.g. after changing the server URL)."""
    _TOOLS_CACHE.clear()
