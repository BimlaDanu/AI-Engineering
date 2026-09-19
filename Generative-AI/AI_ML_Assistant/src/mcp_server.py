"""Synapse tools **as an MCP server** — expose our own tools to any MCP client.

The Model Context Protocol (MCP) is symmetric: :mod:`src.core.mcp_client` lets Synapse app
*consume* tools served by a remote MCP server, and this module is the mirror image — it
*serves* Synapse's own built-in tools (arXiv search, the SymPy calculator, token/cost
estimation) so any MCP client (Claude Desktop, another agent, an IDE) can call them exactly
as Synapse's LLM does.

Single source of truth: the tools are **not** re-implemented here. Each entry in
:data:`src.tools.ALL_TOOLS` is a LangChain ``@tool`` wrapping a plain typed function; we
register that underlying function (``.func``) with :class:`~mcp.server.fastmcp.FastMCP`,
which builds the MCP JSON schema from its signature and docstring. So the calculator's input
guards, the arXiv query builder, and every docstring stay defined in one place — change the
tool, and both the in-app LLM and this MCP server see the change.

Run it (the server speaks over a transport; it is not a web page):

* ``make mcp-serve``            → stdio transport (what Claude Desktop and most clients spawn)
* ``python -m src.mcp_server --http [--host H --port P]`` → streamable-HTTP transport

stdio makes no network connections of its own; it reads/writes the parent process's pipes.
Only the arXiv tool reaches the network, and only when a client actually calls it — the same
contract as inside the app.
"""

from __future__ import annotations

import argparse

from mcp.server.fastmcp import FastMCP

from src.tools import ALL_TOOLS

# Advertised server name; clients show this when listing available tool providers.
SERVER_NAME = "synapse-tools"

# Human-readable guidance surfaced to MCP clients that display server instructions.
SERVER_INSTRUCTIONS = (
    "Domain tools from Synapse, an AI/ML research assistant: search arXiv for papers, "
    "evaluate/simplify/solve maths with SymPy, and estimate token counts and model cost."
)


def build_server() -> FastMCP:
    """Build the MCP server exposing every tool in :data:`src.tools.ALL_TOOLS`.

    Each LangChain ``@tool`` is registered by its underlying function (``.func``) under the
    tool's own ``name`` and ``description``, so the MCP schema is derived from the same typed
    signature and docstring the in-app LLM sees — no logic or metadata is duplicated here.
    Pure and side-effect free (no transport is started), so it is safe to import in tests.
    """
    server = FastMCP(SERVER_NAME, instructions=SERVER_INSTRUCTIONS)
    for langchain_tool in ALL_TOOLS:
        server.add_tool(
            langchain_tool.func,
            name=langchain_tool.name,
            description=langchain_tool.description,
        )
    return server


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the transport-selection flags (stdio by default, ``--http`` for HTTP)."""
    parser = argparse.ArgumentParser(
        prog="python -m src.mcp_server",
        description="Serve Synapse's built-in tools over the Model Context Protocol.",
    )
    parser.add_argument(
        "--http",
        action="store_true",
        help="Serve over streamable HTTP instead of stdio (default: stdio).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host (with --http).")
    parser.add_argument("--port", type=int, default=8000, help="HTTP bind port (with --http).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: build the server and run it on the chosen transport (blocking)."""
    args = _parse_args(argv)
    server = build_server()
    if args.http:
        server.settings.host = args.host
        server.settings.port = args.port
        server.run(transport="streamable-http")
    else:
        server.run(transport="stdio")


if __name__ == "__main__":
    main()
