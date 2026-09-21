"""An MCP client, written to the protocol rather than to a library.

The Model Context Protocol is how an agent borrows somebody else's tools. Its
streamable-HTTP transport is JSON-RPC 2.0 over ``POST``, and that is a wire format
rather than a technology: ``initialize`` to open a session, ``tools/list`` to find
out what is there, ``tools/call`` to use one. Three requests, no SDK. Written this
way here because the reference implementation is a dependency this project does not
declare, and a protocol that can only be spoken through one library is not much of
a protocol.

**Discovery, then a validated call.** The model never names a remote tool from
imagination: :meth:`McpServer.tools` lists what the server advertises, those names
are what the tool description offers, and a call to anything else is refused
before a request is sent. A remote server is not trusted to be sane -- its tool
list is third-party text and is neutralised like any other.

**One string argument, deliberately.** A remote tool taking a single text
parameter is called; one demanding structured arguments is refused with the reason.
The alternative is letting a model fill in an arbitrary schema it has just been
handed by a third party, which is a large step past what this project is willing
to do unattended. The refusal names the tool and its parameters, so a human can
see what was on offer.

**No server, no tool.** There is no default endpoint. Unconfigured, this is not
described to the model at all, exactly as with web search.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src import security
from src.logging_setup import get_logger
from src.tools.http import post_json

ENDPOINT_VARIABLE = "MCP_SERVER_URL"
"""Environment variable holding the MCP server's streamable-HTTP endpoint."""

PROTOCOL_VERSION = "2025-06-18"
"""The protocol revision this client speaks, sent in ``initialize``."""

CLIENT_NAME = "quantumlab-copilot"
"""How this client identifies itself to a server."""

MAX_TOOLS = 12
"""Most remote tools to advertise. A server offering hundreds is summarised."""

MAX_RESULT_CHARACTERS = 1200
"""How much of a remote tool's output to keep."""

Transport = Callable[[str, dict[str, Any]], Any | None]
"""Sends one JSON-RPC request and returns the parsed reply, or ``None``.

The seam every test uses: a fake transport answers ``tools/list`` and
``tools/call`` from a fixture, and no socket is opened.
"""

LOG = get_logger("tools.mcp")


@dataclass(frozen=True, slots=True)
class RemoteTool:
    """One tool a server advertises.

    Attributes:
        name: What to call it.
        description: What the server says it does, neutralised.
        parameters: The names of its input properties, in schema order.
        required: Which of those the server marks as required.
    """

    name: str
    description: str = ""
    parameters: tuple[str, ...] = ()
    required: tuple[str, ...] = ()

    @property
    def single_text_argument(self) -> str | None:
        """The one parameter this tool can be called with, if there is one.

        Returns:
            The parameter name, or ``None`` when the tool needs more than one
            argument -- which is the case this client refuses rather than guesses
            at. A tool with no required parameters is callable with its first
            optional one, and a tool with exactly one required parameter is
            callable with that.
        """
        if len(self.required) == 1:
            return self.required[0]
        if not self.required and len(self.parameters) == 1:
            return self.parameters[0]
        return None

    def summary(self) -> str:
        """The tool as one line for a model to read.

        Returns:
            Its name, what it does, and what it takes.

        Examples:
            >>> RemoteTool("get_weather", "Weather now", ("city",), ("city",)).summary()
            'get_weather(city) — Weather now'
        """
        signature = ", ".join(self.parameters)
        detail = f" — {self.description}" if self.description else ""
        return f"{self.name}({signature}){detail}"


@dataclass(frozen=True, slots=True)
class McpCall:
    """What one remote call produced, including the ways it produced nothing.

    Attributes:
        tool: The remote tool named.
        argument: The text sent to it.
        output: What came back, truncated and neutralised.
        detail: Why there is nothing, when there is nothing.
    """

    tool: str
    argument: str = ""
    output: str = ""
    detail: str = ""

    @property
    def ok(self) -> bool:
        """Whether the call returned something usable."""
        return bool(self.output)

    def context(self) -> str:
        """The output as material for a prompt.

        Returns:
            The result under a standing warning. A remote MCP server is somebody
            else's code answering over the network: nothing here can check what it
            says, and it is labelled at the same level as web search.
        """
        if not self.output:
            return ""
        return (
            f"Output of the remote MCP tool {self.tool} (a third-party server, NOT "
            f"verified by this application):\n{self.output}"
        )

    def explain(self) -> str:
        """One line for the justification block.

        Returns:
            What was called and what came of it.

        Examples:
            >>> McpCall("get_weather", "Vilnius", detail="no such tool").explain()
            'mcp get_weather("Vilnius"): no such tool'
        """
        outcome = "returned output (unverified)" if self.output else (self.detail or "nothing")
        return f'mcp {self.tool}("{self.argument}"): {outcome}'


@dataclass(frozen=True, slots=True)
class McpServer:
    """A configured MCP endpoint, and the two operations worth having.

    Attributes:
        url: The server's streamable-HTTP endpoint.
        transport: How to send a request. Defaults to a live ``POST``.
    """

    url: str
    transport: Transport | None = None

    def _send(self, method: str, params: dict[str, Any]) -> Any | None:
        """Send one JSON-RPC request.

        Args:
            method: The method name, such as ``"tools/list"``.
            params: Its parameters.

        Returns:
            The ``result`` member of the reply, or ``None`` on any failure --
            including a JSON-RPC ``error`` member, which is a failure the transport
            itself reports as success.
        """
        if self.transport is not None:
            payload = self.transport(method, params)
        else:
            payload = post_json(
                self.url,
                {"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
            )
        if not isinstance(payload, dict) or "error" in payload:
            return None
        result = payload.get("result")
        return result if result is not None else payload

    def tools(self) -> tuple[RemoteTool, ...]:
        """Ask the server what it offers.

        Returns:
            The advertised tools, up to :data:`MAX_TOOLS`, or empty when the
            server is unreachable or answers with something unexpected.
        """
        result = self._send("tools/list", {})
        if not isinstance(result, dict):
            return ()
        listed = result.get("tools")
        if not isinstance(listed, list):
            return ()
        found = [_tool_of(entry) for entry in listed[:MAX_TOOLS]]
        return tuple(tool for tool in found if tool is not None)

    def call(self, tool: str, argument: str) -> McpCall:
        """Call one remote tool with one text argument.

        Never raises. An unknown name, a tool needing structured arguments, an
        unreachable server and an unreadable reply all come back as an
        :class:`McpCall` carrying the reason.

        Args:
            tool: The remote tool's name, checked against what the server
                advertises before anything is sent.
            argument: The text to pass.

        Returns:
            The call, successful or not.
        """
        wanted = tool.strip()
        available = self.tools()
        if not available:
            return McpCall(tool=wanted, detail="the MCP server offered no tools")

        match = next((entry for entry in available if entry.name == wanted), None)
        if match is None:
            offered = ", ".join(entry.name for entry in available)
            return McpCall(tool=wanted, detail=f"no such tool on that server; it offers {offered}")

        parameter = match.single_text_argument
        if parameter is None:
            return McpCall(
                tool=wanted,
                detail=(
                    f"{wanted} needs structured arguments ({', '.join(match.parameters)}), "
                    "which this client does not fill in unattended"
                ),
            )

        text = " ".join(argument.split())
        result = self._send("tools/call", {"name": wanted, "arguments": {parameter: text}})
        output = _output_of(result)
        LOG.info("mcp_call", extra={"tool": wanted, "ok": bool(output)})
        if not output:
            return McpCall(tool=wanted, argument=text, detail="the call returned nothing usable")
        return McpCall(tool=wanted, argument=text, output=output)


def endpoint() -> str | None:
    """Read the configured MCP endpoint from the environment.

    Returns:
        The URL, or ``None`` when unset. Blank counts as unset.
    """
    url = os.environ.get(ENDPOINT_VARIABLE, "").strip()
    return url or None


def is_configured() -> bool:
    """Whether an MCP server is configured at all.

    Returns:
        ``True`` when :data:`ENDPOINT_VARIABLE` is set. Read by
        :class:`src.tools.calling.Toolbox`, which does not describe this tool
        otherwise.
    """
    return endpoint() is not None


def _tool_of(entry: Any) -> RemoteTool | None:
    """Read one advertised tool out of a ``tools/list`` reply.

    Args:
        entry: One element of the server's ``tools`` array.

    Returns:
        The tool, or ``None`` if it has no name -- a nameless tool cannot be
        called, so it is not worth carrying.
    """
    if not isinstance(entry, dict):
        return None
    name = _clean(entry.get("name"), 80)
    if not name:
        return None
    schema = entry.get("inputSchema")
    properties = schema.get("properties") if isinstance(schema, dict) else None
    required = schema.get("required") if isinstance(schema, dict) else None
    parameters = tuple(str(key) for key in properties) if isinstance(properties, dict) else ()
    marked = tuple(str(key) for key in required) if isinstance(required, list) else ()
    return RemoteTool(
        name=name,
        description=_clean(entry.get("description"), 200),
        parameters=parameters,
        required=marked,
    )


def _output_of(result: Any) -> str:
    """Read the text out of a ``tools/call`` reply.

    MCP returns content as a list of typed blocks; only text blocks are read,
    because an image or an embedded resource is not something this application can
    put in a prompt.

    Args:
        result: The reply's ``result`` member.

    Returns:
        The joined text, neutralised and truncated, or ``""``.
    """
    if not isinstance(result, dict):
        return ""
    if result.get("isError") is True:
        return ""
    blocks = result.get("content")
    if not isinstance(blocks, list):
        return ""
    texts = [
        _clean(block.get("text"), MAX_RESULT_CHARACTERS)
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    joined = " ".join(text for text in texts if text)
    return joined[:MAX_RESULT_CHARACTERS].strip()


def _clean(value: object, limit: int) -> str:
    """Neutralise and truncate one field of a server's reply.

    Args:
        value: Whatever the server sent.
        limit: Longest string to keep.

    Returns:
        Text safe to show a model.
    """
    text = " ".join(security.neutralise(str(value or "")).split())
    return text[:limit].strip()
