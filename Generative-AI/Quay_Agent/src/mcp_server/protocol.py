"""The JSON-RPC envelope, and what each Model Context Protocol method answers.

Every function here is pure: a decoded request in, a decoded response out, no
input, no output, no process. That is what makes the protocol testable without
starting one -- a test writes a dictionary and reads a dictionary, and the
transport is somebody else's problem.

Three rules hold throughout, and each is a JSON-RPC requirement that is easy to
get subtly wrong.

A notification is never answered. A request with no ``id`` is a notification,
and replying to one corrupts the stream for a strict client. :func:`handle` returns
``None`` for those, and the caller writes nothing.

An error in the tool is not an error in the call. A tool that refuses a
malformed argument has done its job correctly, so it comes back as a successful
JSON-RPC response whose result carries ``isError``. Reserving protocol-level errors
for protocol-level problems is what lets a client tell "I sent something you could
not parse" apart from "the thing you asked for cannot be computed".

No exception escapes. Anything unforeseen becomes ``-32603 Internal error`` with
the exception type named and its message dropped. The type is enough to find the
bug in the log; the message can carry a path or a value from the environment, and
this is a channel to a program that is not this one.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import BaseTool

from src.agent.tools import TOOLS, get_tool, tool_names
from src.logging_setup import get_logger
from src.physics.method_catalogue import all_methods
from src.physics.model import HAMILTONIAN_WITH_LONGITUDINAL_LATEX
from src.security import screen

_log = get_logger("mcp.protocol")

JSONRPC_VERSION = "2.0"

PROTOCOL_VERSION = "2025-06-18"
"""The protocol revision this server implements.

Clients announce their own during the handshake. Where theirs is one this server
also knows, that one is echoed back and used; otherwise this one is offered and the
client decides whether it can live with it. Answering with the client's version
when it is unsupported would be a claim this server cannot honour.
"""

SUPPORTED_VERSIONS: frozenset[str] = frozenset({"2024-11-05", "2025-03-26", PROTOCOL_VERSION})

SERVER_NAME = "quay"
SERVER_VERSION = "0.1.0"

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
"""The JSON-RPC error codes. Standard values, not invented ones.

A client written against the specification branches on these numbers, so a server
that made up its own would be unreadable to every client but its own tests.
"""

PROBLEM_URI = "quay://problem-statement"
CATALOGUE_URI = "quay://method-catalogue"


def ok(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    """Build a successful JSON-RPC response.

    Args:
        request_id: The id from the request, echoed back verbatim. JSON-RPC allows
            a string or a number and a client may use either, so it is passed
            through rather than coerced.
        result: What the method produced.

    Returns:
        The response object.
    """
    return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "result": result}


def fail(request_id: Any, code: int, message: str) -> dict[str, Any]:
    """Build a JSON-RPC error response.

    Args:
        request_id: The id from the request, or ``None`` when the request could not
            be parsed far enough to have one.
        code: One of the codes above.
        message: A short description. Written for a developer reading a client's
            log, so it names what was wrong rather than apologising for it.

    Returns:
        The response object.
    """
    return {
        "jsonrpc": JSONRPC_VERSION,
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def tool_descriptors() -> list[dict[str, Any]]:
    """Describe every tool in the shape ``tools/list`` returns.

    The schema is taken from the tool's own argument model rather than written out
    again here. A second description would be a second thing to update, and the
    one that was missed would be the one a client reads.

    Returns:
        One descriptor per registered tool, in registration order.
    """
    descriptors: list[dict[str, Any]] = []
    for instrument in TOOLS:
        descriptors.append(
            {
                "name": instrument.name,
                "description": instrument.description,
                "inputSchema": _input_schema(instrument),
            }
        )
    return descriptors


def _input_schema(instrument: BaseTool) -> dict[str, Any]:
    """Read one tool's argument schema in the shape the protocol expects.

    Args:
        instrument: The tool.

    Returns:
        A JSON Schema object. A tool declared without an argument model is
        described as taking none, which is truthful and keeps a client from
        refusing to list the rest of them over one odd entry.
    """
    schema = instrument.args_schema
    if isinstance(schema, dict):
        return schema
    if schema is not None and hasattr(schema, "model_json_schema"):
        return dict(schema.model_json_schema())
    return {"type": "object", "properties": {}}


def resource_descriptors() -> list[dict[str, Any]]:
    """Describe the read-only material a client may fetch.

    Two resources, and both are context a client needs before a tool call means
    anything: what problem this server is about, and what methods exist for it.
    Neither is an answer -- the catalogue lists the exact solvers and states that
    they are the grader's, without carrying a way to run one.

    Returns:
        One descriptor per resource.
    """
    return [
        {
            "uri": PROBLEM_URI,
            "name": "The problem statement",
            "description": "The Hamiltonian this server is about, and the sign and "
            "operator conventions every tool result follows.",
            "mimeType": "text/markdown",
        },
        {
            "uri": CATALOGUE_URI,
            "name": "The method catalogue",
            "description": "Every solution method known to this project, with its cost "
            "class, its accuracy class and whether it can be run from here.",
            "mimeType": "text/markdown",
        },
    ]


def read_resource(uri: str) -> str:
    """Fetch one resource by URI.

    Args:
        uri: What to read.

    Returns:
        The resource as Markdown.

    Raises:
        KeyError: If the URI is not one this server serves. The caller turns that
            into ``-32602``, since asking for a resource that does not exist is a
            bad parameter rather than a broken server.
    """
    if uri == PROBLEM_URI:
        return (
            "# The problem\n\n"
            "Every tool on this server is about one model: the one-dimensional "
            "transverse-field Ising chain.\n\n"
            f"$$\n{HAMILTONIAN_WITH_LONGITUDINAL_LATEX}\n$$\n\n"
            "The operators are Pauli matrices with eigenvalues $\\pm 1$, not spin-half "
            "operators. $J$ makes neighbours agree, $g$ tilts every element the same "
            "way, and $h$ pushes sideways so that nothing settles. Energies are "
            "reported per element unless a result says otherwise, which is what makes "
            "two chain lengths comparable.\n\n"
            "The full three-term form is given here, unlike on the application's own "
            "pages, and the difference is deliberate. A page tells a reader what "
            "problem is being solved, and the application solves the two-term problem "
            "-- one ratio, $h/J$, with a closed-form answer to grade against. This is "
            "a *specification*, read by a client that may set any of the three, so it "
            "states all three. $g$ defaults to zero, and at $g = 0$ the chain is "
            'integrable: every honest feasibility verdict about it is "no". A '
            "non-zero $g$ is what makes the question open, and also what leaves the "
            "run with no exact answer to be marked against.\n"
        )
    if uri == CATALOGUE_URI:
        lines = [
            "# Methods",
            "",
            "| Method | Cost | Accuracy | Available here |",
            "| --- | --- | --- | --- |",
        ]
        lines.extend(
            f"| {facts.name} | {facts.cost} | {facts.accuracy} | "
            f"{'yes' if facts.availability == 'agent' else 'no, the grader only'} |"
            for facts in all_methods()
        )
        lines.extend(
            [
                "",
                "The methods marked as the grader's are exact solutions. This server "
                "can say that they exist -- an honest feasibility answer has to -- and "
                "holds no way to evaluate one.",
                "",
            ]
        )
        return "\n".join(lines)
    raise KeyError(uri)


def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Run one tool and wrap what it produced.

    Args:
        name: The tool to run.
        arguments: What to run it with, straight off the wire.

    Returns:
        The ``tools/call`` result. A refusal comes back with ``isError`` set rather
        than as a protocol error, because a tool that declined a bad argument has
        worked correctly.
    """
    try:
        instrument = get_tool(name)
    except KeyError:
        return _tool_error(f"unknown tool {name!r}; this server offers {', '.join(tool_names())}")

    hostile = _screen_arguments(arguments)
    if hostile:
        # The tools take numbers, so this should never fire. It is here because
        # "should never fire" is a property of today's tool list, and the day a
        # tool takes a string is the day nobody remembers to add the screen.
        _log.warning("mcp_arguments_blocked", extra={"tool": name, "categories": hostile})
        return _tool_error(f"the arguments were refused by the input screen: {', '.join(hostile)}")

    try:
        produced = instrument.invoke(arguments)
    except Exception as error:  # a tool refusing bad input is a result, not a fault
        _log.info("mcp_tool_refused", extra={"tool": name, "error_type": type(error).__name__})
        return _tool_error(f"{type(error).__name__}: {error}")

    # These tools report a refusal by returning ``{"error": ...}`` rather than by
    # raising, because the agent's own call sites read the sentence back to a
    # model. The protocol has a flag for the same thing, and a client that
    # branches on it would otherwise read every refusal as a success.
    refused = isinstance(produced, dict) and "error" in produced
    _log.info("mcp_tool_called", extra={"tool": name, "refused": refused})
    return {
        "content": [{"type": "text", "text": json.dumps(produced, indent=2, default=str)}],
        "isError": refused,
    }


def _screen_arguments(arguments: dict[str, Any]) -> list[str]:
    """Check every string argument for prompt injection.

    A tool result goes into the client's model context, so a string that passed
    through a tool unchanged would be an injection with a tool call wrapped round
    it -- which is a laundering step, not a defence.

    Args:
        arguments: What the client sent.

    Returns:
        The distinct attack categories found, or an empty list.
    """
    found: list[str] = []
    for value in arguments.values():
        if isinstance(value, str):
            found.extend(category for category in screen(value).categories if category not in found)
    return found


def _tool_error(message: str) -> dict[str, Any]:
    """Build a failed ``tools/call`` result.

    Args:
        message: What went wrong, in words a developer can act on.

    Returns:
        The result object, with ``isError`` set.
    """
    return {"content": [{"type": "text", "text": message}], "isError": True}


def handle(request: dict[str, Any]) -> dict[str, Any] | None:
    """Answer one decoded JSON-RPC request.

    Args:
        request: The decoded request object.

    Returns:
        The response, or ``None`` for a notification, which by specification is
        never answered.
    """
    request_id = request.get("id")
    is_notification = "id" not in request
    method = request.get("method")

    if not isinstance(method, str):
        return None if is_notification else fail(request_id, INVALID_REQUEST, "no method named")

    params = request.get("params")
    if params is None:
        params = {}
    if not isinstance(params, dict):
        return (
            None if is_notification else fail(request_id, INVALID_PARAMS, "params is not an object")
        )

    try:
        result = _dispatch(method, params)
    except KeyError as error:
        return None if is_notification else fail(request_id, INVALID_PARAMS, str(error))
    except _UnknownMethodError:
        return (
            None if is_notification else fail(request_id, METHOD_NOT_FOUND, f"no method {method!r}")
        )
    except Exception as error:  # never let a bug reach the client as a dead stream
        _log.warning("mcp_internal_error", extra={"method": method, "type": type(error).__name__})
        return (
            None
            if is_notification
            else fail(request_id, INTERNAL_ERROR, f"internal error: {type(error).__name__}")
        )

    return None if is_notification else ok(request_id, result)


class _UnknownMethodError(Exception):
    """Raised by the dispatch table for a method this server does not implement."""


def _dispatch(method: str, params: dict[str, Any]) -> dict[str, Any]:
    """Route one method to its handler.

    Args:
        method: The JSON-RPC method name.
        params: Its parameters.

    Returns:
        The result object.

    Raises:
        _UnknownMethodError: If the method is not implemented.
        KeyError: If a required parameter is missing or names something unknown.
    """
    if method == "initialize":
        return _initialize(params)
    if method == "ping":
        return {}
    if method == "tools/list":
        return {"tools": tool_descriptors()}
    if method == "tools/call":
        name = params.get("name")
        if not isinstance(name, str):
            raise KeyError("tools/call needs a string 'name'")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise KeyError("tools/call 'arguments' must be an object")
        return call_tool(name, arguments)
    if method == "resources/list":
        return {"resources": resource_descriptors()}
    if method == "resources/read":
        uri = params.get("uri")
        if not isinstance(uri, str):
            raise KeyError("resources/read needs a string 'uri'")
        try:
            text = read_resource(uri)
        except KeyError as error:
            raise KeyError(f"no resource at {uri!r}") from error
        return {"contents": [{"uri": uri, "mimeType": "text/markdown", "text": text}]}
    raise _UnknownMethodError(method)


def _initialize(params: dict[str, Any]) -> dict[str, Any]:
    """Answer the handshake.

    Args:
        params: What the client announced.

    Returns:
        The protocol version agreed on, what this server can do, and what it is.
        Only the capabilities actually implemented are declared: a server that
        advertises ``prompts`` and then answers "method not found" is worse than
        one that never mentioned them, because the client has already built its
        interface around the claim.
    """
    asked = params.get("protocolVersion")
    agreed = asked if isinstance(asked, str) and asked in SUPPORTED_VERSIONS else PROTOCOL_VERSION
    return {
        "protocolVersion": agreed,
        "capabilities": {"tools": {"listChanged": False}, "resources": {"listChanged": False}},
        "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
    }


def decode(line: str) -> dict[str, Any] | None:
    """Read one line of the stream into a request object.

    Args:
        line: One line of input, without its newline.

    Returns:
        The decoded object, or ``None`` if the line is blank, is not valid JSON, or
        is valid JSON that is not an object. The caller answers ``None`` with a
        parse error, which is the one case where a response carries a null id.
    """
    stripped = line.strip()
    if not stripped:
        return None
    try:
        loaded = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return loaded if isinstance(loaded, dict) else None
