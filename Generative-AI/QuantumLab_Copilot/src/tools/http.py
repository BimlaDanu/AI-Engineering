"""The one way out to the internet, so there is one place to reason about it.

Three tools now reach outside the process, and they share this. Concentrating that
is the point: a project with one egress function has one place to set a timeout, a
size limit and a user agent, and one place to audit when the question is "what can
this application talk to?".

**No tool ever takes a URL.** Every caller passes a query, and the host is a
constant in the calling module. That is deliberate and it is the whole defence
against the obvious attack: a tool argument is chosen by a language model, which
can be talked into choosing things, and a model that could name a URL could be
talked into naming ``http://169.254.169.254/`` and reading a cloud instance's
credentials back to whoever asked. A model that can only name a *search term*
cannot, no matter how the question is phrased.

**Failure is a return value.** The network is the one dependency guaranteed to
break, and it must break as a caveat under an answer, never as a traceback in
place of one. Everything here returns ``None`` rather than raising, and the caller
turns that into a sentence.

Built on :mod:`urllib.request` rather than on ``httpx`` or ``requests``. Both are
installed -- as transitive dependencies of the model client, which is precisely the
problem. :file:`pyproject.toml` does not declare either, so importing one would
mean depending on a package a resolver update could quietly remove. The standard
library cannot vanish.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

from src.logging_setup import get_logger

DEFAULT_TIMEOUT_S = 8.0
"""How long to wait. Short: this runs while somebody watches a spinner."""

MAX_RESPONSE_BYTES = 400_000
"""Most of a response to read.

A ceiling rather than trust: an endpoint returning a hundred megabytes would
otherwise be a memory problem long before it was a parsing problem. Generous
enough for any search result page, small enough to be harmless.
"""

USER_AGENT = "QuantumLabCopilot/0.1 (educational use)"
"""Identifies this client honestly.

Wikipedia's API asks for a descriptive agent and rate-limits generic ones, so this
names the application and what it is for. It is sent to every host contacted, which
is the reason it says nothing else.
"""

Fetcher = Callable[[str], Any | None]
"""Something that turns a URL into parsed JSON, or ``None``.

The seam every network-facing tool is tested through: a test passes a function
returning a fixture and never opens a socket.
"""

LOG = get_logger("http")


def encode(value: str) -> str:
    """Percent-encode one query-string value.

    Args:
        value: The text to encode.

    Returns:
        The value, safe to interpolate into a query string. ``quote`` with an
        empty safe set, so a slash, an ampersand and a question mark in a search
        term cannot become URL structure.

    Examples:
        >>> encode("ising model & criticality")
        'ising%20model%20%26%20criticality'
    """
    return urllib.parse.quote(value, safe="")


def sanitise_query(query: str, *, limit: int) -> str:
    """Strip a model-chosen search term down to something safe to send.

    Every tool here puts a term the model wrote into somebody else's query
    language -- arXiv's search grammar, a URL path, a JSON body -- and every one
    of those grammars gives meaning to punctuation. The defence is the same one
    used against SQL injection: the untrusted part is not allowed to carry
    syntax at all.

    Args:
        query: What the model asked to search for.
        limit: Longest result to return, in characters. Each caller sets its own,
            because the services differ in what they will accept.

    Returns:
        Letters, digits, spaces, hyphens and full stops only, collapsed to single
        spaces and truncated. Empty if nothing survived, which callers read as
        "there is nothing to search for".

    Examples:
        >>> sanitise_query('ising) OR cat:econ.GN AND all:"free money"', limit=120)
        'ising OR cat econ.GN AND all free money'
        >>> sanitise_query("  transverse   field  ", limit=120)
        'transverse field'
        >>> sanitise_query("   ", limit=120)
        ''
    """
    kept = "".join(
        character if (character.isalnum() or character in " -.") else " " for character in query
    )
    return " ".join(kept.split())[:limit].strip()


def get_json(url: str, *, timeout: float = DEFAULT_TIMEOUT_S) -> Any | None:
    """Fetch a URL and parse the JSON it returns.

    Args:
        url: The address to fetch. Built by the caller from a fixed host and an
            encoded query -- never supplied by a model.
        timeout: Seconds to wait before giving up.

    Returns:
        The parsed body, or ``None`` if anything at all went wrong: a refused
        connection, a timeout, a non-200 status, a truncated body, or a response
        that was not JSON. All five are the same thing to a caller -- no data --
        and distinguishing them would only spread the handling around.
    """
    return _fetch(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout)


def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> Any | None:
    """Post a JSON body and parse the JSON that comes back.

    Args:
        url: The address to post to, from configuration -- never from a model.
        payload: The body, serialised here so no caller hand-builds JSON.
        headers: Extra headers, such as an authorisation token.
        timeout: Seconds to wait before giving up.

    Returns:
        The parsed body, or ``None`` on any failure. See :func:`get_json`.
    """
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "User-Agent": USER_AGENT,
            "Content-Type": "application/json",
            # Both, because MCP's streamable-HTTP transport may answer either as
            # one JSON document or as a stream of server-sent events.
            "Accept": "application/json, text/event-stream",
            **(headers or {}),
        },
    )
    return _fetch(request, timeout)


def _fetch(request: urllib.request.Request, timeout: float) -> Any | None:
    """Perform one request and parse the response.

    Args:
        request: The prepared request.
        timeout: Seconds to wait.

    Returns:
        The parsed body, or ``None``.
    """
    try:
        # `with` so the socket is closed even when parsing raises: a leaked
        # connection per failed search adds up in a long-lived server process.
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read(MAX_RESPONSE_BYTES)
        return parse_body(body)
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as error:
        LOG.warning("fetch_failed", extra={"error_type": type(error).__name__})
        return None


def parse_body(body: bytes) -> Any:
    r"""Parse a response body that may be JSON or server-sent events.

    MCP servers are allowed to answer a JSON-RPC request either way, and which
    one you get depends on the server rather than on the request. Reading both
    here means no caller has to know which it is talking to.

    Args:
        body: The raw response.

    Returns:
        The parsed document. For an event stream, the payload of the last
        ``data:`` line, which for a single JSON-RPC reply is the reply.

    Raises:
        ValueError: The body was neither, which :func:`_fetch` turns into
            ``None``.

    Examples:
        >>> parse_body(b'{"ok": true}')
        {'ok': True}
        >>> parse_body(b'event: message\ndata: {"ok": true}\n\n')
        {'ok': True}
    """
    text = body.decode("utf-8", errors="replace").strip()
    if not text.startswith("event:") and not text.startswith("data:"):
        return json.loads(text)
    payloads = [
        line.removeprefix("data:").strip() for line in text.splitlines() if line.startswith("data:")
    ]
    if not payloads:
        raise ValueError("event stream carried no data")
    return json.loads(payloads[-1])
