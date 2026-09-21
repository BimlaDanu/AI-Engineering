"""The standard-input loop that carries the protocol.

One line of JSON in, at most one line of JSON out. That is the whole transport,
and keeping it that thin is deliberate: everything that could be wrong about a
message is decided in :mod:`src.mcp_server.protocol`, which needs no process to
test, and what is left here is reading, writing and knowing when to stop.

Three things this loop has to get right, none of them obvious
-------------------------------------------------------------

Nothing but protocol may reach standard output. A client parses every line it
receives as JSON, so one stray ``print`` -- a warning, a progress line, a
library's banner -- ends the session with a parse error the client cannot explain.
Logging is configured onto standard error for exactly this reason, and the
configuration happens before the first message is read.

Output is flushed after every message. Standard output is block-buffered when
it is a pipe rather than a terminal, which a client always is. Without the flush
the first response sits in a buffer while the client waits for it and the
handshake never completes -- a hang with no error anywhere.

End of input is a normal exit. A client closing the pipe is how a session
ends. Treating it as a failure would put an error in the client's log every time a
user closed a window.

Running it
----------

::

    python -m src.mcp_server

A client is configured with that command, this repository as the working
directory, and nothing else. The server needs no credential: no tool here calls a
language model.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable
from typing import TextIO

from src.logging_setup import configure_logging, get_logger
from src.mcp_server.protocol import PARSE_ERROR, decode, fail, handle

_log = get_logger("mcp.server")

EXIT_OK = 0


def serve(source: Iterable[str], sink: TextIO) -> int:
    """Answer every message on ``source``, writing responses to ``sink``.

    Kept separate from :func:`main` so that a test can drive it with a list of
    strings and a buffer. A loop that could only be exercised by starting a
    subprocess is a loop whose edge cases go untested.

    Args:
        source: Lines of input, one message each.
        sink: Where to write responses.

    Returns:
        The process exit code.
    """
    for line in source:
        request = decode(line)
        if request is None:
            if line.strip():
                _write(sink, fail(None, PARSE_ERROR, "the line was not a JSON object"))
            continue
        response = handle(request)
        if response is not None:
            _write(sink, response)
    _log.info("mcp_stream_closed")
    return EXIT_OK


def _write(sink: TextIO, message: dict[str, object]) -> None:
    """Write one response and push it out of the buffer.

    Args:
        sink: Where to write.
        message: The response object.
    """
    sink.write(json.dumps(message) + "\n")
    sink.flush()


def main() -> int:
    """Serve the protocol on standard input and output.

    Returns:
        The process exit code.
    """
    # Onto standard error, before anything is read. Standard output belongs to the
    # protocol and a single log line written there would end the session.
    configure_logging(stream=sys.stderr)
    _log.info("mcp_server_start")
    return serve(sys.stdin, sys.stdout)


if __name__ == "__main__":
    sys.exit(main())
