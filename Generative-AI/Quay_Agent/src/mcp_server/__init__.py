"""Serving this project's tools over the Model Context Protocol.

The agent in ``src/agent`` calls its tools directly, in process. This package
offers the same eight tools to anything that speaks MCP -- a desktop assistant, an
IDE, another agent -- without either side importing the other. The tools are
defined once, in :mod:`src.agent.tools`, and both callers read that one tuple; a
tool added there appears here with no further work, and a tool that nobody listed
gains no power in either place.

Why the protocol is implemented here rather than imported
---------------------------------------------------------

MCP over standard input and output is JSON-RPC 2.0 with a small, fixed method set.
Written out it is a few hundred lines of dictionary handling with no dependency, no
transitive tree, and no version to keep in step with the rest of the application --
and, more usefully, every part of it can be tested by feeding a line of JSON in and
asserting on the line that comes out. A client library would have made the same
tests require a running process.

The parts of the specification this implements are the parts a client needs to
discover and call tools: the handshake, ``tools/list``, ``tools/call``,
``resources/list``, ``resources/read`` and ``ping``. Sampling, prompts,
subscriptions and progress notifications are not implemented, and an unknown method
is answered with a proper JSON-RPC "method not found" rather than silence.

What a client is trusted with
-----------------------------

Nothing. Arguments arriving over the protocol are validated against each tool's own
schema before the tool sees them, and any string among them is screened the same way
a typed question is. Seven of the eight tools are arithmetic and simulation over a
spin chain; ``search_arxiv`` reaches a public preprint index and screens everything it
returns. None writes a file, spends a credential or calls a language model, so the
worst a malformed call can do is produce an error object, which is the shape the
protocol has for exactly that.

======================  =====================================================
Module                  What it holds
======================  =====================================================
``protocol.py``         the JSON-RPC envelope and the MCP method handlers
``server.py``           the standard-input loop, and the entry point
======================  =====================================================
"""
