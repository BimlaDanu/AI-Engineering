---
title: The Model Context Protocol (MCP)
subject: ai
topic: mcp
difficulty: intermediate
year: 2026
source: in-house study notes
---

# The Model Context Protocol (MCP)

## The problem it solves

LLMs are only as useful as the context they can see and the actions they can take. Before
MCP, every AI application integrated every data source with bespoke glue code: N assistants
× M tools meant N×M custom connectors, each with its own auth, format, and bugs.

The **Model Context Protocol (MCP)** — an open standard introduced by **Anthropic in
November 2024** — replaces that with one common interface: build one MCP server for your
tool, and *any* MCP-capable assistant can use it; build one MCP client into your assistant,
and it can use *any* MCP server. N×M collapses to N+M. The popular analogy: **MCP is the
"USB-C port" for AI applications** — one plug shape for every device. During 2025 it was
adopted well beyond Anthropic, including by OpenAI and Google DeepMind, making it the
de-facto standard for connecting assistants to external systems.

## Architecture: host, client, server

MCP is a **client–server protocol** with three roles:

- **Host** — the AI application the user interacts with (a chat app, an IDE assistant, an
  agent). It runs the LLM and decides what the model may access.
- **Client** — a connector inside the host that maintains a **one-to-one session** with a
  single server. A host with five servers runs five clients.
- **Server** — a small program exposing some capability: a filesystem, a database, a git
  repo, a web-search service, an internal company API.

Messages are **JSON-RPC 2.0**. Two transports are standard: **stdio** (the host launches
the server as a local subprocess — simple and private) and **HTTP** (for remote servers).
A session starts with an `initialize` handshake where both sides declare their
capabilities and protocol version.

## What a server offers: tools, resources, prompts

Servers expose three kinds of primitives, distinguished by *who controls their use*:

1. **Tools** — *model-controlled* functions the LLM decides to call (e.g.
   `search_flights`, `run_query`). Each declares a name, description, and a JSON schema
   for its inputs — very similar to ordinary function calling.
2. **Resources** — *application-controlled* data the host can read into context (files,
   table schemas, documents), addressed by URI.
3. **Prompts** — *user-controlled* reusable templates the user explicitly invokes (e.g. a
   "summarise this repo" slash command).

Clients can also grant servers capabilities in the other direction, most notably
**sampling** (the server asks the host's LLM to generate text on its behalf — keeping the
API key and model choice with the host).

## MCP vs plain tool calling

Ordinary **function calling** is provider- and app-specific: you register Python functions
with *your* app and *your* LLM SDK (as this assistant does with LangChain `@tool`
functions). **MCP standardises the boundary**: the tool lives in a separate process with a
uniform discovery mechanism (`tools/list`) and calling convention (`tools/call`), so the
same server works with any host, in any language, without redeploying the assistant. Under
the hood the LLM still sees tools the familiar way — MCP is plumbing between the app and
the tool, not a change to the model itself.

## Security considerations

MCP inherits classic risks and adds new ones: a malicious server can lie in its tool
descriptions (**tool poisoning**), returned content can carry **prompt injection**, and a
compromised server may exfiltrate whatever the host sends it. Good practice: only install
trusted servers, keep humans in the loop for destructive actions, scope credentials
narrowly, and treat all tool output as untrusted input.

## Key takeaways

- MCP is an open, JSON-RPC-based standard (Anthropic, Nov 2024) that turns N×M custom
  integrations into N+M reusable ones — "USB-C for AI".
- Hosts run clients; each client talks 1:1 to a server over stdio or HTTP.
- Servers expose **tools** (model-controlled), **resources** (app-controlled), and
  **prompts** (user-controlled).
- It standardises *how* tools are discovered and called; the LLM's tool-use behaviour
  stays the same. Treat third-party servers as untrusted by default.
