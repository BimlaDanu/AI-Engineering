---
title: APIs, SDKs, and How Apps Talk to LLMs
subject: ai
topic: apis
difficulty: beginner
year: 2026
source: in-house study notes
---

# APIs, SDKs, and how apps talk to LLMs

## What is an API?

An **API (Application Programming Interface)** is a contract that lets one program use
another program's capabilities without knowing how it works inside. Think of a restaurant:
you (the client) don't walk into the kitchen — you hand the waiter (the API) a request from
a fixed menu, and the kitchen (the server) returns a dish in a predictable format. The menu
is the API's **specification**: which requests exist, what inputs they need, and what shape
the response will have.

In AI engineering, "the API" almost always means a **web API**: your code sends an HTTP
request over the internet to a provider's server (OpenAI, Anthropic, Google, OpenRouter, …)
and receives a response, usually as **JSON** — a simple text format of nested keys and
values that every language can parse.

## Anatomy of an LLM API call

A chat-completion request has a few standard parts:

- **Endpoint** — the URL that identifies the capability, e.g.
  `https://openrouter.ai/api/v1/chat/completions`. One provider exposes many endpoints
  (chat, embeddings, moderation, …).
- **Method** — the HTTP verb. LLM calls are `POST` (send data, get a result); read-only
  lookups are usually `GET`.
- **Headers** — metadata, most importantly `Authorization: Bearer <API key>`.
- **Body** — JSON payload: the `model` name, the list of `messages`
  (`system` / `user` / `assistant` roles), and parameters like `temperature` or
  `max_tokens`.
- **Response** — JSON containing the assistant's message, a `finish_reason`, and a
  `usage` block with input/output **token** counts — which is exactly what providers
  bill you for.

## API keys, security, and .env files

An **API key** is a secret string that identifies (and bills) your account. Treat it like a
password: keep it in a `.env` file or environment variable, never hard-code it, never
commit it to git, and never print it in logs. Anyone holding the key can spend your money.

## Rate limits and errors

Providers cap how fast you may call them (requests and tokens per minute). Exceeding the
cap returns HTTP **429 Too Many Requests**; robust clients respond with **exponential
backoff** (wait, retry, wait longer). Other statuses worth recognising: **401**
(bad/missing key), **400** (malformed request), **500/503** (provider trouble — retry
later). Production apps must handle all of these gracefully instead of crashing.

## SDKs: APIs made comfortable

An **SDK (Software Development Kit)** is a provider-maintained library that wraps the raw
HTTP calls in native functions — `client.chat.completions.create(...)` in Python instead of
hand-building requests. SDKs handle authentication, retries, timeouts, streaming, and type
checking. Frameworks like **LangChain** sit one level higher still, adding prompt
templates, tool calling, retrievers, and memory on top of any provider's SDK.

## The OpenAI-compatible convention and gateways

OpenAI's chat-completions request format became a de-facto standard, so many providers and
gateways accept the *same* JSON shape. **OpenRouter** exploits this: it is one
OpenAI-compatible endpoint that routes to hundreds of models from different labs — you
switch models by changing a string, not your code. This very assistant calls OpenRouter
through LangChain's `ChatOpenAI` client with a custom `base_url`.

## Beyond one-shot calls

Modern LLM APIs also support **streaming** (tokens arrive as they are generated, via
server-sent events), **function/tool calling** (the model returns a structured request to
run one of your declared functions, you execute it and send the result back), and
**structured outputs** (force the reply to match a JSON schema).

## Key takeaways

- An API is a contract for using someone else's software; LLM APIs are web APIs speaking
  JSON over HTTPS.
- A call = endpoint + method + auth header + JSON body; the response includes token usage,
  which is what you pay for.
- Keep API keys in `.env`; handle 429/401/500 errors with backoff and clear messages.
- SDKs wrap the HTTP details; OpenAI-compatible gateways like OpenRouter make models
  swappable behind one interface.
