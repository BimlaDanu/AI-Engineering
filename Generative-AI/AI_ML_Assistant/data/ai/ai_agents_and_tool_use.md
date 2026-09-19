---
title: AI Agents and Tool Use
subject: ai
topic: agents
difficulty: intermediate
year: 2026
source: In-house study note
---

# AI Agents and Tool Use

An **LLM agent** is a language model given the ability to *act*: rather than emitting one
answer, it runs a loop — **reason → choose an action → observe the result → repeat** — until
it can answer or hits a budget. Actions are **tools**: functions the model may call (search a
knowledge base, run a calculator, hit an API, read a file). This turns a static text predictor
into a system that can gather information and take multi-step actions.

## The plan–act–observe loop

Most agents implement a variant of **ReAct** (Reason + Act), interleaving natural-language
*thoughts* with tool *actions*:

1. **Think** — the model writes a short rationale about what to do next.
2. **Act** — it emits a structured tool call (a name + JSON arguments).
3. **Observe** — the tool runs and its output is fed back into the context.
4. Repeat until the model decides it has enough to answer — or a **step cap** stops it.

The step cap matters: without a hard bound an agent can loop forever or rack up cost. A good
design bounds iterations, tools, and tokens, and falls back to a best-effort grounded answer
when the budget runs out.

## Tool calling

Modern models support **structured tool calling**: you advertise each tool with a name,
description, and a JSON schema for its arguments; the model returns a validated call rather
than free-form prose you must parse. This is far more reliable — the arguments are typed and
schema-checked before execution. Tools should be **idempotent and side-effect-aware**, return
concise results (the model pays tokens to read them), and fail gracefully with a message the
model can reason about.

## Patterns beyond a single loop

- **Routing** — a cheap classifier decides *which* path a query needs (knowledge, tool, or a
  full agent loop), so simple questions skip the expensive machinery.
- **Retrieval-augmented agents** — the agent re-retrieves from a knowledge base mid-loop as
  its understanding of the question sharpens, instead of retrieving once up front.
- **Planner–executor** — one model drafts a multi-step plan, another executes each step.
- **Reflection** — the agent critiques its own draft and revises before finalising.

## Risks and guardrails

Agents amplify both capability and risk. **Prompt injection** via tool outputs or retrieved
text can hijack the loop, so screen inputs and treat tool results as untrusted. Bound the
loop, log every action for observability, require confirmation before irreversible actions,
and keep the agent **grounded** — every factual claim should trace to a tool result or cited
source, not the model's memory.
