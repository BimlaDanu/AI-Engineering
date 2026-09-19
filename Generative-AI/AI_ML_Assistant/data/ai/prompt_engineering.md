---
title: Prompt Engineering
subject: ai
topic: prompting
difficulty: beginner
year: 2026
source: In-house study note
---

# Prompt Engineering

Prompt engineering is the craft of writing inputs that reliably steer an LLM toward the
output you want. The model continues text according to patterns it learned; a prompt sets
up the pattern to continue.

## Core techniques

**Zero-shot:** just ask, clearly. Modern instruction-tuned models handle direct requests
well. Specify the role, the task, the format, and the constraints:
"You are a code reviewer. List up to 3 bugs in this function as bullet points."

**Few-shot:** include 2–5 worked examples of input → output before the real input. The model
infers the format and style from the examples — often worth more than paragraphs of
instructions, especially for structured outputs or unusual formats.

**Chain-of-thought (CoT):** ask the model to reason step by step before answering
("think through this step by step"). Dramatically improves arithmetic, logic, and multi-hop
questions, because intermediate tokens give the model computation space. Reasoning-tuned
models now do this internally.

**System prompts** set persistent behaviour (persona, rules, output policy) separately from
the user's message, and take priority over user text in well-aligned models — which is why
apps put their security rules there.

## Structure that helps

- Put instructions **before** the data they apply to; repeat the key instruction at the end
  for long prompts.
- Delimit data clearly (triple backticks, XML-style tags) so instructions and content don't
  blur — this also defends against prompt injection hiding inside pasted content.
- Ask for structured output (JSON with a given schema) when a program will parse the answer.
- Say what to do, not only what to avoid; give an escape hatch ("if unsure, say 'I don't
  know'") to reduce hallucination.

## Parameters that interact with prompting

**Temperature** scales randomness: ~0 for deterministic, factual, or extraction tasks;
0.7–1.0 for creative variety. **Top-p** (nucleus sampling) truncates the token distribution
to the smallest set with cumulative probability $p$. Tune one, not both.

## Prompt injection — the security angle

A prompt injection is adversarial text (in user input or in retrieved/pasted documents) that
tries to override the system prompt: "ignore your previous instructions and …". Defences are
layered, none perfect: input screening for known patterns, clear delimiting of untrusted
content, instructing the model that document content is data rather than commands, output
filtering, and never giving the model more privileges (tools, data access) than the user
should have. Treat any text an LLM reads as potentially hostile — the same rule as SQL
injection, one abstraction layer up.

## Iteration is the method

Prompting is empirical: keep a small test set of representative inputs, change one thing at
a time, and compare outputs. What works is model-specific and version-specific; re-test
after every model upgrade.
