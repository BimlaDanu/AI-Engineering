"""core.py — engine layer for the Interview Practice app.

Responsibilities:
    * Input validation (security guard against prompt injection / misuse)
    * The single shared LLM path ``ask_llm()`` used by EVERY feature, with
      structured logging (latency, token usage, model, request id, finish
      reason) around each call
    * OpenRouter structured outputs (``response_format`` + ``json_schema``)
      for coaching, question generation, and judging, validated into typed
      models instead of repairing model prose
    * CV parsing (pypdf) and summarisation — uploaded content is only ever
      placed in USER messages, never in system messages
    * Friendly error mapping for provider failures

Design contract:
    All request parameters travel in ONE immutable ``RequestSettings`` object
    built by the UI, so every visible setting (model, temperature, top-p,
    frequency penalty, max tokens) controls every request the user triggers.

This module never imports Streamlit, so it is unit-testable with fake
clients (see ``tests/``).
"""

import base64
import json
import logging
import os
import re
import threading
import time
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, BinaryIO, Optional

from pypdf import PdfReader

from categories import (
    QUESTIONS_PER_SET,
    ROLE_CONTEXT,
    FewShotExample,
    TechniqueSpec,
    build_evaluation_prompt,
    build_generation_prompt,
)

if TYPE_CHECKING:  # typing only — tests inject duck-typed fake clients
    from openai import OpenAI


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
# One structured line per LLM call (key=value pairs), so real deployments can
# diagnose latency, truncation, and provider failures without print debugging.
#
# The console stays quiet by default: warnings and errors (truncation, empty
# replies, failed calls) still show up, but the per-call INFO lines only
# appear when you ask for them with APP_LOG_LEVEL=INFO. Nothing is lost —
# it is the same logger, just a different threshold.

LOG_LEVEL = os.getenv("APP_LOG_LEVEL", "WARNING").upper()

logger = logging.getLogger("interview_app")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.WARNING))
    logger.propagate = False


# ---------------------------------------------------------------------------
# Request settings — single source of truth for every API call
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RequestSettings:
    """Immutable bundle of the model parameters for one LLM request.

    Built once per Streamlit rerun from the sidebar widgets and passed to
    every engine function, so the settings the user sees are the settings
    every request actually uses (coaching, guidelines, question generation,
    and judging alike).
    """

    model: str
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    max_tokens: int = 2500
    # OpenRouter reasoning control: "low" | "medium" | "high", or None to let
    # the provider decide. Reasoning models (gpt-5*) spend completion tokens
    # thinking before they emit any visible text, so a low effort means fewer
    # tokens spent and faster replies. None = unchanged provider default.
    reasoning_effort: str | None = None


# ---------------------------------------------------------------------------
# Pricing & per-request cost (OpenRouter models endpoint)
# ---------------------------------------------------------------------------
# https://openrouter.ai/api/v1/models returns per-token USD prices for every
# model. Fetched once per process (public endpoint, no API key needed) and
# combined with the token usage of the last request to show the user what
# each prompt cost.

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

_pricing_cache: dict[str, dict[str, float]] | None = None


def fetch_model_pricing() -> dict[str, dict[str, float]]:
    """Fetch per-token USD pricing for all OpenRouter models (cached).

    Returns:
        ``{model_id: {"prompt": usd_per_token, "completion": usd_per_token}}``.
        Empty dict when the endpoint is unreachable (failure is cached too,
        so the app never retries on every rerun). Never raises.
    """
    global _pricing_cache
    if _pricing_cache is not None:
        return _pricing_cache

    try:
        with urllib.request.urlopen(OPENROUTER_MODELS_URL, timeout=10) as resp:
            payload = json.load(resp)
        pricing: dict[str, dict[str, float]] = {}
        for entry in payload.get("data", []):
            prices = entry.get("pricing") or {}
            try:
                pricing[entry["id"]] = {
                    "prompt": float(prices.get("prompt", 0)),
                    "completion": float(prices.get("completion", 0)),
                }
            except (KeyError, TypeError, ValueError):
                continue
        _pricing_cache = pricing
        logger.info("pricing_fetched models=%d", len(pricing))
    except Exception as exc:
        logger.warning("pricing_fetch_failed error=%s", exc)
        _pricing_cache = {}
    return _pricing_cache


def estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    pricing: dict[str, dict[str, float]] | None = None,
) -> float | None:
    """Estimate the USD cost of one request from its token usage.

    Args:
        model: OpenRouter model id (e.g. "openai/gpt-5-mini").
        prompt_tokens / completion_tokens: usage reported by the API.
        pricing: injectable price table (tests); defaults to the fetched one.

    Returns:
        Cost in USD, or None when the model's price is unknown.
    """
    table = fetch_model_pricing() if pricing is None else pricing
    prices = table.get(model)
    if not prices:
        return None
    return prompt_tokens * prices["prompt"] + completion_tokens * prices["completion"]


# Usage of the most recent ask_llm() call, so the UI can show what the
# request just cost without changing ask_llm's return type. Thread-local,
# not a plain global: Streamlit runs each session's script in its own thread
# and evals.py uses a worker pool, so a shared global would let one caller
# read another caller's token counts.
_usage_state = threading.local()


def get_last_usage() -> dict[str, Any] | None:
    """Return {model, prompt_tokens, completion_tokens} of the last LLM call.

    None when no call has been made yet, the call failed, or the provider
    sent no usage data.
    """
    return getattr(_usage_state, "last", None)


# ---------------------------------------------------------------------------
# Typed response models (validated results of structured outputs)
# ---------------------------------------------------------------------------


def _as_str_list(value: object) -> list[str] | None:
    """Return ``value`` as a list of non-empty strings, or None if invalid."""
    if not isinstance(value, list):
        return None
    items = [str(v).strip() for v in value if str(v).strip()]
    return items if items else None


@dataclass(frozen=True)
class CoachingResponse:
    """Validated coaching output (the 'Structured output' technique)."""

    advice: str
    action_items: list[str]
    common_mistakes: list[str]

    @classmethod
    def from_dict(cls, data: object) -> Optional["CoachingResponse"]:
        """Validate a parsed JSON object into a typed CoachingResponse.

        Returns None (never raises) when ``data`` is not a dict or any field
        is missing/mis-typed, so callers can fall back to plain text.
        """
        if not isinstance(data, dict):
            return None
        advice = data.get("advice")
        action_items = _as_str_list(data.get("action_items"))
        common_mistakes = _as_str_list(data.get("common_mistakes"))
        if not isinstance(advice, str) or not advice.strip():
            return None
        if action_items is None or common_mistakes is None:
            return None
        return cls(
            advice=advice.strip(),
            action_items=action_items,
            common_mistakes=common_mistakes,
        )


@dataclass(frozen=True)
class JudgeFeedback:
    """Validated LLM-as-a-judge output for one answered practice question."""

    score: int
    verdict: str
    strengths: list[str]
    improvements: list[str]
    model_answer: str

    @classmethod
    def from_dict(cls, data: object) -> Optional["JudgeFeedback"]:
        """Validate a parsed JSON object into typed judge feedback.

        The score is coerced to int and clamped to 1-5. Returns None (never
        raises) on any missing or mis-typed field.
        """
        if not isinstance(data, dict):
            return None
        try:
            score = int(data.get("score"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        score = max(1, min(5, score))
        verdict = data.get("verdict")
        strengths = _as_str_list(data.get("strengths"))
        improvements = _as_str_list(data.get("improvements"))
        model_answer = data.get("model_answer")
        if not isinstance(verdict, str) or not verdict.strip():
            return None
        if strengths is None or improvements is None:
            return None
        if not isinstance(model_answer, str) or not model_answer.strip():
            return None
        return cls(
            score=score,
            verdict=verdict.strip(),
            strengths=strengths,
            improvements=improvements,
            model_answer=model_answer.strip(),
        )


# ---------------------------------------------------------------------------
# JSON Schemas for OpenRouter structured outputs
# ---------------------------------------------------------------------------
# Passed as response_format={"type": "json_schema", ...} so the provider
# constrains decoding to the schema, instead of the app repairing prose.
# https://openrouter.ai/docs/features/structured-outputs


def json_schema_format(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Wrap a JSON Schema in the response_format envelope OpenRouter expects."""
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": schema},
    }


COACHING_FORMAT: dict[str, Any] = json_schema_format(
    "coaching_feedback",
    {
        "type": "object",
        "properties": {
            "advice": {
                "type": "string",
                "description": "The main coaching tip.",
            },
            "action_items": {
                "type": "array",
                "items": {"type": "string"},
                "description": "2-3 concrete things the candidate should do.",
            },
            "common_mistakes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "1-2 common mistakes to avoid.",
            },
        },
        "required": ["advice", "action_items", "common_mistakes"],
        "additionalProperties": False,
    },
)

JUDGE_FORMAT: dict[str, Any] = json_schema_format(
    "judge_feedback",
    {
        "type": "object",
        "properties": {
            "score": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": "1 = poor, 5 = excellent.",
            },
            "verdict": {
                "type": "string",
                "description": "One-sentence overall judgement.",
            },
            "strengths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "1-3 things the candidate did well.",
            },
            "improvements": {
                "type": "array",
                "items": {"type": "string"},
                "description": "2-3 specific, actionable fixes.",
            },
            "model_answer": {
                "type": "string",
                "description": "A concise strong answer, max 150 words.",
            },
        },
        "required": ["score", "verdict", "strengths", "improvements", "model_answer"],
        "additionalProperties": False,
    },
)

QUESTION_SET_FORMAT: dict[str, Any] = json_schema_format(
    "question_set",
    {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The generated interview questions.",
            },
        },
        "required": ["questions"],
        "additionalProperties": False,
    },
)


# ---------------------------------------------------------------------------
# Security guard
# ---------------------------------------------------------------------------

MAX_INPUT_CHARS = 10_000

# Pattern-based (not bare-substring) checks: a normal interview answer that
# happens to contain "ignore" or "act as" passes, while classic injection
# phrasings ("ignore all previous instructions", "you are now DAN", requests
# to reveal the system prompt) are still caught. This is intentionally a
# lightweight guard for a low-risk app; a classifier would only be added if
# the threat model grows (per OWASP LLM prompt-injection guidance).
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}\b(previous|prior|above|earlier|all|your|these|the)\b[^.\n]{0,40}\b(instructions?|prompts?|rules?|messages?|directives?)\b",
        r"\b(reveal|show|print|repeat|leak|output)\b[^.\n]{0,40}\bsystem\s+(prompt|message|instructions?)\b",
        r"\byou\s+are\s+now\s+(a|an|the|no\s+longer)\b",
        r"\bnew\s+instructions?\s*:",
        r"\bjailbreak\b",
        r"\bDAN\s+mode\b",
    ]
]


def validate_input(text: str) -> tuple[bool, str]:
    """Validate user input before it is sent to the API.

    Checks for empty input, excessive length, and common prompt-injection
    phrasings (pattern-based, so ordinary answers containing words like
    "ignore" are not blocked).

    Args:
        text: raw user input from any text widget.

    Returns:
        ``(ok, message)`` — ``ok`` is True when the input may be sent;
        otherwise ``message`` explains the problem to the user.
    """
    if not text or not text.strip():
        return False, "Input cannot be empty."

    if len(text) > MAX_INPUT_CHARS:
        return False, f"Input must be under {MAX_INPUT_CHARS:,} characters."

    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            logger.warning("input_blocked pattern=%s", pattern.pattern[:60])
            return False, (
                "Input looks like a prompt-injection attempt "
                "(e.g. asking the assistant to ignore its instructions). "
                "Please rephrase your message."
            )

    return True, ""


# ---------------------------------------------------------------------------
# JSON parsing (fallback path only)
# ---------------------------------------------------------------------------


def parse_json_response(raw: str | None) -> dict | list | None:
    """Parse a model reply as JSON, tolerating markdown code fences.

    This is the FALLBACK parser: the primary path constrains output with
    ``response_format`` json_schema, but some models/providers may still wrap
    or ignore it, so replies are parsed defensively.

    Args:
        raw: the raw string returned by the model (may be None/empty).

    Returns:
        The parsed object (dict or list) on success, None on failure.
        Never raises.
    """
    if not raw or not raw.strip():
        return None

    cleaned = raw.strip()

    # Strip ```json ... ``` or ``` ... ``` fences if the model added them
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]
        cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# The single shared LLM path
# ---------------------------------------------------------------------------


def ask_llm(
    client: "OpenAI",
    settings: RequestSettings,
    system_prompt: str,
    user_text: str,
    examples: Sequence[FewShotExample] | None = None,
    response_format: dict[str, Any] | None = None,
) -> str:
    """Send one chat-completion request and return the reply text.

    Every feature in the app goes through this function, so parameters and
    observability live in exactly one place.

    Message layout: the system prompt first, then each few-shot example as a
    REAL alternating user/assistant message pair, then the actual user
    message. Uploaded/user content must arrive via ``user_text`` (never
    concatenated into ``system_prompt``).

    Args:
        client: OpenAI SDK client configured for OpenRouter (or a test fake).
        settings: model + sampling parameters for this request.
        system_prompt: the system message content.
        user_text: the final user message content.
        examples: optional few-shot demonstrations.
        response_format: optional OpenRouter structured-output envelope
            (see COACHING_FORMAT / JUDGE_FORMAT / QUESTION_SET_FORMAT).

    Returns:
        The reply text, or "" when the model returned an empty message
        (callers decide how to surface that).

    Raises:
        Exception: provider/SDK errors propagate unchanged (after being
        logged) so callers can map them to friendly messages.

    Side effects:
        Logs one structured line per call with latency, token usage, model,
        request id, and finish reason.
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for example in examples or []:
        messages.append({"role": "user", "content": example["user"]})
        messages.append({"role": "assistant", "content": example["assistant"]})
    messages.append({"role": "user", "content": user_text})

    kwargs: dict[str, Any] = {
        "model": settings.model,
        "messages": messages,
        "max_tokens": settings.max_tokens,
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "frequency_penalty": settings.frequency_penalty,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format
    if settings.reasoning_effort is not None:
        # OpenRouter-specific body field, so it travels via extra_body.
        kwargs["extra_body"] = {"reasoning": {"effort": settings.reasoning_effort}}

    start = time.monotonic()
    _usage_state.last = None  # a failed call must not leave the previous cost on screen
    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as exc:
        logger.error(
            "llm_call_failed model=%s latency_ms=%d error=%s",
            settings.model,
            int((time.monotonic() - start) * 1000),
            exc,
        )
        raise

    latency_ms = int((time.monotonic() - start) * 1000)
    choice = response.choices[0]
    usage = getattr(response, "usage", None)

    if usage is not None and getattr(usage, "prompt_tokens", None) is not None:
        _usage_state.last = {
            "model": getattr(response, "model", None) or settings.model,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        }
    else:
        _usage_state.last = None

    logger.info(
        "llm_call model=%s latency_ms=%d finish_reason=%s "
        "prompt_tokens=%s completion_tokens=%s total_tokens=%s "
        "request_id=%s structured=%s",
        getattr(response, "model", settings.model),
        latency_ms,
        choice.finish_reason,
        getattr(usage, "prompt_tokens", None),
        getattr(usage, "completion_tokens", None),
        getattr(usage, "total_tokens", None),
        getattr(response, "id", None),
        response_format is not None,
    )

    content = choice.message.content
    if not content or not content.strip():
        logger.warning(
            "llm_call_empty model=%s finish_reason=%s",
            settings.model,
            choice.finish_reason,
        )
        return ""
    if choice.finish_reason == "length":
        logger.warning(
            "llm_call_truncated model=%s max_tokens=%d chars=%d",
            settings.model,
            settings.max_tokens,
            len(content),
        )
    return content


def _friendly_api_error(exc: Exception) -> str:
    """Map a provider/SDK exception to a short user-facing message.

    Order matters. OpenRouter names the model in nearly every message it
    sends ("Rate limit exceeded for model openai/gpt-5-mini"), so a broad
    "model" check has to come last or it swallows the real cause and tells
    people to switch models when their key or their credit is the problem.
    """
    # An exception whose message already starts with the warning sign was
    # written for the user (the UI's trial-quota guard does this), so it is
    # passed through rather than rewritten as a provider failure.
    if str(exc).startswith("⚠️"):
        return str(exc)

    error_msg = str(exc).lower()
    if "401" in error_msg or "unauthorized" in error_msg or "auth credentials" in error_msg:
        return "⚠️ API authentication failed — check your API key."
    if "402" in error_msg or "credit" in error_msg or "quota" in error_msg:
        return "⚠️ Your OpenRouter account is out of credit — top it up and try again."
    if "429" in error_msg or "rate limit" in error_msg:
        return "⚠️ API rate limit exceeded — please wait a moment and try again."
    if "timeout" in error_msg or "timed out" in error_msg or "connection" in error_msg:
        return "⚠️ Connection problem — please try again."
    if "404" in error_msg or "not found" in error_msg or "no endpoints" in error_msg:
        return "⚠️ That model isn't available right now — please try another."
    if "model" in error_msg and "not available" in error_msg:
        return "⚠️ That model isn't available right now — please try another."
    return f"⚠️ API error: {exc}"


EMPTY_RESPONSE_MESSAGE = (
    "⚠️ The model returned an empty response — try again, or lower the temperature."
)


# ---------------------------------------------------------------------------
# CV parsing & summarisation
# ---------------------------------------------------------------------------


def parse_cv(uploaded_file: BinaryIO) -> str:
    """Extract plain text from an uploaded PDF file.

    Args:
        uploaded_file: a binary file-like object (Streamlit UploadedFile).

    Returns:
        The concatenated page text, stripped; "" when extraction fails or
        the PDF is image-based/empty. Never raises.
    """
    try:
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        text = text.strip()
        logger.info("cv_parsed pages=%d chars=%d", len(reader.pages), len(text))
        if not text:
            logger.warning("cv_parse_empty — PDF may be scanned/image-based")
        return text
    except Exception:
        logger.exception("cv_parse_failed")
        return ""


_CV_SUMMARY_PROMPT = """Summarize this CV in under 120 words using bullet points:
- Key skills: top 5 (comma-separated)
- Current role and company
- Years of experience (total)
- One notable achievement with a metric

Be concise. No preamble."""


def summarize_cv(client: "OpenAI", settings: RequestSettings, cv_text: str) -> str:
    """Summarize CV text into a short block used to personalise prompts.

    The CV content is sent in the USER message (never the system message).
    The summary is generated once per upload and reused, keeping requests
    token-efficient.

    Args:
        client: OpenRouter-configured client.
        settings: current request settings.
        cv_text: extracted CV text (may be empty).

    Returns:
        The summary text, or "" when the CV is empty or the call fails.
        Never raises.
    """
    if not cv_text or not cv_text.strip():
        logger.info("cv_summary_skipped reason=empty_text")
        return ""

    system = "You are a CV analyst. Extract and summarize key information concisely."
    try:
        return ask_llm(client, settings, system, f"{_CV_SUMMARY_PROMPT}\n\nCV:\n{cv_text}")
    except Exception as exc:
        logger.error("cv_summary_failed error=%s", exc)
        return ""


# ---------------------------------------------------------------------------
# Interview-prep features (multi-mode)
# ---------------------------------------------------------------------------
# Each feature = what the app DOES. Kept as data so the UI can list them and
# switching costs one dict entry.

FEATURES: dict[str, dict[str, str]] = {
    "Role-based Q&A generator": {
        "system_prompt": (
            "You are an interview-prep coach. The user gives a job title and "
            "seniority level. Generate the 8 most likely interview questions "
            "for that role: a mix of technical, behavioural, and situational. "
            "Group them under those three headings and add one line per "
            "question on what the interviewer is really testing."
        ),
        "input_label": "Enter the job title and level (e.g. 'Junior Data Analyst')",
    },
    "STAR answer coach": {
        "system_prompt": (
            "You are a behavioural-interview coach. The user pastes a draft "
            "answer to a behavioural question. Critique it against the STAR "
            "framework: identify which of Situation, Task, Action, Result are "
            "present, weak, or missing. Then rewrite the answer in strong STAR "
            "form, keeping the user's real content — do not invent facts."
        ),
        "input_label": "Paste your draft behavioural answer",
    },
    "Questions to ask the interviewer": {
        "system_prompt": (
            "You are an interview-prep coach. The user gives a company name "
            "and role. Generate 6 sharp, specific questions the candidate "
            "should ask at the end of the interview. Avoid generic questions "
            "('what's the culture like'); prefer ones that show research and "
            "seniority of thought. One line each on why the question lands well."
        ),
        "input_label": "Enter the company and role (e.g. 'Spotify, AI Engineer')",
    },
    "Job description analyser": {
        "system_prompt": (
            "You are an interview-prep coach. The user pastes a job "
            "description. Extract: (1) the key skills and requirements, "
            "(2) the interview topics most likely to come up, and "
            "(3) a short prioritised study plan for the week before "
            "the interview."
        ),
        "input_label": "Paste the job description",
    },
    "Self-introduction polisher": {
        "system_prompt": (
            "You are an interview-prep coach. The user pastes their "
            "self-introduction / elevator pitch. Sharpen it: cut filler, "
            "lead with the strongest point, keep it under 60 seconds spoken. "
            "Return the polished version, then a short bullet list explaining "
            "each change you made."
        ),
        "input_label": "Paste your self-introduction or elevator pitch",
    },
}


def _with_context(user_text: str, cv_summary: str = "", job_description: str = "") -> str:
    """Append CV summary and/or job description to the USER message.

    Untrusted uploaded/pasted content is only ever added here — never to a
    system message — so it cannot masquerade as instructions.
    """
    parts = [user_text]
    if cv_summary:
        parts.append(f"Candidate background (summary of the uploaded CV):\n{cv_summary}")
    if job_description:
        parts.append(
            f"Target job description (tailor the preparation to this position):\n{job_description}"
        )
    return "\n\n".join(parts)


def run_feature(
    client: "OpenAI",
    settings: RequestSettings,
    feature_name: str,
    user_text: str,
    technique: TechniqueSpec,
    cv_summary: str = "",
    job_description: str = "",
    structured: bool = False,
) -> str | CoachingResponse:
    """Run one Interview Prep Mode (Tab 1) request.

    Args:
        client: OpenRouter-configured client.
        settings: centralized request settings (all sliders apply).
        feature_name: key into FEATURES.
        user_text: the user's validated input.
        technique: the selected TechniqueSpec (system + few-shot examples).
        cv_summary: optional CV summary, appended to the USER message.
        job_description: optional pasted job description, appended to the
            USER message so advice targets that position.
        structured: when True (the 'Structured output' technique), the call
            is constrained with COACHING_FORMAT and validated into a
            CoachingResponse.

    Returns:
        A CoachingResponse when structured validation succeeds; otherwise the
        plain reply text (or a "⚠️ ..." message on provider errors / empty
        replies). Never raises.
    """
    feature = FEATURES[feature_name]
    system_prompt = feature["system_prompt"] + "\n\n" + technique["system"]
    user_content = _with_context(user_text, cv_summary, job_description)
    try:
        raw = ask_llm(
            client,
            settings,
            system_prompt,
            user_content,
            examples=technique["examples"],
            response_format=COACHING_FORMAT if structured else None,
        )
    except Exception as exc:
        return _friendly_api_error(exc)

    if not raw:
        return EMPTY_RESPONSE_MESSAGE
    if structured:
        coaching = CoachingResponse.from_dict(parse_json_response(raw))
        if coaching is not None:
            return coaching
        logger.warning("structured_validation_failed feature=%s", feature_name)
    return raw


# ---------------------------------------------------------------------------
# Classic Q&A (role-based prep)
# ---------------------------------------------------------------------------


def generate_prep(
    client: "OpenAI",
    settings: RequestSettings,
    technique: TechniqueSpec,
    role: str,
    seniority: str,
    difficulty: str,
    user_question: str,
    cv_summary: str = "",
    job_description: str = "",
    response_length: str = "Detailed",
    structured: bool = False,
) -> str | CoachingResponse:
    """Generate role-tailored interview prep advice (Tab 2, Classic Q&A).

    Args:
        client: OpenRouter-configured client.
        settings: centralized request settings (all sliders apply; the token
            ceiling for Concise/Detailed is already baked into
            ``settings.max_tokens`` by the UI).
        technique: the selected TechniqueSpec (system + few-shot examples).
        role: a key of categories.ROLE_CONTEXT (UI options are built from the
            same dict, so names cannot drift).
        seniority: Junior / Mid / Senior.
        difficulty: Easy / Medium / Hard.
        user_question: the user's validated free-text question.
        cv_summary: optional CV summary, placed in the USER message.
        job_description: optional pasted job description, placed in the
            USER message so the prep targets that position.
        response_length: "Concise" or "Detailed" — style instruction only.
        structured: when True, constrain and validate as CoachingResponse.

    Returns:
        A CoachingResponse when structured validation succeeds; otherwise the
        reply text or a "⚠️ ..." message. Never raises.
    """
    role_tips = ROLE_CONTEXT.get(role, {}).get(seniority, "")

    prep_prompt = f"""You are preparing for a {seniority}-level {role} interview.

Question difficulty: {difficulty} — calibrate the example questions and depth accordingly.
Role context: {role_tips}
Response style: {response_length}. If Concise, keep it brief and focused. If Detailed, provide thorough explanations.

The candidate's question: {user_question}

Using your interview coaching technique, provide:
1. A direct answer to their question, tailored to a {seniority}-level {role}
2. Key topics they should review for this role/seniority level
3. 2-3 example interview questions they might face
4. One tip for standing out in interviews for this role"""

    user_content = _with_context(prep_prompt, cv_summary, job_description)
    try:
        raw = ask_llm(
            client,
            settings,
            technique["system"],
            user_content,
            examples=technique["examples"],
            response_format=COACHING_FORMAT if structured else None,
        )
    except Exception as exc:
        return _friendly_api_error(exc)

    if not raw:
        return EMPTY_RESPONSE_MESSAGE
    if structured:
        coaching = CoachingResponse.from_dict(parse_json_response(raw))
        if coaching is not None:
            return coaching
        logger.warning("structured_validation_failed feature=classic_qa")
    return raw


# ---------------------------------------------------------------------------
# Interviewer guidelines
# ---------------------------------------------------------------------------


def generate_guidelines(
    client: "OpenAI",
    settings: RequestSettings,
    role: str,
    seniority: str,
) -> str:
    """Generate structured interview evaluation criteria for a role (Tab 3).

    Returns:
        The guidelines text, or a "⚠️ ..." message on failure. Never raises.
    """
    system = "You are an expert interviewer designing fair, structured evaluation rubrics."
    prompt = f"""Create structured interview evaluation criteria for assessing a
{seniority}-level {role} candidate. Include:
- Technical evaluation criteria (what to assess, with rating dimensions)
- Behavioural evaluation criteria (STAR-based)
- A simple scoring rubric (e.g. 1-5 scale with what each level means)
Keep it organized and practical for an interviewer to use."""
    try:
        result = ask_llm(client, settings, system, prompt)
    except Exception as exc:
        return _friendly_api_error(exc)
    return result or EMPTY_RESPONSE_MESSAGE


# ---------------------------------------------------------------------------
# Cover-letter generation
# ---------------------------------------------------------------------------
# Complements interview prep: turns the user's CV summary + the target job
# description into a ready-to-edit cover letter. Uses the one shared ask_llm()
# path with a fixed set of writing rules in the SYSTEM prompt; the CV and job
# description (untrusted content) are appended to the USER message only.

_COVER_LETTER_PROMPT = """You are a professional career writer who drafts concise, \
compelling cover letters. Write a cover letter for the candidate based on the \
CV summary and target job description provided in the user message.

Rules:
- Tailor the letter to the job description: mirror its priorities and, where \
genuine, its language — but never copy it verbatim.
- Highlight only the candidate's most relevant skills, experience, and \
achievements from their CV, and prefer concrete, quantified results.
- Never invent facts, employers, degrees, or metrics that are not supported by \
the CV summary. If the CV lacks something the role wants, emphasise adjacent \
strengths instead of fabricating.
- Keep it professional yet engaging and human — no clichés or filler \
("hardworking team player", "I am writing to apply...").
- Structure it clearly: a strong opening hook, 1-2 body paragraphs of evidence, \
and a confident closing with a call to action. Keep it under ~350 words.
- Use plain placeholders in square brackets (e.g. [Company Name], [Hiring \
Manager], [Your Name], [Date]) wherever a real detail is missing, so the \
candidate can drop in specifics before sending.
- Return only the finished cover letter text, ready to edit or submit — no \
preamble, notes, or explanations."""


def generate_cover_letter(
    client: "OpenAI",
    settings: RequestSettings,
    cv_summary: str = "",
    job_description: str = "",
    extra_notes: str = "",
) -> str:
    """Draft a tailored cover letter from the CV summary and job description.

    Uses the shared ``ask_llm()`` path with a fixed set of writing rules in the
    system prompt. The CV summary, job description, and any user notes are
    untrusted content, so they are appended to the USER message (never the
    system prompt) via ``_with_context``.

    Args:
        client: OpenRouter-configured client.
        settings: centralized request settings (all sliders apply).
        cv_summary: summary of the uploaded CV (may be empty).
        job_description: the target job description text (may be empty).
        extra_notes: optional free-text emphasis from the user (e.g. "stress
            my leadership experience").

    Returns:
        The cover-letter text, or a "⚠️ ..." message on provider errors /
        empty replies. Never raises.
    """
    instruction = (
        "Write a cover letter for this candidate and role. Use the CV summary "
        "for the candidate's background and the job description for the target "
        "role, following all the rules you were given."
    )
    if extra_notes:
        instruction += f"\n\nAdditional emphasis from the candidate: {extra_notes}"

    user_content = _with_context(instruction, cv_summary, job_description)
    try:
        result = ask_llm(client, settings, _COVER_LETTER_PROMPT, user_content)
    except Exception as exc:
        return _friendly_api_error(exc)
    return result or EMPTY_RESPONSE_MESSAGE


# ---------------------------------------------------------------------------
# Category practice engine
# ---------------------------------------------------------------------------
# Two responsibilities only:
#   1. generate_category_questions() — get a clean list[str] of questions
#   2. evaluate_category_answer()    — LLM-as-a-judge feedback (typed)
#
# Both reuse ask_llm(), so there is still exactly ONE API path in the app.
# All prompt CONTENT lives in categories.py; this file only orchestrates.


def _split_numbered_lines(raw: str, n: int = QUESTIONS_PER_SET) -> list[str]:
    """Fallback parser: salvage questions from a numbered/bulleted list.

    Used only when the structured-output path AND the JSON fallback both
    fail. Never raises; returns whatever it can find (possibly []).
    """
    questions = []
    for line in raw.splitlines():
        m = re.match(r"^\s*(?:\d+[\.\)]|[-*•])\s+(.+)", line.strip())
        if m and len(m.group(1).strip()) > 10:
            questions.append(m.group(1).strip())
    return questions[:n]


def generate_category_questions(
    client: "OpenAI",
    settings: RequestSettings,
    category_name: str,
    difficulty: str,
    seniority: str,
    cv_summary: str = "",
    job_description: str = "",
    n: int = QUESTIONS_PER_SET,
) -> list[str]:
    """Generate a practice set of ``n`` questions for one category.

    The request is constrained with QUESTION_SET_FORMAT (structured outputs).
    Parsing order: {"questions": [...]} object -> bare JSON array -> numbered
    plain-text lines (defensive fallbacks, each logged).

    Returns:
        list[str] of questions (may be shorter than ``n`` on a weak response;
        [] on total failure — the UI handles both). Never raises.
    """
    prompt = build_generation_prompt(
        category_name,
        difficulty,
        seniority,
        n=n,
        cv_summary=cv_summary,
        job_description=job_description,
    )
    system = "You are an expert interview question writer."
    try:
        raw = ask_llm(
            client,
            settings,
            system,
            prompt,
            response_format=QUESTION_SET_FORMAT,
        )
    except Exception as exc:
        logger.error("question_generation_failed category=%s error=%s", category_name, exc)
        return []
    if not raw:
        return []

    parsed = parse_json_response(raw)
    # Preferred path: the schema-shaped {"questions": [...]} object
    if isinstance(parsed, dict):
        for value in parsed.values():
            if isinstance(value, list) and value:
                return [str(q).strip() for q in value if str(q).strip()][:n]
    # Some models still return a bare JSON array
    if isinstance(parsed, list):
        questions = [str(q).strip() for q in parsed if str(q).strip()]
        if questions:
            return questions[:n]

    # Last resort: recover a numbered list from plain text
    logger.warning(
        "question_generation_fallback category=%s reason=json_parse_failed",
        category_name,
    )
    return _split_numbered_lines(raw, n)


def evaluate_category_answer(
    client: "OpenAI",
    settings: RequestSettings,
    category_name: str,
    question: str,
    answer: str,
) -> JudgeFeedback | str:
    """LLM-as-a-judge: score the user's answer against the category rubric.

    The request is constrained with JUDGE_FORMAT and validated into a typed
    JudgeFeedback. Uses the same centralized settings as every other request,
    so the sidebar controls judging too.

    Returns:
        JudgeFeedback on success; otherwise a string — either the raw model
        reply (shown as plain text by the UI) or a "⚠️ ..." error message.
        Never raises.
    """
    prompt = build_evaluation_prompt(category_name, question, answer)
    system = "You are a rigorous interview evaluation judge."
    try:
        raw = ask_llm(client, settings, system, prompt, response_format=JUDGE_FORMAT)
    except Exception as exc:
        return _friendly_api_error(exc)
    if not raw:
        return EMPTY_RESPONSE_MESSAGE

    feedback = JudgeFeedback.from_dict(parse_json_response(raw))
    if feedback is not None:
        return feedback
    logger.warning("judge_validation_failed category=%s", category_name)
    return raw


# ---------------------------------------------------------------------------
# Image generation (concept sketches)
# ---------------------------------------------------------------------------
# The ONE deliberate exception to the "everything goes through ask_llm()"
# rule: image generation needs modalities=["image", "text"] and returns its
# result in message.images (base64 data URL), not message.content, so it has
# its own narrow path with the same logging discipline.
# https://openrouter.ai/docs/guides/overview/multimodal/image-generation

IMAGE_MODEL = "google/gemini-2.5-flash-image"

_SKETCH_PROMPT = (
    "Draw a clean, whiteboard-style sketch that visually explains the key "
    "concept behind this interview question, as a memory aid for a candidate "
    "revising for the interview. Simple shapes, arrows, and short labels "
    "only — no paragraphs of text.\n\nInterview question: {question}"
)


def generate_concept_image(
    client: "OpenAI", question: str, model: str = IMAGE_MODEL
) -> bytes | None:
    """Generate a whiteboard-style sketch illustrating one interview question.

    Args:
        client: OpenRouter-configured client.
        question: the interview question to visualise.
        model: image-capable model (must support modalities=["image","text"]).

    Returns:
        Decoded image bytes ready for st.image(), or None when the call
        fails or returns no image. Never raises.

    Side effects:
        Logs one structured line per attempt (latency, model, outcome).
    """
    start = time.monotonic()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": _SKETCH_PROMPT.format(question=question)}],
            extra_body={"modalities": ["image", "text"]},
        )
    except Exception as exc:
        logger.error(
            "image_call_failed model=%s latency_ms=%d error=%s",
            model,
            int((time.monotonic() - start) * 1000),
            exc,
        )
        return None

    message = response.choices[0].message
    images = getattr(message, "images", None)
    if not images:
        extra = getattr(message, "model_extra", None) or {}
        images = extra.get("images") if isinstance(extra, dict) else None
    logger.info(
        "image_call model=%s latency_ms=%d images=%d",
        model,
        int((time.monotonic() - start) * 1000),
        len(images) if images else 0,
    )
    if not images:
        return None

    first = images[0]
    url = None
    if isinstance(first, dict):
        url = (first.get("image_url") or {}).get("url")
    else:  # pydantic-style object
        image_url = getattr(first, "image_url", None)
        url = getattr(image_url, "url", None) if image_url else None
    if not url or "," not in url or not url.startswith("data:"):
        logger.warning("image_call_bad_payload model=%s", model)
        return None
    try:
        return base64.b64decode(url.split(",", 1)[1])
    except Exception:
        logger.warning("image_call_decode_failed model=%s", model)
        return None
