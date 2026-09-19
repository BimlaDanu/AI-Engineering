"""Abstractive summarization through the OpenRouter gateway.

Responses are cached on disk, keyed by model, prompt version, language, length
and a hash of the article text, so a re-run returns the same summaries and the
similarity scores stay reproducible.
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from src import config
from src.log import get_logger

if TYPE_CHECKING:
    from openai import OpenAI

logger = get_logger(__name__)

CACHE_DIR = config.INTERIM_DIR / "llm_cache"

_LANGUAGE_NAMES = {"en": "English", "es": "Spanish", "fr": "French", "de": "German"}

# Part of the cache key. Bump it when the prompt changes, otherwise the cache
# serves summaries written to the old wording.
_PROMPT_VERSION = 2

_SYSTEM_PROMPT = (
    "You are a news editor writing summaries for a press monitoring service. "
    "Summarize only what the article states. Do not add background, opinion or "
    "any fact that is not in the text. Keep the named entities -- people, "
    "organisations and places -- that the article treats as important. "
    "Reply with the summary alone, no preamble and no bullet points."
)


@dataclass(frozen=True)
class SummaryRequest:
    """One article queued for abstractive summarization.

    Attributes:
        pageid: Identifier of the article, used to match the response back.
        lang: Two-letter language code; the summary is written in it.
        text: The article text.
    """

    pageid: int
    lang: str
    text: str


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    """Build the OpenRouter client once per process.

    Returns:
        An OpenAI-compatible client pointed at OpenRouter.

    Raises:
        RuntimeError: If ``OPENROUTER_API_KEY`` is not set.
    """
    from openai import OpenAI

    load_dotenv(config.PROJECT_ROOT / ".env")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to .env before running this stage."
        )
    return OpenAI(base_url=config.LLM_BASE_URL, api_key=api_key)


def build_prompt(text: str, lang: str, n_sentences: int) -> str:
    """Write the user prompt for one article.

    Args:
        text: The article text.
        lang: Two-letter language code.
        n_sentences: Target length of the summary.

    Returns:
        The prompt, asking for a summary in the article's own language.
    """
    language = _LANGUAGE_NAMES.get(lang, lang)
    # A sentence count alone is not enough: asked only for three sentences the
    # model returns three long ones and compresses almost nothing. The word
    # target is what makes the summary shorter.
    target_words = n_sentences * 20
    return (
        f"Summarize the news article below in at most {n_sentences} sentences "
        f"and about {target_words} words in total, written in {language}. "
        f"Keep each sentence under 25 words.\n\nARTICLE:\n{text}"
    )


def _cache_path(request: SummaryRequest, n_sentences: int) -> Path:
    """Return the cache file for a request.

    The key covers everything that changes the output, so switching models or
    prompt length produces a new entry rather than a stale hit.

    Args:
        request: The queued article.
        n_sentences: Target summary length.

    Returns:
        Path of the JSON cache file.
    """
    material = f"{config.LLM_MODEL}|v{_PROMPT_VERSION}|{request.lang}|{n_sentences}|{request.text}"
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]
    return CACHE_DIR / f"{digest}.json"


def summarize_one(request: SummaryRequest, n_sentences: int) -> str:
    """Summarize a single article, using the cache when possible.

    A failed call returns an empty string rather than raising, so one bad
    response cannot lose the whole batch.

    Args:
        request: The article to summarize.
        n_sentences: Target summary length.

    Returns:
        The summary, or an empty string if the call failed.
    """
    path = _cache_path(request, n_sentences)
    if path.exists():
        return str(json.loads(path.read_text(encoding="utf-8"))["summary"])

    try:
        response = _client().chat.completions.create(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(request.text, request.lang, n_sentences)},
            ],
        )
        summary = (response.choices[0].message.content or "").strip()
    except Exception as error:  # noqa: BLE001 - one failure must not stop the batch
        logger.warning("summary failed for pageid %s (%s): %s", request.pageid, request.lang, error)
        return ""

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"summary": summary}, ensure_ascii=False), encoding="utf-8")
    return summary


def summarize_many(requests: list[SummaryRequest], n_sentences: int) -> dict[int, str]:
    """Summarize a batch of articles concurrently.

    Args:
        requests: The articles to summarize.
        n_sentences: Target summary length.

    Returns:
        Summary text keyed by ``pageid``. All requests must be for one
        language, since a ``pageid`` is shared across translations.
    """
    logger.info("requesting %d abstractive summaries from %s", len(requests), config.LLM_MODEL)
    with ThreadPoolExecutor(max_workers=config.LLM_MAX_WORKERS) as pool:
        summaries = list(pool.map(lambda request: summarize_one(request, n_sentences), requests))

    failed = sum(1 for summary in summaries if not summary)
    if failed:
        logger.warning("%d of %d summaries came back empty", failed, len(requests))
    return {request.pageid: summary for request, summary in zip(requests, summaries, strict=True)}
