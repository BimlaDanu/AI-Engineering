"""Offline tests for auxiliary-call token accounting (:class:`src.utils.UsageMeter`).

The final answer reports its own exact usage, but the extra main-model calls a request makes
— query rewriting/expansion and listwise reranking — used to go uncounted, undercounting the
reported tokens and cost. These pin that the meter records exact usage when the provider
reports it and an estimate otherwise, and that the ``retrieve`` step feeds those calls into
the meter — all without a network or a real model.
"""

from __future__ import annotations

from typing import ClassVar

from src.config import RagSettings
from src.core.schemas import Reranking
from src.core.service import AnswerRequest
from src.core.steps import PipelineDeps, retrieve
from src.rag.retriever import RetrievedChunk, rerank_chunks, rewrite_query
from src.utils import StageTimer, UsageMeter


class _RerankingLLM:
    """Structured-output rerank stub: returns a preset ranking, carries no usage_metadata."""

    def __init__(self, ranking: list[int]) -> None:
        self._ranking = ranking

    def with_structured_output(self, schema: object, method: str | None = None) -> object:
        ranking = self._ranking

        class _Structured:
            def invoke(self, _messages: object) -> Reranking:
                return Reranking(ranking=ranking)

        return _Structured()


def _chunk(text: str, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        metadata={"source": "s.md"},
        vector_score=score,
        bm25_score=score,
        score=score,
    )


# --- UsageMeter primitive ---------------------------------------------------
def test_add_accumulates_and_clamps_negatives() -> None:
    meter = UsageMeter()
    meter.add(10, 4)
    meter.add(-5, 2)  # negative inputs are clamped to zero, not subtracted
    assert (meter.input_tokens, meter.output_tokens) == (10, 6)


def test_record_call_uses_exact_usage_metadata_when_present() -> None:
    class _Reply:
        content = "rewritten"
        usage_metadata: ClassVar[dict] = {"input_tokens": 42, "output_tokens": 7}

    meter = UsageMeter()
    meter.record_call(_Reply(), "prompt text ignored", "output ignored")
    assert (meter.input_tokens, meter.output_tokens) == (42, 7)


def test_record_call_estimates_when_usage_absent() -> None:
    # A structured-output result carries no usage_metadata, so tokens are estimated from text.
    meter = UsageMeter()
    meter.record_call(object(), "a fairly long prompt " * 20, "short out")
    assert meter.input_tokens > meter.output_tokens > 0


# --- helpers report into the meter ------------------------------------------
class _RewriteReply:
    content = '"self-attention in transformers"'
    usage_metadata: ClassVar[dict] = {"input_tokens": 15, "output_tokens": 5}


class _RewriteLLM:
    def invoke(self, _messages: object) -> _RewriteReply:
        return _RewriteReply()


def test_rewrite_query_records_exact_usage() -> None:
    meter = UsageMeter()
    out = rewrite_query(_RewriteLLM(), "how does attention work", [], meter=meter)
    assert out == "self-attention in transformers"
    assert (meter.input_tokens, meter.output_tokens) == (15, 5)


def test_rerank_records_estimated_usage_from_the_candidate_catalogue() -> None:
    # Reranking is structured (no usage_metadata), so usage is estimated — and because the
    # candidate catalogue dominates the prompt, the estimate must be non-trivial.
    chunks = [_chunk("passage one " * 30), _chunk("passage two " * 30), _chunk("three " * 30)]
    llm = _RerankingLLM([3, 1, 2])
    meter = UsageMeter()
    rerank_chunks(llm, "q", chunks, top_n=2, meter=meter)
    assert meter.input_tokens > 50  # the catalogue was counted, not skipped
    assert meter.output_tokens > 0


# --- retrieve feeds the request meter ---------------------------------------
class _RewriteRecordingKB:
    def search(self, query, subjects=None, difficulty=None, k=4, alpha=0.7, *, filters=None):
        return [_chunk(f"hit::{query}")]


def _deps(kb, settings, llm) -> PipelineDeps:
    req = AnswerRequest(
        question="how does attention work",
        level="Practitioner",
        model="m",
        subjects=None,
        settings=settings,
    )
    return PipelineDeps(llm=llm, kb=kb, settings=settings, router=None, req=req, timer=StageTimer())


def test_retrieve_accumulates_rewrite_usage_into_deps_meter() -> None:
    settings = RagSettings(multi_query=False, rewrite_query=True)
    deps = _deps(_RewriteRecordingKB(), settings, _RewriteLLM())
    assert deps.usage.input_tokens == 0
    retrieve({"question": "how does attention work", "history": []}, deps)
    assert deps.usage.input_tokens == 15 and deps.usage.output_tokens == 5
