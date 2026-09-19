"""
RAGAs-style evaluation harness for the Synapse app RAG pipeline.

This package evaluates **retrieval and answer quality**, rather than just testing
whether the pipeline works. It implements the four main RAGAs metrics:
**faithfulness**, **answer relevancy**, **context precision**, and
**context recall**. These metrics use an LLM-as-judge with OpenRouter
structured outputs (`response_format` + `json_schema`) through LangChain's
`with_structured_output`. Each judge response is validated as a Pydantic
object, so all metric calculations use typed data instead of parsing
free-form text.

The package is organised into independently testable modules:

- `src.eval.schemas` — Defines the Pydantic schemas returned by the judge.
- `src.eval.judge` — Contains the `Judge` protocol and the `LLMJudge`
  implementation. Tests can use a deterministic stub that follows the
  same protocol.
- `src.eval.metrics` — Implements the four RAGAs metrics. Each metric
  takes a judge and returns a typed `MetricResult`.
- `src.eval.dataset` — Loads the curated golden question/reference dataset.
- `src.eval.runner` — Runs the evaluation pipeline and produces an
  aggregated `EvalReport`.

The harness only calls OpenRouter, the same provider used by the app,
and only when an evaluation is run.
"""

from __future__ import annotations

from src.eval.compare import ABComparison, MetricComparison, Variant, compare_reports
from src.eval.dataset import GoldenSample, load_golden_set
from src.eval.judge import Judge, LLMJudge
from src.eval.metrics import (
    METRICS,
    MetricResult,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from src.eval.runner import (
    EvalReport,
    SampleEvaluation,
    run_evaluation,
    run_evaluations,
)

__all__ = [
    "METRICS",
    "ABComparison",
    "EvalReport",
    "GoldenSample",
    "Judge",
    "LLMJudge",
    "MetricComparison",
    "MetricResult",
    "SampleEvaluation",
    "Variant",
    "answer_relevancy",
    "compare_reports",
    "context_precision",
    "context_recall",
    "faithfulness",
    "load_golden_set",
    "run_evaluation",
    "run_evaluations",
]
