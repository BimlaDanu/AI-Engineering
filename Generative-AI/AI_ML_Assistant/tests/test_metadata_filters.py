"""Offline tests for B3 knowledge-base metadata filters (topic / source / year).

These pin the pure filter logic — :class:`MetadataFilters` (Chroma predicates + BM25-side
``passes``) and :func:`compute_facets` — plus the wiring in the ``retrieve`` step that turns
the raw :class:`RagSettings` lists into a filter and hands it to ``kb.search``. No network,
LLM, or vector store is touched: the KB is a recording fake.
"""

from __future__ import annotations

from src.config import RagSettings
from src.core.service import AnswerRequest
from src.core.steps import PipelineDeps, retrieve
from src.rag.retriever import MetadataFilters, RetrievedChunk, compute_facets
from src.utils import StageTimer


def _chunk(text: str, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        metadata={"source": "s.md"},
        vector_score=score,
        bm25_score=score,
        score=score,
    )


# --- MetadataFilters.active -------------------------------------------------
def test_empty_filter_is_inactive() -> None:
    assert MetadataFilters().active is False


def test_any_facet_makes_it_active() -> None:
    assert MetadataFilters(topics=("attention",)).active is True
    assert MetadataFilters(sources=("paper.md",)).active is True
    assert MetadataFilters(years=("2023",)).active is True


# --- MetadataFilters.conditions: Chroma $in predicates ----------------------
def test_conditions_empty_when_no_facets() -> None:
    assert MetadataFilters().conditions() == []


def test_conditions_builds_one_in_predicate_per_active_facet() -> None:
    filters = MetadataFilters(topics=("t1", "t2"), sources=("a.md",), years=("2024",))
    assert filters.conditions() == [
        {"topic": {"$in": ["t1", "t2"]}},
        {"source": {"$in": ["a.md"]}},
        {"year": {"$in": ["2024"]}},
    ]


# --- MetadataFilters.passes: BM25-side mirror of the Chroma filter ----------
def test_passes_true_when_no_facets_selected() -> None:
    assert MetadataFilters().passes({"topic": "anything"}) is True


def test_passes_checks_topic_source_and_year() -> None:
    filters = MetadataFilters(topics=("attention",), sources=("a.md",), years=("2023",))
    ok = {"topic": "attention", "source": "a.md", "year": "2023"}
    assert filters.passes(ok) is True
    assert filters.passes({**ok, "topic": "other"}) is False
    assert filters.passes({**ok, "source": "b.md"}) is False
    assert filters.passes({**ok, "year": "2020"}) is False


def test_passes_year_compared_as_string() -> None:
    # Years are stored as strings; an int year still matches its string form.
    assert MetadataFilters(years=("2023",)).passes({"year": 2023}) is True


def test_passes_missing_year_never_matches_an_active_year_filter() -> None:
    assert MetadataFilters(years=("2023",)).passes({"topic": "x"}) is False


# --- compute_facets: distinct, sorted values for the UI ---------------------
def test_compute_facets_dedupes_and_sorts() -> None:
    metas = [
        {"topic": "rag", "source": "b.md", "year": "2021"},
        {"topic": "attention", "source": "a.md", "year": "2023"},
        {"topic": "rag", "source": "a.md", "year": "2021"},
    ]
    facets = compute_facets(metas)
    assert facets["topics"] == ["attention", "rag"]
    assert facets["sources"] == ["a.md", "b.md"]
    assert facets["years"] == ["2023", "2021"]  # newest first


def test_compute_facets_ignores_missing_and_empty_values() -> None:
    metas = [{"topic": "rag"}, {"source": "", "year": None}, {}]
    facets = compute_facets(metas)
    assert facets == {"topics": ["rag"], "sources": [], "years": []}


# --- retrieve step: raw settings -> MetadataFilters -> kb.search ------------
class _RecordingKB:
    """Records the ``filters`` it is searched with and returns a couple of chunks."""

    def __init__(self) -> None:
        self.filters_seen: list[MetadataFilters | None] = []

    def search(self, query, subjects=None, difficulty=None, k=4, alpha=0.7, *, filters=None):
        self.filters_seen.append(filters)
        return [_chunk("hit0"), _chunk("hit1")]


def _deps(kb, settings) -> PipelineDeps:
    req = AnswerRequest(
        question="how does attention work",
        level="Practitioner",
        model="m",
        subjects=None,
        settings=settings,
    )
    return PipelineDeps(
        llm=None, kb=kb, settings=settings, router=None, req=req, timer=StageTimer()
    )


def test_retrieve_passes_selected_facets_to_search() -> None:
    kb = _RecordingKB()
    settings = RagSettings(
        rewrite_query=False,
        rerank=False,
        filter_topics=["attention"],
        filter_sources=["a.md"],
        filter_years=["2023"],
    )
    retrieve({"question": "q", "history": []}, _deps(kb, settings))
    passed = kb.filters_seen[0]
    assert passed is not None and passed.active
    assert passed.topics == ("attention",)
    assert passed.sources == ("a.md",)
    assert passed.years == ("2023",)


def test_retrieve_passes_inactive_filter_when_none_selected() -> None:
    kb = _RecordingKB()
    settings = RagSettings(rewrite_query=False, rerank=False)
    retrieve({"question": "q", "history": []}, _deps(kb, settings))
    passed = kb.filters_seen[0]
    assert passed is not None and passed.active is False
