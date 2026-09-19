"""Offline tests for structured arXiv query building and free-text cleaning.

These pin the fix for the playground bug where a raw sentence like
"Andrew Ng ml arxiv papers" returned unrelated papers: a person's name belongs in an
``au:`` field, and filler words ("arxiv", "papers") must be stripped rather than matched.
All pure functions — no network.
"""

from __future__ import annotations

from src.tools.arxiv import _clean_freetext, _clean_terms, build_arxiv_query


def test_clean_terms_drops_filler_words() -> None:
    assert _clean_terms("find recent arxiv papers on transformers") == ["transformers"]
    # A meaningful multi-term topic is preserved in order.
    assert _clean_terms("retrieval augmented generation") == [
        "retrieval",
        "augmented",
        "generation",
    ]


def test_clean_freetext_falls_back_when_all_filler() -> None:
    # If stripping removes everything, keep the original so the search still runs.
    assert _clean_freetext("papers") == "papers"


def test_build_query_puts_author_in_field_and_quotes_multiword() -> None:
    # The bug case: "Andrew Ng" as an author becomes au:"Andrew Ng", not a free keyword.
    assert build_arxiv_query(author="Andrew Ng") == 'au:"Andrew Ng"'
    assert build_arxiv_query(author="Ng") == "au:Ng"


def test_build_query_ands_cleaned_keywords_and_category() -> None:
    q = build_arxiv_query(keywords="recent papers on diffusion models", category="cs.LG")
    assert q == "all:diffusion AND all:models AND cat:cs.LG"


def test_build_query_combines_all_parts() -> None:
    q = build_arxiv_query(keywords="policy gradient", author="Sutton", category="cs.LG")
    # Field order is title, author, keywords, category (AND is commutative on arXiv).
    assert q == "au:Sutton AND all:policy AND all:gradient AND cat:cs.LG"


def test_build_query_empty_when_all_blank() -> None:
    assert build_arxiv_query() == ""
    assert build_arxiv_query(keywords="   ", author="", category="") == ""


def test_kb_hit_markdown_links_and_truncates() -> None:
    from src.rag.retriever import RetrievedChunk
    from src.ui.pages.tools import _kb_hit_markdown

    chunk = RetrievedChunk(
        text="Retrieval-augmented generation combines a retriever with a generator. " * 10,
        metadata={"title": "RAG", "source": "rag.md", "url": "https://example.com/rag"},
        vector_score=0.8,
        bm25_score=0.2,
        score=0.66,
    )
    md = _kb_hit_markdown(chunk)
    assert "[RAG](https://example.com/rag)" in md
    assert "`rag.md`" in md
    assert "0.66" in md
    assert md.endswith("…")


def test_kb_hit_markdown_neutralizes_markdown_and_latex() -> None:
    # KB notes are .md files; their text must not render as headings/bold/math (the bug where
    # a snippet showed up in big bold letters). The snippet line must be plain prose.
    from src.rag.retriever import RetrievedChunk
    from src.ui.pages.tools import _kb_hit_markdown

    chunk = RetrievedChunk(
        text="## Batch norm: $\\hat{x} = \\frac{x - \\mu}{\\sqrt{\\sigma^2}}$ with **weights**",
        metadata={"title": "Classical ML", "source": "classical.md"},
        vector_score=0.5,
        bm25_score=0.5,
        score=0.24,
    )
    snippet_line = _kb_hit_markdown(chunk).split("\n", 1)[1]
    # No markdown/LaTeX control chars, and no leftover LaTeX command names.
    for control in ("#", "$", "`", "*", "[", "]", "{", "}", "\\", "^"):
        assert control not in snippet_line
    for command in ("frac", "hat", "sqrt", "mu", "sigma"):
        assert command not in snippet_line
    assert "weights" in snippet_line and "Batch norm" in snippet_line
