"""Tests for the Wikipedia lookup.

Everything goes through the injected fetcher, so no test opens a socket. What is
pinned: a query cannot carry URL structure, a deep read genuinely reads more, a
third party's malformed JSON degrades instead of raising, and an article is
labelled as background rather than as evidence.
"""

from __future__ import annotations

from typing import Any

from src.tools.wiki import (
    MAX_DEEP_CHARACTERS,
    MAX_EXTRACT_CHARACTERS,
    MAX_SECTIONS,
    clean_query,
    look_up,
)


def responder(pages: dict[str, Any]) -> tuple[Any, list[str]]:
    """A fetcher answering by URL fragment, plus the list of URLs it was given."""
    seen: list[str] = []

    def fetch(url: str) -> Any:
        seen.append(url)
        for fragment, payload in pages.items():
            if fragment in url:
                return payload
        return None

    return fetch, seen


def search_hit(title: str = "Transverse-field Ising model") -> dict[str, Any]:
    """A search response with one result."""
    return {"query": {"search": [{"title": title}]}}


def summary(extract: str = "A model of spins in a transverse field.") -> dict[str, Any]:
    """A REST summary response."""
    return {
        "title": "Transverse-field Ising model",
        "extract": extract,
        "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Ising"}},
    }


def full(extract: str) -> dict[str, Any]:
    """An `action=query` extract response, keyed by an arbitrary page id."""
    return {"query": {"pages": {"1234": {"extract": extract}}}}


def sections(*names: str) -> dict[str, Any]:
    """An `action=parse` sections response."""
    return {"parse": {"sections": [{"line": name} for name in names]}}


# --- the query --------------------------------------------------------------


def test_a_query_cannot_carry_url_structure() -> None:
    # The term is chosen by a model, and the URL it lands in has parameters. The
    # same argument as the arXiv cleaner, one host along.
    assert clean_query("Ising&action=delete&format=xml") == "Ising action delete format xml"


def test_an_empty_query_never_reaches_the_network() -> None:
    fetch, seen = responder({})
    lookup = look_up("!!!", fetch=fetch)
    assert seen == []
    assert not lookup.found
    assert "empty" in lookup.detail


def test_the_query_is_percent_encoded_into_the_url() -> None:
    fetch, seen = responder({"list=search": search_hit(), "summary": summary()})
    look_up("ising model", fetch=fetch)
    assert "ising%20model" in seen[0]


# --- what comes back --------------------------------------------------------


def test_a_found_article_carries_its_citation() -> None:
    fetch, _ = responder({"list=search": search_hit(), "summary": summary()})
    lookup = look_up("ising", fetch=fetch)
    assert lookup.found
    assert lookup.article is not None
    assert "Wikipedia" in lookup.article.citation
    assert "en.wikipedia.org" in lookup.article.citation


def test_an_article_is_offered_as_background_not_as_evidence() -> None:
    # The distinction the whole project rests on: this text has not been checked
    # by anything here, and the prompt has to say so.
    fetch, _ = responder({"list=search": search_hit(), "summary": summary()})
    context = look_up("ising", fetch=fetch).context()
    assert "NOT verified" in context
    assert "NOT part of its reviewed corpus" in context


def test_a_shallow_extract_is_truncated() -> None:
    fetch, _ = responder({"list=search": search_hit(), "summary": summary("x" * 5000)})
    lookup = look_up("ising", fetch=fetch)
    assert lookup.article is not None
    assert len(lookup.article.extract) <= MAX_EXTRACT_CHARACTERS


# --- depth ------------------------------------------------------------------


def test_a_deep_lookup_reads_more_than_a_shallow_one() -> None:
    pages = {
        "list=search": search_hit(),
        "summary": summary("Short lead."),
        "prop=extracts": full("Long body. " * 300),
        "prop=sections": sections("History", "Exact solution"),
    }
    fetch, _ = responder(pages)
    shallow = look_up("ising", fetch=fetch)
    deep = look_up("ising", depth="deep", fetch=fetch)
    assert shallow.article is not None
    assert deep.article is not None
    assert len(deep.article.extract) > len(shallow.article.extract)
    assert deep.article.sections == ("History", "Exact solution")
    assert deep.article.depth == "deep"
    assert "(deep)" in deep.explain()


def test_a_deep_extract_is_still_capped() -> None:
    # An article on quantum mechanics runs to tens of thousands of characters, and
    # all of it would push the verified numbers out of the model's attention.
    pages = {
        "list=search": search_hit(),
        "summary": summary(),
        "prop=extracts": full("y" * 40_000),
        "prop=sections": sections(),
    }
    fetch, _ = responder(pages)
    lookup = look_up("ising", depth="deep", fetch=fetch)
    assert lookup.article is not None
    assert len(lookup.article.extract) <= MAX_DEEP_CHARACTERS


def test_a_deep_lookup_falls_back_to_the_summary_it_already_has() -> None:
    # Two of the four requests failed. A shorter article is a worse answer; no
    # article would be a refusal, and a refusal here would be wrong.
    fetch, _ = responder({"list=search": search_hit(), "summary": summary("Lead only.")})
    lookup = look_up("ising", depth="deep", fetch=fetch)
    assert lookup.found
    assert lookup.article is not None
    assert lookup.article.extract == "Lead only."
    assert lookup.article.sections == ()


def test_the_section_list_is_capped() -> None:
    pages = {
        "list=search": search_hit(),
        "summary": summary(),
        "prop=extracts": full("body"),
        "prop=sections": sections(*[f"Section {index}" for index in range(50)]),
    }
    fetch, _ = responder(pages)
    lookup = look_up("ising", depth="deep", fetch=fetch)
    assert lookup.article is not None
    assert len(lookup.article.sections) == MAX_SECTIONS


# --- the ways it produces nothing ------------------------------------------


def test_no_search_hit_is_a_result_not_an_exception() -> None:
    fetch, _ = responder({"list=search": {"query": {"search": []}}})
    lookup = look_up("nonexistent topic", fetch=fetch)
    assert not lookup.found
    assert "no article matched" in lookup.detail


def test_a_page_with_no_summary_says_so() -> None:
    fetch, _ = responder({"list=search": search_hit(), "summary": {"title": "Ising"}})
    lookup = look_up("ising", fetch=fetch)
    assert not lookup.found
    assert "no summary" in lookup.detail


def test_a_dead_network_is_a_result_not_an_exception() -> None:
    def dead(url: str) -> Any:
        return None

    lookup = look_up("ising", fetch=dead)
    assert not lookup.found
    assert lookup.context() == ""


def test_malformed_third_party_json_degrades() -> None:
    # Somebody else's API, free to change shape. Reading a shape rather than
    # indexing a chain is what keeps that from being an outage here.
    shapes: list[Any] = [[], "text", {"query": "not a dict"}, {"query": {"search": {}}}]
    for payload in shapes:
        fetch, _ = responder({"list=search": payload})
        assert not look_up("ising", fetch=fetch).found
