"""Regression tests for CitationManager citation numbering (agentic-titan#107).

Covers the defect where ``CitationManager.add_source()`` overwrote a
source's provider-supplied ``relevance_score`` with the citation sequence
number. Registering a citation must never mutate the search result.
"""

from __future__ import annotations

from tools.search import CitationManager, SearchSource


def make_source(url: str, score: float = 0.73) -> SearchSource:
    return SearchSource(url, "Title", "Snippet", relevance_score=score)


def test_add_source_preserves_relevance_score() -> None:
    manager = CitationManager()
    source = make_source("https://example.com/ranked", 0.73)
    number = manager.add_source(source)
    assert number == 1
    assert source.relevance_score == 0.73


def test_add_source_preserves_zero_and_non_integer_scores() -> None:
    manager = CitationManager()
    zero = make_source("https://example.com/zero", 0.0)
    fractional = make_source("https://example.com/fractional", 0.125)
    manager.add_source(zero)
    manager.add_source(fractional)
    assert zero.relevance_score == 0.0
    assert fractional.relevance_score == 0.125


def test_duplicate_url_returns_same_citation_without_mutation() -> None:
    manager = CitationManager()
    first = make_source("https://example.com/dup", 0.73)
    duplicate = make_source("https://example.com/dup", 0.91)
    first_number = manager.add_source(first)
    second_number = manager.add_source(duplicate)
    assert first_number == 1
    assert second_number == 1
    # Neither the stored source nor the duplicate object is mutated.
    assert first.relevance_score == 0.73
    assert duplicate.relevance_score == 0.91


def test_citation_numbers_stable_one_based_insertion_order() -> None:
    manager = CitationManager()
    urls = [
        "https://example.com/a",
        "https://example.com/b",
        "https://example.com/c",
    ]
    numbers = [manager.add_source(make_source(u)) for u in urls]
    assert numbers == [1, 2, 3]
    # Re-adding existing URLs keeps the original numbers.
    assert manager.add_source(make_source(urls[0])) == 1
    assert manager.add_source(make_source(urls[2])) == 3
    assert manager.add_source(make_source("https://example.com/d")) == 4


def test_citation_lists_and_formatting_in_insertion_order() -> None:
    manager = CitationManager()
    manager.add_source(SearchSource("https://example.com/a", "Alpha", "s1"))
    manager.add_source(SearchSource("https://example.com/b", "Beta", "s2"))
    citations = manager.get_citations()
    assert citations == [
        "[1] Alpha - https://example.com/a",
        "[2] Beta - https://example.com/b",
    ]
    block = manager.format_citations_block()
    assert "[1] [Alpha](https://example.com/a)" in block
    assert "[2] [Beta](https://example.com/b)" in block
    assert block.index("[1]") < block.index("[2]")
