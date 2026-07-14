"""Tests for HybridRetriever (mocked vector store)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.rag.retriever import HybridRetriever


def _make_doc(content: str, title: str = "Test") -> Document:
    return Document(page_content=content, metadata={"title": title})


class TestRRFFusion:
    """Unit-test the static RRF merge logic without any network calls."""

    def test_deduplication(self) -> None:
        doc = _make_doc("Hello world")
        result = HybridRetriever._rrf_fuse([[doc, doc], [doc]], top_k=5)
        assert len(result) == 1

    def test_higher_rank_wins(self) -> None:
        doc_a = _make_doc("A" * 50)
        doc_b = _make_doc("B" * 50)
        doc_c = _make_doc("C" * 50)
        # doc_a appears first in both lists → highest RRF score
        result = HybridRetriever._rrf_fuse(
            [[doc_a, doc_b, doc_c], [doc_a, doc_c, doc_b]], top_k=3
        )
        assert result[0].page_content.startswith("A")

    def test_top_k_limit(self) -> None:
        docs = [_make_doc(f"Doc {i}" * 10) for i in range(20)]
        result = HybridRetriever._rrf_fuse([docs], top_k=5)
        assert len(result) <= 5

    def test_empty_lists(self) -> None:
        assert HybridRetriever._rrf_fuse([], top_k=5) == []
        assert HybridRetriever._rrf_fuse([[]], top_k=5) == []
