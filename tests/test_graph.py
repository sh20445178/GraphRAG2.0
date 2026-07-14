"""Tests for knowledge graph builder and querier (no LLM calls)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import networkx as nx

from src.knowledge_graph.builder import KnowledgeGraphBuilder, _build_seed_graph
from src.knowledge_graph.query import GraphQuerier


@pytest.fixture()
def tmp_graph_path(tmp_path: Path) -> Path:
    return tmp_path / "test_graph.json"


class TestSeedGraph:
    def test_seed_has_nodes_and_edges(self) -> None:
        g = nx.DiGraph()
        _build_seed_graph(g)
        assert g.number_of_nodes() > 20
        assert g.number_of_edges() > 10

    def test_random_forest_classification_edge(self) -> None:
        g = nx.DiGraph()
        _build_seed_graph(g)
        assert g.has_edge("random-forest", "classification")
        assert g["random-forest"]["classification"]["relation"] == "SUITABLE_FOR"

    def test_bert_is_transformer_variant(self) -> None:
        g = nx.DiGraph()
        _build_seed_graph(g)
        assert g.has_edge("bert", "transformer")


class TestGraphQuerier:
    def _make_querier_with_seed(self, tmp_path: Path) -> GraphQuerier:
        """Writes a seed graph to a temp file and loads GraphQuerier from it."""
        g = nx.DiGraph()
        _build_seed_graph(g)
        path = tmp_path / "graph.json"
        with open(path, "w") as f:
            json.dump(nx.node_link_data(g), f)

        # Patch the settings to point at our temp file
        import src.utils.config as cfg
        original = cfg.get_settings()
        original.graph_store_path = path  # type: ignore[misc]

        from src.knowledge_graph.query import GraphQuerier
        querier = GraphQuerier.__new__(GraphQuerier)
        querier._settings = original  # type: ignore[attr-defined]
        querier._graph = g
        return querier

    def test_get_related_algorithms_classification(self, tmp_path: Path) -> None:
        querier = self._make_querier_with_seed(tmp_path)
        algos = querier.get_related_algorithms("classification")
        assert len(algos) > 0
        names_lower = [a.lower() for a in algos]
        assert any("random forest" in n or "xgboost" in n or "svm" in n for n in names_lower)

    def test_get_context_returns_string(self, tmp_path: Path) -> None:
        querier = self._make_querier_with_seed(tmp_path)
        ctx = querier.get_context("random forest classification", ml_task="classification")
        assert isinstance(ctx, str)
        assert len(ctx) > 0

    def test_empty_graph_returns_empty_string(self) -> None:
        from src.knowledge_graph.query import GraphQuerier
        querier = GraphQuerier.__new__(GraphQuerier)
        querier._graph = nx.DiGraph()  # type: ignore[attr-defined]
        querier._settings = None  # type: ignore[attr-defined]
        assert querier.get_context("anything") == ""
