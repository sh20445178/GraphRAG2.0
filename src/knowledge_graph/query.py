"""
Knowledge Graph Querier
=======================
Retrieves relevant subgraphs and serialises them as text context
for the LLM generator.
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
from loguru import logger

from src.utils.config import get_settings


class GraphQuerier:
    """Queries the persisted knowledge graph for context relevant to a query."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._graph = self._load_graph()

    # ── Public ─────────────────────────────────────────────────────────────

    def get_context(self, query: str, ml_task: str = "general", depth: int = 2) -> str:
        if self._graph.number_of_nodes() == 0:
            return ""

        seed_nodes = self._find_seed_nodes(query, ml_task)
        if not seed_nodes:
            return ""

        subgraph = self._extract_subgraph(seed_nodes, depth)
        return self._serialise(subgraph)

    def get_related_algorithms(self, task: str) -> list[str]:
        """Returns algorithm/architecture node names suitable for a given ML task slug."""
        task_node = self._resolve_task_node(task)
        if not task_node or task_node not in self._graph:
            return []
        algorithms = []
        for node, data in self._graph.nodes(data=True):
            if data.get("label") in {"Algorithm", "Architecture"}:
                if self._graph.has_edge(node, task_node):
                    algorithms.append(data.get("name", node))
        return algorithms

    # ── Internals ──────────────────────────────────────────────────────────

    def _find_seed_nodes(self, query: str, ml_task: str) -> list[str]:
        q_lower = query.lower()
        seeds: list[str] = []
        for node, data in self._graph.nodes(data=True):
            name = data.get("name", node).lower()
            if name in q_lower or node in q_lower:
                seeds.append(node)
        # Always include the task node
        task_node = self._resolve_task_node(ml_task)
        if task_node and task_node in self._graph:
            seeds.append(task_node)
        return list(set(seeds))[:10]

    def _extract_subgraph(self, seeds: list[str], depth: int) -> nx.DiGraph:
        nodes = set(seeds)
        for seed in seeds:
            try:
                neighbourhood = nx.single_source_shortest_path_length(
                    self._graph, seed, cutoff=depth
                )
                nodes.update(neighbourhood.keys())
            except nx.NodeNotFound:
                pass
        return self._graph.subgraph(nodes).copy()

    def _serialise(self, subgraph: nx.DiGraph) -> str:
        if subgraph.number_of_nodes() == 0:
            return ""
        lines: list[str] = ["### Knowledge Graph Context\n"]
        for node, data in subgraph.nodes(data=True):
            label = data.get("label", "")
            name = data.get("name", node)
            desc = data.get("description", "")
            lines.append(f"- **{name}** [{label}]{': ' + desc if desc else ''}")

        lines.append("\n**Relationships:**")
        for src, tgt, data in subgraph.edges(data=True):
            src_name = subgraph.nodes[src].get("name", src)
            tgt_name = subgraph.nodes[tgt].get("name", tgt)
            rel = data.get("relation", "RELATED")
            lines.append(f"  {src_name} --[{rel}]--> {tgt_name}")

        return "\n".join(lines)

    @staticmethod
    def _resolve_task_node(ml_task: str) -> str | None:
        mapping = {
            "classification": "classification",
            "regression": "regression",
            "nlp": "nlp",
            "cv": "cv",
            "computer_vision": "cv",
            "reinforcement_learning": "reinforcement-learning",
            "time_series": "time-series",
            "generative": "generative-ai",
            "generative_ai": "generative-ai",
        }
        return mapping.get(ml_task.lower())

    def _load_graph(self) -> nx.DiGraph:
        path = self._settings.graph_store_path
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            g = nx.node_link_graph(data)
            logger.debug("GraphQuerier loaded graph: %d nodes.", g.number_of_nodes())
            return g
        logger.debug("No graph file found at %s.", path)
        return nx.DiGraph()
