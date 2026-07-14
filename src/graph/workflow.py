"""Assembles the full LangGraph StateGraph for GraphRAG 2.0."""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from src.graph.state import RAGState
from src.graph.nodes import (
    node_query_analyser,
    node_retriever,
    node_graph_enricher,
    node_reranker,
    node_ml_advisor,
    node_generator,
    node_reflection,
    node_regenerator,
)
from src.graph.edges import route_after_reflection, route_by_intent


def build_graph() -> StateGraph:
    """
    GraphRAG 2.0 pipeline:

    query_analyser
         │
    ┌────┴─────┐
    retriever  graph_enricher   ← parallel
    └────┬─────┘
       reranker
         │
      ┌──┴──┐
    ml_advisor (recommend) or generator (other intents)
         │
      generator
         │
      reflection
         │
    ┌────┴─────┐
    END    regenerator → reflection (loop, max 2x)
    """
    builder = StateGraph(RAGState)

    # ── Register nodes ─────────────────────────────────────────────────────
    builder.add_node("query_analyser", node_query_analyser)
    builder.add_node("retriever", node_retriever)
    builder.add_node("graph_enricher", node_graph_enricher)
    builder.add_node("reranker", node_reranker)
    builder.add_node("ml_advisor", node_ml_advisor)
    builder.add_node("generator", node_generator)
    builder.add_node("reflection", node_reflection)
    builder.add_node("regenerator", node_regenerator)

    # ── Entry point ────────────────────────────────────────────────────────
    builder.set_entry_point("query_analyser")

    # ── Linear flow ────────────────────────────────────────────────────────
    builder.add_edge("query_analyser", "retriever")
    builder.add_edge("query_analyser", "graph_enricher")  # parallel branch

    # Both retriever and graph_enricher feed reranker
    builder.add_edge("retriever", "reranker")
    builder.add_edge("graph_enricher", "reranker")

    # ── Intent-based branching ─────────────────────────────────────────────
    builder.add_conditional_edges(
        "reranker",
        route_by_intent,
        {"ml_advisor": "ml_advisor", "generate": "generator"},
    )
    builder.add_edge("ml_advisor", "generator")

    # ── Reflection loop ────────────────────────────────────────────────────
    builder.add_edge("generator", "reflection")
    builder.add_conditional_edges(
        "reflection",
        route_after_reflection,
        {"end": END, "regenerate": "regenerator"},
    )
    builder.add_edge("regenerator", "reflection")

    return builder


def compile_graph():
    """Returns a compiled, runnable LangGraph application."""
    return build_graph().compile()
