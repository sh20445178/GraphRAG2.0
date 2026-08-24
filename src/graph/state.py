"""LangGraph state schema for the GraphRAG 2.0 pipeline."""
from __future__ import annotations

from typing import Annotated, Any
from typing_extensions import TypedDict

from langchain_core.documents import Document
from langgraph.graph.message import add_messages


class RAGState(TypedDict):
    """Shared state that flows through every node in the graph."""

    # ── Input ──────────────────────────────────────────────────────────────
    query: str
    """Raw user query."""

    # ── Conversation history (append-only via add_messages reducer) ────────
    messages: Annotated[list[Any], add_messages]

    # ── Query analysis ──────────────────────────────────────────────────────
    refined_query: str
    """Query after expansion / re-writing."""
    sub_queries: list[str]
    """Decomposed sub-queries for multi-hop retrieval."""
    query_intent: str
    """Detected intent: e.g. 'explain', 'compare', 'code', 'recommend'."""

    # ── Retrieval ───────────────────────────────────────────────────────────
    retrieved_docs: list[Document]
    """Raw docs from vector + graph retrieval."""
    reranked_docs: list[Document]
    """Documents after reranking."""

    # ── Knowledge graph context ─────────────────────────────────────────────
    graph_context: str
    """Serialised subgraph / entity relations relevant to the query."""

    # ── Generation ──────────────────────────────────────────────────────────
    answer: str
    """Current generated answer."""
    code_snippets: list[str]
    """Any generated code blocks."""

    # ── Reflection ──────────────────────────────────────────────────────────
    reflection_score: float
    """Self-critique score in [0, 1]."""
    reflection_feedback: str
    """Critique text used to guide regeneration."""
    iteration: int
    """Reflection iteration counter."""

    # ── ML-specific ─────────────────────────────────────────────────────────
    ml_task: str
    """Detected ML task type: classification, regression, nlp, cv, etc."""
    recommended_models: list[dict[str, Any]]
    """Structured model recommendations."""

    # ── Customer feedback analysis ───────────────────────────────────────────
    feedback_themes: list[dict[str, Any]]
    """Extracted themes with counts, sentiment, and example quotes."""
    sentiment_distribution: dict[str, float]
    """Ratio of positive / neutral / negative feedback."""
    trend_summary: str
    """Structured Markdown trend report produced by trend_summarizer."""


