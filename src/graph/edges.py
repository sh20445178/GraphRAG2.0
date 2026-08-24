"""Routing / conditional-edge logic for the LangGraph workflow."""
from __future__ import annotations

from src.graph.state import RAGState

MAX_ITERATIONS = 2
REFLECTION_THRESHOLD = 0.85


def route_after_reflection(state: RAGState) -> str:
    """
    Decide whether to regenerate or finalise:
      - If score is good enough OR we've hit the iteration cap → END
      - Otherwise → regenerate
    """
    score = state.get("reflection_score", 0.0)
    iteration = state.get("iteration", 0)

    if score >= REFLECTION_THRESHOLD or iteration >= MAX_ITERATIONS:
        return "end"
    return "regenerate"


def route_by_intent(state: RAGState) -> str:
    """After reranking, branch on query intent."""
    intent = state.get("query_intent", "general")
    if intent == "recommend":
        return "ml_advisor"
    if intent == "feedback_trends":
        return "feedback"
    return "generate"
