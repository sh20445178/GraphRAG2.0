"""Tests for the LangGraph workflow (mocked LLM)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import json

import pytest

from src.graph.edges import route_after_reflection, route_by_intent, MAX_ITERATIONS, REFLECTION_THRESHOLD


class TestReflectionRouting:
    def test_routes_to_end_on_high_score(self) -> None:
        state = {"reflection_score": REFLECTION_THRESHOLD, "iteration": 1}
        assert route_after_reflection(state) == "end"  # type: ignore[arg-type]

    def test_routes_to_regenerate_on_low_score(self) -> None:
        state = {"reflection_score": 0.4, "iteration": 1}
        assert route_after_reflection(state) == "regenerate"  # type: ignore[arg-type]

    def test_routes_to_end_on_max_iterations(self) -> None:
        state = {"reflection_score": 0.2, "iteration": MAX_ITERATIONS}
        assert route_after_reflection(state) == "end"  # type: ignore[arg-type]


class TestIntentRouting:
    def test_recommend_intent_goes_to_advisor(self) -> None:
        state = {"query_intent": "recommend"}
        assert route_by_intent(state) == "ml_advisor"  # type: ignore[arg-type]

    def test_other_intents_go_to_generate(self) -> None:
        for intent in ["explain", "code", "compare", "debug", "general"]:
            state = {"query_intent": intent}
            assert route_by_intent(state) == "generate"  # type: ignore[arg-type]
