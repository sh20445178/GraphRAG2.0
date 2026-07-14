"""
ML Model Advisor
================
Recommends the most appropriate ML models/architectures based on:
  - The user's natural-language query
  - The detected ML task
  - Retrieved context documents
  - The knowledge graph
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger

from src.knowledge_graph.query import GraphQuerier
from src.utils.config import get_settings


_ADVISOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a senior ML architect advising on model selection.

ML Task detected: {ml_task}
Knowledge graph says these models are suitable: {graph_models}

Context from documents:
{context}

Provide 3-5 model recommendations as a JSON array. Each object must have:
  - name          : model/algorithm name
  - category      : classical_ml | deep_learning | transformer | ensemble | other
  - suitability   : why it fits this task (2-3 sentences)
  - pros          : list of 3 advantages
  - cons          : list of 2-3 limitations
  - frameworks    : list of recommended Python frameworks
  - starter_code  : minimal Python snippet (as a string) showing how to instantiate the model
  - complexity    : low | medium | high  (training complexity)
  - data_size_req : small | medium | large  (data requirement)

Respond ONLY with a valid JSON array."""),
    ("human", "Query: {query}"),
])


class MLModelAdvisor:
    """Generates structured ML model recommendations."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._graph_querier = GraphQuerier()

    def recommend(
        self,
        query: str,
        ml_task: str = "general",
        context_docs: list[Document] | None = None,
    ) -> list[dict[str, Any]]:
        graph_models = self._graph_querier.get_related_algorithms(ml_task)
        context = self._format_docs(context_docs or [])

        llm = ChatOpenAI(
            model=self._settings.llm_model,
            temperature=0.1,
            openai_api_key=self._settings.openai_api_key,
        )
        chain = _ADVISOR_PROMPT | llm
        result = chain.invoke({
            "query": query,
            "ml_task": ml_task,
            "graph_models": ", ".join(graph_models) if graph_models else "none found in graph",
            "context": context,
        })

        try:
            recommendations: list[dict[str, Any]] = json.loads(result.content)
        except json.JSONDecodeError:
            logger.warning("Could not parse model recommendations JSON.")
            recommendations = []

        return recommendations

    @staticmethod
    def _format_docs(docs: list[Document]) -> str:
        if not docs:
            return "No additional context."
        return "\n\n".join(
            f"[{i+1}] {d.metadata.get('title','')}: {d.page_content[:400]}"
            for i, d in enumerate(docs)
        )
