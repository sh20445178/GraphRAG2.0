"""
Knowledge Graph Builder
=======================
Extracts entities and relations from text using an LLM,
then stores them in a NetworkX DiGraph (persisted as JSON).
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import networkx as nx
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger

from src.knowledge_graph.entities import Entity, Relation, ENTITY_LABELS, RELATION_TYPES
from src.utils.config import get_settings


_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", f"""You are a knowledge-graph extractor specialised in Machine Learning.
Extract entities and relations from the provided text.

Entity labels (use only these): {sorted(ENTITY_LABELS)}
Relation types (use only these): {sorted(RELATION_TYPES)}

Return a JSON object with two lists:
{{
  "entities": [
    {{"id": "<slug>", "label": "<EntityLabel>", "name": "<display name>",
      "description": "<one sentence>"}}
  ],
  "relations": [
    {{"source_id": "<slug>", "target_id": "<slug>", "relation_type": "<TYPE>"}}
  ]
}}

Use short, lowercase hyphenated slugs for ids (e.g. "random-forest", "imagenet").
Only extract clearly stated facts. Respond ONLY with valid JSON."""),
    ("human", "{text}"),
])


class KnowledgeGraphBuilder:
    """Builds and persists an ML knowledge graph."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._graph: nx.DiGraph = nx.DiGraph()
        self._load()

    # ── Public API ─────────────────────────────────────────────────────────

    @property
    def graph(self) -> nx.DiGraph:
        return self._graph

    def add_documents(self, documents: list[Document]) -> None:
        llm = ChatOpenAI(
            model=self._settings.llm_model,
            temperature=0,
            openai_api_key=self._settings.openai_api_key,
        )
        chain = _EXTRACTION_PROMPT | llm
        for doc in documents:
            try:
                self._extract_and_add(doc.page_content[:3000], chain)
            except Exception as exc:
                logger.warning("KG extraction failed for doc: %s", exc)
        self._save()

    def add_seed_knowledge(self) -> None:
        """Pre-populates the graph with a curated ML taxonomy."""
        _build_seed_graph(self._graph)
        self._save()

    # ── Internals ──────────────────────────────────────────────────────────

    def _extract_and_add(self, text: str, chain: Any) -> None:
        result = chain.invoke({"text": text})
        try:
            data: dict[str, Any] = json.loads(result.content)
        except json.JSONDecodeError:
            logger.warning("Could not parse KG extraction JSON.")
            return

        for e in data.get("entities", []):
            self._graph.add_node(
                e["id"],
                label=e.get("label", ""),
                name=e.get("name", e["id"]),
                description=e.get("description", ""),
            )

        for r in data.get("relations", []):
            src, tgt = r.get("source_id"), r.get("target_id")
            if src and tgt and self._graph.has_node(src) and self._graph.has_node(tgt):
                self._graph.add_edge(src, tgt, relation=r.get("relation_type", "RELATED"))

    # ── Persistence ────────────────────────────────────────────────────────

    def _save(self) -> None:
        path = self._settings.graph_store_path
        data = nx.node_link_data(self._graph)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        logger.info("Knowledge graph saved: %d nodes, %d edges.", self._graph.number_of_nodes(), self._graph.number_of_edges())

    def _load(self) -> None:
        path = self._settings.graph_store_path
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self._graph = nx.node_link_graph(data)
            logger.info("Knowledge graph loaded: %d nodes, %d edges.", self._graph.number_of_nodes(), self._graph.number_of_edges())
        else:
            self._graph = nx.DiGraph()


# ── Seed ML taxonomy ──────────────────────────────────────────────────────

def _build_seed_graph(g: nx.DiGraph) -> None:
    """Adds a curated ML taxonomy so the graph is useful even before ingestion."""
    nodes = [
        # Tasks
        ("classification", "Task", "Classification"),
        ("regression", "Task", "Regression"),
        ("nlp", "Task", "Natural Language Processing"),
        ("cv", "Task", "Computer Vision"),
        ("reinforcement-learning", "Task", "Reinforcement Learning"),
        ("time-series", "Task", "Time Series Forecasting"),
        ("generative-ai", "Task", "Generative AI"),
        # Algorithms
        ("random-forest", "Algorithm", "Random Forest"),
        ("xgboost", "Algorithm", "XGBoost"),
        ("lightgbm", "Algorithm", "LightGBM"),
        ("svm", "Algorithm", "Support Vector Machine"),
        ("transformer", "Architecture", "Transformer"),
        ("bert", "Architecture", "BERT"),
        ("gpt", "Architecture", "GPT"),
        ("resnet", "Architecture", "ResNet"),
        ("lstm", "Architecture", "LSTM"),
        ("cnn", "Architecture", "Convolutional Neural Network"),
        # Frameworks
        ("pytorch", "Framework", "PyTorch"),
        ("tensorflow", "Framework", "TensorFlow"),
        ("sklearn", "Framework", "scikit-learn"),
        ("huggingface", "Framework", "HuggingFace Transformers"),
        ("keras", "Framework", "Keras"),
        # Datasets
        ("imagenet", "Dataset", "ImageNet"),
        ("mnist", "Dataset", "MNIST"),
        ("glue", "Dataset", "GLUE Benchmark"),
        # Metrics
        ("accuracy", "Metric", "Accuracy"),
        ("f1-score", "Metric", "F1 Score"),
        ("rmse", "Metric", "RMSE"),
        ("bleu", "Metric", "BLEU Score"),
        ("auc-roc", "Metric", "AUC-ROC"),
        # Optimizers
        ("adam", "Optimizer", "Adam"),
        ("sgd", "Optimizer", "SGD"),
        ("adamw", "Optimizer", "AdamW"),
    ]

    for node_id, label, name in nodes:
        g.add_node(node_id, label=label, name=name, description="")

    edges = [
        ("random-forest", "classification", "SUITABLE_FOR"),
        ("random-forest", "regression", "SUITABLE_FOR"),
        ("xgboost", "classification", "SUITABLE_FOR"),
        ("xgboost", "regression", "SUITABLE_FOR"),
        ("lightgbm", "classification", "SUITABLE_FOR"),
        ("transformer", "nlp", "SUITABLE_FOR"),
        ("bert", "nlp", "SUITABLE_FOR"),
        ("bert", "transformer", "VARIANT_OF"),
        ("gpt", "transformer", "VARIANT_OF"),
        ("gpt", "nlp", "SUITABLE_FOR"),
        ("gpt", "generative-ai", "SUITABLE_FOR"),
        ("resnet", "cv", "SUITABLE_FOR"),
        ("cnn", "cv", "SUITABLE_FOR"),
        ("lstm", "time-series", "SUITABLE_FOR"),
        ("lstm", "nlp", "SUITABLE_FOR"),
        ("svm", "classification", "SUITABLE_FOR"),
        ("random-forest", "sklearn", "USES"),
        ("xgboost", "sklearn", "USES"),
        ("bert", "pytorch", "USES"),
        ("gpt", "pytorch", "USES"),
        ("resnet", "pytorch", "USES"),
        ("bert", "huggingface", "USES"),
        ("gpt", "huggingface", "USES"),
        ("bert", "glue", "EVALUATED_ON"),
        ("resnet", "imagenet", "TRAINED_ON"),
        ("cnn", "mnist", "TRAINED_ON"),
    ]

    for src, tgt, rel in edges:
        g.add_edge(src, tgt, relation=rel)
