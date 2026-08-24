"""
Customer Feedback Indexer
=========================
Ingests customer feedback from CSV or JSON files into the vector store
with rich metadata (date, source, category, rating, sentiment_label).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from loguru import logger

from src.rag.indexer import DocumentIndexer


def _row_to_doc(row: dict[str, Any], source: str) -> Document:
    """Convert a feedback row to a LangChain Document."""
    text = (
        row.get("text")
        or row.get("review")
        or row.get("feedback")
        or row.get("comment")
        or row.get("body")
        or ""
    ).strip()
    if not text:
        return None  # type: ignore[return-value]

    metadata: dict[str, Any] = {
        "source": source,
        "doc_type": "customer_feedback",
        "date": row.get("date") or row.get("created_at") or row.get("timestamp") or "",
        "category": row.get("category") or row.get("topic") or row.get("type") or "general",
        "rating": row.get("rating") or row.get("score") or "",
        "sentiment_label": row.get("sentiment") or row.get("label") or "",
        "product": row.get("product") or row.get("product_name") or "",
        "customer_id": row.get("customer_id") or row.get("user_id") or "",
    }
    # Strip empty metadata values to keep the store lean
    metadata = {k: v for k, v in metadata.items() if v != ""}
    return Document(page_content=text, metadata=metadata)


class FeedbackIndexer:
    """
    Loads customer feedback from CSV or JSON and indexes it via DocumentIndexer.

    CSV format (flexible column names):
        text | review | feedback | comment | body  → feedback text (required)
        date | created_at | timestamp             → date string (optional)
        category | topic | type                  → topic bucket (optional)
        rating | score                            → numeric rating (optional)
        sentiment | label                         → pre-labelled sentiment (optional)
        product | product_name                   → product name (optional)

    JSON format: a list of objects with the same keys as above.
    """

    def __init__(self, backend: str = "openai") -> None:
        self._indexer = DocumentIndexer(backend=backend)

    # ── Public API ─────────────────────────────────────────────────────────

    def index_csv(self, path: str | Path) -> int:
        path = Path(path)
        docs = self._load_csv(path)
        indexed = self._indexer.index_documents(docs)
        logger.info("Indexed %d feedback entries from %s", indexed, path.name)
        return indexed

    def index_json(self, path: str | Path) -> int:
        path = Path(path)
        docs = self._load_json(path)
        indexed = self._indexer.index_documents(docs)
        logger.info("Indexed %d feedback entries from %s", indexed, path.name)
        return indexed

    def index_feedback_file(self, path: str | Path) -> int:
        path = Path(path)
        if path.suffix.lower() == ".csv":
            return self.index_csv(path)
        if path.suffix.lower() == ".json":
            return self.index_json(path)
        raise ValueError(f"Unsupported feedback file format: {path.suffix}. Use .csv or .json")

    def index_feedback_list(self, records: list[dict[str, Any]], source: str = "inline") -> int:
        """Index a list of dicts directly (e.g., from a database query)."""
        docs = [d for r in records if (d := _row_to_doc(r, source)) is not None]
        indexed = self._indexer.index_documents(docs)
        logger.info("Indexed %d feedback entries from in-memory list.", indexed)
        return indexed

    # ── Loaders ────────────────────────────────────────────────────────────

    def _load_csv(self, path: Path) -> list[Document]:
        docs: list[Document] = []
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                doc = _row_to_doc(dict(row), path.name)
                if doc:
                    docs.append(doc)
        logger.debug("Loaded %d rows from CSV %s", len(docs), path.name)
        return docs

    def _load_json(self, path: Path) -> list[Document]:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise ValueError("JSON feedback file must contain a top-level list of objects.")
        docs = [d for row in data if (d := _row_to_doc(row, path.name)) is not None]
        logger.debug("Loaded %d records from JSON %s", len(docs), path.name)
        return docs
