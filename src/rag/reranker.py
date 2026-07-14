"""
Document reranker.
Tries Cohere rerank API first; falls back to a cross-encoder locally.
"""
from __future__ import annotations

from langchain_core.documents import Document
from loguru import logger

from src.utils.config import get_settings


class Reranker:
    def __init__(self) -> None:
        self._settings = get_settings()

    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        if not documents:
            return []
        top_k = self._settings.top_k_rerank

        if self._settings.cohere_api_key:
            return self._cohere_rerank(query, documents, top_k)
        return self._cross_encoder_rerank(query, documents, top_k)

    # ── Cohere ─────────────────────────────────────────────────────────────

    def _cohere_rerank(
        self, query: str, documents: list[Document], top_k: int
    ) -> list[Document]:
        try:
            import cohere  # type: ignore

            co = cohere.Client(self._settings.cohere_api_key)
            texts = [d.page_content for d in documents]
            response = co.rerank(
                query=query,
                documents=texts,
                top_n=top_k,
                model="rerank-english-v3.0",
            )
            return [documents[r.index] for r in response.results]
        except Exception as exc:
            logger.warning("Cohere rerank failed: %s – falling back.", exc)
            return self._cross_encoder_rerank(query, documents, top_k)

    # ── Local cross-encoder fallback ───────────────────────────────────────

    def _cross_encoder_rerank(
        self, query: str, documents: list[Document], top_k: int
    ) -> list[Document]:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            pairs = [[query, d.page_content] for d in documents]
            scores = model.predict(pairs)
            ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
            return [doc for _, doc in ranked[:top_k]]
        except ImportError:
            logger.warning("sentence-transformers not available – returning first %d docs.", top_k)
            return documents[:top_k]
