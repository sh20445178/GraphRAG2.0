"""
RAG 2.0 Hybrid Retriever
========================
Combines:
  1. Dense vector search  (Chroma / cosine similarity)
  2. Sparse keyword search (BM25 via rank_bm25)
  3. De-duplication & score fusion (Reciprocal Rank Fusion)
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from loguru import logger

from src.utils.config import get_settings
from src.utils.embeddings import get_embeddings


class HybridRetriever:
    """Fuses vector and BM25 retrieval with Reciprocal Rank Fusion."""

    def __init__(self, embedding_backend: str = "openai") -> None:
        self._settings = get_settings()
        self._embeddings = get_embeddings(embedding_backend)  # type: ignore[arg-type]
        self._vectorstore: Chroma | None = None

    # ── Public ─────────────────────────────────────────────────────────────

    def retrieve(self, queries: Sequence[str], top_k: int | None = None) -> list[Document]:
        top_k = top_k or self._settings.top_k_retrieval
        all_results: list[list[Document]] = []

        for q in queries:
            dense = self._dense_search(q, top_k)
            sparse = self._sparse_search(q, top_k)
            all_results.append(dense)
            all_results.append(sparse)

        fused = self._rrf_fuse(all_results, top_k=top_k)
        logger.debug("Hybrid retrieval: %d docs for %d queries.", len(fused), len(queries))
        return fused

    # ── Dense retrieval ────────────────────────────────────────────────────

    def _get_vectorstore(self) -> Chroma:
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                collection_name=self._settings.chroma_collection_name,
                embedding_function=self._embeddings,
                persist_directory=str(self._settings.vector_store_path),
            )
        return self._vectorstore

    def _dense_search(self, query: str, k: int) -> list[Document]:
        try:
            vs = self._get_vectorstore()
            results = vs.similarity_search_with_relevance_scores(query, k=k)
            threshold = self._settings.similarity_threshold
            docs = [doc for doc, score in results if score >= threshold]
            return docs
        except Exception as exc:
            logger.warning("Dense search failed: %s", exc)
            return []

    # ── Sparse / BM25 retrieval ────────────────────────────────────────────

    def _sparse_search(self, query: str, k: int) -> list[Document]:
        """BM25 over the current Chroma collection's stored documents."""
        try:
            from rank_bm25 import BM25Okapi  # type: ignore

            vs = self._get_vectorstore()
            # Pull all stored documents (safe for small corpora)
            collection = vs._collection  # type: ignore[attr-defined]
            raw = collection.get(include=["documents", "metadatas"])
            texts: list[str] = raw["documents"] or []
            metadatas: list[dict] = raw["metadatas"] or []

            if not texts:
                return []

            tokenized = [t.lower().split() for t in texts]
            bm25 = BM25Okapi(tokenized)
            scores = bm25.get_scores(query.lower().split())

            # Return top-k by BM25 score
            ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            return [
                Document(page_content=texts[i], metadata=metadatas[i])
                for i in ranked_idx
                if scores[i] > 0
            ]
        except ImportError:
            logger.warning("rank_bm25 not installed – skipping sparse search.")
            return []
        except Exception as exc:
            logger.warning("Sparse search failed: %s", exc)
            return []

    # ── Reciprocal Rank Fusion ─────────────────────────────────────────────

    @staticmethod
    def _rrf_fuse(
        ranked_lists: list[list[Document]],
        top_k: int,
        k: int = 60,
    ) -> list[Document]:
        scores: dict[str, float] = defaultdict(float)
        doc_map: dict[str, Document] = {}

        for ranked in ranked_lists:
            for rank, doc in enumerate(ranked):
                key = doc.page_content[:200]  # use content prefix as dedup key
                scores[key] += 1.0 / (k + rank + 1)
                doc_map[key] = doc

        sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
        return [doc_map[key] for key in sorted_keys[:top_k]]
