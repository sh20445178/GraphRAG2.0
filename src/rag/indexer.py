"""
Document indexer: loads raw files, chunks them, and upserts into ChromaDB.
Supports PDF, DOCX, plain text, Markdown, and arXiv paper IDs.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    ArxivLoader,
)
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from loguru import logger

from src.rag.chunker import SmartChunker
from src.utils.config import get_settings
from src.utils.embeddings import get_embeddings


def _doc_id(doc: Document) -> str:
    content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()[:12]
    source = doc.metadata.get("source", "unknown")
    return f"{source}-{content_hash}"


class DocumentIndexer:
    """Ingests documents and maintains the Chroma vector store."""

    def __init__(self, backend: str = "openai") -> None:
        self._settings = get_settings()
        self._embeddings = get_embeddings(backend)  # type: ignore[arg-type]
        self._chunker = SmartChunker()
        self._vectorstore: Chroma | None = None

    # ── Vector store access ────────────────────────────────────────────────

    def get_vectorstore(self) -> Chroma:
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                collection_name=self._settings.chroma_collection_name,
                embedding_function=self._embeddings,
                persist_directory=str(self._settings.vector_store_path),
            )
        return self._vectorstore

    # ── Ingestion ──────────────────────────────────────────────────────────

    def index_file(self, path: str | Path, metadata: dict | None = None) -> int:
        path = Path(path)
        docs = self._load_file(path)
        if metadata:
            for d in docs:
                d.metadata.update(metadata)
        return self._upsert(docs)

    def index_arxiv(self, arxiv_id: str, max_docs: int = 1) -> int:
        loader = ArxivLoader(query=arxiv_id, load_max_docs=max_docs)
        docs = loader.load()
        logger.info("Loaded %d docs from arXiv: %s", len(docs), arxiv_id)
        return self._upsert(docs)

    def index_documents(self, documents: Sequence[Document]) -> int:
        return self._upsert(list(documents))

    def index_directory(self, directory: str | Path, glob: str = "**/*.*") -> int:
        directory = Path(directory)
        total = 0
        for path in directory.glob(glob):
            if path.suffix.lower() in {".pdf", ".txt", ".md", ".docx"}:
                try:
                    total += self.index_file(path)
                except Exception as exc:
                    logger.warning("Skipping %s: %s", path, exc)
        return total

    # ── Internals ──────────────────────────────────────────────────────────

    def _load_file(self, path: Path) -> list[Document]:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            loader = PyPDFLoader(str(path))
        elif suffix == ".docx":
            loader = Docx2txtLoader(str(path))
        elif suffix in {".txt", ".md"}:
            loader = TextLoader(str(path), encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
        docs = loader.load()
        for d in docs:
            d.metadata.setdefault("source", path.name)
            d.metadata.setdefault("title", path.stem)
        return docs

    def _upsert(self, docs: list[Document]) -> int:
        if not docs:
            return 0
        chunks = self._chunker.chunk(docs)
        ids = [_doc_id(c) for c in chunks]
        vs = self.get_vectorstore()
        vs.add_documents(chunks, ids=ids)
        logger.info("Indexed %d chunks.", len(chunks))
        return len(chunks)
