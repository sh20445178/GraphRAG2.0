"""
Smart document chunker with sentence-aware splitting and metadata enrichment.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from src.utils.config import get_settings


class SmartChunker:
    """Splits documents while preserving semantic boundaries."""

    def __init__(self) -> None:
        s = get_settings()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=s.chunk_size,
            chunk_overlap=s.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
            is_separator_regex=False,
        )

    def chunk(self, documents: Iterable[Document]) -> list[Document]:
        chunks: list[Document] = []
        for doc in documents:
            splits = self._splitter.split_documents([doc])
            for i, chunk in enumerate(splits):
                chunk.metadata.setdefault("chunk_index", i)
                chunk.metadata.setdefault("total_chunks", len(splits))
                chunks.append(chunk)
        logger.debug("Produced %d chunks from documents.", len(chunks))
        return chunks
