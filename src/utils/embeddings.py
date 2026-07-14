"""Embedding utilities – supports OpenAI and local HuggingFace models."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from langchain_core.embeddings import Embeddings

from src.utils.config import get_settings


EmbeddingBackend = Literal["openai", "huggingface"]


@lru_cache(maxsize=4)
def get_embeddings(backend: EmbeddingBackend = "openai") -> Embeddings:
    settings = get_settings()

    if backend == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )

    # Fallback – runs fully locally, no API key required
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
