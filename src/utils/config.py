"""Central configuration loaded from environment / .env file."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    llm_model: str = Field(default="gpt-4o-mini")
    embedding_model: str = Field(default="text-embedding-3-small")

    # Optional services
    cohere_api_key: str = Field(default="", alias="COHERE_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")

    # Vector store
    vector_store_path: Path = Field(default=Path("./data/vector_store"))
    chroma_collection_name: str = Field(default="graphrag_ml")

    # Knowledge graph
    graph_store_path: Path = Field(default=Path("./data/knowledge_graph.json"))

    # RAG hyperparameters
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    top_k_retrieval: int = Field(default=10)
    top_k_rerank: int = Field(default=5)
    similarity_threshold: float = Field(default=0.5)

    # LangSmith tracing
    langchain_tracing_v2: bool = Field(default=False)
    langchain_api_key: str = Field(default="", alias="LANGCHAIN_API_KEY")
    langchain_project: str = Field(default="GraphRAG2.0-ML")

    def ensure_dirs(self) -> None:
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        self.graph_store_path.parent.mkdir(parents=True, exist_ok=True)
        Path("./data/raw").mkdir(parents=True, exist_ok=True)
        Path("./data/processed").mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
