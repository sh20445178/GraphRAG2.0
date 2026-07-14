"""
Example 2 – Graph-enriched RAG
================================
Indexes an arXiv paper, builds graph triples from it, then queries.
Run: python examples/graph_rag.py
"""
from __future__ import annotations

from src.rag.indexer import DocumentIndexer
from src.knowledge_graph.builder import KnowledgeGraphBuilder
from src.graph.workflow import compile_graph


def main() -> None:
    # 1. Seed taxonomy + index a paper (RAG-Fusion: 2402.03367)
    print("Seeding ML taxonomy…")
    builder = KnowledgeGraphBuilder()
    builder.add_seed_knowledge()

    print("Indexing arXiv paper 2402.03367 (RAG-Fusion)…")
    indexer = DocumentIndexer()
    indexer.index_arxiv("2402.03367", max_docs=1)

    # 2. Query graph-enriched pipeline
    graph = compile_graph()
    query = "How does RAG-Fusion improve retrieval compared to standard RAG?"
    print(f"\nQuery: {query}\n{'='*60}")

    result = graph.invoke({"query": query, "messages": []})
    print(result["answer"])

    if result.get("graph_context"):
        print("\n--- Graph Context Used ---")
        print(result["graph_context"][:600])


if __name__ == "__main__":
    main()
