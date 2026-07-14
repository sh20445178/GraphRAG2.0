"""
Example 1 – Basic RAG query
============================
Run: python examples/basic_rag.py
"""
from __future__ import annotations

from src.graph.workflow import compile_graph
from src.knowledge_graph.builder import KnowledgeGraphBuilder


def main() -> None:
    # 1. Seed the knowledge graph on first run
    print("Seeding knowledge graph…")
    builder = KnowledgeGraphBuilder()
    builder.add_seed_knowledge()

    # 2. Compile the LangGraph pipeline
    graph = compile_graph()

    # 3. Ask a question
    query = "What is the best algorithm for binary text classification on a small dataset?"
    print(f"\nQuery: {query}\n{'='*60}")

    result = graph.invoke({"query": query, "messages": []})

    print(f"ML Task Detected : {result['ml_task']}")
    print(f"Query Intent     : {result['query_intent']}")
    print(f"Reflection Score : {result.get('reflection_score', 0):.2f}")
    print(f"\nAnswer:\n{result['answer']}")

    if result.get("recommended_models"):
        print("\n--- Model Recommendations ---")
        for rec in result["recommended_models"]:
            print(f"• {rec['name']} ({rec.get('category', '')})")


if __name__ == "__main__":
    main()
