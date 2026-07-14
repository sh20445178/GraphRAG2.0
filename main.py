"""
GraphRAG 2.0 – Main CLI entry point
====================================
Usage:
  python main.py ask "What model should I use for tabular classification?"
  python main.py index --file paper.pdf
  python main.py index --arxiv 2005.11401
  python main.py seed-graph
  python main.py chat
"""
from __future__ import annotations

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

app = typer.Typer(help="GraphRAG 2.0 – LangGraph ML knowledge assistant")
console = Console()


# ── ask ───────────────────────────────────────────────────────────────────

@app.command()
def ask(
    query: str = typer.Argument(..., help="Natural language question about ML"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run a single query through the full GraphRAG 2.0 pipeline."""
    from src.graph.workflow import compile_graph

    console.print(Panel(f"[bold cyan]Query:[/] {query}", expand=False))

    graph = compile_graph()
    result = graph.invoke({"query": query, "messages": []})

    if verbose:
        console.print(f"\n[dim]Refined query:[/] {result.get('refined_query')}")
        console.print(f"[dim]ML task:[/] {result.get('ml_task')}")
        console.print(f"[dim]Intent:[/] {result.get('query_intent')}")
        console.print(f"[dim]Reflection score:[/] {result.get('reflection_score', 0):.2f}")

    console.print("\n")
    console.print(Markdown(result.get("answer", "No answer generated.")))

    if result.get("recommended_models"):
        console.print("\n[bold green]Model Recommendations:[/]")
        for i, rec in enumerate(result["recommended_models"], 1):
            console.print(f"  {i}. {rec.get('name')} – {rec.get('suitability', '')[:80]}")


# ── index ─────────────────────────────────────────────────────────────────

@app.command()
def index(
    file: str | None = typer.Option(None, "--file", "-f", help="Path to PDF/DOCX/TXT/MD"),
    arxiv: str | None = typer.Option(None, "--arxiv", "-a", help="arXiv paper ID"),
    directory: str | None = typer.Option(None, "--dir", "-d", help="Directory to recursively index"),
    build_graph: bool = typer.Option(True, "--build-graph/--no-graph", help="Also extract KG triples"),
) -> None:
    """Index documents into the vector store (and optionally the knowledge graph)."""
    from src.rag.indexer import DocumentIndexer
    from src.knowledge_graph.builder import KnowledgeGraphBuilder

    indexer = DocumentIndexer()

    if file:
        n = indexer.index_file(file)
        console.print(f"[green]✓[/] Indexed {n} chunks from {file}")

    if arxiv:
        n = indexer.index_arxiv(arxiv)
        console.print(f"[green]✓[/] Indexed {n} chunks from arXiv:{arxiv}")

    if directory:
        n = indexer.index_directory(directory)
        console.print(f"[green]✓[/] Indexed {n} chunks from directory {directory}")

    if build_graph and (file or arxiv or directory):
        console.print("[yellow]Building knowledge graph triples…[/]")
        KnowledgeGraphBuilder().add_seed_knowledge()
        console.print("[green]✓[/] Knowledge graph updated.")


# ── seed-graph ────────────────────────────────────────────────────────────

@app.command(name="seed-graph")
def seed_graph() -> None:
    """Populate the knowledge graph with curated ML taxonomy (no LLM call needed)."""
    from src.knowledge_graph.builder import KnowledgeGraphBuilder

    builder = KnowledgeGraphBuilder()
    builder.add_seed_knowledge()
    g = builder.graph
    console.print(f"[green]✓[/] Seed graph created: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges.")


# ── chat ──────────────────────────────────────────────────────────────────

@app.command()
def chat() -> None:
    """Interactive multi-turn chat with the GraphRAG 2.0 pipeline."""
    from src.graph.workflow import compile_graph

    console.print(Panel("[bold]GraphRAG 2.0 – Interactive ML Assistant[/]\nType 'exit' to quit.", style="cyan"))
    graph = compile_graph()
    messages: list = []

    while True:
        query = console.input("\n[bold cyan]You:[/] ").strip()
        if query.lower() in {"exit", "quit", "q"}:
            break
        if not query:
            continue

        result = graph.invoke({"query": query, "messages": messages})
        messages = result.get("messages", [])
        console.print("\n[bold green]Assistant:[/]")
        console.print(Markdown(result.get("answer", "…")))


if __name__ == "__main__":
    app()
