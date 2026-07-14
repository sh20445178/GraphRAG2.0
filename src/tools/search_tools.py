"""
LangChain-compatible tools that can be used inside a LangGraph
ToolNode or bound to a ReAct agent.
"""
from __future__ import annotations

from langchain_core.tools import tool

from src.rag.indexer import DocumentIndexer
from src.rag.retriever import HybridRetriever
from src.knowledge_graph.builder import KnowledgeGraphBuilder
from src.knowledge_graph.query import GraphQuerier


@tool
def index_arxiv_paper(arxiv_id: str) -> str:
    """Index an arXiv paper by its ID (e.g. '2005.11401') into the vector store."""
    indexer = DocumentIndexer()
    n = indexer.index_arxiv(arxiv_id)
    return f"Indexed {n} chunks from arXiv paper {arxiv_id}."


@tool
def index_local_file(file_path: str) -> str:
    """Index a local PDF, DOCX, TXT, or MD file into the vector store."""
    indexer = DocumentIndexer()
    n = indexer.index_file(file_path)
    return f"Indexed {n} chunks from {file_path}."


@tool
def search_knowledge_base(query: str) -> str:
    """Search the RAG knowledge base for relevant ML information."""
    retriever = HybridRetriever()
    docs = retriever.retrieve([query], top_k=5)
    if not docs:
        return "No relevant documents found."
    parts = [f"[{i+1}] {d.metadata.get('title','')}: {d.page_content[:300]}" for i, d in enumerate(docs)]
    return "\n\n".join(parts)


@tool
def build_knowledge_graph_from_text(text: str) -> str:
    """Extract ML entities and relations from text and add them to the knowledge graph."""
    from langchain_core.documents import Document
    builder = KnowledgeGraphBuilder()
    builder.add_documents([Document(page_content=text)])
    g = builder.graph
    return f"Knowledge graph now has {g.number_of_nodes()} nodes and {g.number_of_edges()} edges."


@tool
def query_knowledge_graph(query: str, ml_task: str = "general") -> str:
    """Get knowledge-graph context relevant to a query and ML task."""
    querier = GraphQuerier()
    return querier.get_context(query=query, ml_task=ml_task) or "No relevant graph context found."
