# GraphRAG 2.0 – LangGraph-Powered ML Knowledge Assistant

A production-ready **RAG 2.0** system built with **LangGraph** and **LangChain**, designed specifically for Machine Learning model development workflows.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────┐
│  Query Analyser │  ← refines query, detects ML task & intent
└────────┬────────┘
         │
    ┌────┴─────┐  (parallel)
    ▼          ▼
┌──────────┐  ┌──────────────────┐
│ Hybrid   │  │ Knowledge Graph  │
│ Retriever│  │ Enricher         │
│(Vec+BM25)│  │(NetworkX + seed) │
└────┬─────┘  └────────┬─────────┘
     └────────┬─────────┘
              ▼
       ┌─────────────┐
       │  Reranker   │  ← Cohere or cross-encoder
       └──────┬──────┘
              │
    ┌─────────┴──────────┐
    ▼                    ▼
┌──────────┐     ┌──────────────┐
│ Generator│     │  ML Advisor  │  ← only for "recommend" intent
└────┬─────┘     └──────┬───────┘
     └──────────┬────────┘
                ▼
         ┌────────────┐
         │ Reflection │  ← self-critique with score
         └──────┬─────┘
    ┌───────────┴───────────┐
    ▼                       ▼
  END (score ≥ 0.85)   Regenerator → Reflection (max 2×)
```

### RAG 2.0 Features

| Feature | Implementation |
|---|---|
| **Hybrid Retrieval** | Dense (Chroma) + Sparse (BM25) fused via Reciprocal Rank Fusion |
| **Knowledge Graph** | NetworkX DiGraph with curated ML taxonomy + LLM triple extraction |
| **Reranking** | Cohere API with cross-encoder local fallback |
| **Multi-hop Queries** | Sub-query decomposition via LLM query analyser |
| **Self-Reflection** | LLM self-critique loop, max 2 regeneration cycles |
| **Intent Routing** | Conditional graph edges based on detected query intent |
| **ML Advisor** | Structured model recommendations with starter code snippets |

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY (minimum required)
```

> **Local / free alternative**: set `EMBEDDING_MODEL` to use HuggingFace BGE embeddings by
> calling `get_embeddings("huggingface")` in `src/utils/embeddings.py`.

### 3. Seed the knowledge graph

```bash
python main.py seed-graph
```

### 4. Ask a question

```bash
python main.py ask "What is the best model for time-series anomaly detection?"
```

### 5. Interactive chat

```bash
python main.py chat
```

---

## CLI Commands

| Command | Description |
|---|---|
| `python main.py ask "<query>"` | Single query through the full pipeline |
| `python main.py ask "<query>" --verbose` | Show pipeline metadata |
| `python main.py seed-graph` | Populate the KG with ML taxonomy |
| `python main.py index --file paper.pdf` | Index a local file |
| `python main.py index --arxiv 2005.11401` | Index an arXiv paper |
| `python main.py index --dir ./papers/` | Recursively index a directory |
| `python main.py chat` | Multi-turn interactive chat |

---

## Project Structure

```
GraphRAG2.0/
├── main.py                     # CLI entry point (Typer)
├── requirements.txt
├── .env.example
├── pyproject.toml
├── src/
│   ├── graph/
│   │   ├── state.py            # LangGraph TypedDict state
│   │   ├── nodes.py            # All graph nodes
│   │   ├── edges.py            # Conditional routing logic
│   │   └── workflow.py         # Graph assembly & compilation
│   ├── rag/
│   │   ├── chunker.py          # Sentence-aware text splitter
│   │   ├── indexer.py          # Document loading & ChromaDB upsert
│   │   ├── retriever.py        # Hybrid retriever (RRF fusion)
│   │   └── reranker.py         # Cohere / cross-encoder reranker
│   ├── knowledge_graph/
│   │   ├── entities.py         # Entity & relation dataclasses
│   │   ├── builder.py          # LLM triple extraction + seed graph
│   │   └── query.py            # Subgraph querier & context serialiser
│   ├── ml/
│   │   ├── model_advisor.py    # Structured ML model recommendations
│   │   ├── code_generator.py   # Training pipeline / eval code gen
│   │   └── evaluator.py        # Metric selection & improvement advice
│   ├── tools/
│   │   ├── search_tools.py     # LangChain @tool wrappers for RAG/KG
│   │   └── ml_tools.py         # LangChain @tool wrappers for ML ops
│   └── utils/
│       ├── config.py           # Pydantic-settings config
│       └── embeddings.py       # OpenAI / HuggingFace embedding factory
├── examples/
│   ├── basic_rag.py            # Simple RAG query demo
│   ├── graph_rag.py            # arXiv ingestion + graph-enriched query
│   └── ml_model_advisor.py     # Full advisor + code gen + eval flow
├── tests/
│   ├── test_retriever.py       # RRF fusion unit tests
│   ├── test_graph.py           # KG builder/querier tests
│   └── test_workflow.py        # Graph routing unit tests
└── data/
    ├── raw/                    # Drop raw documents here
    ├── processed/
    └── vector_store/           # ChromaDB persisted here
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Extending the System

### Add a new LangGraph node

1. Implement `node_my_feature(state: RAGState) -> dict` in `src/graph/nodes.py`
2. Register it in `src/graph/workflow.py` with `builder.add_node(...)`
3. Wire edges as needed

### Add a new document source

Add a new loader in `src/rag/indexer.py` following the existing `_load_file` pattern.

### Use as a LangChain tool in your own agent

```python
from src.tools.search_tools import search_knowledge_base, query_knowledge_graph
from src.tools.ml_tools import recommend_ml_models, generate_ml_code

tools = [search_knowledge_base, query_knowledge_graph, recommend_ml_models, generate_ml_code]
```

---

## License

MIT
