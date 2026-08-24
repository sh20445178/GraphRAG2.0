"""All LangGraph node implementations for the GraphRAG 2.0 pipeline."""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger

from src.graph.state import RAGState
from src.rag.retriever import HybridRetriever
from src.rag.reranker import Reranker
from src.knowledge_graph.query import GraphQuerier
from src.ml.model_advisor import MLModelAdvisor
from src.utils.config import get_settings

# ── Shared LLM instance ────────────────────────────────────────────────────

def _get_llm(temperature: float = 0.0) -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        model=s.llm_model,
        temperature=temperature,
        openai_api_key=s.openai_api_key,
    )


# ═══════════════════════════════════════════════════════════════════════════
# NODE: Query Analyser
# ═══════════════════════════════════════════════════════════════════════════

_QUERY_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a query analyser for a knowledge base that covers Machine Learning topics
and customer feedback analysis.
Given a user query, produce a JSON object with these keys:
  - refined_query  : a clearer, expanded version of the query
  - sub_queries    : list of 2-4 focused sub-questions that together answer the query
  - query_intent   : one of [explain, compare, code, recommend, debug, feedback_trends, general]
                     Use "feedback_trends" when the query is about summarising, analysing, or
                     identifying trends/patterns in customer feedback, reviews, or user opinions.
  - ml_task        : the ML task domain (classification, regression, nlp, cv,
                     reinforcement_learning, time_series, generative, general)

Respond ONLY with valid JSON."""),
    ("human", "{query}"),
])


def node_query_analyser(state: RAGState) -> dict[str, Any]:
    logger.info("Node: query_analyser")
    llm = _get_llm()
    chain = _QUERY_ANALYSIS_PROMPT | llm
    result = chain.invoke({"query": state["query"]})
    try:
        parsed: dict[str, Any] = json.loads(result.content)
    except json.JSONDecodeError:
        parsed = {
            "refined_query": state["query"],
            "sub_queries": [state["query"]],
            "query_intent": "general",
            "ml_task": "general",
        }
    return {
        "refined_query": parsed.get("refined_query", state["query"]),
        "sub_queries": parsed.get("sub_queries", [state["query"]]),
        "query_intent": parsed.get("query_intent", "general"),
        "ml_task": parsed.get("ml_task", "general"),
        "messages": [HumanMessage(content=state["query"])],
        "iteration": 0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# NODE: Hybrid Retriever
# ═══════════════════════════════════════════════════════════════════════════

def node_retriever(state: RAGState) -> dict[str, Any]:
    logger.info("Node: retriever")
    retriever = HybridRetriever()
    queries = [state["refined_query"]] + state.get("sub_queries", [])
    docs = retriever.retrieve(queries)
    return {"retrieved_docs": docs}


# ═══════════════════════════════════════════════════════════════════════════
# NODE: Graph Context Enricher
# ═══════════════════════════════════════════════════════════════════════════

def node_graph_enricher(state: RAGState) -> dict[str, Any]:
    logger.info("Node: graph_enricher")
    querier = GraphQuerier()
    context = querier.get_context(
        query=state["refined_query"],
        ml_task=state.get("ml_task", "general"),
    )
    return {"graph_context": context}


# ═══════════════════════════════════════════════════════════════════════════
# NODE: Reranker
# ═══════════════════════════════════════════════════════════════════════════

def node_reranker(state: RAGState) -> dict[str, Any]:
    logger.info("Node: reranker")
    reranker = Reranker()
    reranked = reranker.rerank(
        query=state["refined_query"],
        documents=state["retrieved_docs"],
    )
    return {"reranked_docs": reranked}


# ═══════════════════════════════════════════════════════════════════════════
# NODE: ML Model Advisor
# ═══════════════════════════════════════════════════════════════════════════

def node_ml_advisor(state: RAGState) -> dict[str, Any]:
    logger.info("Node: ml_advisor")
    advisor = MLModelAdvisor()
    recommendations = advisor.recommend(
        query=state["refined_query"],
        ml_task=state.get("ml_task", "general"),
        context_docs=state.get("reranked_docs", []),
    )
    return {"recommended_models": recommendations}


# ═══════════════════════════════════════════════════════════════════════════
# NODE: Answer Generator
# ═══════════════════════════════════════════════════════════════════════════

_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert ML engineer and researcher.
Use the retrieved context and knowledge-graph information to answer the user's question.

Guidelines:
- Be precise and technically accurate.
- For code requests, provide complete, runnable Python snippets.
- For model recommendations, explain WHY a model fits the task.
- Cite sources by mentioning document titles when available.
- If recommending models, format them as a numbered list with: name, use-case fit, pros/cons.

Knowledge Graph Context:
{graph_context}

Retrieved Documents:
{context}

ML Task: {ml_task}
Query Intent: {query_intent}"""),
    ("human", "{query}"),
])


def _format_docs(docs: list) -> str:
    if not docs:
        return "No relevant documents found."
    parts = []
    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get("title", f"Document {i}")
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{i}] **{title}** ({source})\n{doc.page_content}")
    return "\n\n".join(parts)


def node_generator(state: RAGState) -> dict[str, Any]:
    logger.info("Node: generator")
    llm = _get_llm(temperature=0.2)
    chain = _GENERATION_PROMPT | llm

    docs = state.get("reranked_docs") or state.get("retrieved_docs", [])
    result = chain.invoke({
        "query": state["refined_query"],
        "context": _format_docs(docs),
        "graph_context": state.get("graph_context", ""),
        "ml_task": state.get("ml_task", "general"),
        "query_intent": state.get("query_intent", "general"),
    })
    answer = result.content
    code_snippets = re.findall(r"```(?:python)?\n(.*?)```", answer, re.DOTALL)
    return {
        "answer": answer,
        "code_snippets": code_snippets,
        "messages": [AIMessage(content=answer)],
    }


# ═══════════════════════════════════════════════════════════════════════════
# NODE: Self-Reflection
# ═══════════════════════════════════════════════════════════════════════════

_REFLECTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a quality assessor for ML answers.
Score the answer on the following criteria (each 0–1):
  1. Factual accuracy
  2. Completeness – does it fully address the query?
  3. Code quality (if applicable)
  4. Clarity

Return JSON: {{"score": <average 0-1>, "feedback": "<brief improvement suggestion>"}}
If score >= 0.85 feedback should be "OK".
Respond ONLY with valid JSON."""),
    ("human", "Query: {query}\n\nAnswer:\n{answer}"),
])


def node_reflection(state: RAGState) -> dict[str, Any]:
    logger.info("Node: reflection (iteration %d)", state.get("iteration", 0))
    llm = _get_llm()
    chain = _REFLECTION_PROMPT | llm
    result = chain.invoke({
        "query": state["query"],
        "answer": state.get("answer", ""),
    })
    try:
        parsed = json.loads(result.content)
        score = float(parsed.get("score", 0.5))
        feedback = parsed.get("feedback", "")
    except (json.JSONDecodeError, ValueError):
        score = 0.5
        feedback = "Could not parse reflection."

    return {
        "reflection_score": score,
        "reflection_feedback": feedback,
        "iteration": state.get("iteration", 0) + 1,
    }


# ═══════════════════════════════════════════════════════════════════════════
# NODE: Regenerator (uses reflection feedback)
# ═══════════════════════════════════════════════════════════════════════════

_REGEN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are refining an ML answer based on quality feedback.

Original answer:
{answer}

Critic feedback:
{feedback}

Context:
{context}

Improve the answer addressing ALL feedback points. Maintain technical accuracy."""),
    ("human", "{query}"),
])

_FEEDBACK_REGEN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are refining a customer feedback trend report based on quality feedback.

Original report:
{answer}

Critic feedback:
{feedback}

Feedback excerpts context:
{context}

Revise the report addressing ALL critic feedback. Keep the Markdown section structure:
## Executive Summary / ## Sentiment Overview / ## Key Themes /
## Top Issues & Praise / ## Emerging Trends / ## Actionable Recommendations"""),
    ("human", "{query}"),
])


def node_regenerator(state: RAGState) -> dict[str, Any]:
    logger.info("Node: regenerator")
    llm = _get_llm(temperature=0.3)
    docs = state.get("reranked_docs") or state.get("retrieved_docs", [])

    if state.get("query_intent") == "feedback_trends":
        chain = _FEEDBACK_REGEN_PROMPT | llm
        context = "\n\n".join(
            f"- [{d.metadata.get('date', '')}] {d.page_content[:300]}" for d in docs[:10]
        )
    else:
        chain = _REGEN_PROMPT | llm
        context = _format_docs(docs)

    result = chain.invoke({
        "query": state["refined_query"],
        "answer": state.get("answer", ""),
        "feedback": state.get("reflection_feedback", ""),
        "context": context,
    })
    answer = result.content
    code_snippets = re.findall(r"```(?:python)?\n(.*?)```", answer, re.DOTALL)
    return {
        "answer": answer,
        "code_snippets": code_snippets,
        "messages": [AIMessage(content=answer)],
    }


# ═══════════════════════════════════════════════════════════════════════════
# NODE: Feedback Theme Analyser
# ═══════════════════════════════════════════════════════════════════════════

_FEEDBACK_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a customer feedback analyst.
Analyze the provided customer feedback documents and extract structured insights.

Return a JSON object with exactly these keys:
{{
  "themes": [
    {{
      "theme": "<theme name>",
      "count": <approximate number of mentions>,
      "sentiment": "positive|neutral|negative",
      "examples": ["<short verbatim quote>"]
    }}
  ],
  "sentiment_distribution": {{
    "positive": <fraction 0.0-1.0>,
    "neutral": <fraction 0.0-1.0>,
    "negative": <fraction 0.0-1.0>
  }},
  "top_complaints": ["<issue 1>", "<issue 2>"],
  "top_praises": ["<praise 1>", "<praise 2>"],
  "emerging_issues": ["<issue not yet widespread but appearing>"]
}}

The three sentiment fractions must sum to 1.0.
Respond ONLY with valid JSON."""),
    ("human", "Customer feedback documents:\n\n{documents}"),
])


def _format_feedback_docs(docs: list) -> str:
    if not docs:
        return "No feedback documents available."
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        header_parts = [f"[{i}]"]
        if meta.get("date"):
            header_parts.append(f"Date: {meta['date']}")
        if meta.get("category"):
            header_parts.append(f"Category: {meta['category']}")
        if meta.get("rating"):
            header_parts.append(f"Rating: {meta['rating']}")
        if meta.get("product"):
            header_parts.append(f"Product: {meta['product']}")
        parts.append(" | ".join(header_parts) + "\n" + doc.page_content)
    return "\n\n---\n\n".join(parts)


def node_feedback_analyser(state: RAGState) -> dict[str, Any]:
    logger.info("Node: feedback_analyser")
    llm = _get_llm(temperature=0.0)
    chain = _FEEDBACK_ANALYSIS_PROMPT | llm

    docs = state.get("reranked_docs") or state.get("retrieved_docs", [])
    result = chain.invoke({"documents": _format_feedback_docs(docs[:25])})

    try:
        parsed: dict[str, Any] = json.loads(result.content)
    except json.JSONDecodeError:
        parsed = {
            "themes": [],
            "sentiment_distribution": {"positive": 0.33, "neutral": 0.33, "negative": 0.34},
            "top_complaints": [],
            "top_praises": [],
            "emerging_issues": [],
        }

    return {
        "feedback_themes": parsed.get("themes", []),
        "sentiment_distribution": parsed.get("sentiment_distribution", {}),
        # Append structured findings to graph_context for trend_summarizer
        "graph_context": (
            state.get("graph_context", "") + "\n\n"
            + f"Top complaints: {parsed.get('top_complaints', [])}\n"
            + f"Top praises: {parsed.get('top_praises', [])}\n"
            + f"Emerging issues: {parsed.get('emerging_issues', [])}"
        ).strip(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# NODE: Trend Summarizer
# ═══════════════════════════════════════════════════════════════════════════

_TREND_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a customer-insights analyst producing an executive trend report.
Use the structured analysis and raw feedback excerpts to write a comprehensive Markdown report.

Required sections (use these exact headers):
## Executive Summary
## Sentiment Overview
## Key Themes
## Top Issues & Praise
## Emerging Trends
## Actionable Recommendations

Guidelines:
- Quote specific customer language where possible.
- Be data-driven: reference percentages / counts from the theme analysis.
- Recommendations must be concrete and prioritised.
- Keep the report skimmable with bullet points."""),
    ("human", """User query: {query}

--- Extracted Themes (JSON) ---
{themes}

--- Sentiment Distribution ---
{sentiment}

--- Additional Context ---
{graph_context}

--- Representative Feedback Excerpts ---
{context}"""),
])


def node_trend_summarizer(state: RAGState) -> dict[str, Any]:
    logger.info("Node: trend_summarizer")
    llm = _get_llm(temperature=0.1)
    chain = _TREND_SUMMARY_PROMPT | llm

    docs = state.get("reranked_docs") or state.get("retrieved_docs", [])
    context_excerpts = "\n\n".join(
        f"- [{d.metadata.get('date', '')}] {d.page_content[:300]}" for d in docs[:10]
    )

    result = chain.invoke({
        "query": state.get("query", "Summarize recent customer feedback trends"),
        "themes": json.dumps(state.get("feedback_themes", []), indent=2),
        "sentiment": json.dumps(state.get("sentiment_distribution", {}), indent=2),
        "graph_context": state.get("graph_context", ""),
        "context": context_excerpts,
    })

    summary = result.content
    return {
        "trend_summary": summary,
        "answer": summary,
        "code_snippets": [],
        "messages": [AIMessage(content=summary)],
        "iteration": state.get("iteration", 0),
    }
