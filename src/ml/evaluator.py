"""
ML Evaluator
============
Generates evaluation strategies, metric explanations,
and compares model results.
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.utils.config import get_settings


_EVAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an ML evaluation expert.
Given a task and model results, provide:
1. The appropriate evaluation metrics and why
2. How to interpret the results
3. Potential failure modes / pitfalls
4. Suggestions to improve performance
5. A Python code snippet to compute all relevant metrics

Be concise but thorough. Format your response in Markdown."""),
    ("human", "ML Task: {ml_task}\nModel: {model_name}\nResults: {results}"),
])


class MLEvaluator:
    def __init__(self) -> None:
        self._settings = get_settings()

    def evaluate(
        self,
        ml_task: str,
        model_name: str,
        results: str,
    ) -> str:
        llm = ChatOpenAI(
            model=self._settings.llm_model,
            temperature=0.1,
            openai_api_key=self._settings.openai_api_key,
        )
        chain = _EVAL_PROMPT | llm
        return chain.invoke({
            "ml_task": ml_task,
            "model_name": model_name,
            "results": results,
        }).content
