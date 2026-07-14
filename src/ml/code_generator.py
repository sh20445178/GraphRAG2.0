"""
ML Code Generator
=================
Generates production-quality ML code: training loops, pipelines,
evaluation scripts, and hyperparameter configs.
"""
from __future__ import annotations

from enum import Enum

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.utils.config import get_settings


class CodeTarget(str, Enum):
    TRAINING_PIPELINE = "training_pipeline"
    EVALUATION = "evaluation"
    INFERENCE = "inference"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    DATA_PREPROCESSING = "data_preprocessing"
    FULL_PROJECT = "full_project"


_CODE_GEN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert ML engineer.
Generate clean, well-structured Python code for the requested target.

Requirements:
- Use modern Python (3.10+) with type hints
- Follow PEP 8
- Include brief inline comments for non-obvious logic
- Use the specified framework
- Code must be directly runnable (not pseudocode)
- For training pipelines, include: data loading, preprocessing, model init,
  training loop/fit, validation, model saving

Target: {target}
Framework: {framework}
ML Task: {ml_task}
Model: {model_name}
"""),
    ("human", "{description}"),
])


class MLCodeGenerator:
    """Generates ML code snippets for different pipeline stages."""

    def __init__(self) -> None:
        self._settings = get_settings()

    def generate(
        self,
        description: str,
        target: CodeTarget = CodeTarget.TRAINING_PIPELINE,
        framework: str = "sklearn",
        ml_task: str = "classification",
        model_name: str = "auto",
    ) -> str:
        llm = ChatOpenAI(
            model=self._settings.llm_model,
            temperature=0.1,
            openai_api_key=self._settings.openai_api_key,
        )
        chain = _CODE_GEN_PROMPT | llm
        result = chain.invoke({
            "description": description,
            "target": target.value,
            "framework": framework,
            "ml_task": ml_task,
            "model_name": model_name,
        })
        return result.content
