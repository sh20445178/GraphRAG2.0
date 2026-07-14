"""ML-specific LangChain tools."""
from __future__ import annotations

from langchain_core.tools import tool

from src.ml.model_advisor import MLModelAdvisor
from src.ml.code_generator import MLCodeGenerator, CodeTarget
from src.ml.evaluator import MLEvaluator


@tool
def recommend_ml_models(query: str, ml_task: str = "general") -> str:
    """Recommend ML models for a given task description."""
    advisor = MLModelAdvisor()
    recs = advisor.recommend(query=query, ml_task=ml_task)
    if not recs:
        return "No recommendations generated."
    lines = []
    for i, r in enumerate(recs, 1):
        lines.append(f"{i}. **{r.get('name')}** ({r.get('category')})")
        lines.append(f"   Why: {r.get('suitability')}")
        lines.append(f"   Complexity: {r.get('complexity')} | Data: {r.get('data_size_req')}")
        lines.append(f"   Frameworks: {', '.join(r.get('frameworks', []))}")
    return "\n".join(lines)


@tool
def generate_ml_code(
    description: str,
    target: str = "training_pipeline",
    framework: str = "sklearn",
    ml_task: str = "classification",
) -> str:
    """Generate ML Python code for a training pipeline, evaluation, inference, etc."""
    generator = MLCodeGenerator()
    try:
        code_target = CodeTarget(target)
    except ValueError:
        code_target = CodeTarget.TRAINING_PIPELINE
    return generator.generate(
        description=description,
        target=code_target,
        framework=framework,
        ml_task=ml_task,
    )


@tool
def evaluate_ml_results(ml_task: str, model_name: str, results: str) -> str:
    """Evaluate ML model results and suggest improvements."""
    evaluator = MLEvaluator()
    return evaluator.evaluate(ml_task=ml_task, model_name=model_name, results=results)
