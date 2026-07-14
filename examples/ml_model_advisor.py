"""
Example 3 – Full ML Model Advisor workflow
===========================================
Demonstrates: model recommendation + code generation + evaluation advice.
Run: python examples/ml_model_advisor.py
"""
from __future__ import annotations

from src.knowledge_graph.builder import KnowledgeGraphBuilder
from src.ml.model_advisor import MLModelAdvisor
from src.ml.code_generator import MLCodeGenerator, CodeTarget
from src.ml.evaluator import MLEvaluator


def main() -> None:
    # Seed graph
    builder = KnowledgeGraphBuilder()
    builder.add_seed_knowledge()

    task_description = (
        "I have 50k rows of customer churn data with mixed numerical and "
        "categorical features. I want to predict churn probability with "
        "high recall (minimize false negatives)."
    )
    ml_task = "classification"

    print("=" * 60)
    print("ML TASK:", task_description)
    print("=" * 60)

    # ── Step 1: Get recommendations ────────────────────────────────────────
    print("\n[1] Fetching model recommendations…\n")
    advisor = MLModelAdvisor()
    recommendations = advisor.recommend(query=task_description, ml_task=ml_task)

    for i, rec in enumerate(recommendations[:3], 1):
        print(f"  {i}. {rec.get('name')} ({rec.get('complexity')} complexity)")
        print(f"     {rec.get('suitability', '')[:120]}")
        print(f"     Frameworks: {', '.join(rec.get('frameworks', []))}\n")

    # ── Step 2: Generate starter code for top recommendation ──────────────
    if recommendations:
        top = recommendations[0]
        model_name = top.get("name", "XGBoost")
        framework = (top.get("frameworks") or ["sklearn"])[0]

        print(f"[2] Generating training pipeline code for {model_name}…\n")
        generator = MLCodeGenerator()
        code = generator.generate(
            description=task_description,
            target=CodeTarget.TRAINING_PIPELINE,
            framework=framework,
            ml_task=ml_task,
            model_name=model_name,
        )
        # Print first 60 lines to keep output readable
        lines = code.splitlines()[:60]
        print("\n".join(lines))
        if len(code.splitlines()) > 60:
            print("... (truncated)")

    # ── Step 3: Evaluation advice ─────────────────────────────────────────
    print("\n[3] Getting evaluation guidance…\n")
    evaluator = MLEvaluator()
    eval_advice = evaluator.evaluate(
        ml_task=ml_task,
        model_name=model_name if recommendations else "XGBoost",
        results="accuracy=0.88, precision=0.82, recall=0.71, AUC-ROC=0.91",
    )
    print(eval_advice[:600])


if __name__ == "__main__":
    main()
