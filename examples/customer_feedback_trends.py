"""
Customer Feedback Trends – End-to-end Example
==============================================
Demonstrates the full GraphRAG 2.0 pipeline for summarising customer feedback.

Steps:
  1. Generate synthetic feedback data (no external files needed)
  2. Index the feedback into the Chroma vector store via FeedbackIndexer
  3. Run the feedback-trends pipeline through the compiled LangGraph
  4. Print the structured trend report

Run:
    python examples/customer_feedback_trends.py
"""
from __future__ import annotations

import json
import textwrap
from datetime import date, timedelta
from random import choice, randint, seed as rseed

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

console = Console()


# ── 1. Synthetic feedback data ─────────────────────────────────────────────

def generate_synthetic_feedback(n: int = 60) -> list[dict]:
    """Creates n realistic-looking SaaS product reviews."""
    rseed(42)

    positive_templates = [
        "The new dashboard is incredibly intuitive. Saved our team hours every week.",
        "Onboarding flow improved massively. New users ramp up in minutes now.",
        "The API is well-documented and easy to integrate. Love the SDKs.",
        "Customer support response time is excellent. Issue resolved same day.",
        "Mobile app finally feels polished. Great job on the redesign.",
        "Bulk export feature is a game-changer for our reporting workflow.",
        "SSO integration works flawlessly with our Okta setup.",
        "The new search is blazing fast. Night-and-day compared to before.",
    ]
    neutral_templates = [
        "Works as advertised. Nothing special but gets the job done.",
        "Pricing is fair for what you get. Would be nice to have more flexibility.",
        "Documentation is decent. Could use more code examples.",
        "UI is clean. Some features are buried but manageable.",
        "Onboarding wizard is okay. Took about an hour to set up.",
    ]
    negative_templates = [
        "Billing page is confusing. Charged twice in the same month – still waiting for a refund.",
        "Loading times are unacceptable on large datasets. Often 30+ seconds.",
        "The mobile app crashes whenever I try to export reports.",
        "No dark mode yet. Please add this – it's 2024.",
        "Notification system is broken. I stopped receiving alerts after the last update.",
        "Data import fails silently on CSV files larger than 10 MB.",
        "The API rate limits are way too restrictive for enterprise use.",
        "Search relevance has gotten worse since the April update.",
        "Cannot delete bulk records. Very frustrating for data cleanup.",
    ]

    categories = ["billing", "performance", "mobile", "api", "ui/ux", "notifications", "data", "onboarding"]
    products = ["Analytics Pro", "DataSync", "ReportBuilder", "API Gateway"]
    sentiments = (["positive"] * 40 + ["neutral"] * 20 + ["negative"] * 40)

    records = []
    today = date.today()
    for i in range(n):
        sentiment = choice(sentiments)
        if sentiment == "positive":
            text = choice(positive_templates)
            rating = choice([4, 5])
        elif sentiment == "neutral":
            text = choice(neutral_templates)
            rating = choice([3, 4])
        else:
            text = choice(negative_templates)
            rating = choice([1, 2])

        days_ago = randint(0, 89)
        records.append({
            "text": text,
            "date": str(today - timedelta(days=days_ago)),
            "category": choice(categories),
            "rating": rating,
            "sentiment": sentiment,
            "product": choice(products),
            "customer_id": f"cust_{1000 + i}",
        })
    return records


# ── 2. Index feedback ──────────────────────────────────────────────────────

def index_feedback(records: list[dict]) -> int:
    from src.rag.feedback_indexer import FeedbackIndexer
    indexer = FeedbackIndexer()
    return indexer.index_feedback_list(records, source="synthetic_demo")


# ── 3. Run the pipeline ────────────────────────────────────────────────────

def run_feedback_trends(query: str) -> dict:
    from src.graph.workflow import compile_graph
    graph = compile_graph()
    return graph.invoke({"query": query, "messages": []})


# ── 4. Render results ──────────────────────────────────────────────────────

def print_theme_table(themes: list[dict]) -> None:
    if not themes:
        return
    table = Table(title="Extracted Themes", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Theme", style="bold cyan")
    table.add_column("Sentiment", style="bold")
    table.add_column("Mentions", justify="right")
    table.add_column("Example Quote", max_width=60)

    sentiment_style = {"positive": "green", "neutral": "yellow", "negative": "red"}
    for i, t in enumerate(themes[:10], 1):
        sty = sentiment_style.get(t.get("sentiment", "neutral"), "white")
        examples = t.get("examples", [])
        example = examples[0] if examples else ""
        table.add_row(
            str(i),
            t.get("theme", ""),
            f"[{sty}]{t.get('sentiment', '')}[/{sty}]",
            str(t.get("count", "?")),
            textwrap.shorten(example, width=60, placeholder="…"),
        )
    console.print(table)


def print_sentiment_bar(sentiment: dict) -> None:
    if not sentiment:
        return
    pos = sentiment.get("positive", 0)
    neu = sentiment.get("neutral", 0)
    neg = sentiment.get("negative", 0)
    bar_width = 50
    p_blocks = round(pos * bar_width)
    n_blocks = round(neu * bar_width)
    g_blocks = bar_width - p_blocks - n_blocks

    bar = (
        f"[green]{'█' * p_blocks}[/green]"
        f"[yellow]{'█' * n_blocks}[/yellow]"
        f"[red]{'█' * g_blocks}[/red]"
    )
    console.print(f"\n[bold]Sentiment Distribution:[/bold]  {bar}")
    console.print(
        f"  [green]▲ {pos:.0%} positive[/]  "
        f"[yellow]● {neu:.0%} neutral[/]  "
        f"[red]▼ {neg:.0%} negative[/]\n"
    )


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    console.print(Panel(
        "[bold cyan]GraphRAG 2.0 – Customer Feedback Trend Analysis[/bold cyan]\n"
        "Indexing synthetic SaaS product reviews and generating a trend report.",
        expand=False,
    ))

    # Step 1: Generate & index
    console.print("\n[bold yellow]Step 1:[/] Generating synthetic feedback data…")
    records = generate_synthetic_feedback(n=60)
    console.print(f"  Generated [bold]{len(records)}[/] feedback records.")

    console.print("\n[bold yellow]Step 2:[/] Indexing into vector store…")
    n_indexed = index_feedback(records)
    console.print(f"  Indexed [bold]{n_indexed}[/] chunks.")

    # Step 2: Run multiple queries
    queries = [
        "Summarize recent customer feedback trends over the last 90 days",
        "What are the top complaints and how urgent are they?",
        "Which product areas have the most positive sentiment?",
    ]

    for query in queries:
        console.print(f"\n[bold yellow]Query:[/] {query}")
        result = run_feedback_trends(query)

        intent = result.get("query_intent", "")
        score = result.get("reflection_score", 0.0)
        console.print(f"  [dim]intent={intent}  reflection_score={score:.2f}[/dim]")

        print_theme_table(result.get("feedback_themes") or [])
        print_sentiment_bar(result.get("sentiment_distribution") or {})

        console.print(Panel(
            Markdown(result.get("answer", "No answer generated.")),
            title="[bold green]Trend Report[/bold green]",
            border_style="green",
        ))
        console.rule()


if __name__ == "__main__":
    main()
