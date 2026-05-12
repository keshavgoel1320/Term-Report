"""CLI entry point for the CWM retrieval study."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from cwm_research.schemas import StudyPaths

app = typer.Typer(
    name="cwm-study",
    help="Construction Waste Management Retrieval Study",
    add_completion=False,
)
console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _setup_logging(paths: StudyPaths) -> None:
    """Configure file and console logging."""
    paths.log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.FileHandler(paths.log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    # Quiet noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


def _get_paths() -> StudyPaths:
    return StudyPaths.from_root(PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def validate_db() -> None:
    """Validate the product database for diversity and quality."""
    paths = _get_paths()
    _setup_logging(paths)

    console.print(Panel("[bold]Validating Product Database[/bold]", style="cyan"))

    from cwm_research.database import load_products
    products = load_products(paths.products_path)

    # Print summary
    from collections import Counter
    categories = Counter(p.category for p in products)
    subcategories = Counter(p.subcategory for p in products)
    conditions = Counter(p.condition for p in products)

    console.print(f"\n[green][OK] {len(products)} products loaded and validated[/green]")
    console.print(f"  Categories: {len(categories)}")
    console.print(f"  Subcategories: {len(subcategories)}")
    console.print(f"  Conditions: {dict(conditions)}")
    console.print(f"  Products/category: min={min(categories.values())}, max={max(categories.values())}")

    # Sample a few
    console.print("\n[bold]Sample products:[/bold]")
    for p in products[:3]:
        console.print(f"  [dim]- {p.name} ({p.category}) - {p.materials[0]}, {p.condition}[/dim]")


@app.command()
def run(
    skip_cache: bool = typer.Option(False, "--skip-cache", help="Ignore cached indexes and regenerate"),
) -> None:
    """Run the full study: enrich > index > query > evaluate > report."""
    paths = _get_paths()
    _setup_logging(paths)
    logger = logging.getLogger("cwm_research.cli")

    start_time = time.time()

    console.print(Panel(
        "[bold]CWM Retrieval Study - Full Run[/bold]\n"
        "3 pipelines x 50 queries x 1000 products",
        style="cyan",
    ))

    # ----- Step 1: Load products -----
    console.print("\n[bold cyan]Step 1/7: Loading product database...[/bold cyan]")
    from cwm_research.database import load_products
    products = load_products(paths.products_path)
    console.print(f"  [green][OK] {len(products)} products loaded[/green]")

    # ----- Step 2: Initialise LLM client -----
    console.print("\n[bold cyan]Step 2/7: Initialising Vertex AI...[/bold cyan]")
    from cwm_research.vertex_llm import VertexLLMClient
    llm_client = VertexLLMClient(cache_dir=paths.cache_dir / "llm")
    models = llm_client.discover_models()
    console.print(f"  [green][OK] Generation: {models.generation_model}[/green]")
    console.print(f"  [green][OK] Judge: {models.judge_model}[/green]")

    # ----- Step 3: Enrich products -----
    console.print("\n[bold cyan]Step 3/7: Enriching products with vague user queries...[/bold cyan]")
    from cwm_research.enrichment import enrich_products
    enriched = enrich_products(
        products, llm_client, cache_dir=paths.enrichment_cache_dir,
    )
    successful = sum(1 for e in enriched if e.generated_queries)
    console.print(f"  [green][OK] {successful}/{len(enriched)} products enriched[/green]")

    # ----- Step 4: Build indexes -----
    console.print("\n[bold cyan]Step 4/7: Building search indexes...[/bold cyan]")
    from cwm_research.embeddings import EmbeddingBackend
    from cwm_research.indexing import build_raw_index, build_enriched_index

    embedding_backend = EmbeddingBackend()
    raw_cache = None if skip_cache else paths.index_dir
    enriched_cache = None if skip_cache else paths.enriched_index_dir

    raw_index = build_raw_index(products, embedding_backend, cache_dir=raw_cache)
    enriched_index = build_enriched_index(enriched, products, embedding_backend, cache_dir=enriched_cache)
    console.print(f"  [green][OK] Raw index: {raw_index.size} docs[/green]")
    console.print(f"  [green][OK] Enriched index: {enriched_index.size} docs[/green]")

    # ----- Step 5: Generate test queries -----
    console.print("\n[bold cyan]Step 5/7: Generating test queries...[/bold cyan]")
    from cwm_research.queries import generate_test_queries, load_test_queries

    if paths.queries_path.exists() and not skip_cache:
        queries = load_test_queries(paths.queries_path)
        console.print(f"  [green][OK] Loaded {len(queries)} cached queries[/green]")
    else:
        queries = generate_test_queries(llm_client, output_path=paths.queries_path)

    # ----- Step 6: Run all pipelines -----
    console.print("\n[bold cyan]Step 6/7: Running retrieval pipelines...[/bold cyan]")
    from cwm_research.pipelines import run_pipeline_basic, run_pipeline_enriched, run_pipeline_rewrite

    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

    pipeline_runs: dict[str, list[tuple]] = {
        "basic_hybrid": [],
        "enriched_hybrid": [],
        "rewrite_enriched": [],
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Running pipelines...", total=len(queries) * 3)

        for query in queries:
            # Pipeline 1: Basic
            results, eff_q, meta = run_pipeline_basic(query.text, raw_index, embedding_backend)
            pipeline_runs["basic_hybrid"].append((query, results, eff_q, meta))
            progress.update(task, advance=1)

            # Pipeline 2: Enriched
            results, eff_q, meta = run_pipeline_enriched(query.text, enriched_index, embedding_backend)
            pipeline_runs["enriched_hybrid"].append((query, results, eff_q, meta))
            progress.update(task, advance=1)

            # Pipeline 3: Rewrite + Enriched
            results, eff_q, meta = run_pipeline_rewrite(query.text, enriched_index, embedding_backend, llm_client)
            pipeline_runs["rewrite_enriched"].append((query, results, eff_q, meta))
            progress.update(task, advance=1)

    for name, runs in pipeline_runs.items():
        console.print(f"  [green][OK] {name}: {len(runs)} queries processed[/green]")

    # ----- Step 7: Evaluate and report -----
    console.print("\n[bold cyan]Step 7/7: Evaluating and generating reports...[/bold cyan]")
    from cwm_research.evaluation import evaluate_all_queries
    from cwm_research.reporting import (
        aggregate_metrics,
        write_results_json,
        write_summary_markdown,
        write_methodology_report,
    )

    # Use judge model for evaluation
    judge_client = VertexLLMClient(cache_dir=paths.cache_dir / "judge")
    judge_client._selected_models = models  # reuse discovered models

    evaluated = evaluate_all_queries(queries, pipeline_runs, judge_client)

    write_results_json(paths, evaluated, models, embedding_backend.model_name, len(products), len(queries))
    write_summary_markdown(paths, evaluated, models, embedding_backend.model_name)
    write_methodology_report(paths, evaluated, models, embedding_backend.model_name, len(products), len(queries))

    # ----- Done -----
    elapsed = time.time() - start_time
    console.print(f"\n[bold green]{'='*60}[/bold green]")
    console.print(f"[bold green]Study complete in {elapsed:.1f}s[/bold green]")
    console.print(f"[bold green]{'='*60}[/bold green]")

    # Print final summary
    console.print("\n[bold]Pipeline Comparison:[/bold]")
    console.print(f"  {'Pipeline':<22} {'NDCG@5':>8} {'NDCG@10':>8} {'MRR':>8} {'Judge':>8}")
    console.print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for name, results in evaluated.items():
        m = aggregate_metrics(results)
        console.print(
            f"  {name:<22} {m['mean_ndcg_at_5']:>8.4f} {m['mean_ndcg_at_10']:>8.4f} "
            f"{m['mean_mrr']:>8.4f} {m['mean_judge_score']:>8.4f}"
        )

    console.print(f"\n[dim]Results: {paths.results_path}[/dim]")
    console.print(f"[dim]Summary: {paths.summary_path}[/dim]")
    console.print(f"[dim]Methodology: {paths.methodology_path}[/dim]")
    console.print(f"[dim]Log: {paths.log_path}[/dim]")


if __name__ == "__main__":
    app()
