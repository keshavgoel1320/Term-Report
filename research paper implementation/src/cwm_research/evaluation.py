"""Evaluation: pooled LLM judging, NDCG, MRR computation."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any

from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

from cwm_research.schemas import JudgeScore, QueryPipelineResult, SearchResult, TestQuery
from cwm_research.vertex_llm import VertexLLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------

JUDGE_PROMPT_TEMPLATE = """You are an expert judge evaluating search results for a construction waste management marketplace.

A user searched for: "{query}"

Below are retrieved products. For EACH product, rate its relevance to the user's search on a scale of 1-5:
  1 = Completely irrelevant
  2 = Slightly relevant (tangentially related topic)
  3 = Moderately relevant (right general area but not a great fit)
  4 = Highly relevant (good match for the user's need)
  5 = Perfectly relevant (exactly what the user is looking for)

Products to judge:
{products_text}

Return a JSON array where each element has:
- "product_name": the exact product name
- "score": integer 1-5
- "rationale": brief 1-sentence explanation

Return ALL {count} products in the array."""


def judge_results_pooled(
    query: TestQuery,
    pipeline_results: dict[str, list[SearchResult]],
    llm_client: VertexLLMClient,
) -> dict[str, int]:
    """
    Pool results from all pipelines, deduplicate, judge once.
    Returns: {product_name: score} mapping.
    """
    # Pool unique products across all pipelines
    seen: dict[str, SearchResult] = {}
    for pipeline_name, results in pipeline_results.items():
        for r in results:
            if r.name not in seen:
                seen[r.name] = r

    if not seen:
        return {}

    # Build the products text for judging
    products_list = list(seen.values())
    products_text = "\n".join(
        f"{i+1}. **{r.name}** (Category: {r.category})\n   {r.description[:200]}..."
        if len(r.description) > 200
        else f"{i+1}. **{r.name}** (Category: {r.category})\n   {r.description}"
        for i, r in enumerate(products_list)
    )

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        query=query.text,
        products_text=products_text,
        count=len(products_list),
    )

    try:
        judge_response = llm_client.generate_json(
            prompt,
            cache_key=f"judge_{query.query_id}_{len(seen)}",
            temperature=0.1,
            max_tokens=4096,
        )
    except Exception as exc:
        logger.warning("Judge call failed for %s: %s", query.query_id, exc)
        return {}

    # Parse scores
    scores: dict[str, int] = {}
    if isinstance(judge_response, list):
        for item in judge_response:
            if isinstance(item, dict) and "product_name" in item and "score" in item:
                name = str(item["product_name"]).strip()
                score = int(item["score"])
                score = max(1, min(5, score))  # clamp
                scores[name] = score

    return scores


# ---------------------------------------------------------------------------
# Metric computation using judge scores
# ---------------------------------------------------------------------------

def compute_ndcg(scores: list[int], k: int) -> float:
    """Compute NDCG@k from a ranked list of relevance scores."""
    if not scores:
        return 0.0
    scores_at_k = scores[:k]
    dcg = sum((2**s - 1) / math.log2(i + 2) for i, s in enumerate(scores_at_k))
    ideal = sorted(scores, reverse=True)[:k]
    idcg = sum((2**s - 1) / math.log2(i + 2) for i, s in enumerate(ideal))
    return round(dcg / idcg, 6) if idcg > 0 else 0.0


def compute_mrr(scores: list[int], threshold: int = 3) -> float:
    """Compute MRR: reciprocal rank of first result with score >= threshold."""
    for i, s in enumerate(scores):
        if s >= threshold:
            return round(1.0 / (i + 1), 6)
    return 0.0


def compute_metrics_from_judge(
    results: list[SearchResult],
    judge_scores: dict[str, int],
) -> tuple[float, float, float, float, list[JudgeScore]]:
    """
    Compute NDCG@5, NDCG@10, MRR, and mean judge score using
    the pooled judge scores.
    """
    # Map scores to result order
    ordered_scores = [judge_scores.get(r.name, 1) for r in results]

    ndcg_5 = compute_ndcg(ordered_scores, 5)
    ndcg_10 = compute_ndcg(ordered_scores, 10)
    mrr = compute_mrr(ordered_scores)
    mean_score = round(sum(ordered_scores) / len(ordered_scores), 4) if ordered_scores else 0.0

    # Build JudgeScore objects
    judge_items = [
        JudgeScore(product_name=r.name, score=judge_scores.get(r.name, 1), rationale="")
        for r in results
    ]

    return ndcg_5, ndcg_10, mrr, mean_score, judge_items


# ---------------------------------------------------------------------------
# Full evaluation orchestration
# ---------------------------------------------------------------------------

def evaluate_all_queries(
    queries: list[TestQuery],
    pipeline_runs: dict[str, list[tuple[TestQuery, list[SearchResult], str, dict]]],
    llm_client: VertexLLMClient,
) -> dict[str, list[QueryPipelineResult]]:
    """
    Evaluate all pipelines on all queries using pooled judging.

    Args:
        queries: The 50 test queries
        pipeline_runs: {pipeline_name: [(query, results, effective_query, metadata), ...]}
        llm_client: For judge calls

    Returns:
        {pipeline_name: [QueryPipelineResult, ...]}
    """
    pipeline_names = list(pipeline_runs.keys())
    all_results: dict[str, list[QueryPipelineResult]] = {name: [] for name in pipeline_names}

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Judging and computing metrics...", total=len(queries))

        for q_idx, query in enumerate(queries):
            # Gather results for this query across all pipelines
            query_pipeline_results: dict[str, list[SearchResult]] = {}
            query_pipeline_meta: dict[str, tuple[str, dict]] = {}

            for pipeline_name in pipeline_names:
                run = pipeline_runs[pipeline_name][q_idx]
                _, results, effective_query, metadata = run
                query_pipeline_results[pipeline_name] = results
                query_pipeline_meta[pipeline_name] = (effective_query, metadata)

            # Pool and judge all results for this query at once
            judge_scores = judge_results_pooled(query, query_pipeline_results, llm_client)

            # Compute metrics for each pipeline
            for pipeline_name in pipeline_names:
                results = query_pipeline_results[pipeline_name]
                effective_query, metadata = query_pipeline_meta[pipeline_name]

                ndcg_5, ndcg_10, mrr, mean_score, judge_items = compute_metrics_from_judge(
                    results, judge_scores
                )

                all_results[pipeline_name].append(QueryPipelineResult(
                    query_id=query.query_id,
                    query_text=query.text,
                    effective_query=effective_query,
                    pipeline=pipeline_name,
                    results=results,
                    judge_scores=judge_items,
                    ndcg_at_5=ndcg_5,
                    ndcg_at_10=ndcg_10,
                    mrr=mrr,
                    mean_judge_score=mean_score,
                    metadata=metadata,
                ))

            progress.update(task, advance=1)

    # Log aggregate results
    for pipeline_name, results in all_results.items():
        avg_ndcg_10 = sum(r.ndcg_at_10 for r in results) / len(results) if results else 0
        avg_judge = sum(r.mean_judge_score for r in results) / len(results) if results else 0
        logger.info(
            "%s: avg NDCG@10=%.4f, avg Judge=%.4f",
            pipeline_name, avg_ndcg_10, avg_judge,
        )

    return all_results
