"""Generate results JSON, summary markdown, and methodology report."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from cwm_research.schemas import QueryPipelineResult, StudyPaths
from cwm_research.vertex_llm import SelectedModels

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def aggregate_metrics(results: list[QueryPipelineResult]) -> dict[str, float]:
    """Compute aggregate metrics for a pipeline's results."""
    if not results:
        return {"mean_ndcg_at_5": 0, "mean_ndcg_at_10": 0, "mean_mrr": 0, "mean_judge_score": 0}
    return {
        "mean_ndcg_at_5": round(mean(r.ndcg_at_5 for r in results), 4),
        "mean_ndcg_at_10": round(mean(r.ndcg_at_10 for r in results), 4),
        "mean_mrr": round(mean(r.mrr for r in results), 4),
        "mean_judge_score": round(mean(r.mean_judge_score for r in results), 4),
    }


# ---------------------------------------------------------------------------
# Results JSON
# ---------------------------------------------------------------------------

def write_results_json(
    paths: StudyPaths,
    pipeline_results: dict[str, list[QueryPipelineResult]],
    models: SelectedModels,
    embedding_model: str,
    product_count: int,
    query_count: int,
) -> None:
    """Write the full results to results.json."""
    payload: dict[str, Any] = {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "models": {
                "generation": models.generation_model,
                "judge": models.judge_model,
                "embedding": embedding_model,
            },
            "product_count": product_count,
            "query_count": query_count,
        },
        "pipelines": {},
    }

    for pipeline_name, results in pipeline_results.items():
        metrics = aggregate_metrics(results)
        payload["pipelines"][pipeline_name] = {
            "aggregate_metrics": metrics,
            "per_query": [r.model_dump() for r in results],
        }

    paths.results_path.parent.mkdir(parents=True, exist_ok=True)
    paths.results_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Results written to %s", paths.results_path)


# ---------------------------------------------------------------------------
# Summary markdown
# ---------------------------------------------------------------------------

def write_summary_markdown(
    paths: StudyPaths,
    pipeline_results: dict[str, list[QueryPipelineResult]],
    models: SelectedModels,
    embedding_model: str,
) -> None:
    """Write the results summary comparison table."""
    lines = [
        "# CWM Retrieval Study — Results Summary",
        "",
        "## Pipeline Comparison",
        "",
        "| Pipeline | NDCG@5 | NDCG@10 | MRR | Judge Mean |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for pipeline_name, results in pipeline_results.items():
        m = aggregate_metrics(results)
        lines.append(
            f"| {pipeline_name} | {m['mean_ndcg_at_5']:.4f} | "
            f"{m['mean_ndcg_at_10']:.4f} | {m['mean_mrr']:.4f} | "
            f"{m['mean_judge_score']:.4f} |"
        )

    lines.extend([
        "",
        "## Models Used",
        "",
        f"- **Generation model**: `{models.generation_model}`",
        f"- **Judge model**: `{models.judge_model}`",
        f"- **Embedding model**: `{embedding_model}`",
        "",
        "## Pipeline Definitions",
        "",
        "- **basic_hybrid**: User's vague query → raw BM25+FAISS index (baseline)",
        "- **enriched_hybrid**: User's vague query → enriched index (products augmented with LLM-generated user-like queries)",
        "- **rewrite_enriched**: LLM rewrites user's vague query → enriched index (query-side + document-side enrichment)",
        "",
        "---",
        "",
        "## Per-Query Results",
        "",
        "| Query | Text | Basic NDCG@10 | Enriched NDCG@10 | Rewrite NDCG@10 | Basic Judge | Enriched Judge | Rewrite Judge |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])

    pipeline_names = list(pipeline_results.keys())
    if len(pipeline_names) == 3:
        n_queries = len(pipeline_results[pipeline_names[0]])
        for i in range(n_queries):
            q = pipeline_results[pipeline_names[0]][i]
            vals = []
            for pn in pipeline_names:
                r = pipeline_results[pn][i]
                vals.extend([f"{r.ndcg_at_10:.4f}", f"{r.mean_judge_score:.4f}"])
            text_short = q.query_text[:60] + "..." if len(q.query_text) > 60 else q.query_text
            lines.append(
                f"| {q.query_id} | {text_short} | {vals[0]} | {vals[2]} | {vals[4]} | {vals[1]} | {vals[3]} | {vals[5]} |"
            )

    paths.summary_path.parent.mkdir(parents=True, exist_ok=True)
    paths.summary_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Summary written to %s", paths.summary_path)


# ---------------------------------------------------------------------------
# Methodology report
# ---------------------------------------------------------------------------

def write_methodology_report(
    paths: StudyPaths,
    pipeline_results: dict[str, list[QueryPipelineResult]],
    models: SelectedModels,
    embedding_model: str,
    product_count: int,
    query_count: int,
) -> None:
    """Write the detailed methodology document."""
    # Determine winner
    pipeline_rankings = sorted(
        (
            (name, aggregate_metrics(results)["mean_ndcg_at_10"], aggregate_metrics(results)["mean_judge_score"])
            for name, results in pipeline_results.items()
        ),
        key=lambda x: (x[1], x[2]),
        reverse=True,
    )
    best_pipeline = pipeline_rankings[0][0] if pipeline_rankings else "n/a"

    lines = [
        "# Comparative Retrieval Methodology Report",
        "",
        "## 1. Objective",
        "",
        "This study compares three retrieval strategies for a construction waste management marketplace. "
        "The core question: when users search with vague, incomplete language, does document-side enrichment "
        "or query-side rewriting (or both) improve retrieval quality over a basic hybrid baseline?",
        "",
        "### Why this matters",
        "",
        "Real users of a construction waste marketplace are typically site buyers, contractors, or renovators "
        "who describe their needs informally ('something strong for the floor', 'cheap recycled option for walls'). "
        "They rarely use the technical product names or engineering terminology that populates catalog databases. "
        "This vocabulary gap is the core retrieval challenge.",
        "",
        "## 2. Product Database",
        "",
        f"- **{product_count}** products across **50** categories and **200** subcategories",
        "- Categories span: structural materials, insulation, metals, polymers, ceramics, stone, glass, composites, soil/rubble, and specialty items",
        "- Each product has: name, description, materials, condition (reclaimed/recycled/waste), "
        "dimensions, weight, origin process, reuse applications, compliance tags, and structured attributes",
        "- The database is deterministically generated from a 50-subdomain specification catalog to ensure "
        "perfect diversity: exactly 20 products per category, 5 per subcategory, no duplicates",
        "",
        "## 3. Test Queries",
        "",
        f"- **{query_count}** test queries generated by LLM to simulate real user behavior",
        "- Queries are intentionally vague, incomplete, and use everyday language",
        "- No query directly names a product category or uses catalog terminology",
        "- Queries span: material type, use case, property requirements, cost concerns, and ambiguous descriptions",
        "",
        "## 4. Retrieval Stack",
        "",
        "### Shared components",
        "- **Sparse retrieval**: BM25Okapi over tokenized product documents",
        "- **Dense retrieval**: FAISS inner-product search over L2-normalised embeddings (cosine similarity)",
        "- **Hybrid scoring**: weighted sum of normalised BM25 (0.4) and dense (0.6) scores",
        f"- **Embedding model**: `{embedding_model}`",
        "",
        "### Indexes",
        "- **Raw index**: BM25 + FAISS over original product text (used by Pipeline 1)",
        "- **Enriched index**: BM25 + FAISS over product text augmented with 6 LLM-generated vague user queries per product (used by Pipelines 2 and 3)",
        "",
        "## 5. Pipeline Definitions",
        "",
        "### Pipeline 1: Basic Hybrid Search (`basic_hybrid`)",
        "- The user's vague query is sent directly to the **raw index**",
        "- No LLM processing at query time",
        "- This is the baseline: pure retrieval quality without any artificial enhancement",
        "",
        "### Pipeline 2: Enriched Hybrid Search (`enriched_hybrid`)",
        "- The user's vague query is sent directly to the **enriched index**",
        "- The enrichment is document-side only: each product was augmented with 6 LLM-generated queries "
        "that simulate how a non-expert user might search for that product",
        "- Hypothesis: by embedding informal language alongside technical descriptions, the index bridges "
        "the vocabulary gap between user queries and product catalogs",
        "",
        "### Pipeline 3: LLM Rewrite + Enriched Search (`rewrite_enriched`)",
        "- The user's vague query is first rewritten by an LLM that simulates an internal clarification process",
        "- The LLM asks itself clarifying questions (material type, use case, properties, condition) and produces a richer query",
        "- The rewritten query is then searched against the **enriched index**",
        "- Hypothesis: combining query-side enrichment with document-side enrichment should produce the best results, "
        "as both sides of the vocabulary gap are addressed",
        "",
        "## 6. LLM Stack",
        "",
        f"- **Generation model** (enrichment + rewriting): `{models.generation_model}`",
        f"- **Judge model** (evaluation): `{models.judge_model}`",
        "- Models are selected at runtime by probing Vertex AI for the latest available Gemini models",
        "- All LLM responses are cached to disk for reproducibility",
        "",
        "## 7. Evaluation Methodology",
        "",
        "### Pooled LLM-as-Judge",
        "- For each query, results from all 3 pipelines are pooled and deduplicated",
        "- The LLM judge scores every unique product in the pool on a 1-5 relevance scale",
        "- This ensures fair comparison: all pipelines are judged against the same relevance assessments",
        "- No pre-labeled relevance judgments (qrels) are used — this reflects real marketplace conditions "
        "where ground truth is not available",
        "",
        "### Metrics",
        "- **NDCG@5 and NDCG@10**: Measures ranking quality using graded relevance from judge scores. "
        "Higher = better ranking of relevant items in the top positions.",
        "- **MRR (Mean Reciprocal Rank)**: Measures how quickly the first highly-relevant result (judge score ≥ 3) appears. "
        "Higher = relevant results appear sooner.",
        "- **Mean Judge Score**: Average relevance score across all retrieved items. "
        "Higher = retrieved items are generally more relevant.",
        "",
        "## 8. Results",
        "",
        f"- **Best pipeline by NDCG@10**: `{best_pipeline}`",
        "",
        "### Aggregate Metrics",
        "",
        "| Pipeline | NDCG@5 | NDCG@10 | MRR | Judge Mean |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for name, results in pipeline_results.items():
        m = aggregate_metrics(results)
        lines.append(
            f"| {name} | {m['mean_ndcg_at_5']:.4f} | {m['mean_ndcg_at_10']:.4f} | "
            f"{m['mean_mrr']:.4f} | {m['mean_judge_score']:.4f} |"
        )

    lines.extend([
        "",
        "## 9. Analysis and Inferences",
        "",
    ])

    # Generate analysis based on results
    if pipeline_rankings:
        best_name, best_ndcg, best_judge = pipeline_rankings[0]
        worst_name, worst_ndcg, worst_judge = pipeline_rankings[-1]
        lines.extend([
            f"- `{best_name}` achieved the highest NDCG@10 ({best_ndcg:.4f}), suggesting its approach to "
            "bridging the vocabulary gap is most effective for ranking relevant products.",
            f"- `{worst_name}` had the lowest NDCG@10 ({worst_ndcg:.4f}), which provides insight into "
            "the limitations of its approach.",
            "",
            "### Key observations",
            "",
            "- If **enriched_hybrid** outperforms **basic_hybrid**: Document-side enrichment successfully bridges "
            "the vocabulary gap. Embedding informal language alongside technical descriptions helps match vague queries.",
            "- If **rewrite_enriched** outperforms **enriched_hybrid**: Query-side rewriting adds value beyond "
            "document-side enrichment alone. The LLM's ability to infer user intent and add specificity helps.",
            "- If **basic_hybrid** performs competitively: The raw BM25+semantic hybrid is already robust enough "
            "for vague queries, and LLM-based enrichment may not justify the added complexity.",
            "",
        ])

    lines.extend([
        "## 10. Limitations",
        "",
        "- The product database is synthetic, generated from a specification catalog. Real marketplace data would have more variability.",
        "- LLM-as-judge evaluation, while practical, may have systematic biases (e.g., preferring verbose descriptions, surface-level keyword matching).",
        "- The 50 test queries, while diverse, are still LLM-generated rather than collected from real users.",
        "- Results depend on the specific Gemini model version available at runtime; different model versions may produce different enrichment quality.",
        "- Pooled judging means metrics are relative within the pool — absolute scores should not be compared across separate runs with different judge models.",
        "",
        "## 11. Reproducibility",
        "",
        "- All LLM calls are cached to `.cache/` — rerunning produces identical results as long as the cache is preserved",
        "- Products are loaded from `data/products.json` (deterministically generated, version-controlled)",
        "- Queries are cached after first generation",
        "- Index builds are cached after first construction",
        "- Model selection is logged and recorded in results metadata",
    ])

    paths.methodology_path.parent.mkdir(parents=True, exist_ok=True)
    paths.methodology_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Methodology report written to %s", paths.methodology_path)
