"""LLM-based document enrichment — generates vague user queries per product."""

from __future__ import annotations

import logging
from pathlib import Path

from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

from cwm_research.database import build_document_text
from cwm_research.schemas import EnrichedProduct, Product
from cwm_research.vertex_llm import VertexLLMClient

logger = logging.getLogger(__name__)

ENRICHMENT_PROMPT_TEMPLATE = """You are helping build a search index for a construction waste management marketplace.

For the product below, generate 6 short, vague queries that a NON-EXPERT user might type when looking for this kind of product. Think like a site buyer, contractor, or someone renovating who does not know the technical name. They describe what they need in everyday language.

RULES:
- Each query must be under 15 words
- Use informal, everyday language
- Focus on what the user NEEDS or wants to DO with the material
- Vary the angle: some about the use case, some about properties, some about cost/availability
- Be natural — if a real user would happen to use the same words as the catalog, that is fine

Product:
Name: {name}
Category: {category}
Materials: {materials}
Condition: {condition}
Reuse applications: {reuse_applications}
Description (abbreviated): {description_short}

Return a JSON array of exactly 6 strings."""


def enrich_products(
    products: list[Product],
    llm_client: VertexLLMClient,
    *,
    cache_dir: Path | None = None,
) -> list[EnrichedProduct]:
    """Generate vague user queries for each product and build enriched documents."""
    enriched: list[EnrichedProduct] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Enriching products with LLM...", total=len(products))

        for idx, product in enumerate(products):
            original_text = build_document_text(product)
            cache_key = f"enrich_{product.name}" if cache_dir else None

            try:
                generated_queries = _generate_queries_for_product(
                    product, llm_client, cache_key=cache_key
                )
            except Exception as exc:
                logger.warning("Enrichment failed for %s: %s", product.name, exc)
                generated_queries = []

            combined = original_text + "\n\nUser might search for:\n" + "\n".join(
                f"- {q}" for q in generated_queries
            )

            enriched.append(EnrichedProduct(
                product_index=idx,
                product_name=product.name,
                original_text=original_text,
                generated_queries=generated_queries,
                combined_text=combined,
            ))
            progress.update(task, advance=1)

    successful = sum(1 for e in enriched if e.generated_queries)
    logger.info("Enrichment complete: %d/%d products enriched successfully", successful, len(products))
    return enriched


def _generate_queries_for_product(
    product: Product,
    llm_client: VertexLLMClient,
    *,
    cache_key: str | None = None,
) -> list[str]:
    """Call the LLM to generate vague user queries for a single product."""
    # Abbreviate description to save tokens
    desc_short = product.description[:300]
    if len(product.description) > 300:
        desc_short += "..."

    prompt = ENRICHMENT_PROMPT_TEMPLATE.format(
        name=product.name,
        category=product.category,
        materials=", ".join(product.materials),
        condition=product.condition,
        reuse_applications=", ".join(product.reuse_applications[:3]),
        description_short=desc_short,
    )

    result = llm_client.generate_json(prompt, cache_key=cache_key, temperature=0.4, max_tokens=1024)

    if isinstance(result, list):
        return [str(q).strip() for q in result if q and str(q).strip()][:8]
    return []
