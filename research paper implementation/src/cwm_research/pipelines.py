"""Three retrieval pipelines for the comparative study."""

from __future__ import annotations

import logging

from cwm_research.embeddings import EmbeddingBackend
from cwm_research.indexing import HybridIndex
from cwm_research.retrieval import hybrid_search
from cwm_research.schemas import SearchResult
from cwm_research.vertex_llm import VertexLLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline 1: Basic Hybrid Search on Raw Index
# ---------------------------------------------------------------------------

def run_pipeline_basic(
    query_text: str,
    raw_index: HybridIndex,
    embedding_backend: EmbeddingBackend,
    *,
    top_k: int = 10,
) -> tuple[list[SearchResult], str, dict]:
    """
    Pipeline 1: Send the user's vague query directly to the raw index.
    No LLM processing. This is the baseline.

    Returns: (results, effective_query, metadata)
    """
    results = hybrid_search(query_text, raw_index, embedding_backend, top_k=top_k)
    return results, query_text, {"pipeline": "basic_hybrid", "strategy": "direct"}


# ---------------------------------------------------------------------------
# Pipeline 2: Hybrid Search on Enriched Index
# ---------------------------------------------------------------------------

def run_pipeline_enriched(
    query_text: str,
    enriched_index: HybridIndex,
    embedding_backend: EmbeddingBackend,
    *,
    top_k: int = 10,
) -> tuple[list[SearchResult], str, dict]:
    """
    Pipeline 2: Send the user's vague query to the enriched index.
    The index contains products augmented with LLM-generated vague queries,
    so similarity matching works better for informal user language.

    Returns: (results, effective_query, metadata)
    """
    results = hybrid_search(query_text, enriched_index, embedding_backend, top_k=top_k)
    return results, query_text, {"pipeline": "enriched_hybrid", "strategy": "enriched_index"}


# ---------------------------------------------------------------------------
# Pipeline 3: LLM Query Rewriting + Enriched Index
# ---------------------------------------------------------------------------

QUERY_REWRITE_PROMPT = """You are a search assistant for a construction waste management marketplace.

A user has typed a vague search query. Your job is to REWRITE it into a richer, more specific query that will retrieve better results. Do this by internally simulating a clarification process:

1. Read the user's query
2. Ask yourself: What material might they want? What is the use case? What properties matter (strength, fire resistance, weight, cost, weather resistance)? What condition (reclaimed, recycled, waste)?
3. Based on reasonable assumptions, produce a SINGLE rewritten query that is more detailed and specific

RULES:
- The rewritten query must still be natural language (not structured filters)
- Add plausible specifics based on the original intent
- Do NOT hallucinate requirements the user clearly didn't imply
- Keep it under 50 words
- Think about what construction waste products would actually satisfy this need

User's original query: "{query}"

Return JSON: {{"rewritten_query": "...", "reasoning": "..."}}"""


def run_pipeline_rewrite(
    query_text: str,
    enriched_index: HybridIndex,
    embedding_backend: EmbeddingBackend,
    llm_client: VertexLLMClient,
    *,
    top_k: int = 10,
) -> tuple[list[SearchResult], str, dict]:
    """
    Pipeline 3: LLM rewrites the vague query into a richer form,
    then searches the enriched index with the rewritten query.

    Returns: (results, effective_query, metadata)
    """
    # Rewrite the query
    try:
        rewrite_result = llm_client.generate_json(
            QUERY_REWRITE_PROMPT.format(query=query_text),
            cache_key=f"rewrite_{query_text}",
            temperature=0.3,
            max_tokens=512,
        )
        rewritten = rewrite_result.get("rewritten_query", query_text)
        reasoning = rewrite_result.get("reasoning", "")
    except Exception as exc:
        logger.warning("Query rewrite failed for '%s': %s — falling back to original", query_text, exc)
        rewritten = query_text
        reasoning = f"Rewrite failed: {exc}"

    # Search enriched index with the rewritten query
    results = hybrid_search(rewritten, enriched_index, embedding_backend, top_k=top_k)

    metadata = {
        "pipeline": "rewrite_enriched",
        "strategy": "llm_rewrite_then_enriched_search",
        "original_query": query_text,
        "rewritten_query": rewritten,
        "rewrite_reasoning": reasoning,
    }

    return results, rewritten, metadata
