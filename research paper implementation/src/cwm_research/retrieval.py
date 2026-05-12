"""Hybrid BM25 + dense retrieval search."""

from __future__ import annotations

import logging

import numpy as np

from cwm_research.embeddings import EmbeddingBackend
from cwm_research.indexing import HybridIndex
from cwm_research.schemas import SearchResult

logger = logging.getLogger(__name__)


def hybrid_search(
    query: str,
    index: HybridIndex,
    embedding_backend: EmbeddingBackend,
    *,
    top_k: int = 10,
    bm25_weight: float = 0.4,
    dense_weight: float = 0.6,
) -> list[SearchResult]:
    """Run weighted hybrid search: BM25 + cosine similarity."""
    n = index.size
    if n == 0:
        return []

    # BM25 scores
    query_tokens = query.lower().split()
    bm25_raw = index.bm25.get_scores(query_tokens)
    bm25_max = bm25_raw.max()
    bm25_norm = bm25_raw / bm25_max if bm25_max > 0 else bm25_raw

    # Dense scores (cosine similarity via inner product on L2-normed vectors)
    query_vec = embedding_backend.embed_single_query(query).reshape(1, -1)
    k_search = min(n, max(top_k * 3, 100))  # over-retrieve for fusion
    dense_scores_sparse, dense_indices = index.faiss_index.search(query_vec, k_search)

    dense_full = np.zeros(n, dtype=np.float32)
    for score, idx in zip(dense_scores_sparse[0], dense_indices[0]):
        if 0 <= idx < n:
            dense_full[idx] = score

    dense_max = dense_full.max()
    dense_norm = dense_full / dense_max if dense_max > 0 else dense_full

    # Weighted fusion
    combined = bm25_weight * bm25_norm + dense_weight * dense_norm

    # Get top-k
    top_indices = np.argsort(combined)[::-1][:top_k]

    results: list[SearchResult] = []
    for rank, idx in enumerate(top_indices, start=1):
        results.append(SearchResult(
            rank=rank,
            product_index=int(idx),
            name=index.product_names[idx],
            category=index.product_categories[idx],
            score=round(float(combined[idx]), 6),
            bm25_score=round(float(bm25_norm[idx]), 6),
            dense_score=round(float(dense_norm[idx]), 6),
            description=index.product_descriptions[idx],
        ))

    return results
