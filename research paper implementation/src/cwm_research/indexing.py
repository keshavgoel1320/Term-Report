"""Build and persist BM25 + FAISS hybrid search indexes."""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from rich.console import Console

from cwm_research.database import build_document_text
from cwm_research.embeddings import EmbeddingBackend
from cwm_research.schemas import EnrichedProduct, Product

logger = logging.getLogger(__name__)
console = Console()


class HybridIndex:
    """A combined BM25 + FAISS index over product documents."""

    def __init__(
        self,
        *,
        bm25: BM25Okapi,
        faiss_index: faiss.Index,
        document_texts: list[str],
        product_names: list[str],
        product_categories: list[str],
        product_descriptions: list[str],
    ) -> None:
        self.bm25 = bm25
        self.faiss_index = faiss_index
        self.document_texts = document_texts
        self.product_names = product_names
        self.product_categories = product_categories
        self.product_descriptions = product_descriptions

    @property
    def size(self) -> int:
        return len(self.document_texts)


def build_raw_index(
    products: list[Product],
    embedding_backend: EmbeddingBackend,
    *,
    cache_dir: Path | None = None,
) -> HybridIndex:
    """Build the raw hybrid index (Pipeline 1) on original product text."""
    if cache_dir and _index_cached(cache_dir):
        console.print("[dim]Loading cached raw index...[/dim]")
        return _load_index(cache_dir)

    console.print("[bold cyan]Building raw index over product documents...[/bold cyan]")
    document_texts = [build_document_text(p) for p in products]
    product_names = [p.name for p in products]
    product_categories = [p.category for p in products]
    product_descriptions = [p.description for p in products]

    index = _build_index(
        document_texts=document_texts,
        product_names=product_names,
        product_categories=product_categories,
        product_descriptions=product_descriptions,
        embedding_backend=embedding_backend,
        label="raw",
    )

    if cache_dir:
        _save_index(index, cache_dir)

    return index


def build_enriched_index(
    enriched_products: list[EnrichedProduct],
    products: list[Product],
    embedding_backend: EmbeddingBackend,
    *,
    cache_dir: Path | None = None,
) -> HybridIndex:
    """Build the enriched hybrid index (Pipelines 2 & 3) on enriched text."""
    if cache_dir and _index_cached(cache_dir):
        console.print("[dim]Loading cached enriched index...[/dim]")
        return _load_index(cache_dir)

    console.print("[bold cyan]Building enriched index over enriched documents...[/bold cyan]")
    document_texts = [ep.combined_text for ep in enriched_products]
    product_names = [ep.product_name for ep in enriched_products]
    product_categories = [products[ep.product_index].category for ep in enriched_products]
    product_descriptions = [products[ep.product_index].description for ep in enriched_products]

    index = _build_index(
        document_texts=document_texts,
        product_names=product_names,
        product_categories=product_categories,
        product_descriptions=product_descriptions,
        embedding_backend=embedding_backend,
        label="enriched",
    )

    if cache_dir:
        _save_index(index, cache_dir)

    return index


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_index(
    *,
    document_texts: list[str],
    product_names: list[str],
    product_categories: list[str],
    product_descriptions: list[str],
    embedding_backend: EmbeddingBackend,
    label: str,
) -> HybridIndex:
    """Build BM25 and FAISS components."""
    # BM25
    console.print(f"  [dim]Tokenizing {len(document_texts)} documents for BM25 ({label})...[/dim]")
    tokenized = [doc.lower().split() for doc in document_texts]
    bm25 = BM25Okapi(tokenized)

    # FAISS
    console.print(f"  [dim]Computing dense embeddings for {len(document_texts)} documents ({label})...[/dim]")
    embeddings = embedding_backend.embed_documents(document_texts)

    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)  # inner product on L2-normed = cosine
    faiss_index.add(embeddings)

    console.print(f"  [green][OK] {label.title()} index built: {len(document_texts)} documents, dim={dimension}[/green]")

    return HybridIndex(
        bm25=bm25,
        faiss_index=faiss_index,
        document_texts=document_texts,
        product_names=product_names,
        product_categories=product_categories,
        product_descriptions=product_descriptions,
    )


def _index_cached(cache_dir: Path) -> bool:
    return all(
        (cache_dir / name).exists()
        for name in ("bm25.pkl", "faiss.index", "metadata.json")
    )


def _save_index(index: HybridIndex, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    with (cache_dir / "bm25.pkl").open("wb") as f:
        pickle.dump(index.bm25, f)
    faiss.write_index(index.faiss_index, str(cache_dir / "faiss.index"))
    metadata = {
        "document_texts": index.document_texts,
        "product_names": index.product_names,
        "product_categories": index.product_categories,
        "product_descriptions": index.product_descriptions,
    }
    (cache_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Index saved to %s", cache_dir)


def _load_index(cache_dir: Path) -> HybridIndex:
    with (cache_dir / "bm25.pkl").open("rb") as f:
        bm25 = pickle.load(f)
    faiss_index = faiss.read_index(str(cache_dir / "faiss.index"))
    metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
    logger.info("Index loaded from %s (%d documents)", cache_dir, len(metadata["document_texts"]))
    return HybridIndex(
        bm25=bm25,
        faiss_index=faiss_index,
        document_texts=metadata["document_texts"],
        product_names=metadata["product_names"],
        product_categories=metadata["product_categories"],
        product_descriptions=metadata["product_descriptions"],
    )
