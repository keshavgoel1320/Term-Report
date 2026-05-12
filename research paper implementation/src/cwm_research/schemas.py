"""Data models for the CWM retrieval study."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Product schema — unchanged from the original database
# ---------------------------------------------------------------------------

class Product(BaseModel):
    """A single construction waste management product."""

    name: str
    description: str
    category: str
    subcategory: str = ""
    materials: list[str]
    condition: Literal["new", "reclaimed", "recycled", "waste"]
    weight_kg: float
    dimensions: str
    origin_process: str
    reuse_applications: list[str]
    compliance_tags: list[str]
    tensile_strength: Literal["low", "medium", "high"] | None = None
    flexibility: Literal["low", "medium", "high"] | None = None
    durability: Literal["low", "medium", "high"] | None = None
    corrosion_resistance: Literal["low", "medium", "high"] | None = None
    cost_band: Literal["low", "medium", "high"] | None = None
    weight_class: Literal["light", "medium", "heavy"] | None = None
    environment: list[Literal["indoor", "outdoor", "marine", "high_temperature"]] = Field(default_factory=list)
    price_per_unit_inr: float

    @field_validator("name", "description", "category", "subcategory", "dimensions", "origin_process")
    @classmethod
    def validate_non_empty_strings(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Field must not be empty.")
        return value

    @field_validator("materials", "reuse_applications", "compliance_tags")
    @classmethod
    def validate_non_empty_lists(cls, value: list[str]) -> list[str]:
        if not value or not all(item and item.strip() for item in value):
            raise ValueError("List fields must not be empty.")
        return [item.strip() for item in value]

    @field_validator("weight_kg", "price_per_unit_inr")
    @classmethod
    def validate_positive_numbers(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("Numeric fields must be positive.")
        return round(float(value), 2)


# ---------------------------------------------------------------------------
# Enriched product — product + generated vague queries for indexing
# ---------------------------------------------------------------------------

@dataclass
class EnrichedProduct:
    """A product enriched with LLM-generated vague user queries."""

    product_index: int
    product_name: str
    original_text: str
    generated_queries: list[str]  # 5-8 vague user-like queries
    combined_text: str  # original_text + generated queries joined


# ---------------------------------------------------------------------------
# Test query — intentionally vague
# ---------------------------------------------------------------------------

class TestQuery(BaseModel):
    """A vague, user-like test query."""

    query_id: str
    text: str


# ---------------------------------------------------------------------------
# Search and evaluation results
# ---------------------------------------------------------------------------

class SearchResult(BaseModel):
    """A single retrieved product for a query."""

    rank: int
    product_index: int
    name: str
    category: str
    score: float
    bm25_score: float
    dense_score: float
    description: str


class JudgeScore(BaseModel):
    """LLM judge score for a single retrieved product."""

    product_name: str
    score: int = Field(ge=1, le=5)
    rationale: str


class QueryPipelineResult(BaseModel):
    """Complete result for one query on one pipeline."""

    query_id: str
    query_text: str
    effective_query: str
    pipeline: str
    results: list[SearchResult]
    judge_scores: list[JudgeScore] = Field(default_factory=list)
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    mrr: float = 0.0
    mean_judge_score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Study paths
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StudyPaths:
    """All filesystem paths used by the study."""

    root: Path
    data_dir: Path
    cache_dir: Path
    index_dir: Path
    enriched_index_dir: Path
    products_path: Path
    queries_path: Path
    enrichment_cache_dir: Path
    results_path: Path
    summary_path: Path
    methodology_path: Path
    log_dir: Path
    log_path: Path

    @classmethod
    def from_root(cls, root: Path) -> StudyPaths:
        data_dir = root / "data"
        cache_dir = root / ".cache"
        return cls(
            root=root,
            data_dir=data_dir,
            cache_dir=cache_dir,
            index_dir=cache_dir / "index_raw",
            enriched_index_dir=cache_dir / "index_enriched",
            products_path=data_dir / "products.json",
            queries_path=data_dir / "queries.json",
            enrichment_cache_dir=cache_dir / "enrichment",
            results_path=root / "results.json",
            summary_path=root / "results-summary.md",
            methodology_path=root / "methodology.md",
            log_dir=root / "logs",
            log_path=root / "logs" / "study.log",
        )
