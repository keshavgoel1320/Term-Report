"""Load and validate the product database."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

from cwm_research.schemas import Product

logger = logging.getLogger(__name__)


def load_products(products_path: Path) -> list[Product]:
    """Load products from JSON and validate."""
    if not products_path.exists():
        raise FileNotFoundError(f"Product database not found: {products_path}")

    with products_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    products = [Product.model_validate(item) for item in raw]
    validate_products(products)
    logger.info("Loaded and validated %d products from %s", len(products), products_path)
    return products


def validate_products(products: list[Product]) -> dict[str, int]:
    """Validate the product database and return summary statistics."""
    if not products:
        raise ValueError("Product database is empty.")

    names = [p.name for p in products]
    duplicates = [name for name, count in Counter(names).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate product names: {duplicates[:5]}")

    categories = Counter(p.category for p in products)
    subcategories = Counter(p.subcategory for p in products)

    # Check for empty descriptions
    empty_desc = [p.name for p in products if len(p.description.strip()) < 20]
    if empty_desc:
        raise ValueError(f"Products with empty/short descriptions: {empty_desc[:5]}")

    # Check for empty materials
    empty_mat = [p.name for p in products if not p.materials]
    if empty_mat:
        raise ValueError(f"Products with no materials: {empty_mat[:5]}")

    stats = {
        "product_count": len(products),
        "category_count": len(categories),
        "subcategory_count": len(subcategories),
        "min_products_per_category": min(categories.values()),
        "max_products_per_category": max(categories.values()),
    }
    logger.info(
        "Database validated: %d products, %d categories, %d subcategories",
        stats["product_count"],
        stats["category_count"],
        stats["subcategory_count"],
    )
    return stats


def build_document_text(product: Product) -> str:
    """Build the searchable text representation of a product."""
    parts = [
        f"Product: {product.name}",
        f"Category: {product.category}",
        f"Subcategory: {product.subcategory}" if product.subcategory else "",
        f"Materials: {', '.join(product.materials)}",
        f"Condition: {product.condition}",
        f"Dimensions: {product.dimensions}",
        f"Weight: {product.weight_kg:.1f} kg",
        f"Origin: {product.origin_process}",
        f"Reuse applications: {', '.join(product.reuse_applications)}",
        f"Compliance: {', '.join(product.compliance_tags)}",
        f"Description: {product.description}",
    ]
    return "\n".join(part for part in parts if part)
