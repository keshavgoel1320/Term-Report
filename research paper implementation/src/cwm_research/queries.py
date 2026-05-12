"""Generate 50 vague, user-like test queries via LLM."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from rich.console import Console

from cwm_research.schemas import TestQuery
from cwm_research.vertex_llm import VertexLLMClient

logger = logging.getLogger(__name__)
console = Console()


QUERY_GENERATION_PROMPT = """You are generating test search queries for a construction waste management marketplace.

The marketplace sells RECLAIMED, RECYCLED, and WASTE construction materials. Examples include: salvaged bricks, scrap steel beams, leftover concrete blocks, old timber planks, used tiles, recycled insulation, pipe offcuts, old doors, scrap copper cable, glass cullet, etc.

Generate exactly 50 search queries that REAL NON-EXPERT USERS would type when looking for materials on this platform. Follow these rules strictly:

1. Each query should be SHORT (5-15 words), VAGUE, and INFORMAL
2. Write naturally — use whatever words a real person would actually type
3. Make queries incomplete and ambiguous — like how real people actually search
4. Cover a wide range of use cases: flooring, walls, roofing, drainage, insulation, structural, decorative, outdoor, fire safety, fencing, furniture, gardening, paving, etc.
5. Some queries should mention:
   - A use case without specifying material ("need something for the bathroom floor")
   - A material without specifying use ("got any old wood planks?")
   - A property without context ("anything fire resistant and cheap")
   - Cost concerns ("budget friendly stuff for walls")
   - Environmental preferences ("eco friendly building material")
   - Vague descriptions ("those chunky stone blocks for gardens")
6. Mix query intents:
   - ~15 queries about what the material IS ("cheap metal sheets", "recycled plastic pipes")
   - ~15 queries about what they NEED IT FOR ("something for my driveway", "material for a garden wall")
   - ~10 queries about PROPERTIES ("waterproof and lightweight", "fire resistant ceiling stuff")
   - ~10 queries that are VERY VAGUE ("building materials on a budget", "good stuff for outdoor use")
7. Make sure the 50 queries span at least 20 different use-case areas

Return a JSON array of exactly 50 strings. Each string is one query."""


def generate_test_queries(
    llm_client: VertexLLMClient,
    *,
    output_path: Path | None = None,
    cache_key: str = "test_queries_v1",
) -> list[TestQuery]:
    """Generate 50 vague test queries using the LLM."""
    console.print("[bold cyan]Generating 50 vague test queries via LLM...[/bold cyan]")

    raw_queries = llm_client.generate_json(
        QUERY_GENERATION_PROMPT,
        cache_key=cache_key,
        temperature=0.6,
        max_tokens=4096,
    )

    if not isinstance(raw_queries, list) or len(raw_queries) < 40:
        raise ValueError(f"Expected ~50 queries, got {len(raw_queries) if isinstance(raw_queries, list) else type(raw_queries)}")

    queries = [
        TestQuery(query_id=f"Q{i+1:02d}", text=str(q).strip())
        for i, q in enumerate(raw_queries[:50])
        if q and str(q).strip()
    ]

    console.print(f"  [green][OK] Generated {len(queries)} test queries[/green]")

    # Save to disk
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = [q.model_dump() for q in queries]
        output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved queries to %s", output_path)

    # Preview
    for q in queries[:5]:
        console.print(f"  [dim]{q.query_id}: \"{q.text}\"[/dim]")
    if len(queries) > 5:
        console.print(f"  [dim]... and {len(queries) - 5} more[/dim]")

    return queries


def load_test_queries(path: Path) -> list[TestQuery]:
    """Load previously generated test queries from JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Queries file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [TestQuery.model_validate(item) for item in data]
