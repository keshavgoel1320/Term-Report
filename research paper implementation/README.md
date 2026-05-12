# LLM-Augmented Hybrid Retrieval for Specialised Domain Marketplaces

A comparative study evaluating whether LLM-based document enrichment and query rewriting improve retrieval quality for informal, underspecified user queries in domain-specific product marketplaces.

> **Paper:** Aryan Thakur, *"LLM-Augmented Hybrid Retrieval for Informal Queries in Specialised Domain Marketplaces: A Comparative Study of Document and Query Enrichment Strategies"*, April 2026.

---

## The Problem

Specialised marketplaces (construction materials, industrial equipment, etc.) suffer from a **vocabulary gap**: products are listed by domain experts using precise technical language, while buyers search using vague, informal queries. A user searching *"something strong for the floor"* won't match a listing for *"hand-cleaned kiln-sound solid brick batch with compressive strength certification."*

This study tests whether LLMs can bridge that gap — on the **document side** (enriching listings with informal language) and the **query side** (rewriting vague queries into richer forms).

---

## Pipelines Evaluated

| # | Pipeline | What Happens | Query-Time LLM? |
|---|----------|--------------|:---:|
| 1 | `basic_hybrid` | Vague query → raw BM25 + FAISS index | No |
| 2 | `enriched_hybrid` | Vague query → LLM-enriched index (products + generated user-like queries) | No |
| 3 | `rewrite_enriched` | LLM-rewritten query → LLM-enriched index | Yes |

All pipelines share an identical hybrid retrieval backend: **BM25 (sparse)** + **FAISS with BGE-Large-EN-v1.5 embeddings (dense)**, fused via weighted linear combination (α=0.4 BM25, 0.6 dense).

---

## Key Results

| Pipeline | NDCG@5 | NDCG@10 | MRR | Mean Judge Score |
|----------|-------:|--------:|----:|-----------------:|
| `basic_hybrid` | 0.7864 | 0.9003 | 0.8069 | 2.956 |
| `enriched_hybrid` | 0.7814 | 0.8866 | 0.8869 | 3.246 |
| `rewrite_enriched` | 0.7901 | 0.8980 | 0.8717 | 3.280 |

**Headline findings:**
- Document enrichment improves result relevance by **+9.8%** (judge score) with **zero query-time latency**
- Query rewriting adds a marginal **+1.0%** over enrichment alone, but introduces **rewrite drift** risk on precise queries
- The hybrid BM25 + dense baseline is already strong — NDCG scores are comparable across all pipelines

**Practical recommendation:** Document-side enrichment is the highest-value intervention for production deployment.

---

## Models Used

| Role | Model |
|------|-------|
| Synthetic corpus generation | Gemini 2.5 Flash |
| Document enrichment, query rewriting, query generation | Gemini 2.5 Flash Lite |
| Relevance judging | Gemini 2.5 Flash Lite |
| Embeddings | BAAI/bge-large-en-v1.5 |

---

## Study Scale

- **1,000** products across **50** categories
- **50** intentionally vague test queries
- **3** retrieval pipelines × 50 queries = 150 retrieval runs
- **~1,100** LLM calls (enrichment + rewriting + judging)
- Evaluation via **pooled LLM-as-judge** (TREC-style pooling, no pre-labelled relevance judgments)

---

## Repository Structure

```
├── README.md                  # This file
├── results.json               # Complete per-query results with metadata
├── results-summary.md         # Comparison table across all 50 queries
├── data/
│   ├── products.json          # 1,000-product synthetic corpus
│   └── queries.json           # 50 test queries
├── .cache/llm/                # Cached LLM responses (reproducibility)
├── src/cwm_research/
│   ├── cli.py                 # CLI entry point (typer)
│   ├── database.py            # Product loading and validation
│   ├── embeddings.py          # BGE embedding wrapper
│   ├── enrichment.py          # LLM document enrichment
│   ├── evaluation.py          # Pooled LLM-as-judge evaluation
│   ├── indexing.py            # BM25 + FAISS index construction
│   ├── pipelines.py           # Pipeline orchestration
│   ├── queries.py             # Test query generation
│   ├── reporting.py           # Results and summary generation
│   ├── retrieval.py           # Hybrid score fusion
│   ├── schemas.py             # Pydantic data models
│   └── vertex_llm.py          # Vertex AI client with caching
└── pyproject.toml             # Project configuration
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Google Cloud ADC configured (`gcloud auth application-default login`)
- Vertex AI API enabled on your GCP project

### Run the Study

```bash
# Install dependencies
uv sync

# Validate the product database
uv run python -m cwm_research validate-db

# Run the full study (uses cached LLM responses if available)
uv run python -m cwm_research run
```

The study produces `results.json`, `results-summary.md`, and `methodology.md`.

### Reproducibility

All LLM responses are cached in `.cache/llm/`. Re-running the study with the existing cache produces **identical results** without making any API calls. To run a fresh study, delete the cache directory first.

---

## License

This project is an academic research study. See the research paper for citation details.
