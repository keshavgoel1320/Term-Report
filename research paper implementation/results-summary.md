# CWM Retrieval Study — Results Summary

## Pipeline Comparison

| Pipeline | NDCG@5 | NDCG@10 | MRR | Judge Mean |
| --- | ---: | ---: | ---: | ---: |
| basic_hybrid | 0.7864 | 0.9003 | 0.8069 | 2.9560 |
| enriched_hybrid | 0.7814 | 0.8866 | 0.8869 | 3.2460 |
| rewrite_enriched | 0.7901 | 0.8980 | 0.8717 | 3.2800 |

## Models Used

- **Generation model**: `gemini-2.5-flash-lite`
- **Judge model**: `gemini-2.5-flash-lite`
- **Embedding model**: `BAAI/bge-large-en-v1.5`

## Pipeline Definitions

- **basic_hybrid**: User's vague query → raw BM25+FAISS index (baseline)
- **enriched_hybrid**: User's vague query → enriched index (products augmented with LLM-generated user-like queries)
- **rewrite_enriched**: LLM rewrites user's vague query → enriched index (query-side + document-side enrichment)

---

## Per-Query Results

| Query | Text | Basic NDCG@10 | Enriched NDCG@10 | Rewrite NDCG@10 | Basic Judge | Enriched Judge | Rewrite Judge |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q01 | old wood planks for sale | 1.0000 | 0.8671 | 0.8699 | 3.1000 | 4.1000 | 4.1000 |
| Q02 | salvaged bricks cheap | 1.0000 | 1.0000 | 1.0000 | 4.0000 | 4.0000 | 4.0000 |
| Q03 | recycled concrete blocks near me | 0.8649 | 0.8058 | 0.9139 | 2.7000 | 2.7000 | 3.1000 |
| Q04 | scrap steel beams wanted | 0.9946 | 0.7633 | 0.9277 | 3.1000 | 3.4000 | 3.4000 |
| Q05 | used tiles for kitchen | 0.4677 | 0.5705 | 0.8654 | 1.6000 | 1.9000 | 2.4000 |
| Q06 | leftover insulation material | 1.0000 | 1.0000 | 1.0000 | 4.0000 | 4.0000 | 4.0000 |
| Q07 | pipe offcuts for plumbing | 0.9968 | 0.8391 | 0.4677 | 3.9000 | 3.7000 | 1.6000 |
| Q08 | old doors for shed | 0.7546 | 0.9068 | 0.8210 | 2.5000 | 3.3000 | 3.4000 |
| Q09 | scrap copper cable price | 1.0000 | 1.0000 | 1.0000 | 4.0000 | 4.0000 | 4.0000 |
| Q10 | glass cullet for crafts | 1.0000 | 1.0000 | 1.0000 | 3.0000 | 3.0000 | 3.0000 |
| Q11 | need something for bathroom floor | 0.8279 | 0.5681 | 0.6353 | 1.8000 | 2.3000 | 2.7000 |
| Q12 | material for garden wall | 0.9718 | 0.9399 | 0.8253 | 3.5000 | 3.3000 | 2.9000 |
| Q13 | stuff for paving driveway | 0.6772 | 0.8612 | 0.8940 | 3.9000 | 3.7000 | 3.8000 |
| Q14 | reclaimed timber for furniture | 0.7797 | 0.7741 | 0.8870 | 3.8000 | 3.8000 | 2.5000 |
| Q15 | old roof tiles wanted | 1.0000 | 1.0000 | 1.0000 | 2.2000 | 4.0000 | 3.8000 |
| Q16 | what can I use for soundproofing? | 0.8713 | 0.8401 | 0.9580 | 3.6000 | 3.4000 | 3.5000 |
| Q17 | cheap wall cladding options | 0.9672 | 1.0000 | 0.6879 | 3.3000 | 3.3000 | 2.7000 |
| Q18 | fire resistant ceiling panels | 1.0000 | 1.0000 | 1.0000 | 2.9000 | 3.0000 | 3.0000 |
| Q19 | waterproof building material | 0.5634 | 1.0000 | 1.0000 | 3.0000 | 5.0000 | 4.0000 |
| Q20 | lightweight bricks for extension | 1.0000 | 1.0000 | 0.9944 | 3.0000 | 3.0000 | 2.6000 |
| Q21 | eco friendly building material | 0.8561 | 0.8473 | 0.9851 | 3.5000 | 4.2000 | 4.6000 |
| Q22 | budget friendly stuff for walls | 0.7993 | 0.9305 | 0.9084 | 2.2000 | 2.0000 | 3.6000 |
| Q23 | building materials on a budget | 1.0000 | 0.8210 | 0.8146 | 2.1000 | 2.8000 | 4.2000 |
| Q24 | good stuff for outdoor use | 0.8216 | 0.8122 | 0.9942 | 3.0000 | 4.0000 | 2.9000 |
| Q25 | sustainable building supplies | 1.0000 | 0.9915 | 1.0000 | 3.1000 | 3.8000 | 4.9000 |
| Q26 | demolition materials wanted | 1.0000 | 1.0000 | 1.0000 | 4.0000 | 4.0000 | 4.0000 |
| Q27 | salvaged wood for flooring | 0.6670 | 0.6149 | 0.9886 | 2.2000 | 2.7000 | 3.5000 |
| Q28 | recycled metal for structure | 0.8581 | 0.8989 | 1.0000 | 2.8000 | 4.4000 | 5.0000 |
| Q29 | old glass for windows | 1.0000 | 1.0000 | 0.9723 | 3.0000 | 3.0000 | 3.0000 |
| Q30 | used plastic sheets | 1.0000 | 0.7900 | 0.7725 | 1.0000 | 1.2000 | 2.3000 |
| Q31 | broken concrete for hardcore | 0.9918 | 0.9596 | 0.8238 | 3.9000 | 3.0000 | 3.4000 |
| Q32 | need cheap flooring | 0.9776 | 0.8906 | 0.9099 | 3.7000 | 3.6000 | 3.4000 |
| Q33 | something for a patio | 0.5659 | 0.8430 | 0.6283 | 1.9000 | 3.2000 | 3.0000 |
| Q34 | material for a fence | 0.6237 | 0.4954 | 0.9966 | 1.6000 | 1.4000 | 2.9000 |
| Q35 | insulation for attic | 0.9660 | 0.7802 | 0.8600 | 3.3000 | 3.0000 | 3.2000 |
| Q36 | drainage pipes used | 0.9277 | 0.9173 | 0.6867 | 3.3000 | 3.4000 | 3.2000 |
| Q37 | structural steel scrap | 0.9885 | 0.9598 | 1.0000 | 4.9000 | 4.9000 | 5.0000 |
| Q38 | decorative stone for garden | 0.8334 | 0.7795 | 0.9855 | 3.1000 | 2.8000 | 1.7000 |
| Q39 | old doors interior | 0.9028 | 0.9480 | 0.9828 | 3.7000 | 4.4000 | 4.3000 |
| Q40 | copper wire scrap | 0.9933 | 0.9834 | 0.9704 | 4.8000 | 4.9000 | 4.8000 |
| Q41 | recycled glass for countertops | 1.0000 | 1.0000 | 0.9289 | 3.0000 | 3.0000 | 2.7000 |
| Q42 | anything for a fireplace surround | 0.8967 | 0.7751 | 0.8022 | 2.4000 | 2.5000 | 2.2000 |
| Q43 | cheap roofing material | 1.0000 | 1.0000 | 0.7448 | 3.0000 | 3.0000 | 2.3000 |
| Q44 | stuff for a retaining wall | 0.7644 | 0.7649 | 0.8238 | 3.9000 | 3.8000 | 3.0000 |
| Q45 | recycled wood chips for mulch | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Q46 | salvaged windows | 1.0000 | 0.9182 | 0.7251 | 1.0000 | 3.6000 | 3.0000 |
| Q47 | old plasterboard for sale | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.3000 | 4.0000 |
| Q48 | scrap aluminum sheets | 0.9600 | 0.9802 | 0.9138 | 3.3000 | 3.4000 | 3.3000 |
| Q49 | used carpet tiles | 0.8859 | 0.8921 | 0.9319 | 4.2000 | 4.1000 | 4.1000 |
| Q50 | what about old radiators? | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |