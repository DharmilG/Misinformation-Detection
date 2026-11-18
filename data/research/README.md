# Research data templates — 10-case pilot

Generated from `val.csv` with seed 42 (10 cases, 5 MATCHED / 5 MISMATCHED).
The held-out test split is never sampled.

## Provenance — read this before using the results

The variants and annotations were **drafted by an LLM assistant, then reviewed
and adopted by the project owner on 2026-11-18**. Every filled row records:

- `author` / `annotator`: `human-reviewer`
- `origin`: `AI-generated draft, human-reviewed and adopted`
- `reviewed_on`: `2026-11-18`

The caption text and gold labels are unchanged from the drafts — the review was
an approval, not a rewrite. The `origin` field keeps the drafting history
explicit, so the data is neither misattributed nor silently stripped of its
provenance.

Remaining limits:

- Claim-style variants (V4) were drafted with mixed truth values from image
  inspection. A single reviewer adopted them; they were **not** independently
  adjudicated by a second human.
- The five `wrong_event` / `wrong_location` / `wrong_person` labels were
  likewise adopted by one person. Inter-annotator agreement (Cohen's κ) is
  **not measurable** from this set; it stays `[PLANNED - NOT YET MEASURED]`.
- This is a **10-case pilot for pipeline validation**, not publication-grade
  evidence. A second annotator on ≥30 mismatched cases is needed before these
  results can be reported as research findings.

## Files

| File | Status |
|---|---|
| `study1_cases.jsonl` | filled; `mismatch_type` set for the 5 mismatched cases |
| `study1_variants.jsonl` | 40 rows (V1/V3/V4/V6 × 10), all authored by `assistant-llm` |
| `study1_annotations.jsonl` | 5 annotations, all by `assistant-llm` |
| `study4_manifest.jsonl` | `gold_outcome` mapped from `gold_label` (MATCHED→SUPPORTED, MISMATCHED→CONTRADICTED); evidence IDs empty; `non_rag_outcome` null |

## Why the RAG fields are empty / null

- **`gold_evidence_ids`: `[]`.** No real evidence corpus with stable passage IDs
  exists in this checkout. `retrieve()` returns synthetic or empty passages.
  An empty list is the honest value and is itself evidence for failure mode F1.
- **`non_rag_outcome`: `null`.** This is supposed to be the *baseline arm's
  output*, not a gold label. Copying the gold label into it would fabricate a
  paired comparison. It must be produced by running the baseline; until then it
  stays null. The Study 4 ablation runner does not consume this field.

## Held-out split

All rows come from `val.csv`. The 7,264 test rows were not touched.
