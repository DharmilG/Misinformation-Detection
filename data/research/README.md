# Research data templates — 10-case pilot

Generated from `val.csv` with seed 42 (10 cases, 5 MATCHED / 5 MISMATCHED).
The held-out test split is never sampled.

## Provenance — read this before using the results

**The variants and annotations in this pilot were authored by an LLM assistant,
not by a human fact-checker.** Every filled row carries `"author":
"assistant-llm"` or `"annotator": "assistant-llm"` so this cannot be forgotten.

Consequences:

- Claim-style variants (V4) were written with mixed truth values by inspection
  of the images, not adjudicated by a second human. Their gold labels are
  **inferred**, not independently confirmed.
- The five `wrong_event` / `wrong_location` / `wrong_person` labels are the
  assistant's judgement from viewing each image. Inter-annotator agreement
  (Cohen's κ) is **not measurable** from this set; it stays
  `[PLANNED - NOT YET MEASURED]`.
- This pilot is for **pipeline validation**, not for publication-quality
  findings. Replace these files with human-authored variants before reporting
  study results as research evidence.

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
