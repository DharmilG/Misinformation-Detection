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
| `study1_variants.jsonl` | 40 rows (V1/V3/V4/V6 × 10), AI-drafted then human-reviewed and adopted |
| `study1_annotations.jsonl` | 5 annotations, AI-drafted then human-reviewed and adopted |
| `study4_manifest.jsonl` | `gold_outcome` mapped from `gold_label`; 7/10 have real `gold_evidence_ids`; `non_rag_outcome` null |
| `evidence_corpus.json` | 3,512 real ground-truth captions from `val.csv` |

## Evidence corpus and gold evidence (added for Study 4)

`evidence_corpus.json` is a **closed-world corpus** built by
`scripts/build_evidence_corpus.py` from the 3,512 consistent image–caption pairs
in `val.csv`. Each passage id is the dataset `image_id`, and the passage text is
the real ground-truth caption for that image. Provenance and a SHA-256 manifest
hash are in `evidence_corpus.meta.json`.

This is **not** web evidence and must be reported as closed-world. It is real
data from the dataset, not authored text.

`gold_evidence_ids` in `study4_manifest.jsonl` is derived deterministically: for
a case whose image has a consistent caption in `val.csv`, the gold evidence id
is that image id; otherwise the list is empty. 7 of 10 pilot cases have gold
evidence. The derivations, including the evidence text, are recorded in
`study4_evidence_judgements.jsonl` for audit.

Measured on this corpus (10-case pilot): retrieval recall@5 = 0.714,
precision@5 = 0.143, and the gold passage is rank 1 for several mismatched
cases. Assessment then fails in 10/10 cases (no generation backend), so RAG
changes no decision. The diagnosis is F6 (generation), not F1 (retrieval).

## Why some RAG fields are still null

- **`gold_evidence_ids`: `[]` for 3 of 10 cases.** Those images have no
  consistent caption in `val.csv`, so no ground-truth evidence exists for them.
  An empty list is the honest annotation, not a gap to fill by guessing.
- **`non_rag_outcome`: `null`.** This is supposed to be the *baseline arm's
  output*, not a gold label. Copying the gold label into it would fabricate a
  paired comparison. It must be produced by running the baseline; until then it
  stays null. The Study 4 ablation runner does not consume this field.

## Held-out split

All rows come from `val.csv`. The 7,264 test rows were not touched.
