# Research results — out-of-context image–caption verification project

**Status:** research instrument complete; first valid research results partially obtained.
**Date of this report:** 2026-11-18.
**Scope of evidence:** a 10-case pilot (validation split only) plus a full conformance suite. The 7,264-row held-out test set was not touched during exploration.

---

## 1. Executive summary

The project verifies whether a caption matches an image (MATCHED / MISMATCHED). The classifier is real but modest: **macro F1 0.692, ROC AUC 0.778**. At that level it is a triage signal, not an automatic judge.

The most defensible contribution is **methodological**. While building the RAG study we found that three apparent "model failures" were actually **integration defects**:

1. the retriever was a stub returning fabricated passages;
2. the retriever's passage schema did not match what the assessor expected;
3. the assessor's neutral-confidence formula made its citation threshold unreachable.

All three were found, fixed, and pinned with conformance cases. A fourth limitation remains: the deterministic assessor labels plain captions `unresolved`, so it cannot produce SUPPORTED/CONTRADICTED for newsroom-style text. That is a genuine capability limit, not a bug to tune away.

**We therefore do not yet have a valid answer to "does RAG beat non-RAG?"** The only recorded comparison (`reports/rag_non_rag_eval.json`) is a 50-row fixture check, and our pilot was invalidated by the defects above. What we *do* have is a working pipeline with measured retrieval quality and per-stage attribution, ready for the first real experiment.

---

## 2. What the system does and what was measured

### Classifier (trained, held-out result)

| Quantity | Value |
|---|---|
| Candidate | D-C (MLP head on frozen CLIP ViT-B/32) |
| Test accuracy | 0.692 |
| Test macro F1 | 0.692 |
| Test ROC AUC | 0.778 |
| Operating threshold | 0.45065754652023315 (selected on validation) |
| Calibration temperature | 1.285 (fitted on 7,024 validation rows; original value unrecorded) |
| Best epoch / seed | 10 / 42 |
| Rows | 25,000 train / 7,024 val / 7,264 test |
| Confusion matrix | `[[2404, 1228], [1009, 2623]]` |

### Conformance suite

| Measure | Value |
|---|---|
| Registered case ids | **404** |
| Result | **403 passed, 1 skipped, 0 failed** |
| Skipped case | `parity.skip_when_reference_absent` (reference features absent — expected) |
| Test modules | 63, mirrored against `src/` |
| doc/00 | byte-identical (SHA-256 verified through the normalised hash) |

### Research additions

| Artifact | Value |
|---|---|
| Pilot cases | 10 from `val.csv` (5 MATCHED / 5 MISMATCHED) |
| Caption variants | 40 (V1/V3/V4/V6); AI-drafted, human-reviewed and adopted |
| Mismatch annotations | 5, single reviewer; κ not measurable |
| Evidence corpus | 3,512 real ground-truth captions from `val.csv` |
| Corpus manifest SHA-256 | `04974b961de841715e863b8e83a709bbeae03b26eb43db6ca5cecb51fe411c58` |
| Cases with gold evidence | 7 of 10 (the other 3 images have no consistent caption) |

---

## 3. The three defects we found and fixed

### Defect 1 — the retriever was a stub

`retrieval_api.retrieve()` fabricated passages of the form
`"Relevant passage i concerning {query}"` and returned empty results when no
index object was supplied. A real TF-IDF retriever already existed in
`src/retrieval/corpus.py` but was never called.

- **Symptom:** every case attributed to F1 (no evidence); all RAG conditions scored identically.
- **Fix:** `retrieve()` now delegates to the real corpus and adapts real passage ids, excerpts and scores. Missing corpus returns explicit `UNAVAILABLE` rather than a silent empty success.
- **Pin:** `retriever_api.uses_real_corpus`, `retriever_api.no_synthetic_passages`, `retriever_api.unavailable_is_explicit`, `retriever_api.real_result_schema`, `retriever_api.empty_query_ok`.

### Defect 2 — passage schema mismatch

`retrieval_api.RetrievedPassage` is flat (`.chunk_id`, `.text`, `.score`), but
`rag.assessment.run_assessment` expects `reporting.model.RetrievedPassage` with
`.passage.excerpt`. Every assessment raised internally and returned status FAILED.

The real cause was invisible because the diagnostics recorded only `status`, not
`error_detail`. The hidden message was:

```
'RetrievedPassage' object has no attribute 'passage'
```

This led to an **incorrect F6 ("generation unavailable") diagnosis**, which I explicitly retract.

- **Fix:** an adapter in `rag_diagnostics.py` converts flat passages into the nested schema; `error_detail` is now always recorded.
- **Pin:** `rag_diag.adapter_converts_flat_passages`, `rag_diag.schema_mismatch_attributed`.

### Defect 3 — unreachable citation threshold

`_score_passage` returned confidence `0.3 × overlap` for neutral passages. Since
overlap ≤ 1.0, confidence could never exceed the 0.35 citation threshold — an
exact-match passage was structurally uncitable.

- **Fix:** neutral confidence is now the overlap itself (capped at 0.9), making the threshold meaningful.
- **Pin:** `rag_generation.neutral_overlap_reaches_threshold`.
- **Effect:** one citation per case is now accepted, but the outcome is still INCONCLUSIVE (see the remaining limitation below).

> **Lesson recorded:** a per-stage tracer must record *why* a stage failed, not just that it failed. Status alone produced two wrong diagnoses.

---

## 4. RAG vs non-RAG — the actual numbers

### The only recorded comparison (not a valid experiment)

From `reports/rag_non_rag_eval.json`, a **50-row fixture check of the evaluator**:

| Arm | Accuracy | Macro F1 |
|---|---|---|
| non-RAG | 0.500 | 0.167 |
| RAG | 0.120 | 0.095 |
| Paired Δ | **−0.380** | — |
| Same-prediction rate | 0.060 | |
| Citation validity | 0.000 | |
| Semantic support accuracy | 0.000 | |

This is a pipeline sanity check, **not evidence about RAG**. Both arms are far below the deployed classifier, and the outcomes are a four-class claim layer rather than binary image–caption agreement.

### Our pilot (10 cases, closed-world corpus)

| Stage | Measurement |
|---|---|
| Retrieval recall@5 | **0.714286** |
| Retrieval precision@5 | 0.142857 |
| Empty retrieval rate | 0.000 |
| Synthetic passage rate | 0.000 (regression guard) |
| Outcomes (C5) | 10 × INCONCLUSIVE |
| Accuracy, every condition C0–C6 | 0.500 |
| Macro F1, every condition | 0.333333 |
| RAG Δ (C5 vs C0) | 0.000, bootstrap CI [0, 0] |
| Same-prediction rate | 1.000 |

**Why no difference?** Retrieval is not the bottleneck — the true caption is rank 1 for several mismatched cases. The assessor accepts a citation but classifies it `unresolved`; `run_assessment` counts only `supported` and `contradicted` toward an outcome, so support_count = contra_count = 0 and the outcome is INCONCLUSIVE. With no outcome, C5's explicit mapping cannot change the label, so RAG and non-RAG produce identical decisions.

**Conclusion:** we cannot yet say RAG is better or worse. The observed equality is caused by a downstream capability limit, not by evidence being unhelpful. Answering the question requires an assessor that can resolve plain captions — a small local vision–language model was designed for this and remains unbuilt.

---

## 5. Study 1 and Study 2 pilot numbers

These are **pipeline validation**, not research findings: n = 10, one reviewer, no κ, AI-drafted variants.

| Measurement | Value |
|---|---|
| V0 accuracy | 0.500 |
| ECE (V0) | 0.104 |
| Δ V1 paraphrase | 0.000 |
| Δ V2 shorten | 0.000 |
| Δ V3 expand | 0.000 |
| Δ V4 claim-style | **+0.200** |
| Δ V5 negate | 0.000 |
| Δ V6 translate | 0.000 |

The +0.2 for claim-style captions is **two cases out of ten** and is the opposite of hypothesis H2. It must not be reported as evidence that claim-style captions are easier.

| Signal | AURC | Selective accuracy @50% |
|---|---|---|
| CLIP cosine | 0.761151 | 0.200 |
| D-C score | 0.418135 | 0.400 |
| Calibrated probability | 0.418135 | 0.400 |

Directionally, the classifier score ranks errors better than raw cosine. With n = 10 this is a hypothesis, not a result.

---

## 6. Limitations

**Model**
- 0.692 macro F1 — roughly three errors in ten
- One validation split used for model selection, threshold selection and temperature fitting; overfit risk documented in doc/03
- Test split evaluated once and cannot be reused for tuning
- Original calibration temperature unrecorded; T = 1.285 was refit

**Evidence and RAG**
- Shipped `data/corpus.json` has six passages
- Our 3,512-passage corpus is **closed-world** (dataset captions), not web evidence — must be reported as such
- The deterministic assessor cannot resolve plain captions (no stance vocabulary)
- `non_rag_outcome` is null; no genuine baseline arm output has been produced

**Research quality**
- 10 cases, single reviewer, κ unmeasurable
- Variants were AI-drafted then human-adopted; recorded in `data/research/README.md`
- Three of ten cases have no gold evidence at all

**Deployment**
- `models/` is empty; no `release.json`; the verified loader path cannot be used
- No API, no server, no privacy or retention design
- CLIP is English-centric; multilingual behaviour untested

**Documented in project docs:** doc/03 §14 (split usage), doc/07 §12 (limitations table, C-04/C-05), data/research/README.md.

---

## 7. Where this can be used

| Setting | Fit | Constraint |
|---|---|---|
| **Fact-checking desk triage** | Best | Assistive only; a human makes the call |
| Archive / mis-caption audit | Good | Works on stored image–caption pairs |
| **Research instrument** | Strongest current use | Reproducible, conformance-tested, per-stage attribution |
| Product listings / insurance / rentals | Adjacent | Requires retraining and revalidation |
| Automated takedown or publication | **Not suitable** | 0.692 macro F1 is too low |

---

## 8. What to fix next, in order

1. **Real assessor.** Wire a small local vision–language model (Ollama, `minicpm-v4.6:1b` or `qwen3.5:4b`) so evidence can actually resolve a claim. Design already in `C:\Users\Ronny\.opencode\plan\`.
2. **Re-run Study 4** to get the first valid RAG vs non-RAG answer.
3. **Produce a real baseline arm** for `non_rag_outcome` — it is currently null, so the paired protocol is incomplete.
4. **Package the checkpoint** with `release.json` and verify it via `scripts/verify_checkpoint.py`.
5. **Scale the study** to 100–300 cases with human-authored variants and a second annotator for κ.
6. **Monitor drift and recalibrate** before any deployment.

Do not tune the keyword assessor to produce a favourable RAG result — that would invalidate the experiment.

---

## 9. Documentation map — where things live

| Location | Contains | Committed? |
|---|---|---|
| `result.md` (this file) | Consolidated conclusions, numbers, limitations | Yes |
| `doc/00`–`doc/07` | Contracts, architecture, layout, traceability | Yes |
| `data/research/README.md` | Data provenance, corpus scope, review history | Yes |
| `C:\Users\Ronny\.opencode\plan\` | Study designs, implementation plans | No — outside the repo |
| `reports/conformance.json` | 404 case outcomes | No — `reports/` is gitignored |
| `reports/study1/`, `reports/study2/`, `reports/study4/` | Generated measurements | No — gitignored |
| Study `.md` files in the plan dir | Result templates, mostly placeholders | No |

This report is the first **committed** record of the findings. Previously a reader could recover the design but not the results.

---

## 10. Reproduction

```bash
# Full conformance suite
python -m tests.run --tier all --report reports/conformance.json

# Build the closed-world evidence corpus
python scripts/build_evidence_corpus.py --split val --out data/research/evidence_corpus.json

# Study 1
python scripts/run_failure_audit.py \
  --manifest data/research/study1_cases.jsonl \
  --variants data/research/study1_variants.jsonl \
  --checkpoint dataset/models/checkpoint --temperature 1.285 \
  --temperature-provenance "fitted_on_val_7024_rows" \
  --output reports/study1/audit_results.jsonl
python scripts/report_failure_audit.py --input reports/study1/audit_results.jsonl \
  --output reports/study1/results.json

# Study 2
python scripts/run_selective_prediction.py --input reports/study1/audit_results.jsonl \
  --output reports/study2/signals.json

# Study 4
python scripts/run_rag_ablation.py \
  --manifest data/research/study4_manifest.jsonl \
  --checkpoint dataset/models/checkpoint --temperature 1.285 \
  --temperature-provenance "fitted_on_val_7024_rows" \
  --corpus data/research/evidence_corpus.json \
  --output reports/study4/rag_traces.jsonl
python scripts/report_rag_ablation.py --input reports/study4/rag_traces.jsonl \
  --output reports/study4/results.json
```

---

## 11. Research integrity notes

- No result in this report is derived from the 7,264-row test split during exploration.
- No evidence passage is fabricated. The corpus is real dataset captions, labelled closed-world.
- `gold_evidence_ids` is derived deterministically from consistent dataset pairs and recorded in `study4_evidence_judgements.jsonl`.
- Variant data is AI-drafted and human-adopted, with `origin`, `author` and `reviewed_on` fields on every row.
- Two earlier diagnoses were wrong and are explicitly retracted in §3. No erroneous conclusion was written into the study documents.
- All unmeasured values remain `[PLANNED - NOT YET MEASURED]`; no number was estimated into a report.
