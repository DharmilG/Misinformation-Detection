# Research results — out-of-context image–caption verification project

**Status:** research instrument complete; definitive RAG comparison run under a frozen protocol and reported as inconclusive.
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

**The definitive RAG comparison has now been run under the pre-registered protocol in `research/protocol.md`.** Its result is **inconclusive**: Δ accuracy = 0.000 with a bootstrap 95% CI of [−0.600, +0.600]. Critically, the local assessor returned `SUPPORTED` for all 10 claims, including the false ones, so it did not discriminate. The result is reported as-is; the assessor was not re-prompted after the run.

---

## In plain words

**Same score, different answers.**

Both methods scored 50%. But they did not give the same answers — RAG changed every single one.
The plain classifier called every pair MISMATCHED. With RAG, every pair became MATCHED.
The test set has 5 genuinely matching and 5 genuinely mismatching pairs, so flipping every answer keeps the score at exactly 50%. That is a coincidence, not a tie.
The AI judge called every claim SUPPORTED, even the false ones. So it was not really judging the evidence. We cannot say RAG helped or hurt yet.

**Why did RAG flip every answer?**

1. The evidence corpus is a bag of real captions. Each caption really belongs to some photo.
2. A false caption paired with the wrong photo still exists in that bag — it belongs to a different photo.
3. RAG searched by the caption's own words, so it found the caption itself at rank 1.
4. The AI judge only saw text, never the photo. "These words appear in the evidence" looked like "these words are supported".
5. It answered SUPPORTED for all ten claims, and SUPPORTED was converted into MATCHED. That is why all ten answers flipped.

Concrete example: on the five false pairs, the passage the judge cited was the pair's own caption id (`77605`, `1566593`, `1039824`, `311815`, `1575962`), not the true caption for that image — the true caption for v0006 was `107208`.

**Why were half the answers still wrong?**

For the five genuinely matching pairs, the accident gave the right answer — the caption really does belong to that photo. For the five false pairs it gave the wrong answer — that caption belongs to a different photo. The plain classifier got the false pairs right; RAG got the true pairs right. Each method got exactly one half right, so the 50% is a coincidence of a balanced set, not a tie.

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
| Registered case ids | **414** |
| Result | **413 passed, 1 skipped, 0 failed** |
| Skipped case | `parity.skip_when_reference_absent` (reference features absent — expected) |
| Test modules | 65, mirrored against `src/` |
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

### Definitive run — pre-registered protocol, local assessor

Executed with `research/protocol.md` frozen in advance; model `qwen3:8b` via local
Ollama, temperature 0, prompt `rag-v1`, closed-world corpus hash `04974b96…`,
baseline arm from real no-evidence classifier predictions.

| Arm | Accuracy | Macro F1 |
|---|---|---|
| C1 non-RAG (classifier, no evidence) | 0.500 | 0.333333 |
| C5 RAG (retrieval + local assessor) | 0.500 | 0.333333 |
| **Paired Δ (C5 − C1)** | **0.000** | — |
| **Bootstrap 95% CI (1,000 draws, seed 42)** | **[−0.600, +0.600]** | — |
| Same-prediction rate C1/C5 | 0.000 | |

| Stage | Measurement |
|---|---|
| Retrieval recall@5 | **0.714286** |
| Retrieval precision@5 | 0.142857 |
| Empty retrieval rate | 0.000 |
| Synthetic passage rate | 0.000 |
| Assessor status | OK in 10/10 |
| Assessor outcomes | **SUPPORTED in 10/10** |
| Real citation ids recorded | yes (e.g. 713948, 62103, 175912) |

**Protocol verdict: INCONCLUSIVE** — the CI spans zero, so neither "RAG helps"
nor "RAG hurts" can be claimed.

**Why it is inconclusive rather than a clean null.** The C5 arm does not merely
match C1 — it disagrees with it on **every** case (same-prediction rate 0.000).
The assessor labelled all ten claims `SUPPORTED`, including the five genuinely
out-of-context ones, so RAG flipped every MISMATCHED prediction to MATCHED. The
net accuracy happened to stay 0.500. This is a **degenerate assessor**, not a
neutral one, and it means the comparison did not measure the intended effect.

**Conclusion that can be derived:** retrieval is demonstrably not the bottleneck
(recall@5 = 0.714, true caption often rank 1). The bottleneck is claim assessment:
the deterministic heuristic cannot resolve plain captions, and the local
`qwen3:8b` assessor defaults to agreement. Both were observed under fixed
protocol. Whether RAG helps therefore remains **unanswered**, not answered
negatively. Per the protocol, the assessor and prompt were not tuned after the run.

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

1. **Non-degenerate assessor.** The protocol run showed `qwen3:8b` returns `SUPPORTED` for all claims, including false ones. This needs a **new pre-registered follow-up study** with a different model or prompt — not a silent post-hoc re-prompt of this run.
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
| `research/protocol.md` | Pre-registered RAG protocol and decision rule | Yes |
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
