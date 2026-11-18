# Pre-registered protocol — RAG vs non-RAG on out-of-context verification

**Frozen before the local-assessor run.** Any change after the definitive run
invalidates the result and must be recorded as an exploratory follow-up.

## Research question

Does retrieval-augmented assessment improve binary verification accuracy over
the same classifier without evidence, on the 10-case validation pilot?

## Arms

| Arm | Contents |
|---|---|
| C1 (non-RAG) | frozen CLIP + D-C classifier, no evidence |
| C5 (RAG) | C1 + real TF-IDF retrieval + local assessor + explicit outcome mapping |

Paired by case id. Identical claim text in both arms. The non-RAG arm receives
no evidence.

## Primary metric

**Paired Δ accuracy = accuracy(C5) − accuracy(C1).**

## Secondary metrics

- macro F1 per arm
- same-prediction rate
- outcome distribution (SUPPORTED / CONTRADICTED / INCONCLUSIVE / CONFLICTING)
- retrieval recall@5 and precision@5 against `gold_evidence_ids`

## Uncertainty

1,000 bootstrap resamples, seed 42, 95% percentile CI on the paired Δ.

## Decision rule — fixed in advance

| Outcome | Condition |
|---|---|
| **RAG helps** | Δ > 0 **and** CI lower bound > 0 |
| **RAG hurts** | Δ < 0 **and** CI upper bound < 0 |
| **Inconclusive** | CI spans zero |

An inconclusive result is reported as inconclusive. The assessor, prompt,
corpus and thresholds are **not** tuned after this run to change the outcome.

## Fixed inputs

| Input | Value |
|---|---|
| Checkpoint | `dataset/models/checkpoint` |
| Weights SHA-256 | recorded in `result.md` |
| Threshold | 0.45065754652023315 |
| Temperature | 1.285 (fitted on 7,024 validation rows) |
| Corpus | `data/research/evidence_corpus.json`, 3,512 passages |
| Corpus SHA-256 | `04974b961de841715e863b8e83a709bbeae03b26eb43db6ca5cecb51fe411c58` |
| Gold evidence | `data/research/study4_manifest.jsonl`, 7/10 cases |
| Assessor model | `qwen3:8b` via local Ollama, temperature 0 |
| Prompt version | `rag-v1` (recorded in the run manifest) |
| Split | validation-derived pilot only; 7,264 held-out test rows untouched |

## Validity limits — accepted before the run

- 10 cases, one reviewer, κ unmeasurable
- closed-world corpus of dataset captions, not web evidence
- 3 cases have no gold evidence; they remain in the accuracy comparison
- no human-subject study yet

These are limitations of scope, not defects to fix post hoc.

## Reporting

The outcome goes into `result.md` verbatim with all secondary metrics. A
negative or inconclusive result is reported as such. No number is estimated
into the report; unmeasured fields stay `[PLANNED - NOT YET MEASURED]`.
