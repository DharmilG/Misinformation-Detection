# AGENTS.md

Guidance for agents working in this repository.

## Documentation

`doc/` is the single source of truth. `doc/00_PROJECT_OVERVIEW.md` is the project overview and must stay byte-identical — a test asserts its SHA-256.

Read the document that owns a contract before changing it. Each document owns one area and points to the others rather than restating them:

| Document | Owns |
|---|---|
| `01_CLIP_FEATURE_EXTRACTION.md` | CLIP encoder, feature semantics, feature cache, parity diagnostic |
| `02_DATASET_PREPARATION.md` | Dataset source, labels, splits, leakage audit, manifests |
| `03_CLASSIFIER_TRAINING_AND_CHECKPOINTS.md` | Classifier head, training, selection, checkpoint format, release |
| `04_INPUT_AND_INFERENCE_PIPELINE.md` | Input validation, preprocessing, inference path, status model |
| `05_REPORTING_AND_UI.md` | Report model, comparison rules, rendering, export |
| `06_RAG_NON_RAG_EVALUATION.md` | Evidence corpus, retrieval, generation, eligibility, outcomes, RAG/non-RAG |
| `07_INTEGRATION_TESTING.md` | Test tiers, traceability, cross-component tests, reproduction, sign-off |

## Running the tests

```bash
python -m tests.run                     # everything
python -m tests.run --tier fast         # no model weights, no dataset
python -m tests.run --tier data         # the dataset, no model weights
python -m tests.run --tier slow         # the model weights
python -m tests.run -k truncate         # only the cases whose id contains "truncate"
python -m tests.run --list              # every id, tier, and the clause it enforces
python -m tests.run --report reports/conformance.json
```

Tiers are additive: `--tier "data and slow"` selects both. A tier that needs something absent **skips with a reason**; it never fails for a missing dependency.

First-time setup:

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt pandas pytest
```

The CLIP weights (~600 MB) download on the first `slow` run. `dataset/` must be present for the `data` tier.

## Known gaps

Tests that assert a requirement the implementation does not yet meet are marked `xfail(strict=True)` with a one-line reason. The suite is therefore green today, and each gap stays pinned and named.

`strict=True` matters: when a gap is fixed, the test reports **unexpectedly passing**, which fails the run. Remove the `xfail` marker at that point — do not leave it, or the fix is invisible.
