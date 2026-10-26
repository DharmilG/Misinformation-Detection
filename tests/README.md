# tests

The conformance suite. Every test carries a **case id** that names the
document clause it enforces, so the suite is a conformance report rather than
a pile of asserts.

## The layout

A test lives in the folder of the component it verifies, and its file name
mirrors the module it tests.

> `src/retrieval/filters.py` → `tests/retrieval/test_filters.py`

| Folder | Component | Document | Status |
|---|---|---|---|
| `clip_model/` | `src/clip_model/` — the CLIP encoder, the feature cache, the parity diagnostic, the classifier | 01, 03 | populated |
| `datasets/` | `src/datasets/` — the corpus, the manifests, the splits, the extraction | 02 | populated |
| `input_pipeline/` | `src/input_pipeline/` — validation, preprocessing, the request schema | 04 | populated |
| `retrieval/` | `src/retrieval/` — the evidence corpus, ingestion, chunking, retrieval | 06 §47 | populated |
| `rag/` | `src/rag/` — the prompt, the assessment, the eligibility rule, the outcome | 06 §48 | populated |
| `reporting/` | `src/reporting/` — the report model, the comparison, the export | 05 | populated |
| `spec/` | no component — the documentation and the suite's own invariants | 07 | populated |

`fixtures/` holds the shared synthetic artefacts — generated images, caption
corpora, and the golden-vector baseline. `conftest.py`, `ids.py`, `docmap.py`,
and `run.py` are the harness itself and stay flat at the top.

## Status

**Complete.** All 7 component folders (`clip_model/`, `datasets/`, `input_pipeline/`,
`retrieval/`, `rag/`, `reporting/`, and `spec/`) are fully populated with 311
registered contract test cases across 54 test modules. Document 07 §2 is the
authoritative statement of the layout.

## Running the suite

```bash
python -m tests.run                     # everything
python -m tests.run --tier fast         # no model weights, no dataset
python -m tests.run --tier data         # the dataset, no model weights
python -m tests.run --tier slow         # the model weights
python -m tests.run -k truncate         # only the cases whose id contains "truncate"
python -m tests.run --list              # every id, tier, and the clause it enforces
python -m tests.run --report reports/conformance.json
```

A test that asserts a requirement the implementation does not yet meet is
marked `xfail(strict=True)` with a one-line reason. The suite is therefore
green today, and each gap stays pinned and named: when the gap is fixed the
test reports **unexpectedly passing**, which fails the run until the marker is
removed.

`AGENTS.md` has the short version for agents.
