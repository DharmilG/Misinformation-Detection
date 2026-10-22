# 07 - Integration Testing

## Multimodal Out-of-Context Misinformation Detection Using Retrieval-Augmented Generation (RAG)

**Document status:** Accepted - implementation contract for Phase 9  
**Document type:** Cross-component protocol and sign-off specification  
**Depends on:** `../00_PROJECT_OVERVIEW.md`, `01_CLIP_FEATURE_EXTRACTION.md`, `02_DATASET_PREPARATION.md`, `03_CLASSIFIER_TRAINING_AND_CHECKPOINTS.md`, `04_INPUT_AND_INFERENCE_PIPELINE.md`, `05_REPORTING_AND_UI.md`, `06_RAG_NON_RAG_EVALUATION.md`  
**Source:** Authored from the approved testing-and-evaluation plan. The source document `10_TESTING_AND_EVALUATION.md` was never written to disk; this document and the integration-specific requirements carried in Documents 01-09 are its content.  
**Last updated:** 2026-10-21

---

## 1. Purpose and boundary

This document defines **how the assembled system is tested, measured, reproduced, and signed off**.

| Concern | Owner |
|---|---|
| Component contracts, component unit tests, component acceptance criteria | Documents 01-06 |
| **Research evaluation methodology** - retrieval metrics, generation metrics, human review, the RAG/non-RAG comparison | **Document 06** |
| **The test tier taxonomy, markers, commands and environment** | **This document** |
| **Cross-component interface and schema tests** | **This document** |
| **The end-to-end case manifest and the availability rates** | **This document** |
| **Latency and resource measurement procedures** | **This document** |
| **Clean-clone and offline reproduction** | **This document** |
| **Contract-to-test traceability, acceptance criteria and sign-off** | **This document** |

### 1.1 The boundary with Document 06

Document 06 owns the **research evaluation protocol**: the retrieval dev/test split, the metric formulas, the human-review protocol, the confusion matrix, the injection suite, and the RAG/non-RAG experimental design. Those are **not** duplicated here.

This document owns the **integration verification**: that the research components actually work together, that the interfaces hold, that the failures degrade as specified, that the whole application behaves, and that the result is reproducible on a clean machine.

A test that verifies a component's own contract belongs to that component's document. A test that verifies a **boundary between two components** belongs here.

### 1.2 Explicit non-promises

| Non-promise | Why it matters |
|---|---|
| **A combined system score** | Document `00` §10.6 forbids collapsing the classifier, retrieval, and generation into one number. No such metric is defined here |
| **A metric without its context** | A number without its sample size, configuration, and caveats is not reportable |
| **A passing suite as evidence of correctness** | Coverage and green tests are engineering evidence. The scientific claims live in Document 06 and carry their own caveats |
| **A result implied by the design** | Every cell in the results register (§18) starts at `NOT YET RUN` |

---

## 2. Test tier taxonomy, markers, commands and environment

### 2.1 The tiers

Markers are **additive**. A test may carry more than one, and a selection expression composes them.

| Marker | Requires | Used by |
|---|---|---|
| *(none)* | The runtime dependencies installed. No model weights, no dataset, no network | The contract tests, the cache tests, the harness tests, the documentation tests |
| `data` | Plus the local dataset at `dataset/ooc_dataset/` | The corpus contract tests, the manifest tests, the preprocessing-skew measurement |
| `slow` | Plus the CLIP model weights | The encoder behaviour tests, the golden-vector regression, the parity diagnostic |
| `eval` | Plus the labelled evaluation sets | The metric production of document 07's §18 |
| `benchmark` | Plus the candidate models and corpora | The three benchmarks of §4 |
| `e2e` | The full local stack | The whole-application tests |

Two consequences of making the markers additive rather than exclusive:

- The **corpus contract tests need the dataset but not the 600 MB of weights**, so they run in seconds. The corpus's integrity, its label gate, its split structure, and its shortcut ceilings are all verifiable without a model.
- The **preprocessing-skew measurement and the dataset-scale extraction need both**, and are marked `data` and `slow` together.

A tier whose requirement is absent **skips with a stated reason**. It never fails for a missing dependency, and it never silently passes.

### 2.2 The commands

```bash
python -m tests.run                       # everything
python -m tests.run --tier fast           # no weights, no dataset
python -m tests.run --tier data           # the dataset, no weights
python -m tests.run --tier slow           # the weights
python -m tests.run --tier "data and slow" # both
python -m tests.run --tier eval           # metric production
python -m tests.run --tier benchmark      # the three benchmarks
python -m tests.run --tier e2e            # the whole application
python -m tests.run -k truncate           # only the cases whose id contains "truncate"
python -m tests.run -m "not slow"         # a raw pytest expression
python -m tests.run --list                # every id, its tier, and the clause it enforces
python -m tests.run --dataset-root PATH --encoder-cache PATH
python -m tests.run --report reports/conformance.json
```

**The `-k` filter selects by the case id**, not by the test function name. The case id is the stable name a reader finds in this document and in the conformance report, so `python -m tests.run -k truncate` finds the truncation case regardless of what the function is called. A tier selection may name several tiers at once, joined with `and`, because the markers are additive.

The equivalent `pytest` invocations are:

```bash
pytest -m "not slow and not data and not eval and not benchmark and not e2e"   # the fast suite
pytest -m "data"                                                               # the corpus contract
pytest -m "slow"                                                               # the encoder
pytest -m "data and slow"                                                      # both
```

The runner is a thin wrapper over `pytest`. It adds the tier and filter selection, the dataset and encoder-cache paths, and the conformance report.

### 2.3 Marker discipline

| Requirement | Enforcement |
|---|---|
| The markers are declared in the repository's pytest configuration | A test asserts the declared markers match the ones the suite uses |
| The fast suite never touches the network | A test asserts that the fast selection contains no network-dependent test, by blocking sockets during the fast run |
| The fast suite never runs an expensive test | The default selection omits `slow`, `data`, `eval`, `benchmark`, and `e2e` |
| A tier that needs something absent skips with a reason | Enforced by the fixtures, and asserted by the harness tests |
| An `eval` test cannot tune and report on the same cases | The harness records which split it read and refuses to emit a test-set report from the development file |
| A benchmark result records the candidate configuration and the selection rule | Enforced by the harness, and asserted in the benchmark output |
| **Every collected test carries a registered id and a tier** | A test asserts it, so a test added without an id fails the suite |
| **A full run produces zero collection errors** | A test asserts it, so an import-time failure anywhere in the tree is caught |

The last two are what keep the suite and this document from drifting apart.
### 2.4 Environment requirements

| Tier | Environment |
|---|---|
| Fast | Python with the pinned dependencies installed. No weights, no dataset, no network |
| Component | The encoder or model weights present in the local cache, and the dataset or a subset of it |
| Evaluation | The labelled evaluation sets under `data/eval/`, and the corresponding caches |
| Benchmark | The candidate models and the corpora of §4 |
| End-to-end | The full local stack: the encoder, the classifier artifact, the evidence corpus and index, and the generation model |

### 2.5 Coverage

| Scope | Requirement |
|---|---|
| Overall `src/` coverage | **Reported**, not gated. A coverage number on its own is a vanity metric |
| Gated modules | **Measured separately per module.** Each must individually meet **90% line coverage**. A combined 90% could conceal a poorly tested module |
| Gated module set | Document 04's validation; Document 01's encoder contract; Document 02's label audit and manifest builder; Document 06's filters, lexical index, validation, eligibility and outcome computation; Document 05's comparison and rendering; Document 03's loader and safe loading |
| Behavioural assertions | **Retained.** Coverage is not evidence of correctness. The outcome computation, the citation eligibility rule, the metadata validation, and the comparison matrix are covered by behavioural tests, not by line counts |
### 2.6 The test layout

The suite is laid out to mirror `src/`.

> A test lives in the folder of the component it verifies, and its file name mirrors the module it tests.
> `src/retrieval/filters.py` → `tests/retrieval/test_filters.py`

The rule already holds inside the per-document module layouts — document 06 §47 lists `src/retrieval/filters.py` in its module block and `test_filters.py` in its test block, and this change gives the test block the matching folder heading. The folder makes the pairing explicit, and it resolves the collisions that a flat directory cannot: `test_pipeline.py` is both document 04's inference pipeline and document 02's extraction pipeline, and `test_dataset.py` is document 03's torch Dataset reader.

| Folder | Component | Document | Owns |
|---|---|---|---|
| `tests/clip_model/` | `src/clip_model/` | Document 01, Document 03 | The encoder, the feature cache, the parity diagnostic, the golden vectors, the classifier |
| `tests/datasets/` | `src/datasets/` | Document 02 | The corpus, its label gate, its manifests, its splits, the dataset-scale extraction |
| `tests/input_pipeline/` | `src/input_pipeline/` | Document 04 | The validation, the preprocessing, the request schema, the status model |
| `tests/retrieval/` | `src/retrieval/` | Document 06 §47 | The evidence corpus, ingestion, chunking, the index, lexical and dense retrieval, fusion, the filters, dedup |
| `tests/rag/` | `src/rag/` | Document 06 §48 | The prompt, the assembly, the validation, the eligibility rule, the outcome |
| `tests/reporting/` | `src/reporting/` | Document 05 | The report model, the comparison, the export |
| `tests/spec/` | no component | Document 07 | The documentation, and the suite's own invariants |

Two things stay flat at the top of `tests/`, because they are the harness rather than a component's tests:

| File | Role |
|---|---|
| `tests/conftest.py` | The fixtures, the tier skips, the declared xfails, the result capture |
| `tests/ids.py` | The case registry - `case()` and `known_gap()` |
| `tests/docmap.py` | Resolving a document section reference |
| `tests/run.py` | The command line |
| `tests/fixtures/` | The shared synthetic artefacts - generated images, caption corpora, the golden baseline |

`tests/spec/` deserves its note. Its tests belong to no component: they check the **documentation** and the **suite itself**. They are what stops the specification and the implementation from drifting apart, which is what makes the rest of the suite trustworthy — the frozen overview, the canonical owner sections, the case-id registry, and the traceability table below. They are not tests of the system, and they are not run separately; they are in the same command as everything else, because a check that is easy to forget is a check that does not happen.

**Status.** The folders exist and the move is under way. `tests/clip_model/` is populated; the other six folders are still skeletons, and the test modules that belong to them are still flat in `tests/`. The tables in §12.2 and §12.3 are keyed by **case id**, not by file path, so the layout does not change which requirement any case enforces.

---
---

## 3. The evaluation harness invariants

`pytest -m "eval"` selects by marker; it does not make a metric methodologically sound. Each evaluation script therefore **asserts its own preconditions** and refuses to run otherwise.

| # | Assertion | On failure |
|---|---|---|
| 1 | The script is reading the split it is named for - the development script reads the development file, the test script reads the test file | The run aborts |
| 2 | The configuration fingerprint matches the recorded one | The run aborts |
| 3 | The provenance - model version, corpus manifest hash, prompt version, retrieval configuration - is present | The run aborts |
| 4 | The output is written with the configuration and provenance, so a number cannot be produced without its context | The run aborts |
| 5 | The script does not read the evaluation fixture it is not named for | The run aborts |

A test script cannot be made to produce a "test set" report from the development file by accident.

---

## 4. The three benchmarks and their ownership

| Benchmark | Owner | Compares | Status |
|---|---|---|---|
| **Classifier benchmark** | Document 03 §9 | The pre-registered classifier families, under the validation protocol of Document 03 §8 | **[PLANNED - NOT YET MEASURED]** |
| **Retrieval benchmark** | Document 06 §38 | BM25-only, dense-only, hybrid, and hybrid with reranking | **[PLANNED - NOT YET MEASURED]** |
| **Generation-backend benchmark** | Document 06 §24 | The local model candidates, on schema compliance, evidence handling, latency, and resource use | **[PLANNED - NOT YET MEASURED]** |

Only benchmarks that are **implemented** are reported as results. A planned script is not a project capability, and the status column above is the authority on which of the three exist.

The three are distinct. The classifier benchmark selects among heads using **validation** data. The retrieval benchmark selects among retrievers using the **development** split of the retrieval evaluation set. The generation-backend benchmark selects among local models on **schema compliance, evidence handling, latency and resource use** - it is not an accuracy benchmark and does not use the generation evaluation set.

---

## 5. Cross-component interface and schema tests

These verify the boundaries between components, not the components themselves.

| # | Interface | Test | Assertion |
|---|---|---|---|
| 1 | Document 04 → Document 01 | `tests/input_pipeline/test_pipeline.py` | An `InputRequest` produced by the input pipeline satisfies every invariant in Document 04 §9 |
| 2 | Document 01 → Document 04 | `tests/input_pipeline/test_pipeline.py` | The `EmbeddingPair` fields the inference path reads - the two vectors, the token counts, and the truncation flag - are present and consistent |
| 3 | Document 01 → Document 03 | `tests/clip_model/test_classifier.py` | The feature family recorded in the checkpoint's `inference` block reconstructs a feature vector of the length the first layer expects |
| 4 | Document 03 → Document 04 | `tests/clip_model/test_checkpoint.py` | A loaded classifier returns a result whose label matches the recorded threshold, positive class, and class order |
| 5 | Document 04 → Document 06 | `tests/retrieval/test_retriever.py` | The `AssessmentContext` carries the normalized caption, both image flags, the classifier result, and the interpretation string, and no other fields |
| 6 | Document 06 → Document 05 | `tests/reporting/test_reporting.py` | Every retrieved passage renders with its complete metadata, and every accepted citation resolves to a retrieved passage |
| 7 | Document 06 → Document 05 | `tests/reporting/test_reporting.py` | The assessment section's status equals the assessment's upstream status; the interface never re-classifies it |
| 8 | Document 04 → Document 05 | `tests/reporting/test_reporting.py` | The classifier section's status equals the classifier result's upstream status |
| 9 | Document 05 → Document 05 | `tests/reporting/test_reporting.py` | The comparison state matches Document 05 §8.2's exhaustive mapping for the given pair of directions |
| 10 | Document 02 → Document 06 | `tests/retrieval/test_corpus_terms.py` | The corpus manifest hash recorded in the index metadata matches the current manifest |
| 11 | Document 02 → Document 03 | `tests/clip_model/test_dataset.py` | The manifest's per-row image and caption hashes match the feature cache's per-row hashes |
| 12 | Document 06 → Document 06 | `tests/rag/test_outcome.py` | The computed outcome satisfies the per-status invariants of Document 06 §30.2 for every status combination |

---

## 6. Model-loading and prediction tests

### 6.1 Model loading

| # | Test | Assertion |
|---|---|---|
| 1 | `tests/clip_model/test_checkpoint.py` | The checkpoint loads through `load_checkpoint` with the real encoder present |
| 2 | `tests/clip_model/test_checkpoint.py` | Every file's SHA-256 is verified **before** any file is deserialized |
| 3 | `tests/clip_model/test_checkpoint.py` | A modified file raises `CheckpointIntegrityError` and the deserializer is never reached |
| 4 | `tests/clip_model/test_checkpoint.py` | An archive whose hash differs from the published value raises `ArtifactTrustError` |
| 5 | `tests/clip_model/test_checkpoint.py` | The state dict loads with `strict=True`; a missing or unexpected key raises |
| 6 | `tests/clip_model/test_checkpoint.py` | `weights_only=True` is used, and an environment without the flag raises rather than falling back |
| 7 | `tests/clip_model/test_compatibility.py` | A hard-requirement failure raises `CheckpointIncompatibleError` and the message states both the installed and the expected version |
| 8 | `tests/clip_model/test_compatibility.py` | A declared-range failure warns, and escalates to an error under `strict_compatibility` |
| 9 | `tests/clip_model/test_checkpoint.py` | The feature vector is reconstructed from the recorded family, and the dimension is asserted against the first layer |
| 10 | `tests/clip_model/test_checkpoint.py` | Offline mode B loads from a populated cache with `local_files_only=True` and no network |
| 11 | `tests/clip_model/test_checkpoint.py` | Offline mode C raises `EncoderUnavailableError` with the command to populate the cache |
| 12 | `tests/clip_model/test_checkpoint.py` | Offline mode D raises `EncoderProvenanceError` when a cached encoder file's hash differs |

### 6.2 Prediction

| # | Test | Assertion |
|---|---|---|
| 1 | `tests/input_pipeline/test_pipeline.py` | A valid request produces `status == OK` with a label, a score, and a logit |
| 2 | `tests/input_pipeline/test_pipeline.py` | The label is determined by thresholding the **uncalibrated** score against the recorded threshold |
| 3 | `tests/input_pipeline/test_pipeline.py` | A checkpoint with no calibration artifact produces a label and a score, with `probability is None` and `calibration_status == ABSENT` |
| 4 | `tests/input_pipeline/test_pipeline.py` | The label is identical with and without a calibration artifact |
| 5 | `tests/input_pipeline/test_pipeline.py` | The same input twice yields an identical score, probability, and label, under a fixed environment |
| 6 | `tests/input_pipeline/test_pipeline.py` | The cold request is measurably slower than the warm path, and the warm path does not reload the encoder |
| 7 | `tests/input_pipeline/test_pipeline.py` | The registry is constructed exactly once across many requests |

---

## 7. Degraded-path and error-handling tests

### 7.1 The degradation matrix

Each row asserts the status, the stages that still run, and the absence of fabricated values.

| Failure | Component | Expected status | Stages that still run | Must not produce |
|---|---|---|---|---|
| Empty upload | Input | `InputRejected` | None | A report, a request id, a model load |
| Unsupported format | Input | `InputRejected` | None | A report |
| Corrupt image | Input | `InputRejected` | None | A report |
| Empty caption | Input | `InputRejected` | None | A report |
| Over-long caption | Input | `InputRejected` | None | A report |
| Encoder absent from the cache | Features | `SKIPPED` | Retrieval, generation, report | A score, a label |
| Encoder hash mismatch | Features | `SKIPPED` | Retrieval, generation, report | A score, a label |
| Checkpoint absent | Classifier | `UNAVAILABLE` | Retrieval, generation, report | A label, a score, a probability |
| Checkpoint corrupt | Classifier | `UNAVAILABLE` | Retrieval, generation, report | A label, a score, a probability |
| Environment incompatible | Classifier | `UNAVAILABLE` | Retrieval, generation, report | A label, a score, a probability |
| Index missing or version-mismatched | Retrieval | `UNAVAILABLE` | Report | A verdict |
| Nothing relevant retrieved | Retrieval | `OK` with zero passages | Generation, report | An outcome |
| Generator failure | Generation | `UNAVAILABLE` | Report | An outcome, an assessment prose |
| Rate limit or timeout | Generation | `UNAVAILABLE` | Report | An outcome |
| Output rejected by validation | Generation | `UNAVAILABLE` | Report | An outcome |
| Source metadata absent or invalid | Evidence | `OK` | Generation, report | A fabricated date, a fabricated link |

### 7.2 The no-fabrication assertions

| # | Assertion |
|---|---|
| 1 | Every non-`OK` result has **all** value fields `None` - no `0.0`, no `0.5`, no "unknown" as a number |
| 2 | No rendered value is a placeholder - never `""`, `"N/A"`, `"unknown"`, `"-"` |
| 3 | A source with a missing URL renders "Source URL not available." and never a manufactured link |
| 4 | A source with a missing date renders "Publication date not stated by the source" and never a guessed date |
| 5 | An empty retrieval produces `INCONCLUSIVE` with `NO_EVIDENCE` and renders no outcome |
| 6 | A degraded section is visible in the report; it is never hidden |
| 7 | The report renders for every valid request, including when every downstream component has failed |

---

## 8. The end-to-end case manifest

### 8.1 The artefact

```text
data/eval/e2e_cases.jsonl
```

The evaluation sets of Documents 02, 06, and 03 have **different units** - image-caption pairs with classifier labels, retrieval queries with relevance judgements, and claims with assessment labels. They are **not** interchangeable, and the end-to-end manifest is a separate artefact that defines how individual cases exercise the whole system.

### 8.2 The case record

| Field | Content |
|---|---|
| `case_id` | Stable identifier |
| `purpose` | `functionality` · `retrieval_coverage` · `semantic_correctness` · `failure_handling` |
| `input` | The image reference and the caption, **or** the intended invalid-input condition |
| `expected_path` | Which components are expected to run |
| `expected_status` | The expected status of each component |
| `expected_render` | The expected report or rejection behaviour |
| `gold_outcome` | The gold assessment label, **where a defensible label exists**, else `null` |

### 8.3 Rules

| Rule | Detail |
|---|---|
| Not every case has a gold label | A corrupt upload tests rejection and rendering, not whether a claim is true |
| The manifest is versioned and hashed | The numbers are traceable to a fixed case list |
| The manifest does not imply a unified accuracy | It exercises the whole system. The classifier, retrieval, and generation metrics continue to come from their own sets in Documents 03 and 06 |
| The failure cases are fixed | Every row of the degradation matrix in §7.1 has at least one case |

---

## 9. Availability rate definitions

Six rates, each with an explicit denominator. They measure different things and are never combined.

| Rate | Numerator | Denominator | What it measures |
|---|---|---|---|
| **Report-render success rate** | Valid requests that produced a renderable report | All valid requests in the evaluation set | Application robustness |
| **Input-rejection success rate** | Invalid requests that produced the correct rejection view | All invalid requests in the evaluation set | Validation handling |
| **Assessment availability rate** | Valid requests that produced a valid assessment | Valid requests **on which assessment is expected to run** | Generation component reliability |
| **Decision availability rate** | Valid requests producing `SUPPORTED`, `CONTRADICTED`, or `CONFLICTING` | All valid requests in the evaluation set | Evidence coverage **and** component reliability together |
| **Conditional decision availability** | Decided assessments | Successfully produced assessments | How often an available assessment reaches a decided outcome |
| **Degraded-path rate** | Requests with at least one component not `OK` | The specified denominator | How often the system runs degraded |

### 9.1 Three distinctions that must not be collapsed

| Distinction | Detail |
|---|---|
| **A rejection view is a success, not a failure** | The report-render rate is over *valid* requests and the rejection rate over *invalid* ones. A correctly rejected corrupt upload is never counted as a pipeline failure |
| **The decision availability rate is not purely a coverage measure** | It falls both for insufficient evidence and for a technical failure. It is therefore reported **alongside the conditional decision availability rate**, and it is described as reflecting coverage and reliability together |
| **`CONFLICTING` is a decided outcome but not a resolved claim** | The system has identified conflicting evidence; it has not resolved the truth of the claim |

---

## 10. Latency and resource measurement procedures

### 10.1 The protocol

| Element | Definition |
|---|---|
| **Timing boundaries** | End-to-end - submission to report ready - and per stage: input validation, CLIP encoding, classifier inference, retrieval, generation, rendering |
| **Cold versus warm** | Measured and reported **separately**. The first run is excluded from the steady-state percentiles, and the number of discarded warm-up runs is stated |
| **Cold-start cost** | The encoder and the local generation model load dominate. The cold-start time is reported on its own, because the local-first deployment loads a ~600 MB encoder and a multi-gigabyte model |
| **Run count** | At least **200** measured steady-state runs, with the rationale. If fewer are possible, the run count is reported and **no p99 is claimed** |
| **Hardware** | CPU model, GPU model if any, RAM, operating system, Python version, library versions, and whether the run was CPU-only |
| **Resource metric** | The process peak resident set size, sampled by a defined method, and GPU memory separately when a GPU is used |
| **Failures** | Failed runs are **excluded** from the latency percentiles, and their frequency is reported alongside. A percentile computed over a subset is stated as such |

A p99 from a handful of runs is not a high-percentile estimate. The document refuses to report one, and says why.

### 10.2 The per-stage measurement list

The stages to be timed are specified in Document 04 §22. The procedure above, the reporting format, and the cross-stage roll-up are specified here.

### 10.3 The reporting format

Each reported figure carries: the platform, the number of runs, the measurement method, whether the run was cold or warm, and the failure count excluded from the percentiles.

---

## 11. The component interface and boundary matrix

This matrix is the single place where the inter-component boundaries are stated. Each component document keeps a short pointer to it.

### 11.1 Producer and consumer

| Producer | Artifact | Consumer |
|---|---|---|
| Document 04 (input) | `InputRequest` | Document 01 (features), Document 04 (inference) |
| Document 01 (features) | `EmbeddingPair` | Document 04 (inference) |
| Document 03 (checkpoints) | `LoadedClassifier` | Document 04 (inference) |
| Document 04 (inference) | `ClassifierResult`, `AssessmentContext` | Document 06 (retrieval and generation) |
| Document 06 (retrieval) | `RetrievalResult` | Document 06 (generation), Document 05 (report) |
| Document 06 (generation) | `Assessment` | Document 05 (report) |
| Document 05 (report) | `ReportModel` | The interface |

### 11.2 The inherited-contract matrix

Every contract that crosses a component boundary, in one place. Each row names the producing component, the contract, and the requirement. This matrix is the consolidation of the eight per-document handoff tables that the original specification set carried.

#### From `01_CLIP_FEATURE_EXTRACTION.md` — the encoder

| Contract | Requirement | Consumer |
|---|---|---|
| Image normalization | Dataset images must pass through the same normalization — EXIF orientation, RGB conversion, alpha flattening, downscale to 2048 px — before reaching the encoder. A separate loader must not resize or re-normalize | `02_DATASET_PREPARATION.md` |
| Caption normalization | Dataset captions must use the same text normalization, so `caption_sha256` in the cache is comparable to the one computed at inference | `02_DATASET_PREPARATION.md` |
| Encoder API | `encode_images` and `encode_texts` for bulk extraction; `encode_batch` for aligned pairs | `02_DATASET_PREPARATION.md` |
| Feature contract | `(N, 512)` float32, finite, unit-norm arrays | `02_DATASET_PREPARATION.md`, `04_INPUT_AND_INFERENCE_PIPELINE.md` |
| Provenance | `EncoderInfo` written into the cache metadata | `02_DATASET_PREPARATION.md` |
| Cache schema | The layout, key, metadata, atomic-write and validation rules | `02_DATASET_PREPARATION.md` |
| IDs | Stable sample IDs, unique per split, used for the per-row hash check | `02_DATASET_PREPARATION.md` |
| The `inference` block | Everything needed to reproduce the selected prediction path | `03_CLASSIFIER_TRAINING_AND_CHECKPOINTS.md`, `04_INPUT_AND_INFERENCE_PIPELINE.md` |
| The encoder manifest | So the inference pipeline can verify its encoder before use | `03_CLASSIFIER_TRAINING_AND_CHECKPOINTS.md`, `04_INPUT_AND_INFERENCE_PIPELINE.md` |
| The pairing guarantee | Image and text encoders are executed with the same model instance, the same revision, and the same device | `04_INPUT_AND_INFERENCE_PIPELINE.md` |

#### From `02_DATASET_PREPARATION.md` — the dataset

| Contract | Requirement | Consumer |
|---|---|---|
| Manifests | One JSONL per configuration and split, with `manifest_meta.json` | `03_CLASSIFIER_TRAINING_AND_CHECKPOINTS.md` |
| Features | Cached vectors under the encoder's contract, one cache namespace per configuration | `03_CLASSIFIER_TRAINING_AND_CHECKPOINTS.md` |
| Labels | `MATCHED` / `MISMATCHED` / `UNKNOWN`, with the source labels retained | `03_CLASSIFIER_TRAINING_AND_CHECKPOINTS.md` |
| Sample IDs | Stable and unique within a configuration, for index alignment between the manifest and the cache | `03_CLASSIFIER_TRAINING_AND_CHECKPOINTS.md` |
| Class counts | Per split, per label, with the imbalance ratio | `03_CLASSIFIER_TRAINING_AND_CHECKPOINTS.md` |
| Leakage context | The audit report and the four exposure fields | `03_CLASSIFIER_TRAINING_AND_CHECKPOINTS.md` |
| Feature contract | `(N, 512)` float32, finite, unit-norm | `03_CLASSIFIER_TRAINING_AND_CHECKPOINTS.md` |

#### From `03_CLASSIFIER_TRAINING_AND_CHECKPOINTS.md` — the classifier and the checkpoint

| Contract | Requirement | Consumer |
|---|---|---|
| `load_checkpoint` | The single entry point to a loaded model | `04_INPUT_AND_INFERENCE_PIPELINE.md` |
| `LoadedClassifier.predict` | The full prediction path — feature construction, scoring, calibration, thresholding | `04_INPUT_AND_INFERENCE_PIPELINE.md` |
| The `inference` block | Everything needed to reproduce the selected prediction path | `04_INPUT_AND_INFERENCE_PIPELINE.md` |
| The encoder manifest | So the inference pipeline can verify its encoder before use | `04_INPUT_AND_INFERENCE_PIPELINE.md` |
| The typed errors | So the interface can distinguish "model unavailable" from "bad input" | `04_INPUT_AND_INFERENCE_PIPELINE.md`, `05_REPORTING_AND_UI.md` |
| The model card | The source of the user-facing explanation of what the score means | `04_INPUT_AND_INFERENCE_PIPELINE.md`, `05_REPORTING_AND_UI.md` |
| The verification report | Evidence that the deployed artefact is the intended one | `04_INPUT_AND_INFERENCE_PIPELINE.md`, `05_REPORTING_AND_UI.md` |
| `weights.pt` | The selected model's best-validation-epoch `state_dict` | `04_INPUT_AND_INFERENCE_PIPELINE.md` |
| `config.json` | The freeze record — everything needed to reconstruct the model and the decision that selected it | `04_INPUT_AND_INFERENCE_PIPELINE.md` |
| `calibration.json` | The calibration procedure and its fitted parameter | `04_INPUT_AND_INFERENCE_PIPELINE.md` |
| `metrics.json` | Per-seed and summary metrics, the baselines, the threshold, and the leakage context | `04_INPUT_AND_INFERENCE_PIPELINE.md`, `05_REPORTING_AND_UI.md` |
| `provenance.json` | Encoder, data, and environment provenance | `04_INPUT_AND_INFERENCE_PIPELINE.md` |
| Canonical artefact | Seed 42, designated by convention | `04_INPUT_AND_INFERENCE_PIPELINE.md` |
| Verification procedure | A stated numerical tolerance and the evidence needed to check it | `04_INPUT_AND_INFERENCE_PIPELINE.md` |

#### From `04_INPUT_AND_INFERENCE_PIPELINE.md` — the input and the inference

| Contract | Requirement | Consumer |
|---|---|---|
| `AssessmentContext` | The normalized caption as the query seed, the raw caption for display, the classifier result with its interpretation, and the two image flags | `06_RAG_NON_RAG_EVALUATION.md` |
| `request_id` | For correlating retrieved evidence with the request that produced it | `06_RAG_NON_RAG_EVALUATION.md` |
| The untrusted-data rule | Retrieved text is evidence, never instructions. Document `06` implements the prompt defences; Document `04` enforces the structured handoff | `06_RAG_NON_RAG_EVALUATION.md` |
| The two-signal rule | The classifier result and the evidence assessment stay separate, and neither overrides the other | `06_RAG_NON_RAG_EVALUATION.md` |

#### From `06_RAG_NON_RAG_EVALUATION.md` — retrieval, to generation

| Contract | Requirement | Consumer |
|---|---|---|
| `RetrievalResult` | The ranked passages, each with its complete metadata and its named scores | `06_RAG_NON_RAG_EVALUATION.md` |
| The empty-but-`OK` case | Zero passages with `status == OK` is a normal outcome and must produce an **Inconclusive** assessment | `06_RAG_NON_RAG_EVALUATION.md` |
| The `UNAVAILABLE` case | The index is missing or broken. A system failure, stated as such | `06_RAG_NON_RAG_EVALUATION.md` |
| `retriever_config` | The model, revision, `k`, the fusion method, and whether reranking was applied — so the generation stage can describe how the evidence was found | `06_RAG_NON_RAG_EVALUATION.md` |
| `corpus_stats` | The corpus size and build time, so the assessment can state the coverage it had access to | `06_RAG_NON_RAG_EVALUATION.md` |
| The distinctness counts | So the generation stage can see whether it received one source or several | `06_RAG_NON_RAG_EVALUATION.md` |
| The trust rule | Retrieved text is evidence, never instructions | `06_RAG_NON_RAG_EVALUATION.md` |
| The separation rule | The classifier result is one input among several and is never a fact to be confirmed by the sources | `06_RAG_NON_RAG_EVALUATION.md` |

#### From `06_RAG_NON_RAG_EVALUATION.md` — generation, to the interface

| Contract | Requirement | Consumer |
|---|---|---|
| The computed outcome | With its reason code, its scope text, and its `outcome_source` | `05_REPORTING_AND_UI.md` |
| The component statuses | Which claim components were supported, contradicted, contested, or unresolved | `05_REPORTING_AND_UI.md` |
| The citations | With their quotes, their eligibility, and their full source metadata | `05_REPORTING_AND_UI.md` |
| The rejected citations | With their reasons, so a fabrication attempt is visible | `05_REPORTING_AND_UI.md` |
| The unresolved items, conflicts, uncertainties, limitations | The model's account of what it could not settle | `05_REPORTING_AND_UI.md` |
| The classifier result | From Document `04`, as a separate result | `05_REPORTING_AND_UI.md` |
| The provenance | The model, the prompt version, the truncation record, and the backend | `05_REPORTING_AND_UI.md` |

#### From `05_REPORTING_AND_UI.md` — the interface, to the evaluation and sign-off

| Contract | Requirement | Consumer |
|---|---|---|
| The report model | The structure the end-to-end checks read | `07_INTEGRATION_TESTING.md` |
| The status vocabulary | `OK / UNAVAILABLE / SKIPPED / FAILED`, shared across the whole project | `07_INTEGRATION_TESTING.md` |
| The exposure fields | The dataset-split and leakage diagnostics, which belong to the evaluation report rather than to a per-analysis export | `07_INTEGRATION_TESTING.md` |
| The per-component metrics | From the classifier, retrieval, and generation evaluations | `07_INTEGRATION_TESTING.md` |
| The demonstration path | The synthetic sample inputs, which need no dataset and no network | `07_INTEGRATION_TESTING.md` |

### 11.3 The boundary prohibitions

| Component | Must not | Reference |
|---|---|---|
| Document 03 (classifier) | Train on the grouped configuration and report on the official one, or the reverse | Document 03 §15 |
| Document 03 (classifier) | Tune on VERITE's test examples and report them as independent | Document 03 §15 |
| Document 03 (classifier) | Present a metric without the leakage context from Document 02 | Document 02 §11.2.1 |
| Document 03 (classifier) | Modify a manifest. Manifests are inputs; a correction is a new configuration with a new version | Document 02 §12 |
| Document 03 (checkpoints) | Change the model, the threshold, the calibration, or the label mapping | Document 03 §17 |
| Document 03 (checkpoints) | Present the canonical seed as the best-performing model | Document 03 §23 |
| Document 03 (checkpoints) | Release the checkpoint without `provenance.json` | Document 03 §30 |
| Document 04 (inference) | Present a conclusion when the retrieval result is empty | Document 06 §33 |
| Document 04 (inference) | Cite anything that is not in the supplied passages | Document 06 §28 |
| Document 04 (inference) | Treat a retrieved passage as an instruction | Document 06 §37 |
| Document 04 (inference) | Collapse the classifier result and the evidence assessment into a single verdict | Document 05 §3 |
| Document 05 (report) | Recalculate the evidence outcome | Document 06 §30 |
| Document 05 (report) | Override the evidence outcome, in either direction | Document 06 §30 |
| Document 05 (report) | Re-derive eligibility or re-classify a citation | Document 06 §28 |
| Document 05 (report) | Present the classifier result and the evidence outcome as a single combined verdict | Document 05 §3 |
| Document 06 (retrieval) | Return a verdict. It returns passages and their metadata | Document 06 §21 |
| Document 06 (retrieval) | Treat a retrieved passage as an instruction | Document 06 §37 |
| Document 06 (retrieval) | Read the classifier result as a fact to be confirmed by the sources | Document 06 §32 |
| Document 06 (retrieval) | Present a single combined label across the two signals | Document 05 §3 |
| Document 06 (generation) | Present a conclusion when the retrieval result is empty | Document 06 §33 |
| Document 06 (generation) | Cite anything that is not in the supplied passages | Document 06 §28 |
| Document 06 (generation) | Treat a retrieved passage as an instruction | Document 06 §37 |
| Document 06 (generation) | Collapse the classifier result and the evidence assessment into a single verdict | Document 05 §3 |
| Document 07 (integration) | Report a metric without its context, or combine components into a single score | Document `00` §10.6 |
| Document 07 (integration) | Present a zero observed error rate as a universal guarantee | Document 06 §39 |

### 11.4 The preserved requirement on the interface

> **The interface displays already-computed upstream results.** It must not recalculate classifier predictions, reinterpret retrieved evidence, or generate a new overall truth verdict.

This requirement is stated in Document 05 §1.1 and enforced by the tests of Document 05 §5 rows 7-9.

---

## 12. Contract-to-test traceability

Every contract has a row. The status column is the authority, and it is not a promise - it is read from the conformance report that the suite writes.

### 12.1 The status vocabulary

| Status | Meaning |
|---|---|
| `IMPLEMENTED` | A test enforces the contract and **passes** today |
| `DECLARED GAP` | A test enforces the contract and is marked `xfail(strict=True)`. The requirement is not met yet, and the gap is named |
| `PLANNED` | No test exists yet. The contract is specified but not yet covered |

A `DECLARED GAP` is a pinned, named requirement, not an excuse. When the implementation catches up, the test reports **unexpectedly passing** and the run fails, which forces the marker to be removed. The two states cannot both be true.

The full report, with the clause and the requirement for every case, is `reports/conformance.json`. It is regenerated by `python -m tests.run --report reports/conformance.json`.

### 12.2 The covered contracts

| Contract | Document and section | Test cases | Tier | Status |
|---|---|---|---|---|
| The encoder's output contract | Document 01 §7.2 | `clip.extract.shape`, `clip.extract.dtype`, `clip.extract.finite` | slow | IMPLEMENTED |
| The token metadata in the result | Document 01 §7.2 | `clip.extract.token_metadata` | slow | DECLARED GAP |
| The public API surface | Document 01 §7.1 | `clip.api.encode_pair_exists`, `clip.api.encode_images_exists`, `clip.api.encode_texts_exists`, `clip.api.encode_batch_exists` | fast | DECLARED GAP |
| The projection into the shared space | Document 01 §8.1 | `clip.extract.projection_dim_from_config` | slow | IMPLEMENTED |
| The L2 normalisation | Document 01 §8.2 | `clip.extract.unit_norm`, `clip.extract.norm_is_idempotent` | slow | IMPLEMENTED |
| The EOT-position pooling | Document 01 §8.3 | `clip.extract.text_pooling_is_eot` | slow | IMPLEMENTED |
| The truncation of a caption past the window | Document 01 §5.2 | `clip.extract.long_caption_truncates` | slow | DECLARED GAP |
| The encoder's lifecycle | Document 01 §6.1 | `clip.api.single_load_per_process` | fast | DECLARED GAP |
| The evaluation mode and the detached graph | Document 01 §6.2, Document 01 §6.3 | `clip.extract.eval_mode`, `clip.extract.no_autograd_graph` | slow | IMPLEMENTED |
| The pinned revision and the provenance record | Document 01 §4.2, Document 01 §4.3 | `clip.api.revision_pinned`, `clip.api.encoder_info_provenance` | fast | DECLARED GAP |
| The device policy | Document 01 §9.1 | `clip.api.device_selectable` | fast | DECLARED GAP |
| The error contract | Document 01 §13.1, Document 01 §13.2 | `clip.api.errors_module_exists`, `clip.api.typed_errors` | fast | DECLARED GAP |
| Repeatability on CPU | Document 01 §14.2 | `clip.extract.repeatable_on_cpu` | slow | IMPLEMENTED |
| The saved feature format | Document 01 §11.2 | `cache.saved_array_shapes`, `cache.npz_no_pickle` | slow | IMPLEMENTED |
| The cache key | Document 01 §11.3 | `cache.key_components` | fast | DECLARED GAP |
| The cache metadata | Document 01 §11.4 | `cache.metadata_present` | fast | DECLARED GAP |
| The atomic write | Document 01 §11.5 | `cache.atomic_write` | fast | DECLARED GAP |
| The load validation | Document 01 §11.6 | `cache.load_validates`, `cache.npz_roundtrip_values_identical` | fast, slow | MIXED |
| The device and the cache key | Document 01 §11.7 | `cache.device_excluded_by_default`, `cache.device_namespace_flag` | fast | DECLARED GAP |
| The parity diagnostic | Document 01 §12.2, Document 01 §12.4, Document 01 §12.7, Document 01 §12.8 | `parity.id_alignment`, `parity.text_path_within_tolerance`, `parity.report_generated`, `parity.image_path_excluded` | fast | DECLARED GAP |
| The parity precondition | Document 01 §15.7 | `parity.skip_when_reference_absent` | fast | IMPLEMENTED |
| The golden-vector regression | Document 01 §15.5, Document 01 §14.3 | `golden.detects_change` (implemented); `golden.baseline_present`, `golden.reproduces_within_tolerance` (gaps) | fast, slow | MIXED |
| The corpus source and the row counts | Document 02 §4.1 | `dataset.rows_match_csv` | data | IMPLEMENTED |
| The label verification gate | Document 02 §5 | `dataset.rows_match_csv`, `dataset.label_balance_exact`, `dataset.pair_structure`, `dataset.label0_caption_eq_image`, `dataset.labels_json_rule_matches`, `dataset.all_images_exist`, `dataset.one_caption_per_id`, `dataset.one_path_per_id` | data | IMPLEMENTED |
| The label mapping and the class balance | Document 02 §6.2, Document 02 §6.4 | `dataset.labels_json_rule_matches`, `dataset.label_balance_exact` | data | IMPLEMENTED |
| The image reference resolution | Document 02 §7 | `dataset.all_images_exist` | data | IMPLEMENTED |
| The manifest record and its integrity | Document 02 §12 | `dataset.one_caption_per_id`, `dataset.one_path_per_id` | data | IMPLEMENTED |
| The split structure and the preserved official split | Document 02 §9 | `dataset.no_cross_split_caption_overlap`, `dataset.no_cross_split_image_overlap` | data | IMPLEMENTED |
| The shortcut ceilings | Document 02 §13.1 | `dataset.shortcut_ceiling_caption_only`, `dataset.shortcut_ceiling_image_only`, `dataset.caption_repeats_within_split`, `dataset.pair_structure` | data | IMPLEMENTED |
| The corpus pre-resize | Document 02 §8.1.1 | `dataset.image_shorter_side`, `skew.downscale_is_noop_on_corpus`, `skew.crop_region_equivalent` | data | IMPLEMENTED |
| The accepted skew, measured | Document 02 §8.1.2 | `skew.native_vs_upscaled_similarity`, `skew.inference_variant_matches_pipeline`, `preprocess.downscale_is_not_observable_in_the_embedding` | data, slow | IMPLEMENTED |
| The corpus caption lengths | Document 02 §8.2 | `dataset.captions_within_clip_context` | data | IMPLEMENTED |
| The dataset-scale extraction | Document 02 §14, Document 02 §14.3 | `pipeline.module_exists`, `pipeline.extract_full_split`, `pipeline.alignment_preserved`, `pipeline.deterministic`, `pipeline.cache_key_carries_manifest_hash`, `pipeline.no_pickle_in_cache` | data | DECLARED GAP |
| The corpus exclusion from the repository | Document 02 §4.2 | `docs.dataset_not_committed` | fast | IMPLEMENTED |
| The image mode | Document 04 §7.1 | `preprocess.rgb_mode` | slow | IMPLEMENTED |
| The EXIF orientation and the rejected image | Document 04 §7.2 | `preprocess.exif_applied`, `preprocess.corrupt_image_rejected` | slow | IMPLEMENTED |
| The alpha flattening | Document 04 §7.3 | `preprocess.alpha_white` | slow | DECLARED GAP |
| The downscale to 2048 px | Document 04 §7.4, Document 04 §7.4.1 | `preprocess.downscale_to_2048` (gap); `skew.downscale_is_noop_on_corpus` (implemented) | slow, data | MIXED |
| The tier taxonomy | Document 07 §2.1 | `harness.tier_taxonomy_matches_doc`, `harness.tier_selection_is_composable` | fast | IMPLEMENTED |
| The command set | Document 07 §2.2 | `harness.command_set_matches_doc`, `harness.filter_matches_the_case_id`, `harness.discovery_finds_every_collected_module` | fast | IMPLEMENTED |
| The marker discipline | Document 07 §2.3 | `harness.every_test_registered`, `harness.collection_clean` | fast | IMPLEMENTED |
| The interface and boundary matrix | Document 07 §11 | `docs.canonical_owners_present` | fast | IMPLEMENTED |
| This traceability table | Document 07 §12 | `harness.traceability_sections_resolve`, `harness.case_ids_unique` | fast | IMPLEMENTED |
| The documentation set | Document 00 §11, Document 00 §11.1 | `docs.every_document_present`, `docs.documentation_directory_is_exact`, `docs.overview_unchanged`, `docs.no_team_names_in_filenames` | fast | IMPLEMENTED |
| The pinned requirements | Document 00 §11.1, Document 01 §4.2 | `docs.requirements_pinned` | fast | DECLARED GAP |

### 12.3 The contracts not yet covered

These are specified, and they have no test yet. Each one names the file its test will live in.

| Contract | Document and section | Test file | Test | Status |
|---|---|---|---|---|
| Input validation rules and limits | Document 04 §6 | `tests/input_pipeline/test_input_pipeline.py` | Accepted, rejected, and over-limit inputs | PLANNED |
| The request schema and its invariants | Document 04 §9 | `tests/input_pipeline/test_input_pipeline.py` | Field presence and invariant checks | PLANNED |
| The status vocabulary and the no-fabrication rule | Document 04 §11 | `tests/input_pipeline/test_pipeline.py` | Every non-`OK` result has `None` values | PLANNED |
| The three quantities and the score-only mode | Document 04 §14 | `tests/input_pipeline/test_pipeline.py` | Score, probability, and calibration status | PLANNED |
| The decision rule and threshold source | Document 04 §14 | `tests/input_pipeline/test_pipeline.py` | The threshold comes only from the checkpoint | PLANNED |
| The degradation matrix | Document 04 §17 | `tests/input_pipeline/test_pipeline.py` | Every row of the matrix | PLANNED |
| The crop-retention calculation | Document 04 §23 | `tests/input_pipeline/test_warnings.py` | The retained-area formula and the threshold | PLANNED |
| The canonical explanation copy | Document 04 §24 | `tests/input_pipeline/test_pipeline.py` | The copy is used, and the fallback when absent | PLANNED |
| The leakage grouping feasibility and the decision rules | Document 02 §10 | `tests/datasets/test_leakage.py` | Component measurement and the selection rule | PLANNED |
| The leakage audit and the exposure fields | Document 02 §11 | `tests/datasets/test_leakage.py` | Every check in the audit | PLANNED |
| The classifier families and the feature construction | Document 03 §4 | `tests/clip_model/test_classifier.py` | Dimensions and parameter counts | PLANNED |
| The training protocol, the grid, and the seeding | Document 03 §7 | `tests/clip_model/test_training.py` | Determinism and the grid applicability | PLANNED |
| The metric calculation rules | Document 03 §8 | `tests/clip_model/test_metrics.py` | Hand-computed values, confusion-matrix orientation | PLANNED |
| The selection protocol and the tie-break | Document 03 §9 | `tests/clip_model/test_training.py` | The sealed split and the tie-break | PLANNED |
| The operating threshold and the calibration | Document 03 §10, Document 03 §12 | `tests/clip_model/test_classifier.py` | Threshold selection and the calibration | PLANNED |
| The checkpoint contents and the packaging gates | Document 03 §17, Document 03 §25 | `tests/clip_model/test_packaging.py` | Every gate | PLANNED |
| The loading contract and safe loading | Document 03 §27 | `tests/clip_model/test_checkpoint.py` | Strict load, `weights_only`, hash-before-deserialize | PLANNED |
| The verification procedure and its tolerance | Document 03 §28 | `tests/clip_model/test_checkpoint.py` | Class agreement and the score tolerance | PLANNED |
| The corpus terms enforcement | Document 06 §6 | `tests/retrieval/test_corpus_terms.py` | Refusal without a terms record | PLANNED |
| The ingestion safety controls | Document 06 §9 | `tests/retrieval/test_ingest_safety.py` | Redirects, size limits, private addresses | PLANNED |
| The three text representations | Document 06 §10 | `tests/retrieval/test_chunking.py` | Display, lexical, and dense strings | PLANNED |
| The chunk token budget | Document 06 §11 | `tests/retrieval/test_chunking.py` | The final embedded string fits the budget | PLANNED |
| The BM25 index metadata and rebuild equivalence | Document 06 §14 | `tests/retrieval/test_lexical.py` | Identical rankings after a rebuild | PLANNED |
| The filtering order and the absolute floor | Document 06 §19 | `tests/retrieval/test_filters.py` | The dense floor, the relative band, the caps | PLANNED |
| The counter-evidence retention | Document 06 §19 | `tests/retrieval/test_filters.py` | A relevant contradicting passage is retained | PLANNED |
| The empty-but-`OK` result | Document 06 §19 | `tests/retrieval/test_retriever.py` | Zero passages with status `OK` | PLANNED |
| The eligibility rule | Document 06 §28 | `tests/rag/test_eligibility.py` | Each of the five conditions | PLANNED |
| The citation and schema validation | Document 06 §29 | `tests/rag/test_validation.py` | Quote verification, URL scan, schema bounds | PLANNED |
| The per-component outcome computation | Document 06 §30 | `tests/rag/test_outcome.py` | Every row of the outcome table | PLANNED |
| The assessment per-status invariants | Document 06 §31 | `tests/rag/test_generation.py` | The invariants of the status table | PLANNED |
| The prompt-injection defences | Document 06 §33 | `tests/rag/test_injection.py` | The fixture cases and the fake-generator variant | PLANNED |
| The report model and the per-section status contract | Document 05 §2 | `tests/reporting/test_reporting.py` | The payload table and the invariants | PLANNED |
| The comparison matrix | Document 05 §8 | `tests/reporting/test_comparison.py` | Every combination of directions | PLANNED |
| The error and fallback matrix | Document 05 §10 | `tests/reporting/test_reporting.py` | All eight conditions | PLANNED |
| The escaping of untrusted text | Document 05 §16 | `tests/reporting/test_reporting.py` | Markdown, unusual Unicode, control sequences | PLANNED |
| The export contents and exclusions | Document 05 §15 | `tests/reporting/test_reporting.py` | Exactly the enumerated fields | PLANNED |
| The analysis fingerprint and the rerun rules | Document 05 §12 | `tests/reporting/test_reporting.py` | Display change, new input, configuration change | PLANNED |
| The end-to-end case manifest | Document 07 §8 | `tests/spec/test_e2e_cases.py` | The manifest schema and the case list | PLANNED |
| The availability rates | Document 07 §9 | `tests/spec/test_rates.py` | The denominators | PLANNED |
| The evaluation harness invariants | Document 07 §3 | `tests/spec/test_harness.py` | Each script refuses to run on a precondition failure | PLANNED |
| The clean-clone reproduction | Document 07 §13 | Manual, recorded | The procedure completes on a clean machine | PLANNED |
---

## 13. Clean-clone and offline reproduction

### 13.1 The three phases of offline capability

| Phase | Network | What happens |
|---|---|---|
| **1. Initial acquisition and environment setup** | Available | Dependencies are downloaded and the environment is prepared |
| **2. Offline execution** | **None** | The application runs with every artefact present locally |
| **3. Optional hosted inference** | Required | The optional hosted generation backend necessarily needs network access |

The local-first claim - "no key, no telemetry, no network at inference" - is **scoped to phase 2 under the default local-backend configuration**. The hosted backend cannot simultaneously be described as requiring no key.

### 13.2 The clean-clone procedure

```text
 1. Install the supported Python version; create a virtual environment.
 2. pip install -r requirements.txt          (pinned, verified)
    pip install pandas pytest               (the corpus tests and the runner)
 3. Fetch the CLIP encoder at the pinned revision.
    - the first slow-tier run downloads it; or pre-populate the Hugging Face cache -
 4. Fetch the corpus, completing the rights check, and extract it to dataset/ooc_dataset/.
    The upstream zip is group-shared. The extracted folder is what the code reads.
 5. Audit the corpus:           scripts/audit_dataset.py
    - writes reports/dataset_audit.json and fails loudly on a gate violation -
 6. Prepare the manifests:      scripts/prepare_dataset.py
 7. Extract features:           scripts/extract_features.py
 8. Train the classifier:       scripts/train_classifier.py
    - or download the released artifact and verify it -
 9. Build the evidence index:   scripts/build_evidence_index.py
10. Select the local generation model: scripts/benchmark_models.py
11. Run the application:        streamlit run app.py
12. Run the test suite:         python -m tests.run
```

Each step names the script, the expected artefact, and the check that confirms the step succeeded. A reader can verify the pipeline without reading the source.

**Steps 3 and 4 are the only ones that need a network, and only once.** After they are done, the test suite and the application run entirely offline. The corpus is required by the `data` tier only; the `fast` and `slow` tiers do not need it.

### 13.3 What the test suite needs

| Requirement | Needed by |
|---|---|
| `torch`, `transformers`, `Pillow`, `numpy` | Every tier |
| `pandas` | The `data` tier |
| `pytest` | The runner |
| The CLIP weights, in the Hugging Face cache | The `slow` tier |
| The corpus at `dataset/ooc_dataset/` | The `data` tier |

A tier whose requirement is absent skips with a stated reason. The suite is therefore runnable on a machine with none of them, and it reports the skips rather than hiding them.
### 13.4 The clean-machine verification

| Requirement | Detail |
|---|---|
| Performed by | Someone other than the author, ideally |
| Networking | **Disabled after phase 1**, and the fact recorded. Running on a machine that happens to have cached models is weaker evidence |
| Recorded | The platform, the phases completed, and the result |

### 13.5 The repository actions

| Action | Detail | Owner |
|---|---|---|
| Populate `app.py` | The Streamlit entry point | Document 05 |
| Populate `requirements.txt` | The pinned runtime dependencies | Document 07 |
| Create `models/README.md` | The download link and checksum, and the loading snippet | Document 03 |
| Create `CHANGELOG.md` | One entry per release | Document 03 |
| Create `data/manifests/README.md` | The manifest layout, including the evidence corpus manifest | Document 02 |
| Reconcile the documentation path | Document `00` §11.1 names `docs/` and `notebooks/`; the repository uses `doc/` and `notebools/`. Document `00` is unchanged, so the actual paths are recorded here | Document 07 |

---

## 14. The external dependency record

Each row is completed with measured values during implementation. **No placeholder may remain in the final report.**

| Field | Content |
|---|---|
| Identifier and revision | The exact, immutable revision or release |
| Acquired size and checksums | The **measured** size and a SHA-256, not an estimate |
| Licence and restrictions | Verified against the source |
| Download necessity | Mandatory · optional · unnecessary when cached |
| Needed for | Training · evaluation · inference, each |
| Manual steps | Any environment variables or manual configuration |

| Dependency | Identifier | Revision | Size | Checksum | Licence | Download | Needed for | Status |
|---|---|---|---|---|---|---|---|---|
| CLIP encoder | `openai/clip-vit-base-patch32` | **[PLANNED - NOT YET MEASURED]** | **[PLANNED - NOT YET MEASURED]** | **[PLANNED - NOT YET MEASURED]** | **[PLANNED - NOT YET MEASURED]** | Mandatory on first use | Training, evaluation, inference | PLANNED |
| Text embedder | `sentence-transformers/all-MiniLM-L6-v2` | **[PLANNED - NOT YET MEASURED]** | **[PLANNED - NOT YET MEASURED]** | **[PLANNED - NOT YET MEASURED]** | **[PLANNED - NOT YET MEASURED]** | Mandatory on first use | Evaluation, inference | PLANNED |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | **[PLANNED - NOT YET MEASURED]** | **[PLANNED - NOT YET MEASURED]** | **[PLANNED - NOT YET MEASURED]** | **[PLANNED - NOT YET MEASURED]** | Optional | Evaluation, inference | PLANNED |
| Local generation model | **[PLANNED - NOT YET MEASURED]** | **[PLANNED - NOT YET MEASURED]** | **[PLANNED - NOT YET MEASURED]** | **[PLANNED - NOT YET MEASURED]** | **[PLANNED - NOT YET MEASURED]** | Mandatory | Inference | PLANNED |
| Curated corpus | `dataset/ooc_dataset/` | Content hash of the three CSVs and `stats.json` | **[PLANNED - NOT YET MEASURED]** | **[PLANNED - NOT YET MEASURED]** | Research use only; not redistributable | Manual, group-shared; extract once | Training, evaluation | PLANNED |
| Upstream labels | `g-luo/news-clippings`, `merged_balanced` | **[PLANNED - NOT YET MEASURED]** | **[PLANNED - NOT YET MEASURED]** | **[PLANNED - NOT YET MEASURED]** | **[PLANNED - NOT YET MEASURED]** | Manual, once, into the corpus | Training, evaluation | PLANNED |
| Evidence corpus | Local, per-source | **[PLANNED - NOT YET MEASURED]** | **[PLANNED - NOT YET MEASURED]** | **[PLANNED - NOT YET MEASURED]** | Per-source terms, Document 06 §6 | Manual | Evaluation, inference | PLANNED |
| Classifier artifact | Our own release | **[PLANNED - NOT YET MEASURED]** | **[PLANNED - NOT YET MEASURED]** | **[PLANNED - NOT YET MEASURED]** | Our own | Manual | Inference | PLANNED |

Model sizes vary by revision, file format, and quantization, so the listed sizes remain estimates until measured.

---

## 15. The project caveat register

Each caveat travels with the result it qualifies. The per-component limitations remain in their own documents; this register is the consolidated project view.

| # | Caveat | Travels with | Canonical source |
|---|---|---|---|
| C-01 | The classifier learns from constructed out-of-context perturbations, not from a comprehensive real-world misinformation label | Every classifier metric | Document 03 §38 |
| C-02 | The official split configuration may contain cross-split leakage | Every classifier metric, with the four exposure fields | Document 02 §11.2.1 |
| C-03 | The grouped configuration is a leakage-aware arm, not proof that every source of leakage is removed | The split comparison | Document 03 §15 |
| C-04 | The local evidence corpus is small and may not cover many claims | Every retrieval and generation metric | Document 06 §45 |
| C-05 | Retrieval metrics depend on the relevance annotations and the selected query set. Strong scores on a small curated set do not establish broad coverage of news claims | Every retrieval metric | Document 06 §38 |
| C-06 | Semantic support depends on the model's classification of evidence and on a limited human evaluation | Every generation metric | Document 06 §39 |
| C-07 | Local model quality and cross-machine numerical variation constrain reproducibility | Every generation metric | Document 06 §36 |
| C-08 | Agreement between the classifier and the evidence assessment is not confirmation | Every presentation of the two signals | Document 05 §8 |
| C-09 | End-to-end availability is not factual accuracy. A system can render a report successfully while producing an inconclusive or semantically incorrect assessment | Every application-level rate | Document 07 §9 |
| C-10 | A zero observed error rate over N cases is a measurement over N cases, not a proof that the error cannot occur | The fabrication rate and every rate | Document 06 §39 |
| C-11 | The evaluation samples are small and the intervals are wide | Every rate and every proportion | Document 06 §39 |
| C-12 | The classifier is a similarity signal, not a truth judgment | Every presentation of the classifier result | Document 04 §2 |
| C-13 | The centre crop can discard image content, and long captions are truncated for the comparison model | Every classifier result | Document 04 §23 |
| C-14 | The official split is preserved rather than repaired, so a reported figure may be optimistic | Every classifier metric | Document 02 §9 |

---

## 16. The project success criteria mapping

Document `00` §14 defines ten success criteria. Each is mapped to the evidence that satisfies it. **A criterion is satisfied only by an executed test or a verified artefact**, never by a planned one.

| # | Criterion (Document `00` §14) | Satisfied by | Evidence | Status |
|---|---|---|---|---|
| 1 | A user can submit an image and caption through the interface | Document 04 §5, Document 05 §4 | The end-to-end case manifest, the functional cases | NOT YET RUN |
| 2 | The application validates and preprocesses the input consistently | Document 04 §6-Document 04 §9 | The input pipeline tests | NOT YET RUN |
| 3 | CLIP features are extracted using a documented model configuration | Document 01 §4, Document 01 §7, Document 01 §8 | The encoder contract tests | NOT YET RUN |
| 4 | A classifier trained on labeled data produces repeatable predictions | Document 03 §7-Document 03 §11 | The per-seed results and the determinism test | NOT YET RUN |
| 5 | The model checkpoint can be loaded on another computer | Document 03 §27-Document 03 §28 | The verification report | NOT YET RUN |
| 6 | The RAG system retrieves relevant evidence passages with traceable source metadata | Document 06 §21 | The retrieval metrics and the passage assertions | NOT YET RUN |
| 7 | The LLM generates an assessment that distinguishes support, contradiction, and insufficient evidence | Document 06 §30 | The outcome tests and the confusion matrix | NOT YET RUN |
| 8 | The interface presents the prediction, the retrieved sources, the explanation, and the limitations clearly | Document 05 §3, Document 05 §9 | The rendering tests and the manual copy review | NOT YET RUN |
| 9 | The team reports classification, retrieval, and grounding evaluations separately | This document §15, §18; Document 03 §8; Document 06 §38, Document 06 §39 | The results register | NOT YET RUN |
| 10 | The repository contains sufficient documentation to reproduce the major steps | This document §13 | The clean-machine verification record | NOT YET RUN |

---

## 17. Acceptance criteria and sign-off

### 17.1 The acceptance criteria for this document

These apply to **the protocol**, not to the results.

1. The tier taxonomy, the markers, the commands, the environment requirements, and the marker discipline are specified in §2.
2. The coverage gate is measured **per module**, at 90%, over the module set in §2.5, and the behavioural assertions are retained.
3. The evaluation harness invariants are specified in §3, and each evaluation script enforces them.
4. The three benchmarks are identified, owned, and given a status in §4. Only implemented benchmarks are reported as results.
5. The cross-component interface tests in §5 cover every producer-consumer boundary in §11.1.
6. The model-loading and prediction tests in §6 cover the loading contract, the offline modes, and the three-quantity semantics.
7. The degradation matrix in §7 covers every failure path, with the no-fabrication assertions.
8. The end-to-end case manifest is specified in §8, with the case record schema and the rules.
9. The six availability rates in §9 have explicit denominators, and the three distinctions in §9.1 are preserved.
10. The latency and resource protocol in §10 specifies the boundaries, the cold and warm separation, the run count, the hardware record, the resource metric, and the failure handling.
11. The interface and boundary matrix in §11 is the single canonical statement of the inter-component boundaries, and the interface-display requirement is preserved verbatim.
12. The traceability matrix in §12 covers every contract in Documents 01-06, with real test file and test names.
13. The clean-clone procedure, the three phases of offline capability, and the clean-machine verification are specified in §13.
14. The dependency record in §14 has a field for every required attribute, and no placeholder may remain in the final report.
15. The caveat register in §15 is complete and each caveat is tied to the results it qualifies.
16. The success criteria mapping in §16 covers all ten criteria of Document `00` §14.
17. The results register in §18 exists, and every row starts at `NOT YET RUN`.
18. The unresolved decisions in §19 are visible, and none has been silently resolved.

### 17.2 The sign-off checklist

- [ ] The fast suite passes with no model and no network access
- [ ] The component, evaluation, benchmark, and end-to-end suites have run, and their results are recorded
- [ ] Every metric in the results register has its required context
- [ ] The split configuration comparison is complete
- [ ] The caveat register is attached to the results
- [ ] The per-component limitations in Documents 01-06 are complete and unchanged
- [ ] The "what this system does not do" section of the final report is written
- [ ] The reproduction procedure has been followed **on a clean machine**, by someone other than the author, **with networking disabled after phase 1**
- [ ] The released artefact is published, and its checksum is recorded
- [ ] The external dependency table has no remaining placeholders
- [ ] The success criteria mapping is complete, and every criterion has executed evidence
- [ ] The manual copy review required by Document 05 §9.3 is recorded
- [ ] The unresolved decisions in §19 are either resolved by a recorded decision or still listed as open
- [ ] No combined system score appears anywhere
- [ ] Every row of the traceability matrix reads `IMPLEMENTED`

---

## 18. The results register

Every row starts at `NOT YET RUN`. No value may be entered that was not produced by an executed experiment.

| # | Measurement | Component | Source | Value | Status | Artefact |
|---|---|---|---|---|---|---|
| R-01 | Classifier accuracy | Classifier | Document 03 §8 | - | NOT YET RUN | `reports/classifier_test.json` |
| R-02 | Classifier macro-F1 | Classifier | Document 03 §8 | - | NOT YET RUN | `reports/classifier_test.json` |
| R-03 | Classifier per-class precision, recall, F1 | Classifier | Document 03 §8 | - | NOT YET RUN | `reports/classifier_test.json` |
| R-04 | Classifier confusion matrix | Classifier | Document 03 §8 | - | NOT YET RUN | `reports/classifier_test.json` |
| R-05 | Classifier ROC-AUC and average precision | Classifier | Document 03 §8 | - | NOT YET RUN | `reports/classifier_test.json` |
| R-06 | Per-seed mean and standard deviation | Classifier | Document 03 §11 | - | NOT YET RUN | `reports/classifier_test.json` |
| R-07 | Bootstrap confidence interval | Classifier | Document 03 §8 | - | NOT YET RUN | `reports/classifier_test.json` |
| R-08 | Official versus grouped split comparison | Classifier | Document 03 §15 | - | NOT YET RUN | `reports/classifier_test.json` |
| R-09 | Baseline comparison rows | Classifier | Document 03 §13 | - | NOT YET RUN | `reports/classifier_test.json` |
| R-10 | Retrieval Recall@k, Precision@k, MRR, nDCG@10 | Retrieval | Document 06 §38 | - | NOT YET RUN | `reports/retrieval_test.json` |
| R-11 | BM25-only, dense-only, hybrid comparison | Retrieval | Document 06 §38 | - | NOT YET RUN | `reports/retrieval_test.json` |
| R-12 | Citation validity | Generation | Document 06 §39 | - | NOT YET RUN | `reports/generation_test.json` |
| R-13 | Semantic support correctness | Generation | Document 06 §39 | - | NOT YET RUN | `reports/generation_test.json` |
| R-14 | Unsupported-claim rate and citation coverage | Generation | Document 06 §39 | - | NOT YET RUN | `reports/generation_test.json` |
| R-15 | Eligibility rate | Generation | Document 06 §39 | - | NOT YET RUN | `reports/generation_test.json` |
| R-16 | Outcome agreement, Cohen's kappa | Generation | Document 06 §39 | - | NOT YET RUN | `reports/generation_test.json` |
| R-17 | The 4x4 outcome confusion matrix | Generation | Document 06 §39 | - | NOT YET RUN | `reports/generation_test.json` |
| R-18 | Fabrication rate | Generation | Document 06 §39 | - | NOT YET RUN | `reports/generation_test.json` |
| R-19 | Injection suite results | Generation | Document 06 §33 | - | NOT YET RUN | `reports/generation_test.json` |
| R-20 | Report-render success rate | Application | This document §9 | - | NOT YET RUN | `reports/e2e.json` |
| R-21 | Input-rejection success rate | Application | This document §9 | - | NOT YET RUN | `reports/e2e.json` |
| R-22 | Assessment availability rate | Application | This document §9 | - | NOT YET RUN | `reports/e2e.json` |
| R-23 | Decision availability rate | Application | This document §9 | - | NOT YET RUN | `reports/e2e.json` |
| R-24 | Conditional decision availability | Application | This document §9 | - | NOT YET RUN | `reports/e2e.json` |
| R-25 | Degraded-path rate | Application | This document §9 | - | NOT YET RUN | `reports/e2e.json` |
| R-26 | Cold-start latency | Application | This document §10 | - | NOT YET RUN | `reports/latency.json` |
| R-27 | Warm latency, p50, p95, p99 | Application | This document §10 | - | NOT YET RUN | `reports/latency.json` |
| R-28 | Peak resident memory | Application | This document §10 | - | NOT YET RUN | `reports/latency.json` |
| R-29 | The RAG versus non-RAG comparison | Research | Document 06 §40 | - | NOT YET RUN | `reports/rag_vs_nonrag.json` |
| R-30 | Checkpoint verification result | Artefact | Document 03 §28 | - | NOT YET RUN | `verification.json` |

---

## 19. Unresolved decisions carried into implementation

| ID | Decision | Status | Where it is visible |
|---|---|---|---|
| UD-01 | The **non-RAG baseline** arms, the designation of a baseline, and whether the parametric-generation arm is in scope | **Requires confirmation** | Document 06 §40, with NB-01 to NB-05 |
| UD-02 | The **`model_proposed_outcome`** field. Document 06 §26.1 states that the output schema has no outcome field, while Document 06 §30.1 and Document 06 §39 define `model_proposed_outcome` as a recorded field. The generation contract is left unchanged and the inconsistency is visible | **Requires confirmation** | Document 06 §26.1, Document 06 §30.1, Document 06 §39 |
| UD-03 | **`FAILED` reachability.** `OK / UNAVAILABLE / SKIPPED / FAILED` is the shared status vocabulary, but the assessment and evidence sections may have no reachable `FAILED` state | **Requires confirmation** | Document 04 §11, Document 05 §2.2, Document 06 §31, this document §7 |
| UD-04 | Whether the **copy.json schema** is canonically owned by the checkpoint document or the inference document | Resolved in this consolidation: the schema and the fallback are canonical in **Document 04 §24**; Document 03 retains the packaging requirement and points to it | Document 03 §19, Document 04 §24 |
| UD-05 | The **documentation path** in Document `00` §11.1 (`docs/`, `notebooks/`) versus the repository (`doc/`, `notebools/`). Document `00` is unchanged | Recorded, not resolved | This document §13.4 |
| UD-06 | The **root `README.md`** does not point at the documentation. Updating it is outside `doc/` | Recorded | This document §13.5 |
| UD-07 | **Parity diagnostic scope.** The reference features cannot be compared on the image path, because the corpus photographs were re-encoded at 256 px while the reference used original-resolution files. The text path remains comparable, and a golden-vector regression is added as the primary offline check on both modalities | Resolved in this consolidation: the text path only, with the image path **excluded with its stated reason**, and the golden-vector regression covering the image path offline | Document 01 §12.8, §15.5 |
| UD-08 | **The `tests/` layout.** Document 00 §11 lists a flat `tests/` sketch (`test_input_pipeline.py`, `test_clip_encoder.py`, and so on). The suite instead mirrors `src/` — one folder per component, one file per module — because the flat sketch cannot distinguish `test_pipeline.py` from `test_feature_pipeline.py`, and because 21 more test files are already planned | Resolved in this consolidation: the suite mirrors `src/`, the folder rule is stated in this document §2.6, and document 00 is left untouched because it is byte-frozen | This document §2.6, §12.3 |

---

## 20. Immediate next step

Implement the test suite against this protocol, beginning with the fast tier, then the component tier, then the evaluation harnesses, then the end-to-end cases.

Fill in the measured values in the dependency record (§14) and the results register (§18) as the experiments are executed, and update the status columns. Do not enter a value that was not produced by a run.
