# 02 - Dataset Preparation

## Multimodal Out-of-Context Misinformation Detection Using Retrieval-Augmented Generation (RAG)

**Document status:** Accepted - implementation contract for Phase 3  
**Document type:** Component specification  
**Depends on:** `../00_PROJECT_OVERVIEW.md`, `01_CLIP_FEATURE_EXTRACTION.md`  
**Consolidated from:** `03_DATASET_PREPARATION.md`  
**Last updated:** 2026-10-21

---

## 1. Purpose and boundary

This document defines how the project turns a public dataset of labeled image–caption pairs into **reproducible, leakage-audited splits, versioned manifests, and cached CLIP features** that a classifier can be trained and evaluated on.

The output of this document is a set of artifacts:

| Artifact | Description |
|---|---|
| Label audit report | What the source annotations actually mean |
| Grouping feasibility report | Measured connected-component structure of the leakage graph |
| Leakage audit report | Every cross-split overlap found, measured and reported |
| Split manifests | One JSONL file per configuration and split |
| Feature caches | CLIP vectors for each manifest, under document 01's cache contract |

### 1.1 Explicit non-promises

| Non-promise | Why it matters |
|---|---|
| **A verified-clean dataset** | The official splits are audited, not assumed clean. Overlaps are reported, not hidden. |
| **A perfect label mapping** | The mapping is derived from the actual annotation files. Records that cannot be decided are `UNKNOWN`, not guessed. |
| **A leakage-free grouped split** | The grouped configuration is an experimental arm with measured residual leakage, not a guarantee. |
| **Published dataset sizes** | Counts are read from the downloaded artifacts and recorded. Published figures are treated as expectations, not facts. |
| **A redistributable dataset** | Images and annotations stay out of Git and out of distributed artifacts. |

### 1.2 Separation from the RAG evidence corpus

This document prepares **labeled image–caption pairs for classifier training**.

Document 06 prepares the **evidence corpus for retrieval**: articles, fact-checks, source metadata, chunking, and embeddings.

These are different pipelines with different storage, different manifests, and different purposes.

| | Labeled pairs (this document) | Evidence corpus (document 06) |
|---|---|---|
| Content | Image + caption + a class label | Article text + source metadata |
| Purpose | Train and evaluate the mismatch classifier | Retrieve passages for the grounded assessment |
| Output | Manifests, feature caches | Chunked documents, vector index |
| Label semantics | `MATCHED` / `MISMATCHED` / `UNKNOWN` | No labels; relevance is judged at retrieval time |
| Never used as | Evidence for an assessment | Classifier training data |

A labeled pair dataset is not a RAG knowledge base. No artifact produced here is treated as retrievable evidence.

---

## 2. Two kinds of data

Document 00 §6 identifies two distinct data requirements. This document covers only the first.

| Requirement | Contents | Owner |
|---|---|---|
| Labeled image–caption pairs | Image reference, caption text, class label, provenance | **This document** |
| Evidence documents | Article body, title, publisher, publication date, URL | Document 06 |

Consequences of keeping them separate:

- A classifier label and an evidence relevance judgment are different things and are never conflated.
- The retrieval index is built from curated source articles, not from dataset annotations.
- The classifier can be retrained without rebuilding the index, and the index can be rebuilt without retraining the classifier.

---

## 3. Roles of the candidate datasets

| Dataset | Role in this project | Status | Notes |
|---|---|---|---|
| **NewsCLIPpings** | Primary training and evaluation source | **Required** | Directly targets out-of-context image–caption mismatch; provides precomputed CLIP ViT-B/32 features for baseline comparison |
| **VERITE** | Independent, fine-grained evaluation | **Optional** | Three classes (`truthful`, `miscaptioned`, `out-of-context`); access may be restricted |
| **M4FC** | Real-world fact-checking and evidence | **Deferred** | More setup than the pipeline needs at this stage; relevant to the evidence side |

### 3.1 Why NewsCLIPpings is the primary

- It is built specifically for out-of-context image–caption detection.
- It provides both matched and generated mismatched examples for the same underlying images.
- Its annotations carry a binary falsification indicator, which gives a workable starting label.
- It publishes precomputed CLIP ViT-B/32 features, which allows a baseline comparison against our own extractor.

### 3.2 Why VERITE is separate and optional

VERITE distinguishes three failure modes rather than two. A binary classifier trained on NewsCLIPpings cannot natively report that distinction, so any binary evaluation against VERITE requires an explicit, justified mapping. The three-class labels are retained in the manifest so the distinction remains reportable.

VERITE access may require institutional verification, and some original image URLs are no longer live. The loader is therefore optional, and nothing in the primary pipeline depends on it.

### 3.3 Why M4FC is deferred

M4FC is most relevant to the evidence-oriented part of the project, which document 06 owns. It supplies image URLs rather than image files, requires more setup, and may include graphic content. It is revisited after the classifier and retrieval pipelines are stable.

---

## 4. NewsCLIPpings acquisition

### 4.1 Source

The project uses a **curated subset** of NewsCLIPpings, not the upstream release directly. The subset carries the upstream labels and fills in the caption text and photographs from VisualNews, so the rows are usable without a separate image corpus.

| Property | Value |
|---|---|
| Local path | `dataset/ooc_dataset/` (gitignored) |
| Label source | `https://huggingface.co/datasets/g-luo/news-clippings`, the `merged_balanced` split only |
| Text and photo source | VisualNews `data.json` - `https://github.com/FuxiaoLiu/VisualNews-Repository` |
| Paper | *NewsCLIPpings: Automatic Generation of Out-of-Context Multimodal Media* (EMNLP 2021) |
| Manifest storage | `data/manifests/<configuration>/` (gitignored) |
| Feature cache | `data/processed/features/<dataset_slug>/<model_slug>/<revision>/<extractor_version>/<split>/` (gitignored) |

#### Contents

| File | Content |
|---|---|
| `train.csv` | 25,000 rows - 12,500 pairs |
| `val.csv` | 7,024 rows - 3,512 pairs |
| `test.csv` | 7,264 rows - 3,632 pairs |
| `labels.json` | What `0` and `1` mean, and the rule that produced them |
| `stats.json` | Row counts, label counts, average caption length, and the construction settings |
| `images/` | 35,789 photographs, shorter side resized to 256 px, JPEG |

#### How it was built

1. The labels come from the upstream `merged_balanced` split, which mixes every way the mismatches were generated and has equal real and fake pairs. The upstream split carries identifiers only.
2. The caption text and the photo paths are joined in from VisualNews `data.json`.
3. Train is 12,500 pairs drawn at random with **seed 42**. Val and test are not subsampled.
4. The photographs live inside a 98 GB archive, so only the 35,789 that are referenced were pulled.
5. The photographs were resized to a shorter side of 256 px at JPEG quality 90. Any pair with a missing photograph or an empty caption was dropped.

#### The version identity

The subset has no upstream release tag of its own, so its version identity is **the three CSVs plus `stats.json`**, hashed. The hash is recorded in every manifest and in the feature cache metadata, so a changed corpus invalidates every derived artefact automatically.

### 4.2 Storage rules
## 5. Label verification gate

**No label mapping is implemented until this section's verification report exists.**

The subset in §4 already ships a computed `label` column, so the gate is no longer about discovering the field name. It is about **confirming that the shipped column is what it claims to be**, against the files themselves. A plausible label column that is wrong in some way nobody notices is still the most damaging failure mode in this project, and a shipped column is not evidence of correctness.

### 5.1 The checks

| # | Check | What it establishes |
|---|---|---|
| 1 | The CSV schema is exactly `image_path`, `caption`, `label`, `caption_id`, `image_id` | The rows are what the rest of this document assumes |
| 2 | `labels.json` states a rule, and the observed values agree with it | The label column means what the file says it means |
| 3 | Every split is **exactly** 50% `label` 0 and 50% `label` 1 | The balance is as documented, not approximately so |
| 4 | Every `caption_id` forms **exactly one pair**: two rows, one of each label | The pair structure is intact and the split is not cutting a pair |
| 5 | On every `label` 0 row, `caption_id == image_id` | The consistent rows really are the caption's own photograph |
| 6 | `caption_id` values are **disjoint across the three splits** | No caption is seen in two splits |
| 7 | `image_id` values are **disjoint across the three splits** | No photograph is seen in two splits |
| 8 | Every referenced `image_path` exists on disk | No row points at a missing file |
| 9 | Each `caption_id` has exactly one caption text, and each `image_id` exactly one `image_path` | The per-row hashes are well defined and stable |
| 10 | The photographs have a shorter side of 256 px | The pre-resize of §8.4 is present and recorded |
| 11 | The captions fit inside the CLIP context window | No caption needs truncation, so the corpus does not exercise that path |

### 5.2 Verification method

The check runs as a reproducible script, `scripts/audit_dataset.py`, over the local dataset. It produces:

| Output | Content |
|---|---|
| Schema report | The columns present, their types, and a sample of observed values |
| Label report | The value distribution, the `labels.json` rule, and whether the two agree |
| Balance report | Counts per class, per split, and overall |
| Pair report | The pair structure, and the label agreement within each pair |
| Split report | The overlap counts for both `caption_id` and `image_id`, per pair of splits |
| Integrity report | Missing files, and the one-text-per-id and one-path-per-id checks |
| Geometry report | The shorter-side distribution and the caption length distribution |
| Ceiling report | The shortcut ceilings of §X |
| Verdict | The gate's outcome, and the count of any rows quarantined |

The script is deterministic and re-runnable. It writes `reports/dataset_audit.json` and **fails loudly** when a check fails, so a corrupt or substituted corpus cannot be used unnoticed.

### 5.3 The gate

| Condition | Action |
|---|---|
| Every check passes | Proceed to implement the mapping in §6 |
| The schema differs from the expected one | Update §6.2 to the observed schema; do not force the expectation |
| The class balance is not exactly 50/50 | Record the observed counts, and state whether the imbalance is a property of the corpus or of a corruption |
| A `caption_id` does not form a clean pair | Report it, quarantine the affected rows, and state the count |
| The splits overlap | Report the overlap. The official split is never silently repaired - see §9.2 |
| A referenced file is missing | Quarantine the row, keep it in the manifest with its flag set, and report the count |
| A `caption_id` or `image_id` maps to two different texts or paths | Report it as an integrity failure and stop; the manifest's per-row hashes are not well defined without it |

### 5.4 Why this gate still exists

The upstream `merged_balanced` labels were re-sourced, re-joined against VisualNews, and re-split. That is exactly the kind of pipeline in which a column can be correct in principle and wrong in practice. The gate makes that failure mode visible before any metric is computed, and it costs one script.

---
|---|---|
| Field inventory | Every field name with its type and a sample of observed values |
| Value distribution | Count per distinct value of the falsification field |
| Class balance | Counts per class, per split, and overall |
| Worked examples | A small number of records quoted in full for each class, so the semantics can be read directly |
| Ambiguity report | Records with a missing, null, or internally inconsistent flag, with their identifiers |
| Duplicate report | Repeated records, both exact and by shared image |
| Coverage report | Which records appear in which provided split file |

The script is deterministic and re-runnable. Its output is archived as a project artifact under `data/processed/audits/` and summarized in this document once it has been run.

### 5.3 The gate

| Condition | Action |
|---|---|
| Verification report produced and reviewed | Proceed to implement the mapping in §6 |
| Field name differs from the expected one | Update the mapping table to the observed name; do not force the expectation |
| Value domain differs from boolean | Map every observed value explicitly; unrecognised values become `UNKNOWN` |
| A second manipulation field exists | Record it in the manifest as additional source metadata; decide whether it refines the binary label, and document the decision |
| Flag granularity is per image | Record the granularity in the manifest schema and adjust §10's grouping accordingly |
| Ambiguous records exist | Define the `UNKNOWN` rule from the observed ambiguity and record the count |
| Split files do not cover every record | Report uncovered records explicitly; assign them to no split, and state the count |

### 5.4 Why this gate exists

The most common failure mode in this project would not be a code bug. It would be a mapping that is plausible, runs without error, produces reasonable-looking accuracy, and is wrong in a way nobody notices. The gate makes that failure mode visible before any metric is computed.

---

## 6. Label mapping

The mapping is implemented **after** §5's report and is derived from it.

### 6.1 Project label scheme

```python
from enum import Enum

class Label(str, Enum):
    MATCHED = "MATCHED"
    MISMATCHED = "MISMATCHED"
    UNKNOWN = "UNKNOWN"
```

| Label | Meaning |
|---|---|
| `MATCHED` | The caption and the image are a genuine pairing according to the source annotation |
| `MISMATCHED` | The caption and the image are a deliberately falsified pairing according to the source annotation |
| `UNKNOWN` | The source annotation does not determine the class |

`UNKNOWN` is a real outcome, not a placeholder for "not yet implemented". Records carrying it stay in the manifest and are excluded from training by default. The count is reported for every configuration.

### 6.2 Mapping table

The source column is the subset's `label` column. The `labels.json` file states the rule directly - `falsified False -> 0, falsified True -> 1` - and §5's gate confirms that the observed values agree with it.

| Source field | Source value | Project label | Rationale |
|---|---|---|---|
| `label` | `0` | `MATCHED` | `labels.json`: "consistent". The caption and the photograph belong together |
| `label` | `1` | `MISMATCHED` | `labels.json`: "out_of_context". A real caption paired with a different real photograph |
| `label` | absent, null, or empty | `UNKNOWN` | Not decidable from the annotation |
| `label` | any value not listed above | `UNKNOWN` | Conservative. Counted and reported, never coerced |

**One source condition maps to exactly one project label.** The subset carries a single derived column, so there is nothing to decompose, and the upstream perturbation-type field is not exposed in the subset. That loss is recorded in §21.

### 6.3 Source labels are always retained

Every manifest record keeps the original annotation fields, so a project label can always be re-derived and a source-class result reported separately.

| Field | Purpose |
|---|---|
| `source_label` | The human-readable source form: `consistent` or `out_of_context` |
| `source_label_raw` | The value exactly as it appears in the CSV: the integer `0` or `1` |
| `source_extra` | The source-native identifiers, `caption_id` and `image_id` |
| `caption_id` | The VisualNews identifier of the caption. It is the **pair group key** (§10) |
| `image_id` | The VisualNews identifier of the photograph |

`caption_id` and `image_id` are promoted to first-class manifest fields rather than being buried in `source_extra`, because §10's grouping and §12's per-row hashes are both defined in terms of them.

### 6.4 Class balance

| Report | Content |
|---|---|
| Per split, per label | Record counts |
| Overall | Counts and ratios |
| Imbalance ratio | `max(count) / min(count)` over the decided labels |
| `UNKNOWN` count | Reported separately, never folded into either class |

**The corpus is exactly balanced.** Every split is 50% `MATCHED` and 50% `MISMATCHED`, so the imbalance ratio is **1.0** in every split, and the inverse-frequency class weights of document 03 §5.2 are therefore **all exactly 1.0**.

## 7. Image reference resolution

Every manifest record points at an image that must exist and decode.

### 7.1 Path construction

The subset's `image_path` is **already relative to the dataset root**, for example `images/2ebf6acf9bcc.jpg`. The resolution rule is therefore a single join:

```text
resolved_path = dataset_root / record.image_path
```

where `dataset_root` is `dataset/ooc_dataset/`. No path rewriting, no id-to-filename mapping, and no directory guessing are involved.

The path is stored in the manifest **relative to the dataset root**, not as an absolute path. A manifest containing absolute paths is broken on every other machine, which is the failure document `00` §10.5 explicitly warns against.

| Property | Rule |
|---|---|
| The stored path | Relative to the dataset root, exactly as the CSV gives it |
| The dataset root | Configuration, never hardcoded in the manifest |
| The identity of the image | Its SHA-256, not its path. Two rows may point at the same file |

### 7.2 Validation

| Check | Failure |
|---|---|
| The file exists at the resolved path | Record quarantined, counted |
| The file decodes and passes document 04's image checks | Record quarantined, counted |
| The image is within document 04's size limits | Record quarantined, counted |

Document 04's validation is reused, not reimplemented. A corpus image that the application would reject is excluded from training for the same reason it would be rejected at inference - otherwise the model trains on a distribution the user never produces.

## 8. Preprocessing parity

Consistency between training and inference is the single most important property in this project. A model trained on differently prepared images and served differently prepared images is not the model that was evaluated.

### 8.1 Images

Dataset images pass through the **same** document 04 normalization as user uploads:

```text
decode → EXIF orientation → RGB conversion → alpha flattening → downscale to 2048 px
```

then through document 01's processor:

```text
resize to 224 → center crop → tensor → mean/std normalization
```

No separate corpus loader resizes, crops, or normalizes. If document 04's normalization is not exposed as a reusable function, it is refactored to expose it - **without changing the upload contract in `04`**.

#### 8.1.1 The corpus arrives pre-resized

The corpus photographs were resized by the dataset author to a shorter side of **255–256 px** and re-encoded as JPEG at quality 90, before this project saw them. The 2048 px downscale in document 04 is therefore a **no-op** on every corpus image.

| Path | Chain | Difference |
|---|---|---|
| **Training** | 256 px shorter side → 2048 downscale (no-op) → processor 256 → 224 | Short resampling chain; JPEG-90 artefacts already baked in |
| **Inference** | original resolution → 2048 downscale → processor 2048 → 224 | Long resampling chain; sharper source |

Both paths end at the same 224×224 centre crop of the same relative field of view. They differ in the resampling chain and in the compression already applied to the corpus images.

#### 8.1.2 The skew is accepted, and measured

**This is accepted, not fixed.** Obtaining original-resolution photographs means the 98 GB VisualNews archive, which is not practical for this project.

Accepting it is only defensible if the difference is **measured** rather than argued. The measurement is part of the test suite:

| The measurement | What it records |
|---|---|
| The downscale is a no-op | A corpus image is unchanged by document 04's 2048 px step |
| The native-versus-upscaled embedding similarity | The same photograph encoded at its native 256 px and upscaled to the inference resolution, compared by cosine similarity |
| The equivalent crop region | Both variants yield the same relative 224×224 crop |
| The inference variant | The inference path matches the documented chain exactly |

The measured similarity is recorded in the dataset audit and in the final report. It is a **measurement, not a pass/fail assertion** - the specification is that the difference is small and known, and the number is what establishes that.

**Measured.** Over a sample of 20 photographs drawn from the three splits, the cosine similarity between a corpus photograph at its native 255-256 px and the same photograph upscaled to the inference resolution is:

| Statistic | Value |
|---|---|
| Mean cosine similarity | **0.9972** |
| Minimum | **0.9937** |
| Maximum | **0.9989** |
| Sample size | 20 |

So the accepted skew costs about **0.3%** of the embedding on average, and at worst **0.6%**. The measurement is regenerated by the test suite and written to `reports/skew_measurement.json`, so it stays honest as the encoder changes.

**Why the difference is small.** Document 04 §7.4.2 records the measurement that explains it: CLIP's processor resizes the shortest side to 224 px and centre-crops, and the relative crop region is the same whether or not the image was reduced first. The 2048 px step is therefore almost invisible in the feature, and the corpus's 255-256 px pre-resize is bounded by the same effect. The measurement is still worth taking - "small" is a number here, not an argument.

### 8.2 Captions

Corpus captions use the same text normalization as user input: Unicode NFKC, whitespace collapsing, and the same length rules as document 04 §8.

The corpus captions average 16.5 words, with a maximum of 30 words. **No corpus caption approaches the CLIP context window**, so the corpus never exercises the truncation path. That is worth stating, because it means a truncation defect would not be caught by training on this corpus alone - it would only be caught by a synthetic test, which §15 of document 01 requires.

The per-row `caption_sha256` is the hash of the **normalized** caption, so it is comparable with the one computed at inference.

### 8.3 What is explicitly not done

| Not done | Why |
|---|---|
| A corpus-specific resize or crop | Creates a second preprocessing path |
| Training-time augmentation | Document 03 owns it, and it must be consistent between training and inference |
| Per-corpus colour handling | Document 04's ICC policy applies to every image, corpus or user |
| Batch-level statistics | The processor's fixed mean and standard deviation are used, not corpus-derived ones |
| Re-deriving the corpus labels | The labels are verified in §5, not recomputed |

---
No separate dataset-specific loader resizes, crops, or normalizes. If document 04 exposes its normalization as a reusable function, the dataset builder calls that function. If it does not, it is refactored to expose it — **without changing the upload contract in `04`**.

The image SHA-256 recorded in the manifest is the same hash document 01's cache uses, computed from the same normalized pixels. This is what makes the per-row cache validation in `01` §11.6 meaningful.

### 8.2 Captions

Dataset captions use the **same document 04 text normalization** as user input. The `caption_sha256` in the manifest is the hash of the normalized caption, so a caption typed by a user and a caption read from an annotation produce comparable hashes.

### 8.3 What is explicitly not done

| Not done | Why |
|---|---|
| Dataset-specific resizing | Creates a second preprocessing path |
| Training-time augmentation | Augmentation belongs to document 03 if it is used at all, and it must be applied consistently |
| Per-dataset color handling | Document 04's ICC policy applies to all images, including dataset images |
| Batch-level statistics | The CLIP processor's fixed mean/std are used, not dataset-derived ones |

---

## 9. Split configurations

A **configuration** is the complete description of how a set of manifests was produced:

```text
configuration = (dataset, dataset_version, split_strategy, grouping_config, seed, manifest_version)
```

The official and grouped splits are **separate configurations**, not two options for the same thing. Each has its own manifests, its own cache namespace, and its own results.

### 9.1 Configurations

| Configuration | Definition | Role |
|---|---|---|
| `newsclippings-official-v1` | The release's own split files, used byte-for-byte | **Primary** — comparable with published work |
| `newsclippings-grouped-v1` | Our own split over leakage-safe groups, seed 42, 80/10/10 | **Experimental** — leak-controlled comparison |
| `newsclippings-dev-v1` | Seeded, class-balanced subsample of the official configuration | Fast iteration |

### 9.2 The official split is never silently repaired

This is a hard rule.

| Rule | Behaviour |
|---|---|
| Record assignment | Preserved exactly as the release provides it |
| Records moved between splits | **Never** |
| Records dropped to improve a ratio | **Never** |
| Overlap found by the audit | **Reported**, not acted upon |
| Splitting method changed | A new configuration, with a new name and its own artifacts |

The official configuration exists so that results are comparable with the published literature. A quietly "fixed" official split is neither comparable nor an audit — it is a third thing with the first one's name.

### 9.3 The grouped configuration is an experimental arm

The grouped split is a separate configuration reported alongside the official one, never as a replacement and never as a corrected version of it.

| Aspect | Official configuration | Grouped configuration |
|---|---|---|
| Assignment source | Release split files | Our seeded algorithm over groups |
| Leakage status | Audited, overlaps reported | Grouped, residual leakage measured |
| Primary role | Headline results | Leakage-sensitivity comparison |
| Reported as | "`newsclippings-official-v1`" | "`newsclippings-grouped-v1`" |

Every metric in documents 03 and 05 names the configuration it came from. A result without a configuration name is incomplete.

### 9.4 The dev configuration

A seeded, class-balanced subsample for fast iteration:

| Split | Target size | Source |
|---|---|---|
| train | 2,000 | Official configuration's train split, sampled per class |
| val | 500 | Official configuration's val split, sampled per class |
| test | 500 | Official configuration's test split, sampled per class |

The dev configuration is drawn **from the official configuration** so that dev results remain roughly comparable to full results. It is a compute-saving tool, not a separate experimental claim, and it is named in any report that uses it.

---

## 10. Leakage grouping and feasibility

A grouped split is only meaningful if the groups it splits on actually capture the leakage relationships. This section defines those relationships, and — importantly — requires that their **feasibility be measured before** a grouped split is built.

### 10.1 The leakage relationships

| Leak type | Description | Why it inflates results |
|---|---|---|
| **A. Same image, different captions** | An image appears in train with one caption and in test with another | The model can memorize the image's appearance, and the true caption is the most semantically similar one available |
| **B. Same caption, different images** | A caption appears in train with a non-matching image and in test with its true image | The model learns the caption→image association, which is exactly the test-time answer |
| **C. Near-duplicate images** | The same photo, resized or re-encoded, in both splits | A weaker version of A |
| **D. Same event, different images** | Different photos of the same incident | The weakest of these, and the most expensive to exclude |

Leak types A and B are the ones the grouping algorithm must address.

### 10.2 The bipartite leakage graph

Records are edges in a bipartite graph:

```text
image node  ──── record ────  caption-source node
```

An edge connects the image to the caption-source node of the record. A record's caption-source node is the image whose original caption the record's caption derives from, where the source provides that provenance.

Connected components of this graph are **leakage-safe groups**: no component may straddle a split boundary, because any two records in the same component share an image or a caption source.

### 10.3 The feasibility question

Union-find over image and caption-source links is transitive. Transitivity can produce a **giant component** — one component containing a large share of the dataset. If that happens, an 80/10/10 grouped split is not achievable while preserving the guarantee.

The size and structure of the components are therefore **measured before** the algorithm is committed to. No assumption is made that a reasonable split ratio is reachable.

### 10.4 Link set configurations

Grouping strength is configurable, because the right choice depends on a measurement that has not been taken yet.

| Link set | Links | Component definition | Feasibility |
|---|---|---|---|
| `L0` | Image only | One component per source image | Always feasible; components are bounded by the number of caption variants of one image |
| `L1` | Image + caption-source | Bipartite components (§10.2) | May produce a giant component |
| `L2` | Image + exact normalized caption-text duplicate | Components from image links and shared caption strings | Intermediate; caption-text links are exact and cheap to compute |

`L2` is included because caption-source provenance may be partial or inconsistently recorded, while exact caption-text duplication is always computable from the manifest itself.

### 10.5 The feasibility report

Produced by `scripts/audit_leakage_groups.py` before any grouped split is built.

| Measurement | Purpose |
|---|---|
| Component count per link set | Overall structure |
| Component size distribution — min, median, p90, p99, max | Whether a giant component exists |
| Largest component as a share of all records | The decisive figure |
| Share of records in components above the threshold | Blast radius |
| Edge counts by link type | Which relationship dominates |
| Achievable split ratios per link set | Whether 80/10/10 is reachable at all |
| Number of singleton components | Sanity check on the grouping logic |

The report is archived under `data/processed/audits/` and summarized in this document once produced.

### 10.6 Decision rules

| Condition | Action |
|---|---|
| Largest component ≤ 10% of records | Use the strongest link set that qualifies |
| `L1` exceeds the threshold but `L2` qualifies | Use `L2`; record that the stronger set was rejected and why |
| Only `L0` qualifies | Use `L0`; **report the residual caption-source leakage explicitly** and state that leak type B is not controlled |
| No link set qualifies | Do not build a grouped split; report why; the official configuration stands as primary |
| A component is too large to place in a target split | **Keep it intact in one split** and report the resulting ratio deviation |

**Records are never dropped to make a split hit a target ratio.** A split that contains 71/12/17 instead of 80/10/10 is a valid split with a documented deviation. A split that hit the ratio by discarding hard records is a biased split that looks clean.

### 10.7 Group key recording

Every split manifest records the grouping key, the link set, the seed, the achieved ratios, and the group statistics from the feasibility report. The grouping is reproducible from the recorded parameters.

---

## 11. Leakage detection and audit

The audit runs on **every** configuration, including the official one, and its output is a **report**.

### 11.1 Checks

| Check | Method | Cost |
|---|---|---|
| Exact image duplicate across splits | SHA-256 of the normalized image, dictionary lookup | O(N) |
| Near-duplicate image across splits | 64-bit pHash, Hamming distance ≤ 6 | O(N) with LSH shortlisting |
| Caption-source overlap across splits | Union-find component membership | O(N) |
| **Exact caption-text duplicate across splits** | Hash of the normalized caption | O(N) |
| Paired samples | Shared image or shared caption source | O(N) |
| Record duplication | Hash of the full record content | O(N) |

Two implementation notes that keep the audit tractable:

- **Near-duplicate detection over tens of thousands of images cannot be O(N²).** A 4-band LSH over the 64-bit pHash shortlists candidates before the exact Hamming comparison. The Hamming threshold is a configurable constant, recorded in the report.
- **Caption-text duplication is exact and cheap.** It catches the most common form of leak type B even when caption-source provenance is incomplete, so it is checked regardless of which link set was used.

### 11.2 The audit report

| Field | Content |
|---|---|
| Configuration and split | Which arms were compared |
| Link set and grouping parameters | Used to produce the grouping |
| Exact image duplicates | Count of straddling pairs, affected records per split |
| Near-duplicate images | Count, worst observed Hamming distance, affected records |
| Caption-source overlaps | Count of straddling pairs |
| Exact caption-text duplicates | Count of straddling pairs, affected records |
| Paired samples | Count |
| Exposure fields | The four fields defined in §11.2.1, reported separately |
| Interpretation | What the overlap implies for reading the configuration's metrics |

#### 11.2.1 Exposure field definitions

Exposure is reported as **four separate fields**, not as a single number. These are different forms of overlap with different consequences, and combining them without justification would hide which kind of leak is present.

| Field | Definition | Denominator |
|---|---|---|
| `exposure_exact_image_pct` | Percentage of test records whose image has a **byte-identical** match in the training split | All test records |
| `exposure_near_image_pct` | Percentage of test records whose image is a **near-duplicate** (pHash Hamming distance ≤ the configured threshold) of a training-split image | All test records |
| `exposure_shared_caption_source_pct` | Percentage of test records that **share an image or a caption-source link** with any training-split record, according to the grouping link set | All test records |
| `exposure_any_pct` | Percentage of test records affected by **at least one** of the three above | All test records |

Rules:

- The three component fields are always reported individually, including when one of them is zero.
- `exposure_any_pct` is a union, not an average, and it **over-counts** records that appear in more than one category. It is reported with that caveat attached.
- A record is "affected" if **at least one** training-split record matches it on the relevant relation. Pairs are counted separately from affected records; a record matching five training records is one affected record and five pairs.
- The percentage is computed over **all** test records, including quarantined ones, so the denominator matches the manifest's test row count. The quarantined count is reported alongside.
- The exposure fields are the **only** leakage figures downstream documents may cite. Document 03 consumes these values and does not compute its own.

**Why these stay separate.** Exact image duplication means the model can recognise the photograph. Near-duplication means the same, in a weaker form. Shared caption-source means the model can learn the caption-to-image association. A configuration with high `exposure_exact_image_pct` and low `exposure_shared_caption_source_pct` is a materially different problem from the reverse, and a single blended percentage would make the two indistinguishable.

### 11.3 The audit does not repair

**The audit reports. It does not modify.**

| Finding | Action |
|---|---|
| Cross-split overlap in the official configuration | Report it. The official assignment is unchanged. |
| Cross-split overlap in the grouped configuration | Report it. It means the grouping link set was insufficient — recorded, and it informs the next configuration. |
| A test-split exposure above a stated share | Report it prominently alongside any metric computed on that split, so the metric is read with the right caveat |

**A reported leak is more useful than a hidden one.** A silently repaired official split produces numbers that cannot be compared with anything and cannot be traced back to a known state. A reported leak produces numbers with a stated limitation attached, and a reader can judge them.

Document 03 and the final report must include the audit's exposure fields (§11.2.1) next to the accuracy or F1 score, so that no metric is presented without its leakage context. Where a single headline number is required by a template, the four fields are shown together rather than collapsed into one.

---

## 12. Manifest format

### 12.1 Format

**JSONL** — one JSON object per line, UTF-8, LF line endings.

| Property | Reason |
|---|---|
| One record per line | Streamable, so a large manifest never needs to be fully loaded to be filtered or counted |
| Line-delimited | Diffable in Git, so a manifest change is reviewable even for large files |
| No trailing structure | A truncated write cannot corrupt every subsequent record |
| Schema-versioned | A format change is explicit |

### 12.2 Record schema

```json
{
  "sample_id": "nclp_train_000001",
  "dataset": "ooc-newsclippings",
  "dataset_version": "<hash of the three CSVs and stats.json>",
  "source_record_id": "<row index within the source split>",
  "source_file": "train.csv",
  "configuration": "newsclippings-official-v1",
  "split": "train",
  "image_path": "images/2ebf6acf9bcc.jpg",
  "image_sha256": "<hex of normalized pixels>",
  "caption_raw": "…",
  "caption": "<normalized by document 04>",
  "caption_sha256": "<hex of the normalized caption>",
  "label": "MISMATCHED",
  "source_label": "out_of_context",
  "source_label_raw": 1,
  "source_extra": {},
  "caption_id": "93574",
  "image_id": "243354",
  "pair_group_id": "93574",
  "event_id": "93574",
  "flags": {"missing_image": false, "quarantine_reason": null},
  "manifest_version": 1
}
```

| Field | Purpose |
|---|---|
| `sample_id` | Stable, unique within a configuration. Used by caches and by document 03 |
| `dataset`, `dataset_version` | The corpus identity and its hash |
| `source_record_id` | The row index within the source CSV, so any row can be traced back |
| `source_file` | The CSV the row came from |
| `configuration` | The configuration that produced this record |
| `image_path` | Relative to the dataset root, exactly as the CSV gives it (§7.1) |
| `image_sha256` | Hash of the normalized pixels - the same hash the feature cache validates against |
| `caption` | The normalized caption, as the classifier and the retriever will see it |
| `caption_sha256` | Hash of the normalized caption |
| `label` | The project label |
| `source_label`, `source_label_raw`, `source_extra` | Original annotation, always retained (§6.3) |
| `caption_id` | The source-native caption identifier. It is the **pair group key** |
| `image_id` | The source-native photograph identifier |
| `pair_group_id` | The pair this row belongs to. Equal to its `caption_id` |
| `event_id` | The grouping key this record belongs to. Equal to its `pair_group_id` for this corpus |
| `flags` | Quarantine and other per-record state |
| `manifest_version` | The manifest schema version |

**Why the pair group and the event are the same thing here.** The upstream corpus pairs each caption with exactly one photograph, so `caption_id` **is** the event identifier for this dataset. `pair_group_id` and `event_id` are kept as separate fields because the grouped configuration in §10 groups on both image and caption-source links, and a future corpus may separate them. On this corpus they coincide, and §10 says so rather than leaving a reader to assume it.
### 12.3 Storage layout

```text
data/manifests/
    README.md
    newsclippings-official-v1/
        train.jsonl
        val.jsonl
        test.jsonl
        manifest_meta.json
    newsclippings-grouped-v1/
        train.jsonl
        val.jsonl
        test.jsonl
        manifest_meta.json
    newsclippings-dev-v1/
        train.jsonl
        val.jsonl
        test.jsonl
        manifest_meta.json
```

**`data/manifests/` is excluded by `.gitignore`**, with `data/manifests/README.md` as the only tracked file beneath it. The manifests contain dataset captions, so they inherit the corpus's redistribution restriction (§4.3) even though they contain no pixels.
### 12.4 `manifest_meta.json`

Written alongside the split files, recording everything needed to reproduce them:

```json
{
  "manifest_schema_version": 1,
  "configuration": "newsclippings-official-v1",
  "dataset": "newsclippings",
  "dataset_version": "<release tag or commit>",
  "source_url": "https://github.com/g-luo/news_clippings",
  "accessed_at": "<date>",
  "split_strategy": "official",
  "grouping": {"link_set": null, "group_key": "image_id", "seed": null},
  "target_ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
  "achieved_ratios": {"train": 0.83, "val": 0.09, "test": 0.08},
  "counts": {"train": 0, "val": 0, "test": 0},
  "label_counts": {"MATCHED": 0, "MISMATCHED": 0, "UNKNOWN": 0},
  "quarantined": {"train": 0, "val": 0, "test": 0},
  "label_mapping_version": 1,
  "preprocessing_version": 1,
  "feature_extractor_version": 1,
  "manifest_sha256": {"train": "<hex>", "val": "<hex>", "test": "<hex>"},
  "audit_reports": {
    "labels": "data/processed/audits/labels_<...>.json",
    "groups": "data/processed/audits/groups_<...>.json",
    "leakage": "data/processed/audits/leakage_<...>.json"
  },
  "created_at": "<timestamp>"
}
```

The per-split `manifest_sha256` is the hash of the split file's bytes. Any manual edit to a manifest is therefore detectable, and a manifest whose hash does not match the metadata is rejected by the loader.

### 12.5 Stable sample IDs

| Requirement | Rule |
|---|---|
| Uniqueness | Unique within a configuration |
| Stability | The same record receives the same ID across regenerations of the same configuration |
| Determinism | Derived from the configuration name, split, and the record's index in a deterministic ordering |
| Cross-configuration | The `source_record_id` links the same underlying record across configurations |

Regenerating a manifest from unchanged inputs must produce byte-identical output. IDs are therefore not random and not based on a run timestamp.

### 12.6 Integrity checks on load

| Check | Failure |
|---|---|
| `manifest_schema_version` supported | Reject |
| `manifest_sha256` matches the file bytes | Reject |
| `row_count` matches the metadata counts | Reject |
| `sample_id` unique within the split | Reject |
| `label` is a known label | Reject |
| `caption` equals the normalization of `caption_raw` | Reject |
| `caption_sha256` matches `caption` | Reject |
| `image_sha256` matches the referenced image's normalized pixels | Reject |
| `caption_id` and `image_id` are present and non-empty | Reject |
| `pair_group_id` equals the record's `caption_id` | Reject |
| `caption_id` appears exactly twice in its split, with one row of each label | Reject the split |
| `configuration` matches the directory | Reject |

The caption checks make the manifest self-verifying: if the normalization in document 04 changes without a version bump, the manifest fails to load rather than silently mixing normalization states.

The identifier checks are what make the manifest usable for training at all. They enforce the pair structure of §5 directly on the artefact, so a manifest that was built from a corrupted or substituted corpus is rejected at load rather than at training time.

## 13. Class balance and metadata

The manifest supports the analysis document 03 needs, without making the decisions.

| Reported | Detail |
|---|---|
| Per split, per label | Record counts |
| Overall | Counts and ratios |
| Imbalance ratio | `max / min` over the decided labels |
| `UNKNOWN` | Count, reported separately |
| Quarantined | Count per split, with reasons |
| Sub-sampling feasibility | Whether each class has enough records for the dev configuration |

**The corpus is exactly balanced** (§6.4), so the imbalance ratio is 1.0 in every split and the inverse-frequency weights of document 03 §5.2 are all exactly 1.0. The sub-sampling feasibility check is therefore always satisfied.

The imbalance ratio matters for more than reporting. If one class has a very small count, the grouped split may not be able to place that class across three splits at all, which interacts with §10.6's decision rules and is recorded in the feasibility report.

### 13.1 How much information the label carries

The corpus has a structure that makes the strength of any classifier measurable **in advance**, rather than only after the fact. Two shortcut ceilings follow from it, and both are computed by `scripts/audit_dataset.py` and recorded in the dataset audit.

| Probe | Ceiling | Why |
|---|---|---|
| **Caption only** | **Exactly 50%** | Every `caption_id` appears in **two** rows, one of each label. The caption is therefore identical for a matched and a mismatched pair, and carries **no** information about the label. Verified in §5: no `caption_id` has a group size other than two, and no group has two rows of the same label |
| **Image only, within a split** | **94.8%** on train, 84.6% on val and test | A predictor that memorises which photographs it has seen can do this well. On train, 11,199 photographs appear only with label 0 and 11,199 only with label 1, while 1,301 appear with both labels. Predicting the observed label for the unambiguous photographs and either label for the rest scores `(11,199 + 11,199 + 1,301) / 25,000` |
| **Image only, across splits** | **Approximately 50%** | The photographs are disjoint across splits, and the marginal label distribution of photographs is balanced by construction, so nothing generalises from the training photographs to the test photographs |

#### What this means for the evaluation

| Consequence | Statement |
|---|---|
| The task genuinely requires matching | The caption alone is provably uninformative, so a model scoring well is doing image–caption matching rather than reading the answer off the text |
| **Training accuracy is not informative** | A model that memorised training photographs would reach 94.8% without understanding anything. Only val and test accuracy is meaningful |
| The image-only probe needs a number to read against | Document 03 §12.3's probe is not just a diagnostic here - it is measured against a known 94.8% ceiling, so an image-only score near 85% is a memorisation artefact rather than a semantic result |
| A constant predictor scores 50% | Accuracy alone is uninformative on a balanced corpus, so document 03's metric set is reported in full |

The ceilings are properties of the corpus, not of the model, and they hold for any classifier trained on it. They are recorded here so that a later reader can interpret the classifier's numbers correctly rather than mistaking a memorisation artefact for a result.

---

## 14. Dataset feature extraction

### 14.1 Encoder

Features are produced by this project's own extractor, per document 01 §12.6.

| Source | Used for |
|---|---|
| **Our extractor** | Every configuration's training and evaluation features |
| NewsCLIPpings reference embeddings | **Baseline comparison only**, in a separate cache namespace, never mixed into a training set |

The reason is a training/inference consistency guarantee. A classifier trained on reference features and served features from our extractor is a model whose feature distribution differs between training and serving, and no downstream tuning reliably repairs that.

### 14.2 Cache layout

```text
data/processed/features/<dataset_slug>/<model_slug>/<revision>/<extractor_version>/<split>/
```

The configuration is not in the path, because the **features do not depend on the configuration** - the same photograph and caption produce the same vector whichever split they were assigned to. The dataset slug and the extractor version are, so features from a different corpus or a different encoder can never share a cache directory by accident. The configuration is recorded in the cache metadata, and the manifest hash in the cache key (§11.3) ties the cache to the exact rows it was built from.

### 14.3 Process

1. Load the manifest for the split.
2. For each record with a usable image, load the image and apply document 04's normalization.
3. Batch the normalized images and call document 01's `encode_images`.
4. Separately call `encode_texts` on the normalized captions.
5. Write the cache with document 01's atomic procedure, populating the per-row `image_sha256` and `caption_hashes` files from the manifest, **and the `caption_id` and `image_id` arrays**.

**The cache is written in manifest row order, which is the source CSV's row order.** The order is therefore deterministic and the rows line up with the CSVs by construction, without a join. `caption_id` and `image_id` are stored alongside the vectors precisely so that alignment is checkable rather than assumed.

Because the manifest's hashes are exactly what document 01 §11.6 validates, a cache is provably built for that manifest's records. Because the identifiers are stored, a reader can confirm the alignment without recomputing anything.

**The corpus's CSVs are in pair order**, with a matched row immediately followed by its mismatched row. The cache preserves that order for reproducibility; **the training sampler shuffles** it, with the seed recorded in document 03 §6.4. Shuffling belongs to the sampler, not to the feature store, so the features stay aligned with the CSVs.

---
### 14.4 Quarantined records

Records with `flags.missing_image = true` have no image to encode and are not written to the cache. The cache's `row_count` is the number of encodable records, and the manifest's quarantined count is reported in the metadata so the two can be reconciled.

### 14.5 Progress and resumption

Extraction over tens of thousands of images takes minutes. The script prints progress, and an interrupted run leaves no partial cache — document 01 §11.5's atomic write means a rerun starts clean rather than resuming from a corrupt state.

### 14.6 Reference-feature comparison

When the NewsCLIPpings reference embeddings are available, they are loaded into a **separate** cache namespace for the baseline comparison described in document 03. This comparison is a diagnostic about the reference data, not part of the final model's data path.

---

## 15. VERITE as an independent benchmark

### 15.1 Access

| Item | Requirement |
|---|---|
| Repository | `https://github.com/stevejpapad/image-text-verification` |
| Dataset | Hosted on Hugging Face, potentially with institutional-email verification |
| Images | Some original URLs are no longer live; the current distribution path may require access |
| Licence | Verify the current terms before use |

Access is a prerequisite, not a given. The loader is **optional** and the primary pipeline does not depend on it. If access is not obtained, the project proceeds with NewsCLIPpings only and says so in the final report.

### 15.2 Three-class retention

VERITE distinguishes `truthful`, `miscaptioned`, and `out-of-context`. A binary classifier cannot natively report that distinction.

The rule, per document 00 §6.2: **do not silently merge labels.** VERITE records therefore carry:

| Field | Purpose |
|---|---|
| `label` | The project binary label, derived by the mapping below |
| `source_label` | The original three-class label, always retained |
| `source_label_raw` | The raw value from the annotation file |

The three-class result is reported **separately** wherever the model can support it, and the binary mapping is stated explicitly with its justification.

### 15.3 Binary mapping

| VERITE class | Project label | Justification |
|---|---|---|
| `truthful` | `MATCHED` | The pairing is genuine |
| `miscaptioned` | `MISMATCHED` | The caption is wrong for this image |
| `out-of-context` | `MISMATCHED` | The caption belongs to a different image or event |

The mapping merges two distinct error types into one. The cost of the merge is stated: a binary evaluation on VERITE cannot distinguish a miscaption from an out-of-context swap, and the three-class breakdown is reported separately to recover that distinction. The mapping is also stated in the results, never left implicit.

### 15.4 Evaluation protocol

| Rule | Reason |
|---|---|
| VERITE is **never** used for training in the primary configuration | It is an independent benchmark, not a training source |
| **No tuning on the benchmark's test examples** | Tuning on test and then reporting that test is not an independent evaluation |
| No model selection on the benchmark | Same reason |
| Reported with the binary mapping stated | So a reader can reconstruct what was measured |
| Reported separately from NewsCLIPpings metrics | Different class distributions and construction |

---

## 16. M4FC

### 16.1 Deferral

M4FC is deferred to after the classifier and retrieval pipelines are stable.

| Reason | Detail |
|---|---|
| Belongs to the evidence side | Its main relevance is real-world claims and evidence resources — document 06's domain |
| Different distribution | Real-world claims differ from constructed out-of-context perturbations |
| More setup | Image URLs rather than image files, collection instructions to follow |
| Content warning | Some images may be graphic; the repository carries a content warning that must be reviewed first |

### 16.2 Prerequisites before it is attempted

| Prerequisite | Detail |
|---|---|
| Stable primary pipeline | NewsCLIPpings training and evaluation complete and documented |
| Verified access terms | Repository and dataset licences re-checked at the time of use |
| Content review | The collection instructions and content warning reviewed |
| Label semantics verified | As in §5, for whichever task subset is used |
| Task selection | The six fact-checking tasks are not all suited to a mismatch classifier; the chosen subset is justified |

### 16.3 What it would add

Real-world claims with real-world evidence, and a bridge between the classifier and the evidence side of the project. It is an extension, not a requirement.

---

## 17. Reproducibility record

Document 00 §5.2 sets the minimum record. This document satisfies it and adds the audit artefacts.

| Item | Source |
|---|---|
| Dataset name | `ooc-newsclippings` |
| Dataset version | The SHA-256 of the three CSVs and `stats.json`, recorded in `manifest_meta.json` |
| Local path | `dataset/ooc_dataset/` |
| Label source | `https://huggingface.co/datasets/g-luo/news-clippings`, the `merged_balanced` split |
| Text and photo source | VisualNews `data.json` |
| Access date | `accessed_at` |
| Corpus construction | The upstream `merged_balanced` labels joined to VisualNews text and photographs; train subsampled to 12,500 pairs at **seed 42**; val and test not subsampled; photographs resized to a 256 px shorter side at JPEG quality 90 |
| Exact split-generation method | `split_strategy`, `grouping`, `target_ratios` |
| Random seed | 42 for every configuration we generate |
| CLIP model identifier and preprocessing | Carried from document 01 in the feature cache metadata |
| Label mapping and class weighting | `label_mapping_version` and the mapping table; weighting is document 03's |
| Label verification report | `audit_reports.labels` |
| Grouping feasibility report | `audit_reports.groups` |
| Leakage audit report | `audit_reports.leakage` |
| Shortcut ceilings | `audit_reports.ceilings` |
| Quarantined record counts | `quarantined` |
| Known limitations | This document, §21 |

The record is reproducible: given the same corpus, the same code, and the same recorded parameters, the manifests regenerate byte-identically. The corpus itself is identified by content hash rather than by a release tag, so a substituted or modified corpus is detected rather than silently used.

---
## 18. Testing strategy

### 18.1 Test tiers and commands

The test tier taxonomy, the marker definitions, the environment requirements, the complete command set, the coverage gate, and the evaluation-harness invariants are specified centrally in `07_INTEGRATION_TESTING.md` §2 and §3.

The tests listed in the remaining subsections of this section are this component's own tests, executed under those tiers. They are **unit and component tests**: they verify this component's contracts, not the boundaries between components, which are tested in `07_INTEGRATION_TESTING.md` §5.

```bash
# This component's fast suite
pytest -m "not slow and not eval and not benchmark and not e2e"
```
### 18.2 Fast tests

Use synthetic manifests and small synthetic images.

| Test | Assertion |
|---|---|
| Manifest round-trip | Write then read a manifest; every field survives |
| Manifest determinism | Regenerating from the same inputs produces byte-identical output |
| Manifest hash | `manifest_sha256` detects a single-character edit |
| Caption integrity | A manifest whose `caption` is not the normalization of `caption_raw` is rejected |
| Label vocabulary | An unknown label string is rejected |
| Duplicate sample IDs | Rejected on load |
| Quarantine flag | A quarantined record is retained in the manifest and excluded from extraction |
| Split group integrity | No group ID appears in two splits of the grouped configuration |
| Split ratio reporting | Achieved ratios are recorded and may deviate without error |
| Union-find correctness | On a synthetic graph, components match a brute-force connected-components computation |
| LSH shortlisting | Shortlisting never omits a pair within the Hamming threshold |
| Hamming threshold | A pHash pair at distance 5 is flagged, at distance 7 is not |
| Config isolation | Manifests and caches for two configurations cannot collide |

### 18.3 Slow tests

| Test | Assertion |
|---|---|
| Label audit runs | The script produces a report on the real annotations |
| Official split preserved | Regenerating the official configuration reproduces the release's assignment exactly |
| Grouping feasibility | The report is produced and its largest-component figure is recorded |
| Leakage audit | The report is produced and lists every check in §11.1 |
| Manifest integrity | Every manifest record's `image_sha256` matches the referenced image |
| Feature extraction | Cached features exist for every encodable record, with matching row counts |
| Cache validation | Document 01's per-row hash check passes against the manifest |
| Real leakage sample | On a small real subset, the audit detects at least one known cross-split image duplication if the dataset has one |

### 18.4 Audits as tests

The three audit scripts are deterministic and re-runnable. Their outputs are archived, and a test asserts that the reports exist, are parseable, and contain every required field — so an audit cannot silently stop running.

---

## 19. Module layout and scripts

### 19.1 Modules

```text
src/datasets/
├── __init__.py
├── schemas.py             # Label, ManifestRecord, SplitConfiguration
├── labels.py              # verified source → project label mapping
├── newsclippings.py       # annotation reader, path resolution
├── verite.py              # optional VERITE reader
├── splits.py              # configuration builders: official, grouped, dev
├── grouping.py            # link sets, union-find, component measurement
├── leakage.py             # duplicate, near-duplicate, caption-text checks
├── manifests.py           # JSONL read/write, hashing, integrity checks
└── extraction.py          # manifest → feature cache, using document 01
```

`src/clip_model/dataset.py`, listed in document 00's structure, holds the **torch Dataset** that document 03 uses to read manifests and cached features. It is the training-side reader and is specified in document 03. The preparation logic lives in `src/datasets/`.

### 19.2 Scripts

```text
scripts/
├── audit_dataset.py             # §5 label gate, split structure, and the §13.1 ceilings
├── audit_leakage_groups.py      # §10 feasibility report
├── audit_leakage.py             # §11 leakage audit report
├── prepare_dataset.py           # validate → map → split → write manifests
└── extract_features.py          # manifest → feature cache
```

`scripts/audit_dataset.py` exists today. It runs the checks of §5 against the local corpus, verifies the split structure, computes the shortcut ceilings of §13.1, writes `reports/dataset_audit.json`, and exits non-zero when a check fails, so a corrupt or substituted corpus cannot be used unnoticed.

All scripts are re-runnable and deterministic, and accept `--config` and `--output` arguments so results are written to a known location rather than wherever the run happened to be.

---

## 20. Acceptance criteria

Phase 3 is complete when:

1. The label verification report (§5) has been produced, reviewed, and its findings are recorded in this document.
2. The label mapping table (§6.2) is derived from the observed annotation semantics, and every source value maps to exactly one project label.
3. `UNKNOWN` is defined from observed ambiguity, and its count is reported per configuration.
4. Source labels are retained in every manifest record, so multi-class reporting remains possible.
5. The grouping feasibility report (§10.5) has been produced, and the largest-component figure is recorded in this document.
6. The grouping link set was **chosen by measurement**, and the decision rule in §10.6 is applied and recorded.
7. If no link set qualified, no grouped split was built, and the reason is recorded.
8. The official configuration's record assignment is byte-for-byte the release's own; no record was moved or dropped to change a split.
9. The leakage audit has been run on every configuration, including the official one, and every check in §11.1 is present in the report.
10. The audit reports cross-split overlaps and does not repair them.
11. The four exposure fields in §11.2.1 are available to accompany any metric computed on that split.
12. Manifests are JSONL, one file per configuration and split, with `manifest_meta.json` recording the full reproducibility record.
13. Manifests are deterministic — regeneration from unchanged inputs produces byte-identical files.
14. All nine integrity checks in §12.6 are enforced on load.
15. Sample IDs are stable, unique within a configuration, and shared `source_record_id` values link records across configurations.
16. Dataset images and captions pass through document 04's normalization, with no separate dataset-specific preprocessing path.
17. Features are extracted by this project's own extractor for every configuration, and reference embeddings are never mixed into a training set.
18. Each configuration has its own cache namespace, and the caches pass document 01's per-row hash validation against their manifests.
19. Quarantined records are retained in the manifest, excluded from extraction, and counted per split.
20. The VERITE loader is optional, the primary pipeline does not depend on it, and the three-class labels are retained with the binary mapping stated explicitly.
21. No dataset content is committed to Git, and the rights check in §4.3 is recorded.
22. The fast test suite passes with no dataset, no weights, and no network access.

---

## 21. Known limitations

| Limitation | Consequence | Status |
|---|---|---|
| The official split may contain cross-split leakage | Metrics on the official configuration may be optimistic | Audited and reported; the official split is never silently repaired |
| The grouped configuration is an experimental arm | Headline results come from the configuration with the known leakage caveat | Both are reported side by side |
| `L0` grouping does not control caption-source leakage | Residual leak type B remains | Only used if the stronger link sets are infeasible; stated explicitly |
| A giant component may force a large ratio deviation | The grouped split may not be 80/10/10 | Kept intact and reported; no records dropped |
| Dataset images may be partially unavailable | Some records are quarantined | Counted per split and reported |
| Label semantics depend on the release | A different release may map differently | The verification gate and the recorded mapping version handle this |
| The binary label collapses two error types | The classifier cannot distinguish a miscaption from an out-of-context swap | Accepted for v1; the source label is retained for future multi-class work |
| Class imbalance may be substantial | Accuracy may be a misleading headline metric | Counts and imbalance ratio are reported; document 03 chooses metrics |
| VERITE may not be accessible | The independent benchmark may be unavailable | The loader is optional and the dependency is one-way |
| Published dataset counts may differ from the artifacts | External expectations may not match | Actual counts are read from the artifacts and recorded |
| Dataset rights may prohibit redistribution | The project cannot ship the data | Data stays local; manifests and caches are reviewed separately |
| Manifest regeneration depends on code and release | A code change alters the manifests | Version fields and hashes detect it; manifests are regenerable from the source data |

---

## 22. Interface summary

| Direction | Detail |
|---|---|
| Provides | The split manifests, the split configurations, the label mapping, the leakage audit report, the four exposure fields, the corpus manifest, and the reproducibility record |
| Receives | The source dataset under the recorded terms, and the encoder and feature-cache contract from `01_CLIP_FEATURE_EXTRACTION.md` |
| Owns | The label verification gate, the label mapping, the split configurations, the leakage grouping and its feasibility measurement, the leakage audit, the manifest schema and its integrity checks, and the dataset feature-extraction run |
| Must not | Modify a manifest after it is written, silently repair the official split, move a record between splits to improve a ratio, redistribute dataset content, or train the classifier on the VERITE test examples |

The complete cross-component interface matrix, and the prohibitions that apply to every component, are specified in `07_INTEGRATION_TESTING.md` §11.

## 23. Immediate next step

Implement `src/datasets/` and `scripts/` against this contract, beginning with the label verification audit (§5) and the grouping feasibility audit (§10.5), because both gate later work. Then build the manifests, then extract features.

Fill in the measured values during implementation — the label mapping table, the largest-component figure, the achieved split ratios, the record counts, and the audit headline numbers — and update the corresponding sections of this document.

The next specification after this one is `03_CLASSIFIER_TRAINING_AND_CHECKPOINTS.md`.
