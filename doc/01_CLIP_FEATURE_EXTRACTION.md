# 01 - CLIP Feature Extraction

## Multimodal Out-of-Context Misinformation Detection Using Retrieval-Augmented Generation (RAG)

**Document status:** Accepted - implementation contract for Phase 2  
**Document type:** Component specification  
**Depends on:** `../00_PROJECT_OVERVIEW.md`, `04_INPUT_AND_INFERENCE_PIPELINE.md`  
**Consolidated from:** `02_CLIP_FEATURE_EXTRACTION.md` (primary), with the encoder file manifest from `05_MODEL_CHECKPOINTS.md`  
**Last updated:** 2026-10-21

---

## 1. Purpose and boundary

This document defines how the project turns a validated image and caption into **CLIP feature vectors**. It is the contract between the input pipeline, dataset preparation, classifier training, and inference.

CLIP is used as a **frozen feature extractor**. This component:

1. Loads a pinned CLIP checkpoint exactly once, locally.
2. Converts images and captions into projected 512-dimensional embeddings.
3. Returns those embeddings with a complete provenance record.
4. Caches dataset-level features so training does not re-encode the same data.

**This component does not decide whether a caption is misleading.** It produces vectors. A classifier trained on those vectors (document 03) produces a score, and retrieved evidence (document 06) produces a separate factual assessment. Neither is guaranteed to be correct.

### 1.1 Explicit non-promises

The following are **not** claimed, tested, or implied anywhere in this document:

| Non-promise | Why it matters |
|---|---|
| **Perfect parity with NewsCLIPpings** | Different checkpoints and preprocessing produce measurably different vectors. Parity is a diagnostic, not a guarantee. |
| **Identical outputs across devices** | CUDA, MPS, and CPU perform floating-point operations in different orders. Outputs are close, not identical. |
| **Guaranteed misinformation detection** | CLIP features describe semantic similarity, not truth. A perfectly matched image-caption pair can still be a false news claim. |
| **A calibrated probability of falsity** | A cosine similarity is a similarity score, not a probability. The classifier head is not calibrated unless `03` explicitly calibrates it. |
| **Complete coverage of the caption** | CLIP truncates at 77 tokens. Long captions are only partially encoded. |

Anything in this document that sounds like a threshold is an **engineering gate**, not a scientific claim. Thresholds are stated with their rationale and their failure path.

---

## 2. Local execution and model acquisition

The application runs **locally** on a single machine. The model is downloaded once and then used offline.

### 2.1 Acquisition

| Property | Value |
|---|---|
| Hub | Hugging Face Hub |
| Model ID | `openai/clip-vit-base-patch32` |
| Class | `CLIPModel` (full model, not `CLIPVisionModel` or `CLIPTextModel` alone) |
| Processor | `CLIPProcessor` (image processor + tokenizer) |
| Weights format | `safetensors` |
| `trust_remote_code` | `False` — always, with no override path |
| Cache location | Default Hugging Face cache, overridable via `HF_HOME` |
| Download size | Approximately 600 MB (one-time) |

### 2.2 Offline operation

`local_files_only=True` prevents the Hub client from downloading missing files. **It does not guarantee the model can load.** If the cache is incomplete, loading fails.

Three scenarios are distinguished and tested separately:

| Scenario | Setup | Expected result |
|---|---|---|
| **First run, network available** | Model absent from cache | Files download, model loads, `INFO` log records the acquisition |
| **Offline, cache complete** | All required files cached, `local_files_only=True` | Model loads successfully with no network access |
| **Offline, cache missing or incomplete** | Files absent, `local_files_only=True` | `ModelUnavailableError` with an actionable message |

A missing model **must raise**. It must never produce randomly initialized, zeroed, or otherwise fabricated embeddings. A failed component stops the pipeline and the UI reports the component as unavailable.

### 2.3 Network policy

| Phase | Network |
|---|---|
| First-time setup (model download) | Allowed, one-time |
| Model loading from complete cache | Not allowed, not needed |
| Feature extraction (inference) | Never |
| Feature extraction (dataset) | Never |
| Tests marked `slow` without a cache | May download; marked and skippable |

The feature extraction path contains no HTTP client. The only network-capable dependency is the Hub client inside `transformers`, which is constrained by `local_files_only`.

---

## 3. Scope and out of scope

### 3.1 In scope

- Pinning the CLIP checkpoint and its exact revision.
- Device resolution and model placement.
- Image and text encoding through the official processor.
- Explicit normalization, dimensional checks, and finiteness checks.
- Tokenizer token counting and truncation detection.
- A complete provenance record for every produced feature.
- Dataset feature caching with strict validation.
- A parity diagnostic against the NewsCLIPpings reference features.

### 3.2 Out of scope

- Training or fine-tuning CLIP. The weights are frozen.
- Classifier architecture, loss functions, or training loops — document 03.
- Dataset construction, annotation parsing, or split logic — document 02.
- Checkpoint save/load conventions for the classifier — document 03.
- Retrieval, prompting, or report generation — documents 06–05.
- Zero-shot classification with prompt templates.
- Image augmentation of any kind. Features are extracted from the image as submitted.
- Embedding visualization.

### 3.3 Relationship to document 04

The separation of preprocessing responsibilities is strict and must not be blurred:

```text
Document 04  decode → validate → EXIF orientation → RGB conversion → downscale to 2048 px (no crop)
Document 01  CLIP processor: resize to 224 → center crop → tensor → mean/std normalization
              CLIP tokenizer: tokenize → truncate to 77 → embed
Document 02  load dataset images → apply the SAME document 04 normalization → document 01
```

**No image is ever resized or normalized twice.** Document 04 never applies CLIP's mean/std normalization, and document 01 never applies EXIF correction or alpha flattening.

---

## 4. Model selection and provenance

### 4.1 Why `openai/clip-vit-base-patch32`

- It is the ViT-B/32 architecture, which corresponds to the CLIP variant used by the NewsCLIPpings reference features.
- It is a single well-documented checkpoint with published precomputed feature artifacts.
- It runs on CPU, so the project does not require a GPU to demonstrate.
- Image and text encoders share one projection space, so the two output vectors are directly comparable.

### 4.2 Pinning

| Item | Requirement |
|---|---|
| Model ID | `openai/clip-vit-base-patch32` |
| Revision | A specific commit SHA, recorded in `constants.py` and in every cache's metadata |
| `transformers` version | Pinned in `requirements.txt` |
| `torch` version | Pinned in `requirements.txt` |
| `numpy` version | Pinned in `requirements.txt` |
| `pillow` version | Pinned in `requirements.txt` |
| `safetensors` version | Pinned in `requirements.txt` |

**The revision is a commit SHA, not `main`.** A moving tag means the weights can change silently between two runs, which would invalidate every cached feature and every reported metric.

### 4.3 Provenance record

At load time the encoder captures:

- `model_id` and `model_revision`.
- `weights_sha256` — SHA-256 of the loaded `safetensors` weight file, computed once and cached in memory for the process lifetime.
- Library versions: `transformers`, `torch`, `numpy`, `pillow`, `safetensors`.
- `model.config.projection_dim` as read from the checkpoint — never hardcoded.
- `preprocessing_version` from document 04.
- `feature_extractor_version` (defined in §4.4).

This record is embedded in every `EncoderInfo` and therefore travels with every cache and every inference result.

### 4.4 Version constants and the bump policy

```python
# src/clip_model/constants.py
MODEL_ID = "openai/clip-vit-base-patch32"
MODEL_REVISION = "<commit sha>"          # filled at implementation time
FEATURE_EXTRACTOR_VERSION = 1            # bump on any behaviour change
TOKENIZER_MAX_LENGTH = 77
EMBEDDING_DIM = 512
```

**Bump policy.** Any change to how features are produced requires incrementing `FEATURE_EXTRACTOR_VERSION`:

- Changing the normalization behaviour (order, epsilon, dtype).
- Changing the pooling or projection path.
- Changing the tokenizer configuration or `max_length`.
- Changing the image preprocessing in document 04, which also increments document 04's `PREPROCESSING_VERSION`.

Because both versions are recorded in cache metadata and validated on load, an incompatible cache is rejected automatically rather than silently reused.

---

### 4.5 The encoder file manifest

A model ID and a commit SHA identify the **intended** revision. They do not guarantee that the files resolved on a given machine are the expected ones: a cache can be partially populated, a mirror can serve different bytes, or a file can be replaced. The encoder is therefore verified by a **file manifest**.

The manifest is computed at packaging time from the resolved cache and stored in the checkpoint release:

```json
"encoder": {
  "model_id": "openai/clip-vit-base-patch32",
  "revision": "<commit sha>",
  "weights_sha256": "<hex of model.safetensors>",
  "files": [
    {"path": "model.safetensors", "sha256": "<hex>"},
    {"path": "config.json", "sha256": "<hex>"},
    {"path": "preprocessor_config.json", "sha256": "<hex>"},
    {"path": "tokenizer.json", "sha256": "<hex>"},
    {"path": "tokenizer_config.json", "sha256": "<hex>"},
    {"path": "vocab.json", "sha256": "<hex>"},
    {"path": "merges.txt", "sha256": "<hex>"},
    {"path": "special_tokens_map.json", "sha256": "<hex>"}
  ],
  "processor": {"size": 224, "crop_size": 224, "resample": 3,
                "mean": [0.48145466, 0.4578275, 0.40821073],
                "std": [0.26862954, 0.26130258, 0.27577711],
                "do_normalize": true},
  "tokenizer": {"model_max_length": 77, "truncation": true,
                "padding": "max_length", "bos_token_id": 49406,
                "eos_token_id": 49407}
}
```

| Block | What it pins |
|---|---|
| `files` | The exact bytes of every file that affects the encoder's behaviour |
| `processor` | Image size, crop policy, resample filter, mean/std, and whether normalization is applied |
| `tokenizer` | Context length, truncation and padding policy, and the special token ids that define the EOT pooling position |

The processor and tokenizer blocks are the details §8 relies on. Recording them means a consumer does not have to trust that the installed processor behaves as this document describes.

**Canonical owner.** The manifest **definition** is canonical here. The checkpoint's `release.json` stores it, and the loader verifies the resolved files against it. The verification procedure, the four offline modes, and the `EncoderProvenanceError` behaviour are specified in `03_CLASSIFIER_TRAINING_AND_CHECKPOINTS.md` §21.
## 5. Preprocessing contract

### 5.1 Image path

| Step | Owner | Detail |
|---|---|---|
| Decode, EXIF orientation, RGB conversion, alpha flattening | Document 04 | Produces an RGB PIL image, longest side ≤ 2048 px |
| Resize to 224 px | CLIP image processor | Shortest-edge resize to 224 |
| Center crop to 224×224 | CLIP image processor | Crop may remove image edges |
| Scale to `[0, 1]` and apply mean/std normalization | CLIP image processor | OpenAI CLIP mean/std |
| Tensor conversion | CLIP image processor | `float32` |
| Batch assembly | Document 01 | Stack to `(N, 3, 224, 224)` |

**No manual resize, crop, or normalization is performed by this document.** The processor is the only component that touches model-specific pixel values.

The center crop can discard content at the edges of a non-square image. This is inherent to CLIP's preprocessing and is recorded as a known limitation (§17), not a defect to work around here.

### 5.2 Text path

| Step | Owner | Detail |
|---|---|---|
| Unicode NFKC normalization, whitespace collapsing | Document 04 | Produces the normalized caption |
| Tokenization | CLIP tokenizer | BPE, `add_special_tokens=True` |
| Truncation | CLIP tokenizer | `truncation=True`, `max_length=77` |
| Padding | CLIP tokenizer | `padding="max_length"`, `max_length=77` |
| Token embedding + positional embedding | CLIP text model | EOT position used for pooling |
| Projection to shared space | CLIP text projection | 512-d |
| L2 normalization | Document 01 | Explicit (§8) |

Padding to `max_length=77` rather than `"longest"` is deliberate. The CLIP text model locates its EOT token positionally, so a fixed 77-token context makes the pooling position deterministic and independent of the batch composition. A batch of one long caption and a batch of one short caption therefore produce the same result for the long caption.

**No prompt templates are applied.** The caption is encoded as written. Prompt engineering belongs to a zero-shot approach, which is out of scope.

### 5.3 Image-text alignment guarantee

Image and text encoders are independently testable, but they must be executed with the same model instance, the same revision, and the same device. A pair is only meaningful when both vectors come from one consistent configuration. `encode_batch` enforces this by construction.

### 5.4 Interaction with a pre-resized corpus

The evaluation corpus described in document 02 is **not** delivered at original resolution. Its photographs were resized by the dataset author to a shorter side of 255-256 pixels and re-encoded as JPEG at quality 90, before this project ever saw them.

This has one concrete consequence for the preprocessing chain, and it is recorded here rather than left to be discovered.

| Path | Chain | Effect |
|---|---|---|
| **Training** (corpus image) | 255-256 px shorter side → document 04's 2048 px downscale is a **no-op** → processor 256 → 224 | The crop region is correct; the resampling chain is short |
| **Inference** (user upload) | original resolution → document 04's 2048 px downscale → processor 2048 → 224 | The crop region is the same relative region; the resampling chain is long and the source is sharper |

The two paths end at the same 224×224 centre crop of the same relative field of view. They differ in the resampling chain and in the JPEG artefacts already baked into the corpus images.

**This is accepted, not fixed.** Obtaining original-resolution photographs means the 98 GB VisualNews archive, which is not practical for this project. The difference is small, it is symmetric, and it is **measured** rather than asserted: document 02 §8.1.2 records the measured embedding difference between a corpus image at its native size and the same image upsampled to the inference resolution - a mean cosine similarity of **0.9972** over a 20-image sample, with a minimum of **0.9937**.

**Verification.** A `data`-tier test asserts that document 04's 2048 px downscale leaves a corpus image unchanged, and another measures the embedding difference between the native and upscaled forms of the same photograph.

---

## 6. Encoder lifecycle

### 6.1 Lazy singleton

The model is loaded on first use and reused for the process lifetime.

- Loading is guarded by a `threading.Lock` so concurrent Streamlit reruns cannot trigger two loads.
- The loaded model and processor are module-level state in `src/clip_model/model_loader.py`.
- A `reset_model_cache()` function exists for tests that need to reload with a different configuration.

Loading is not performed at import time. Importing `src.clip_model` must not download or load anything, so that test collection and static analysis stay fast and offline.

### 6.2 Runtime mode

```python
model.eval()                 # disable dropout
torch.inference_mode()       # disable autograd graph construction
```

`model.eval()` is required: dropout would make repeated calls on the same input return different vectors, destroying the determinism guarantees in §14.

`torch.inference_mode()` is preferred over `torch.no_grad()` because it also avoids version-counter bookkeeping on inference tensors.

### 6.3 No gradients reach CLIP

The frozen encoder is never part of the training graph. Document 03 trains only a classification head, on detached NumPy arrays, outside any autograd context. CLIP weights are never updated, and the checkpoint written by document 03 contains only the head.

### 6.4 Thread safety

The loaded model is shared. `torch.inference_mode()` calls are safe to issue from multiple threads on CPU. On CUDA, concurrent calls are safe but serialize on the default stream. The project is single-user; no explicit locking is required around inference calls, and none is added.

---

## 7. Output schema and public API

### 7.1 Public methods

```python
class ClipEncoder:
    def encode_pair(self, image: Image.Image, text: str) -> EmbeddingPair: ...
    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray: ...
    def encode_texts(self, texts: Sequence[str]) -> np.ndarray: ...
    def encode_batch(self, pairs: Sequence[tuple[Image.Image, str]]) -> BatchEmbeddings: ...
```

| Method | Returns | Purpose |
|---|---|---|
| `encode_pair` | `EmbeddingPair` | Single-example inference |
| `encode_images` | `(N, 512)` float32 array | Batched image extraction |
| `encode_texts` | `(N, 512)` float32 array | Batched text extraction |
| `encode_batch` | `BatchEmbeddings` | Aligned image-caption dataset convenience |

`encode_images` and `encode_texts` are independently testable and independently usable. `encode_batch` is a thin wrapper that calls them with shared configuration; it introduces no separate code path.

`encode_images` and `encode_texts` return the raw NumPy arrays. The richer metadata is available from `encode_pair` and `encode_batch`, and from the encoder's `info` property, so a caller that only needs vectors is not forced to construct provenance objects.

### 7.2 Guaranteed output properties

For every vector produced by any of the four methods:

| Property | Guarantee |
|---|---|
| Shape | `(512,)` for a single vector, `(N, 512)` for a batch |
| dtype | `float32` |
| Values | All finite — no `NaN`, no `Inf` |
| Norm | L2 norm equal to 1.0 within 1e-6 |
| Space | The shared CLIP image–text projection space |
| Order | Image and text vectors in a pair are aligned to the same input |

For every caption encoded:

| Property | Guarantee |
|---|---|
| `text_token_count` | Tokens actually fed to the model, including BOS and EOT, ≤ 77 |
| `text_original_token_count` | Tokens before truncation, including BOS and EOT |
| `text_truncated` | `True` when the caption exceeded the context window |
| Count basis | CLIP BPE tokens, not words, not characters, not bytes |

### 7.3 Schema

```python
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class EncoderInfo:
    model_id: str
    model_revision: str
    weights_sha256: str
    embedding_dim: int                # read from config, asserted 512
    tokenizer_max_length: int         # 77
    normalized: bool                  # True
    dtype: str                        # "float32"
    device: str                       # "cpu" | "mps" | "cuda" | "cuda:0"
    clip_logit_scale: float           # diagnostic only — see §8.4
    transformers_version: str
    torch_version: str
    numpy_version: str
    preprocessing_version: int        # from document 04
    feature_extractor_version: int    # from §4.4

@dataclass(frozen=True)
class EmbeddingPair:
    image_embedding: np.ndarray       # (512,), float32, finite, L2-normalized
    text_embedding: np.ndarray        # (512,), float32, finite, L2-normalized
    text_token_count: int             # including BOS+EOT, <= 77
    text_original_token_count: int    # including BOS+EOT, before truncation
    text_truncated: bool
    encoder: EncoderInfo

@dataclass(frozen=True)
class BatchEmbeddings:
    ids: tuple[str, ...]              # stable sample IDs, aligned to rows
    image_embeddings: np.ndarray      # (N, 512), float32, L2-normalized
    text_embeddings: np.ndarray       # (N, 512), float32, L2-normalized
    text_token_counts: np.ndarray     # (N,) int32
    text_original_token_counts: np.ndarray  # (N,) int32
    text_truncated: np.ndarray        # (N,) bool
    encoder: EncoderInfo
```

The public schema contains **NumPy arrays, not PyTorch tensors**. Torch tensors are an internal implementation detail of the encoder. Document 03 converts the NumPy arrays to tensors when it builds a training batch, and the conversion is explicit and visible in that document.

Arrays returned by the encoder are marked read-only (`arr.flags.writeable = False`) so a consumer cannot accidentally corrupt a cached or shared array in place.

### 7.4 NumPy as the interchange format

NumPy was chosen deliberately over `torch.Tensor`:

- The cache is stored as `.npy` files, so the in-memory and on-disk representations match exactly.
- Document 02 and 03 can load features without importing `torch`, which keeps dataset inspection tooling light.
- Serialization avoids `torch.save` pickle risks entirely.

---

## 8. Feature semantics and verified behavior

This section defines what the vectors *mean*, and states how each claim is **verified at runtime and by test** rather than assumed.

### 8.1 Projection

`get_image_features` and `get_text_features` return vectors in the shared CLIP projection space, dimension `config.projection_dim`. The encoder reads this value from the loaded configuration and asserts it equals 512 for the pinned checkpoint.

The encoder uses the full `CLIPModel`. It does **not** use `vision_model.last_hidden_state` (which is 768-dimensional for ViT-B/32 and is not in the shared space) and does not use the pre-projection `pooler_output` of the text model.

**Verification.** A slow test asserts `embedding.shape[-1] == model.config.projection_dim` and that the value is 512. If a future revision changes the projection dimension, the test fails rather than the pipeline silently producing differently shaped vectors.

### 8.2 Normalization

Normalization is applied **explicitly by this document**, once, after extraction:

```python
vector = vector / np.linalg.norm(vector, axis=-1, keepdims=True)
```

CLIP's own `forward` method normalizes internally for the contrastive loss, but `get_image_features` / `get_text_features` do **not** guarantee normalized output. Whether the installed `transformers` version returns already-normalized vectors is **verified rather than assumed**.

**Measured.** With `transformers` 5.x, `CLIPModel.forward` already returns L2-normalised `image_embeds` and `text_embeds`: the raw projection has a norm of about 7.86, and the returned tensor has a norm of exactly 1.0. The explicit normalization above is therefore **idempotent on this library version** - it changes nothing, because normalizing a unit vector again returns the same vector.

That is a fact about the library, not a licence to drop the step. The explicit normalization is kept because it makes the contract independent of the library version, and because it is what the cache metadata's `normalized` flag describes. Two tests assert both facts: that the returned vectors are unit-norm to within 1e-6, and that normalizing them twice changes nothing.

The encoder performs a one-time check at load time on a synthetic probe:

1. Extract a probe image vector and a probe text vector.
2. Record their raw norms.
3. Apply the explicit normalization.
4. Assert the final norms are within 1e-6 of 1.0.
5. Record whether the raw vectors were already unit-norm in the `EncoderInfo` provenance log line.

This makes the double-normalization question a measured fact rather than a hidden assumption. Normalizing an already-unit vector is a no-op numerically, so the operation is safe under either behaviour — but the behaviour is now known and recorded.

**Verification.** A slow test asserts `np.allclose(np.linalg.norm(v), 1.0, atol=1e-6)` for both modalities, and a second test asserts that the encoder's explicit normalization is idempotent when applied twice (an already-normalized vector is unchanged within 1e-6).

### 8.3 Text pooling

CLIP pools the text representation at the **EOT** token position, not by mean pooling over the sequence. The HF implementation locates the EOT position and applies the text projection to that pooled representation.

**Verification.** A slow test compares `get_text_features` output against a manual computation — text model output, EOT-position selection, then `text_projection` — and asserts agreement within 1e-5. This pins the pooling behaviour to the installed implementation instead of relying on documentation that may drift.

### 8.4 `logit_scale` semantics

`model.logit_scale` is a **learned scalar parameter** of the pretrained CLIP checkpoint. It scales the cosine similarity between image and text vectors into logits used by the contrastive objective, equivalent to a learned temperature.

It is **not**:

- a probability that a caption is misleading,
- a classifier output,
- a confidence score,
- comparable across checkpoints.

The encoder records it in `EncoderInfo.clip_logit_scale` **as a diagnostic value only**. Document 03 will not use it as a feature or as a score. The field is retained because it is useful when comparing this implementation against the NewsCLIPpings reference features, which were produced within a CLIP training framework where that scale is meaningful.

### 8.5 What the vectors mean, and what they do not

A cosine similarity between an image vector and a text vector measures **semantic relatedness within CLIP's training distribution**. A high value means the text plausibly describes the image. A low value means it probably does not.

It does not measure:

- whether the claim is true,
- whether the image is authentic or unedited,
- whether the event happened when or where the text says,
- whether the pairing is a deliberate out-of-context manipulation.

A semantically matching but false claim, and a semantically mismatching but truthful caption, are both possible. This is exactly why document 00 separates the classifier stage from the evidence stage.

---

## 9. Device policy and numerics

### 9.1 Resolution

Resolution order for the `auto` setting:

```text
CUDA  →  Apple MPS  →  CPU
```

| Setting | Behaviour |
|---|---|
| `auto` (default) | Use the first available device in the order above |
| `cpu` | Force CPU; error if unavailable |
| `cuda` / `cuda:N` | Force CUDA; error if unavailable |
| `mps` | Force Apple MPS; error if unavailable |

An explicitly requested device that is unavailable raises `DeviceUnavailableError`. There is no silent fallback from an explicit request, because a silent fallback would change the numerics without changing the recorded configuration.

### 9.2 dtype

`float32` throughout. Mixed precision (`float16`, `bfloat16`) is not used in v1. It would change feature values, complicate the cache contract, and provide no meaningful speedup for the CPU path that this project defaults to.

### 9.3 Cross-device behaviour

CUDA, MPS, and CPU perform floating-point operations in different orders. The same input on two devices produces vectors that are **close but not bit-identical**.

The project therefore makes **no claim of identical outputs across devices**. What is guaranteed:

| Property | Guarantee |
|---|---|
| Same device, same input, repeated calls | Bit-identical on CPU; within 1e-5 on GPU |
| Different devices, same input | Within a documented cosine tolerance (§14.3) |
| Features cached on device A, used on device B | Cached vectors are reused unchanged; no re-derivation happens |

**The practical consequence for the project:** a classifier trained on cached features is applied to features from the same cache during evaluation, so the train and inference distributions match exactly. Only live user inference runs on a possibly different device, and the drift is bounded by the tolerance in §14.3.

### 9.4 Batch-size effects

Results are identical whether images are encoded one at a time or in a batch, up to floating-point associativity in the matrix multiply. A tolerance of 1e-5 (cosine) covers this.

---

## 10. Batching and performance

### 10.1 Batch size

| Context | Default batch size | Notes |
|---|---|---|
| Inference (single pair) | 1 | `encode_pair` |
| Dataset extraction | 32 | Configurable, recorded in cache metadata |
| Very large batches | Auto-halved on OOM | With a clear log message |

An out-of-memory error on CUDA triggers one automatic halving of the batch size, up to two attempts, before surfacing `FeatureExtractionError`. The final batch size used is recorded in the cache metadata so that a partially different extraction can be detected.

### 10.2 Resource expectations

Approximate, for `openai/clip-vit-base-patch32` on CPU:

| Resource | Approximate value |
|---|---|
| Model load time | 3–10 s, once per process |
| Inference latency (single pair) | 100–400 ms |
| Dataset throughput (batch 32) | 20–60 images/second |
| Resident memory | Approximately 1.5–2.0 GB including the model |

These are planning figures, not guarantees. They are recorded to set expectations and to notice a regression on a different machine.

### 10.3 Memory discipline

`torch.inference_mode()` prevents graph retention. Encoded batches are moved to CPU and converted to NumPy immediately, so the GPU working set does not grow with the dataset size. Only one batch is resident at a time.

---

## 11. Feature cache

### 11.1 Purpose

Dataset features are cached so that classifier training and every re-run of an experiment do not re-encode the same images and captions. The cache is a **derived artifact**: it can always be rebuilt from the source dataset, and it is excluded from Git.

### 11.2 Layout

```text
data/processed/features/
    <dataset_slug>/
        <model_slug>/
            <revision>/
                <extractor_version>/
                    <split_name>/
                        image_embeddings.npy        # (N, 512) float32
                        text_embeddings.npy         # (N, 512) float32
                        ids.json                    # stable sample IDs, order matters
                        image_hashes.json           # per-row SHA-256 of the source image
                        caption_hashes.json         # per-row SHA-256 of the normalized caption
                    metadata.json               # written LAST
```

`data/processed/` is already excluded by `.gitignore`.

### 11.3 Cache key

The cache is valid for a given set of inputs only if every input that can change the output is unchanged. The key is composed of:

| Component | Source | Rationale |
|---|---|---|
| `image_sha256` | Per row, from document 04 | Content hash of the normalized source image |
| `caption_sha256` | Per row, SHA-256 of the **normalized** caption | Raw caption would miss normalization changes |
| `source_manifest_sha256` | Document 02 | A changed manifest means different rows, even if every individual row still hashes the same |
| `model_id` | `constants.py` | Different checkpoint, different features |
| `model_revision` | `constants.py` | Same model at a different revision differs |
| `preprocessing_version` | Document 04 | Change in EXIF/RGB/downscale behaviour alters pixels |
| `feature_extractor_version` | `constants.py` | Change in pooling, normalization, or tokenizer alters vectors |
| `dtype` | `float32` | A dtype change alters stored values |
| `normalized` | `True` | Normalized and unnormalized vectors are not interchangeable |
| `tokenizer_max_length` | `77` | Truncation length changes the text vector |
| `library_versions` | `transformers`, `torch`, `numpy`, `pillow` | Different library versions can change outputs |
| `encoder_file_manifest` | §4.5 | A replaced weight file produces different vectors under the same revision |

**Device is not part of the key by default**, so a cache is shareable across machines. The cross-device difference is bounded and measured in §14.3, and the `device_namespace` flag in §11.7 segregates caches by device when strict reproducibility is required.

**Device is not part of the key by default.** See §11.7.

### 11.4 Metadata

`metadata.json` is written **last**, after all arrays and side files are in place. It records:

```json
{
  "cache_schema_version": 1,
  "model_id": "openai/clip-vit-base-patch32",
  "model_revision": "<commit sha>",
  "weights_sha256": "...",
  "preprocessing_version": 1,
  "feature_extractor_version": 1,
  "embedding_dim": 512,
  "dtype": "float32",
  "normalized": true,
  "tokenizer_max_length": 77,
  "device": "cpu",
  "batch_size": 32,
  "split_name": "test",
  "row_count": 1000,
  "ids_sha256": "...",
  "library_versions": {"transformers": "...", "torch": "...", "numpy": "...", "pillow": "..."},
  "created_at": "2026-10-20T12:00:00Z",
  "complete": true
}
```

`ids_sha256` is a hash of the ordered ID list, so a reordering of rows is detected even if the row count is unchanged.

### 11.5 Atomic writes

An interrupted write must never leave a cache that appears valid.

Procedure:

1. Write to a temporary directory alongside the target.
2. Write each `.npy` and JSON side file there.
3. Write `metadata.json` there, last, with `complete: true`.
4. Atomically move the temporary directory into place.
5. Remove any existing target directory only after the new one is fully in place.

On load, a cache is accepted only if `metadata.json` exists, parses, has `complete: true`, and every file it references exists with the expected shape and row count. Any failure raises `FeatureCacheError` and the cache is treated as absent — never partially consumed.

### 11.6 Validation on load

| Check | Failure action |
|---|---|
| `metadata.json` present and parseable | `FeatureCacheError` |
| `complete == true` | `FeatureCacheError` |
| `cache_schema_version` supported | `FeatureCacheError` |
| `model_id` and `model_revision` match the current configuration | `FeatureCacheError` |
| `preprocessing_version` and `feature_extractor_version` match | `FeatureCacheError` |
| `dtype`, `normalized`, `embedding_dim`, `tokenizer_max_length` match | `FeatureCacheError` |
| `row_count` equals the number of IDs and both arrays' first dimension | `FeatureCacheError` |
| `ids_sha256` matches the recomputed hash of the ID list | `FeatureCacheError` |
| Per-row `image_sha256` / `caption_sha256` match the requested samples | `FeatureCacheError` |
| All values finite | `FeatureCacheError` |
| All row norms within 1e-6 of 1.0 | `FeatureCacheError` |

The per-row hash check is what makes the cache trustworthy: it confirms the cache was built for *these* samples, not merely for samples with the same count.

### 11.7 Device and cache validity

Device is **excluded from the cache key by default**, so a cache built on one machine is usable on another.

The risk this creates is a train/inference distribution mismatch: features extracted on CUDA and features extracted on CPU are not bit-identical (§9.3). The project handles this as follows:

1. Device is **recorded in metadata** for every cache.
2. A **one-time cross-device drift check** (§14.3) quantifies the difference on a small sample, and the result is recorded in the documentation.
3. The `device_namespace` configuration flag, when enabled, includes the device in the cache key, segregating caches per device. Default is **off**.
4. If only one device is available locally, the drift check is deferred and the limitation is recorded in §17. A teammate on a different device can enable `device_namespace` to force segregation.

This choice favours sharing cached features over strict per-device reproducibility, which is the right trade-off for a single-user local project.

### 11.8 Cache invalidation

Bumping `preprocessing_version` or `feature_extractor_version` automatically invalidates every existing cache, because the metadata check in §11.6 compares those values. There is no manual cleanup step and no risk of silently reusing stale features.

---

## 12. NewsCLIPpings parity diagnostic

### 12.1 Purpose and framing

The NewsCLIPpings repository provides precomputed CLIP ViT-B/32 image and text embeddings. Comparing our implementation against them answers a specific question: **are the reference features and our features measuring the same thing, closely enough to be substituted?**

The answer is **not assumed in either direction**. Parity is a **diagnostic**, not a guarantee, and the thresholds below are **provisional engineering gates** chosen to catch gross incompatibilities — not established scientific tolerances, and not claims about the equivalence of two independent implementations.

### 12.2 Preconditions

Before any numeric comparison, the following must be confirmed and recorded. A comparison performed before these checks is invalid.

| # | Precondition | How to establish it |
|---|---|---|
| 1 | **Sample ID alignment** | Match rows by stable sample ID, not by row order. Verify ID uniqueness and coverage, and confirm the join is one-to-one. |
| 2 | **Feature dimensionality** | Confirm the reference arrays are 512-dimensional and check their dtype. |
| 3 | **Reference normalization state** | Compute row norms of the reference features. If they are already unit-norm, they are normalized; if not, they are unnormalized. This determines whether a comparison needs normalizing one side. |
| 4 | **Checkpoint provenance** | Determine which CLIP checkpoint and revision generated the reference features, and whether the text features are projected into the shared space. |
| 5 | **Preprocessing used** | Determine the resize, crop, and normalization applied when the reference features were generated. |
| 6 | **Pooling behaviour** | Determine whether the reference text features use EOT pooling, consistent with §8.3. |
| 7 | **Ordering of comparison** | Images are compared to images, texts to texts. Never compare an image vector to a text vector. |

If precondition 1 cannot be satisfied, the parity test is **not run** and the reason is recorded. A misaligned comparison produces a meaningless number, and reporting it would be worse than reporting nothing.

### 12.3 Metrics

Computed per modality, over the aligned sample set:

| Metric | Definition |
|---|---|
| Mean cosine similarity | Mean of per-sample cosine between our vector and the reference vector |
| Median cosine similarity | Middle value of the per-sample distribution |
| 1st percentile cosine | Low tail of the distribution |
| 5th percentile cosine | Slightly wider low tail |
| Minimum cosine | Worst single sample |
| Max absolute difference | Element-wise maximum difference, complementing the cosine view |

A cosine-only view hides magnitude differences, and a difference-only view hides direction differences. Both are reported.

### 12.4 Provisional thresholds

| Gate | Threshold | Rationale |
|---|---|---|
| Mean cosine | ≥ 0.99 | Catches systematic preprocessing or projection differences |
| 1st percentile cosine | ≥ 0.98 | Catches a minority of badly handled samples |

These thresholds are **provisional**. They are engineering gates chosen so that a clearly incompatible implementation is caught early. They are not derived from theory, not published, and not claims about what parity "should" be. If the project later has a defensible basis for tighter or looser gates, the thresholds change and this section is updated.

**A passing result is a permission to investigate further, not a proof of interchangeability.**

### 12.5 When parity fails

A failure does **not** mean the Hugging Face implementation is unsuitable, and does **not** trigger an automatic switch to another backend.

The investigation order is:

1. **Sample alignment** — the most common cause of a spurious failure. Re-verify the ID join.
2. **Normalization state** — compare like with like. Normalize the unnormalized side.
3. **Preprocessing** — resize method, crop policy, and whether the reference was produced from the original file or from an already-downscaled image.
4. **Checkpoint provenance** — a different checkpoint or revision of the same architecture.
5. **Tokenizer and truncation** — tokenization differences change the text vector.
6. **Pooling** — confirm EOT pooling in the reference.
7. **Implementation differences** — the remaining possibilities in the two libraries.

Only after this investigation, if the reference features remain incompatible, is switching backends considered — and that decision is documented with its reasoning.

### 12.6 Decision record — which features are used for the final model

**The final classifier is trained and evaluated on features produced by this project's own extractor, not on the NewsCLIPpings reference features.**

The reference features are used for exactly two purposes:

1. **The parity diagnostic** in this section.
2. **A baseline comparison**, to check that this implementation does not behave differently from the published reference in a way that would invalidate the comparison.

Reasons:

- Training and inference must use one implementation. Mixing encoders creates a feature-distribution mismatch that no amount of downstream tuning reliably repairs.
- The reference features may have been generated with a different preprocessing pipeline, and that difference is not fully recoverable from the published artifacts.
- If parity passes, using our own extractor anyway costs one feature-extraction pass over the dataset and removes the entire class of mismatch problems.

This decision is recorded here so that a later reader does not assume the reference features were used and silently reproduce an inconsistent setup.

### 12.7 Output

The diagnostic produces a report containing the preconditions, their verification results, all metrics from §12.3, the pass/fail outcome against §12.4, and the investigation notes if it failed. The report is stored under `data/processed/` and excluded from Git.

### 12.8 Outcome for the curated corpus

The corpus in document 02 is a **derived** subset of the upstream NewsCLIPpings release. Its labels come from the upstream `merged_balanced` split, but its caption text and photographs were re-sourced from VisualNews and **re-encoded at 256 px shorter side**. That single fact decides what the parity diagnostic can and cannot check.

| Modality | Comparable with the reference features? | Reason |
|---|---|---|
| **Text** | **Yes** | The caption text is the upstream VisualNews text, unmodified. The tokenizer and the text encoder see the same input. |
| **Image** | **No** | The reference features were computed on original-resolution photographs. The corpus photographs are 255-256 px re-encodes, so the pixels are not the same even where the scene is. |

The image path therefore **fails for an explainable reason**, and §12.5's investigation order reaches that conclusion before any implementation defect is suspected. The diagnostic must record the exclusion and its reason rather than reporting a large image-path difference as a parity failure.

| Consequence | Handling |
|---|---|
| The text path remains a valid independent check | It confirms the tokenizer, the EOT pooling, the projection, and the normalisation |
| The image path cannot validate the image preprocessing | §5.4 records the corpus resolution instead, and §15 records the golden-vector regression that covers the image path offline |
| The §12.6 decision record is unaffected | The final model uses this project's own extractor regardless of the outcome |

**Verification.** An `optional`-tier test asserts that the image path is **excluded with its stated reason**, rather than silently skipped, so the exclusion is visible in the report.

---

## 13. Error handling

### 13.1 Error types

| Exception | Raised when | User-facing behaviour |
|---|---|---|
| `ModelUnavailableError` | Checkpoint cannot be found or loaded, including an incomplete offline cache | "The CLIP model is not available locally. Run the setup step once with an internet connection, or set `HF_HOME` to an existing cache." |
| `DeviceUnavailableError` | An explicitly requested device is not available | "The requested device (`cuda`) is not available on this machine." |
| `FeatureExtractionError` | Encoding fails, an OOM persists, or output validation fails | "Feature extraction failed. The analysis cannot continue." |
| `FeatureCacheError` | A cache is missing, incomplete, or inconsistent | "The feature cache is missing or out of date. It will be rebuilt." |

### 13.2 Rules

- **No silent fallback.** A failed load never produces zero vectors, random vectors, or a differently-shaped result.
- **No silent device substitution.** An explicit device request that fails raises; it does not quietly fall back to CPU.
- **No partial consumption.** A cache that fails any validation check is treated as absent.
- **No traceback in the UI.** The Streamlit layer catches these exceptions and shows `str(exc)`. The full traceback goes to the log at `ERROR` level.
- **Errors are distinguishable by type.** The UI needs to tell "the model is missing" apart from "your image is bad" apart from "the cache is stale", because the user action differs for each.

### 13.3 What the pipeline does on failure

The classifier stage cannot run without features. On a `ModelUnavailableError` or `FeatureExtractionError`:

1. The feature stage reports the failure.
2. The classifier stage is skipped — it does not emit a default or placeholder prediction.
3. The retrieval and assessment stages may still run, because they do not depend on CLIP. The UI presents the evidence-based assessment and states that the image-caption consistency result is unavailable.

This follows the rule in document 00: a failed component must not silently generate a fabricated prediction or report.

---

## 14. Numerical invariants and determinism

### 14.1 Invariants checked on every batch

| Invariant | Check |
|---|---|
| Shape | `image_embeddings.shape == (N, 512)` |
| Dtype | `float32` exactly |
| Finiteness | No `NaN`, no `Inf` |
| Norm | Every row within 1e-6 of 1.0 |
| Token counts | `0 < text_token_count <= 77`; `text_original_token_count >= text_token_count` |
| Truncation flag | `text_truncated == (text_original_token_count > 77)` |
| Row alignment | `len(ids) == N` |

A failure raises `FeatureExtractionError` and the batch is not returned. These checks are cheap relative to the encoding cost.

### 14.2 Determinism

| Condition | Guarantee |
|---|---|
| Same device, same input, repeated calls | Bit-identical on CPU |
| Same device, GPU, repeated calls | Within 1e-5 cosine |
| Same device, different batch size | Within 1e-5 cosine |
| Different devices, same input | Within 1e-4 cosine (see §14.3) |

`model.eval()` is mandatory for this table to hold; dropout would break the first two rows.

### 14.3 The cross-device tolerance and how it is established

The 1e-4 cross-device figure is a **provisional bound**, to be established by measurement rather than assumed:

1. Extract features for a fixed set of 50 images and 50 captions on the available device.
2. If a second device is available locally, extract the same set on it.
3. Compute the per-sample cosine similarity between the two device's outputs.
4. Record the minimum observed value in this document.

The recorded minimum defines the project's actual cross-device tolerance, and the table above is updated to match. If only one device is available, the measurement is deferred, the limitation is recorded in §17, and the 1e-4 figure is explicitly labelled as **unverified** until a second device is available.

**Device drift is a bounded difference, not a source of silent inconsistency.** It is quantified here, recorded in cache metadata, and optional to eliminate entirely via the `device_namespace` flag (§11.7).

---

## 15. Testing strategy

### 15.1 Test tiers and commands

The test tier taxonomy, the marker definitions, the environment requirements, the complete command set, the coverage gate, and the evaluation-harness invariants are specified centrally in `07_INTEGRATION_TESTING.md` §2 and §3.

The tests listed in the remaining subsections of this section are this component's own tests, executed under those tiers. They are **unit and component tests**: they verify this component's contracts, not the boundaries between components, which are tested in `07_INTEGRATION_TESTING.md` §5.

```bash
# This component's fast suite
pytest -m "not slow and not data and not eval and not benchmark and not e2e"
```

### 15.2 The two styles of test

The suite distinguishes two kinds of assertion, and the distinction is deliberate.

| Style | What it asserts | Typical outcome |
|---|---|---|
| **Behavioural conformance** | The API that exists today satisfies the behavioural clauses | Mostly **passes**. It is the real baseline |
| **Interface conformance** | The API surface, the cache contract, the error contract, and the dataset pipeline exist as specified | **Fails** until the implementation is extended. Each failure names the clause and the required change |

A uniformly red suite would hide the fact that the projection, the pooling, the shapes, the dtype, and the finiteness are already correct. A uniformly green suite would hide the gaps. The split keeps both visible.

### 15.3 Fast tests

- Schema construction and field presence.
- `EncoderInfo` provenance fields are populated.
- Cache key composition: changing any single component changes the key.
- Cache validation rejects a mismatched revision, a wrong `feature_extractor_version`, a wrong row count, a reordered ID list, a non-finite value, and a wrong norm.
- Cache write is atomic: an interrupted write leaves no loadable cache.
- `metadata.json` is written last.
- Device resolution logic with mocked `torch.cuda.is_available` / `torch.backends.mps.is_available`.
- Error types are raised for the right conditions with mocked loaders.
- Token counting logic against a stub tokenizer implementing the same interface, including the BOS/EOT convention.
- The public API surface required by §7.1 is present: `encode_pair`, `encode_images`, `encode_texts`, `encode_batch`.
- The model revision is pinned, and the encoder reports its own provenance.

### 15.4 Slow tests

- Model loads, is in `eval()` mode, and reports the expected `projection_dim`.
- `encode_pair` returns finite, unit-norm, float32 vectors of shape `(512,)`.
- `encode_images` / `encode_texts` return `(N, 512)` with matching row counts.
- Normalization behaviour: raw norms are recorded, normalized norms are within 1e-6 of 1.0, and double normalization is a no-op.
- Pooling: `get_text_features` matches the manual EOT-position projection within 1e-5.
- Token counting against the real tokenizer: BOS and EOT included, `max_length` 77, truncation flag correct for a caption that exceeds the window.
- A caption beyond the window is **truncated**, not rejected by a position-embedding error.
- Repeatability: the same input twice gives bit-identical output on CPU.
- Batch-size invariance: batch-1 and batch-32 results agree within 1e-5.
- **Offline loading with a complete cache** succeeds with `local_files_only=True` and makes no network call.
- **Offline loading with an empty cache** raises `ModelUnavailableError` with an actionable message.
- **First-run with network** downloads and loads; skipped automatically when the environment is offline.
- A missing model never yields a fabricated vector.
- The weights are loaded once per process, not once per instance.
- The device can be selected, and an unavailable device raises rather than falling back silently.

### 15.5 Golden-vector regression

A recorded set of image and caption pairs with their expected vectors, checked against the current build within the tolerance of §14.3.

This is the **primary, always-available, offline** check on the encoder chain. The parity diagnostic in §12 needs a download and covers the text path only; the golden vectors need neither and cover both modalities. They detect any change in the processor, the tokenizer, the pooling, the projection, or the normalisation.

### 15.6 Dataset-tier tests

The corpus-facing tests are specified in document 02 and execute under the `data` marker, which requires the dataset but **not** the model weights. They verify the label gate, the split structure, the leakage properties, the manifest contract, and the shortcut ceilings of document 02 §X.

The preprocessing-skew measurement of §5.4 and the dataset-scale extraction of document 02 §14 require both the dataset and the weights, and are marked `data` and `slow` together.

### 15.7 Parity tests

Run only when the NewsCLIPpings reference features are present locally, and only for the text path. See §12 and §12.8.

---

## 16. Acceptance criteria

Phase 2 is complete when:

1. The model ID, revision, and all dependency versions are pinned in `constants.py` and `requirements.txt`, and the revision is a commit SHA rather than a tag.
2. `MODEL_REVISION` is a specific commit SHA recorded in the source.
3. The checkpoint's license is recorded in the documentation, along with the weights SHA-256 and the model-card URL.
4. The model loads lazily, once per process, under a lock, in `eval()` mode, with `torch.inference_mode()`.
5. `encode_pair`, `encode_images`, `encode_texts`, and `encode_batch` exist with the signatures in §7.1, and the image and text paths are independently testable.
6. Every output vector is `(512,)`, float32, finite, and L2-normalized within 1e-6, verified by test rather than by inspection.
7. The projection dimension is read from the checkpoint configuration and asserted, not hardcoded.
8. The normalization behaviour of the installed `transformers` version is measured at load time and recorded, and no double-normalization assumption is made.
9. Text pooling is verified against a manual EOT-position computation.
10. Token counts are CLIP tokens including BOS and EOT, `max_length` is 77, and the truncation flag is correct — all covered by test.
11. `EncoderInfo` carries the model ID, revision, weights hash, library versions, device, dtype, and both version numbers.
12. The cache key includes every input that can change the output, and device is excluded by default with the `device_namespace` flag available (§11.7).
13. Cache writes are atomic, metadata is written last, and every validation check in §11.6 is enforced.
14. `ModelUnavailableError`, `DeviceUnavailableError`, `FeatureExtractionError`, and `FeatureCacheError` exist, are raised appropriately, and never lead to a silent fallback or a fabricated vector.
15. The three offline scenarios — first run, complete cache, missing cache — are tested separately.
16. The parity diagnostic runs only after the preconditions in §12.2 are established, reports all metrics from §12.3, and treats the thresholds as provisional.
17. The decision record in §12.6 exists: the final model uses this project's own extractor.
18. The cross-device tolerance has been measured on a second device, or the limitation is explicitly recorded as unverified.
19. The fast test suite passes with no model download and no network access.
20. `01` and `04` preprocessing are not applied twice for any input, verified by a test that runs a known image through both stages.

---

## 17. Known limitations

| Limitation | Consequence | Status |
|---|---|---|
| Cross-device outputs are close, not identical | Small train/inference drift if the device changes | Quantified in §14.3; `device_namespace` flag available |
| Cross-device tolerance not yet measured | The 1e-4 figure is unverified | Deferred until a second device is available |
| Parity thresholds are provisional | A pass is not proof of interchangeability | Documented in §12.4 |
| NewsCLIPpings reference features may not be reproducible | The final model uses our own extractor | Decision recorded in §12.6 |
| CLIP truncates at 77 tokens | Long captions are only partially encoded | Tokens are counted and truncation is flagged |
| Center crop may remove image edges | Content outside the 224×224 crop is invisible to the model | Inherent to CLIP preprocessing |
| ICC profiles are ignored (document 04) | Wide-gamut and CMYK images shift slightly | Accepted for v1 |
| CLIP is biased toward its training distribution | Performance varies across topics, regions, and languages | Documented; evaluate per subset |
| Semantic similarity is not truth | A matching pair can still be a false claim | The reason document 00 separates the stages |
| Weights are frozen | No domain adaptation to the dataset | Acceptable for v1 |
| First run requires internet | The model is unavailable until downloaded | One-time setup, documented in §2 |
| Text truncation detection is count-based | A caption may be cut mid-token without notice | `text_truncated` flag surfaces it |

---

## 18. Interface summary

| Direction | Detail |
|---|---|
| Provides | The `EmbeddingPair` — the image and text embeddings, the token counts, and the truncation flag — and the `EncoderInfo` provenance record |
| Receives | The validated RGB image and the normalized caption from `04_INPUT_AND_INFERENCE_PIPELINE.md` |
| Owns | The encoder configuration and pinning, the feature semantics and their verification, the device policy, the feature cache contract, the encoder file manifest, and the NewsCLIPpings parity diagnostic |
| Must not | Fine-tune CLIP, alter the projection or pooling semantics, normalize a feature a second time, or accept an encoder whose file manifest does not match |

The complete cross-component interface matrix, and the prohibitions that apply to every component, are specified in `07_INTEGRATION_TESTING.md` §11.

## 19. Immediate next step

Implement `src/clip_model/` against this contract, beginning with the encoder and the output schema, then the cache, then the parity diagnostic. Fill in the concrete `MODEL_REVISION` commit SHA, the license text, and the weights SHA-256 during implementation, and update §4 and §14.3 in this document with the measured values.

**Conformance status.** The suite in `tests/` asserts this document's clauses. Its report, `reports/conformance.json`, records for each case the clause it enforces and whether the current implementation satisfies it. The gaps it lists are the requirements still to be implemented; the specification in this document is unchanged by them.

The next specification after this one is `02_DATASET_PREPARATION.md`.
