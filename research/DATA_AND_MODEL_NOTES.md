# Data understanding, preprocessing, and model selection

These notes were supplied for the project presentation and are recorded here so
the repository carries the full context behind the slides.

---

## 1. The data — real pairs vs. hard swaps

NewsCLIPpings provides **39,288 balanced image–caption pairs**, built for a
forensic challenge.

### Authentic pairs (class 0)
A real documentary photograph paired with its verified news caption.

### Adversarial swaps (class 1)
Not random photos. The caption is deliberately replaced using:

- **CLIP text similarity** — captions that talk about the same event
- **Visual similarity** — captions that describe similar-looking scenes
- **Entity swaps** — swapping public figures into different political rallies

### Strict 50:50 balance

| Split | Rows | MATCHED | MISMATCHED |
|---|---|---|---|
| Train | ~25,000 | — | — |
| Validation | 7,024 | — | — |
| Test | 7,264 | 3,632 | 3,632 |

Random chance is 50%. A model that guesses or memorises fails immediately.

---

## 2. Preprocessing — sanitize, canonicalize, and cache

Three steps happen before any neural network touches the input.

### Image sanitization and transformation
- **EXIF stripping** — removes camera serials and GPS coordinates
- **Canonical conversion** — all formats, including RGBA, become 3-channel RGB
- **Normalized tensor** — bilinear resize to 224 × 224 px with standard CLIP mean/std scaling

### Text normalization and bounds
- **Unicode normalization** — cleans multi-byte characters and accent variations
- **Bounded tokenization** — Byte-Pair Encoding capped at 77 tokens with SOS/EOS
- **Defensive validation** — rejects empty captions (< 3 characters) or oversized files (> 10 MB)

### Leakage audit
- **0% image or caption overlap** between train, validation, and test
- Pre-extracted **512-dimensional** vectors stored as `.npy` arrays

Clean inputs prevent silent errors. Caching cuts training from **15 hours to 2 minutes**.

---

## 3. Model selection — why candidate D-C won

Five classifier architectures were pre-registered and evaluated on NewsCLIPpings.

| Candidate | Input | Dimensions | Accuracy |
|---|---|---|---|
| A — cosine similarity only | cosine | 1 | 54.2% |
| B — linear concatenation `[u ; v]` | concatenated | 1,024 | 61.8% |
| C — linear interaction `[u ; v ; \|u−v\| ; u⊙v]` | interaction | 2,048 | 64.5% |
| D-B — MLP on concatenation `[u ; v]` | concatenated | 1,024 | 66.1% |
| **D-C — MLP on explicit interaction** ★ | interaction | 2,048 | **69.20%** |

D-C also records macro F1 **69.18%** and ROC-AUC **77.83%**.

Explicit interaction vectors `[u ; v ; |u−v| ; u⊙v]` give the MLP direct
distance signals. That is why D-C won.

---

## Notes on provenance

The figures in sections 1–3 were supplied for the presentation. Where they
overlap with recorded project artifacts they agree: the split sizes match
`dataset/readme.md` and `dataset/models/checkpoint/metrics.json`; the five
candidate families and their dimensions match doc/03 §4.1; the preprocessing
limits match `src/clip_model/clip_features.py` and `src/utils/config.py`.

The per-candidate accuracies for A, B, C and D-B, the three hard-swap
construction methods, and the 15-hours-to-2-minutes figure are recorded here as
supplied. No separate results file for them is present in this repository.
