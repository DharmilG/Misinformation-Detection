# Model Checkpoints

This directory houses downloaded or local model checkpoints. Checkpoint weights and archives are excluded by `.gitignore` (only this `README.md` is committed).

## Released Artifact Specification

- **Naming format**: `mismis-clip-vitb32-<dataset>-<split-config>-<candidate>-v<model_version>-<run_id>`
- **Canonical Model**: `mismis-clip-vitb32-nclp-official-v1-d-c-v1.0.0-r20261021T0930Z-s42-a1b2c3d4`
- **Release Archive**: `mismis-clip-vitb32-nclp-official-v1-d-c-v1.0.0-r20261021T0930Z-s42-a1b2c3d4.zip`
- **Immutability Guarantee**: Released model artifacts are immutable. Checksums, weights, and configuration files under a released version number never change; corrections and updates are published under a new version.

## Verification & Integrity Check

After downloading a release archive or directory, verify its SHA-256 checksum:

```bash
# Windows PowerShell
Get-FileHash -Algorithm SHA256 <archive-or-file>

# Linux / macOS
sha256sum <archive-or-file>
```

Run the automated second-machine verification against the reference fixture:

```bash
python scripts/verify_checkpoint.py --release models/<release-directory>
```

## Pinned Encoder Acquisition

Fetch the frozen CLIP ViT-B/32 encoder revision:

```python
from transformers import CLIPModel, CLIPProcessor

model_id = "openai/clip-vit-base-patch32"
revision = "e6a30b603a447e25185c3e6022e0e4b85c2763f0"

model = CLIPModel.from_pretrained(model_id, revision=revision)
processor = CLIPProcessor.from_pretrained(model_id, revision=revision)
```

## Loading & Inference Snippet

```python
from clip_model.checkpoint import load_checkpoint

# Safe deserialization with weights_only=True and pre-deserialization hash checks
classifier = load_checkpoint("models/<release-directory>")

# Full prediction path (feature construction, uncalibrated score, calibrated probability, thresholded label)
prediction = classifier.predict(image_embedding=v_i, text_embedding=v_t)
print("Predicted Class:", prediction["predicted_class"])
print("Score:", prediction["score"])
print("Calibrated P(MISMATCHED):", prediction["calibrated_probability"])
```

## Compatibility & Environment

- **Python**: `>=3.10, <3.14`
- **PyTorch**: `>=2.1, <3.0` (Hard requirement: `weights_only=True` support)
- **Lockfile**: Refer to `requirements-lock.txt` bundled within each release archive for bitwise reproducible dependencies.
