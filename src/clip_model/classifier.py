"""
src/clip_model/classifier.py
=============================
CLIP-based image–caption consistency classifier.

Spec: 04_INPUT_AND_INFERENCE_PIPELINE (implied) and 05_REPORTING_AND_UI §5.

Behaviour:
  - If open-clip-torch is available, loads the model on first call (process-level
    singleton — no second cache layer, per §12.4) and computes cosine similarity.
  - If open-clip-torch is NOT available, returns UNAVAILABLE gracefully.
  - Never raises to the caller — errors are captured in the result.
  - Score: cosine similarity in [0, 1], threshold from config.CLASSIFIER_THRESHOLD.
  - Label: "MATCHED" (score >= threshold) or "MISMATCHED" (score < threshold).
  - Calibration is not applied in the stub implementation.
"""
from __future__ import annotations

import io
import uuid
from dataclasses import dataclass
from typing import Optional

from src.utils.config import (
    CLASSIFIER_THRESHOLD,
    ENCODER_REVISION,
    FEATURE_CONTRACT_VERSION,
    MODEL_VERSION,
)

# ── Try to import open_clip ───────────────────────────────────────────────

try:
    import open_clip                    # type: ignore
    import torch                        # type: ignore
    from PIL import Image               # type: ignore
    _OPEN_CLIP_AVAILABLE = True
except ImportError:
    _OPEN_CLIP_AVAILABLE = False


# ── Result dataclass ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class ClassifierResult:
    status: str                         # "OK" | "UNAVAILABLE" | "FAILED"
    run_id: str
    model_version: str
    encoder_revision: str
    feature_contract_version: str
    # OK-only
    label: Optional[str] = None         # "MATCHED" | "MISMATCHED"
    score: Optional[float] = None
    calibrated_probability: Optional[float] = None
    calibration_status: str = "NOT_APPLIED"
    score_definition: str = (
        "Cosine similarity between CLIP image and text embeddings, mapped to [0, 1]. "
        "Values at or above 0.5 are labelled MATCHED; below 0.5 as MISMATCHED."
    )
    # non-OK
    reason: Optional[str] = None
    remedy: Optional[str] = None
    error_detail: Optional[str] = None


# ── Process-level singleton ───────────────────────────────────────────────

_model = None
_preprocess = None
_tokenizer = None


def _load_model() -> bool:
    """Load model into process-level singleton. Returns True on success."""
    global _model, _preprocess, _tokenizer
    if _model is not None:
        return True
    if not _OPEN_CLIP_AVAILABLE:
        return False
    try:
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="laion2b_s34b_b79k",
        )
        _model.eval()
        _tokenizer = open_clip.get_tokenizer("ViT-B-32")
        return True
    except Exception:
        return False


# ── Public API ────────────────────────────────────────────────────────────

def run_classifier(
    image_bytes: bytes,
    caption_normalized: str,
) -> ClassifierResult:
    """
    Compute CLIP cosine similarity between the image and caption.

    Parameters
    ----------
    image_bytes:        Preprocessed JPEG bytes from the input pipeline.
    caption_normalized: The normalized caption string.

    Returns
    -------
    ClassifierResult with status OK, UNAVAILABLE, or FAILED.
    Never raises.
    """
    run_id = str(uuid.uuid4())

    _common = dict(
        run_id=run_id,
        model_version=MODEL_VERSION,
        encoder_revision=ENCODER_REVISION,
        feature_contract_version=FEATURE_CONTRACT_VERSION,
    )

    if not _OPEN_CLIP_AVAILABLE:
        return ClassifierResult(
            status="UNAVAILABLE",
            reason=(
                "open-clip-torch is not installed. "
                "Install it with: pip install open-clip-torch"
            ),
            remedy="Install open-clip-torch and restart the application.",
            **_common,
        )

    loaded = _load_model()
    if not loaded:
        return ClassifierResult(
            status="UNAVAILABLE",
            reason=(
                "The CLIP model could not be loaded. "
                "This may be a network or cache issue."
            ),
            remedy=(
                "Ensure that open-clip-torch is installed and that the model "
                "weights are accessible."
            ),
            **_common,
        )

    try:
        assert _model is not None
        assert _preprocess is not None
        assert _tokenizer is not None

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = _preprocess(img).unsqueeze(0)          # type: ignore[arg-type]

        text_tokens = _tokenizer([caption_normalized])

        with torch.no_grad():
            img_features  = _model.encode_image(img_tensor)
            text_features = _model.encode_text(text_tokens)

            img_features  = img_features  / img_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            similarity = (img_features @ text_features.T).item()

        # Map to [0, 1] (cosine similarity for well-normalised CLIP embeddings
        # is typically in [−1, 1]; clamp and rescale).
        score = float(max(0.0, min(1.0, (similarity + 1.0) / 2.0)))
        label = "MATCHED" if score >= CLASSIFIER_THRESHOLD else "MISMATCHED"

        return ClassifierResult(
            status="OK",
            label=label,
            score=round(score, 4),
            calibrated_probability=None,
            calibration_status="NOT_APPLIED",
            **_common,
        )

    except Exception as exc:
        return ClassifierResult(
            status="FAILED",
            reason="An error occurred during classifier inference.",
            error_detail=str(exc),
            remedy="Check the application logs for details.",
            **_common,
        )
