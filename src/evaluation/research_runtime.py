"""Research-only loader for the working classifier checkpoint.

The released loading path (``clip_model.checkpoint.load_checkpoint``) requires a
``release.json`` hash manifest (doc/03 §27.3).  The working checkpoint under
``dataset/models/checkpoint`` does not have one, so research scripts use this
loader instead.

It is deliberately labelled *research-only*: it deserialises with
``weights_only=True`` on CPU, loads the state dict with ``strict=True``, and
records the weight hash and the provenance of the calibration temperature in
every result row.  It is not a substitute for a released checkpoint.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from ..clip_model.classifier import build_features, create_model, predict_calibrated_labels
    from ..clip_model.classifier_calibration import (
        compute_calibrated_probability,
        fit_temperature,
    )
    from ..clip_model.clip_features import CLIPFeatureExtractor
except ImportError:  # imported as a top-level module from a script
    from src.clip_model.classifier import build_features, create_model, predict_calibrated_labels
    from src.clip_model.classifier_calibration import (
        compute_calibrated_probability,
        fit_temperature,
    )
    from src.clip_model.clip_features import CLIPFeatureExtractor


@dataclass(frozen=True)
class ResearchRuntime:
    """Everything a research script needs to score one image-caption pair."""

    extractor: CLIPFeatureExtractor
    model: torch.nn.Module
    candidate: str
    threshold: float
    temperature: float
    temperature_provenance: str
    checkpoint_dir: str
    weights_sha256: str

    def as_provenance(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "threshold": self.threshold,
            "temperature": self.temperature,
            "temperature_provenance": self.temperature_provenance,
            "checkpoint_dir": self.checkpoint_dir,
            "weights_sha256": self.weights_sha256,
            "encoder_model_id": self.extractor.model_name,
            "encoder_revision": self.extractor.revision,
        }


def weights_sha256(path: str | Path) -> str:
    """SHA-256 of the weight file, streamed so large files are safe."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_research_runtime(
    checkpoint_dir: str | Path,
    *,
    temperature: float | None = None,
    temperature_provenance: str = "explicit",
    device: str = "cpu",
) -> ResearchRuntime:
    """Load the working checkpoint without release verification.

    ``temperature`` must be supplied explicitly, or fitted separately with
    :func:`fit_temperature_from_features`.  Defaulting to 1.0 silently would
    make calibrated probabilities meaningless, so this loader refuses to guess.
    """
    base = Path(checkpoint_dir)
    weights = base / "weights.pt"
    config_path = base / "config.json"
    metrics_path = base / "metrics.json"
    for required in (weights, config_path, metrics_path):
        if not required.is_file():
            raise FileNotFoundError(f"checkpoint is incomplete: missing {required}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    candidate = str(config.get("candidate") or "D-C")
    threshold = float(metrics["threshold"])

    state = torch.load(weights, map_location="cpu", weights_only=True)
    model = create_model(candidate)
    model.load_state_dict(state, strict=True)
    model.eval()

    extractor = CLIPFeatureExtractor(device=device)

    if temperature is None:
        raise ValueError(
            "no calibration temperature supplied; pass --temperature or fit one "
            "with fit_temperature_from_features() and record its provenance"
        )

    return ResearchRuntime(
        extractor=extractor,
        model=model,
        candidate=candidate,
        threshold=threshold,
        temperature=float(temperature),
        temperature_provenance=temperature_provenance,
        checkpoint_dir=str(base),
        weights_sha256=weights_sha256(weights),
    )


def fit_temperature_from_features(
    runtime: ResearchRuntime,
    features_root: str | Path,
    *,
    split: str = "val",
) -> float:
    """Fit temperature on a split whose CLIP features are already extracted.

    This is used only because the working checkpoint did not record the
    temperature at training time.  The returned value must be stored with its
    provenance (the split and the feature directory) in every result report.
    """
    directory = Path(features_root) / split
    image = torch.from_numpy(np.load(directory / "image_embeddings.npy", allow_pickle=False))
    text = torch.from_numpy(np.load(directory / "text_embeddings.npy", allow_pickle=False))
    labels = torch.from_numpy(np.load(directory / "labels.npy", allow_pickle=False)).long()
    features = build_features(runtime.candidate, image, text)
    with torch.no_grad():
        logits = runtime.model(features)
    return float(fit_temperature(logits, labels))


def score_pair(runtime: ResearchRuntime, image: Any, caption: str) -> dict[str, Any]:
    """Score one (image, caption) pair with all research signals.

    ``image`` may be a path or a PIL image.  ``encode_pair`` is used rather than
    ``encode_images`` because only the former applies the doc/01 §7 /
    doc/04 §7 preprocessing chain (EXIF, alpha, downscale).
    """
    encoded = runtime.extractor.encode_pair(image, caption)
    image_emb = encoded["image_features"]
    text_emb = encoded["text_features"]
    image_emb = torch.nn.functional.normalize(image_emb.float(), p=2, dim=-1)
    text_emb = torch.nn.functional.normalize(text_emb.float(), p=2, dim=-1)
    cosine = float((image_emb * text_emb).sum(dim=-1)[0])

    features = build_features(runtime.candidate, image_emb, text_emb)
    with torch.no_grad():
        logits = runtime.model(features)
    # compute_calibrated_probability already selects the positive class.
    probability = float(compute_calibrated_probability(logits, runtime.temperature)[0])
    raw_probability = float(torch.softmax(logits.float(), dim=-1)[0, 1])
    label = predict_calibrated_labels(
        logits,
        temperature=runtime.temperature,
        threshold=runtime.threshold,
    )[0]

    return {
        "cosine": cosine,
        "d_c_score": raw_probability,
        "calibrated_probability": probability,
        "logit_0": float(logits[0, 0]),
        "logit_1": float(logits[0, 1]),
        "predicted_label": int(label),
        "text_token_count": int(encoded["text_token_count"]),
        "text_truncated": bool(encoded["text_truncated"]),
    }
