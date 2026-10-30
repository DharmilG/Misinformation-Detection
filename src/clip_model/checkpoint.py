"""Classifier checkpoint loading, verification, and LoadedClassifier API (Document 03 §27, §28).

Implements safe checkpoint deserialization with weights_only=True, hash integrity
verification, strict state_dict loading, second-machine verification evaluation,
and LoadedClassifier interface.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from classifier import build_features, create_model
from torch import nn

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class CheckpointIntegrityError(Exception):
    """Raised when file hash verification fails before deserialization (doc/03 §27.3)."""


class CheckpointNotFoundError(FileNotFoundError):
    """Raised when a checkpoint directory or required file is not found."""


class CheckpointIncompatibleError(Exception):
    """Raised when environment or model compatibility checks fail."""


# ---------------------------------------------------------------------------
# Safe loading and integrity verification (doc/03 §27.2, §27.3)
# ---------------------------------------------------------------------------

def compute_sha256(path: str | Path) -> str:
    """Compute hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_weights_safe(weights_path: str | Path) -> dict[str, Any]:
    """Load model weights safely with weights_only=True on CPU (doc/03 §27.3)."""
    p = Path(weights_path)
    if not p.is_file():
        raise CheckpointNotFoundError(f"Weights file not found: {p}")
    return torch.load(p, map_location="cpu", weights_only=True)


def verify_checkpoint_integrity(
    checkpoint_dir: str | Path,
    manifest: dict[str, Any] | Sequence[dict[str, Any]],
) -> None:
    """Verify file hashes against manifest before any deserialization (doc/03 §27.3)."""
    base = Path(checkpoint_dir)

    entries: dict[str, str] = {}
    if isinstance(manifest, dict):
        for k, v in manifest.items():
            if isinstance(v, dict):
                entries[k] = v.get("sha256", "")
            else:
                entries[k] = str(v)
    elif isinstance(manifest, (list, tuple)):
        for item in manifest:
            path_key = item.get("path") or item.get("file")
            if path_key and "sha256" in item:
                entries[path_key] = item["sha256"]

    for file_rel, expected_hash in entries.items():
        target = base / file_rel
        if not target.is_file():
            raise CheckpointIntegrityError(f"Missing checkpoint file: {file_rel}")
        actual_hash = compute_sha256(target)
        if actual_hash.lower() != expected_hash.lower():
            raise CheckpointIntegrityError(
                f"Checksum mismatch for {file_rel}: expected {expected_hash}, got {actual_hash}"
            )


def load_state_dict_strict(model: nn.Module, state_dict: dict[str, Any]) -> None:
    """Load model state_dict with strict=True (doc/03 §27.2)."""
    model.load_state_dict(state_dict, strict=True)


# ---------------------------------------------------------------------------
# Second-machine verification evaluation (doc/03 §28.3)
# ---------------------------------------------------------------------------

def evaluate_verification_agreement(
    reference_predictions: Sequence[dict[str, Any]],
    actual_predictions: Sequence[dict[str, Any]],
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Evaluate second-machine verification criteria against reference fixture.

    Acceptance criteria (doc/03 §28.3):
    - 100% class agreement across all samples.
    - Max score difference <= tolerance (1e-4).
    - Max calibrated probability difference <= tolerance (1e-4).
    """
    if len(reference_predictions) != len(actual_predictions):
        return {
            "passed": False,
            "reason": f"Sample count mismatch: {len(reference_predictions)} vs {len(actual_predictions)}",
            "samples_checked": len(actual_predictions),
            "class_agreements": 0,
            "max_score_difference": float("inf"),
            "max_probability_difference": float("inf"),
        }

    total = len(reference_predictions)
    class_agreements = 0
    max_score_diff = 0.0
    max_prob_diff = 0.0

    for ref, act in zip(reference_predictions, actual_predictions):
        ref_class = ref.get("reference_class") or ref.get("predicted_class")
        act_class = act.get("predicted_class") or act.get("class")
        if ref_class == act_class:
            class_agreements += 1

        ref_score = float(ref.get("reference_score", ref.get("score", 0.0)))
        act_score = float(act.get("score", 0.0))
        score_diff = abs(ref_score - act_score)
        max_score_diff = max(max_score_diff, score_diff)

        ref_prob = float(ref.get("reference_calibrated_probability", ref.get("calibrated_probability", 0.0)))
        act_prob = float(act.get("calibrated_probability", 0.0))
        prob_diff = abs(ref_prob - act_prob)
        max_prob_diff = max(max_prob_diff, prob_diff)

    passed = (
        class_agreements == total
        and max_score_diff <= tolerance + 1e-9
        and max_prob_diff <= tolerance + 1e-9
    )

    return {
        "passed": passed,
        "samples_checked": total,
        "class_agreements": class_agreements,
        "max_score_difference": float(max_score_diff),
        "max_probability_difference": float(max_prob_diff),
        "tolerance": tolerance,
    }


# ---------------------------------------------------------------------------
# LoadedClassifier API (doc/03 §27.1)
# ---------------------------------------------------------------------------

class LoadedClassifier:
    """The reconstructed classification head and inference API (doc/03 §27.1)."""

    model: nn.Module
    config: dict[str, Any]
    calibration: dict[str, Any]
    metrics: dict[str, Any]
    provenance: dict[str, Any]
    copy_info: dict[str, Any]
    threshold: float
    feature_family: str
    temperature: float

    def __init__(
        self,
        model: nn.Module,
        config: dict[str, Any],
        calibration: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        provenance: dict[str, Any] | None = None,
        copy_info: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.model.eval()
        self.config = config
        self.calibration = calibration or {}
        self.metrics = metrics or {}
        self.provenance = provenance or {}
        self.copy_info = copy_info or {}

        # Operating threshold and temperature
        inf_block = self.config.get("inference", {})
        self.threshold = float(inf_block.get("operating_threshold", 0.5))
        self.feature_family = inf_block.get("feature_family", "C")
        self.temperature = float(self.calibration.get("temperature", 1.0)) if self.calibration else 1.0

    def _prepare_features(
        self,
        image: torch.Tensor | None = None,
        text: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features is not None:
            feat = features
        elif image is not None and text is not None:
            feat = build_features(self.feature_family, image, text)
        else:
            raise ValueError("Must provide either (image, text) or precomputed features")
        if feat.dim() == 1:
            feat = feat.unsqueeze(0)
        return feat.float()

    def score(
        self,
        image: torch.Tensor | None = None,
        text: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
    ) -> float:
        """Return uncalibrated positive-class score (softmax probability of MISMATCHED)."""
        feat = self._prepare_features(image, text, features)
        with torch.no_grad():
            logits = self.model(feat)
            probs = F.softmax(logits, dim=-1)
            return float(probs[0, 1].item())

    def predict_proba(
        self,
        image: torch.Tensor | None = None,
        text: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Return calibrated class probabilities."""
        feat = self._prepare_features(image, text, features)
        with torch.no_grad():
            logits = self.model(feat)
            scaled = logits / max(self.temperature, 1e-6)
            probs = F.softmax(scaled, dim=-1)[0]
            return {
                "MATCHED": float(probs[0].item()),
                "MISMATCHED": float(probs[1].item()),
            }

    def predict(
        self,
        image: torch.Tensor | None = None,
        text: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Run full prediction path: score, calibrated proba, and thresholded label."""
        raw_score = self.score(image, text, features)
        proba = self.predict_proba(image, text, features)
        predicted_class = "MISMATCHED" if raw_score >= self.threshold else "MATCHED"

        return {
            "predicted_class": predicted_class,
            "score": raw_score,
            "calibrated_probability": proba["MISMATCHED"],
            "threshold": self.threshold,
        }


def load_checkpoint(
    path: str | Path,
    *,
    strict_compatibility: bool = False,
    verify_integrity: bool = True,
) -> LoadedClassifier:
    """Load classifier checkpoint directory adhering to doc/03 §27 sequence."""
    base = Path(path)
    if not base.is_dir():
        raise CheckpointNotFoundError(f"Checkpoint directory not found: {base}")

    # Load release.json or individual configs
    release_path = base / "release.json"
    manifest: dict[str, Any] | None = None
    if release_path.is_file():
        with open(release_path, "r", encoding="utf-8") as f:
            rel_data = json.load(f)
            manifest = {item["path"]: item["sha256"] for item in rel_data.get("files", []) if "path" in item}

    # Verify integrity before deserialization
    if verify_integrity and manifest:
        verify_checkpoint_integrity(base, manifest)

    # Parse config
    config_path = base / "config.json"
    if not config_path.is_file():
        raise CheckpointNotFoundError("Missing config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Optional metadata
    cal_data = None
    cal_path = base / "calibration.json"
    if cal_path.is_file():
        with open(cal_path, "r", encoding="utf-8") as f:
            cal_data = json.load(f)

    metrics_data = None
    met_path = base / "metrics.json"
    if met_path.is_file():
        with open(met_path, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

    prov_data = None
    prov_path = base / "provenance.json"
    if prov_path.is_file():
        with open(prov_path, "r", encoding="utf-8") as f:
            prov_data = json.load(f)

    copy_data = None
    copy_path = base / "copy.json"
    if copy_path.is_file():
        with open(copy_path, "r", encoding="utf-8") as f:
            copy_data = json.load(f)

    # Rebuild model
    inf_block = config.get("inference", {})
    family = inf_block.get("feature_family", config.get("candidate", "C"))
    hidden_dim = config.get("hyperparameters", {}).get("hidden_dim", 256)
    dropout = config.get("hyperparameters", {}).get("dropout", 0.1)

    model = create_model(family, hidden_dim=hidden_dim, dropout=dropout)

    # Load state dict
    weights_path = base / "weights.pt"
    state_dict = load_weights_safe(weights_path)
    load_state_dict_strict(model, state_dict)

    return LoadedClassifier(
        model=model,
        config=config,
        calibration=cal_data,
        metrics=metrics_data,
        provenance=prov_data,
        copy_info=copy_data,
    )
