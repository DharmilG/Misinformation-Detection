"""Canonical explanation copy and copy validation of Document 04 §25.

Provides copy schema validation, fallback copy constants, and copy file loader.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIRED_COPY_KEYS = {
    "copy_version",
    "what_this_measures",
    "not_a_truth_claim",
    "probability_note",
    "score_note",
}

FALLBACK_COPY: dict[str, str] = {
    "copy_version": "fallback",
    "what_this_measures": (
        "Estimates the semantic consistency between the provided image and caption "
        "using a pre-trained multimodal embedding model."
    ),
    "not_a_truth_claim": (
        "This score is an automated statistical estimate of multimodal alignment and "
        "does not constitute a definitive judgment of factual truth."
    ),
    "probability_note": (
        "Calibrated probability represents estimated likelihood under temperature-scaled "
        "validation data."
    ),
    "score_note": (
        "Uncalibrated raw mismatch score evaluated against the operating threshold."
    ),
}


class CheckpointIntegrityError(ValueError):
    """Raised when release copy or checkpoint metadata fails integrity checks."""


def validate_copy_payload(payload: dict[str, Any]) -> None:
    """Validate that copy payload contains all required non-empty string keys."""
    if not isinstance(payload, dict):
        raise CheckpointIntegrityError("Copy payload must be a dictionary.")

    missing = REQUIRED_COPY_KEYS - set(payload.keys())
    if missing:
        raise CheckpointIntegrityError(f"Copy payload missing required keys: {missing}")

    for key in REQUIRED_COPY_KEYS:
        val = payload.get(key)
        if not isinstance(val, str) or not val.strip():
            raise CheckpointIntegrityError(
                f"Copy key '{key}' must be a non-empty string."
            )


def load_copy_file(path: str | Path | None = None) -> dict[str, str]:
    """Load and validate copy file, or return FALLBACK_COPY if path is None."""
    if path is None:
        # Check standard models release copy if it exists
        default_path = Path("models/test_release/copy.json")
        if default_path.is_file():
            try:
                data = json.loads(default_path.read_text(encoding="utf-8"))
                validate_copy_payload(data)
                return data
            except Exception:
                pass
        return dict(FALLBACK_COPY)

    p = Path(path)
    if not p.is_file():
        raise CheckpointIntegrityError(f"Copy file not found: {p}")

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CheckpointIntegrityError(f"Failed to parse copy file as JSON: {exc}") from exc

    validate_copy_payload(data)
    return data
