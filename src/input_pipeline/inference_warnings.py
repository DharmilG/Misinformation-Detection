"""Prediction context warnings and crop-retention calculation of Document 04 §24.

Calculates retained area fraction under center cropping and cleans warning collections.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def calculate_crop_retention(width: int, height: int) -> float:
    """Compute retained area fraction as min(W, H) / max(W, H) from original dimensions."""
    if width <= 0 or height <= 0:
        return 0.0
    return float(min(width, height) / max(width, height))


def check_crop_warnings(
    width: int, height: int, threshold: float = 0.5
) -> list[str]:
    """Check if crop retention is below threshold and return context warnings."""
    fraction = calculate_crop_retention(width, height)
    warnings: list[str] = []
    if fraction < threshold:
        warnings.append("LOW_CROP_RETENTION")
    return warnings


def clean_warnings(raw_warnings: Sequence[Any]) -> list[Any]:
    """Preserve emission order and eliminate duplicate warning items."""
    return list(dict.fromkeys(raw_warnings).keys())
