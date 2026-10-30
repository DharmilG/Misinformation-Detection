"""Leakage grouping and exposure audit of Document 02 §10 and §11.

Defines link sets L0/L1/L2, calculates bipartite split exposure, and evaluates
connected-component feasibility thresholds.
"""

from __future__ import annotations

from typing import Any

LINK_SETS: dict[str, str] = {
    "L0": "image only",
    "L1": "image + caption source",
    "L2": "image + text",
}


def audit_exposure(splits: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    """Audit cross-split leakage without modifying records in place."""
    train_records = splits.get("train", [])
    test_records = splits.get("test", [])
    val_records = splits.get("val", [])

    train_images = {r.get("image_id") for r in train_records if r.get("image_id")}
    train_captions = {r.get("caption_id") for r in train_records if r.get("caption_id")}

    eval_records = test_records or val_records
    if not eval_records:
        return {
            "exposure_exact_image_pct": 0.0,
            "exposure_near_image_pct": 0.0,
            "exposure_shared_caption_source_pct": 0.0,
            "exposure_any_pct": 0.0,
        }

    total = len(eval_records)
    exact_image_count = 0
    near_image_count = 0
    caption_count = 0
    any_count = 0

    for r in eval_records:
        img_leak = r.get("image_id") in train_images
        cap_leak = r.get("caption_id") in train_captions

        if img_leak:
            exact_image_count += 1
        if cap_leak:
            caption_count += 1
        if img_leak or cap_leak:
            any_count += 1

    return {
        "exposure_exact_image_pct": float(exact_image_count / total * 100.0),
        "exposure_near_image_pct": float(near_image_count / total * 100.0),
        "exposure_shared_caption_source_pct": float(caption_count / total * 100.0),
        "exposure_any_pct": float(any_count / total * 100.0),
    }


def compute_exposure_union(records: list[dict[str, Any]]) -> dict[str, float]:
    """Compute exposure_any_pct as set union over affected records."""
    if not records:
        return {"exposure_any_pct": 0.0}

    affected = 0
    for r in records:
        if (
            r.get("exact_image_leak")
            or r.get("shared_caption_leak")
            or r.get("near_image_leak")
        ):
            affected += 1

    return {"exposure_any_pct": float(affected / len(records) * 100.0)}


def is_feasible_component_size(largest_share: float, threshold: float = 0.10) -> bool:
    """Check if largest connected component is <= threshold for grouping feasibility."""
    return largest_share <= threshold
