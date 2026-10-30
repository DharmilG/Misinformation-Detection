"""Split configurations and boundaries of Document 02 §9.

Defines the official, grouped, and balanced dev subsample split configurations.
"""

from __future__ import annotations

import random
from typing import Any

OFFICIAL_CONFIGURATION: str = "newsclippings-official-v1"
GROUPED_CONFIGURATION: str = "newsclippings-grouped-v1"
DEV_TARGETS: dict[str, int] = {"train": 2000, "val": 500, "test": 500}


def subsample_dev(
    splits: dict[str, list[dict[str, Any]]],
    targets: dict[str, int] | None = None,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Sample class-balanced dev subset preserving official distribution."""
    target_counts = targets or DEV_TARGETS
    rng = random.Random(seed)
    dev_splits: dict[str, list[dict[str, Any]]] = {}

    for split_name, target in target_counts.items():
        records = splits.get(split_name, [])
        half = target // 2

        matched = [r for r in records if str(r.get("label")) in ("MATCHED", "Label.MATCHED", "0")]
        mismatched = [r for r in records if str(r.get("label")) in ("MISMATCHED", "Label.MISMATCHED", "1")]

        sampled_matched = rng.sample(matched, min(half, len(matched)))
        sampled_mismatched = rng.sample(mismatched, min(half, len(mismatched)))

        combined = sampled_matched + sampled_mismatched
        dev_splits[split_name] = combined

    return dev_splits
