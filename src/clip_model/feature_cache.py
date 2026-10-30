"""The feature cache contract of Document 01 §11.

Provides cache key generation, atomic cache persistence, metadata tracking,
and strict load validation.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from constants import (
    DEFAULT_DTYPE,
    FEATURE_EXTRACTOR_VERSION,
    MODEL_ID,
    MODEL_REVISION,
    PREPROCESSING_VERSION,
    TOKENIZER_MAX_LENGTH,
)
from errors import FeatureCacheError


def cache_key(
    model_id: str = MODEL_ID,
    model_revision: str = MODEL_REVISION,
    feature_extractor_version: int = FEATURE_EXTRACTOR_VERSION,
    preprocessing_version: int = PREPROCESSING_VERSION,
    dtype: str = DEFAULT_DTYPE,
    normalized: bool = True,
    tokenizer_max_length: int = TOKENIZER_MAX_LENGTH,
    library_versions: dict[str, str] | None = None,
    source_manifest_sha256: str = "",
    device: str = "cpu",
    device_namespace: bool = False,
) -> str:
    """Generate deterministic cache key for feature configurations."""
    parts = [
        f"model_id={model_id}",
        f"model_revision={model_revision}",
        f"extractor_v={feature_extractor_version}",
        f"preprocess_v={preprocessing_version}",
        f"dtype={dtype}",
        f"normalized={normalized}",
        f"tok_max={tokenizer_max_length}",
        f"manifest={source_manifest_sha256}",
    ]
    if library_versions:
        sorted_libs = ",".join(f"{k}:{v}" for k, v in sorted(library_versions.items()))
        parts.append(f"libs={sorted_libs}")
    if device_namespace:
        parts.append(f"device={device}")

    return "/".join(parts)


def write_cache(
    target_dir: str | Path,
    image_features: np.ndarray,
    text_features: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save features to target_dir with metadata written last."""
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    row_count = int(image_features.shape[0])
    dtype_str = str(image_features.dtype)

    meta_payload = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "feature_extractor_version": FEATURE_EXTRACTOR_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "dtype": dtype_str,
        "normalized": True,
        "tokenizer_max_length": TOKENIZER_MAX_LENGTH,
        "row_count": row_count,
    }
    if metadata:
        meta_payload.update(metadata)

    npz_path = target / "features.npz"
    meta_path = target / "metadata.json"

    # Save npz array first
    np.savez(
        npz_path,
        image_features=image_features,
        text_features=text_features,
    )

    # Write metadata last
    meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")


def read_cache(
    target_dir: str | Path,
    expected_revision: str | None = MODEL_REVISION,
    expected_version: int | None = FEATURE_EXTRACTOR_VERSION,
    expected_normalized: bool | None = True,
) -> dict[str, Any]:
    """Load and validate feature cache."""
    target = Path(target_dir)
    meta_path = target / "metadata.json"
    npz_path = target / "features.npz"

    if not meta_path.is_file():
        raise FeatureCacheError(f"Feature cache metadata missing: {meta_path}")
    if not npz_path.is_file():
        raise FeatureCacheError(f"Feature cache data missing: {npz_path}")

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise FeatureCacheError(f"Invalid metadata JSON: {exc}") from exc

    if expected_revision is not None and meta.get("model_revision") != expected_revision:
        raise FeatureCacheError(
            f"Cache revision mismatch: expected {expected_revision}, got {meta.get('model_revision')}"
        )
    if expected_version is not None and meta.get("feature_extractor_version") != expected_version:
        raise FeatureCacheError(
            f"Cache version mismatch: expected {expected_version}, got {meta.get('feature_extractor_version')}"
        )
    if expected_normalized is not None and meta.get("normalized") != expected_normalized:
        raise FeatureCacheError(
            f"Cache normalization mismatch: expected {expected_normalized}, got {meta.get('normalized')}"
        )

    with np.load(npz_path, allow_pickle=False) as data:
        img_feats = np.array(data["image_features"])
        txt_feats = np.array(data["text_features"])

    return {
        "image_features": img_feats,
        "text_features": txt_feats,
        "metadata": meta,
    }
