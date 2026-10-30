"""Manifest record schema and atomic writer of Document 02 §12.

Defines the 22-field JSONL manifest record schema, deterministic sample_id generator,
and atomic file writing.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

MANIFEST_VERSION: int = 1


def format_sample_id(split: str, index: int) -> str:
    """Format sample_id as nclp_<split>_<index:06d>."""
    return f"nclp_{split}_{index:06d}"


def hash_caption(caption: str) -> str:
    """Generate 64-character lowercase SHA-256 digest of caption."""
    return hashlib.sha256(caption.encode("utf-8")).hexdigest()


def create_record(
    source_record_id: int | str,
    source_file: str,
    split: str,
    configuration: str = "newsclippings-official-v1",
    image_path: str = "images/sample.jpg",
    image_sha256: str = "0" * 64,
    caption_raw: str = "",
    caption: str = "",
    label: str = "MATCHED",
    source_label: str = "consistent",
    source_label_raw: Any = 0,
    caption_id: str = "",
    image_id: str = "",
    sample_id: str | None = None,
    dataset: str = "newsclippings",
    dataset_version: str = "1.0.0",
    source_extra: dict[str, Any] | None = None,
    pair_group_id: str = "",
    event_id: str = "",
    flags: list[str] | None = None,
) -> dict[str, Any]:
    """Create a complete 22-field manifest record."""
    s_id = sample_id or format_sample_id(split, int(source_record_id) if str(source_record_id).isdigit() else 0)
    cap_hash = hash_caption(caption)

    return {
        "sample_id": s_id,
        "dataset": dataset,
        "dataset_version": dataset_version,
        "source_record_id": source_record_id,
        "source_file": source_file,
        "configuration": configuration,
        "split": split,
        "image_path": image_path,
        "image_sha256": image_sha256,
        "caption_raw": caption_raw,
        "caption": caption,
        "caption_sha256": cap_hash,
        "label": label,
        "source_label": source_label,
        "source_label_raw": source_label_raw,
        "source_extra": source_extra or {},
        "caption_id": caption_id,
        "image_id": image_id,
        "pair_group_id": pair_group_id or f"pg_{image_id}",
        "event_id": event_id or f"ev_{caption_id}",
        "flags": flags or [],
        "manifest_version": MANIFEST_VERSION,
    }


def write_manifest(target_path: str | Path, records: list[dict[str, Any]]) -> None:
    """Atomically write manifest records to JSONL file."""
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = target.with_name(f"{target.name}.tmp_{os.getpid()}")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    tmp_path.replace(target)
