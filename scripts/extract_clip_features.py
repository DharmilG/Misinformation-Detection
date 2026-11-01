"""Extract aligned frozen CLIP features from the audited OOC dataset."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
CLIP_DIR = ROOT / "src" / "clip_model"
if str(CLIP_DIR) not in sys.path:
    sys.path.insert(0, str(CLIP_DIR))

from clip_features import CLIPFeatureExtractor


def _read_split(dataset_root: Path, split: str) -> list[dict[str, str]]:
    with (dataset_root / f"{split}.csv").open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def extract_split(
    dataset_root: Path,
    output_root: Path,
    split: str,
    batch_size: int,
    device: str | None,
) -> Path:
    records = _read_split(dataset_root, split)
    if not records:
        raise ValueError(f"{split}.csv is empty")
    extractor = CLIPFeatureExtractor(device=device)
    destination = output_root / split
    partial = destination / ".partial"
    partial.mkdir(parents=True, exist_ok=True)
    image_features: list[np.ndarray] = []
    text_features: list[np.ndarray] = []
    labels: list[int] = []
    sample_ids: list[str] = []

    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        batch_id = start // batch_size
        batch_file = partial / f"batch_{batch_id:06d}.npz"
        if batch_file.is_file():
            saved = np.load(batch_file, allow_pickle=False)
            image_features.append(saved["image"])
            text_features.append(saved["text"])
            labels.extend(saved["labels"].tolist())
            sample_ids.extend(saved["sample_ids"].tolist())
        else:
            pairs = [(dataset_root / row["image_path"], row["caption"]) for row in batch]
            features = extractor.encode_batch(pairs, device=device)
            image = features["image_features"].numpy().astype(np.float32)
            text = features["text_features"].numpy().astype(np.float32)
            batch_labels = np.asarray([int(row["label"]) for row in batch], dtype=np.int64)
            batch_ids = np.asarray(
                [f"{row['caption_id']}:{row['image_id']}" for row in batch],
                dtype="U",
            )
            temporary = batch_file.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez(handle, image=image, text=text, labels=batch_labels, sample_ids=batch_ids)
            temporary.replace(batch_file)
            image_features.append(image)
            text_features.append(text)
            labels.extend(batch_labels.tolist())
            sample_ids.extend(batch_ids.tolist())
        print(f"{split}: {min(start + batch_size, len(records))}/{len(records)}", flush=True)

    np.save(destination / "image_embeddings.npy", np.concatenate(image_features))
    np.save(destination / "text_embeddings.npy", np.concatenate(text_features))
    np.save(destination / "labels.npy", np.asarray(labels, dtype=np.int64))
    (destination / "sample_ids.json").write_text(json.dumps(sample_ids), encoding="utf-8")
    metadata = {
        "split": split,
        "rows": len(records),
        "dataset_csv_sha256": hashlib.sha256((dataset_root / f"{split}.csv").read_bytes()).hexdigest(),
        "model_id": extractor.info.model_id,
        "model_revision": extractor.info.model_revision,
        "weights_sha256": extractor.info.weights_sha256,
        "feature_dim": int(image_features[0].shape[-1]),
    }
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for batch_file in partial.glob("batch_*.npz"):
        batch_file.unlink()
    partial.rmdir()
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset" / "ooc_dataset")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "features")
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=("train", "val", "test"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    for split in args.splits:
        extract_split(args.dataset_root, args.output_root, split, args.batch_size, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
