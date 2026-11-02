"""Extract aligned frozen CLIP features from the audited OOC dataset."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
from io import BytesIO
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


def _pack_array(value: np.ndarray) -> bytes:
    buffer = BytesIO()
    np.save(buffer, value, allow_pickle=False)
    return buffer.getvalue()


def _load_array(value: bytes) -> np.ndarray:
    return np.load(BytesIO(value), allow_pickle=False)


def _write_npy_atomic(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    temporary.replace(path)


def extract_split(
    dataset_root: Path,
    output_root: Path,
    split: str,
    batch_size: int,
    device: str | None,
    checkpoint_path: Path,
) -> Path:
    records = _read_split(dataset_root, split)
    if not records:
        raise ValueError(f"{split}.csv is empty")
    extractor = CLIPFeatureExtractor(device=device)
    destination = output_root / split
    csv_hash = hashlib.sha256((dataset_root / f"{split}.csv").read_bytes()).hexdigest()
    expected_batches = (len(records) + batch_size - 1) // batch_size
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(checkpoint_path) as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS extraction_runs ("
            "split TEXT PRIMARY KEY, rows INTEGER NOT NULL, batch_size INTEGER NOT NULL, "
            "dataset_csv_sha256 TEXT NOT NULL, model_id TEXT NOT NULL, model_revision TEXT NOT NULL, "
            "weights_sha256 TEXT NOT NULL, device TEXT NOT NULL, status TEXT NOT NULL)"
        )
        db.execute(
            "CREATE TABLE IF NOT EXISTS extraction_batches ("
            "split TEXT NOT NULL, batch_id INTEGER NOT NULL, image BLOB NOT NULL, text BLOB NOT NULL, "
            "labels BLOB NOT NULL, sample_ids BLOB NOT NULL, PRIMARY KEY (split, batch_id))"
        )
        existing = db.execute(
            "SELECT rows, batch_size, dataset_csv_sha256, model_revision FROM extraction_runs WHERE split=?",
            (split,),
        ).fetchone()
        if existing and existing != (len(records), batch_size, csv_hash, extractor.info.model_revision):
            raise ValueError(f"Cannot resume {split}: dataset, batch size, or model revision changed.")
        db.execute(
            "INSERT OR IGNORE INTO extraction_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (split, len(records), batch_size, csv_hash, extractor.info.model_id,
             extractor.info.model_revision, extractor.info.weights_sha256,
             device or extractor.device, "in_progress"),
        )
        legacy_partial = destination / ".partial"
        for legacy_file in sorted(legacy_partial.glob("batch_*.npz")):
            batch_id = int(legacy_file.stem.split("_")[1])
            already_saved = db.execute(
                "SELECT 1 FROM extraction_batches WHERE split=? AND batch_id=?",
                (split, batch_id),
            ).fetchone()
            if already_saved:
                continue
            with np.load(legacy_file, allow_pickle=False) as saved:
                db.execute(
                    "INSERT INTO extraction_batches VALUES (?, ?, ?, ?, ?, ?)",
                    (split, batch_id, _pack_array(saved["image"]), _pack_array(saved["text"]),
                     _pack_array(saved["labels"]), _pack_array(saved["sample_ids"])),
                )
        db.commit()
    image_features: list[np.ndarray] = []
    text_features: list[np.ndarray] = []
    labels: list[int] = []
    sample_ids: list[str] = []

    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        batch_id = start // batch_size
        with sqlite3.connect(checkpoint_path) as db:
            saved = db.execute(
                "SELECT image, text, labels, sample_ids FROM extraction_batches WHERE split=? AND batch_id=?",
                (split, batch_id),
            ).fetchone()
        if saved:
            image = _load_array(saved[0])
            text = _load_array(saved[1])
            batch_labels = _load_array(saved[2])
            batch_ids = _load_array(saved[3])
            if len(image) != len(batch) or len(text) != len(batch):
                raise ValueError(f"Invalid saved batch {split}:{batch_id}")
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
            with sqlite3.connect(checkpoint_path) as db:
                db.execute(
                    "INSERT INTO extraction_batches VALUES (?, ?, ?, ?, ?, ?)",
                    (split, batch_id, _pack_array(image), _pack_array(text),
                     _pack_array(batch_labels), _pack_array(batch_ids)),
                )
                db.commit()
        image_features.append(image)
        text_features.append(text)
        labels.extend(batch_labels.tolist())
        sample_ids.extend(batch_ids.tolist())
        print(f"{split}: {min(start + batch_size, len(records))}/{len(records)}", flush=True)

    with sqlite3.connect(checkpoint_path) as db:
        completed = db.execute(
            "SELECT COUNT(*) FROM extraction_batches WHERE split=?", (split,)
        ).fetchone()[0]
    if completed != expected_batches:
        raise RuntimeError(f"Cannot consolidate incomplete {split} extraction")
    _write_npy_atomic(destination / "image_embeddings.npy", np.concatenate(image_features))
    _write_npy_atomic(destination / "text_embeddings.npy", np.concatenate(text_features))
    _write_npy_atomic(destination / "labels.npy", np.asarray(labels, dtype=np.int64))
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
    with sqlite3.connect(checkpoint_path) as db:
        db.execute("UPDATE extraction_runs SET status='complete' WHERE split=?", (split,))
        db.commit()
    legacy_partial = destination / ".partial"
    if legacy_partial.is_dir():
        for legacy_file in legacy_partial.glob("batch_*.npz"):
            legacy_file.unlink()
        legacy_partial.rmdir()
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset" / "ooc_dataset")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "features")
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=("train", "val", "test"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="One SQLite file containing all resumable extraction batches.")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    checkpoint = args.checkpoint or (args.output_root / "extraction_checkpoint.sqlite")
    for split in args.splits:
        extract_split(args.dataset_root, args.output_root, split, args.batch_size, args.device, checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
