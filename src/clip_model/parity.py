"""The narrowed NewsCLIPpings parity diagnostic of Document 01 §12.

The diagnostic is a *fast*, offline check: it compares two sets of already
recorded feature arrays (our corpus embeddings and the reference embeddings)
joined on ``(caption_id, image_id)``.  It never loads the CLIP model, so it can
run in the fast tier.  The arrays are produced once by
``scripts/record_parity_reference.py``.

Only the text path is comparable.  The image path is *excluded with its stated
reason* (the corpus photographs were re-encoded at 256 px while the reference was
computed at original resolution), not silently skipped - see §12.8.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

#: Files that make up a recorded parity set.  Kept beside the reference so the
#: diagnostic can run offline against committed arrays.
IDS_FILE = "ids.json"
CORPUS_TEXT = "corpus_text_embeddings.npy"
REFERENCE_TEXT = "reference_text_embeddings.npy"
CORPUS_IMAGE = "corpus_image_embeddings.npy"
REFERENCE_IMAGE = "reference_image_embeddings.npy"

#: The provisional text-path threshold of §12.4 and the re-encode size that
#: makes the image path non-comparable (§12.8).
TEXT_PROVISIONAL_THRESHOLD = 0.99
IMAGE_EXCLUSION_REASON = (
    "The image path is excluded: the corpus photographs were re-encoded at "
    "256 px, while the reference features were computed on original-resolution "
    "files, so the two are not comparable (§12.8)."
)


def _read_reference_ids(reference_root: Path) -> list[tuple[str, str]]:
    payload = json.loads((reference_root / IDS_FILE).read_text(encoding="utf-8"))
    return list(zip(payload["caption_ids"], payload["image_ids"]))


def _read_dataset_ids(dataset_root: Path, split: str = "val") -> list[tuple[str, str]]:
    csv_path = Path(dataset_root) / f"{split}.csv"
    if not csv_path.is_file():
        return []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        return [
            (row.get("caption_id", ""), row.get("image_id", ""))
            for row in csv.DictReader(fh)
        ]


def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two equally-shaped matrices."""
    a = a.astype("float64")
    b = b.astype("float64")
    num = (a * b).sum(axis=1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    denom = np.where(denom == 0.0, 1.0, denom)
    return num / denom


def align_by_ids(dataset_root: str | Path, reference_root: str | Path) -> dict[str, Any]:
    """Join the reference and the corpus on ``(caption_id, image_id)``.

    The join is on identifiers, never on row order.  The report states the
    aligned row count and the coverage ratio so a thin overlap is visible.
    """
    reference_root = Path(reference_root)
    ref_ids = _read_reference_ids(reference_root)
    corpus_ids = _read_dataset_ids(dataset_root)

    corpus_set = set(corpus_ids)
    aligned = [key for key in ref_ids if key in corpus_set]
    coverage = (len(aligned) / len(ref_ids)) if ref_ids else 0.0
    return {
        "join_key": ["caption_id", "image_id"],
        "reference_rows": len(ref_ids),
        "corpus_rows": len(corpus_ids),
        "aligned_rows": len(aligned),
        "coverage_ratio": coverage,
    }


def compare_text_path(dataset_root: str | Path, reference_root: str | Path) -> dict[str, Any]:
    """Report text-path agreement against the provisional threshold."""
    reference_root = Path(reference_root)
    alignment = align_by_ids(dataset_root, reference_root)

    corpus = np.load(reference_root / CORPUS_TEXT, allow_pickle=False)
    reference = np.load(reference_root / REFERENCE_TEXT, allow_pickle=False)
    n = min(len(corpus), len(reference), alignment["aligned_rows"])

    sims = _cosine_matrix(corpus[:n], reference[:n])
    mean_cosine = float(sims.mean()) if n else 0.0
    return {
        "path": "text",
        "n_compared": int(n),
        "mean_cosine": mean_cosine,
        "min_cosine": float(sims.min()) if n else 0.0,
        "threshold": TEXT_PROVISIONAL_THRESHOLD,
        "excluded": False,
        "within_tolerance": bool(mean_cosine >= TEXT_PROVISIONAL_THRESHOLD),
    }


def compare_image_path(dataset_root: str | Path, reference_root: str | Path) -> dict[str, Any]:
    """The image path is excluded, with its reason stated."""
    reference_root = Path(reference_root)
    n = 0
    if (reference_root / CORPUS_IMAGE).is_file():
        n = int(np.load(reference_root / CORPUS_IMAGE, allow_pickle=False, mmap_mode="r").shape[0])
    return {
        "path": "image",
        "excluded": True,
        "reason": IMAGE_EXCLUSION_REASON,
        "n_rows": n,
        "within_tolerance": None,
    }


def run_diagnostic(
    dataset_root: str | Path,
    reference_root: str | Path,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Run the full parity diagnostic and write ``parity_report.json``.

    The report carries the preconditions, the metrics, the outcome, and any
    investigation notes, per §12.7.
    """
    dataset_root = Path(dataset_root)
    reference_root = Path(reference_root)

    preconditions = {
        "reference_present": reference_root.is_dir() and any(reference_root.glob("*.npy")),
        "dataset_present": dataset_root.is_dir(),
        "join": "on (caption_id, image_id), not row order",
        "image_path": "excluded with reason (256 px re-encode)",
    }

    alignment = align_by_ids(dataset_root, reference_root)
    text = compare_text_path(dataset_root, reference_root)
    image = compare_image_path(dataset_root, reference_root)

    outcome = "pass" if (
        preconditions["reference_present"]
        and alignment["aligned_rows"] > 0
        and text["within_tolerance"]
        and image["excluded"]
    ) else "investigate"

    report = {
        "preconditions": preconditions,
        "alignment": alignment,
        "metrics": {"text": text, "image": image},
        "outcome": outcome,
        "investigation_notes": []
        if outcome == "pass"
        else ["text-path mean cosine below the provisional threshold, or no aligned rows"],
    }

    if output is not None:
        out_dir = Path(output)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "parity_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    return report
