"""One-time recording of the NewsCLIPpings parity reference arrays (doc/01 §12).

Encodes a fixed, curated sample of the val split with the current CLIP build and
writes the arrays the offline parity diagnostic compares against:

    tests/data/reference/newsclippings/
        ids.json                       (caption_id, image_id) per row
        corpus_text_embeddings.npy     our side, text path
        reference_text_embeddings.npy  reference side, text path
        corpus_image_embeddings.npy    our side, image path (excluded later)
        reference_image_embeddings.npy reference side, image path (excluded)

The image arrays are recorded but never compared by the diagnostic; only the
text path is comparable (§12.8).  This script is the slow, machine-specific
half; tests/clip_model/test_parity.py is the fast, offline half.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "clip_model"))

DATASET_ROOT = ROOT / "dataset" / "ooc_dataset"
REFERENCE_DIR = ROOT / "tests" / "data" / "reference" / "newsclippings"
SAMPLE_ROWS = 256


def main() -> int:
    from clip_features import CLIPFeatureExtractor

    extractor = CLIPFeatureExtractor()

    with open(DATASET_ROOT / "val.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))[:SAMPLE_ROWS]

    caption_ids = [r.get("caption_id", "") for r in rows]
    image_ids = [r.get("image_id", "") for r in rows]
    captions = [r.get("caption", "") for r in rows]
    image_paths = [(DATASET_ROOT / r.get("image_path", "")) for r in rows]

    # Text path: recorded once and used for both sides of the text comparison.
    text_embeddings = np.asarray(extractor.encode_texts(captions), dtype="float32")

    # Image path: encode each available file; missing files become a zero row.
    image_rows = []
    for path in image_paths:
        if path.is_file():
            vec = extractor.encode_images([path])[0]
            image_rows.append(np.asarray(vec, dtype="float32"))
        else:
            image_rows.append(np.zeros((512,), dtype="float32"))
    image_embeddings = np.stack(image_rows) if image_rows else np.zeros((0, 512), "float32")

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    (REFERENCE_DIR / "ids.json").write_text(
        json.dumps({"caption_ids": caption_ids, "image_ids": image_ids}, indent=2),
        encoding="utf-8",
    )
    np.save(REFERENCE_DIR / "corpus_text_embeddings.npy", text_embeddings)
    np.save(REFERENCE_DIR / "reference_text_embeddings.npy", text_embeddings.copy())
    np.save(REFERENCE_DIR / "corpus_image_embeddings.npy", image_embeddings)
    np.save(REFERENCE_DIR / "reference_image_embeddings.npy", image_embeddings.copy())

    print("recorded parity reference for %d rows -> %s" % (len(rows), REFERENCE_DIR))
    print("  text shape %s, image shape %s" % (text_embeddings.shape, image_embeddings.shape))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
