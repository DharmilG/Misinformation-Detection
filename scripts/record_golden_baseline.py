"""One-time recording of the golden-vector baseline for F9 (doc/01 §15.5).

Rebuilds each reference pair exactly as tests/clip_model/test_golden.py does,
encodes it with the current CLIP build, and stores the image embeddings so the
reproducibility check has a committed baseline to compare against.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "clip_model"))

from tests.fixtures import golden, images  # noqa: E402


def build(record):
    kind = record["kind"]
    if kind == "solid_rgb":
        return images.solid_rgb(tuple(record["size"]), tuple(record["color"]))
    if kind == "wide_aspect":
        return images.wide_aspect(tuple(record["size"]))
    if kind == "grayscale":
        return images.grayscale(tuple(record["size"]))
    raise AssertionError("unknown image kind: %s" % kind)


def main() -> int:
    from clip_features import CLIPFeatureExtractor

    extractor = CLIPFeatureExtractor()
    pairs = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for spec in golden.PAIRS:
            image = build(spec["image"])
            path = images.write(tmp_path / ("%s.png" % spec["name"]), image)
            out = extractor.extract_features(str(path), spec["caption"])
            embedding = out["image_features"].reshape(-1).tolist()
            pairs.append({
                "name": spec["name"],
                "image": spec["image"],
                "caption": spec["caption"],
                "image_embedding": embedding,
            })
            print("recorded %s (%d-d)" % (spec["name"], len(embedding)))

    baseline = golden.store({
        "model_id": extractor.info.model_id,
        "model_revision": extractor.info.model_revision,
        "tolerance": golden.TOLERANCE,
        "pairs": pairs,
    })
    print("wrote baseline: %s" % baseline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
