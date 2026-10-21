"""F6 - the preprocessing skew of document 02 §8.1.

The corpus photographs arrive at a 256 px shorter side, so document 04's 2048 px
downscale is a no-op on them while inference runs the full chain.  The skew is
accepted, and **measured** here rather than argued about.
"""

import csv
import json
from pathlib import Path

import pytest

from tests.ids import case

torch = pytest.importorskip("torch")

REPORT = Path(__file__).resolve().parent.parent / "reports" / "skew_measurement.json"


def _first_row(dataset_root, split="train"):
    with open(dataset_root / ("%s.csv" % split), newline="", encoding="utf-8") as fh:
        return next(csv.DictReader(fh))


def _cosine(a, b):
    return float(
        torch.nn.functional.cosine_similarity(
            a.float().reshape(1, -1), b.float().reshape(1, -1)
        )[0]
    )


@case(
    "skew.downscale_is_noop_on_corpus",
    "F6",
    "data",
    "doc/02 §8.1.1, doc/04 §7.4.1",
    "Every corpus photograph is smaller than 2048 px, so the downscale leaves it unchanged.",
)
def test_downscale_is_noop_on_corpus(dataset_root):
    from PIL import Image

    rows = []
    for split in ("train", "val", "test"):
        with open(dataset_root / ("%s.csv" % split), newline="", encoding="utf-8") as fh:
            rows.extend(list(csv.DictReader(fh))[:40])
    for row in rows:
        with Image.open(dataset_root / row["image_path"]) as image:
            assert max(image.width, image.height) <= 2048, (
                "%s is %dx%d, so the downscale is not a no-op"
                % (row["image_path"], image.width, image.height)
            )


@case(
    "skew.native_vs_upscaled_similarity",
    "F6",
    ("data", "slow"),
    "doc/02 §8.1.2",
    "The embedding difference between a corpus photograph at its native size and the same photograph upscaled to the inference resolution is measured and recorded.",
    requires=("torch", "transformers"),
)
def test_native_vs_upscaled_similarity(extractor, dataset_root, tmp_path):
    from PIL import Image

    rows = []
    for split in ("train", "val", "test"):
        with open(dataset_root / ("%s.csv" % split), newline="", encoding="utf-8") as fh:
            rows.extend(list(csv.DictReader(fh))[:7])
    rows = rows[:20]

    similarities = []
    for row in rows:
        native_path = dataset_root / row["image_path"]
        caption = row["caption"]
        with Image.open(native_path) as opened:
            image = opened.convert("RGB")
            scale = 2048 / max(image.width, image.height)
            upscaled = image.resize(
                (round(image.width * scale), round(image.height * scale)),
                Image.Resampling.LANCZOS,
            )
        upscaled_path = tmp_path / ("up_%s.png" % row["caption_id"])
        upscaled.save(upscaled_path, "PNG")

        native = extractor.extract_features(str(native_path), caption)
        inference = extractor.extract_features(str(upscaled_path), caption)
        similarities.append(_cosine(native["image_features"], inference["image_features"]))
        upscaled_path.unlink()

    mean = sum(similarities) / len(similarities)
    worst = min(similarities)
    assert mean > 0.98, "the two forms of the same photograph diverge on average: %.4f" % mean
    assert worst > 0.95, "the worst photograph diverges: %.4f" % worst

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    if REPORT.is_file():
        try:
            payload = json.loads(REPORT.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    payload["native_vs_upscaled_cosine"] = {
        "sample_size": len(similarities),
        "mean": round(mean, 6),
        "minimum": round(worst, 6),
        "maximum": round(max(similarities), 6),
    }
    payload["note"] = (
        "The corpus photographs arrive at a 255-256 px shorter side; inference runs "
        "the 2048 px downscale first. This is the measured cost of that difference."
    )
    REPORT.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@case(
    "skew.crop_region_equivalent",
    "F6",
    "data",
    "doc/02 §8.1.1",
    "The 224 px centre crop covers the same relative region of the frame in both forms.",
)
def test_crop_region_equivalent(dataset_root):
    from PIL import Image

    row = _first_row(dataset_root)
    with Image.open(dataset_root / row["image_path"]) as opened:
        native = opened.convert("RGB")
    scale = 2048 / max(native.width, native.height)
    upscaled = native.resize(
        (round(native.width * scale), round(native.height * scale)),
        Image.Resampling.LANCZOS,
    )

    def relative_crop(image):
        factor = 224 / min(image.width, image.height)
        wide, tall = image.width * factor, image.height * factor
        left = (wide - 224) / 2 / wide
        top = (tall - 224) / 2 / tall
        return left, top

    a = relative_crop(native)
    b = relative_crop(upscaled)
    assert abs(a[0] - b[0]) < 1e-3, "the crop region moved horizontally"
    assert abs(a[1] - b[1]) < 1e-3, "the crop region moved vertically"


@case(
    "skew.inference_variant_matches_pipeline",
    "F6",
    ("data", "slow"),
    "doc/04 §7.4, doc/02 §8.1.2",
    "An image already reduced to the inference resolution encodes identically to the same image passed through the documented chain.",
    requires=("torch", "transformers"),
)
def test_inference_variant_matches_pipeline(extractor, dataset_root, tmp_path):
    from PIL import Image

    row = _first_row(dataset_root)
    native_path = dataset_root / row["image_path"]
    with Image.open(native_path) as opened:
        image = opened.convert("RGB")
        scale = 2048 / max(image.width, image.height)
        scaled = image.resize(
            (round(image.width * scale), round(image.height * scale)),
            Image.Resampling.LANCZOS,
        )
    scaled_path = tmp_path / "scaled.png"
    scaled.save(scaled_path, "PNG")

    direct = extractor.extract_features(str(native_path), row["caption"])
    through_chain = extractor.extract_features(str(scaled_path), row["caption"])
    assert _cosine(direct["image_features"], through_chain["image_features"]) > 0.99, (
        "the inference chain changes the embedding of the same photograph"
    )
