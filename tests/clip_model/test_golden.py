"""F9 - the golden-vector regression of document 01 §15.5.

The primary, offline, both-modality check on the encoder chain.  The parity
diagnostic needs a download and covers the text path only, so it cannot stand
in for this.
"""

import json
from pathlib import Path

import pytest

from tests.fixtures import golden
from tests.ids import case, known_gap

torch = pytest.importorskip("torch")


def _cosine(a, b):
    return float(
        torch.nn.functional.cosine_similarity(
            a.float().reshape(1, -1), b.float().reshape(1, -1)
        )[0]
    )


@case(
    "golden.detects_change",
    "F9",
    "fast",
    "doc/01 §15.5",
    "The comparison detects a deliberately altered baseline, so the regression cannot pass silently.",
)
def test_detects_change():
    expected = {"image": [0.0] * 512}
    actual = {"image": [1.0] * 512}
    result = golden.compare(expected, actual, golden.TOLERANCE)
    assert result["matches"] is False, "an altered baseline was accepted"
    assert "image" in result["mismatched"], "the comparison did not name the pair that moved"

    unchanged = golden.compare(expected, dict(expected), golden.TOLERANCE)
    assert unchanged["matches"] is True, "an identical baseline was rejected"


@known_gap(
    "golden.baseline_present",
    "F9",
    "fast",
    "doc/01 §15.5",
    "A recorded baseline of image and caption embeddings exists, so the regression has something to compare against.",
    "the baseline has not been recorded yet; it requires one run of the encoder on the reference machine",
)
def test_baseline_present():
    baseline = golden.load()
    assert baseline is not None, "the golden baseline has not been recorded"
    assert len(baseline["pairs"]) == len(golden.PAIRS)


@known_gap(
    "golden.reproduces_within_tolerance",
    "F9",
    "slow",
    "doc/01 §14.3, §15.5",
    "The current build reproduces the recorded baseline within the tolerance.",
    "the baseline has not been recorded yet",
    requires=("torch", "transformers"),
)
def test_reproduces_within_tolerance(extractor, tmp_path):
    baseline = golden.load()
    assert baseline is not None, "the golden baseline has not been recorded"

    from tests.fixtures import images

    worst = 1.0
    for record in baseline["pairs"]:
        image = _build(image_record=record["image"])
        path = images.write(tmp_path / ("%s.png" % record["name"]), image)
        out = extractor.extract_features(str(path), record["caption"])
        similarity = _cosine(
            out["image_features"], torch.tensor(record["image_embedding"]).reshape(1, -1)
        )
        worst = min(worst, similarity)
    assert worst >= 1.0 - golden.TOLERANCE, "the worst case diverged: %.6f" % worst


def _build(image_record):
    """Rebuild one of the recorded images."""
    from tests.fixtures import images

    kind = image_record["kind"]
    if kind == "solid_rgb":
        return images.solid_rgb(tuple(image_record["size"]), tuple(image_record["color"]))
    if kind == "wide_aspect":
        return images.wide_aspect(tuple(image_record["size"]))
    if kind == "grayscale":
        return images.grayscale(tuple(image_record["size"]))
    raise AssertionError("unknown image kind: %s" % kind)
