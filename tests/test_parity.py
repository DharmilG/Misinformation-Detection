"""F8 - the narrowed NewsCLIPpings parity diagnostic of document 01 §12.

The image path cannot be compared: the corpus photographs were re-encoded at
256 px while the reference features were computed on original-resolution files.
The text path remains comparable, and the image path is **excluded with its
stated reason** rather than silently skipped.
"""

import json
from pathlib import Path

import pytest

from tests.ids import case, known_gap

REFERENCE_ROOT = Path(__file__).resolve().parent.parent / "data" / "reference" / "newsclippings"


def _reference_available():
    return REFERENCE_ROOT.is_dir() and any(REFERENCE_ROOT.glob("*.npy"))


@case(
    "parity.skip_when_reference_absent",
    "F8",
    "fast",
    "doc/01 §15.7",
    "The diagnostic skips cleanly, with a stated reason, when the reference features are not present locally.",
)
def test_skip_when_reference_absent():
    if _reference_available():
        pytest.skip("the reference features are present, so the diagnostic runs")
    pytest.skip("the NewsCLIPpings reference features are not present at %s" % REFERENCE_ROOT)


@known_gap(
    "parity.id_alignment",
    "F8",
    "fast",
    "doc/01 §12.2",
    "The diagnostic joins the reference and the corpus on (caption_id, image_id), not on row order, and reports the coverage.",
    "the diagnostic does not exist, so there is no join",
)
def test_id_alignment(dataset_root):
    from parity import align_by_ids

    report = align_by_ids(dataset_root, REFERENCE_ROOT)
    assert report["aligned_rows"] > 0, "no rows aligned on (caption_id, image_id)"
    assert report["coverage_ratio"] > 0.0


@known_gap(
    "parity.text_path_within_tolerance",
    "F8",
    "fast",
    "doc/01 §12.4",
    "The text embeddings agree with the reference within the provisional threshold.",
    "the diagnostic does not exist, so the text path is never compared",
)
def test_text_path_within_tolerance(dataset_root):
    from parity import compare_text_path

    report = compare_text_path(dataset_root, REFERENCE_ROOT)
    assert report["mean_cosine"] >= 0.99, "the text path diverged: %.4f" % report["mean_cosine"]


@known_gap(
    "parity.image_path_excluded",
    "F8",
    "fast",
    "doc/01 §12.8",
    "The image path is excluded from the comparison, and the report states why: the corpus photographs were re-encoded at 256 px.",
    "the diagnostic does not exist, so the exclusion cannot be stated",
)
def test_image_path_excluded(dataset_root):
    from parity import compare_image_path

    report = compare_image_path(dataset_root, REFERENCE_ROOT)
    assert report["excluded"] is True, "the image path was not excluded"
    assert "256" in report["reason"], "the exclusion does not state its reason"


@known_gap(
    "parity.report_generated",
    "F8",
    "fast",
    "doc/01 §12.7",
    "The diagnostic writes a report containing the preconditions, the metrics, the outcome, and any investigation notes.",
    "the diagnostic does not exist, so no report is produced",
)
def test_report_generated(dataset_root, tmp_path):
    from parity import run_diagnostic

    report = run_diagnostic(dataset_root, REFERENCE_ROOT, output=tmp_path)
    assert (tmp_path / "parity_report.json").is_file()
    payload = json.loads((tmp_path / "parity_report.json").read_text(encoding="utf-8"))
    for key in ("preconditions", "metrics", "outcome"):
        assert key in payload, "the report is missing %s" % key
