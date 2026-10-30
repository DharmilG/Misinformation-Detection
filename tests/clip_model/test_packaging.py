"""The classifier artifact packaging gates, copy schema, and release layout (Document 03 §17, §19, §25).

Tests complete release file layout, machine-readable copy.json schema,
packaging gates against local absolute paths, and synthetic fixture rights.
"""

from pathlib import Path
import pytest

from tests.ids import case

RELEASE_FILES = {
    "release.json",
    "weights.pt",
    "config.json",
    "calibration.json",
    "metrics.json",
    "copy.json",
    "MODEL_CARD.md",
    "requirements-lock.txt",
    "reference_fixture",
}

COPY_KEYS = {
    "copy_version",
    "what_this_measures",
    "not_a_truth_claim",
    "probability_note",
    "score_note",
}


def _packaging_module():
    import importlib
    return importlib.import_module("classifier_packaging")


@case(
    "packaging.module_exists",
    "F2",
    "fast",
    "doc/03 §17",
    "The project provides a packaging module to create, validate, and verify release archives.",
)
def test_packaging_module_exists():
    _packaging_module()


@case(
    "packaging.release_layout_complete",
    "F2",
    "fast",
    "doc/03 §19.1",
    "The release layout contains exactly the required 9 files and directories.",
)
def test_release_layout_complete(tmp_path):
    mod = _packaging_module()
    layout = mod.get_expected_release_files()
    assert set(layout) == RELEASE_FILES


@case(
    "packaging.copy_file_schema",
    "F2",
    "fast",
    "doc/03 §19.1.1",
    "The copy.json file contains all required non-empty explanation string keys.",
)
def test_copy_file_schema():
    mod = _packaging_module()
    copy_payload = mod.generate_copy_json()
    missing = COPY_KEYS - set(copy_payload.keys())
    assert not missing, "copy.json is missing keys: %s" % missing
    for k in COPY_KEYS:
        assert isinstance(copy_payload[k], str) and len(copy_payload[k].strip()) > 0, "%s is empty or not a string" % k


@case(
    "packaging.gate_no_absolute_paths",
    "F2",
    "fast",
    "doc/03 §25",
    "Packaging Gate 3 asserts that no private local machine absolute paths are present in release metadata.",
)
def test_gate_no_absolute_paths():
    mod = _packaging_module()
    # Payload with Windows absolute path fails gate
    bad_metadata = {"config": {"path": "C:\\Users\\Developer\\Project\\data"}}
    assert mod.check_absolute_paths_gate(bad_metadata) is False

    # Payload with relative paths passes gate
    clean_metadata = {"config": {"path": "data/processed/features"}}
    assert mod.check_absolute_paths_gate(clean_metadata) is True


@case(
    "packaging.gate_fixture_rights",
    "F2",
    "fast",
    "doc/03 §20.3, §25",
    "Packaging Gate 7 asserts that reference fixture samples are synthetic with committed generator.py.",
)
def test_gate_fixture_rights(tmp_path):
    mod = _packaging_module()
    fixture_dir = tmp_path / "reference_fixture"
    fixture_dir.mkdir()
    # Missing generator.py fails gate
    assert mod.check_fixture_rights_gate(fixture_dir) is False

    # Adding committed generator script passes gate
    (fixture_dir / "generator.py").write_text("# synthetic image generator", encoding="utf-8")
    (fixture_dir / "README.md").write_text("Synthetic rights free", encoding="utf-8")
    assert mod.check_fixture_rights_gate(fixture_dir) is True
