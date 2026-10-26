"""The canonical explanation copy and end-to-end inference handoff of Document 04 §25.

Tests the five canonical explanation copy keys, fallback copy schema,
and rejection of malformed copy files missing mandatory limitation statements.
"""

import pytest

from tests.ids import case, known_gap

REQUIRED_COPY_KEYS = {
    "copy_version",
    "what_this_measures",
    "not_a_truth_claim",
    "probability_note",
    "score_note",
}


def _copy_module():
    import importlib
    return importlib.import_module("inference_copy")


@known_gap(
    "inference_e2e.module_exists",
    "F2",
    "fast",
    "doc/04 §25",
    "The project provides a module handling canonical explanation copy and validation.",
    "there is no inference_copy module",
)
def test_inference_e2e_module_exists():
    _copy_module()


@known_gap(
    "inference_e2e.fallback_copy_schema",
    "F2",
    "fast",
    "doc/04 §25.2",
    "FALLBACK_COPY provides all five required keys with copy_version == 'fallback'.",
    "fallback copy is not defined",
)
def test_fallback_copy_schema():
    mod = _copy_module()
    fallback = mod.FALLBACK_COPY
    missing = REQUIRED_COPY_KEYS - set(fallback.keys())
    assert not missing, "FALLBACK_COPY missing keys: %s" % missing
    assert fallback["copy_version"] == "fallback"


@known_gap(
    "inference_e2e.copy_keys_non_empty",
    "F2",
    "fast",
    "doc/04 §25.1",
    "Every required copy key is a non-empty string and not whitespace only.",
    "copy key validation is not implemented",
)
def test_copy_keys_non_empty():
    mod = _copy_module()
    copy_data = mod.load_copy_file()
    for key in REQUIRED_COPY_KEYS:
        val = copy_data.get(key)
        assert isinstance(val, str) and len(val.strip()) > 0, "key %s is empty or not string" % key


@known_gap(
    "inference_e2e.malformed_copy_raises",
    "F2",
    "fast",
    "doc/04 §25.1",
    "A copy file missing any required key or having empty strings raises CheckpointIntegrityError.",
    "copy integrity validation is not implemented",
)
def test_malformed_copy_raises():
    mod = _copy_module()
    bad_payload = {"copy_version": "1.0.0", "what_this_measures": "Something"}  # missing not_a_truth_claim etc.
    with pytest.raises(Exception) as excinfo:
        mod.validate_copy_payload(bad_payload)
    assert "Integrity" in type(excinfo.value).__name__ or "copy" in str(excinfo.value).lower()
