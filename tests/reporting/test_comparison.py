"""The comparison section and descriptive relationship mapping of Document 05 §8.

Tests the two direction axes (classifier direction and assessment direction),
the exhaustive 9-cell mapping producing exactly 5 comparison states, the approved
descriptive sentences, absence of forbidden terms (agree, confirm, verified, prove),
and presence of the permanent note ('Two signals pointing the same way does not make
either one correct').
"""

import pytest

from tests.ids import case, known_gap


def _comparison_module():
    import importlib
    return importlib.import_module("src.reporting.comparison")


@known_gap(
    "comparison.direction_axes",
    "F2",
    "fast",
    "doc/05 §8.1",
    "Classifier direction maps to MISMATCH_INDICATED, NO_MISMATCH_INDICATED, or UNAVAILABLE; assessment direction maps to CONTRADICTS_CLAIM, SUPPORTS_CLAIM, UNRESOLVED, or UNAVAILABLE.",
    "comparison direction axis mapping is not yet implemented",
)
def test_comparison_direction_axes():
    mod = _comparison_module()
    assert hasattr(mod, "ClassifierDirection")
    assert hasattr(mod, "AssessmentDirection")
    clf_dirs = {d.name for d in mod.ClassifierDirection}
    asmt_dirs = {d.name for d in mod.AssessmentDirection}
    assert clf_dirs == {"MISMATCH_INDICATED", "NO_MISMATCH_INDICATED", "UNAVAILABLE"}
    assert asmt_dirs == {"CONTRADICTS_CLAIM", "SUPPORTS_CLAIM", "UNRESOLVED", "UNAVAILABLE"}


@known_gap(
    "comparison.five_states_defined",
    "F2",
    "fast",
    "doc/05 §8.2",
    "Five comparison states: CONSISTENT_PATTERN, DIFFERENT_PATTERN, ASSESSMENT_UNRESOLVED, ASSESSMENT_UNAVAILABLE, CLASSIFIER_UNAVAILABLE.",
    "comparison state definitions are not yet implemented",
)
def test_comparison_five_states_defined():
    mod = _comparison_module()
    assert hasattr(mod, "ComparisonState")
    expected = {
        "CONSISTENT_PATTERN",
        "DIFFERENT_PATTERN",
        "ASSESSMENT_UNRESOLVED",
        "ASSESSMENT_UNAVAILABLE",
        "CLASSIFIER_UNAVAILABLE",
    }
    actual = {s.name for s in mod.ComparisonState}
    assert actual == expected


@known_gap(
    "comparison.exhaustive_mapping_matrix",
    "F2",
    "fast",
    "doc/05 §8.2",
    "Exhaustive mapping defines exactly 5 comparison states across all 9 direction combinations.",
    "comparison mapping matrix is not yet implemented",
)
def test_comparison_exhaustive_mapping_matrix():
    mod = _comparison_module()
    assert hasattr(mod, "map_comparison_state")
    
    # 9 cells from doc 05 §8.2:
    cases = [
        ("MISMATCH_INDICATED", "CONTRADICTS_CLAIM", "CONSISTENT_PATTERN"),
        ("MISMATCH_INDICATED", "SUPPORTS_CLAIM", "DIFFERENT_PATTERN"),
        ("MISMATCH_INDICATED", "UNRESOLVED", "ASSESSMENT_UNRESOLVED"),
        ("MISMATCH_INDICATED", "UNAVAILABLE", "ASSESSMENT_UNAVAILABLE"),
        ("NO_MISMATCH_INDICATED", "SUPPORTS_CLAIM", "CONSISTENT_PATTERN"),
        ("NO_MISMATCH_INDICATED", "CONTRADICTS_CLAIM", "DIFFERENT_PATTERN"),
        ("NO_MISMATCH_INDICATED", "UNRESOLVED", "ASSESSMENT_UNRESOLVED"),
        ("NO_MISMATCH_INDICATED", "UNAVAILABLE", "ASSESSMENT_UNAVAILABLE"),
        ("UNAVAILABLE", "CONTRADICTS_CLAIM", "CLASSIFIER_UNAVAILABLE"),
        ("UNAVAILABLE", "SUPPORTS_CLAIM", "CLASSIFIER_UNAVAILABLE"),
        ("UNAVAILABLE", "UNRESOLVED", "CLASSIFIER_UNAVAILABLE"),
        ("UNAVAILABLE", "UNAVAILABLE", "CLASSIFIER_UNAVAILABLE"),
    ]
    for clf_dir, asmt_dir, expected_state in cases:
        assert mod.map_comparison_state(clf_dir, asmt_dir) == expected_state


@known_gap(
    "comparison.wording_no_forbidden_terms",
    "F2",
    "fast",
    "doc/05 §8.3",
    "Comparison sentences describe directional relationship without forbidden terms (agree, confirm, verified, prove).",
    "comparison sentence generation is not yet implemented",
)
def test_comparison_wording_no_forbidden_terms():
    mod = _comparison_module()
    assert hasattr(mod, "get_comparison_sentence")
    forbidden = ["agree", "confirm", "verified", "prove"]
    states = [
        "CONSISTENT_PATTERN",
        "DIFFERENT_PATTERN",
        "ASSESSMENT_UNRESOLVED",
        "ASSESSMENT_UNAVAILABLE",
        "CLASSIFIER_UNAVAILABLE",
    ]
    for state in states:
        sentence = mod.get_comparison_sentence(state).lower()
        for term in forbidden:
            assert term not in sentence, f"Forbidden term '{term}' found in {state} sentence: {sentence}"


@case(
    "comparison.permanent_note_present",
    "F2",
    "fast",
    "doc/05 §8.4",
    "Comparison section always includes the permanent note: 'Two signals pointing the same way does not make either one correct.'",
)
def test_comparison_permanent_note_present():
    mod = _comparison_module()
    assert hasattr(mod, "PERMANENT_NOTE")
    expected_note = "Two signals pointing the same way does not make either one correct."
    assert mod.PERMANENT_NOTE == expected_note


@known_gap(
    "comparison.meaningful_distinction_not_conflict",
    "F2",
    "fast",
    "doc/05 §8.5",
    "A coherent image-caption pair whose claim is contradicted yields DIFFERENT_PATTERN as a meaningful distinction, not a system failure.",
    "different pattern explanatory semantics are not yet implemented",
)
def test_comparison_meaningful_distinction_not_conflict():
    mod = _comparison_module()
    assert hasattr(mod, "get_comparison_sentence")
    sentence = mod.get_comparison_sentence("DIFFERENT_PATTERN")
    assert "one is about the image–caption relationship, the other is about the claim" in sentence


@known_gap(
    "comparison.classifier_unavailable_preserves_assessment",
    "F2",
    "fast",
    "doc/05 §8.2",
    "CLASSIFIER_UNAVAILABLE comparison state explicitly states claim assessment below is unaffected.",
    "classifier unavailable comparison sentence is not yet implemented",
)
def test_comparison_classifier_unavailable_preserves_assessment():
    mod = _comparison_module()
    assert hasattr(mod, "get_comparison_sentence")
    sentence = mod.get_comparison_sentence("CLASSIFIER_UNAVAILABLE")
    assert "unaffected" in sentence.lower()
