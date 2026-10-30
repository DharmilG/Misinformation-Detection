"""
tests/test_comparison.py
=========================
Tests for src/reporting/comparison.py.

Spec: 05_REPORTING_AND_UI §8.

Fast tests — no model, no network, no I/O.
"""
from __future__ import annotations

import pytest

from src.reporting import copy as C
from src.reporting.comparison import (
    PERMANENT_NOTE,
    compute_comparison,
)
from src.reporting.model import AssessmentSection, ClassifierSection


# ── Helpers ───────────────────────────────────────────────────────────────

def _make_clf(
    status: str = "OK",
    label: str | None = "MATCHED",
) -> ClassifierSection:
    return ClassifierSection(
        status=status,
        run_id="r1",
        model_version="v1",
        encoder_revision="enc1",
        label=label,
        score=0.75 if label == "MATCHED" else 0.25,
    )


def _make_asm(
    status: str = "OK",
    outcome: str | None = "SUPPORTED",
) -> AssessmentSection:
    return AssessmentSection(
        status=status,
        prompt_version="pv1",
        scope_qualifier=C.SCOPE_QUALIFIER,
        outcome=outcome,
    )


# ── Five-state tests ──────────────────────────────────────────────────────

class TestFiveStates:

    def test_consistent_pattern_both_match_support(self) -> None:
        comp = compute_comparison(_make_clf(label="MATCHED"), _make_asm(outcome="SUPPORTED"))
        assert comp.state == "CONSISTENT_PATTERN"

    def test_consistent_pattern_both_mismatch_contradict(self) -> None:
        comp = compute_comparison(_make_clf(label="MISMATCHED"), _make_asm(outcome="CONTRADICTED"))
        assert comp.state == "CONSISTENT_PATTERN"

    def test_different_pattern_mismatch_but_supported(self) -> None:
        comp = compute_comparison(_make_clf(label="MISMATCHED"), _make_asm(outcome="SUPPORTED"))
        assert comp.state == "DIFFERENT_PATTERN"

    def test_different_pattern_matched_but_contradicted(self) -> None:
        comp = compute_comparison(_make_clf(label="MATCHED"), _make_asm(outcome="CONTRADICTED"))
        assert comp.state == "DIFFERENT_PATTERN"

    def test_assessment_unresolved_inconclusive(self) -> None:
        comp = compute_comparison(_make_clf(label="MATCHED"), _make_asm(outcome="INCONCLUSIVE"))
        assert comp.state == "ASSESSMENT_UNRESOLVED"

    def test_assessment_unresolved_conflicting(self) -> None:
        comp = compute_comparison(_make_clf(label="MATCHED"), _make_asm(outcome="CONFLICTING"))
        assert comp.state == "ASSESSMENT_UNRESOLVED"

    def test_assessment_unavailable_when_asm_failed(self) -> None:
        comp = compute_comparison(_make_clf(label="MATCHED"), _make_asm(status="FAILED"))
        assert comp.state == "ASSESSMENT_UNAVAILABLE"

    def test_assessment_unavailable_when_asm_skipped(self) -> None:
        comp = compute_comparison(_make_clf(label="MATCHED"), _make_asm(status="SKIPPED"))
        assert comp.state == "ASSESSMENT_UNAVAILABLE"

    def test_classifier_unavailable_when_clf_failed(self) -> None:
        comp = compute_comparison(_make_clf(status="FAILED", label=None), _make_asm())
        assert comp.state == "CLASSIFIER_UNAVAILABLE"

    def test_classifier_unavailable_when_clf_unavailable(self) -> None:
        comp = compute_comparison(_make_clf(status="UNAVAILABLE", label=None), _make_asm())
        assert comp.state == "CLASSIFIER_UNAVAILABLE"

    def test_classifier_unavailable_takes_priority_over_asm_unavailable(self) -> None:
        """CLASSIFIER_UNAVAILABLE wins regardless of assessment state."""
        comp = compute_comparison(
            _make_clf(status="UNAVAILABLE", label=None),
            _make_asm(status="FAILED"),
        )
        assert comp.state == "CLASSIFIER_UNAVAILABLE"


# ── Sentence tests ─────────────────────────────────────────────────────────

class TestSentences:

    def test_all_states_have_non_empty_sentence(self) -> None:
        cases = [
            (_make_clf(label="MATCHED"),    _make_asm(outcome="SUPPORTED")),
            (_make_clf(label="MISMATCHED"), _make_asm(outcome="CONTRADICTED")),
            (_make_clf(label="MISMATCHED"), _make_asm(outcome="SUPPORTED")),
            (_make_clf(label="MATCHED"),    _make_asm(outcome="CONTRADICTED")),
            (_make_clf(label="MATCHED"),    _make_asm(outcome="INCONCLUSIVE")),
            (_make_clf(label="MATCHED"),    _make_asm(status="FAILED")),
            (_make_clf(status="FAILED", label=None), _make_asm()),
        ]
        for clf, asm in cases:
            comp = compute_comparison(clf, asm)
            assert comp.sentence, f"Empty sentence for state {comp.state}"

    def test_forbidden_words_absent_from_sentences(self) -> None:
        cases = [
            (_make_clf(label="MATCHED"),    _make_asm(outcome="SUPPORTED")),
            (_make_clf(label="MISMATCHED"), _make_asm(outcome="CONTRADICTED")),
            (_make_clf(label="MISMATCHED"), _make_asm(outcome="SUPPORTED")),
            (_make_clf(label="MATCHED"),    _make_asm(outcome="CONTRADICTED")),
            (_make_clf(label="MATCHED"),    _make_asm(outcome="INCONCLUSIVE")),
            (_make_clf(label="MATCHED"),    _make_asm(status="FAILED")),
            (_make_clf(status="FAILED", label=None), _make_asm()),
        ]
        for clf, asm in cases:
            comp = compute_comparison(clf, asm)
            sentence_lower = comp.sentence.lower()
            for word in C.FORBIDDEN_COMPARISON_WORDS:
                assert word not in sentence_lower, (
                    f"Forbidden word '{word}' found in sentence for state {comp.state}: "
                    f"'{comp.sentence}'"
                )


# ── Permanent note ─────────────────────────────────────────────────────────

class TestPermanentNote:

    def test_permanent_note_is_non_empty_string(self) -> None:
        assert isinstance(PERMANENT_NOTE, str)
        assert len(PERMANENT_NOTE.strip()) > 0

    def test_permanent_note_does_not_use_forbidden_words(self) -> None:
        note_lower = PERMANENT_NOTE.lower()
        for word in C.FORBIDDEN_COMPARISON_WORDS:
            assert word not in note_lower, (
                f"Forbidden word '{word}' found in PERMANENT_NOTE"
            )

    def test_permanent_note_does_not_contain_verdict_phrases(self) -> None:
        note_lower = PERMANENT_NOTE.lower()
        for phrase in C.FORBIDDEN_VERDICT_PHRASES:
            assert phrase not in note_lower
