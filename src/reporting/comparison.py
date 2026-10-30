"""
src/reporting/comparison.py
============================
Exhaustive five-state comparison mapping.

Spec: 05_REPORTING_AND_UI.md §8

The five states (§8.2):
    CONSISTENT_PATTERN
    DIFFERENT_PATTERN
    ASSESSMENT_UNRESOLVED
    ASSESSMENT_UNAVAILABLE
    CLASSIFIER_UNAVAILABLE

Rules:
  - If classifier is not OK  →  CLASSIFIER_UNAVAILABLE  (regardless of assessment)
  - If assessment is not OK  →  ASSESSMENT_UNAVAILABLE
  - If assessment outcome is INCONCLUSIVE or CONFLICTING  →  ASSESSMENT_UNRESOLVED
  - Otherwise map (clf_direction, asm_direction) via the §8.2 table

Forbidden words in any sentence: "agree", "confirm", "verified", "prove".
The permanent note (§8.4) is a module-level constant consumed by both the
renderer and the tests — it is never constructed inline in app.py.
"""
from __future__ import annotations

from src.reporting.model import ClassifierSection, AssessmentSection, ComparisonSection

# §8.4 — always rendered alongside any comparison state
PERMANENT_NOTE = "Two signals pointing the same way does not make either one correct."

# §8.3 — exact documented sentences, keyed by (clf_direction, asm_direction)
_SENTENCES: dict[tuple[str, str], tuple[str, str]] = {
    # (clf_direction, asm_direction) → (state, sentence)
    ("MISMATCH_INDICATED", "CONTRADICTS_CLAIM"): (
        "CONSISTENT_PATTERN",
        "Both results point in the same direction: "
        "the comparison indicates a mismatch, and the retrieved evidence contradicts the claim.",
    ),
    ("MISMATCH_INDICATED", "SUPPORTS_CLAIM"): (
        "DIFFERENT_PATTERN",
        "The two results point in different directions. "
        "The comparison indicates a mismatch, and the retrieved evidence supports the claim. "
        "These measure different things — one is about the image–caption relationship, "
        "the other is about the claim.",
    ),
    ("NO_MISMATCH_INDICATED", "SUPPORTS_CLAIM"): (
        "CONSISTENT_PATTERN",
        "Both results point in the same direction: "
        "the comparison indicates no mismatch, and the retrieved evidence supports the claim.",
    ),
    ("NO_MISMATCH_INDICATED", "CONTRADICTS_CLAIM"): (
        "DIFFERENT_PATTERN",
        "The two results point in different directions. "
        "The comparison indicates no mismatch, and the retrieved evidence contradicts the claim. "
        "These measure different things — one is about the image–caption relationship, "
        "the other is about the claim.",
    ),
}

_SENTENCE_UNRESOLVED = (
    "ASSESSMENT_UNRESOLVED",
    "The retrieved evidence did not settle the claim, so the two results cannot be compared.",
)
_SENTENCE_ASM_UNAVAILABLE = (
    "ASSESSMENT_UNAVAILABLE",
    "No claim assessment was produced, so the two results cannot be compared.",
)
_SENTENCE_CLF_UNAVAILABLE = (
    "CLASSIFIER_UNAVAILABLE",
    "The image–caption comparison was unavailable. The claim assessment below is unaffected.",
)


def _clf_direction(clf: ClassifierSection) -> str:
    """Map classifier section to a direction string, or 'UNAVAILABLE'."""
    if clf.status != "OK":
        return "UNAVAILABLE"
    if clf.label == "MISMATCHED":
        return "MISMATCH_INDICATED"
    if clf.label == "MATCHED":
        return "NO_MISMATCH_INDICATED"
    # status is OK but label is absent — treat as UNAVAILABLE defensively
    return "UNAVAILABLE"


def _asm_direction(asm: AssessmentSection) -> str:
    """Map assessment section to a direction string."""
    if asm.status != "OK":
        return "UNAVAILABLE"
    if asm.outcome in ("INCONCLUSIVE", "CONFLICTING") or asm.outcome is None:
        return "UNRESOLVED"
    if asm.outcome == "CONTRADICTED":
        return "CONTRADICTS_CLAIM"
    if asm.outcome == "SUPPORTED":
        return "SUPPORTS_CLAIM"
    # Defensive: unknown outcome treated as unresolved
    return "UNRESOLVED"


def compute_comparison(
    clf: ClassifierSection,
    asm: AssessmentSection,
) -> ComparisonSection:
    """
    Compute the comparison state from the §8.2 exhaustive mapping.
    Always returns a well-defined ComparisonSection — never raises.
    """
    cd = _clf_direction(clf)

    # Rule: CLASSIFIER_UNAVAILABLE wins regardless of assessment state
    if cd == "UNAVAILABLE":
        state, sentence = _SENTENCE_CLF_UNAVAILABLE
        return ComparisonSection(state=state, sentence=sentence)

    ad = _asm_direction(asm)

    if ad == "UNAVAILABLE":
        state, sentence = _SENTENCE_ASM_UNAVAILABLE
        return ComparisonSection(state=state, sentence=sentence)

    if ad == "UNRESOLVED":
        state, sentence = _SENTENCE_UNRESOLVED
        return ComparisonSection(state=state, sentence=sentence)

    state, sentence = _SENTENCES[(cd, ad)]
    return ComparisonSection(state=state, sentence=sentence)
