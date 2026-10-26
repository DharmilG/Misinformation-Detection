"""The Assessment result object and per-status invariants of Document 06 §31 and §34.

Tests Assessment schema invariants (status != OK implies outcome is None, prose fields None),
absence of placeholder text (no 'N/A', '-', or empty strings), and generation provenance.
"""

import pytest

from tests.ids import case, known_gap


def _generation_module():
    import importlib
    return importlib.import_module("rag_generation")


@known_gap(
    "rag_generation.module_exists",
    "F2",
    "fast",
    "doc/06 §31",
    "The project provides an assessment generation and result object module.",
    "there is no rag_generation module",
)
def test_generation_module_exists():
    _generation_module()


@known_gap(
    "rag_generation.assessment_schema_fields",
    "F2",
    "fast",
    "doc/06 §31.1",
    "Assessment dataclass defines status, outcome, outcome_source, citations, provenance, and warnings.",
    "Assessment schema is not implemented",
)
def test_assessment_schema_fields():
    mod = _generation_module()
    cls = getattr(mod, "Assessment", None)
    assert cls is not None
    required = {
        "status",
        "outcome",
        "outcome_source",
        "outcome_reason",
        "scope_text",
        "claim_restated",
        "claim_components",
        "evidence_summary",
        "citations",
        "rejected_citations",
        "unresolved",
        "conflicts",
        "uncertainties",
        "limitations",
        "provenance",
        "warnings",
        "error",
    }
    annotations = getattr(cls, "__annotations__", {})
    missing = required - set(annotations.keys())
    assert not missing, "Assessment missing fields: %s" % missing


@known_gap(
    "rag_generation.per_status_invariants",
    "F2",
    "fast",
    "doc/06 §31.2",
    "When status != OK, outcome is None, outcome_source is 'none', and all prose payload fields are None.",
    "per-status invariants are not enforced",
)
def test_per_status_invariants():
    mod = _generation_module()
    failed_assessment = mod.create_failed_assessment(status="UNAVAILABLE", error_message="Model timed out")
    assert failed_assessment.status == "UNAVAILABLE"
    assert failed_assessment.outcome is None
    assert failed_assessment.outcome_source == "none"
    assert failed_assessment.evidence_summary is None
    assert failed_assessment.claim_restated is None
    assert failed_assessment.error is not None


@known_gap(
    "rag_generation.no_placeholder_text_in_failed",
    "F2",
    "fast",
    "doc/06 §31.2",
    "Non-OK and no-evidence statuses never contain placeholder strings such as 'N/A', 'unknown', or empty strings.",
    "no-placeholder invariant is not enforced",
)
def test_no_placeholder_text_in_failed():
    mod = _generation_module()
    no_ev_assessment = mod.create_no_evidence_assessment()
    for field in ("evidence_summary", "claim_restated", "scope_text"):
        val = getattr(no_ev_assessment, field, None)
        assert val is None, "Placeholder text '%s' found in field %s" % (val, field)
