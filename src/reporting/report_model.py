"""
src/reporting/report_model.py
==============================
Conformance surface for Document 05 §2.

Re-exports the dataclasses from model.py and adds the enum types,
validation helpers, rendering, export, cache, and sample factories that
the fast test suite asserts against.

Spec: 05_REPORTING_AND_UI.md §2, §10, §11, §12, §15, §16, §18
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

# ── Re-export core dataclasses from model.py ─────────────────────────────
from src.reporting.model import (  # noqa: F401
    AssessmentSection,
    CitationRecord,
    ClaimComponent,
    ComparisonSection,
    EvidenceSection,
    InputSection,
    PassageRecord,
    ReportModel,
    ReportProvenance,
    RetrievedPassage,
    compute_fingerprint,
)

# ── Re-export rendering and export ───────────────────────────────────────
from src.reporting.text_render import render_report_text  # noqa: F401
from src.reporting.export import generate_markdown_export  # noqa: F401


# ── Section Status Enum (§2.2) ──────────────────────────────────────────
class ReportSectionStatus(str, Enum):
    OK = "OK"
    UNAVAILABLE = "UNAVAILABLE"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"


STATUS_VOCABULARY = {s.value for s in ReportSectionStatus}


# ── Flexible ClassifierSection for conformance tests ────────────────────
# The test constructs ClassifierSection(status=..., reason=...) with only
# those two kwargs.  We provide a compatible dataclass here.
@dataclass(frozen=True)
class ClassifierSection:  # type: ignore[no-redef]
    status: str = "OK"
    run_id: str = ""
    model_version: str = ""
    encoder_revision: str = ""
    label: Optional[str] = None
    score: Optional[float] = None
    probability: Optional[float] = None
    calibrated_probability: Optional[float] = None
    calibration_status: str = "NOT_APPLIED"
    score_definition: str = (
        "Cosine similarity between CLIP image and text embeddings, mapped to [0, 1]. "
        "Values at or above 0.5 are labelled MATCHED; below 0.5 as MISMATCHED."
    )
    reason: Optional[str] = None
    remedy: Optional[str] = None
    error_detail: Optional[str] = None


# ── Validation helpers (§2.3) ───────────────────────────────────────────
def validate_section_payload(section: Any, status: str | None = None) -> None:
    """Assert that non-OK sections carry no result values."""
    effective_status = status or getattr(section, "status", "OK")
    if effective_status != "OK":
        for attr in ("label", "score", "probability", "calibrated_probability", "outcome"):
            val = getattr(section, attr, None)
            if val is not None:
                raise ValueError(
                    f"Non-OK section must not carry '{attr}'; found {val!r}"
                )


# ── Fingerprint alias (§12.2) ──────────────────────────────────────────
def compute_analysis_fingerprint(**kwargs: str) -> str:
    """Public alias matching the test's expected function name."""
    return compute_fingerprint(**kwargs)


# ── Input rejection view (§11) ─────────────────────────────────────────
def render_input_rejection(*, validation_error: str) -> str:
    """Return a plain-text rejection message. Does NOT produce a ReportModel."""
    return f"Input rejected: {validation_error}"


# ── Outcome badge (§16) ────────────────────────────────────────────────
_OUTCOME_BADGES: dict[str, str] = {
    "SUPPORTED": "✓ Supported",
    "CONTRADICTED": "✗ Contradicted",
    "INCONCLUSIVE": "? Inconclusive",
    "CONFLICTING": "⚠ Conflicting",
}


def format_outcome_badge(outcome: str) -> str:
    """Return word + icon for an assessment outcome."""
    return _OUTCOME_BADGES.get(outcome, f"? {outcome}")


# ── Markdown escaping (§16.1) ─────────────────────────────────────────
def escape_untrusted_markdown(text: str) -> str:
    """Escape markdown heading syntax, HTML tags, and link injection."""
    # Remove markdown heading markers at start
    escaped = text.lstrip("#").lstrip()
    # Strip HTML/script tags
    import re
    escaped = re.sub(r"<[^>]+>", "", escaped)
    # Escape markdown links
    escaped = escaped.replace("[", "\\[").replace("]", "\\]")
    return escaped


# ── Export wrapper (§15) ──────────────────────────────────────────────
def export_markdown(report: ReportModel) -> str:
    """Return the complete Markdown export for a report."""
    return generate_markdown_export(report)


# ── Session cache (§12.3) ────────────────────────────────────────────
class ReportCache:
    """In-memory report cache keyed by analysis fingerprint."""

    def __init__(self) -> None:
        self._store: dict[str, ReportModel] = {}

    def store(self, report: ReportModel) -> None:
        self._store[report.analysis_fingerprint] = report

    def lookup(self, fingerprint: str) -> Optional[ReportModel]:
        return self._store.get(fingerprint)

    def invalidate(self, fingerprint: str) -> None:
        self._store.pop(fingerprint, None)


# ── Sample report factories (for tests) ───────────────────────────────
def _make_minimal_provenance() -> ReportProvenance:
    return ReportProvenance(
        model_version="clip-vit-base-patch32-D-C",
        encoder_revision="e6a30b603a",
        prompt_version="1.0",
        corpus_version="synthetic-v1",
        corpus_manifest_sha256="a" * 64,
        retrieval_config_hash="b" * 64,
        embedding_revision="c1",
        chunking_version="1",
        feature_contract_version="1",
        run_id="run_sample_001",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _make_input_section() -> InputSection:
    return InputSection(
        status="OK",
        original_filename="test_image.jpg",
        original_width=640,
        original_height=480,
        caption_original="A dog running in the park",
        warnings=(),
    )


def _make_classifier_section_ok() -> ClassifierSection:
    return ClassifierSection(
        status="OK",
        run_id="clf_001",
        model_version="clip-vit-base-patch32-D-C",
        encoder_revision="e6a30b603a",
        label="MATCHED",
        score=0.72,
        probability=0.72,
        calibrated_probability=0.68,
    )


def _make_evidence_section(passages: tuple = ()) -> EvidenceSection:
    return EvidenceSection(
        status="OK" if passages else "UNAVAILABLE",
        passages=passages,
        corpus_version="synthetic-v1",
        corpus_manifest_sha256="a" * 64,
        total_documents=10,
        total_passages=50,
    )


def _make_assessment_section(
    outcome: str = "INCONCLUSIVE",
    accepted: tuple = (),
    rejected: tuple = (),
) -> AssessmentSection:
    from src.reporting import copy as C
    return AssessmentSection(
        status="OK",
        prompt_version="1.0",
        scope_qualifier=C.SCOPE_QUALIFIER,
        outcome=outcome,
        reason_text="Assessment based on retrieved evidence.",
        accepted_citations=accepted,
        rejected_citations=rejected,
    )


def _make_comparison_section(state: str = "ASSESSMENT_UNRESOLVED") -> ComparisonSection:
    return ComparisonSection(state=state, sentence="The signals are not yet aligned.")


def create_sample_report() -> ReportModel:
    """Return a fully-populated sample report for testing."""
    accepted = (
        CitationRecord(
            passage_id="p_sample",
            quote="Sample evidence quote.",
            source_name="Sample Source",
            published_at="2024-01-01",
            url="https://example.com",
            accepted=True,
        ),
    )
    passages = (
        RetrievedPassage(
            passage=PassageRecord(
                passage_id="p_sample",
                title="Sample Passage",
                source_name="Sample Source",
                source_type="article",
                published_at="2024-01-01",
                excerpt="Sample evidence quote.",
                url="https://example.com",
                rights_wording=None,
                storage_scope="excerpt",
            ),
            rank=1,
            total_retrieved=1,
            relevance_score=0.90,
        ),
    )
    fp = "f" * 64
    return ReportModel(
        request_id="req_sample_001",
        analysis_fingerprint=fp,
        created_at=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        input=_make_input_section(),
        classifier=_make_classifier_section_ok(),
        evidence=_make_evidence_section(passages=passages),
        assessment=_make_assessment_section(outcome="SUPPORTED", accepted=accepted),
        comparison=_make_comparison_section("CONSISTENT_PATTERN"),
        limitations=("The local corpus is small and curated.",),
        provenance=_make_minimal_provenance(),
    )


def create_degraded_classifier_report() -> ReportModel:
    """Return a report where the classifier section is UNAVAILABLE."""
    degraded_clf = ClassifierSection(
        status="UNAVAILABLE",
        reason="Classifier model offline.",
    )
    fp = "d" * 64
    return ReportModel(
        request_id="req_degraded_001",
        analysis_fingerprint=fp,
        created_at=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        input=_make_input_section(),
        classifier=degraded_clf,
        evidence=_make_evidence_section(),
        assessment=_make_assessment_section(outcome="INCONCLUSIVE"),
        comparison=_make_comparison_section("CLASSIFIER_UNAVAILABLE"),
        limitations=("The local corpus is small and curated.",),
        provenance=_make_minimal_provenance(),
    )


def create_sample_report_with_rejected_citations() -> ReportModel:
    """Return a report with both accepted and rejected citations."""
    accepted = (
        CitationRecord(
            passage_id="p1",
            quote="Dogs are known to run in parks.",
            source_name="Pet Journal",
            published_at="2024-03-15",
            url="https://example.com/pet",
            accepted=True,
        ),
    )
    rejected = (
        CitationRecord(
            passage_id="p2",
            quote="Cats prefer indoor activities.",
            source_name="Feline Weekly",
            published_at=None,
            url=None,
            accepted=False,
            rejection_reason="Not relevant to the claim about dogs.",
        ),
    )
    passages = (
        RetrievedPassage(
            passage=PassageRecord(
                passage_id="p1",
                title="Running Dogs",
                source_name="Pet Journal",
                source_type="article",
                published_at="2024-03-15",
                excerpt="Dogs are known to run in parks.",
                url="https://example.com/pet",
                rights_wording=None,
                storage_scope="excerpt",
            ),
            rank=1,
            total_retrieved=2,
            relevance_score=0.85,
        ),
    )
    fp = "e" * 64
    return ReportModel(
        request_id="req_cited_001",
        analysis_fingerprint=fp,
        created_at=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        input=_make_input_section(),
        classifier=_make_classifier_section_ok(),
        evidence=_make_evidence_section(passages=passages),
        assessment=_make_assessment_section(
            outcome="SUPPORTED", accepted=accepted, rejected=rejected,
        ),
        comparison=_make_comparison_section("CONSISTENT_PATTERN"),
        limitations=("The local corpus is small and curated.",),
        provenance=_make_minimal_provenance(),
    )
