"""
src/reporting/model.py
======================
ReportModel and all section dataclasses.

Spec: 05_REPORTING_AND_UI.md §2

Status vocabulary (from docs 04 and 06, never extended here):
    OK | UNAVAILABLE | SKIPPED | FAILED

Invariants enforced:
  - InputSection.status is always "OK" (a report only exists for valid input)
  - ComparisonSection always has a well-defined state (see comparison.py)
  - Every value comes from a stored record; no placeholder strings
  - UNAVAILABLE/SKIPPED/FAILED sections carry no result values

Analysis fingerprint (§12.2):
    sha256(image_sha256 + caption_sha256 + model_version + run_id +
           encoder_revision + feature_contract_version +
           corpus_manifest_sha256 + prompt_version +
           retrieval_config_hash + embedding_revision + chunking_version)
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ── Primitive status type ─────────────────────────────────────────────────
# Exactly four values; this module never introduces a fifth.
SectionStatus = str   # literal "OK" | "UNAVAILABLE" | "SKIPPED" | "FAILED"


# ── Warning codes (§13) ───────────────────────────────────────────────────
IMAGE_DOWNSCALED          = "IMAGE_DOWNSCALED"
EXIF_ROTATED              = "EXIF_ROTATED"
ALPHA_FLATTENED           = "ALPHA_FLATTENED"
TEXT_NORMALIZED           = "TEXT_NORMALIZED"
CAPTION_TRUNCATED_FOR_CLIP = "CAPTION_TRUNCATED_FOR_CLIP"
LOW_CROP_RETENTION        = "LOW_CROP_RETENTION"
CLASSIFIER_UNAVAILABLE_W  = "CLASSIFIER_UNAVAILABLE"   # warning variant


# ── Section: Input (§4) ───────────────────────────────────────────────────
@dataclass(frozen=True)
class InputSection:
    """
    Status is always OK.
    A report is only constructed when document 04 accepted the request.
    """
    status: SectionStatus           # always "OK"
    original_filename: str
    original_width: int
    original_height: int
    caption_original: str           # verbatim; the normalized form is internal
    warnings: tuple[str, ...]       # WarningCode strings from §13


# ── Citation (used inside AssessmentSection) ──────────────────────────────
@dataclass(frozen=True)
class CitationRecord:
    passage_id: str
    quote: str                      # verbatim, escaped at render time
    source_name: str
    published_at: Optional[str]     # ISO date or None → "Publication date not stated…"
    url: Optional[str]              # validated http/https or None → "Source URL not available."
    accepted: bool
    rejection_reason: Optional[str] = None   # present iff accepted is False


# ── Claim component (used inside AssessmentSection) ───────────────────────
@dataclass(frozen=True)
class ClaimComponent:
    description: str
    status: str          # "supported" | "contradicted" | "contested" | "unresolved"
    citations: tuple[CitationRecord, ...]


# ── Passage record (used inside EvidenceSection) ──────────────────────────
@dataclass(frozen=True)
class PassageRecord:
    passage_id: str
    title: str
    source_name: str
    source_type: str
    published_at: Optional[str]    # None → "Publication date not stated by the source"
    excerpt: str
    url: Optional[str]             # None or invalid → "Source URL not available."
    rights_wording: Optional[str]  # shown only when present
    storage_scope: Optional[str]   # "excerpt" | "citation_metadata" | None


@dataclass(frozen=True)
class RetrievedPassage:
    passage: PassageRecord
    rank: int
    total_retrieved: int
    relevance_score: Optional[float]   # labelled as ranking value, never probability


# ── Section: Classifier (§5) ──────────────────────────────────────────────
@dataclass(frozen=True)
class ClassifierSection:
    status: SectionStatus
    run_id: str
    model_version: str
    encoder_revision: str
    # --- OK-only fields ---
    label: Optional[str] = None            # "MISMATCHED" | "MATCHED"
    score: Optional[float] = None          # uncalibrated
    calibrated_probability: Optional[float] = None
    calibration_status: str = "NOT_APPLIED"   # "APPLIED" | "NOT_APPLIED"
    score_definition: str = (
        "Cosine similarity between CLIP image and text embeddings, mapped to [0, 1]. "
        "Values at or above 0.5 are labelled MATCHED; below 0.5 as MISMATCHED."
    )
    # --- non-OK fields ---
    reason: Optional[str] = None
    remedy: Optional[str] = None
    error_detail: Optional[str] = None


# ── Section: Evidence (§6) ────────────────────────────────────────────────
@dataclass(frozen=True)
class EvidenceSection:
    status: SectionStatus
    passages: tuple[RetrievedPassage, ...]
    corpus_version: str
    corpus_manifest_sha256: str
    total_documents: int
    total_passages: int
    reason: Optional[str] = None      # non-OK sections


# ── Section: Assessment (§7) ──────────────────────────────────────────────
@dataclass(frozen=True)
class AssessmentSection:
    status: SectionStatus
    prompt_version: str
    scope_qualifier: str               # always visible (§9.1 level 2)
    # --- OK-only fields ---
    outcome: Optional[str] = None     # "SUPPORTED"|"CONTRADICTED"|"INCONCLUSIVE"|"CONFLICTING"
    reason_text: Optional[str] = None
    components: tuple[ClaimComponent, ...] = field(default_factory=tuple)
    accepted_citations: tuple[CitationRecord, ...] = field(default_factory=tuple)
    rejected_citations: tuple[CitationRecord, ...] = field(default_factory=tuple)
    unresolved_items: tuple[str, ...] = field(default_factory=tuple)
    conflicts: tuple[str, ...] = field(default_factory=tuple)
    uncertainties: tuple[str, ...] = field(default_factory=tuple)
    limitations: tuple[str, ...] = field(default_factory=tuple)
    # --- non-OK fields ---
    reason: Optional[str] = None
    error_detail: Optional[str] = None


# ── Section: Comparison (§8) ──────────────────────────────────────────────
@dataclass(frozen=True)
class ComparisonSection:
    """
    Always constructible. State is one of five documented values:
        CONSISTENT_PATTERN | DIFFERENT_PATTERN | ASSESSMENT_UNRESOLVED
        | ASSESSMENT_UNAVAILABLE | CLASSIFIER_UNAVAILABLE
    """
    state: str
    sentence: str


# ── Provenance ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ReportProvenance:
    model_version: str
    encoder_revision: str
    prompt_version: str
    corpus_version: str
    corpus_manifest_sha256: str
    retrieval_config_hash: str
    embedding_revision: str
    chunking_version: str
    feature_contract_version: str
    run_id: str
    created_at: datetime


# ── Root report model (§2.1) ──────────────────────────────────────────────
@dataclass(frozen=True)
class ReportModel:
    request_id: str
    analysis_fingerprint: str
    created_at: datetime
    input: InputSection
    classifier: ClassifierSection
    evidence: EvidenceSection
    assessment: AssessmentSection
    comparison: ComparisonSection
    limitations: tuple[str, ...]       # page-level (§9.1 level 3)
    provenance: ReportProvenance


# ── Fingerprint (§12.2) ───────────────────────────────────────────────────
def compute_fingerprint(
    *,
    image_sha256: str,
    caption_sha256: str,
    model_version: str,
    run_id: str,
    encoder_revision: str,
    feature_contract_version: str,
    corpus_manifest_sha256: str,
    prompt_version: str,
    retrieval_config_hash: str,
    embedding_revision: str,
    chunking_version: str,
) -> str:
    """
    Compute the analysis fingerprint per spec §12.2.
    Changing any one component invalidates a cached report.
    """
    blob = "".join([
        image_sha256,
        caption_sha256,
        model_version,
        run_id,
        encoder_revision,
        feature_contract_version,
        corpus_manifest_sha256,
        prompt_version,
        retrieval_config_hash,
        embedding_revision,
        chunking_version,
    ])
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
