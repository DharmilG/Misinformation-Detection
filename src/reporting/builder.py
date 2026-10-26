"""
src/reporting/builder.py
========================
Assembles a ReportModel from upstream pipeline results.

Spec: 05_REPORTING_AND_UI §2 (model assembly), §12 (session state).

This is the only place where a ReportModel is constructed.
Business logic (comparison, fingerprint) stays here — not in app.py.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from src.clip_model.classifier import ClassifierResult
from src.input_pipeline.validation import InputRequest
from src.rag.assessment import Assessment
from src.reporting import copy as C
from src.reporting.comparison import compute_comparison
from src.reporting.model import (
    AssessmentSection,
    ClassifierSection,
    ComparisonSection,
    EvidenceSection,
    InputSection,
    ReportModel,
    ReportProvenance,
    compute_fingerprint,
)
from src.retrieval.corpus import RetrievalResult
from src.utils.config import (
    CHUNKING_VERSION,
    CORPUS_MANIFEST_SHA256,
    CORPUS_VERSION,
    EMBEDDING_REVISION,
    ENCODER_REVISION,
    FEATURE_CONTRACT_VERSION,
    MODEL_VERSION,
    PROMPT_VERSION,
    RETRIEVAL_CONFIG_HASH,
)


# ── Section builders ──────────────────────────────────────────────────────

def _build_input_section(
    req: InputRequest,
    warnings: list[str],
) -> InputSection:
    return InputSection(
        status="OK",
        original_filename=req.image.original_filename,
        original_width=req.image.original_width,
        original_height=req.image.original_height,
        caption_original=req.caption.original,
        warnings=tuple(warnings),
    )


def _build_classifier_section(result: ClassifierResult) -> ClassifierSection:
    return ClassifierSection(
        status=result.status,
        run_id=result.run_id,
        model_version=result.model_version,
        encoder_revision=result.encoder_revision,
        label=result.label,
        score=result.score,
        calibrated_probability=result.calibrated_probability,
        calibration_status=result.calibration_status,
        score_definition=result.score_definition,
        reason=result.reason,
        remedy=result.remedy,
        error_detail=result.error_detail,
    )


def _build_evidence_section(retrieval: RetrievalResult) -> EvidenceSection:
    return EvidenceSection(
        status=retrieval.status,
        passages=retrieval.passages,
        corpus_version=retrieval.corpus_version,
        corpus_manifest_sha256=retrieval.corpus_manifest_sha256,
        total_documents=retrieval.total_documents,
        total_passages=retrieval.total_passages,
        reason=retrieval.reason,
    )


def _build_assessment_section(assessment: Assessment) -> AssessmentSection:
    return AssessmentSection(
        status=assessment.status,
        prompt_version=assessment.prompt_version,
        scope_qualifier=assessment.scope_qualifier,
        outcome=assessment.outcome,
        reason_text=assessment.reason_text,
        components=assessment.components,
        accepted_citations=assessment.accepted_citations,
        rejected_citations=assessment.rejected_citations,
        unresolved_items=assessment.unresolved_items,
        conflicts=assessment.conflicts,
        uncertainties=assessment.uncertainties,
        limitations=assessment.limitations,
        reason=assessment.reason,
        error_detail=assessment.error_detail,
    )


# ── Page-level system limitations ─────────────────────────────────────────

_PAGE_LIMITATIONS: tuple[str, ...] = C.METHODOLOGY_LIMITATIONS


# ── Public API ─────────────────────────────────────────────────────────────

def build_report(
    *,
    request: InputRequest,
    warnings: list[str],
    clf_result: ClassifierResult,
    retrieval: RetrievalResult,
    assessment: Assessment,
    request_id: Optional[str] = None,
) -> ReportModel:
    """
    Assemble a complete, immutable ReportModel.

    Parameters
    ----------
    request     : validated InputRequest from the input pipeline
    warnings    : warning codes from the input pipeline
    clf_result  : ClassifierResult from the CLIP classifier
    retrieval   : RetrievalResult from the TF-IDF retriever
    assessment  : Assessment from the RAG assessment module
    request_id  : optional stable id for this analysis run (new UUID if absent)

    Returns
    -------
    ReportModel (frozen dataclass; all mutations rejected)
    """
    now = datetime.now(tz=timezone.utc)
    rid = request_id or str(uuid.uuid4())

    inp  = _build_input_section(request, warnings)
    clf  = _build_classifier_section(clf_result)
    ev   = _build_evidence_section(retrieval)
    asm  = _build_assessment_section(assessment)
    comp = compute_comparison(clf, asm)

    fingerprint = compute_fingerprint(
        image_sha256=request.image.image_sha256,
        caption_sha256=request.caption.caption_sha256,
        model_version=clf_result.model_version,
        run_id=clf_result.run_id,
        encoder_revision=clf_result.encoder_revision,
        feature_contract_version=FEATURE_CONTRACT_VERSION,
        corpus_manifest_sha256=retrieval.corpus_manifest_sha256,
        prompt_version=assessment.prompt_version,
        retrieval_config_hash=retrieval.retrieval_config_hash,
        embedding_revision=retrieval.embedding_revision,
        chunking_version=retrieval.chunking_version,
    )

    provenance = ReportProvenance(
        model_version=clf_result.model_version,
        encoder_revision=clf_result.encoder_revision,
        prompt_version=assessment.prompt_version,
        corpus_version=retrieval.corpus_version,
        corpus_manifest_sha256=retrieval.corpus_manifest_sha256,
        retrieval_config_hash=retrieval.retrieval_config_hash,
        embedding_revision=retrieval.embedding_revision,
        chunking_version=retrieval.chunking_version,
        feature_contract_version=FEATURE_CONTRACT_VERSION,
        run_id=clf_result.run_id,
        created_at=now,
    )

    return ReportModel(
        request_id=rid,
        analysis_fingerprint=fingerprint,
        created_at=now,
        input=inp,
        classifier=clf,
        evidence=ev,
        assessment=asm,
        comparison=comp,
        limitations=_PAGE_LIMITATIONS,
        provenance=provenance,
    )
