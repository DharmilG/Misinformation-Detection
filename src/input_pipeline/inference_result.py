"""Inference result schema and assessment context handoff of Document 04 §15, §19, §20.

Defines ClassifierResult invariants, CalibrationStatus, AssessmentContext, and safe
prompt assembly with quarantined untrusted passages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import uuid


class CalibrationStatus(str, Enum):
    APPLIED = "APPLIED"
    ABSENT = "ABSENT"


@dataclass(frozen=True)
class ClassifierResult:
    status: str
    request_id: str
    label: str | None
    score: float | None
    logit_positive: float | None
    probability: float | None
    calibration_status: str | None
    threshold: float | None
    class_order: tuple[str, ...]
    positive_class: str
    feature_family: str
    model_version: str
    run_id: str
    encoder_revision: str
    warnings: tuple[str, ...]
    error: str | None
    elapsed_ms: float


@dataclass(frozen=True)
class AssessmentContext:
    request_id: str
    caption: str
    caption_raw: str
    image_validated: bool
    image_features_available: bool
    classifier: ClassifierResult | None
    interpretation: str
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class SafePrompt:
    system_prompt: str
    user_prompt: str


def create_ok_classifier_result(
    score: float,
    logit_positive: float,
    probability: float | None,
    threshold: float = 0.50,
    calibrated: bool = True,
    request_id: str | None = None,
    feature_family: str = "Family_D_C",
    model_version: str = "1.0.0",
    run_id: str = "run_default",
    encoder_revision: str = "openai/clip-vit-base-patch32@main",
    warnings: tuple[str, ...] = (),
    elapsed_ms: float = 12.5,
) -> ClassifierResult:
    """Create a validated ClassifierResult with status == OK."""
    label = "MISMATCHED" if score >= threshold else "MATCHED"
    prob = probability if calibrated else None
    cal_status = CalibrationStatus.APPLIED if (calibrated and probability is not None) else CalibrationStatus.ABSENT

    return ClassifierResult(
        status="OK",
        request_id=request_id or str(uuid.uuid4()),
        label=label,
        score=float(score),
        logit_positive=float(logit_positive),
        probability=float(prob) if prob is not None else None,
        calibration_status=cal_status,
        threshold=float(threshold),
        class_order=("MATCHED", "MISMATCHED"),
        positive_class="MISMATCHED",
        feature_family=feature_family,
        model_version=model_version,
        run_id=run_id,
        encoder_revision=encoder_revision,
        warnings=tuple(warnings),
        error=None,
        elapsed_ms=float(elapsed_ms),
    )


def create_assessment_context(
    image_validated: bool,
    image_features_available: bool,
    request_id: str | None = None,
    caption: str = "Sample caption",
    caption_raw: str = "Sample caption",
    classifier: ClassifierResult | None = None,
    interpretation: str = "Assessment context",
    warnings: tuple[str, ...] = (),
) -> AssessmentContext:
    """Create an AssessmentContext with separate image_validated and image_features_available flags."""
    return AssessmentContext(
        request_id=request_id or str(uuid.uuid4()),
        caption=caption,
        caption_raw=caption_raw,
        image_validated=bool(image_validated),
        image_features_available=bool(image_features_available),
        classifier=classifier,
        interpretation=interpretation,
        warnings=tuple(warnings),
    )


def build_safe_generation_prompt(
    system_instructions: str,
    passages: list[str],
    user_query: str = "Evaluate the claim.",
) -> SafePrompt:
    """Quarantine untrusted external retrieved strings into data delimiters."""
    # Ensure untrusted text is never inserted into the system instructions
    clean_sys = system_instructions
    evidence_block = "\n--- BEGIN EVIDENCE ---\n"
    for i, p in enumerate(passages, 1):
        evidence_block += f"[{i}] {p}\n"
    evidence_block += "--- END EVIDENCE ---\n"

    user_prompt = f"{user_query}\n{evidence_block}"
    return SafePrompt(system_prompt=clean_sys, user_prompt=user_prompt)
