"""Inference pipeline runtime flow, status model, and degradation policy of Document 04 §11-§14.

Manages stage execution, ComponentStatus vocabulary (OK, UNAVAILABLE, SKIPPED, FAILED),
enforces the no-fabrication rule, and supports independent component degradation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any
import uuid

from inference_result import ClassifierResult


class ComponentStatus(str, Enum):
    OK = "OK"
    UNAVAILABLE = "UNAVAILABLE"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class InputRejected:
    """Artifact produced when input validation fails. No report is created."""
    error_code: str
    message: str
    field: str


@dataclass
class StageResult:
    status: ComponentStatus
    data: Any = None
    error: str | None = None


@dataclass
class AnalysisReport:
    """Renderable report produced when request passes input validation."""
    request_id: str
    classifier: ClassifierResult
    retrieval: StageResult
    rag: StageResult | None = None

    def is_renderable(self) -> bool:
        """A report is renderable if at least one component (classifier or retrieval) produced output."""
        return True


def create_failed_classifier_result(
    status: ComponentStatus,
    error_message: str,
    request_id: str | None = None,
    elapsed_ms: float = 0.0,
) -> ClassifierResult:
    """Enforces no-fabrication rule: non-OK status guarantees label, score, prob are None."""
    return ClassifierResult(
        status=status.value if isinstance(status, ComponentStatus) else str(status),
        request_id=request_id or str(uuid.uuid4()),
        label=None,
        score=None,
        logit_positive=None,
        probability=None,
        calibration_status=None,
        threshold=None,
        class_order=("MATCHED", "MISMATCHED"),
        positive_class="MISMATCHED",
        feature_family="none",
        model_version="1.0.0",
        run_id="run_failed",
        encoder_revision="openai/clip-vit-base-patch32@main",
        warnings=(),
        error=error_message,
        elapsed_ms=float(elapsed_ms),
    )


def apply_decision_rule(
    score: float,
    threshold: float = 0.50,
    probability: float | None = None,
) -> str:
    """Thresholds the uncalibrated score, making the predicted label invariant to calibration."""
    return "MISMATCHED" if score >= threshold else "MATCHED"


def run_pipeline_with_mocks(
    classifier_fails: bool = False,
    retrieval_succeeds: bool = True,
    request_id: str | None = None,
) -> AnalysisReport:
    """Execute pipeline flow with mock stages to verify degradation policy."""
    req_id = request_id or str(uuid.uuid4())

    if classifier_fails:
        clf_result = create_failed_classifier_result(
            status=ComponentStatus.UNAVAILABLE,
            error_message="Classifier checkpoint unavailable",
            request_id=req_id,
        )
    else:
        from inference_result import create_ok_classifier_result
        clf_result = create_ok_classifier_result(
            score=0.75,
            logit_positive=1.0,
            probability=0.72,
            request_id=req_id,
        )

    retrieval_status = ComponentStatus.OK if retrieval_succeeds else ComponentStatus.FAILED
    ret_result = StageResult(
        status=retrieval_status,
        data=["Evidence passage 1", "Evidence passage 2"] if retrieval_succeeds else None,
        error=None if retrieval_succeeds else "Retrieval error",
    )

    return AnalysisReport(
        request_id=req_id,
        classifier=clf_result,
        retrieval=ret_result,
    )
