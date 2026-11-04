"""Availability rate definitions per Document 07 §9.

Six rates with explicit denominators, never collapsed:
  - report_render_success_rate
  - input_rejection_success_rate
  - assessment_availability_rate
  - decision_availability_rate
  - conditional_decision_availability_rate
  - degraded_path_rate

Key distinctions:
  - Rejection is success for invalid inputs, not pipeline failure.
  - CONFLICTING is decided but does NOT resolve the claim.
  - Decision availability combines evidence coverage and component reliability.
"""
from __future__ import annotations

from typing import Any


def compute_availability_rates(
    *,
    valid_requests_count: int,
    renderable_reports_count: int,
    invalid_requests_count: int,
    correct_rejections_count: int,
    assessment_expected_count: int,
    produced_assessments_count: int,
    decided_assessments_count: int,
    degraded_requests_count: int,
) -> dict[str, float]:
    """Compute the six availability rates with explicit denominators."""
    return {
        "report_render_success_rate": (
            renderable_reports_count / valid_requests_count
            if valid_requests_count else 0.0
        ),
        "input_rejection_success_rate": (
            correct_rejections_count / invalid_requests_count
            if invalid_requests_count else 0.0
        ),
        "assessment_availability_rate": (
            produced_assessments_count / assessment_expected_count
            if assessment_expected_count else 0.0
        ),
        "decision_availability_rate": (
            decided_assessments_count / valid_requests_count
            if valid_requests_count else 0.0
        ),
        "conditional_decision_availability_rate": (
            decided_assessments_count / produced_assessments_count
            if produced_assessments_count else 0.0
        ),
        "degraded_path_rate": (
            degraded_requests_count / valid_requests_count
            if valid_requests_count else 0.0
        ),
    }


def classify_evaluation_outcome(
    *,
    is_valid_input: bool,
    rejection_view_rendered: bool = False,
) -> dict[str, Any]:
    """Classify an evaluation outcome.

    For invalid inputs, a rendered rejection view is a validation success,
    not a pipeline failure.
    """
    if not is_valid_input:
        if rejection_view_rendered:
            return {
                "category": "validation_success",
                "is_pipeline_failure": False,
            }
        return {
            "category": "validation_failure",
            "is_pipeline_failure": True,
        }
    return {
        "category": "pipeline_execution",
        "is_pipeline_failure": False,
    }


def format_rate_report(
    *,
    decision_availability: float,
    conditional_decision_availability: float,
) -> str:
    """Format a human-readable rate report.

    Decision availability reflects evidence coverage and component reliability
    together. Conditional decision availability isolates the reliability factor.
    """
    return (
        f"Decision availability rate: {decision_availability:.1%} — "
        "reflects evidence coverage and component reliability together. "
        f"Conditional decision availability: {conditional_decision_availability:.1%} — "
        "among produced assessments only."
    )


# Outcomes that are "decided" (the system produced a determination)
_DECIDED_OUTCOMES = {"SUPPORTED", "CONTRADICTED", "CONFLICTING"}

# Outcomes that "resolve" the truth of the claim (CONFLICTING does NOT resolve)
_RESOLVED_OUTCOMES = {"SUPPORTED", "CONTRADICTED"}


def is_decided_outcome(outcome: str) -> bool:
    """Return True if the outcome represents a system decision."""
    return outcome in _DECIDED_OUTCOMES


def is_resolved_claim(outcome: str) -> bool:
    """Return True if the outcome resolves the truth of the claim.

    CONFLICTING is decided but does NOT resolve — the evidence points
    in contradictory directions.
    """
    return outcome in _RESOLVED_OUTCOMES
