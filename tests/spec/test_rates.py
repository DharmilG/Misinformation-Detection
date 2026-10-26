"""F10 - availability rate definitions and non-collapsing distinctions of Document 07 §9.

Tests the mathematical definitions of the six availability rates, their explicit
denominators, and the three architectural non-collapsing distinctions:
1. Rejection view is success for invalid requests, not pipeline failure;
2. Decision availability reflects both evidence coverage and component reliability;
3. CONFLICTING is a decided outcome, not a resolved claim.
"""

import pytest

from tests.ids import case, known_gap


def _rates_module():
    import importlib
    return importlib.import_module("src.evaluation.rates")


@known_gap(
    "rates.definitions_and_denominators",
    "F10",
    "fast",
    "doc/07 §9",
    "Six availability rates are defined with explicit denominators and never collapsed.",
    "availability rates evaluation module is not yet implemented",
)
def test_rates_definitions_and_denominators():
    mod = _rates_module()
    assert hasattr(mod, "compute_availability_rates")
    rates = mod.compute_availability_rates(
        valid_requests_count=100,
        renderable_reports_count=98,
        invalid_requests_count=20,
        correct_rejections_count=20,
        assessment_expected_count=95,
        produced_assessments_count=90,
        decided_assessments_count=70,
        degraded_requests_count=5,
    )
    # Check all 6 rates are present:
    expected_rates = {
        "report_render_success_rate",
        "input_rejection_success_rate",
        "assessment_availability_rate",
        "decision_availability_rate",
        "conditional_decision_availability_rate",
        "degraded_path_rate",
    }
    assert expected_rates.issubset(rates.keys())
    assert rates["report_render_success_rate"] == 98 / 100
    assert rates["input_rejection_success_rate"] == 20 / 20
    assert rates["assessment_availability_rate"] == 90 / 95
    assert rates["decision_availability_rate"] == 70 / 100
    assert rates["conditional_decision_availability_rate"] == 70 / 90


@known_gap(
    "rates.rejection_is_success_for_invalid",
    "F10",
    "fast",
    "doc/07 §9.1",
    "A rejection view is counted as a success for invalid requests, not a pipeline failure.",
    "rejection rate categorization is not yet implemented",
)
def test_rates_rejection_is_success_for_invalid():
    mod = _rates_module()
    assert hasattr(mod, "classify_evaluation_outcome")
    outcome = mod.classify_evaluation_outcome(
        is_valid_input=False,
        rejection_view_rendered=True,
    )
    assert outcome["category"] == "validation_success"
    assert outcome["is_pipeline_failure"] is False


@known_gap(
    "rates.decision_availability_dual_factor",
    "F10",
    "fast",
    "doc/07 §9.1",
    "Decision availability rate reflects evidence coverage and reliability together, reported alongside conditional decision availability.",
    "dual factor rate reporting is not yet implemented",
)
def test_rates_decision_availability_dual_factor():
    mod = _rates_module()
    assert hasattr(mod, "format_rate_report")
    report = mod.format_rate_report(
        decision_availability=0.70,
        conditional_decision_availability=0.85,
    )
    assert "evidence coverage and component reliability together" in report.lower()


@known_gap(
    "rates.conflicting_is_decided_not_resolved",
    "F10",
    "fast",
    "doc/07 §9.1",
    "CONFLICTING is a decided outcome in availability rates, but recorded as not resolving the truth of the claim.",
    "conflicting outcome distinction is not yet implemented",
)
def test_rates_conflicting_is_decided_not_resolved():
    mod = _rates_module()
    assert hasattr(mod, "is_decided_outcome")
    assert hasattr(mod, "is_resolved_claim")
    assert mod.is_decided_outcome("CONFLICTING") is True
    assert mod.is_resolved_claim("CONFLICTING") is False
