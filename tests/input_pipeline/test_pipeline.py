"""The inference pipeline runtime flow, status model, and degradation policy of Document 04 §11, §14, §17.

Tests component status vocabulary (OK, UNAVAILABLE, SKIPPED, FAILED), the no-fabrication rule,
degradation independence between classifier and retrieval stages, and threshold decision invariance.
"""

import pytest

from tests.ids import case


def _pipeline_module():
    import importlib
    return importlib.import_module("inference_pipeline")


@case(
    "inference_pipeline.module_exists",
    "F2",
    "fast",
    "doc/04 §13",
    "The project provides an inference pipeline module managing stage execution and degradation.",
)
def test_pipeline_module_exists():
    _pipeline_module()


@case(
    "inference_pipeline.status_vocabulary",
    "F2",
    "fast",
    "doc/04 §11.1",
    "ComponentStatus enum defines exactly OK, UNAVAILABLE, SKIPPED, and FAILED.",
)
def test_status_vocabulary():
    mod = _pipeline_module()
    status_cls = getattr(mod, "ComponentStatus", None)
    assert status_cls is not None
    names = {s.name for s in status_cls}
    assert names == {"OK", "UNAVAILABLE", "SKIPPED", "FAILED"}


@case(
    "inference_pipeline.no_fabrication_rule",
    "F2",
    "fast",
    "doc/04 §11.2",
    "A non-OK component status guarantees that label, score, and probability are None (never 0.0, 0.5, or default).",
)
def test_no_fabrication_rule():
    mod = _pipeline_module()
    for status in (mod.ComponentStatus.UNAVAILABLE, mod.ComponentStatus.SKIPPED, mod.ComponentStatus.FAILED):
        res = mod.create_failed_classifier_result(status=status, error_message="Stage failed")
        assert res.label is None, "label was fabricated under status %s" % status
        assert res.score is None, "score was fabricated under status %s" % status
        assert res.probability is None, "probability was fabricated under status %s" % status


@case(
    "inference_pipeline.two_runtime_artifacts",
    "F2",
    "fast",
    "doc/04 §11.3",
    "Validation rejection produces InputRejected (no report), while valid requests always produce renderable AnalysisReport.",
)
def test_two_runtime_artifacts():
    mod = _pipeline_module()
    assert hasattr(mod, "InputRejected")
    assert hasattr(mod, "AnalysisReport")


@case(
    "inference_pipeline.degradation_classifier_failure_retrieval_runs",
    "F2",
    "fast",
    "doc/04 §17.2",
    "A failure in the classifier head does not stop the evidence retrieval stage.",
)
def test_degradation_classifier_failure_retrieval_runs():
    mod = _pipeline_module()
    # Mock request with broken classifier checkpoint
    report = mod.run_pipeline_with_mocks(classifier_fails=True, retrieval_succeeds=True)
    assert report.classifier.status in (mod.ComponentStatus.UNAVAILABLE, mod.ComponentStatus.SKIPPED)
    assert report.retrieval.status == mod.ComponentStatus.OK
    assert report.is_renderable() is True


@case(
    "inference_pipeline.decision_rule_uncalibrated_threshold",
    "F2",
    "fast",
    "doc/04 §14.4",
    "The decision rule thresholds the uncalibrated score, making the predicted label invariant to calibration.",
)
def test_decision_rule_uncalibrated_threshold():
    mod = _pipeline_module()
    score = 0.65
    threshold = 0.50
    label_uncalibrated = mod.apply_decision_rule(score=score, threshold=threshold)
    label_calibrated = mod.apply_decision_rule(score=score, threshold=threshold, probability=0.85)
    assert label_uncalibrated == "MISMATCHED"
    assert label_calibrated == "MISMATCHED"
