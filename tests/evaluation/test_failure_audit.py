"""Fast conformance cases for the Study 1 reporter."""

from scripts.report_failure_audit import (  # noqa: E402
    PLACEHOLDER,
    accuracy_of,
    bootstrap_ci,
    group_accuracy,
    paired_delta,
    report,
)
from tests.ids import case  # noqa: E402


def _row(case_id, variant, correct, gold="MISMATCHED", mtype="wrong_time", strata=None):
    return {
        "id": case_id,
        "variant": variant,
        "correct": correct,
        "gold_label": gold,
        "predicted_label": gold if correct else ("MATCHED" if gold == "MISMATCHED" else "MISMATCHED"),
        "mismatch_type": mtype,
        "strata": strata or {"source": "wire"},
        "calibrated_probability": 0.6 if correct else 0.4,
    }


@case(
    "audit.paired_delta",
    "F10",
    "fast",
    "doc/07 §12.2",
    "Paired deltas are computed only on case ids present in both variants.",
)
def test_paired_delta():
    rows = [
        _row("c1", "V0", True),
        _row("c1", "V1", False),
        _row("c2", "V0", False),
        _row("c2", "V1", True),
        _row("c3", "V1", True),  # no V0 - must be ignored
    ]
    result = paired_delta(rows, "V1", draws=20)
    assert result["n"] == 2
    assert result["delta"] == 0.0  # one -1 and one +1 cancel


@case(
    "audit.bootstrap_ci",
    "F10",
    "fast",
    "doc/07 §12.2",
    "A bootstrap interval contains the point estimate and never inverts.",
)
def test_bootstrap_ci():
    ci = bootstrap_ci([0.0, 1.0, 0.0, 1.0], draws=50, seed=1)
    assert ci[0] <= ci[1]
    assert 0.0 <= ci[0] <= ci[1] <= 1.0
    assert bootstrap_ci([], draws=10) == [PLACEHOLDER, PLACEHOLDER]


@case(
    "audit.group_accuracy",
    "F10",
    "fast",
    "doc/07 §12.2",
    "Per-slice accuracy groups only V0 rows and reports each named group.",
)
def test_group_accuracy():
    rows = [
        _row("c1", "V0", True, strata={"source": "wire"}),
        _row("c2", "V0", False, strata={"source": "social"}),
        _row("c2", "V1", True, strata={"source": "social"}),
    ]
    groups = group_accuracy(rows, "source")
    assert groups == {"social": 0.0, "wire": 1.0}


@case(
    "audit.placeholder_on_missing",
    "F10",
    "fast",
    "doc/07 §12.2",
    "A report with no measurements emits the exact placeholder, never a guessed number.",
)
def test_placeholder_on_missing():
    result = report([])
    assert result["overall_accuracy_V0"] == PLACEHOLDER
    assert result["inter_annotator_kappa"] == PLACEHOLDER
    assert all(v["delta"] == PLACEHOLDER for v in result["paired_delta"].values())


@case(
    "audit.accuracy_uses_gold_not_correct_flag",
    "F10",
    "fast",
    "doc/07 §12.2",
    "Reported accuracy is recomputed from labels, not trusted from a stored flag.",
)
def test_accuracy_uses_gold_not_correct_flag():
    rows = [
        {"gold_label": "MATCHED", "predicted_label": "MATCHED"},
        {"gold_label": "MISMATCHED", "predicted_label": "MATCHED"},
    ]
    assert accuracy_of(rows) == 0.5
