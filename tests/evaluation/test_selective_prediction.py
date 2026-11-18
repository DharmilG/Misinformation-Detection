"""Fast conformance cases for the Study 2 selective-prediction metrics."""

from src.evaluation.selective_prediction import (  # noqa: E402
    all_signals,
    aurc,
    confidence_for,
    overconfidence_rate,
    risk_coverage,
    selective_accuracy,
    stratified,
)
from scripts.run_selective_prediction import extract_v0  # noqa: E402
from tests.ids import case  # noqa: E402


def _row(cid, correct, score, mtype="wrong_time"):
    return {
        "id": cid,
        "variant": "V0",
        "correct": correct,
        "calibrated_probability": score,
        "d_c_score": score,
        "cosine": score,
        "gold_label": "MISMATCHED",
        "predicted_label": "MISMATCHED" if correct else "MATCHED",
        "mismatch_type": mtype,
    }


@case(
    "selective.risk_coverage_monotone",
    "F10",
    "fast",
    "doc/07 §12.2",
    "Coverage increases monotonically and risk never exceeds one.",
)
def test_risk_coverage_monotone():
    rows = [_row("a", True, 0.9), _row("b", False, 0.2), _row("c", True, 0.7)]
    curve = risk_coverage(rows, "calibrated_probability")
    coverages = [p["coverage"] for p in curve]
    assert all(a <= b for a, b in zip(coverages, coverages[1:]))
    assert coverages[-1] == 1.0
    assert all(0.0 <= p["risk"] <= 1.0 for p in curve)


@case(
    "selective.aurc_bounds",
    "F10",
    "fast",
    "doc/07 §12.2",
    "A perfect ranking has a lower AURC than a reversed one.",
)
def test_aurc_bounds():
    good = [_row("a", True, 0.9), _row("b", True, 0.8), _row("c", False, 0.1)]
    bad = [_row("a", False, 0.9), _row("b", False, 0.8), _row("c", True, 0.1)]
    assert aurc(risk_coverage(good, "calibrated_probability")) < aurc(
        risk_coverage(bad, "calibrated_probability")
    )


@case(
    "selective.selective_accuracy_at_coverage",
    "F10",
    "fast",
    "doc/07 §12.2",
    "Selective accuracy at 70% coverage uses the largest eligible point.",
)
def test_selective_accuracy_at_coverage():
    curve = risk_coverage([_row("a", True, 0.9), _row("b", False, 0.1)], "d_c_score")
    assert selective_accuracy(curve, 0.7) == 1.0
    assert selective_accuracy([], 0.7) == "[PLANNED - NOT YET MEASURED]"


@case(
    "selective.overconfidence_rate",
    "F10",
    "fast",
    "doc/07 §12.2",
    "Only errors above the confidence threshold count as overconfident.",
)
def test_overconfidence_rate():
    rows = [
        _row("a", False, 0.95),
        _row("b", False, 0.4),  # near the boundary - not overconfident
        _row("c", True, 0.99),
    ]
    assert overconfidence_rate(rows, "calibrated_probability", threshold=0.8) == 0.5
    assert overconfidence_rate(
        [_row("ok", True, 0.9)], "calibrated_probability", threshold=0.8
    ) == "[PLANNED - NOT YET MEASURED]"


@case(
    "selective.stratified_by_failure_mode",
    "F10",
    "fast",
    "doc/07 §12.2",
    "AUC results can be broken down by the Study 1 failure taxonomy.",
)
def test_stratified_by_failure_mode():
    rows = [
        _row("a", True, 0.9, "wrong_time"),
        _row("b", False, 0.1, "wrong_place"),
    ]
    groups = stratified(rows, "mismatch_type")
    assert set(groups) == {"wrong_time", "wrong_place"}


@case(
    "selective.v0_extraction",
    "F10",
    "fast",
    "doc/07 §12.2",
    "Only the unperturbed V0 row per case enters the signal analysis.",
)
def test_v0_extraction():
    rows = [
        _row("a", True, 0.9),
        {**_row("a", False, 0.1), "variant": "V4"},
        _row("b", False, 0.2),
    ]
    assert [r["id"] for r in extract_v0(rows)] == ["a", "b"]


@case(
    "selective.placeholder_on_missing",
    "F10",
    "fast",
    "doc/07 §12.2",
    "An empty signal set emits the exact placeholder rather than a guessed AURC.",
)
def test_placeholder_on_missing():
    assert all_signals([])["cosine"]["aurc"] == "[PLANNED - NOT YET MEASURED]"
    assert confidence_for("d_c_score", 0.5) == 0.0
