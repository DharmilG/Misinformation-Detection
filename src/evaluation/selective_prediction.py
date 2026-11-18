"""Selective-prediction metrics for Study 2.

Given a set of scored cases, these functions ask whether confidence separates
correct from incorrect decisions.  All functions take plain lists of dicts so
they can be tested without model weights.

Signals:
    ``d_c_score``, ``calibrated_probability`` - distance from 0.5 is confidence.
    ``cosine`` - magnitude is confidence (direction is handled by the decision).
    ``disagreement`` - low disagreement is high confidence.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable


def confidence_for(signal: str, value: float) -> float:
    """Map a raw signal to a confidence in [0, 1]."""
    v = float(value)
    if signal in ("d_c_score", "calibrated_probability", "cosine_probability"):
        return min(1.0, max(0.0, abs(v - 0.5) * 2.0))
    if signal == "cosine":
        return min(1.0, abs(v))
    if signal == "disagreement":
        return min(1.0, max(0.0, 1.0 - v))
    raise ValueError(f"unknown signal: {signal}")


def _correct(row: dict[str, Any]) -> float:
    if "correct" in row and isinstance(row["correct"], bool):
        return 1.0 if row["correct"] else 0.0
    gold = row.get("gold_label")
    predicted = row.get("predicted_label")
    return 1.0 if gold is not None and gold == predicted else 0.0


def risk_coverage(
    rows: list[dict[str, Any]], signal: str
) -> list[dict[str, float]]:
    """Return (coverage, risk, accuracy) points, most-confident first."""
    if not rows:
        return []
    scored = sorted(
        rows,
        key=lambda r: confidence_for(signal, float(r.get(signal, 0.0))),
        reverse=True,
    )
    points = []
    for index, row in enumerate(scored, 1):
        accepted = scored[:index]
        correct = sum(_correct(r) for r in accepted)
        points.append(
            {
                "coverage": round(index / len(scored), 6),
                "risk": round(1.0 - correct / index, 6),
                "accuracy": round(correct / index, 6),
            }
        )
    return points


def aurc(curve: list[dict[str, float]]) -> float:
    """Area under the risk-coverage curve.  Lower is better."""
    if not curve:
        raise ValueError("empty risk-coverage curve")
    return round(sum(point["risk"] for point in curve) / len(curve), 6)


def selective_accuracy(
    curve: list[dict[str, float]], coverage: float
) -> float | str:
    """Accuracy at the largest recorded coverage not exceeding ``coverage``."""
    eligible = [p for p in curve if p["coverage"] <= coverage + 1e-9]
    if not eligible:
        return "[PLANNED - NOT YET MEASURED]"
    return max(eligible, key=lambda p: p["coverage"])["accuracy"]


def overconfidence_rate(
    rows: list[dict[str, Any]], signal: str, *, threshold: float
) -> float | str:
    """Fraction of errors whose confidence clears the acceptance threshold."""
    errors = [r for r in rows if _correct(r) == 0.0]
    if not errors:
        return "[PLANNED - NOT YET MEASURED]"
    confident = [
        r for r in errors if confidence_for(signal, float(r.get(signal, 0.0))) >= threshold
    ]
    return round(len(confident) / len(errors), 6)


def stratified(
    rows: list[dict[str, Any]], key: str
) -> dict[str, list[dict[str, Any]]]:
    """Group rows by a flat field or by a named ``strata`` entry."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        value = row.get(key)
        if value is None:
            value = (row.get("strata") or {}).get(key)
        if value is None:
            continue
        groups.setdefault(str(value), []).append(row)
    return groups


def all_signals(
    rows: list[dict[str, Any]], *, signals: Iterable[str] = ("cosine", "d_c_score", "calibrated_probability")
) -> dict[str, Any]:
    """Full Study 2 result block, reusing the shared bootstrap helper."""
    from scripts.report_failure_audit import bootstrap_ci

    result: dict[str, Any] = {}
    for signal in signals:
        curve = risk_coverage(rows, signal)
        if not curve:
            result[signal] = {"aurc": "[PLANNED - NOT YET MEASURED]"}
            continue
        result[signal] = {
            "aurc": aurc(curve),
            "selective_accuracy_50": selective_accuracy(curve, 0.5),
            "selective_accuracy_70": selective_accuracy(curve, 0.7),
            "overconfidence_rate": overconfidence_rate(rows, signal, threshold=0.8),
            "curve": curve,
        }
        risks = [p["risk"] for p in curve]
        result[signal]["aurc_ci95"] = bootstrap_ci(risks)
    return result
