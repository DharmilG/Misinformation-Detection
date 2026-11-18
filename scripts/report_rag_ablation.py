"""Study 4 reporter - stage-attributed metrics from rag_ablation traces.

The paired-delta and same-prediction formulas are the ones used by
``scripts/evaluate_rag_non_rag.py``; the stage diagnostics are new.  A field with
no data stays as ``[PLANNED - NOT YET MEASURED]``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clip_model.metrics import (  # noqa: E402
    compute_accuracy,
    compute_macro_f1,
    compute_roc_auc,
)
from src.clip_model.classifier_calibration import compute_ece  # noqa: E402
from scripts.report_failure_audit import bootstrap_ci  # noqa: E402

PLACEHOLDER = "[PLANNED - NOT YET MEASURED]"
OUTCOMES = ("SUPPORTED", "CONTRADICTED", "INCONCLUSIVE", "CONFLICTING")


def _binary(value: str) -> int:
    return 1 if value == "MISMATCHED" else 0


def _group(rows: list[dict[str, Any]], condition: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("condition") == condition]


def condition_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gold = [_binary(str(r["gold_label"])) for r in rows]
    predicted = [_binary(str(r["predicted_label"])) for r in rows]
    confidence = [float(r.get("confidence", 0.0)) for r in rows]
    result: dict[str, Any] = {
        "rows": len(rows),
        "accuracy": round(float(compute_accuracy(gold, predicted)), 6),
        "macro_f1": round(float(compute_macro_f1(y_true=gold, y_pred=predicted)), 6),
        "confidence": confidence,
    }
    if len(set(gold)) > 1:
        result["roc_auc"] = round(float(compute_roc_auc(gold, confidence)), 6)
    else:
        result["roc_auc"] = PLACEHOLDER

    probs = [float(r.get("d_c_score", r.get("confidence", 0.5))) for r in rows]
    ece, _ = compute_ece(probs, gold)
    result["ece"] = round(float(ece), 6)
    result.pop("confidence")
    return result


def paired(base: list[dict[str, Any]], other: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {str(r["case_id"]): r for r in base}
    shared = [(by_id[str(r["case_id"])], r) for r in other if str(r["case_id"]) in by_id]
    if not shared:
        return {"n": 0, "delta": PLACEHOLDER, "ci95": [PLACEHOLDER, PLACEHOLDER],
                "same_prediction_rate": PLACEHOLDER}
    # Delta is (other correct) - (base correct), paired by case id.
    diffs = [
        float(_binary(str(o["predicted_label"])) == _binary(str(o["gold_label"])))
        - float(_binary(str(b["predicted_label"])) == _binary(str(b["gold_label"])))
        for b, o in shared
    ]
    same = sum(str(b["predicted_label"]) == str(o["predicted_label"]) for b, o in shared)
    return {
        "n": len(shared),
        "delta": round(sum(diffs) / len(diffs), 6),
        "ci95": bootstrap_ci(diffs),
        "same_prediction_rate": round(same / len(shared), 6),
    }


def retrieval_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    recalls: list[float] = []
    precisions: list[float] = []
    synthetic = 0
    empty = 0
    for row in rows:
        outputs = row.get("retrieval_outputs") or {}
        if row.get("retrieval_error"):
            continue
        ids = set(str(i) for i in outputs.get("ids", ()) if i is not None)
        gold = set(str(i) for i in (row.get("gold_evidence_ids") or ()))
        k = max(1, len(ids))
        if not ids:
            empty += 1
        if outputs.get("synthetic_passages"):
            synthetic += 1
        if gold:
            recalls.append(len(ids & gold) / len(gold))
            precisions.append(len(ids & gold) / k)
    def avg(values: list[float]) -> Any:
        return round(sum(values) / len(values), 6) if values else PLACEHOLDER
    return {
        "recall_at_k": avg(recalls),
        "precision_at_k": avg(precisions),
        "synthetic_passage_rate": round(synthetic / len(rows), 6) if rows else PLACEHOLDER,
        "empty_retrieval_rate": round(empty / len(rows), 6) if rows else PLACEHOLDER,
    }


def stage_rates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def rate(prefix: str, key: str, value: Any) -> Any:
        present = [r for r in rows if r.get(f"{prefix}_outputs") is not None]
        if not present:
            return PLACEHOLDER
        hits = sum(
            (r.get(f"{prefix}_outputs") or {}).get(key) == value for r in present
        )
        return round(hits / len(present), 6)

    return {
        "eligibility_rejection_rate": rate("eligibility", "status", "REJECTED"),
        "injection_flag_rate": rate("injection", "result", True),
    }


def outcome_distribution(rows: list[dict[str, Any]]) -> dict[str, int | str]:
    counts: dict[str, int | str] = {name: 0 for name in OUTCOMES}
    for row in rows:
        outcome = (row.get("outcome_outputs") or {}).get("outcome")
        if outcome in counts:
            counts[outcome] = int(counts[outcome]) + 1
    return counts


def primary_failure(row: dict[str, Any]) -> str:
    """Assign one failure mode per case using the documented precedence."""
    retrieval = row.get("retrieval_outputs") or {}
    if not retrieval or row.get("retrieval_error"):
        return "F1"
    if retrieval.get("synthetic_passages") or retrieval.get("passage_count", 0) == 0:
        return "F1"

    # A failed/unavailable assessment means no generator produced a claim
    # assessment.  The evidence was found but never used - this is a
    # generation failure, not a retrieval or evidence-relevance failure.
    assessment = row.get("assessment_outputs") or {}
    detail = str(assessment.get("error_detail") or "")
    if "has no attribute" in detail or ("passage" in detail and "error" in detail.lower()):
        # A schema/interface mismatch, not a model failure.  This was the real
        # cause of every FAILED assessment before the adapter was added.
        return "F10"
    if assessment.get("status") in ("FAILED", "UNAVAILABLE"):
        return "F6"
    if assessment.get("status") in ("SKIPPED", None):
        return "F7"

    if row.get("eligibility_error") or row.get("injection_error"):
        return "F4"
    eligibility = row.get("eligibility_outputs") or {}
    if any(s == "REJECTED" for s in eligibility.get("statuses", ())):
        return "F3"
    injection = row.get("injection_outputs") or {}
    if injection.get("result") is True:
        return "F4"
    return "F5"


def report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    conditions = sorted({str(r["condition"]) for r in rows})
    metrics = {c: condition_metrics(_group(rows, c)) for c in conditions}
    base = _group(rows, "C0") or _group(rows, "C1")
    paired_by_condition = {
        c: paired(base, _group(rows, c)) for c in conditions if c != (base[0]["condition"] if base else None)
    }
    rag_rows = _group(rows, "C5") or _group(rows, "C4")
    result: dict[str, Any] = {
        "measurement_count": len(rows),
        "conditions": metrics,
        "paired_vs_baseline": paired_by_condition,
        "retrieval": retrieval_metrics(_group(rows, "C2") or rag_rows),
        "stage_rates": stage_rates(rag_rows),
        "outcome_distribution": outcome_distribution(rag_rows),
        "citation_validity_rate": PLACEHOLDER,
        "semantic_support_accuracy": PLACEHOLDER,
        "primary_failure_modes": {
            mode: sum(primary_failure(r) == mode for r in rag_rows)
            for mode in ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10")
        },
        "verdict_RQ1": PLACEHOLDER,
        "verdict_RQ2": PLACEHOLDER,
        "verdict_RQ3": PLACEHOLDER,
        "verdict_RQ4": PLACEHOLDER,
        "interpretation": PLACEHOLDER,
        "reproduction": PLACEHOLDER,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    path = Path(args.input)
    with path.open("r", encoding="utf-8-sig") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    result = report(rows)
    result["source"] = str(path)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
