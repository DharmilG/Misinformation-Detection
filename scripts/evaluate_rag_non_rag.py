"""Evaluate paired RAG and non-RAG outcomes from a labeled JSONL manifest.

The manifest must contain real, precomputed outputs from both arms. This
script never infers gold labels from model predictions.
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

from src.rag.rag_evaluation import compute_generation_metrics, compute_outcome_confusion_matrix


OUTCOMES = ["SUPPORTED", "CONTRADICTED", "INCONCLUSIVE", "CONFLICTING"]


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("id", "gold_outcome", "non_rag_outcome", "rag_outcome"):
                if key not in row:
                    raise ValueError(f"{path}:{line_number}: missing '{key}'")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no evaluation rows")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Evaluation row ids must be unique")
    return rows


def _accuracy(gold: list[str], predicted: list[str]) -> float:
    return sum(a == b for a, b in zip(gold, predicted)) / len(gold)


def _arm_metrics(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    gold = [row["gold_outcome"] for row in rows]
    predicted = [row[field] for row in rows]
    unknown = sorted(set(gold + predicted) - set(OUTCOMES))
    if unknown:
        raise ValueError(f"Unknown outcome values: {unknown}")
    per_class: dict[str, dict[str, float]] = {}
    for label in OUTCOMES:
        tp = sum(g == label and p == label for g, p in zip(gold, predicted))
        fp = sum(g != label and p == label for g, p in zip(gold, predicted))
        fn = sum(g == label and p != label for g, p in zip(gold, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
    return {
        "rows": len(rows),
        "accuracy": _accuracy(gold, predicted),
        "macro_f1": sum(item["f1"] for item in per_class.values()) / len(OUTCOMES),
        "per_class": per_class,
        "confusion_matrix": compute_outcome_confusion_matrix(gold, predicted, OUTCOMES),
    }


def evaluate(path: Path) -> dict[str, Any]:
    rows = _load_rows(path)
    claims = [row.get("claim") for row in rows if row.get("claim") is not None]
    if claims and len(claims) != len(rows):
        raise ValueError("Every row must provide the same claim field when claim is present")
    if claims and len(set(claims)) != len(claims):
        raise ValueError("Paired evaluation claims must be unique")
    non_rag = _arm_metrics(rows, "non_rag_outcome")
    rag = _arm_metrics(rows, "rag_outcome")
    paired = {
        "rag_minus_non_rag_accuracy": rag["accuracy"] - non_rag["accuracy"],
        "same_prediction_rate": sum(
            row["rag_outcome"] == row["non_rag_outcome"] for row in rows
        ) / len(rows),
    }
    generation_rows = [
        row["rag_generation"]
        for row in rows
        if isinstance(row.get("rag_generation"), dict)
    ]
    return {
        "protocol": {
            "baseline": "classifier_only",
            "paired": True,
            "identical_claims": True,
            "non_rag_evidence_empty": True,
        },
        "non_rag": non_rag,
        "rag": rag,
        "rag_generation": compute_generation_metrics(generation_rows),
        "paired": paired,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
