"""Separated RAG/non-RAG evaluation metrics and paired protocol."""
from __future__ import annotations


def compute_generation_metrics(rows: list[dict]) -> dict[str, float]:
    total = len(rows)
    if not total:
        return {"citation_validity_rate": 0.0, "semantic_support_accuracy": 0.0}
    return {
        "citation_validity_rate": sum(bool(r.get("quote_verified")) for r in rows) / total,
        "semantic_support_accuracy": sum(bool(r.get("human_support_agreed")) for r in rows) / total,
    }


def compute_outcome_confusion_matrix(y_true: list[str], y_pred: list[str], classes: list[str]) -> list[list[int]]:
    matrix = [[0 for _ in classes] for _ in classes]
    positions = {value: i for i, value in enumerate(classes)}
    for truth, prediction in zip(y_true, y_pred):
        if truth in positions and prediction in positions:
            matrix[positions[truth]][positions[prediction]] += 1
    return matrix


def get_comparison_protocol() -> dict[str, bool]:
    return {"paired": True, "identical_claims": True, "identical_model": True, "non_rag_evidence_empty": True}
