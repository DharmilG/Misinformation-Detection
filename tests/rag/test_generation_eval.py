"""The RAG and non-RAG generation evaluation protocol of Document 06 §34, §39, and §40-§44.

Tests citation validity versus semantic support separation, 4x4 outcome confusion matrix,
eligibility/rejection rate tracking, and the RAG/non-RAG experimental comparison protocol.
"""

import pytest

from tests.ids import case, known_gap


def _eval_module():
    import importlib
    return importlib.import_module("rag_evaluation")


@known_gap(
    "rag_eval.module_exists",
    "F2",
    "fast",
    "doc/06 §39",
    "The project provides a generation and RAG evaluation module.",
    "there is no rag_evaluation module",
)
def test_generation_eval_module_exists():
    _eval_module()


@known_gap(
    "rag_eval.citation_validity_vs_semantic_support_separated",
    "F2",
    "fast",
    "doc/06 §39.1",
    "Citation validity (deterministic quote verification) and semantic support (human/LLM agreement) are reported as distinct metrics.",
    "separated evaluation metrics are not implemented",
)
def test_citation_validity_vs_semantic_support_separated():
    mod = _eval_module()
    sample_eval = [
        {"quote_verified": True, "human_support_agreed": True},
        {"quote_verified": True, "human_support_agreed": False},  # valid quote, wrong semantic label
    ]
    metrics = mod.compute_generation_metrics(sample_eval)
    assert "citation_validity_rate" in metrics
    assert "semantic_support_accuracy" in metrics
    assert metrics["citation_validity_rate"] == 1.0
    assert metrics["semantic_support_accuracy"] == 0.5


@known_gap(
    "rag_eval.four_by_four_confusion_matrix",
    "F2",
    "fast",
    "doc/06 §39.2",
    "The evaluation produces a 4x4 confusion matrix over [SUPPORTED, CONTRADICTED, CONFLICTING, INCONCLUSIVE].",
    "4x4 confusion matrix generation is not implemented",
)
def test_four_by_four_confusion_matrix():
    mod = _eval_module()
    classes = ["SUPPORTED", "CONTRADICTED", "CONFLICTING", "INCONCLUSIVE"]
    y_true = ["SUPPORTED", "CONTRADICTED", "INCONCLUSIVE", "CONFLICTING"]
    y_pred = ["SUPPORTED", "CONTRADICTED", "INCONCLUSIVE", "INCONCLUSIVE"]
    cm = mod.compute_outcome_confusion_matrix(y_true, y_pred, classes=classes)
    assert len(cm) == 4
    assert len(cm[0]) == 4


@known_gap(
    "rag_eval.rag_vs_non_rag_protocol",
    "F2",
    "fast",
    "doc/06 §40",
    "The RAG vs non-RAG comparison evaluates identical claims under identical models with and without evidence.",
    "RAG comparative experiment runner is not implemented",
)
def test_rag_vs_non_rag_protocol():
    mod = _eval_module()
    protocol = mod.get_comparison_protocol()
    assert protocol["paired"] is True
    assert protocol["identical_claims"] is True
    assert protocol["non_rag_evidence_empty"] is True
