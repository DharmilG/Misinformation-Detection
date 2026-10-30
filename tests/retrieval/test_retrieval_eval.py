"""The retrieval evaluation protocol and ranking metrics of Document 06 §9, §24, §25, and §38.

Tests evaluation metrics computation (Hit@k, Mean Reciprocal Rank MRR, NDCG@k),
and split isolation between threshold-tuning dev queries and sealed test queries.
"""

import pytest

from tests.ids import case


def _eval_module():
    import importlib
    return importlib.import_module("retrieval_eval")


@case(
    "retrieval_eval.module_exists",
    "F2",
    "fast",
    "doc/06 §9, §38",
    "The project provides a retrieval evaluation and metrics module.",
)
def test_retrieval_eval_module_exists():
    _eval_module()


@case(
    "retrieval_eval.dev_test_split_isolation",
    "F2",
    "fast",
    "doc/06 §38",
    "Retrieval dev set tunes thresholds (tau_dense, alpha_relative) while the test set remains sealed.",
)
def test_dev_test_split_isolation():
    mod = _eval_module()
    splits = mod.load_evaluation_query_splits()
    dev_ids = {q["query_id"] for q in splits["dev"]}
    test_ids = {q["query_id"] for q in splits["test"]}
    assert not (dev_ids & test_ids), "leakage detected: dev and test query sets overlap"


@case(
    "retrieval_eval.metrics_hit_mrr_ndcg",
    "F2",
    "fast",
    "doc/06 §38",
    "Calculates Hit@k, MRR, and NDCG@k correctly across evaluated query rankings.",
)
def test_metrics_hit_mrr_ndcg():
    mod = _eval_module()
    # Query 1: relevant item at rank 2 -> Reciprocal Rank = 0.5, Hit@5 = 1.0
    # Query 2: relevant item not in top 5 -> RR = 0.0, Hit@5 = 0.0
    results = [
        {"ranks": [2], "k": 5},
        {"ranks": [10], "k": 5},
    ]
    metrics = mod.compute_evaluation_metrics(results)
    assert abs(metrics["mrr"] - 0.25) < 1e-5
    assert abs(metrics["hit_at_5"] - 0.50) < 1e-5
