"""The relevance filtering and diversity capping algorithm of Document 06 §19 and §21.

Tests dense floor absolute gate (tau_dense), relative score drop band (alpha_relative),
soft diversity caps (only applied while alternatives exist), and retention of contradictory passages.
"""

import pytest

from tests.ids import case


def _filters_module():
    import importlib
    return importlib.import_module("retrieval_filters")


@case(
    "retrieval_filters.module_exists",
    "F2",
    "fast",
    "doc/06 §19",
    "The project provides a retrieval relevance filtering and diversity capping module.",
)
def test_filters_module_exists():
    _filters_module()


@case(
    "retrieval_filters.dense_floor_absolute_gate",
    "F2",
    "fast",
    "doc/06 §19.1, §19.2",
    "Candidates whose dense cosine similarity is below tau_dense are dropped at Stage A.",
)
def test_dense_floor_absolute_gate():
    mod = _filters_module()
    candidates = [
        {"chunk_id": "c1", "dense_score": 0.65},
        {"chunk_id": "c2", "dense_score": 0.35},
    ]
    # tau_dense = 0.50
    filtered = mod.apply_dense_floor(candidates, tau_dense=0.50)
    assert len(filtered) == 1
    assert filtered[0]["chunk_id"] == "c1"


@case(
    "retrieval_filters.relative_drop_band",
    "F2",
    "fast",
    "doc/06 §19.2",
    "Passages whose final score is below alpha_relative * top_score are dropped at Stage B.",
)
def test_relative_drop_band():
    mod = _filters_module()
    # Top score is 0.80. If alpha_relative = 0.5, cutoff is 0.40
    candidates = [
        {"chunk_id": "c1", "score": 0.80},
        {"chunk_id": "c2", "score": 0.50},
        {"chunk_id": "c3", "score": 0.30},
    ]
    passed = mod.apply_relative_drop(candidates, alpha_relative=0.50)
    ids = [c["chunk_id"] for c in passed]
    assert ids == ["c1", "c2"]


@case(
    "retrieval_filters.soft_diversity_caps",
    "F2",
    "fast",
    "doc/06 §19.2",
    "Soft diversity caps limit chunks per document/source only while alternatives exist, never discarding the last passage.",
)
def test_soft_diversity_caps():
    mod = _filters_module()
    # 3 chunks from docA, 1 from docB. Max 2 per doc allowed.
    # While alternatives exist (docB), docA chunk 3 is capped out.
    candidates = [
        {"chunk_id": "docA::c1", "doc_id": "docA", "score": 0.9},
        {"chunk_id": "docA::c2", "doc_id": "docA", "score": 0.8},
        {"chunk_id": "docA::c3", "doc_id": "docA", "score": 0.7},
        {"chunk_id": "docB::c1", "doc_id": "docB", "score": 0.6},
    ]
    capped = mod.apply_diversity_caps(candidates, max_per_doc=2, target_k=3)
    capped_ids = [c["chunk_id"] for c in capped]
    assert "docA::c3" not in capped_ids
    assert "docB::c1" in capped_ids


@case(
    "retrieval_filters.contradictory_relevant_passages_retained",
    "F2",
    "fast",
    "doc/06 §19.3",
    "Filtering measures relevance only; passages that contradict the claim but are relevant are never filtered out.",
)
def test_contradictory_relevant_passages_retained():
    mod = _filters_module()
    contradictory_passage = {
        "chunk_id": "c_factcheck",
        "dense_score": 0.78,
        "score": 0.75,
        "is_counter_evidence": True,
    }
    retained = mod.filter_candidates([contradictory_passage], tau_dense=0.50, alpha_relative=0.50)
    assert len(retained) == 1
