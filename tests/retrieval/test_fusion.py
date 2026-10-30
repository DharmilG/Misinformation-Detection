"""The Reciprocal Rank Fusion (RRF) algorithm of Document 06 §16 and §19.

Tests RRF formula score(d) = sum(1 / (60 + rank)), constant k=60,
deterministic tie-breaking by chunk_id, and single-retriever graceful degradation.
"""

import pytest

from tests.ids import case


def _fusion_module():
    import importlib
    return importlib.import_module("retrieval_fusion")


@case(
    "rrf_fusion.module_exists",
    "F2",
    "fast",
    "doc/06 §16",
    "The project provides a Reciprocal Rank Fusion module.",
)
def test_fusion_module_exists():
    _fusion_module()


@case(
    "rrf_fusion.constant_k_60",
    "F2",
    "fast",
    "doc/06 §16.1",
    "RRF computes scores with standard smoothing constant k=60: 1 / (60 + rank).",
)
def test_constant_k_60():
    mod = _fusion_module()
    # Rank 1: score = 1 / (60 + 1) = 1 / 61
    score_rank1 = mod.compute_rrf_component(rank=1, k=60)
    assert abs(score_rank1 - (1.0 / 61.0)) < 1e-6

    # Candidate ranked #1 in dense and #2 in lexical: 1/61 + 1/62
    dense_rankings = [("c1", 1), ("c2", 2)]
    lexical_rankings = [("c2", 1), ("c1", 2)]
    fused = mod.fuse_rankings(dense_rankings, lexical_rankings, k=60)
    # c1: 1/61 + 1/62 = c2: 1/62 + 1/61 (equal scores)
    expected = (1.0 / 61.0) + (1.0 / 62.0)
    assert abs(fused["c1"] - expected) < 1e-6


@case(
    "rrf_fusion.tie_breaking_by_chunk_id",
    "F2",
    "fast",
    "doc/06 §16.1",
    "Ties in fused RRF score are broken deterministically by chunk_id alphabetical order.",
)
def test_tie_breaking_by_chunk_id():
    mod = _fusion_module()
    # Identical scores for c_beta and c_alpha
    ranked = mod.sort_fused_candidates({"c_beta": 0.03, "c_alpha": 0.03})
    assert ranked[0][0] == "c_alpha"
    assert ranked[1][0] == "c_beta"


@case(
    "rrf_fusion.graceful_single_retriever_degradation",
    "F2",
    "fast",
    "doc/06 §16.2",
    "When one retriever returns no candidates, RRF degrades gracefully without crashing or zeroing output.",
)
def test_graceful_single_retriever_degradation():
    mod = _fusion_module()
    dense_rankings = [("c1", 1), ("c2", 2)]
    lexical_rankings = []  # Empty lexical results
    fused = mod.fuse_rankings(dense_rankings, lexical_rankings, k=60)
    assert len(fused) == 2
    assert fused["c1"] > fused["c2"]
