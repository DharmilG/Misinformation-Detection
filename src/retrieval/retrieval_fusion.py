"""Reciprocal Rank Fusion (RRF) algorithm of Document 06 §16 and §19.

Fuses dense and lexical rankings using standard smoothing constant k=60,
breaking ties deterministically by chunk_id alphabetical order.
"""

from __future__ import annotations

from typing import Sequence


def compute_rrf_component(rank: int, k: int = 60) -> float:
    """Compute standard RRF score component 1 / (k + rank)."""
    return 1.0 / float(k + rank)


def fuse_rankings(
    dense_rankings: Sequence[tuple[str, int]],
    lexical_rankings: Sequence[tuple[str, int]],
    k: int = 60,
) -> dict[str, float]:
    """Fuse dense and lexical rankings with single-retriever graceful fallback."""
    scores: dict[str, float] = {}

    for chunk_id, rank in dense_rankings:
        scores[chunk_id] = scores.get(chunk_id, 0.0) + compute_rrf_component(rank, k=k)

    for chunk_id, rank in lexical_rankings:
        scores[chunk_id] = scores.get(chunk_id, 0.0) + compute_rrf_component(rank, k=k)

    return scores


def sort_fused_candidates(fused_scores: dict[str, float]) -> list[tuple[str, float]]:
    """Sort fused candidates descending by score, breaking ties alphabetically by chunk_id."""
    return sorted(fused_scores.items(), key=lambda item: (-item[1], item[0]))
