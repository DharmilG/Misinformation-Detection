"""Relevance filtering and diversity capping algorithm of Document 06 §19 and §21.

Applies absolute dense floor (tau_dense), relative score drop band (alpha_relative),
and soft diversity caps while preserving relevant contradictory evidence.
"""

from __future__ import annotations

from typing import Any


def apply_dense_floor(
    candidates: list[dict[str, Any]], tau_dense: float = 0.50
) -> list[dict[str, Any]]:
    """Stage A: drop candidates whose dense score is below absolute gate."""
    return [c for c in candidates if c.get("dense_score", 0.0) >= tau_dense]


def apply_relative_drop(
    candidates: list[dict[str, Any]], alpha_relative: float = 0.50
) -> list[dict[str, Any]]:
    """Stage B: drop candidates whose score is below alpha_relative * top_score."""
    if not candidates:
        return []
    top_score = max(c.get("score", 0.0) for c in candidates)
    cutoff = alpha_relative * top_score
    return [c for c in candidates if c.get("score", 0.0) >= cutoff]


def apply_diversity_caps(
    candidates: list[dict[str, Any]], max_per_doc: int = 2, target_k: int = 3
) -> list[dict[str, Any]]:
    """Soft diversity caps: limit chunks per doc only while alternatives exist."""
    sorted_candidates = sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)
    doc_counts: dict[str, int] = {}
    selected: list[dict[str, Any]] = []

    for c in sorted_candidates:
        doc_id = c.get("doc_id", c.get("document_id", "default"))
        count = doc_counts.get(doc_id, 0)

        # Check if alternative documents exist
        has_alternatives = any(
            cand.get("doc_id", cand.get("document_id", "default")) != doc_id
            for cand in sorted_candidates[len(selected):]
        )

        if count < max_per_doc or not has_alternatives:
            selected.append(c)
            doc_counts[doc_id] = count + 1

        if len(selected) >= target_k:
            break

    return selected


def filter_candidates(
    candidates: list[dict[str, Any]],
    tau_dense: float = 0.50,
    alpha_relative: float = 0.50,
) -> list[dict[str, Any]]:
    """Apply content-neutral relevance filtering without dropping counter-evidence."""
    stage_a = apply_dense_floor(candidates, tau_dense=tau_dense)
    stage_b = apply_relative_drop(stage_a, alpha_relative=alpha_relative)
    return stage_b
