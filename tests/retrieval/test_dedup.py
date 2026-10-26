"""The exact and near-duplicate passage deduplication of Document 06 §19 and §22.

Tests exact SHA-256 text hash collapsing and near-duplicate collapsing
(Jaccard/cosine threshold >= 0.95) to maximize information density in the retrieval output.
"""

import pytest

from tests.ids import case, known_gap


def _dedup_module():
    import importlib
    return importlib.import_module("retrieval_dedup")


@known_gap(
    "retrieval_dedup.module_exists",
    "F2",
    "fast",
    "doc/06 §19",
    "The project provides a retrieval deduplication module.",
    "there is no retrieval_dedup module",
)
def test_dedup_module_exists():
    _dedup_module()


@known_gap(
    "retrieval_dedup.exact_hash_collapse",
    "F2",
    "fast",
    "doc/06 §19",
    "Passages with identical normalized text content are collapsed to a single representative chunk.",
    "exact duplicate deduplication is not implemented",
)
def test_exact_hash_collapse():
    mod = _dedup_module()
    passages = [
        {"chunk_id": "c1", "text": "Official election results announced.", "score": 0.85},
        {"chunk_id": "c2", "text": "Official election results announced.", "score": 0.80},
    ]
    deduped = mod.collapse_exact_duplicates(passages)
    assert len(deduped) == 1
    assert deduped[0]["chunk_id"] == "c1"  # Keeps highest scored


@known_gap(
    "retrieval_dedup.near_duplicate_collapse",
    "F2",
    "fast",
    "doc/06 §19",
    "Passages with near-identical wording (similarity >= 0.95) are collapsed to preserve prompt budget.",
    "near duplicate deduplication is not implemented",
)
def test_near_duplicate_collapse():
    mod = _dedup_module()
    passages = [
        {"chunk_id": "c1", "text": "The mayor announced the new budget in city hall yesterday.", "score": 0.9},
        {"chunk_id": "c2", "text": "The mayor announced the new budget in city hall yesterday!", "score": 0.88},
    ]
    deduped = mod.collapse_near_duplicates(passages, threshold=0.95)
    assert len(deduped) == 1
    assert deduped[0]["chunk_id"] == "c1"
