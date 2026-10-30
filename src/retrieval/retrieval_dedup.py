"""Exact and near-duplicate passage deduplication of Document 06 §19 and §22.

Collapses identical text hashes and near-duplicate passages (similarity >= 0.95)
to maximize information density.
"""

from __future__ import annotations

import difflib
from typing import Any


def collapse_exact_duplicates(passages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse passages with identical normalized text, keeping highest score."""
    # Sort descending by score first
    sorted_passages = sorted(passages, key=lambda x: x.get("score", 0.0), reverse=True)
    seen_texts: set[str] = set()
    deduped: list[dict[str, Any]] = []

    for p in sorted_passages:
        norm_text = p.get("text", "").strip()
        if norm_text not in seen_texts:
            seen_texts.add(norm_text)
            deduped.append(p)

    return deduped


def _text_similarity(s1: str, s2: str) -> float:
    """Calculate character sequence matcher ratio."""
    return difflib.SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


def collapse_near_duplicates(
    passages: list[dict[str, Any]], threshold: float = 0.95
) -> list[dict[str, Any]]:
    """Collapse passages with wording similarity >= threshold, keeping highest score."""
    sorted_passages = sorted(passages, key=lambda x: x.get("score", 0.0), reverse=True)
    retained: list[dict[str, Any]] = []

    for p in sorted_passages:
        t = p.get("text", "").strip()
        is_dup = False
        for rep in retained:
            rep_text = rep.get("text", "").strip()
            if _text_similarity(t, rep_text) >= threshold:
                is_dup = True
                break
        if not is_dup:
            retained.append(p)

    return retained
