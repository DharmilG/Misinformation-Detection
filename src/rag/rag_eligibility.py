"""Deterministic citation eligibility gates."""
from __future__ import annotations

import unicodedata


def _quote_verified(quote: str, passage: str) -> bool:
    normalize = lambda value: " ".join(unicodedata.normalize("NFC", value).split())
    return normalize(quote) in normalize(passage)


def evaluate_citation_eligibility(citation: dict, known_components: set[str], valid_passages: dict[str, str]) -> dict:
    pid = citation.get("passage_id")
    if pid not in valid_passages:
        return {**citation, "eligible": False, "reason": "UNKNOWN_PASSAGE", "ineligibility_reason": "UNKNOWN_PASSAGE"}
    if not _quote_verified(str(citation.get("quote", "")), valid_passages[pid]):
        return {**citation, "eligible": False, "reason": "QUOTE_NOT_VERBATIM", "ineligibility_reason": "QUOTE_NOT_VERBATIM"}
    relation = citation.get("relation")
    if relation == "NEUTRAL":
        return {**citation, "eligible": False, "reason": "RELATION_NEUTRAL", "ineligibility_reason": "RELATION_NEUTRAL"}
    if relation not in {"SUPPORTS", "CONTRADICTS"}:
        return {**citation, "eligible": False, "reason": "INVALID_RELATION", "ineligibility_reason": "INVALID_RELATION"}
    if not citation.get("context_sufficient", False):
        return {**citation, "eligible": False, "reason": "INSUFFICIENT_CONTEXT", "ineligibility_reason": "INSUFFICIENT_CONTEXT"}
    if citation.get("claim_component_id") not in known_components:
        return {**citation, "eligible": False, "reason": "UNKNOWN_COMPONENT", "ineligibility_reason": "UNKNOWN_COMPONENT"}
    return {**citation, "eligible": True, "reason": None}


def process_citations(citations: list[dict], known_components: set[str], valid_passages: dict[str, str]) -> list[dict]:
    return [evaluate_citation_eligibility(c, known_components, valid_passages) for c in citations]
