"""Claim and component outcome rules from Document 06 §30."""
from __future__ import annotations


def compute_component_status(has_eligible_support: bool, has_eligible_contradict: bool) -> str:
    if has_eligible_support and has_eligible_contradict:
        return "CONTESTED"
    if has_eligible_support:
        return "SUPPORTED"
    if has_eligible_contradict:
        return "CONTRADICTED"
    return "UNRESOLVED"


def compute_claim_outcome(component_statuses: dict[str, str], has_citations: bool) -> dict:
    if not has_citations:
        return {"outcome": "INCONCLUSIVE", "reason": "NO_EVIDENCE"}
    if not component_statuses or all(s == "UNRESOLVED" for s in component_statuses.values()):
        return {"outcome": "INCONCLUSIVE", "reason": "NO_ELIGIBLE_EVIDENCE"}
    if any(s == "CONTESTED" for s in component_statuses.values()):
        return {"outcome": "CONFLICTING", "reason": "CONFLICTING_EVIDENCE"}
    if all(s == "SUPPORTED" for s in component_statuses.values()):
        return {"outcome": "SUPPORTED", "reason": None}
    if any(s == "CONTRADICTED" for s in component_statuses.values()):
        return {"outcome": "CONTRADICTED", "reason": "PARTIAL_CONTRADICTION"}
    return {"outcome": "INCONCLUSIVE", "reason": "INSUFFICIENT_EVIDENCE"}
