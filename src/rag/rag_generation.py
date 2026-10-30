"""Assessment schema and deterministic local generation interface."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from .rag_eligibility import process_citations
    from .rag_outcome import compute_claim_outcome, compute_component_status
    from .rag_validation import validate_schema
except ImportError:
    from rag_eligibility import process_citations
    from rag_outcome import compute_claim_outcome, compute_component_status
    from rag_validation import validate_schema


@dataclass(frozen=True)
class Assessment:
    status: str
    outcome: str | None
    outcome_source: str
    outcome_reason: str | None
    scope_text: str | None
    claim_restated: str | None
    claim_components: Any
    evidence_summary: str | None
    citations: Any
    rejected_citations: Any
    unresolved: Any
    conflicts: Any
    uncertainties: Any
    limitations: Any
    provenance: Any
    warnings: tuple[str, ...]
    error: Any
    model_proposed_outcome: str | None = None


def create_failed_assessment(status: str = "UNAVAILABLE", error_message: str = "Generation failed") -> Assessment:
    return Assessment(
        status=status, outcome=None, outcome_source="none", outcome_reason=None,
        scope_text=None, claim_restated=None, claim_components=None,
        evidence_summary=None, citations=None, rejected_citations=None,
        unresolved=None, conflicts=None, uncertainties=None, limitations=None,
        provenance=None, warnings=(), error={"message": error_message},
    )


def create_no_evidence_assessment() -> Assessment:
    return Assessment(
        status="OK", outcome="INCONCLUSIVE", outcome_source="none",
        outcome_reason="NO_EVIDENCE", scope_text=None, claim_restated=None,
        claim_components=None, evidence_summary=None, citations=None,
        rejected_citations=None, unresolved=None, conflicts=None,
        uncertainties=None, limitations=None, provenance=None, warnings=(),
        error=None,
    )


def assess_structured_output(
    *,
    claim: str,
    generated: dict,
    passages: dict[str, str],
) -> Assessment:
    """Validate a model-shaped payload and compute outcome from citations only."""
    schema = validate_schema(generated)
    if not schema["valid"]:
        return create_failed_assessment(error_message=schema["reason"])

    raw_components = generated.get("claim_components") or [{"id": "K1", "text": claim}]
    components = {str(c.get("id")) for c in raw_components if isinstance(c, dict) and c.get("id")}
    processed = process_citations(generated.get("citations", []), components, passages)
    statuses: dict[str, str] = {}
    for component in components:
        rows = [c for c in processed if c.get("claim_component_id") == component and c.get("eligible")]
        statuses[component] = compute_component_status(
            any(c.get("relation") == "SUPPORTS" for c in rows),
            any(c.get("relation") == "CONTRADICTS" for c in rows),
        )
    result = compute_claim_outcome(statuses, bool(processed))
    if result["reason"] == "NO_EVIDENCE":
        return create_no_evidence_assessment()
    accepted = tuple(c for c in processed if c.get("eligible"))
    rejected = tuple(c for c in processed if not c.get("eligible"))
    return Assessment(
        status="OK",
        outcome=result["outcome"],
        outcome_source="computed",
        outcome_reason=result["reason"],
        scope_text=None,
        claim_restated=generated.get("claim_restated", claim),
        claim_components=tuple(raw_components),
        evidence_summary=generated.get("evidence_summary"),
        citations=accepted,
        rejected_citations=rejected,
        unresolved=tuple(generated.get("unresolved", ())),
        conflicts=tuple(generated.get("conflicts", ())),
        uncertainties=tuple(generated.get("uncertainties", ())),
        limitations=tuple(generated.get("limitations", ())),
        provenance=None,
        warnings=(),
        error=None,
    )
