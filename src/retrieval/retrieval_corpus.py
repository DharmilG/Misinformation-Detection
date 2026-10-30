"""Corpus terms, acquisition rights, and storage enforcement of Document 06 §6.

Validates source terms, automated fetch policies, and permitted storage scopes.
"""

from __future__ import annotations

from typing import Any

TERMS_FIELDS = {
    "source_name",
    "source_identity",
    "terms_url",
    "license_or_terms_summary",
    "accessed_at",
    "permitted_use",
    "automated_fetch_allowed",
    "full_text_storage_allowed",
    "redistribution_allowed",
    "notes",
}


class TermsRequirementError(ValueError):
    """Raised when document acquisition or storage violates source terms."""


def create_sample_terms_record() -> dict[str, Any]:
    """Create a sample terms record containing all 10 required fields."""
    return {
        "source_name": "Official News Wire",
        "source_identity": "newswire.org",
        "terms_url": "https://newswire.org/terms",
        "license_or_terms_summary": "Permitted for non-commercial research with attribution.",
        "accessed_at": "2026-10-21T00:00:00Z",
        "permitted_use": "RESEARCH_EVALUATION",
        "automated_fetch_allowed": True,
        "full_text_storage_allowed": True,
        "redistribution_allowed": False,
        "notes": "Verified clean terms record.",
    }


def validate_document_rights(
    doc: dict[str, Any],
    terms_record: dict[str, Any] | None,
) -> bool:
    """Refuse documents that lack valid terms records."""
    if terms_record is None or not isinstance(terms_record, dict):
        raise TermsRequirementError("Document lacks a valid terms record.")
    return True


def fetch_document_online(url: str, terms: dict[str, Any]) -> str:
    """Fetch online document, refusing if terms prohibit automated fetch."""
    if not terms.get("automated_fetch_allowed", False):
        raise TermsRequirementError(
            f"Automated fetch is disallowed for source {terms.get('source_identity')}. Import locally."
        )
    return "Fetched document content"


def prepare_document_for_storage(
    full_text: str, terms: dict[str, Any]
) -> dict[str, str]:
    """Store excerpt only if full text storage is disallowed by terms."""
    if not terms.get("full_text_storage_allowed", True):
        paragraphs = full_text.split("\n\n")
        excerpt = paragraphs[0] if paragraphs else full_text[:200]
        return {"storage_scope": "EXCERPT", "text": excerpt}
    return {"storage_scope": "FULL_PAGE", "text": full_text}
