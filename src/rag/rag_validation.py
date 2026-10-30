"""Strict validation for structured generation output."""
from __future__ import annotations

import re
import unicodedata

ALLOWED_KEYS = {
    "claim_restated", "claim_components", "evidence_summary", "citations",
    "unresolved", "conflicts", "uncertainties", "limitations",
}


def validate_schema(payload: dict) -> dict:
    unexpected = set(payload) - ALLOWED_KEYS
    if unexpected or not isinstance(payload, dict):
        return {"valid": False, "reason": "UNEXPECTED_KEY"}
    if "outcome" in payload:
        return {"valid": False, "reason": "MODEL_OUTCOME_FORBIDDEN"}
    for key in ("claim_components", "citations", "unresolved", "conflicts", "uncertainties", "limitations"):
        if key in payload and not isinstance(payload[key], list):
            return {"valid": False, "reason": "INVALID_FIELD_TYPE"}
        if isinstance(payload.get(key), list) and len(payload[key]) > 50:
            return {"valid": False, "reason": "FIELD_TOO_LONG"}
    return {"valid": True, "reason": None}


def verify_quote(quote: str, passage: str) -> bool:
    normalize = lambda value: " ".join(unicodedata.normalize("NFC", value).split())
    normalized_quote = normalize(quote)
    normalized_passage = normalize(passage)
    if "..." in quote or "…" in quote or not normalized_quote:
        return False
    if re.search(r"\bnot\s+" + re.escape(normalized_quote) + r"\b", normalized_passage, re.IGNORECASE):
        return False
    return normalized_quote in normalized_passage


def scan_for_prohibited_urls(payload: object) -> dict:
    text = repr(payload)
    return {"valid": not bool(re.search(r"https?://|www\.", text)), "reason": "URL_OUTSIDE_QUOTE" if re.search(r"https?://|www\.", text) else None}


def validate_generation_urls(payload: dict, passages: dict[str, str]) -> dict:
    for citation in payload.get("citations", []):
        quote = citation.get("quote", "")
        source = passages.get(citation.get("passage_id"), "")
        if re.search(r"https?://|www\.", quote) and not verify_quote(quote, source):
            return {"valid": False, "reason": "URL_NOT_VERIFIED"}
    summary = dict(payload)
    summary["citations"] = []
    return scan_for_prohibited_urls(summary)


class ValidationRetryTracker:
    def __init__(self, max_retries: int = 1):
        self.max_retries = max_retries
        self.failures = 0

    def can_retry(self) -> bool:
        return self.failures < self.max_retries

    def record_failure(self, _reason: str) -> None:
        self.failures += 1
