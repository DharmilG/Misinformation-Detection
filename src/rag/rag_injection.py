"""Structural prompt-injection defenses."""
from __future__ import annotations

import re

def is_injection_detected(payload: dict) -> bool:
    return bool(re.search(r"https?://|www\.", repr(payload)))


def validate_citations_against_supplied(payload: dict, supplied_passages: dict[str, str]) -> dict:
    rejected = [
        {"passage_id": c.get("passage_id"), "reason": "UNKNOWN_PASSAGE"}
        for c in payload.get("citations", [])
        if c.get("passage_id") not in supplied_passages
    ]
    return {"rejected_citations": rejected}


def evaluate_injected_scenario(passages: list[dict], raw_model_text: str):
    class Result:
        outcome = "INCONCLUSIVE"
    return Result()
