"""
src/rag/assessment.py
======================
Keyword-heuristic claim assessment over retrieved passages.

Spec: 05_REPORTING_AND_UI §7, 04_INPUT_AND_INFERENCE_PIPELINE (implied).

Behaviour:
  - No LLM or external API required.
  - Uses keyword matching to classify each passage as supporting or
    contradicting the claim expressed in the caption.
  - Returns INCONCLUSIVE when no passages are retrieved or when the
    evidence does not settle the claim.
  - All returned values are derived from actual corpus content;
    nothing is fabricated.
  - Never raises to the caller.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

from src.reporting import copy as C
from src.reporting.model import (
    CitationRecord,
    ClaimComponent,
    RetrievedPassage,
)
from src.utils.config import PROMPT_VERSION


# ── Result dataclass ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class Assessment:
    status: str                          # "OK" | "UNAVAILABLE" | "SKIPPED" | "FAILED"
    prompt_version: str
    scope_qualifier: str
    # OK-only
    outcome: Optional[str] = None        # "SUPPORTED"|"CONTRADICTED"|"INCONCLUSIVE"|"CONFLICTING"
    reason_text: Optional[str] = None
    components: tuple[ClaimComponent, ...] = field(default_factory=tuple)
    accepted_citations: tuple[CitationRecord, ...] = field(default_factory=tuple)
    rejected_citations: tuple[CitationRecord, ...] = field(default_factory=tuple)
    unresolved_items: tuple[str, ...] = field(default_factory=tuple)
    conflicts: tuple[str, ...] = field(default_factory=tuple)
    uncertainties: tuple[str, ...] = field(default_factory=tuple)
    limitations: tuple[str, ...] = field(default_factory=tuple)
    # non-OK
    reason: Optional[str] = None
    error_detail: Optional[str] = None


# ── Keyword lists ─────────────────────────────────────────────────────────

_CONTRADICT_WORDS = frozenset([
    "false", "incorrect", "wrong", "denied", "debunked", "hoax", "fabricated",
    "misleading", "misrepresented", "not true", "no evidence", "disproven",
    "refuted", "disputed", "unverified", "misinformation", "fake",
])

_SUPPORT_WORDS = frozenset([
    "confirmed", "true", "accurate", "authentic", "verified",
    "reported", "documented", "published", "official", "evidence shows",
    "according to", "stated", "announced",
])


def _score_passage(passage_excerpt: str, caption_words: set[str]) -> tuple[str, float]:
    """
    Return ('supports'|'contradicts'|'unresolved', confidence 0.0–1.0).
    Purely keyword-based heuristic.
    """
    text = passage_excerpt.lower()
    words = set(re.findall(r"\b\w+\b", text))

    overlap = len(words & caption_words) / max(len(caption_words), 1)

    contra_hits = sum(1 for w in _CONTRADICT_WORDS if w in text)
    support_hits = sum(1 for w in _SUPPORT_WORDS if w in text)

    if contra_hits > support_hits:
        return "contradicted", min(0.9, 0.4 + 0.1 * contra_hits + 0.2 * overlap)
    elif support_hits > contra_hits:
        return "supported", min(0.9, 0.4 + 0.1 * support_hits + 0.2 * overlap)
    else:
        return "unresolved", 0.3 * overlap


# ── Public API ────────────────────────────────────────────────────────────

def run_assessment(
    caption_normalized: str,
    retrieved_passages: tuple[RetrievedPassage, ...],
) -> Assessment:
    """
    Assess the claim in the caption against the retrieved passages.

    Returns Assessment with status OK, UNAVAILABLE, SKIPPED, or FAILED.
    Never raises.
    """
    _common = dict(
        prompt_version=PROMPT_VERSION,
        scope_qualifier=C.SCOPE_QUALIFIER,
    )

    if not retrieved_passages:
        return Assessment(
            status="OK",
            outcome="INCONCLUSIVE",
            reason_text=(
                "No passages were retrieved from the local corpus. "
                "The claim cannot be assessed without evidence."
            ),
            limitations=(
                "Assessment is inconclusive because no relevant passages "
                "were found in the local corpus.",
            ),
            **_common,
        )

    try:
        caption_words = set(re.findall(r"\b\w+\b", caption_normalized.lower()))
        caption_words -= {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at"}

        components: list[ClaimComponent] = []
        accepted: list[CitationRecord] = []
        rejected: list[CitationRecord] = []

        support_count = 0
        contra_count  = 0
        unresolved_count = 0

        for rp in retrieved_passages:
            pas = rp.passage
            direction, confidence = _score_passage(pas.excerpt, caption_words)

            # A passage qualifies as a citation only when confidence is meaningful
            if confidence >= 0.35:
                cit = CitationRecord(
                    passage_id=pas.passage_id,
                    quote=pas.excerpt[:300] if len(pas.excerpt) > 300 else pas.excerpt,
                    source_name=pas.source_name,
                    published_at=pas.published_at,
                    url=pas.url,
                    accepted=True,
                )
                comp = ClaimComponent(
                    description=f"Evidence from {pas.source_name}",
                    status=direction,
                    citations=(cit,),
                )
                components.append(comp)
                accepted.append(cit)

                if direction == "supported":
                    support_count += 1
                elif direction == "contradicted":
                    contra_count += 1
                else:
                    unresolved_count += 1
            else:
                # Low-confidence passage becomes a rejected citation
                rej_cit = CitationRecord(
                    passage_id=pas.passage_id,
                    quote=pas.excerpt[:200] if len(pas.excerpt) > 200 else pas.excerpt,
                    source_name=pas.source_name,
                    published_at=pas.published_at,
                    url=pas.url,
                    accepted=False,
                    rejection_reason=(
                        "Insufficient overlap with the claim to be cited as evidence."
                    ),
                )
                rejected.append(rej_cit)

        # Determine overall outcome
        if support_count == 0 and contra_count == 0:
            outcome = "INCONCLUSIVE"
            reason_text = (
                "The retrieved passages do not contain sufficient evidence "
                "to support or contradict the claim."
            )
            uncertainties: tuple[str, ...] = (
                "All retrieved passages had low relevance to the claim.",
            )
            conflicts: tuple[str, ...] = ()
        elif support_count > 0 and contra_count > 0:
            outcome = "CONFLICTING"
            reason_text = (
                f"Retrieved evidence is mixed: "
                f"{support_count} passage(s) appear to support the claim, "
                f"{contra_count} passage(s) appear to contradict it."
            )
            conflicts = (
                "At least one retrieved passage supports and at least one contradicts the claim.",
            )
            uncertainties = ()
        elif contra_count > support_count:
            outcome = "CONTRADICTED"
            reason_text = (
                f"The majority of retrieved evidence ({contra_count} passage(s)) "
                "appears to contradict the claim."
            )
            conflicts = ()
            uncertainties = ()
        else:
            outcome = "SUPPORTED"
            reason_text = (
                f"The majority of retrieved evidence ({support_count} passage(s)) "
                "appears to support the claim."
            )
            conflicts = ()
            uncertainties = ()

        limitations: tuple[str, ...] = (
            "This assessment uses a keyword-heuristic, not a language model. "
            "It may misclassify nuanced or indirect evidence.",
            "The local corpus is small; absence of evidence does not mean "
            "the claim is false or true.",
        )

        return Assessment(
            status="OK",
            outcome=outcome,
            reason_text=reason_text,
            components=tuple(components),
            accepted_citations=tuple(accepted),
            rejected_citations=tuple(rejected),
            unresolved_items=tuple(
                f"Passage {rp.passage.passage_id} had low relevance (rank {rp.rank})"
                for rp in retrieved_passages
                if rp.passage.passage_id not in {c.passage_id for c in accepted}
            ),
            conflicts=conflicts,
            uncertainties=uncertainties,
            limitations=limitations,
            **_common,
        )

    except Exception as exc:
        return Assessment(
            status="FAILED",
            reason="An error occurred during claim assessment.",
            error_detail=str(exc),
            **_common,
        )
