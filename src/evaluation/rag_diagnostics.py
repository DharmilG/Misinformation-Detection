"""Per-stage instrumentation for the RAG ablation study.

This module wraps the existing RAG modules and records what each stage receives
and produces.  It never modifies them.  Without this trace, a RAG failure can
only be observed at the final label, which is not enough to answer *why* it
failed.

Non-OK assessment statuses (UNAVAILABLE, SKIPPED, FAILED) are retained as
first-class outcomes: a pipeline that silently skips is a possible cause of the
INCONCLUSIVE collapse recorded in reports/rag_non_rag_eval.json.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

try:
    from ..rag import rag_eligibility, rag_injection, rag_outcome
    from ..rag.assessment import run_assessment
    from ..retrieval.retrieval_api import retrieve
except ImportError:  # imported as a top-level module from a script
    from src.rag import rag_eligibility, rag_injection, rag_outcome
    from src.rag.assessment import run_assessment
    from src.retrieval.retrieval_api import retrieve


@dataclass(frozen=True)
class StageRecord:
    stage: str
    condition: str
    case_id: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    latency_ms: float
    error: str | None = None


@dataclass(frozen=True)
class CaseTrace:
    case_id: str
    condition: str
    gold_label: str
    predicted_label: str
    confidence: float
    stages: tuple[StageRecord, ...] = field(default_factory=tuple)


def _timed(
    stage: str,
    condition: str,
    case_id: str,
    inputs: dict[str, Any],
    call: Callable[[], Any],
    summarise: Callable[[Any], dict[str, Any]],
) -> tuple[Any, StageRecord]:
    """Run ``call``, always returning a record even if it raises."""
    started = time.perf_counter()
    try:
        result = call()
        record = StageRecord(
            stage=stage,
            condition=condition,
            case_id=case_id,
            inputs=inputs,
            outputs=summarise(result),
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )
        return result, record
    except Exception as exc:  # a failure is data, not a control-flow event
        return None, StageRecord(
            stage=stage,
            condition=condition,
            case_id=case_id,
            inputs=inputs,
            outputs={},
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            error=f"{type(exc).__name__}: {exc}",
        )


def trace_retrieval(
    case_id: str,
    condition: str,
    query: str,
    *,
    top_k: int = 5,
    index: Any = None,
    corpus_path: Any = None,
) -> tuple[Any, StageRecord]:
    def call():
        return retrieve(query, index=index, top_k=top_k, corpus_path=corpus_path)

    def summarise(result):
        passages = list(getattr(result, "passages", ()) or ())
        return {
            "status": getattr(result, "status", None),
            "reason": getattr(result, "reason", None),
            "passage_count": len(passages),
            "ids": [getattr(p, "chunk_id", None) for p in passages],
            "scores": [float(getattr(p, "score", 0.0)) for p in passages],
            # A stub retriever fabricates text from the query.  Recording this
            # makes failure mode F1 (no real evidence) directly measurable.
            "synthetic_passages": all(
                str(getattr(p, "text", "")).startswith("Relevant passage ")
                for p in passages
            )
            if passages
            else False,
        }

    return _timed(
        "retrieval",
        condition,
        case_id,
        {"query": query, "top_k": top_k, "corpus_path": str(corpus_path) if corpus_path else None},
        call,
        summarise,
    )


def _as_assessment_passages(passages: tuple[Any, ...]) -> tuple[Any, ...]:
    """Adapt flat retrieval passages to the nested schema run_assessment expects.

    ``retrieval_api.RetrievedPassage`` is flat (chunk_id/text/score), while
    ``rag.assessment.run_assessment`` expects ``reporting.model.RetrievedPassage``
    with ``.passage.excerpt``.  Without this adapter every assessment fails with
    "'RetrievedPassage' object has no attribute 'passage'" - a silent integration
    failure that looks like a missing generation backend.
    """
    try:
        from ..reporting.model import PassageRecord, RetrievedPassage as ReportPassage
    except ImportError:
        from src.reporting.model import PassageRecord, RetrievedPassage as ReportPassage

    adapted = []
    for index, passage in enumerate(passages, start=1):
        if hasattr(passage, "passage"):
            adapted.append(passage)
            continue
        excerpt = str(getattr(passage, "text", ""))
        record = PassageRecord(
            passage_id=str(getattr(passage, "chunk_id", f"chunk_{index}")),
            title=excerpt[:120],
            source_name="local corpus",
            source_type="dataset caption",
            published_at=None,
            excerpt=excerpt,
            url=None,
            rights_wording=None,
            storage_scope=None,
        )
        adapted.append(
            ReportPassage(
                passage=record,
                rank=index,
                total_retrieved=len(passages),
                relevance_score=float(getattr(passage, "score", 0.0)),
            )
        )
    return tuple(adapted)


def trace_assessment(
    case_id: str, condition: str, caption: str, passages: tuple[Any, ...]
) -> tuple[Any, StageRecord]:
    result, record = _timed(
        "assessment",
        condition,
        case_id,
        {"caption": caption, "passage_ids": [getattr(p, "chunk_id", None) for p in passages]},
        lambda: run_assessment(caption, _as_assessment_passages(passages)),
        lambda r: {
            "status": getattr(r, "status", None),
            "outcome": getattr(r, "outcome", None),
            "reason_text": getattr(r, "reason_text", None),
            # Non-OK statuses hide the real cause here.  Never drop these.
            "reason": getattr(r, "reason", None),
            "error_detail": getattr(r, "error_detail", None),
            "accepted_citation_ids": [
                getattr(c, "source_id", None) for c in (getattr(r, "accepted_citations", ()) or ())
            ],
            "rejected_citation_ids": [
                getattr(c, "source_id", None) for c in (getattr(r, "rejected_citations", ()) or ())
            ],
            "unresolved_items": list(getattr(r, "unresolved_items", ()) or ()),
            "conflicts": list(getattr(r, "conflicts", ()) or ()),
            "uncertainties": list(getattr(r, "uncertainties", ()) or ()),
            "component_count": len(getattr(r, "components", ()) or ()),
        },
    )
    return result, record


def trace_eligibility(
    case_id: str,
    condition: str,
    citations: list[dict[str, Any]],
    known_components: set[str],
    valid_passages: dict[str, str],
) -> tuple[Any, StageRecord]:
    return _timed(
        "eligibility",
        condition,
        case_id,
        {"citation_count": len(citations), "known_components": sorted(known_components)},
        lambda: rag_eligibility.process_citations(citations, known_components, valid_passages),
        lambda r: {
            "count": len(r) if isinstance(r, list) else None,
            "statuses": [
                item.get("status") if isinstance(item, dict) else None
                for item in (r if isinstance(r, list) else [])
            ],
        },
    )


def trace_injection(
    case_id: str,
    condition: str,
    payload: dict[str, Any],
    passages: dict[str, str],
) -> tuple[Any, StageRecord]:
    return _timed(
        "injection",
        condition,
        case_id,
        {"payload_keys": sorted(payload) if isinstance(payload, dict) else []},
        lambda: rag_injection.validate_citations_against_supplied(payload, passages),
        lambda r: {"result": r if isinstance(r, (bool, str, int, float)) else str(r)},
    )


def trace_outcome(
    case_id: str,
    condition: str,
    component_statuses: dict[str, str],
    has_citations: bool,
) -> tuple[Any, StageRecord]:
    return _timed(
        "outcome",
        condition,
        case_id,
        {"component_statuses": dict(component_statuses), "has_citations": has_citations},
        lambda: rag_outcome.compute_claim_outcome(component_statuses, has_citations),
        lambda r: {"outcome": r.get("outcome") if isinstance(r, dict) else None, "raw": str(r)},
    )


def to_row(trace: CaseTrace) -> dict[str, Any]:
    """Flatten a trace into one JSONL row."""
    row: dict[str, Any] = {
        "case_id": trace.case_id,
        "condition": trace.condition,
        "gold_label": trace.gold_label,
        "predicted_label": trace.predicted_label,
        "confidence": trace.confidence,
    }
    for record in trace.stages:
        prefix = record.stage
        row[f"{prefix}_inputs"] = record.inputs
        row[f"{prefix}_outputs"] = record.outputs
        row[f"{prefix}_latency_ms"] = record.latency_ms
        row[f"{prefix}_error"] = record.error
    return row


def stage_names(trace: CaseTrace) -> tuple[str, ...]:
    """Stages present in a trace, in order (useful for completeness checks)."""
    return tuple(record.stage for record in trace.stages)
