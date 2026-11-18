"""Top-level retriever API and integration contract of Document 06 §18, §19, §23.

Provides retrieve() strictly decoupled from classifier signals, with a
structured RetrievalResult schema and explicit unavailability handling.

The default path uses the real local TF-IDF corpus in ``corpus.py``.  It never
fabricates passage text.  ``retrieve_with_mock_index`` remains a conformance
helper and is the only source of synthetic passages in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    from src.retrieval import corpus as _corpus
    from src.retrieval.retrieval_lexical import rank_candidates_in_memory
except ImportError:  # module imported with src/retrieval on sys.path
    import corpus as _corpus  # type: ignore[no-redef]
    from retrieval_lexical import rank_candidates_in_memory  # type: ignore[no-redef]


@dataclass(frozen=True)
class RetrievedPassage:
    chunk_id: str
    document_id: str
    text: str
    score: float
    dense_score: float = 0.0
    lexical_score: float = 0.0


@dataclass(frozen=True)
class RetrievalResult:
    status: str
    query: str
    passages: list[RetrievedPassage] = field(default_factory=list)
    reason: str | None = None


def _from_corpus(query: str, top_k: int) -> RetrievalResult:
    """Delegate to the real TF-IDF retriever and adapt its result schema."""
    result = _corpus.retrieve_passages(query, top_k=top_k)
    passages = [
        RetrievedPassage(
            chunk_id=str(item.passage.passage_id),
            document_id=str(item.passage.passage_id),
            text=str(item.passage.excerpt),
            score=float(item.relevance_score),
            dense_score=0.0,  # no dense model is in use
            lexical_score=float(item.relevance_score),
        )
        for item in result.passages
    ]
    return RetrievalResult(
        status=result.status,
        query=query,
        passages=passages,
        reason=result.reason,
    )


def _from_index(index: Any, query: str, top_k: int) -> RetrievalResult:
    """Rank a caller-supplied in-memory corpus with BM25, without synthetic text."""
    if not isinstance(index, list) or not index:
        return RetrievalResult(status="OK", query=query, passages=[])
    ranked = rank_candidates_in_memory(index, query)[:top_k]
    by_id = {str(d.get("chunk_id", f"c{i}")): d for i, d in enumerate(index)}
    passages = []
    for chunk_id, score in ranked:
        if score <= 0.0:
            continue
        doc = by_id.get(chunk_id, {})
        passages.append(
            RetrievedPassage(
                chunk_id=chunk_id,
                document_id=str(doc.get("document_id", chunk_id)),
                text=str(doc.get("text", "")),
                score=float(score),
                dense_score=0.0,
                lexical_score=float(score),
            )
        )
    return RetrievalResult(status="OK", query=query, passages=passages)


def retrieve(
    query: str,
    index: Any = None,
    top_k: int = 5,
    tau_dense: float = 0.50,
    alpha_relative: float = 0.50,
) -> RetrievalResult:
    """Retrieve top-k real evidence passages for a query.

    ``index=None`` uses the local TF-IDF corpus.  A supplied list of chunk
    dicts is ranked with in-memory BM25.  Classifier signals are never inputs.
    """
    if not query.strip():
        return RetrievalResult(status="OK", query=query, passages=[])
    if index is None:
        return _from_corpus(query, top_k)
    return _from_index(index, query, top_k)


def retrieve_with_mock_index(
    query: str, empty_index: bool = False
) -> RetrievalResult:
    """Mock execution helper for conformance testing."""
    if empty_index:
        return RetrievalResult(status="OK", query=query, passages=[])

    passages = [
        RetrievedPassage(
            chunk_id="chunk_1",
            document_id="doc_1",
            text="Verified factual reporting on the subject.",
            score=0.88,
            dense_score=0.82,
            lexical_score=0.79,
        )
    ]
    return RetrievalResult(status="OK", query=query, passages=passages)
