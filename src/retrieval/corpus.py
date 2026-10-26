"""
src/retrieval/corpus.py
========================
TF-IDF retrieval over the local corpus.

Spec: 05_REPORTING_AND_UI §6, 04_INPUT_AND_INFERENCE_PIPELINE.

Behaviour:
  - Loads data/corpus.json on first call (process-level singleton).
  - Builds a TF-IDF index with scikit-learn.
  - Returns top-K passages ranked by cosine similarity.
  - If the corpus file is missing, returns UNAVAILABLE gracefully.
  - Never raises to the caller.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Optional

from src.reporting.model import PassageRecord, RetrievedPassage
from src.utils.config import (
    CHUNKING_VERSION,
    CORPUS_MANIFEST_SHA256,
    CORPUS_PATH,
    CORPUS_VERSION,
    EMBEDDING_REVISION,
    RETRIEVAL_CONFIG_HASH,
    RETRIEVAL_TOP_K,
)

try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity        # type: ignore
    import numpy as np                                            # type: ignore
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# ── Result dataclass ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class RetrievalResult:
    status: str                              # "OK" | "UNAVAILABLE" | "FAILED"
    passages: tuple[RetrievedPassage, ...]
    corpus_version: str
    corpus_manifest_sha256: str
    total_documents: int
    total_passages: int
    retrieval_config_hash: str
    embedding_revision: str
    chunking_version: str
    reason: Optional[str] = None


# ── Process-level singleton ───────────────────────────────────────────────

_corpus_records: list[dict] = []
_vectorizer: Optional["TfidfVectorizer"] = None
_tfidf_matrix = None
_corpus_loaded: bool = False
_corpus_load_error: Optional[str] = None


def _load_corpus() -> None:
    global _corpus_records, _vectorizer, _tfidf_matrix
    global _corpus_loaded, _corpus_load_error

    if _corpus_loaded:
        return

    _corpus_loaded = True

    if not CORPUS_PATH.exists():
        _corpus_load_error = (
            f"Corpus file not found at {CORPUS_PATH}. "
            "Create data/corpus.json to enable evidence retrieval."
        )
        return

    try:
        with CORPUS_PATH.open("r", encoding="utf-8") as fh:
            _corpus_records = json.load(fh)
    except Exception as exc:
        _corpus_load_error = f"Could not load corpus: {exc}"
        return

    if not _SKLEARN_AVAILABLE:
        _corpus_load_error = (
            "scikit-learn is not installed. "
            "Install it with: pip install scikit-learn"
        )
        return

    if not _corpus_records:
        # Empty corpus is valid; retrieval will return no passages.
        return

    try:
        texts = [
            f"{r.get('title', '')} {r.get('source_name', '')} {r.get('excerpt', '')}".strip()
            for r in _corpus_records
        ]
        _vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            stop_words="english",
        )
        _tfidf_matrix = _vectorizer.fit_transform(texts)
    except Exception as exc:
        _corpus_load_error = f"TF-IDF index construction failed: {exc}"


def _make_passage_record(raw: dict) -> PassageRecord:
    return PassageRecord(
        passage_id=raw.get("passage_id", str(uuid.uuid4())),
        title=raw.get("title", "Untitled"),
        source_name=raw.get("source_name", "Unknown source"),
        source_type=raw.get("source_type", "unknown"),
        published_at=raw.get("published_at"),          # None is valid
        excerpt=raw.get("excerpt", ""),
        url=raw.get("url"),                            # None is valid
        rights_wording=raw.get("rights_wording"),
        storage_scope=raw.get("storage_scope"),
    )


# ── Public API ────────────────────────────────────────────────────────────

def retrieve_passages(
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
) -> RetrievalResult:
    """
    Retrieve the top-k passages from the local corpus for the given query.

    Returns RetrievalResult with status OK, UNAVAILABLE, or FAILED.
    Never raises.
    """
    _load_corpus()

    _common = dict(
        corpus_version=CORPUS_VERSION,
        corpus_manifest_sha256=CORPUS_MANIFEST_SHA256,
        total_documents=len(_corpus_records),
        total_passages=len(_corpus_records),
        retrieval_config_hash=RETRIEVAL_CONFIG_HASH,
        embedding_revision=EMBEDDING_REVISION,
        chunking_version=CHUNKING_VERSION,
    )

    if _corpus_load_error:
        return RetrievalResult(
            status="UNAVAILABLE",
            passages=(),
            reason=_corpus_load_error,
            **_common,
        )

    if not _corpus_records:
        return RetrievalResult(
            status="OK",
            passages=(),
            **_common,
        )

    if not _SKLEARN_AVAILABLE or _vectorizer is None or _tfidf_matrix is None:
        return RetrievalResult(
            status="UNAVAILABLE",
            passages=(),
            reason="TF-IDF index is not available.",
            **_common,
        )

    try:
        query_vec = _vectorizer.transform([query])
        scores = cosine_similarity(query_vec, _tfidf_matrix).flatten()

        # Rank descending; take top_k
        top_indices = scores.argsort()[::-1][:top_k]
        top_indices = [i for i in top_indices if scores[i] > 0.0]

        retrieved: list[RetrievedPassage] = []
        for rank, idx in enumerate(top_indices, start=1):
            raw = _corpus_records[idx]
            record = _make_passage_record(raw)
            rp = RetrievedPassage(
                passage=record,
                rank=rank,
                total_retrieved=len(top_indices),
                relevance_score=round(float(scores[idx]), 4),
            )
            retrieved.append(rp)

        return RetrievalResult(
            status="OK",
            passages=tuple(retrieved),
            **_common,
        )

    except Exception as exc:
        return RetrievalResult(
            status="FAILED",
            passages=(),
            reason=f"Retrieval failed: {exc}",
            **_common,
        )
