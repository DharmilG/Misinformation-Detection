"""
src/retrieval/corpus.py
=======================

TF-IDF retrieval over a local corpus (Document 06 §10, §12).

Behaviour:
  - Loads a JSON list of passage records on first use (process-level cache).
  - Builds a TF-IDF index with scikit-learn.
  - Returns top-K passages ranked by cosine similarity.
  - A missing or unreadable corpus returns UNAVAILABLE with a reason.
  - Never raises to the caller.

``corpus_path`` selects a specific corpus file.  The default is
``src.utils.config.CORPUS_PATH``.  Research runs can pass a closed-world corpus
built from real dataset captions without changing the shipped corpus.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
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


# ── Index construction ───────────────────────────────────────────────────

def _build_index(records: list[dict]) -> tuple[object, object, Optional[str]]:
    """Return (vectorizer, matrix, error) for a list of passage records."""
    if not _SKLEARN_AVAILABLE:
        return None, None, "scikit-learn is not installed. Install it with: pip install scikit-learn"
    if not records:
        return None, None, None
    try:
        texts = [
            f"{r.get('title', '')} {r.get('source_name', '')} {r.get('excerpt', '')}".strip()
            for r in records
        ]
        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            stop_words="english",
        )
        matrix = vectorizer.fit_transform(texts)
        return vectorizer, matrix, None
    except Exception as exc:
        return None, None, f"TF-IDF index construction failed: {exc}"


def _load_from_file(path: Path) -> dict:
    entry: dict = {"records": [], "vectorizer": None, "matrix": None, "error": None}
    if not path.exists():
        entry["error"] = (
            f"Corpus file not found at {path}. Create data/corpus.json to enable evidence retrieval."
        )
        return entry
    try:
        with path.open("r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        if not isinstance(loaded, list):
            entry["error"] = "Corpus file must contain a JSON list of passage records."
            return entry
        entry["records"] = loaded
    except Exception as exc:
        entry["error"] = f"Could not load corpus: {exc}"
        return entry
    vectorizer, matrix, error = _build_index(entry["records"])
    entry.update(vectorizer=vectorizer, matrix=matrix, error=error)
    return entry


# ── Process-level caches ─────────────────────────────────────────────────

_default_index: Optional[dict] = None
_custom_indexes: dict[str, dict] = {}

# Kept for callers that inspected the original module globals.
_corpus_records: list[dict] = []
_vectorizer = None
_tfidf_matrix = None
_corpus_loaded = False
_corpus_load_error: Optional[str] = None


def _get_index(corpus_path: Optional[str | Path]) -> dict:
    global _default_index, _corpus_records, _vectorizer, _tfidf_matrix
    global _corpus_loaded, _corpus_load_error

    if corpus_path is None:
        if _default_index is None:
            _default_index = _load_from_file(Path(CORPUS_PATH))
            _corpus_records = _default_index["records"]
            _vectorizer = _default_index["vectorizer"]
            _tfidf_matrix = _default_index["matrix"]
            _corpus_load_error = _default_index["error"]
            _corpus_loaded = True
        return _default_index

    key = str(corpus_path)
    if key not in _custom_indexes:
        _custom_indexes[key] = _load_from_file(Path(corpus_path))
    return _custom_indexes[key]


def _make_passage_record(raw: dict) -> PassageRecord:
    return PassageRecord(
        passage_id=raw.get("passage_id", str(uuid.uuid4())),
        title=raw.get("title", "Untitled"),
        source_name=raw.get("source_name", "Unknown source"),
        source_type=raw.get("source_type", "unknown"),
        published_at=raw.get("published_at"),
        excerpt=raw.get("excerpt", ""),
        url=raw.get("url"),
        rights_wording=raw.get("rights_wording"),
        storage_scope=raw.get("storage_scope"),
    )


def _rank(
    query: str,
    top_k: int,
    records: list[dict],
    vectorizer: object,
    matrix: object,
) -> tuple[RetrievedPassage, ...]:
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, matrix).flatten()
    top_indices = scores.argsort()[::-1][:top_k]
    top_indices = [i for i in top_indices if scores[i] > 0.0]

    retrieved: list[RetrievedPassage] = []
    for rank, idx in enumerate(top_indices, start=1):
        raw = records[idx]
        record = _make_passage_record(raw)
        retrieved.append(
            RetrievedPassage(
                passage=record,
                rank=rank,
                total_retrieved=len(top_indices),
                relevance_score=round(float(scores[idx]), 4),
            )
        )
    return tuple(retrieved)


# ── Public API ────────────────────────────────────────────────────────────

def retrieve_passages(
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
    corpus_path: Optional[str | Path] = None,
) -> RetrievalResult:
    """Retrieve top-k passages from a local corpus. Never raises."""
    index = _get_index(corpus_path)
    records = index["records"]

    _common = dict(
        corpus_version=CORPUS_VERSION,
        corpus_manifest_sha256=CORPUS_MANIFEST_SHA256,
        total_documents=len(records),
        total_passages=len(records),
        retrieval_config_hash=RETRIEVAL_CONFIG_HASH,
        embedding_revision=EMBEDDING_REVISION,
        chunking_version=CHUNKING_VERSION,
    )

    if index["error"]:
        return RetrievalResult(
            status="UNAVAILABLE", passages=(), reason=index["error"], **_common
        )
    if not records:
        return RetrievalResult(status="OK", passages=(), **_common)
    if index["vectorizer"] is None or index["matrix"] is None:
        return RetrievalResult(
            status="UNAVAILABLE", passages=(), reason="TF-IDF index is not available.", **_common
        )

    try:
        passages = _rank(query, top_k, records, index["vectorizer"], index["matrix"])
        return RetrievalResult(status="OK", passages=passages, **_common)
    except Exception as exc:
        return RetrievalResult(
            status="FAILED", passages=(), reason=f"Retrieval failed: {exc}", **_common
        )
