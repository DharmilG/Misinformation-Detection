"""Lexical BM25 retriever and deterministic analyzer of Document 06 §14.

Implements simple word analyzer (keeps stopwords, no stemming, tokens >= 2 chars),
BM25 scoring formula (k1=1.5, b=0.75, +1.0 IDF smoothing), and rebuild equivalence.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any


def analyze_text(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric characters, and keep tokens >= 2 chars."""
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if len(t) >= 2]


def calculate_bm25_term_score(
    tf: int,
    doc_len: int,
    avgdl: float,
    df: int,
    total_docs: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """Calculate BM25 term score with Lucene-style +1.0 IDF smoothing."""
    idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
    len_norm = 1.0 - b + b * (doc_len / max(avgdl, 1e-6))
    term_freq_comp = (tf * (k1 + 1.0)) / (tf + k1 * len_norm)
    return float(idf * term_freq_comp)


def rank_candidates_in_memory(
    corpus: list[dict[str, Any]], query: str
) -> list[tuple[str, float]]:
    """Score and rank corpus chunks against query terms using BM25."""
    query_tokens = analyze_text(query)
    total_docs = len(corpus)
    if total_docs == 0:
        return []

    doc_tokens: list[list[str]] = []
    total_len = 0
    for doc in corpus:
        toks = analyze_text(doc.get("text", ""))
        doc_tokens.append(toks)
        total_len += len(toks)

    avgdl = total_len / float(total_docs) if total_docs > 0 else 1.0

    # Calculate document frequency (df) for query tokens
    dfs: dict[str, int] = {}
    for t in query_tokens:
        dfs[t] = sum(1 for d in doc_tokens if t in d)

    scored: list[tuple[str, float]] = []
    for i, doc in enumerate(corpus):
        c_id = doc.get("chunk_id", f"c_{i}")
        d_toks = doc_tokens[i]
        d_len = len(d_toks)
        doc_score = 0.0

        for t in query_tokens:
            tf = d_toks.count(t)
            if tf > 0:
                doc_score += calculate_bm25_term_score(
                    tf=tf,
                    doc_len=d_len,
                    avgdl=avgdl,
                    df=dfs[t],
                    total_docs=total_docs,
                )
        scored.append((c_id, doc_score))

    return sorted(scored, key=lambda x: (-x[1], x[0]))


def save_index(corpus: list[dict[str, Any]], target_dir: str | Path) -> None:
    """Save lexical corpus index to disk."""
    p = Path(target_dir)
    p.mkdir(parents=True, exist_ok=True)
    (p / "lexical_index.json").write_text(json.dumps(corpus), encoding="utf-8")


def rank_candidates_from_disk(
    target_dir: str | Path, query: str
) -> list[tuple[str, float]]:
    """Load persisted lexical index and compute rankings matching in-memory output."""
    p = Path(target_dir) / "lexical_index.json"
    corpus = json.loads(p.read_text(encoding="utf-8"))
    return rank_candidates_in_memory(corpus, query)
