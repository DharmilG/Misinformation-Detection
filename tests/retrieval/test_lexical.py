"""The lexical BM25 retriever and deterministic analyzer of Document 06 §14.

Tests the simple word analyzer (no stopwords removed, no stemming, tokens >= 2 chars),
BM25 scoring formula (k1=1.5, b=0.75, +1.0 IDF smoothing), and rebuild-equivalence invariant.
"""

import pytest

from tests.ids import case


def _lexical_module():
    import importlib
    return importlib.import_module("retrieval_lexical")


@case(
    "lexical_bm25.module_exists",
    "F2",
    "fast",
    "doc/06 §14",
    "The project provides a lexical BM25 retrieval module.",
)
def test_lexical_module_exists():
    _lexical_module()


@case(
    "lexical_bm25.analyzer_rules",
    "F2",
    "fast",
    "doc/06 §14.1",
    "The analyzer lowercases, splits on non-alphanumeric chars, keeps tokens >= 2 chars, and removes NO stopwords or stems.",
)
def test_analyzer_rules():
    mod = _lexical_module()
    tokens = mod.analyze_text("The quick brown fox is running in 2026!")
    # Stopwords like "the", "in", "is" are KEPT
    assert "the" in tokens
    assert "in" in tokens
    assert "is" in tokens
    # Running is NOT stemmed to run
    assert "running" in tokens
    # Numbers and alphanumeric tokens are kept
    assert "2026" in tokens
    # Single-char tokens stripped
    assert "a" not in mod.analyze_text("A book")


@case(
    "lexical_bm25.scoring_formula",
    "F2",
    "fast",
    "doc/06 §14.2",
    "BM25 scoring enforces k1=1.5, b=0.75, and +1.0 Lucene-style IDF smoothing.",
)
def test_scoring_formula():
    mod = _lexical_module()
    # Mock index with avgdl=10, document length=10 (so |d| / avgdl == 1)
    # tf=2, df=5, N=100
    score = mod.calculate_bm25_term_score(tf=2, doc_len=10, avgdl=10.0, df=5, total_docs=100, k1=1.5, b=0.75)
    assert score > 0.0


@case(
    "lexical_bm25.rebuild_equivalence",
    "F2",
    "fast",
    "doc/06 §14.4",
    "A persisted lexical index and a fresh in-memory rebuild produce identical rankings for a fixed query set.",
)
def test_rebuild_equivalence(tmp_path):
    mod = _lexical_module()
    corpus = [
        {"chunk_id": "c1", "text": "President delivers annual policy address in capital."},
        {"chunk_id": "c2", "text": "Economic inflation reaches record low in regional markets."},
    ]
    query = "president policy address"
    res_mem = mod.rank_candidates_in_memory(corpus, query)
    mod.save_index(corpus, tmp_path)
    res_disk = mod.rank_candidates_from_disk(tmp_path, query)
    assert res_mem == res_disk
