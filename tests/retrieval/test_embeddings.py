"""The dense passage and query embedding contract of Document 06 §10 and §12.

Tests the EvidenceEmbedder contract, 384-d float32 embeddings,
unit-length L2 normalization, and absence of NaN or Inf values.
"""

import pytest

from tests.ids import case, known_gap

np = pytest.importorskip("numpy")


def _embeddings_module():
    import importlib
    return importlib.import_module("retrieval_embeddings")


@known_gap(
    "evidence_embed.module_exists",
    "F2",
    "fast",
    "doc/06 §12",
    "The project provides an EvidenceEmbedder module for dense passage and query encoding.",
    "there is no retrieval_embeddings module",
    requires=("numpy",),
)
def test_evidence_embed_module_exists():
    _embeddings_module()


@known_gap(
    "evidence_embed.dimension_384",
    "F2",
    "fast",
    "doc/06 §12",
    "EvidenceEmbedder produces 384-dimensional float32 embeddings matching all-MiniLM-L6-v2.",
    "dense embedder architecture is not implemented",
    requires=("numpy",),
)
def test_dimension_384():
    mod = _embeddings_module()
    embedder = mod.get_evidence_embedder()
    vecs = embedder.embed_texts(["A sample passage for embedding."])
    assert tuple(vecs.shape) == (1, 384)
    assert vecs.dtype == np.float32


@known_gap(
    "evidence_embed.unit_length_normalized",
    "F2",
    "fast",
    "doc/06 §12",
    "Embedded vectors are strictly L2 unit-length normalized within 1e-6 tolerance.",
    "embedding normalization is not implemented",
    requires=("numpy",),
)
def test_unit_length_normalized():
    mod = _embeddings_module()
    embedder = mod.get_evidence_embedder()
    vecs = embedder.embed_texts(["Text A", "Text B"])
    norms = np.linalg.norm(vecs, axis=-1)
    for norm in norms:
        assert abs(norm - 1.0) < 1e-6


@known_gap(
    "evidence_embed.no_nan_or_inf",
    "F2",
    "fast",
    "doc/06 §12",
    "Embedded representations contain no NaN or Inf values across all dimensions.",
    "embedding finiteness checks are not implemented",
    requires=("numpy",),
)
def test_no_nan_or_inf():
    mod = _embeddings_module()
    embedder = mod.get_evidence_embedder()
    vecs = embedder.embed_texts(["Passage one", "Passage two"])
    assert not np.isnan(vecs).any()
    assert not np.isinf(vecs).any()
