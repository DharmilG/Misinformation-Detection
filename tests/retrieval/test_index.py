"""The evidence index storage, safe loading, and invalidation rules of Document 06 §13 and §14.

Tests pickle-free storage (allow_pickle=False on .npy), index_meta.json schema,
and automatic index invalidation on version mismatches (chunking, cleaning, embedding).
"""

from pathlib import Path
import pytest

from tests.ids import case, known_gap

np = pytest.importorskip("numpy")

INDEX_FILES = {
    "embeddings.npy",
    "chunks.jsonl",
    "index_meta.json",
}


def _index_module():
    import importlib
    return importlib.import_module("retrieval_index")


@known_gap(
    "evidence_index.module_exists",
    "F2",
    "fast",
    "doc/06 §13",
    "The project provides an evidence index management and loading module.",
    "there is no retrieval_index module",
    requires=("numpy",),
)
def test_evidence_index_module_exists():
    _index_module()


@known_gap(
    "evidence_index.no_pickle_in_index",
    "F2",
    "fast",
    "doc/06 §13.2",
    "The index avoids Python pickle entirely; embeddings are loaded with allow_pickle=False.",
    "safe index loading is not implemented",
    requires=("numpy",),
)
def test_no_pickle_in_index(tmp_path):
    mod = _index_module()
    npy_path = tmp_path / "embeddings.npy"
    np.save(npy_path, np.zeros((2, 384), dtype=np.float32), allow_pickle=False)
    # mod.load_embeddings must use allow_pickle=False
    arr = mod.load_embeddings_safe(npy_path)
    assert tuple(arr.shape) == (2, 384)


@known_gap(
    "evidence_index.metadata_schema",
    "F2",
    "fast",
    "doc/06 §13.3",
    "index_meta.json records required versions, models, counts, and lexical analyzer parameters.",
    "index metadata schema is not implemented",
    requires=("numpy",),
)
def test_metadata_schema():
    mod = _index_module()
    meta = mod.generate_index_metadata()
    required_keys = {
        "index_schema_version",
        "corpus_manifest_sha256",
        "document_count",
        "chunk_count",
        "chunking_version",
        "cleaning_version",
        "url_canonicalization_version",
        "embedding_model_id",
        "embedding_dim",
        "lexical",
    }
    missing = required_keys - set(meta.keys())
    assert not missing, "index_meta.json missing keys: %s" % missing


@known_gap(
    "evidence_index.version_mismatch_invalidates",
    "F2",
    "fast",
    "doc/06 §13.4",
    "Mismatches in chunking, cleaning, or embedding versions invalidate the index, triggering rebuild.",
    "index version invalidation check is not implemented",
    requires=("numpy",),
)
def test_version_mismatch_invalidates():
    mod = _index_module()
    current_meta = {"chunking_version": 2, "cleaning_version": 1, "embedding_dim": 384}
    stale_meta = {"chunking_version": 1, "cleaning_version": 1, "embedding_dim": 384}
    assert mod.is_index_valid(stale_meta, current_meta) is False
    assert mod.is_index_valid(current_meta, current_meta) is True
