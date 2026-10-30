"""Evidence index storage, safe loading, and invalidation rules of Document 06 §13-§14.

Manages pickle-free array loading (allow_pickle=False), index_meta.json schema,
and automatic version invalidation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

INDEX_FILES = {
    "embeddings.npy",
    "chunks.jsonl",
    "index_meta.json",
}


def load_embeddings_safe(npy_path: str | Path) -> np.ndarray:
    """Load embeddings array safely with allow_pickle=False."""
    return np.load(npy_path, allow_pickle=False)


def generate_index_metadata(
    corpus_manifest_sha256: str = "0" * 64,
    document_count: int = 100,
    chunk_count: int = 500,
    chunking_version: int = 1,
    cleaning_version: int = 1,
    url_canonicalization_version: int = 1,
    embedding_model_id: str = "all-MiniLM-L6-v2",
    embedding_dim: int = 384,
    lexical_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate index metadata conforming to Document 06 §13.3 schema."""
    return {
        "index_schema_version": 1,
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "document_count": document_count,
        "chunk_count": chunk_count,
        "chunking_version": chunking_version,
        "cleaning_version": cleaning_version,
        "url_canonicalization_version": url_canonicalization_version,
        "embedding_model_id": embedding_model_id,
        "embedding_dim": embedding_dim,
        "lexical": lexical_params or {"k1": 1.5, "b": 0.75, "analyzer": "simple_word"},
    }


def is_index_valid(stored_meta: dict[str, Any], current_meta: dict[str, Any]) -> bool:
    """Verify stored index versions match current configuration, invalidating on divergence."""
    checked_keys = ("chunking_version", "cleaning_version", "embedding_dim")
    for k in checked_keys:
        if stored_meta.get(k) != current_meta.get(k):
            return False
    return True
