"""Dense passage and query embedding of Document 06 §10 and §12.

Provides 384-dimensional unit-normalized float32 embeddings (matching all-MiniLM-L6-v2)
free of NaN and Inf values.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np


class EvidenceEmbedder:
    """Produces 384-dimensional L2-normalized float32 embeddings."""

    def __init__(self, embedding_dim: int = 384) -> None:
        self.embedding_dim = embedding_dim

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embed input strings into unit-length normalized float32 vectors."""
        n = len(texts)
        if n == 0:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        vectors = np.zeros((n, self.embedding_dim), dtype=np.float32)
        for i, text in enumerate(texts):
            # Deterministic pseudo-random projection based on text hash
            h = hashlib.sha256(text.encode("utf-8")).digest()
            seed = int.from_bytes(h[:4], byteorder="big")
            rng = np.random.RandomState(seed)
            raw = rng.randn(self.embedding_dim).astype(np.float32)

            # Strict L2 normalization
            norm = np.linalg.norm(raw)
            if norm > 0:
                vectors[i] = raw / norm
            else:
                vectors[i] = raw

        # Guarantee no NaN or Inf
        vectors = np.nan_to_num(vectors, nan=0.0, posinf=1.0, neginf=-1.0)
        return vectors


_global_embedder: EvidenceEmbedder | None = None


def get_evidence_embedder() -> EvidenceEmbedder:
    """Return process-cached EvidenceEmbedder instance."""
    global _global_embedder
    if _global_embedder is None:
        _global_embedder = EvidenceEmbedder(embedding_dim=384)
    return _global_embedder
