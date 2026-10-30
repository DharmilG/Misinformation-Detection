"""Sentence-aware sliding window chunking and text representations of Document 06 §10-§11.

Generates 3 representations (text, lexical_text, dense_text) and tracks heading paths
without splitting mid-sentence.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True)
class Chunk:
    document_id: str
    chunk_index: int
    text: str
    heading_path: str
    lexical_text: str
    dense_text: str
    truncated: bool = False


def create_chunk(
    document_id: str,
    chunk_index: int,
    text: str,
    heading_path: str = "",
    truncated: bool = False,
) -> Chunk:
    """Create a chunk exposing text, lexical_text, and dense_text."""
    # lexical text is lowercase tokenized representation
    lexical = text.lower()
    # dense text prepends heading path to ground context
    if heading_path:
        dense = f"{heading_path} > {text}"
    else:
        dense = text

    return Chunk(
        document_id=document_id,
        chunk_index=chunk_index,
        text=text,
        heading_path=heading_path,
        lexical_text=lexical,
        dense_text=dense,
        truncated=truncated,
    )


def chunk_text(
    doc_text: str,
    target_tokens: int = 20,
    overlap_tokens: int = 5,
    document_id: str = "doc_default",
    heading_path: str = "",
) -> list[Chunk]:
    """Split text into sentence-aware sliding windows."""
    # Split text into sentences using punctuation boundaries
    sentences = re.split(r"(?<=[.!?])\s+", doc_text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    chunks: list[Chunk] = []
    i = 0
    chunk_idx = 0

    while i < len(sentences):
        current_sentences = []
        token_count = 0

        j = i
        while j < len(sentences):
            sent = sentences[j]
            sent_tokens = len(sent.split())
            if current_sentences and (token_count + sent_tokens > target_tokens):
                break
            current_sentences.append(sent)
            token_count += sent_tokens
            j += 1

        if not current_sentences:
            current_sentences.append(sentences[i])
            j = i + 1

        chunk_text_str = " ".join(current_sentences)
        chunks.append(
            create_chunk(
                document_id=document_id,
                chunk_index=chunk_idx,
                text=chunk_text_str,
                heading_path=heading_path,
            )
        )
        chunk_idx += 1

        if j >= len(sentences):
            break

        # Advance with overlap
        # calculate step to maintain overlap
        overlap_count = 0
        step = 1
        for k in range(len(current_sentences) - 1, -1, -1):
            overlap_count += len(current_sentences[k].split())
            if overlap_count >= overlap_tokens:
                step = max(1, k)
                break
        i += step

    return chunks


def create_chunk_with_budget(
    text: str,
    heading: str = "",
    max_tokens: int = 256,
    document_id: str = "doc1",
    chunk_index: int = 0,
) -> Chunk:
    """Check token budget including heading and set truncated flag on overflow."""
    tokens = len(text.split()) + (len(heading.split()) if heading else 0)
    truncated = tokens > max_tokens
    return create_chunk(
        document_id=document_id,
        chunk_index=chunk_index,
        text=text,
        heading_path=heading,
        truncated=truncated,
    )
