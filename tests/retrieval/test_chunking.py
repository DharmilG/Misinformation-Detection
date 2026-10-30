"""The sentence-aware sliding window chunking and text representations of Document 06 §10 and §11.

Tests ~200-token body targets with ~50-token overlap, no mid-sentence splits,
heading path tracking, and the generation of three distinct representations (text, lexical_text, dense_text).
"""

import pytest

from tests.ids import case


def _chunking_module():
    import importlib
    return importlib.import_module("retrieval_chunking")


@case(
    "chunking.module_exists",
    "F2",
    "fast",
    "doc/06 §11",
    "The project provides a sentence-aware text chunking module.",
)
def test_chunking_module_exists():
    _chunking_module()


@case(
    "chunking.sliding_window_sentences",
    "F2",
    "fast",
    "doc/06 §11.1",
    "Chunking applies sentence-aware sliding windows with ~200 token targets and never splits mid-sentence.",
)
def test_sliding_window_sentences():
    mod = _chunking_module()
    sentences = [
        "First sentence establishes the context.",
        "Second sentence adds verifiable details.",
        "Third sentence introduces a critical fact.",
        "Fourth sentence provides historical background.",
        "Fifth sentence concludes the section.",
    ]
    doc_text = " ".join(sentences)
    chunks = mod.chunk_text(doc_text, target_tokens=20, overlap_tokens=5)
    assert len(chunks) > 1
    # Verify every chunk starts and ends with whole sentences (no dangling split words)
    for c in chunks:
        assert any(c.text.startswith(s[:10]) for s in sentences)
        assert c.text.endswith((".", "!", "?"))


@case(
    "chunking.heading_path_tracked",
    "F2",
    "fast",
    "doc/06 §11.1, §11.2",
    "Chunks retain heading_path metadata, which is prepended into dense_text.",
)
def test_heading_path_tracked():
    mod = _chunking_module()
    heading = "2023 Election > Official Audit"
    body = "The recount matched original ballot tallies."
    chunk = mod.create_chunk(document_id="doc1", chunk_index=0, text=body, heading_path=heading)
    assert chunk.heading_path == heading
    assert heading in chunk.dense_text
    assert body in chunk.dense_text


@case(
    "chunking.three_representations",
    "F2",
    "fast",
    "doc/06 §10.3",
    "Every chunk exposes three distinct representations: text, lexical_text, and dense_text.",
)
def test_three_representations():
    mod = _chunking_module()
    chunk = mod.create_chunk(
        document_id="doc1",
        chunk_index=0,
        text="Case-Sensitive Title: 42 Items!",
        heading_path="Section 1",
    )
    assert chunk.text == "Case-Sensitive Title: 42 Items!"
    # lexical_text is lowercased / tokenized
    assert chunk.lexical_text.lower() == chunk.lexical_text
    # dense_text includes heading prefix
    assert "Section 1" in chunk.dense_text


@case(
    "chunking.token_budget_context_fit",
    "F2",
    "fast",
    "doc/06 §11.2",
    "Chunk token budget accounts for special tokens and heading reserve, setting truncated flag on overflow.",
)
def test_token_budget_context_fit():
    mod = _chunking_module()
    # Chunk fitting within budget
    c_normal = mod.create_chunk_with_budget(text="Short body text.", heading="H1", max_tokens=256)
    assert c_normal.truncated is False

    # Chunk overflowing context limit
    huge_text = "word " * 300
    c_overflow = mod.create_chunk_with_budget(text=huge_text, heading="H1", max_tokens=256)
    assert c_overflow.truncated is True
