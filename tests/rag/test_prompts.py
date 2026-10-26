"""The prompt architecture, versioning, and template contracts of Document 06 §32.

Tests system prompt construction, PROMPT_VERSION tracking, classifier context inclusion
with the model interpretation string, and strict evidence data delimiters.
"""

import pytest

from tests.ids import case, known_gap


def _prompts_module():
    import importlib
    return importlib.import_module("rag_prompts")


@known_gap(
    "rag_prompts.module_exists",
    "F2",
    "fast",
    "doc/06 §32",
    "The project provides a prompt construction and template module.",
    "there is no rag_prompts module",
)
def test_prompts_module_exists():
    _prompts_module()


@known_gap(
    "rag_prompts.version_pinned",
    "F2",
    "fast",
    "doc/06 §32",
    "PROMPT_VERSION is recorded and versioned with prompt wording changes.",
    "PROMPT_VERSION is not defined",
)
def test_version_pinned():
    mod = _prompts_module()
    version = getattr(mod, "PROMPT_VERSION", None)
    assert version is not None
    assert isinstance(version, (str, int))


@known_gap(
    "rag_prompts.classifier_context_block",
    "F2",
    "fast",
    "doc/06 §32",
    "Classifier context block embeds the copy.json interpretation string and explicitly warns against treating it as proof.",
    "classifier context prompt template is not implemented",
)
def test_classifier_context_block():
    mod = _prompts_module()
    block = mod.format_classifier_context_block(
        label="MISMATCHED",
        score=0.78,
        interpretation="This score estimates whether the image and caption look mismatched.",
    )
    assert "MISMATCHED" in block
    assert "0.78" in block
    assert "not evidence about whether the claim is true" in block.lower() or "not proof" in block.lower()


@known_gap(
    "rag_prompts.evidence_delimiter",
    "F2",
    "fast",
    "doc/06 §32, §33.1",
    "Retrieved evidence passages are enclosed in explicit data boundaries, never treated as instructions.",
    "evidence prompt delimiting is not implemented",
)
def test_evidence_delimiter():
    mod = _prompts_module()
    passages = [{"chunk_id": "c1", "text": "President arrived in Berlin on Monday."}]
    rendered = mod.format_evidence_passages(passages)
    assert "c1" in rendered
    # Must use explicit boundary markers
    assert "<evidence>" in rendered or "BEGIN EVIDENCE" in rendered
