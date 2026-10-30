"""The corpus acquisition, terms, and rights enforcement of Document 06 §6.

Tests mandatory terms record precondition, refusal on absent terms, automated fetch
permissions check, and permitted storage scopes (FULL_PAGE, EXCERPT, METADATA_ONLY).
"""

import pytest

from tests.ids import case

TERMS_FIELDS = {
    "source_name",
    "source_identity",
    "terms_url",
    "license_or_terms_summary",
    "accessed_at",
    "permitted_use",
    "automated_fetch_allowed",
    "full_text_storage_allowed",
    "redistribution_allowed",
    "notes",
}


def _corpus_module():
    import importlib
    return importlib.import_module("retrieval_corpus")


@case(
    "corpus_terms.module_exists",
    "F2",
    "fast",
    "doc/06 §6",
    "The project provides a corpus terms and rights management module.",
)
def test_corpus_terms_module_exists():
    _corpus_module()


@case(
    "corpus_terms.terms_record_schema",
    "F2",
    "fast",
    "doc/06 §6.1",
    "Terms records define all required metadata, rights, and permission fields.",
)
def test_terms_record_schema():
    mod = _corpus_module()
    sample_terms = mod.create_sample_terms_record()
    missing = TERMS_FIELDS - set(sample_terms.keys())
    assert not missing, "terms record missing fields: %s" % missing


@case(
    "corpus_terms.refuse_without_terms",
    "F2",
    "fast",
    "doc/06 §6.2",
    "Ingestion layer strictly refuses any document that lacks a valid terms record.",
)
def test_refuse_without_terms():
    mod = _corpus_module()
    doc_without_terms = {"url": "https://example.com/article", "text": "Content"}
    with pytest.raises(Exception) as excinfo:
        mod.validate_document_rights(doc_without_terms, terms_record=None)
    assert "Terms" in type(excinfo.value).__name__ or "terms" in str(excinfo.value).lower()


@case(
    "corpus_terms.automated_fetch_disallowed_requires_local_import",
    "F2",
    "fast",
    "doc/06 §6.2",
    "When automated_fetch_allowed is false, the fetch path refuses and requires local file import.",
)
def test_automated_fetch_disallowed_requires_local_import():
    mod = _corpus_module()
    terms = {"automated_fetch_allowed": False, "source_identity": "blocked_source.com"}
    with pytest.raises(Exception):
        mod.fetch_document_online(url="https://blocked_source.com/news", terms=terms)


@case(
    "corpus_terms.full_text_storage_disallowed_stores_excerpt",
    "F2",
    "fast",
    "doc/06 §6.2, §8.1",
    "When full_text_storage_allowed is false, only metadata or permitted excerpt is stored, never full text.",
)
def test_full_text_storage_disallowed_stores_excerpt():
    mod = _corpus_module()
    terms = {"full_text_storage_allowed": False}
    full_text = "Paragraph 1\n\nParagraph 2\n\nParagraph 3" * 20
    stored_doc = mod.prepare_document_for_storage(full_text=full_text, terms=terms)
    assert stored_doc["storage_scope"] in ("EXCERPT", "METADATA_ONLY")
    assert stored_doc["storage_scope"] != "FULL_PAGE"
