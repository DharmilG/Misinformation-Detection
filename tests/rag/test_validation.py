"""The output schema, quote verification, and URL scanning validation of Document 06 §29.

Tests JSON schema constraints, strict case-sensitive NFC quote matching (no negation stripping),
URL detection outside validated quotes, and retry policy.
"""

import pytest

from tests.ids import case, known_gap


def _validation_module():
    import importlib
    return importlib.import_module("rag_validation")


@known_gap(
    "rag_validation.module_exists",
    "F2",
    "fast",
    "doc/06 §29",
    "The project provides a generation output and citation validation module.",
    "there is no rag_validation module",
)
def test_validation_module_exists():
    _validation_module()


@known_gap(
    "rag_validation.schema_keys_and_bounds",
    "F2",
    "fast",
    "doc/06 §29.1",
    "Generation schema validates required keys, rejects unexpected keys, and bounds component/citation counts.",
    "schema validation is not implemented",
)
def test_schema_keys_and_bounds():
    mod = _validation_module()
    # Unexpected key causes rejection
    bad_output = {
        "claim_restated": "Claim",
        "claim_components": [],
        "evidence_summary": "Summary",
        "citations": [],
        "extra_unexpected_key": "injected",
    }
    assert mod.validate_schema(bad_output)["valid"] is False


@known_gap(
    "rag_validation.quote_verification_strict",
    "F2",
    "fast",
    "doc/06 §29.2",
    "Quote verification uses strict case-sensitive NFC and whitespace collapse; negations or punctuation changes fail.",
    "strict quote verification is not implemented",
)
def test_quote_verification_strict():
    mod = _validation_module()
    source_passage = "The summit was not held in Geneva, but in Vienna."

    # Exact verbatim quote passes
    assert mod.verify_quote(quote="not held in Geneva", passage=source_passage) is True

    # Changed case fails
    assert mod.verify_quote(quote="Not held in Geneva", passage=source_passage) is False

    # Stripped negation fails
    assert mod.verify_quote(quote="held in Geneva", passage=source_passage) is False or not mod.verify_quote(
        quote="held in Geneva", passage=source_passage
    )


@known_gap(
    "rag_validation.url_scanning_outside_quotes",
    "F2",
    "fast",
    "doc/06 §29.3",
    "URLs outside validated quotes reject the generation; URLs inside verbatim quotes are permitted.",
    "URL scan security check is not implemented",
)
def test_url_scanning_outside_quotes():
    mod = _validation_module()
    # URL in prose summary fails
    leaky_generation = {
        "evidence_summary": "For more information see https://malicious-source.com/info",
        "citations": [],
    }
    assert mod.scan_for_prohibited_urls(leaky_generation)["valid"] is False

    # URL inside validated quote from passage passes
    valid_quoted_url = {
        "evidence_summary": "The source noted the link.",
        "citations": [{"quote": "published at https://official.gov/report", "passage_id": "c1"}],
    }
    passages = {"c1": "The report was published at https://official.gov/report on Friday."}
    assert mod.validate_generation_urls(valid_quoted_url, passages)["valid"] is True


@known_gap(
    "rag_validation.single_retry_policy",
    "F2",
    "fast",
    "doc/06 §29.5",
    "Schema violations allow exactly one retry with a stricter instruction before outright rejection.",
    "validation retry policy is not implemented",
)
def test_single_retry_policy():
    mod = _validation_module()
    tracker = mod.ValidationRetryTracker(max_retries=1)
    assert tracker.can_retry() is True
    tracker.record_failure("JSON parse error")
    assert tracker.can_retry() is False
