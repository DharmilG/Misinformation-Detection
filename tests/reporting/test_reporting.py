"""The report model, per-section status contract, error and fallback rendering,
analysis fingerprint, export, and escaping of Document 05.

Tests the immutable ReportModel, strict status inheritance (OK / UNAVAILABLE /
SKIPPED / FAILED), visible degraded sections with None values, rejection views,
fingerprint hashing, markdown export schemas, and escaping of untrusted text.
"""

import pytest

from tests.ids import case, known_gap


def _reporting_module():
    import importlib
    return importlib.import_module("src.reporting.report_model")


@known_gap(
    "reporting.model_structure",
    "F2",
    "fast",
    "doc/05 §2.1",
    "ReportModel structure defines request_id, analysis_fingerprint, created_at, input, classifier, evidence, assessment, comparison, limitations, provenance.",
    "ReportModel is not yet implemented",
)
def test_reporting_model_structure():
    mod = _reporting_module()
    assert hasattr(mod, "ReportModel")
    fields = getattr(mod.ReportModel, "__dataclass_fields__", {})
    required = {
        "request_id",
        "analysis_fingerprint",
        "created_at",
        "input",
        "classifier",
        "evidence",
        "assessment",
        "comparison",
        "limitations",
        "provenance",
    }
    assert required.issubset(fields.keys())


@known_gap(
    "reporting.per_section_status_contract",
    "F2",
    "fast",
    "doc/05 §2.2",
    "Per-section status strictly reuses OK / UNAVAILABLE / SKIPPED / FAILED without new status vocabulary; input is always OK.",
    "reporting status contract is not yet implemented",
)
def test_reporting_per_section_status_contract():
    mod = _reporting_module()
    allowed_statuses = {"OK", "UNAVAILABLE", "SKIPPED", "FAILED"}
    assert hasattr(mod, "ReportSectionStatus") or hasattr(mod, "STATUS_VOCABULARY")
    status_enum = getattr(mod, "ReportSectionStatus", None)
    if status_enum:
        names = {s.name for s in status_enum}
        assert names == allowed_statuses


@known_gap(
    "reporting.required_payload_per_status",
    "F2",
    "fast",
    "doc/05 §2.3",
    "Non-OK status produces None for result values; OK status requires validated payload.",
    "section payload validation is not yet implemented",
)
def test_reporting_required_payload_per_status():
    mod = _reporting_module()
    assert hasattr(mod, "validate_section_payload")
    # For a degraded/unavailable section, result fields must be None
    degraded_sec = mod.ClassifierSection(status="UNAVAILABLE", reason="Model offline")
    assert degraded_sec.label is None
    assert degraded_sec.score is None
    assert degraded_sec.probability is None


@known_gap(
    "reporting.no_placeholders_in_payload",
    "F2",
    "fast",
    "doc/05 §2.4",
    "Rendered report never contains placeholder strings like empty string, N/A, unknown, or dash.",
    "report renderer placeholder checking is not yet implemented",
)
def test_reporting_no_placeholders_in_payload():
    mod = _reporting_module()
    assert hasattr(mod, "render_report_text")
    # Any placeholder like "N/A", "unknown", or "-" must be rejected
    report = mod.create_sample_report()
    rendered = mod.render_report_text(report)
    for forbidden in ["\"N/A\"", "unknown", " - "]:
        assert forbidden not in rendered


@known_gap(
    "reporting.error_fallback_matrix",
    "F2",
    "fast",
    "doc/05 §10",
    "Degraded sections remain visible with failure reasons while other unaffected sections continue to render.",
    "error and fallback matrix rendering is not yet implemented",
)
def test_reporting_error_fallback_matrix():
    mod = _reporting_module()
    assert hasattr(mod, "render_report_text")
    report = mod.create_degraded_classifier_report()
    rendered = mod.render_report_text(report)
    # Degraded section is visible
    assert "IMAGE-CAPTION CONSISTENCY" in rendered.upper()
    assert "unavailable" in rendered.lower()
    # No score or label rendered for degraded section
    assert "Score" not in rendered
    # But assessment section still renders
    assert "CLAIM ASSESSMENT" in rendered.upper()


@known_gap(
    "reporting.input_rejection_view",
    "F2",
    "fast",
    "doc/05 §11",
    "Invalid input renders rejection view with verbatim validation error; no report model or request ID is constructed.",
    "input rejection view is not yet implemented",
)
def test_reporting_input_rejection_view():
    mod = _reporting_module()
    assert hasattr(mod, "render_input_rejection")
    res = mod.render_input_rejection(validation_error="Unsupported file format: bmp")
    assert "Unsupported file format: bmp" in res
    assert not hasattr(res, "request_id")
    assert not isinstance(res, getattr(mod, "ReportModel", object))


@known_gap(
    "reporting.analysis_fingerprint_components",
    "F2",
    "fast",
    "doc/05 §12.2",
    "Analysis fingerprint is SHA-256 over image, caption, model, run ID, encoder, corpus manifest, prompt, retrieval config, and chunking hashes.",
    "analysis fingerprinting is not yet implemented",
)
def test_reporting_analysis_fingerprint_components():
    mod = _reporting_module()
    assert hasattr(mod, "compute_analysis_fingerprint")
    fp = mod.compute_analysis_fingerprint(
        image_sha256="a" * 64,
        caption_sha256="b" * 64,
        model_version="v1.0",
        run_id="run_42",
        encoder_revision="rev1",
        feature_contract_version="1",
        corpus_manifest_sha256="c" * 64,
        prompt_version="1.0",
        retrieval_config_hash="d" * 64,
        embedding_revision="e1",
        chunking_version="1",
    )
    assert isinstance(fp, str) and len(fp) == 64


@known_gap(
    "reporting.rerun_and_cache_invalidation",
    "F2",
    "fast",
    "doc/05 §12.3",
    "Display interaction reuses cached report without rerun; submission or config change computes new fingerprint and invalidates cache.",
    "session cache invalidation policy is not yet implemented",
)
def test_reporting_rerun_and_cache_invalidation():
    mod = _reporting_module()
    assert hasattr(mod, "ReportCache")
    cache = mod.ReportCache()
    report = mod.create_sample_report()
    cache.store(report)
    assert cache.lookup(report.analysis_fingerprint) is report
    # Configuration change invalidates
    assert cache.lookup("different_fingerprint_hash") is None


@known_gap(
    "reporting.export_markdown_enumerated_fields",
    "F2",
    "fast",
    "doc/05 §15.1",
    "Markdown export contains exactly the enumerated submission, consistency, assessment, evidence, comparison, limitation, and provenance fields.",
    "markdown export serializer is not yet implemented",
)
def test_reporting_export_markdown_enumerated_fields():
    mod = _reporting_module()
    assert hasattr(mod, "export_markdown")
    report = mod.create_sample_report()
    md = mod.export_markdown(report)
    assert "# Analysis Report" in md
    assert "Image-Caption Consistency" in md
    assert "Claim Assessment" in md
    assert "Retrieved Passages" in md
    assert "Accepted Citations" in md
    assert "Provenance" in md


@known_gap(
    "reporting.export_retrieved_vs_cited_distinction",
    "F2",
    "fast",
    "doc/05 §15.2",
    "Export preserves separation between all retrieved passages and accepted citations, including rejected citations with reasons.",
    "retrieved vs cited distinction in export is not yet implemented",
)
def test_reporting_export_retrieved_vs_cited_distinction():
    mod = _reporting_module()
    assert hasattr(mod, "export_markdown")
    report = mod.create_sample_report_with_rejected_citations()
    md = mod.export_markdown(report)
    assert "Retrieved Passages" in md
    assert "Accepted Citations" in md
    assert "Rejected Citations" in md


@known_gap(
    "reporting.export_excludes_dataset_diagnostics",
    "F2",
    "fast",
    "doc/05 §15.3",
    "Per-analysis export excludes dataset-split and leakage diagnostics which belong exclusively to Document 07 evaluation reports.",
    "export diagnostic boundary enforcement is not yet implemented",
)
def test_reporting_export_excludes_dataset_diagnostics():
    mod = _reporting_module()
    assert hasattr(mod, "export_markdown")
    report = mod.create_sample_report()
    md = mod.export_markdown(report)
    assert "leakage" not in md.lower()
    assert "exposure_any_pct" not in md
    assert "split_diagnostics" not in md


@known_gap(
    "reporting.untrusted_text_escaped",
    "F2",
    "fast",
    "doc/05 §16.1",
    "All untrusted strings (caption, passage excerpt, quotes, titles, URLs) are escaped against markdown and control sequence injection.",
    "untrusted text escaping is not yet implemented",
)
def test_reporting_untrusted_text_escaped():
    mod = _reporting_module()
    assert hasattr(mod, "escape_untrusted_markdown")
    malicious = "# Injected Title\n<script>alert(1)</script>\n[link](http://evil.com)"
    escaped = mod.escape_untrusted_markdown(malicious)
    assert not escaped.startswith("# ")
    assert "<script>" not in escaped


@known_gap(
    "reporting.outcome_word_and_icon",
    "F2",
    "fast",
    "doc/05 §16",
    "Assessment outcome is rendered as both word and symbol (Supported ✓, Contradicted ✗, Inconclusive ?, Conflicting ⚠), never colour alone.",
    "accessible outcome encoding is not yet implemented",
)
def test_reporting_outcome_word_and_icon():
    mod = _reporting_module()
    assert hasattr(mod, "format_outcome_badge")
    assert "✓" in mod.format_outcome_badge("SUPPORTED")
    assert "Supported" in mod.format_outcome_badge("SUPPORTED")
    assert "✗" in mod.format_outcome_badge("CONTRADICTED")
    assert "?" in mod.format_outcome_badge("INCONCLUSIVE")
    assert "⚠" in mod.format_outcome_badge("CONFLICTING")


@known_gap(
    "reporting.pure_text_rendering",
    "F2",
    "fast",
    "doc/05 §18.1",
    "Pure render_report_text function renders complete report without Streamlit dependencies or side effects.",
    "pure text rendering layer is not yet implemented",
)
def test_reporting_pure_text_rendering():
    mod = _reporting_module()
    assert hasattr(mod, "render_report_text")
    report = mod.create_sample_report()
    text = mod.render_report_text(report)
    assert isinstance(text, str)
    assert len(text) > 0
