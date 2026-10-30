"""
tests/test_text_render.py
==========================
Tests for src/reporting/text_render.py.

Spec: 05_REPORTING_AND_UI §18 and §20 acceptance criteria.

Fast tests — no model, no network, no I/O.
All assertions operate on the plain-string output of render_report_text().
"""
from __future__ import annotations

import importlib
import inspect
import sys
from datetime import datetime, timezone

import pytest

from tests.ids import case

from src.reporting import copy as C
from src.reporting.comparison import PERMANENT_NOTE
from src.reporting.model import (
    AssessmentSection,
    CitationRecord,
    ClassifierSection,
    ClaimComponent,
    ComparisonSection,
    EvidenceSection,
    InputSection,
    PassageRecord,
    ReportModel,
    ReportProvenance,
    RetrievedPassage,
)
from src.reporting.text_render import render_report_text


# ── Fixtures ───────────────────────────────────────────────────────────────

def _prov() -> ReportProvenance:
    return ReportProvenance(
        model_version="test-model-v1",
        encoder_revision="enc1",
        prompt_version="pv1",
        corpus_version="cv1",
        corpus_manifest_sha256="abc123",
        retrieval_config_hash="rch1",
        embedding_revision="er1",
        chunking_version="chv1",
        feature_contract_version="fcv1",
        run_id="run-001",
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


def _inp(caption: str = "A test caption") -> InputSection:
    return InputSection(
        status="OK",
        original_filename="test.jpg",
        original_width=640,
        original_height=480,
        caption_original=caption,
        warnings=(),
    )


def _clf_ok() -> ClassifierSection:
    return ClassifierSection(
        status="OK",
        run_id="run-001",
        model_version="test-model-v1",
        encoder_revision="enc1",
        label="MATCHED",
        score=0.72,
    )


def _clf_unavailable() -> ClassifierSection:
    return ClassifierSection(
        status="UNAVAILABLE",
        run_id="run-001",
        model_version="test-model-v1",
        encoder_revision="enc1",
        reason="Model not installed.",
        remedy="Install open-clip-torch.",
    )


def _passage(
    passage_id: str = "p1",
    title: str = "Test Article",
    published_at: str | None = "2023-01-01",
    url: str | None = "https://example.com",
    storage_scope: str | None = None,
) -> RetrievedPassage:
    return RetrievedPassage(
        passage=PassageRecord(
            passage_id=passage_id,
            title=title,
            source_name="Test Source",
            source_type="news",
            published_at=published_at,
            excerpt="A relevant excerpt about the topic.",
            url=url,
            rights_wording=None,
            storage_scope=storage_scope,
        ),
        rank=1,
        total_retrieved=1,
        relevance_score=0.85,
    )


def _ev_ok(passages: tuple = ()) -> EvidenceSection:
    return EvidenceSection(
        status="OK",
        passages=passages,
        corpus_version="cv1",
        corpus_manifest_sha256="abc123",
        total_documents=6,
        total_passages=6,
    )


def _ev_unavailable() -> EvidenceSection:
    return EvidenceSection(
        status="UNAVAILABLE",
        passages=(),
        corpus_version="cv1",
        corpus_manifest_sha256="abc123",
        total_documents=0,
        total_passages=0,
        reason="Corpus not found.",
    )


def _asm_ok(outcome: str = "INCONCLUSIVE") -> AssessmentSection:
    return AssessmentSection(
        status="OK",
        prompt_version="pv1",
        scope_qualifier=C.SCOPE_QUALIFIER,
        outcome=outcome,
        reason_text="Insufficient evidence.",
    )


def _asm_unavailable() -> AssessmentSection:
    return AssessmentSection(
        status="UNAVAILABLE",
        prompt_version="pv1",
        scope_qualifier=C.SCOPE_QUALIFIER,
        reason="Assessment unavailable.",
    )


def _comp(state: str = "ASSESSMENT_UNRESOLVED") -> ComparisonSection:
    return ComparisonSection(
        state=state,
        sentence="The retrieved evidence did not settle the claim.",
    )


def _make_report(
    clf: ClassifierSection | None = None,
    ev: EvidenceSection | None = None,
    asm: AssessmentSection | None = None,
    comp: ComparisonSection | None = None,
    inp: InputSection | None = None,
    limitations: tuple[str, ...] = ("System limitation A.", "System limitation B."),
) -> ReportModel:
    return ReportModel(
        request_id="req-001",
        analysis_fingerprint="fp" * 32,
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        input=inp or _inp(),
        classifier=clf or _clf_ok(),
        evidence=ev or _ev_ok(),
        assessment=asm or _asm_ok(),
        comparison=comp or _comp(),
        limitations=limitations,
        provenance=_prov(),
    )


# ── No Streamlit import ────────────────────────────────────────────────────

class TestNoDependency:

    @case("reporting.text_render.test_text_render_does_not_import_streamlit.1", "F2", "fast", "doc/05 ?18", "test text render does not import streamlit")
    def test_text_render_does_not_import_streamlit(self) -> None:
        """render_report_text must have zero Streamlit imports."""
        import src.reporting.text_render as mod
        source = inspect.getsource(mod)
        assert "import streamlit" not in source
        assert "from streamlit" not in source
        assert "import st" not in source

    @case("reporting.text_render.test_text_render_has_no_io.2", "F2", "fast", "doc/05 ?18", "test text render has no io")
    def test_text_render_has_no_io(self) -> None:
        """Must not open files or write to disk."""
        import src.reporting.text_render as mod
        source = inspect.getsource(mod)
        assert "open(" not in source
        assert "write(" not in source


# ── Output completeness ────────────────────────────────────────────────────

class TestOutputCompleteness:

    @case("reporting.text_render.test_returns_non_empty_string.3", "F2", "fast", "doc/05 ?18", "test returns non empty string")
    def test_returns_non_empty_string(self) -> None:
        text = render_report_text(_make_report())
        assert isinstance(text, str)
        assert len(text) > 100

    @case("reporting.text_render.test_contains_all_section_headings.4", "F2", "fast", "doc/05 ?18", "test contains all section headings")
    def test_contains_all_section_headings(self) -> None:
        text = render_report_text(_make_report())
        for heading in [
            C.H_INPUT, C.H_CLASSIFIER, C.H_RETRIEVED,
            C.H_ASSESSMENT, C.H_COMPARISON, C.H_SYSTEM_LIMS, C.H_PROVENANCE,
        ]:
            assert heading in text, f"Missing heading: {heading}"

    @case("reporting.text_render.test_not_a_truth_claim_always_present.5", "F2", "fast", "doc/05 ?18", "test not a truth claim always present")
    def test_not_a_truth_claim_always_present(self) -> None:
        """§9.1 level 1 — always visible."""
        for clf in [_clf_ok(), _clf_unavailable()]:
            text = render_report_text(_make_report(clf=clf))
            assert C.NOT_A_TRUTH_CLAIM in text, (
                f"NOT_A_TRUTH_CLAIM missing for clf.status={clf.status}"
            )

    @case("reporting.text_render.test_scope_qualifier_always_present_when_asm_ok.6", "F2", "fast", "doc/05 ?18", "test scope qualifier always present when asm ok")
    def test_scope_qualifier_always_present_when_asm_ok(self) -> None:
        """§9.1 level 2 — always visible when assessment is OK."""
        text = render_report_text(_make_report(asm=_asm_ok()))
        assert C.SCOPE_QUALIFIER in text

    @case("reporting.text_render.test_permanent_note_always_present.7", "F2", "fast", "doc/05 ?18", "test permanent note always present")
    def test_permanent_note_always_present(self) -> None:
        """§8.4 — permanent note in every rendered comparison state."""
        text = render_report_text(_make_report())
        assert PERMANENT_NOTE in text

    @case("reporting.text_render.test_system_limitations_all_rendered.8", "F2", "fast", "doc/05 ?18", "test system limitations all rendered")
    def test_system_limitations_all_rendered(self) -> None:
        lims = ("Limitation alpha.", "Limitation beta.")
        text = render_report_text(_make_report(limitations=lims))
        for lim in lims:
            assert lim in text

    @case("reporting.text_render.test_no_combined_verdict.9", "F2", "fast", "doc/05 ?18", "test no combined verdict")
    def test_no_combined_verdict(self) -> None:
        """Never merge classifier + assessment into a single verdict."""
        text = render_report_text(_make_report())
        for phrase in C.FORBIDDEN_VERDICT_PHRASES:
            assert phrase.lower() not in text.lower(), (
                f"Forbidden verdict phrase found: '{phrase}'"
            )


# ── Null-metadata handling ─────────────────────────────────────────────────

class TestNullMetadata:

    @case("reporting.text_render.test_null_date_renders_no_date_string.10", "F2", "fast", "doc/05 ?18", "test null date renders no date string")
    def test_null_date_renders_no_date_string(self) -> None:
        p = _passage(published_at=None)
        ev = _ev_ok(passages=(p,))
        text = render_report_text(_make_report(ev=ev))
        assert C.NO_DATE in text

    @case("reporting.text_render.test_null_url_renders_no_url_string.11", "F2", "fast", "doc/05 ?18", "test null url renders no url string")
    def test_null_url_renders_no_url_string(self) -> None:
        p = _passage(url=None)
        ev = _ev_ok(passages=(p,))
        text = render_report_text(_make_report(ev=ev))
        assert C.NO_URL in text

    @case("reporting.text_render.test_invalid_url_not_rendered_as_link.12", "F2", "fast", "doc/05 ?18", "test invalid url not rendered as link")
    def test_invalid_url_not_rendered_as_link(self) -> None:
        p = _passage(url="javascript:evil()")
        ev = _ev_ok(passages=(p,))
        text = render_report_text(_make_report(ev=ev))
        assert "javascript:" not in text

    @case("reporting.text_render.test_storage_scope_excerpt_rendered_when_present.13", "F2", "fast", "doc/05 ?18", "test storage scope excerpt rendered when present")
    def test_storage_scope_excerpt_rendered_when_present(self) -> None:
        p = _passage(storage_scope="excerpt")
        ev = _ev_ok(passages=(p,))
        text = render_report_text(_make_report(ev=ev))
        assert C.STORAGE_EXCERPT in text

    @case("reporting.text_render.test_storage_scope_citation_rendered_when_present.14", "F2", "fast", "doc/05 ?18", "test storage scope citation rendered when present")
    def test_storage_scope_citation_rendered_when_present(self) -> None:
        p = _passage(storage_scope="citation_metadata")
        ev = _ev_ok(passages=(p,))
        text = render_report_text(_make_report(ev=ev))
        assert C.STORAGE_CITATION in text

    @case("reporting.text_render.test_storage_scope_none_not_rendered.15", "F2", "fast", "doc/05 ?18", "test storage scope none not rendered")
    def test_storage_scope_none_not_rendered(self) -> None:
        p = _passage(storage_scope=None)
        ev = _ev_ok(passages=(p,))
        text = render_report_text(_make_report(ev=ev))
        assert C.STORAGE_EXCERPT not in text
        assert C.STORAGE_CITATION not in text


# ── Empty evidence ─────────────────────────────────────────────────────────

class TestEmptyEvidence:

    @case("reporting.text_render.test_exact_empty_evidence_wording.16", "F2", "fast", "doc/05 ?18", "test exact empty evidence wording")
    def test_exact_empty_evidence_wording(self) -> None:
        """§6.3 — exact required wording when nothing was retrieved."""
        ev = _ev_ok(passages=())
        text = render_report_text(_make_report(ev=ev))
        assert C.EMPTY_EVIDENCE in text


# ── Relevance label ────────────────────────────────────────────────────────

class TestRelevanceLabel:

    @case("reporting.text_render.test_relevance_labelled_as_ranking_value.17", "F2", "fast", "doc/05 ?18", "test relevance labelled as ranking value")
    def test_relevance_labelled_as_ranking_value(self) -> None:
        p = _passage()
        ev = _ev_ok(passages=(p,))
        text = render_report_text(_make_report(ev=ev))
        assert C.RELEVANCE_LABEL in text


# ── Degraded section states ────────────────────────────────────────────────

class TestDegradedStates:

    @case("reporting.text_render.test_classifier_unavailable_renders_status.18", "F2", "fast", "doc/05 ?18", "test classifier unavailable renders status")
    def test_classifier_unavailable_renders_status(self) -> None:
        text = render_report_text(_make_report(clf=_clf_unavailable()))
        assert "UNAVAILABLE" in text

    @case("reporting.text_render.test_evidence_unavailable_renders_status.19", "F2", "fast", "doc/05 ?18", "test evidence unavailable renders status")
    def test_evidence_unavailable_renders_status(self) -> None:
        text = render_report_text(_make_report(ev=_ev_unavailable()))
        assert "UNAVAILABLE" in text

    @case("reporting.text_render.test_assessment_unavailable_renders_status.20", "F2", "fast", "doc/05 ?18", "test assessment unavailable renders status")
    def test_assessment_unavailable_renders_status(self) -> None:
        text = render_report_text(_make_report(asm=_asm_unavailable()))
        assert "UNAVAILABLE" in text


# ── Markdown injection ─────────────────────────────────────────────────────

class TestMarkdownInjection:

    @case("reporting.text_render.test_malicious_caption_escaped.21", "F2", "fast", "doc/05 ?18", "test malicious caption escaped")
    def test_malicious_caption_escaped(self) -> None:
        inp = _inp(caption="# Injected heading\n**bold**\n[link](http://evil.com)")
        text = render_report_text(_make_report(inp=inp))
        # The raw markdown syntax should not appear verbatim as structural markup
        # (the heading marker # should be escaped)
        assert "# Injected heading" not in text or "\\#" in text
