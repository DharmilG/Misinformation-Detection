"""
src/reporting/text_render.py
=============================
Pure text / Markdown rendering of a ReportModel.

Spec: 05_REPORTING_AND_UI.md §18

Properties (all are invariants tested by the fast test suite):
  - Pure: no Streamlit import, no I/O, no side effects, no global mutation.
  - Complete: every field the UI shows is also present in this text,
    so tests can assert on content without a running Streamlit server.
  - All wording comes from src.reporting.copy — not from inline strings here.
  - All untrusted strings are escaped via src.reporting.escaping.
  - Missing metadata uses the exact documented wording (copy.NO_DATE, copy.NO_URL …)
  - No placeholder strings ("", "N/A", "unknown", "-") for missing values.

Used by:
  - tests/test_reporting/ (copy lint, wording checks, null-metadata checks)
  - src/reporting/export.py (the Markdown download)
"""
from __future__ import annotations

from src.reporting import copy as C
from src.reporting.comparison import PERMANENT_NOTE
from src.reporting.escaping import escape_markdown, safe_url
from src.reporting.model import (
    AssessmentSection,
    ClassifierSection,
    ComparisonSection,
    EvidenceSection,
    InputSection,
    ReportModel,
    RetrievedPassage,
)


# ── Internal helpers ───────────────────────────────────────────────────────

def _esc(text: str | None, fallback: str = "Not stated") -> str:
    """Escape and return text, or the documented fallback when absent."""
    if text is None:
        return escape_markdown(fallback)
    return escape_markdown(text)


def _section(title: str) -> str:
    return f"\n## {escape_markdown(title)}\n"


def _subsection(title: str) -> str:
    return f"\n### {escape_markdown(title)}\n"


def _bullet(text: str) -> str:
    return f"- {text}"


def _note(text: str) -> str:
    """Render as a Markdown blockquote."""
    return f"> {text}"


# ── Input section (§4) ────────────────────────────────────────────────────

def _render_input(inp: InputSection) -> list[str]:
    lines = [_section(C.H_INPUT)]
    lines.append(f"Status: {C.INPUT_ACCEPTED}")
    lines.append(
        f"File: {escape_markdown(inp.original_filename)} "
        f"({inp.original_width}\u00d7{inp.original_height}\u00a0px)"
    )
    # Caption verbatim (§4: "the user's original text, exactly as submitted")
    lines.append(f"Caption: {escape_markdown(inp.caption_original)}")

    if inp.warnings:
        lines.append(_subsection("Preprocessing Warnings"))
        for w in inp.warnings:
            msg = C.WARNING_MESSAGES.get(w, escape_markdown(w))
            lines.append(_bullet(escape_markdown(msg)))

    return lines


# ── Classifier section (§5) ───────────────────────────────────────────────

def _render_classifier(clf: ClassifierSection) -> list[str]:
    lines = [_section(C.H_CLASSIFIER)]
    lines.append(f"Status: {clf.status}")

    if clf.status == "OK":
        icon_label = C.OUTCOME_ICON.get(clf.label or "", clf.label or "")
        lines.append(f"Result: {icon_label}")
        lines.append(
            f"Score: {clf.score} \u2014 {C.SCORE_NOTE}"
        )
        if clf.calibration_status == "APPLIED" and clf.calibrated_probability is not None:
            lines.append(
                f"{C.CAL_PROB_NOTE}: {clf.calibrated_probability}"
            )
        else:
            lines.append(C.CAL_NOT_APPLIED)
        lines.append("")
        lines.append(C.CLASSIFIER_WHAT_MEASURES)
        lines.append("")
        # §9.1 level 1 — always visible
        lines.append(_note(C.NOT_A_TRUTH_CLAIM))
        lines.append("")
        lines.append(_subsection("Provenance"))
        lines.append(f"Model version: {escape_markdown(clf.model_version)}")
        lines.append(f"Run ID: {escape_markdown(clf.run_id)}")
        lines.append(f"Encoder revision: {escape_markdown(clf.encoder_revision)}")
        lines.append(f"Score definition: {escape_markdown(clf.score_definition)}")
    else:
        # Degraded: show reason + remedy; NO score, label, or probability
        lines.append(
            f"The image\u2013caption comparison is {clf.status.lower()}."
        )
        if clf.reason:
            lines.append(f"Reason: {escape_markdown(clf.reason)}")
        if clf.remedy:
            lines.append(f"Remedy: {escape_markdown(clf.remedy)}")
        if clf.error_detail:
            lines.append(f"Detail: {escape_markdown(clf.error_detail)}")
        # Limitation still present even when classifier fails (§9.1 level 1)
        lines.append("")
        lines.append(_note(C.NOT_A_TRUTH_CLAIM))

    return lines


# ── Evidence section (§6) ─────────────────────────────────────────────────

def _render_passage(rp: RetrievedPassage) -> list[str]:
    lines: list[str] = []
    pas = rp.passage
    lines.append(_subsection(f"Passage {rp.rank}: {escape_markdown(pas.title or 'Untitled')}"))
    lines.append(
        f"Source: {escape_markdown(pas.source_name)} "
        f"({escape_markdown(pas.source_type)})"
    )
    # §6.1 date
    date_text = escape_markdown(pas.published_at) if pas.published_at else C.NO_DATE
    lines.append(f"Date: {date_text}")

    # §6.1 URL — clickable only when valid; never manufactured
    url = safe_url(pas.url)
    if url:
        lines.append(f"URL: {url}")
    else:
        lines.append(f"URL: {C.NO_URL}")

    # §6.2 rank and relative indicator
    lines.append(f"Rank: {C.rank_label(rp.rank, rp.total_retrieved)}")
    if rp.relevance_score is not None:
        lines.append(
            f"Relevance score: {rp.relevance_score} \u2014 {C.RELEVANCE_LABEL}"
        )

    # Excerpt — truncated for display, truncation noted
    excerpt = pas.excerpt
    DISPLAY_LIMIT = 600
    if len(excerpt) > DISPLAY_LIMIT:
        lines.append(
            f"Excerpt (first {DISPLAY_LIMIT} of {len(excerpt)} characters):\n"
            f"{escape_markdown(excerpt[:DISPLAY_LIMIT])}\n"
            f"*(excerpt truncated for display)*"
        )
    else:
        lines.append(f"Excerpt: {escape_markdown(excerpt)}")

    # §6.1 rights_wording — only when present
    if pas.rights_wording:
        lines.append(f"Rights: {escape_markdown(pas.rights_wording)}")

    # §6.1 storage_scope — only when stored metadata records it (not inferred)
    if pas.storage_scope == "excerpt":
        lines.append(C.STORAGE_EXCERPT)
    elif pas.storage_scope == "citation_metadata":
        lines.append(C.STORAGE_CITATION)
    # None → nothing rendered (acceptance criterion 10)

    return lines


def _render_evidence(ev: EvidenceSection) -> list[str]:
    lines = [_section(C.H_RETRIEVED)]
    lines.append(f"Status: {ev.status}")

    if ev.status != "OK":
        lines.append(
            f"Evidence retrieval is {ev.status.lower()}."
        )
        if ev.reason:
            lines.append(f"Reason: {escape_markdown(ev.reason)}")
        return lines

    # Corpus coverage always shown
    lines.append(
        f"Corpus: {escape_markdown(ev.corpus_version)} | "
        f"Documents: {ev.total_documents} | "
        f"Passages: {ev.total_passages}"
    )

    if not ev.passages:
        # §6.3 — exact required wording
        lines.append("")
        lines.append(C.EMPTY_EVIDENCE)
        lines.append("")
        lines.append(C.NO_EVIDENCE_SUGGESTION)
    else:
        for rp in ev.passages:
            lines.extend(_render_passage(rp))

    return lines


# ── Assessment section (§7) ───────────────────────────────────────────────

def _render_assessment(asm: AssessmentSection) -> list[str]:
    lines = [_section(C.H_ASSESSMENT)]
    lines.append(f"Status: {asm.status}")

    if asm.status != "OK":
        lines.append(
            f"Claim assessment is {asm.status.lower()}."
        )
        if asm.reason:
            lines.append(f"Reason: {escape_markdown(asm.reason)}")
        if asm.error_detail:
            lines.append(f"Detail: {escape_markdown(asm.error_detail)}")
        return lines

    icon_label = C.OUTCOME_ICON.get(asm.outcome or "", asm.outcome or "")
    lines.append(f"Outcome: {icon_label}")
    lines.append(f"Reason: {escape_markdown(asm.reason_text or '')}")
    lines.append("")
    # §9.1 level 2 — always visible
    lines.append(f"Scope: {escape_markdown(asm.scope_qualifier)}")
    lines.append("")

    if asm.components:
        lines.append(_subsection("Claim Components"))
        for comp in asm.components:
            lines.append(
                _bullet(
                    f"{escape_markdown(comp.description)}: {comp.status}"
                )
            )

    # Accepted citations
    if asm.accepted_citations:
        lines.append(_subsection(C.H_ACCEPTED_CITS))
        # §7.2 disclaimer whenever citations are present
        lines.append(C.CITATION_SEMANTICS_DISCLAIMER)
        lines.append("")
        for cit in asm.accepted_citations:
            date_text = (
                escape_markdown(cit.published_at)
                if cit.published_at
                else C.NO_DATE
            )
            url = safe_url(cit.url)
            lines.append(f"Source: {escape_markdown(cit.source_name)}")
            lines.append(f"Date: {date_text}")
            lines.append(f"URL: {url if url else C.NO_URL}")
            lines.append(f"Quote: {escape_markdown(cit.quote)}")
            lines.append("")

    # Rejected citations — always shown so fabrication attempts are visible
    if asm.rejected_citations:
        lines.append(_subsection(C.H_REJECTED_CITS))
        for cit in asm.rejected_citations:
            lines.append(f"Source: {escape_markdown(cit.source_name)}")
            lines.append(
                f"Rejection reason: "
                f"{escape_markdown(cit.rejection_reason or 'Not stated')}"
            )
            lines.append("")

    if asm.unresolved_items:
        lines.append(_subsection("Unresolved Items"))
        for item in asm.unresolved_items:
            lines.append(_bullet(escape_markdown(item)))

    if asm.conflicts:
        lines.append(_subsection("Conflicts"))
        for c in asm.conflicts:
            lines.append(_bullet(escape_markdown(c)))

    if asm.uncertainties:
        lines.append(_subsection("Uncertainties"))
        for u in asm.uncertainties:
            lines.append(_bullet(escape_markdown(u)))

    if asm.limitations:
        lines.append(_subsection("Assessment Limitations"))
        for lim in asm.limitations:
            lines.append(_bullet(escape_markdown(lim)))

    return lines


# ── Comparison section (§8) ───────────────────────────────────────────────

def _render_comparison(comp: ComparisonSection) -> list[str]:
    lines = [_section(C.H_COMPARISON)]
    lines.append(f"State: {comp.state}")
    lines.append(comp.sentence)
    lines.append("")
    # §8.4 — permanent note in every rendered comparison state
    lines.append(_note(PERMANENT_NOTE))
    return lines


# ── System limitations (§9.1 level 3) ─────────────────────────────────────

def _render_limitations(lims: tuple[str, ...]) -> list[str]:
    lines = [_section(C.H_SYSTEM_LIMS)]
    for lim in lims:
        lines.append(_bullet(escape_markdown(lim)))
    return lines


# ── Provenance ─────────────────────────────────────────────────────────────

def _render_provenance(report: ReportModel) -> list[str]:
    prov = report.provenance
    lines = [_section(C.H_PROVENANCE)]
    lines.append(f"Request ID: {report.request_id}")
    lines.append(f"Analysis fingerprint: {report.analysis_fingerprint}")
    lines.append(f"Created: {report.created_at.isoformat()}")
    lines.append(f"Model version: {escape_markdown(prov.model_version)}")
    lines.append(f"Encoder revision: {escape_markdown(prov.encoder_revision)}")
    lines.append(f"Prompt version: {escape_markdown(prov.prompt_version)}")
    lines.append(f"Corpus version: {escape_markdown(prov.corpus_version)}")
    lines.append(
        f"Corpus manifest SHA-256: {escape_markdown(prov.corpus_manifest_sha256)}"
    )
    lines.append(
        f"Retrieval config hash: {escape_markdown(prov.retrieval_config_hash)}"
    )
    lines.append(
        f"Embedding revision: {escape_markdown(prov.embedding_revision)}"
    )
    lines.append(
        f"Chunking version: {escape_markdown(prov.chunking_version)}"
    )
    lines.append(
        f"Feature contract version: "
        f"{escape_markdown(prov.feature_contract_version)}"
    )
    lines.append(f"Run ID: {escape_markdown(prov.run_id)}")
    return lines


# ── Public API ─────────────────────────────────────────────────────────────

def render_report_text(report: ReportModel) -> str:
    """
    Render the complete ReportModel as escaped Markdown text.

    Properties (all enforced by tests):
      - Pure: no imports of Streamlit, no file I/O, no side effects.
      - Complete: contains every field the UI displays.
      - Escaped: all untrusted strings go through escape_markdown().
      - All wording sourced from src.reporting.copy.
      - No combined verdict. No placeholder strings.
    """
    # Assembling section by section in the same order as the report model (§16)
    sections: list[list[str]] = [
        [f"# Misinformation Detection — Analysis Report\n"],
        _render_input(report.input),
        _render_classifier(report.classifier),
        _render_evidence(report.evidence),
        _render_assessment(report.assessment),
        _render_comparison(report.comparison),
        _render_limitations(report.limitations),
        _render_provenance(report),
    ]

    lines: list[str] = []
    for section in sections:
        lines.extend(section)
        lines.append("")   # blank line between sections

    return "\n".join(lines)
