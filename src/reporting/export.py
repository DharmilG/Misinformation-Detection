"""
src/reporting/export.py
=======================
Markdown export — thin wrapper over render_report_text().

Spec: 05_REPORTING_AND_UI.md §15

Guarantees (§15 requirements):
  - Contains all retrieved passages (not only accepted citations).
  - Accepted and rejected citations are separate, clearly labelled.
  - Does NOT include dataset-split or leakage diagnostics.
  - Does NOT include the uploaded image bytes.
  - Includes the export disclosure string from copy.EXPORT_DISCLOSURE.
  - Pure function — no Streamlit import, no I/O.

Used by app.py's st.download_button.
"""
from __future__ import annotations

from datetime import datetime, timezone

from src.reporting import copy as C
from src.reporting.model import ReportModel
from src.reporting.text_render import render_report_text


def generate_markdown_export(report: ReportModel) -> str:
    """
    Return the complete Markdown export string for a report.

    The string is safe to write to a .md file.  It begins with the
    export disclosure note so the recipient knows what the file contains.
    """
    disclosure = (
        f"> **Export note:** {C.EXPORT_DISCLOSURE}\n\n"
        "---\n\n"
    )
    body = render_report_text(report)
    return disclosure + body


def get_export_filename(report: ReportModel) -> str:
    """
    Return a safe, descriptive filename for the download.

    Format: misinformation-report-<YYYYMMDD-HHMMSS>-<first8 of fingerprint>.md
    """
    ts = report.created_at
    # Ensure timezone-aware timestamp for consistent output
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    ts_str = ts.strftime("%Y%m%d-%H%M%S")
    fp_short = report.analysis_fingerprint[:8]
    return f"misinformation-report-{ts_str}-{fp_short}.md"
