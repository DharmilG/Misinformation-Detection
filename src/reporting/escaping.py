"""
src/reporting/escaping.py
==========================
Safe escaping of untrusted content for both the UI and the Markdown export.

Spec: 05_REPORTING_AND_UI.md §16.1

Untrusted sources:
    caption, retrieved passage excerpts, titles, source names, quotations

Rules:
  - escape_markdown()  → prevents injection of headings, links, emphasis in .md
  - escape_display()   → escapes HTML entities and strips control characters
  - safe_url()         → returns None unless the URL scheme is http or https
  - All functions are pure (no I/O, no side effects)
"""
from __future__ import annotations

import html
import re
from urllib.parse import urlparse

# Markdown special structural characters that could alter layout if unescaped
_MD_SPECIAL = r"\`*_{}[]()#|><~"


def escape_markdown(text: str) -> str:
    """
    Escape a string so that it renders as inert literal text in Markdown.
    Prevents injection of headings (#), links ([text](url)), emphasis (*_),
    tables (|), blockquotes (>), code fences (`), and HTML tags (<>).
    Also strips ASCII control characters (except \n and \t).
    """
    if not text:
        return ""
    # Strip control characters (0x00–0x08, 0x0B–0x0C, 0x0E–0x1F, 0x7F)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Escape each Markdown-special character with a backslash
    result = []
    for ch in text:
        if ch in _MD_SPECIAL:
            result.append("\\" + ch)
        else:
            result.append(ch)
    return "".join(result)


def escape_display(text: str) -> str:
    """
    Escape HTML entities and strip control characters from text destined
    for Streamlit display to prevent XSS / markup injection.
    Returns an empty string for None or empty input.
    """
    if not text:
        return ""
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return html.escape(cleaned)


def safe_url(url: str | None) -> str | None:
    """
    Return the URL string if it is a valid http or https URL.
    Returns None for None, empty strings, javascript:, ftp:, data:,
    relative paths, and anything else that is not http/https.
    Never manufacture a URL; only pass through valid ones.
    """
    if not url:
        return None
    url = url.strip()
    try:
        parsed = urlparse(url)
        if parsed.scheme.lower() in ("http", "https") and parsed.netloc:
            return url
    except Exception:
        pass
    return None
