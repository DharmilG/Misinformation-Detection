"""
tests/test_escaping.py
=======================
Tests for src/reporting/escaping.py.

Spec: 05_REPORTING_AND_UI §19.

Fast tests — no model, no network, no I/O.
"""
from __future__ import annotations

import pytest

from tests.ids import case

from src.reporting.escaping import escape_display, escape_markdown, safe_url


class TestEscapeMarkdown:

    @case("reporting.escaping.test_heading_injection_blocked.1", "F2", "fast", "doc/05 ?19", "test heading injection blocked")
    def test_heading_injection_blocked(self) -> None:
        result = escape_markdown("# Injected heading")
        assert result.startswith("\\#") or "# Injected heading" not in result

    @case("reporting.escaping.test_link_injection_blocked.2", "F2", "fast", "doc/05 ?19", "test link injection blocked")
    def test_link_injection_blocked(self) -> None:
        raw = "[evil](javascript:alert(1))"
        result = escape_markdown(raw)
        assert "[evil]" not in result or "javascript:" not in result

    @case("reporting.escaping.test_backtick_escaped.3", "F2", "fast", "doc/05 ?19", "test backtick escaped")
    def test_backtick_escaped(self) -> None:
        result = escape_markdown("Use `code` here")
        assert "\\`" in result or "`code`" not in result

    @case("reporting.escaping.test_asterisk_escaped.4", "F2", "fast", "doc/05 ?19", "test asterisk escaped")
    def test_asterisk_escaped(self) -> None:
        result = escape_markdown("**bold**")
        assert "\\*" in result

    @case("reporting.escaping.test_plain_text_preserved.5", "F2", "fast", "doc/05 ?19", "test plain text preserved")
    def test_plain_text_preserved(self) -> None:
        result = escape_markdown("Hello world")
        assert "Hello world" in result

    @case("reporting.escaping.test_unicode_preserved.6", "F2", "fast", "doc/05 ?19", "test unicode preserved")
    def test_unicode_preserved(self) -> None:
        result = escape_markdown("Héllo wörld — café")
        assert "Héllo" in result
        assert "café" in result

    @case("reporting.escaping.test_none_returns_empty_string.7", "F2", "fast", "doc/05 ?19", "test none returns empty string")
    def test_none_returns_empty_string(self) -> None:
        result = escape_markdown(None)   # type: ignore[arg-type]
        # Should not raise; returns safe string
        assert isinstance(result, str)

    @case("reporting.escaping.test_control_characters_removed.8", "F2", "fast", "doc/05 ?19", "test control characters removed")
    def test_control_characters_removed(self) -> None:
        # Null bytes and low-ASCII control chars should not survive
        result = escape_markdown("Hello\x00World\x01")
        assert "\x00" not in result
        assert "\x01" not in result

    @case("reporting.escaping.test_code_fence_injection_blocked.9", "F2", "fast", "doc/05 ?19", "test code fence injection blocked")
    def test_code_fence_injection_blocked(self) -> None:
        result = escape_markdown("```\nrm -rf /\n```")
        # The triple backtick should be escaped
        assert "```" not in result or "rm -rf" not in result


class TestEscapeDisplay:

    @case("reporting.escaping.test_html_entities_escaped.10", "F2", "fast", "doc/05 ?19", "test html entities escaped")
    def test_html_entities_escaped(self) -> None:
        result = escape_display("<script>alert(1)</script>")
        assert "<script>" not in result

    @case("reporting.escaping.test_plain_text_preserved.11", "F2", "fast", "doc/05 ?19", "test plain text preserved")
    def test_plain_text_preserved(self) -> None:
        result = escape_display("Normal text")
        assert "Normal text" in result

    @case("reporting.escaping.test_unicode_preserved.12", "F2", "fast", "doc/05 ?19", "test unicode preserved")
    def test_unicode_preserved(self) -> None:
        result = escape_display("日本語テスト")
        assert "日本語" in result


class TestSafeUrl:

    @case("reporting.escaping.test_https_url_accepted.13", "F2", "fast", "doc/05 ?19", "test https url accepted")
    def test_https_url_accepted(self) -> None:
        url = "https://example.com/article"
        result = safe_url(url)
        assert result == url

    @case("reporting.escaping.test_http_url_accepted.14", "F2", "fast", "doc/05 ?19", "test http url accepted")
    def test_http_url_accepted(self) -> None:
        url = "http://example.com/article"
        result = safe_url(url)
        assert result == url

    @case("reporting.escaping.test_javascript_url_rejected.15", "F2", "fast", "doc/05 ?19", "test javascript url rejected")
    def test_javascript_url_rejected(self) -> None:
        result = safe_url("javascript:alert(1)")
        assert result is None

    @case("reporting.escaping.test_data_url_rejected.16", "F2", "fast", "doc/05 ?19", "test data url rejected")
    def test_data_url_rejected(self) -> None:
        result = safe_url("data:text/html,<h1>XSS</h1>")
        assert result is None

    @case("reporting.escaping.test_ftp_url_rejected.17", "F2", "fast", "doc/05 ?19", "test ftp url rejected")
    def test_ftp_url_rejected(self) -> None:
        result = safe_url("ftp://example.com/file.txt")
        assert result is None

    @case("reporting.escaping.test_none_returns_none.18", "F2", "fast", "doc/05 ?19", "test none returns none")
    def test_none_returns_none(self) -> None:
        result = safe_url(None)
        assert result is None

    @case("reporting.escaping.test_empty_string_returns_none.19", "F2", "fast", "doc/05 ?19", "test empty string returns none")
    def test_empty_string_returns_none(self) -> None:
        result = safe_url("")
        assert result is None

    @case("reporting.escaping.test_relative_url_rejected.20", "F2", "fast", "doc/05 ?19", "test relative url rejected")
    def test_relative_url_rejected(self) -> None:
        result = safe_url("/relative/path")
        assert result is None
