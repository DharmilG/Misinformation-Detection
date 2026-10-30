"""The ingestion pipeline safety controls of Document 06 §9.

Tests HTTPS-only scheme restriction, private address SSRF blocking,
response size limit enforcement, and redirect re-validation.
"""

import pytest

from tests.ids import case


def _ingest_module():
    import importlib
    return importlib.import_module("retrieval_ingest")


@case(
    "ingest_safety.module_exists",
    "F2",
    "fast",
    "doc/06 §9",
    "The project provides a document ingestion and safety module.",
)
def test_ingest_safety_module_exists():
    _ingest_module()


@case(
    "ingest_safety.schemes_https_only",
    "F2",
    "fast",
    "doc/06 §9.2",
    "Only https schemes are permitted; http requests are refused unless explicit override is configured.",
)
def test_schemes_https_only():
    mod = _ingest_module()
    with pytest.raises(Exception):
        mod.validate_target_url("http://insecure-source.com/article")
    # https passes validation
    assert mod.validate_target_url("https://secure-source.com/article") is True


@case(
    "ingest_safety.private_addresses_rejected",
    "F2",
    "fast",
    "doc/06 §9.2",
    "SSRF guard resolves DNS and rejects loopback (127.0.0.1), private RFC1918, link-local, and reserved IP ranges.",
)
def test_private_addresses_rejected():
    mod = _ingest_module()
    for bad_ip in ("127.0.0.1", "10.0.0.1", "192.168.1.1", "169.254.169.254", "::1"):
        assert mod.is_ip_allowed(bad_ip) is False, "Private/reserved IP was not blocked: %s" % bad_ip


@case(
    "ingest_safety.response_size_limit_raises",
    "F2",
    "fast",
    "doc/06 §9.2",
    "Payload exceeding max response byte size raises an error rather than silently truncating.",
)
def test_response_size_limit_raises():
    mod = _ingest_module()
    max_bytes = 1024 * 1024  # 1 MB
    huge_stream = b"X" * (max_bytes + 10)
    with pytest.raises(Exception):
        mod.read_response_safe(huge_stream, max_bytes=max_bytes)


@case(
    "ingest_safety.redirect_limit_and_revalidation",
    "F2",
    "fast",
    "doc/06 §9.2",
    "Redirects are bounded in number and each hop is re-validated against all safety rules.",
)
def test_redirect_limit_and_revalidation():
    mod = _ingest_module()
    # Excessive redirect chain fails
    redirect_chain = ["https://hop%d.com" % i for i in range(15)]
    with pytest.raises(Exception):
        mod.validate_redirect_chain(redirect_chain, max_redirects=5)

    # Redirect to private IP fails
    with pytest.raises(Exception):
        mod.validate_redirect_target(source_url="https://valid.com", target_url="http://127.0.0.1/admin")
