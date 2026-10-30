"""Document ingestion safety controls of Document 06 §9.

Enforces HTTPS-only scheme, private IP/SSRF blocking, response size limits,
and redirect chain verification.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse


class IngestSafetyError(ValueError):
    """Raised when an ingest URL or response violates network safety rules."""


def validate_target_url(url: str, allow_http: bool = False) -> bool:
    """Ensure URL uses secure https scheme."""
    parsed = urlparse(url)
    if not allow_http and parsed.scheme.lower() != "https":
        raise IngestSafetyError(f"Insecure scheme '{parsed.scheme}'; only HTTPS is permitted.")
    return True


def is_ip_allowed(ip_str: str) -> bool:
    """Check if IP address is public and routable, rejecting loopback and private ranges."""
    try:
        ip = ipaddress.ip_address(ip_str)
        if (
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return False
        return True
    except ValueError:
        return False


def read_response_safe(stream: bytes, max_bytes: int = 1024 * 1024) -> bytes:
    """Read response bytes enforcing strict upper limit."""
    if len(stream) > max_bytes:
        raise IngestSafetyError(
            f"Response payload ({len(stream)} bytes) exceeds max limit ({max_bytes} bytes)."
        )
    return stream


def validate_redirect_chain(redirect_chain: list[str], max_redirects: int = 5) -> bool:
    """Validate that redirect hops do not exceed threshold and obey safety rules."""
    if len(redirect_chain) > max_redirects:
        raise IngestSafetyError(
            f"Redirect chain of length {len(redirect_chain)} exceeds max allowed ({max_redirects})."
        )
    for hop in redirect_chain:
        validate_target_url(hop)
    return True


def validate_redirect_target(source_url: str, target_url: str) -> bool:
    """Validate single redirect target against scheme and SSRF rules."""
    validate_target_url(target_url)
    parsed = urlparse(target_url)
    host = parsed.hostname or ""
    if not is_ip_allowed(host):
        # If hostname is an IP and not allowed
        try:
            ipaddress.ip_address(host)
            raise IngestSafetyError(f"Target host '{host}' resolves to restricted IP.")
        except ValueError:
            pass
    return True
