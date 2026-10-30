"""Environment and model compatibility validation (Document 03 §22).

Enforces hard requirements (torch_min, weights_only_supported), checks supported
version ranges with actionable warnings / strict errors, and verifies lockfile presence.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

try:
    from packaging.specifiers import SpecifierSet
    from packaging.version import parse as parse_version
except ImportError:
    # Minimal fallback parser
    def parse_version(v: str):
        parts = []
        for x in str(v).split("."):
            num = ""
            for char in x:
                if char.isdigit():
                    num += char
                else:
                    break
            parts.append(int(num) if num else 0)
        return tuple(parts)

    class SpecifierSet:
        def __init__(self, spec: str) -> None:
            self.spec = spec

        def contains(self, v: str) -> bool:
            pv = parse_version(v)
            for clause in self.spec.split(","):
                clause = clause.strip()
                if clause.startswith(">="):
                    if pv < parse_version(clause[2:]):
                        return False
                elif clause.startswith("<="):
                    if pv > parse_version(clause[2:]):
                        return False
                elif clause.startswith(">"):
                    if pv <= parse_version(clause[1:]):
                        return False
                elif clause.startswith("<"):
                    if pv >= parse_version(clause[1:]):
                        return False
                elif clause.startswith("==") and pv != parse_version(clause[2:]):
                    return False
            return True


# ---------------------------------------------------------------------------
# Exceptions (doc/03 §22.4)
# ---------------------------------------------------------------------------

class CheckpointIncompatibleError(Exception):
    """Raised when environment or model compatibility requirements fail (doc/03 §22.4)."""


# ---------------------------------------------------------------------------
# Hard requirements checking (doc/03 §22.3, §22.4)
# ---------------------------------------------------------------------------

def check_compatibility(
    env: dict[str, Any],
    hard_requirements: dict[str, Any],
) -> None:
    """Enforce hard requirements. A violation raises CheckpointIncompatibleError with installed and expected versions."""
    # Check torch_min
    if "torch_min" in hard_requirements:
        expected_min = str(hard_requirements["torch_min"])
        installed_torch = str(env.get("torch", ""))
        if not installed_torch or parse_version(installed_torch) < parse_version(expected_min):
            raise CheckpointIncompatibleError(
                f"Incompatible PyTorch version: installed version {installed_torch} is below "
                f"expected minimum version {expected_min}"
            )

    # Check weights_only_supported
    if hard_requirements.get("weights_only_supported", False):
        supported = env.get("weights_only_supported")
        if supported is False:
            raise CheckpointIncompatibleError(
                f"Incompatible environment: weights_only_supported is {supported}; "
                f"expected weights_only_supported=True"
            )

    # Check numpy major version
    if "numpy_major_allowed" in hard_requirements:
        allowed = hard_requirements["numpy_major_allowed"]
        installed_numpy = str(env.get("numpy", ""))
        if installed_numpy:
            major = parse_version(installed_numpy)
            major_num = major[0] if isinstance(major, tuple) else getattr(major, "major", None)
            if major_num not in allowed:
                raise CheckpointIncompatibleError(
                    f"Incompatible NumPy version: installed version {installed_numpy} (major {major_num}) "
                    f"not in allowed major versions {allowed}"
                )


# ---------------------------------------------------------------------------
# Supported ranges checking (doc/03 §22.4)
# ---------------------------------------------------------------------------

def check_supported_ranges(
    env: dict[str, Any],
    ranges: dict[str, str],
    strict: bool = False,
) -> None:
    """Check supported version ranges. Emits warnings unless strict is set to True."""
    for package, spec_str in ranges.items():
        if package in env:
            installed = str(env[package])
            spec = SpecifierSet(spec_str)
            if not spec.contains(installed):
                msg = (
                    f"Package '{package}' version {installed} is outside supported range {spec_str}"
                )
                if strict:
                    raise CheckpointIncompatibleError(msg)
                else:
                    warnings.warn(msg, UserWarning, stacklevel=2)


# ---------------------------------------------------------------------------
# Lockfile verification (doc/03 §22.5)
# ---------------------------------------------------------------------------

def verify_lockfile(release_dir: str | Path) -> bool:
    """Verify that requirements-lock.txt exists in release and is non-empty."""
    lockfile = Path(release_dir) / "requirements-lock.txt"
    if not lockfile.is_file():
        return False
    text = lockfile.read_text(encoding="utf-8").strip()
    return len(text) > 0
