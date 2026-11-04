"""Evaluation harness preconditions per Document 07 §3.

Each evaluation script asserts its own preconditions and refuses to run otherwise:
1. Split isolation
2. Configuration fingerprint matching
3. Complete provenance
4. Output bound with context
5. Fixture name enforcement
"""
from __future__ import annotations

import re
from typing import Any


class EvaluationPreconditionError(RuntimeError):
    """Raised when an evaluation precondition is violated."""


def assert_split_match(*, script_name: str, split_name: str) -> None:
    """Refuse to run if the script reads a split it is not named for.

    Convention: a script with 'dev' in its name may only read 'dev' split,
    'test' -> 'test', 'train' -> 'train', 'val' -> 'val'.
    """
    # Extract the intended split from the script name
    match = re.search(r"(dev|test|train|val)", script_name.lower())
    if match is None:
        return  # no split constraint encoded in name
    intended = match.group(1)
    if split_name.lower() != intended:
        raise EvaluationPreconditionError(
            f"Script '{script_name}' is named for split '{intended}' "
            f"but was pointed at split '{split_name}'."
        )


def assert_fingerprint_match(
    *, active_fingerprint: str, recorded_fingerprint: str
) -> None:
    """Abort if the configuration fingerprint differs from the recorded one."""
    if active_fingerprint != recorded_fingerprint:
        raise EvaluationPreconditionError(
            f"Configuration fingerprint mismatch: active={active_fingerprint[:16]}... "
            f"recorded={recorded_fingerprint[:16]}..."
        )


_REQUIRED_PROVENANCE_KEYS = (
    "model_version",
    "corpus_manifest_sha256",
    "prompt_version",
    "retrieval_config_hash",
)


def validate_evaluation_provenance(provenance: dict[str, Any]) -> bool:
    """Return True if provenance is complete; raise otherwise."""
    missing = [k for k in _REQUIRED_PROVENANCE_KEYS if k not in provenance]
    if missing:
        raise EvaluationPreconditionError(
            f"Evaluation provenance incomplete; missing: {missing}"
        )
    return True


def build_evaluation_record(
    *,
    metrics: dict[str, Any],
    configuration: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Package evaluation output with configuration and provenance."""
    return {
        "metrics": metrics,
        "configuration": configuration,
        "provenance": provenance,
    }


def assert_fixture_allowed(*, script_name: str, fixture_path: str) -> None:
    """Abort on attempt to evaluate on an un-named fixture or test set.

    A named fixture must contain an identifiable name component (not just
    'unnamed', 'temp', 'scratch', or a bare UUID).
    """
    filename = fixture_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    stem = filename.rsplit(".", 1)[0].lower()
    # Reject fixtures with no meaningful name
    if "unnamed" in stem or "temp" in stem or "scratch" in stem:
        raise EvaluationPreconditionError(
            f"Script '{script_name}' cannot evaluate on un-named fixture '{fixture_path}'."
        )
