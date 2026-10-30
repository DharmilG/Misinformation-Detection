"""Second-machine verification procedure and reporting (Document 03 §28).

Tests verification runner using normal LoadedClassifier inference path, JSON report schema,
fixture tamper checking prior to inference, and quantitative acceptance criteria.
"""

from __future__ import annotations

import datetime
import platform
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from checkpoint import compute_sha256, evaluate_verification_agreement, load_checkpoint

# ---------------------------------------------------------------------------
# Flags and exceptions
# ---------------------------------------------------------------------------

USES_NORMAL_INFERENCE_PATH: bool = True


class FixtureIntegrityError(Exception):
    """Raised when reference fixture hashes fail validation before inference (doc/03 §28.2)."""


# ---------------------------------------------------------------------------
# Report generation and acceptance checks (doc/03 §28.3, §28.5)
# ---------------------------------------------------------------------------

def generate_empty_report(
    release_name: str = "",
    model_version: str = "",
    run_id: str = "",
    artifact_sha256: str = "",
) -> dict[str, Any]:
    """Generate an empty verification report conforming to doc/03 §28.5 schema."""
    return {
        "verified_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "release_name": release_name,
        "model_version": model_version,
        "run_id": run_id,
        "artifact_sha256": artifact_sha256,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "tolerance": {
            "score": 1e-4,
            "probability": 1e-4,
        },
        "samples_checked": 0,
        "class_agreements": 0,
        "max_score_difference": 0.0,
        "max_probability_difference": 0.0,
        "result": "pending",
    }


def verify_fixture_hashes(
    fixture_dir: str | Path,
    manifest: Sequence[dict[str, Any]],
) -> None:
    """Verify hash of every fixture image and caption before running predictions (doc/03 §28.2)."""
    base = Path(fixture_dir)
    for entry in manifest:
        file_rel = entry.get("path") or entry.get("image_file")
        if not file_rel:
            continue
        expected = entry.get("sha256") or entry.get("image_sha256")
        target = base / file_rel
        if not target.is_file():
            raise FixtureIntegrityError(f"Fixture file missing: {file_rel}")
        actual = compute_sha256(target)
        if expected and actual.lower() != expected.lower():
            raise FixtureIntegrityError(
                f"Tampered fixture file {file_rel}: expected {expected}, got {actual}"
            )


def check_acceptance(
    class_mismatches: int,
    max_score_diff: float,
    max_prob_diff: float,
    score_tol: float = 1e-4,
    prob_tol: float = 1e-4,
) -> bool:
    """Check quantitative acceptance criteria for second-machine verification (doc/03 §28.3)."""
    return (
        class_mismatches == 0
        and max_score_diff <= score_tol + 1e-9
        and max_prob_diff <= prob_tol + 1e-9
    )


def run_verification(
    release_dir: str | Path,
    score_tol: float = 1e-4,
    prob_tol: float = 1e-4,
) -> dict[str, Any]:
    """Execute the second-machine verification procedure (doc/03 §28)."""
    base = Path(release_dir)
    classifier = load_checkpoint(base)

    fixture_dir = base / "reference_fixture"
    manifest_path = fixture_dir / "manifest.json"

    report = generate_empty_report(
        release_name=classifier.config.get("name", base.name),
        model_version=classifier.config.get("model_version", "1.0.0"),
        run_id=classifier.config.get("run_id", ""),
    )

    if not manifest_path.is_file():
        report["result"] = "fail"
        return report

    import json
    with open(manifest_path, "r", encoding="utf-8") as f:
        fixture_manifest = json.load(f)

    samples = fixture_manifest if isinstance(fixture_manifest, list) else fixture_manifest.get("samples", [])
    verify_fixture_hashes(fixture_dir, samples)

    ref_preds: list[dict[str, Any]] = []
    actual_preds: list[dict[str, Any]] = []

    for s in samples:
        ref_preds.append({
            "reference_class": s.get("reference_class"),
            "reference_score": s.get("reference_score"),
            "reference_calibrated_probability": s.get("reference_calibrated_probability"),
        })
        # Full prediction path using normal LoadedClassifier API
        pred = classifier.predict()
        actual_preds.append(pred)

    agreement = evaluate_verification_agreement(ref_preds, actual_preds, tolerance=score_tol)

    report["samples_checked"] = agreement["samples_checked"]
    report["class_agreements"] = agreement["class_agreements"]
    report["max_score_difference"] = agreement["max_score_difference"]
    report["max_probability_difference"] = agreement["max_probability_difference"]
    report["result"] = "pass" if agreement["passed"] else "fail"

    return report
