"""Classifier artifact packaging gates, copy schema, and release layout (Document 03 §17, §19, §25).

Defines release file layout, user-facing copy.json generator, and packaging gates
asserting privacy (no absolute paths) and synthetic fixture rights.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Release file layout (doc/03 §17, §19.1)
# ---------------------------------------------------------------------------

RELEASE_FILES: set[str] = {
    "release.json",
    "weights.pt",
    "config.json",
    "calibration.json",
    "metrics.json",
    "copy.json",
    "MODEL_CARD.md",
    "requirements-lock.txt",
    "reference_fixture",
}

COPY_KEYS: set[str] = {
    "copy_version",
    "what_this_measures",
    "not_a_truth_claim",
    "probability_note",
    "score_note",
}


def get_expected_release_files() -> set[str]:
    """Return the set of expected release files and directories."""
    return set(RELEASE_FILES)


# ---------------------------------------------------------------------------
# copy.json generator (doc/03 §19.1.1)
# ---------------------------------------------------------------------------

def generate_copy_json(version: str = "1.0.0") -> dict[str, str]:
    """Generate the machine-readable user-facing explanation copy (doc/03 §19.1.1)."""
    return {
        "copy_version": version,
        "what_this_measures": (
            "This score estimates whether the image and caption look like the "
            "mismatched or matched examples the model was trained on."
        ),
        "not_a_truth_claim": (
            "A mismatch score is not proof that the claim is false, and a match "
            "score is not proof that it is true. Use the retrieved sources to "
            "assess the claim itself."
        ),
        "probability_note": (
            "The probability reflects the model's confidence under its training "
            "data. It does not measure factual accuracy."
        ),
        "score_note": (
            "The score is the model's uncalibrated output. A calibrated probability "
            "is shown only when the release includes a valid calibration artifact."
        ),
    }


# ---------------------------------------------------------------------------
# Packaging Gate 3: No absolute paths (doc/03 §25)
# ---------------------------------------------------------------------------

ABSOLUTE_PATH_PATTERNS = [
    re.compile(r"^[a-zA-Z]:[\\/]", re.IGNORECASE),
    re.compile(r"^/(home|Users|root)/", re.IGNORECASE),
    re.compile(r"[a-zA-Z]:\\Users\\", re.IGNORECASE),
]


def check_absolute_paths_gate(payload: Any) -> bool:
    """Scan payload recursively to ensure no private local absolute paths exist (Gate 3)."""
    if isinstance(payload, str):
        for pat in ABSOLUTE_PATH_PATTERNS:
            if pat.search(payload):
                return False
        return True
    elif isinstance(payload, dict):
        for k, v in payload.items():
            if not check_absolute_paths_gate(k) or not check_absolute_paths_gate(v):
                return False
        return True
    elif isinstance(payload, (list, tuple, set)):
        for item in payload:
            if not check_absolute_paths_gate(item):
                return False
        return True
    return True


# ---------------------------------------------------------------------------
# Packaging Gate 7: Synthetic fixture rights (doc/03 §20.3, §25)
# ---------------------------------------------------------------------------

def check_fixture_rights_gate(fixture_dir: str | Path) -> bool:
    """Assert reference fixture samples are synthetic with generator.py and README (Gate 7)."""
    p = Path(fixture_dir)
    if not p.is_dir():
        return False

    generator_script = p / "generator.py"
    readme_file = p / "README.md"

    if not generator_script.is_file():
        return False
    return readme_file.is_file()
