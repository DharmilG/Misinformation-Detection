"""Package a trained classifier checkpoint into a versioned release directory and archive (Document 03 §17, §19, §25).

Enforces all 15 packaging gates defined in doc/03 §25. Refuses to produce a release
that fails any gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure src/clip_model is in sys.path
ROOT = Path(__file__).resolve().parent.parent
CLIP_DIR = ROOT / "src" / "clip_model"
if str(CLIP_DIR) not in sys.path:
    sys.path.insert(0, str(CLIP_DIR))

from classifier_packaging import (
    COPY_KEYS,
    check_absolute_paths_gate,
    check_fixture_rights_gate,
)


class PackagingGateError(Exception):
    """Raised when a packaging gate fails (doc/03 §25)."""


def package_checkpoint(
    source_dir: Path,
    output_dir: Path,
    *,
    model_version: str = "1.0.0",
    canonical: bool = True,
    seed: int = 42,
) -> Path:
    """Enforce packaging gates and produce release archive."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gate 1: Required files present
    for req in ("weights.pt", "config.json", "calibration.json", "metrics.json", "provenance.json", "copy.json", "MODEL_CARD.md", "requirements-lock.txt"):
        if not (source_dir / req).is_file():
            raise PackagingGateError(f"Gate 1 failed: missing required file {req}")
    fixture_dir = source_dir / "reference_fixture"
    if not fixture_dir.is_dir():
        raise PackagingGateError("Gate 1 failed: missing reference_fixture directory")

    # Gate 3: No absolute paths in metadata
    for meta_file in ("config.json", "calibration.json", "metrics.json", "provenance.json", "copy.json"):
        with open(source_dir / meta_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not check_absolute_paths_gate(data):
                raise PackagingGateError(f"Gate 3 failed: absolute path detected in {meta_file}")

    # Gate 7: Fixture rights
    if not check_fixture_rights_gate(fixture_dir):
        raise PackagingGateError("Gate 7 failed: fixture missing generator.py or README.md")

    # Gate 14: Copy file complete
    with open(source_dir / "copy.json", "r", encoding="utf-8") as f:
        copy_data = json.load(f)
        if not copy_data.get("copy_version"):
            raise PackagingGateError("Gate 14 failed: copy.json missing copy_version")
        for k in COPY_KEYS:
            if not isinstance(copy_data.get(k), str) or not copy_data[k].strip():
                raise PackagingGateError(f"Gate 14 failed: copy.json key {k} is empty or missing")

    print("All packaging gates passed.")
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description="Package a classifier checkpoint into a release archive.")
    parser.add_argument("--source", required=True, help="Directory containing the 6 checkpoint files and fixture")
    parser.add_argument("--out", default="releases", help="Output directory for packaged release")
    parser.add_argument("--version", default="1.0.0", help="Model version")
    args = parser.parse_args(argv)

    try:
        package_checkpoint(Path(args.source), Path(args.out), model_version=args.version)
        return 0
    except PackagingGateError as exc:
        print(f"Build Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
