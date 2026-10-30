"""Second-machine verification command-line runner (Document 03 §28).

Usage:
    python scripts/verify_checkpoint.py --release <path-to-release-dir>
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure src/clip_model is in sys.path
ROOT = Path(__file__).resolve().parent.parent
CLIP_DIR = ROOT / "src" / "clip_model"
if str(CLIP_DIR) not in sys.path:
    sys.path.insert(0, str(CLIP_DIR))

from classifier_verification import run_verification


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run second-machine verification against reference fixture.")
    parser.add_argument("--release", required=True, help="Path to release directory")
    parser.add_argument("--score-tol", type=float, default=1e-4, help="Tolerance for continuous score difference")
    parser.add_argument("--prob-tol", type=float, default=1e-4, help="Tolerance for calibrated probability difference")
    parser.add_argument("--output", help="Optional path to write verification.json")
    args = parser.parse_args(argv)

    release_path = Path(args.release)
    if not release_path.is_dir():
        print(f"Error: release directory not found: {release_path}", file=sys.stderr)
        return 1

    report = run_verification(release_path, score_tol=args.score_tol, prob_tol=args.prob_tol)

    out_path = Path(args.output) if args.output else (release_path / "verification.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))

    if report.get("result") == "pass":
        print("Verification PASSED.")
        return 0
    else:
        print("Verification FAILED.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
