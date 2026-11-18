"""Start the localhost Study 3 instrument."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.study_server import Experiment, load_cases, serve  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, help="JSONL of consented cases")
    parser.add_argument("--output", required=True, help="answers JSONL")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    cases = load_cases(args.cases)
    experiment = Experiment(cases=cases, output=Path(args.output), port=args.port)
    print(json.dumps({"cases": len(cases), "output": args.output}, indent=2))
    serve(experiment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
