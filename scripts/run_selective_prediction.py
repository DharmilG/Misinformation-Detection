"""Study 2 driver - extract V0 rows from a Study 1 audit for signal analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.selective_prediction import all_signals  # noqa: E402

SIGNALS = ("cosine", "d_c_score", "calibrated_probability")


def extract_v0(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one row per case, from the unperturbed V0 variant."""
    seen: set[str] = set()
    result = []
    for row in rows:
        if row.get("variant") != "V0":
            continue
        cid = str(row.get("id", ""))
        if cid in seen:
            continue
        seen.add(cid)
        result.append(row)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="audit_results.jsonl")
    parser.add_argument("--output", required=True, help="study2 signals JSON")
    args = parser.parse_args(argv)

    path = Path(args.input)
    with path.open("r", encoding="utf-8-sig") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    v0 = extract_v0(rows)
    if not v0:
        print("no V0 rows found in the audit", file=sys.stderr)
        return 2

    result = {
        "source": str(path),
        "cases": len(v0),
        "signals": all_signals(v0, signals=SIGNALS),
        "by_mismatch_type": {},
    }
    from src.evaluation.selective_prediction import stratified

    for name, group in stratified(v0, "mismatch_type").items():
        result["by_mismatch_type"][name] = {
            "cases": len(group),
            "signals": all_signals(group, signals=SIGNALS),
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"cases": len(v0), "output": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
