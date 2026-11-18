"""Study 4 fix experiments - apply one fix at a time to existing traces.

This is deliberately a *post-processing* script: it does not modify src/rag/.
Only fixes that can be expressed as a decision rule over recorded traces are
implemented.  Fixes that need upstream changes are named, with the reason they
cannot be evaluated yet, rather than being silently faked.

Usage:
    python scripts/run_rag_fixes.py --input rag_traces.jsonl \
        --fix quality-gate --output reports/study4/fix_quality_gate.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Fixes implementable as a rule over recorded traces.
TRACE_FIXES = ("adaptive", "quality-gate", "conservative")

# Fixes that need changes upstream of the trace.
UPSTREAM_FIXES = {
    "hybrid": "requires a real retrieval index and a re-ranker; retrieve() currently returns synthetic passages",
    "eligibility-soft": "requires rag_eligibility to emit an UNCERTAIN status instead of discarding",
    "injection-tune": "requires a configurable decision rule in rag_injection",
    "constrained-generation": "requires generator access and the ValidationRetryTracker wired into the RAG path",
    "soft-weighting": "requires the evidence signal to be a weighted continuous input to the decision",
    "explicit-mapping": "already applied in condition C5 of run_rag_ablation.py; compare C5 against C4",
}


def _label(row: dict[str, Any]) -> str:
    return str(row.get("predicted_label", "MATCHED"))


def _has_real_evidence(row: dict[str, Any]) -> bool:
    retrieval = row.get("retrieval_outputs") or {}
    return bool(retrieval) and not retrieval.get("synthetic_passages", False) and bool(
        retrieval.get("passage_count", 0)
    )


def apply_fix(rows: list[dict[str, Any]], fix: str) -> list[dict[str, Any]]:
    """Return a copy of the C5 rows with one fix applied."""
    selected = [dict(row) for row in rows if row.get("condition") == "C5"]
    baseline = {str(r["case_id"]): r for r in rows if r.get("condition") == "C1"}

    for row in selected:
        fallback = baseline.get(str(row["case_id"]))
        if fix == "adaptive":
            # The C6 trace already contains the adaptive rule.  C5 -> C6.
            adaptive = next(
                (
                    r for r in rows
                    if r.get("condition") == "C6" and str(r.get("case_id")) == str(row["case_id"])
                ),
                None,
            )
            if adaptive is not None:
                row["predicted_label"] = _label(adaptive)
                row["applied_fix"] = "adaptive"
            continue
        if fix == "quality-gate":
            if not _has_real_evidence(row) and fallback is not None:
                row["predicted_label"] = _label(fallback)
                row["applied_fix"] = "quality-gate"
            continue
        if fix == "conservative":
            outcome = (row.get("outcome_outputs") or {}).get("outcome")
            if outcome in (None, "INCONCLUSIVE", "CONFLICTING") and fallback is not None:
                row["predicted_label"] = _label(fallback)
                row["applied_fix"] = "conservative"
            continue
        raise ValueError(f"unknown trace fix: {fix}")
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="rag_traces.jsonl")
    parser.add_argument("--fix", required=True, choices=tuple(TRACE_FIXES) + tuple(UPSTREAM_FIXES))
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if args.fix in UPSTREAM_FIXES:
        print(
            json.dumps(
                {
                    "fix": args.fix,
                    "status": "BLOCKED",
                    "reason": UPSTREAM_FIXES[args.fix],
                    "measurement": "[PLANNED - NOT YET MEASURED]",
                },
                indent=2,
            )
        )
        return 0

    path = Path(args.input)
    with path.open("r", encoding="utf-8-sig") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    fixed = apply_fix(rows, args.fix)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in fixed:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"fix": args.fix, "rows": len(fixed), "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
