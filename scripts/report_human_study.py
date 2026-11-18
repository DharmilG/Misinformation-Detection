"""Study 3 reporter - aggregate anonymous answers into the result template."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.study_server import load_cases  # noqa: E402


def report(answers: list[dict], cases: list[dict]) -> dict:
    gold = {str(c["id"]): c["gold_label"] for c in cases}
    wrong_verdict = {str(c["id"]): c.get("model_was_wrong") for c in cases}
    by_condition: dict[str, dict] = {}
    for condition in ("C0", "C1", "C2", "C3"):
        rows = [r for r in answers if r.get("condition") == condition]
        if not rows:
            by_condition[condition] = {"n": 0, "accuracy": "[PLANNED - NOT YET MEASURED]"}
            continue
        correct = sum(gold.get(r["case_id"]) == r["decision"] for r in rows)
        by_condition[condition] = {
            "n": len(rows),
            "accuracy": round(correct / len(rows), 6),
            "mean_time_ms": round(sum(r["elapsed_ms"] for r in rows) / len(rows), 3),
            "mean_confidence": round(sum(r["confidence"] for r in rows) / len(rows), 6),
            "confidence_accuracy_gap": round(
                sum(r["confidence"] / 7 for r in rows) / len(rows) - correct / len(rows), 6
            ),
        }
    followed = [
        r for r in answers
        if wrong_verdict.get(r["case_id"]) is True and r["condition"] != "C0"
    ]
    return {
        "participants": len({r["participant"] for r in answers}),
        "answers": len(answers),
        "by_condition": by_condition,
        "automation_bias_rate": (
            round(sum(r["decision"] != gold.get(r["case_id"]) for r in followed) / len(followed), 6)
            if followed
            else "[PLANNED - NOT YET MEASURED]"
        ),
        "abstention_compliance": "[PLANNED - NOT YET MEASURED]",
        "qualitative_themes": "[PLANNED - NOT YET MEASURED]",
        "verdict": "[PLANNED - NOT YET MEASURED]",
        "interpretation": "[PLANNED - NOT YET MEASURED]",
        "reproduction": "[PLANNED - NOT YET MEASURED]",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    answers = load_cases(args.answers)
    cases = load_cases(args.cases)
    result = report(answers, cases)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
