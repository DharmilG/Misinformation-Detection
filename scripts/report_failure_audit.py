"""Study 1 reporter - turn paired audit rows into the study result template.

Every number written here is derived from an audit_results.jsonl produced by
``scripts/run_failure_audit.py``.  Fields with no data stay as the literal
``[PLANNED - NOT YET MEASURED]`` so the report can be copied verbatim.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clip_model.metrics import compute_accuracy  # noqa: E402
from src.clip_model.classifier_calibration import compute_ece  # noqa: E402

PLACEHOLDER = "[PLANNED - NOT YET MEASURED]"
VARIANTS = ("V1", "V2", "V3", "V4", "V5", "V6")


def bootstrap_ci(values: list[float], *, draws: int = 1000, seed: int = 42) -> list[float]:
    """Percentile bootstrap CI for the mean of ``values``."""
    if not values:
        return [PLACEHOLDER, PLACEHOLDER]  # type: ignore[list-item]
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(draws):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * draws)]
    hi = means[min(draws - 1, int(0.975 * draws))]
    return [round(lo, 6), round(hi, 6)]


def _correct(row: dict[str, Any]) -> float:
    return 1.0 if row.get("correct") is True else 0.0


def accuracy_of(rows: Iterable[dict[str, Any]]) -> float | str:
    rows = list(rows)
    if not rows:
        return PLACEHOLDER
    gold = [1 if row.get("gold_label") == "MISMATCHED" else 0 for row in rows]
    predicted = [1 if row.get("predicted_label") == "MISMATCHED" else 0 for row in rows]
    return round(float(compute_accuracy(gold, predicted)), 6)


def paired_delta(
    rows: list[dict[str, Any]], variant: str, *, draws: int = 1000, seed: int = 42
) -> dict[str, Any]:
    """Paired delta between a variant and V0 on the same case ids."""
    base = {str(row["id"]): _correct(row) for row in rows if row.get("variant") == "V0"}
    other = {
        str(row["id"]): _correct(row) for row in rows if row.get("variant") == variant
    }
    ids = sorted(set(base) & set(other))
    if not ids:
        return {"n": 0, "delta": PLACEHOLDER, "ci95": [PLACEHOLDER, PLACEHOLDER]}
    diffs = [other[i] - base[i] for i in ids]
    return {
        "n": len(ids),
        "delta": round(sum(diffs) / len(diffs), 6),
        "ci95": bootstrap_ci(diffs, draws=draws, seed=seed),
    }


def group_accuracy(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """Accuracy grouped by ``mismatch_type`` or by a named ``strata`` key."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("variant") != "V0":
            continue
        if key == "mismatch_type":
            value = row.get("mismatch_type")
        else:
            value = (row.get("strata") or {}).get(key)
        if value is None:
            continue
        groups.setdefault(str(value), []).append(row)
    if not groups:
        return {PLACEHOLDER: PLACEHOLDER}
    return {name: accuracy_of(group) for name, group in sorted(groups.items())}


def report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    base = [row for row in rows if row.get("variant") == "V0"]
    result: dict[str, Any] = {
        "measurement_count": len(rows),
        "overall_accuracy_V0": accuracy_of(base),
        "paired_delta": {variant: paired_delta(rows, variant) for variant in VARIANTS},
        "per_mismatch_type_accuracy": group_accuracy(rows, "mismatch_type"),
    }
    strata_keys: set[str] = set()
    for row in base:
        strata_keys.update((row.get("strata") or {}).keys())
    result["per_stratum_accuracy"] = {
        key: group_accuracy(rows, key) for key in sorted(strata_keys)
    }

    # ECE on V0, using the calibrated probability and the predicted score.
    if base:
        probs = [float(r["calibrated_probability"]) for r in base]
        labels = [1 if r["gold_label"] == "MISMATCHED" else 0 for r in base]
        ece, _ = compute_ece(probs, labels)
        result["ece_V0"] = round(float(ece), 6)
    else:
        result["ece_V0"] = PLACEHOLDER

    result["inter_annotator_kappa"] = PLACEHOLDER
    result["verdict"] = PLACEHOLDER
    result["interpretation"] = PLACEHOLDER
    result["reproduction"] = PLACEHOLDER
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="audit_results.jsonl")
    parser.add_argument("--output", required=True, help="study1 results JSON")
    args = parser.parse_args(argv)

    path = Path(args.input)
    rows: list[dict[str, Any]] = []
    # utf-8-sig accepts both plain UTF-8 and a BOM (common from Windows editors).
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    result = report(rows)
    result["source"] = str(path)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
