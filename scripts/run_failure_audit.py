"""Study 1 driver - produce paired measurements for the failure-mode audit.

Reads a case manifest and a human-authored variant manifest, scores every
(image, caption variant) pair with the existing encoder and classifier, and
writes one self-contained JSON object per pair.

No model is trained and no implementation module is modified.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.research_runtime import (  # noqa: E402
    load_research_runtime,
    score_pair,
)


LABEL_TO_INT = {"MATCHED": 0, "MISMATCHED": 1}
INT_TO_LABEL = {value: key for key, value in LABEL_TO_INT.items()}
VARIANTS = ("V0", "V1", "V2", "V3", "V4", "V5", "V6")
MECHANICAL = {"V2", "V5"}


class AuditInputError(ValueError):
    """Raised when a manifest is malformed."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # utf-8-sig accepts both plain UTF-8 and a BOM (common from Windows editors).
    with path.open("r", encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AuditInputError(f"{path}:{number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise AuditInputError(f"{path}:{number}: expected a JSON object")
            rows.append(row)
    return rows


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except OSError:
        return "unknown"


def shortening(caption: str) -> str:
    """V2 - keep the first clause, at most six words.  Deterministic."""
    head = caption.split(".")[0].split(",")[0]
    return " ".join(head.split()[:6]) or caption.strip()


def negation(caption: str) -> str:
    """V5 - negate the caption.  The gold label flips by construction."""
    return "It is not true that " + caption.rstrip(".")


def mechanical_variant(
    variant: str, caption: str, gold_label: str
) -> tuple[str, str]:
    """Return (caption, gold_label) for a mechanically generated variant."""
    if variant == "V2":
        return shortening(caption), gold_label
    if variant == "V5":
        return negation(caption), INT_TO_LABEL[1 - LABEL_TO_INT[gold_label]]
    raise AuditInputError(f"variant {variant} is not mechanical")


def load_inputs(
    manifest: Path, variants_path: Path | None
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    cases = _read_jsonl(manifest)
    if not cases:
        raise AuditInputError(f"{manifest} contains no cases")

    authored: dict[str, list[dict[str, Any]]] = {}
    if variants_path is not None:
        for row in _read_jsonl(variants_path):
            authored.setdefault(str(row["id"]), []).append(row)

    by_id: dict[str, dict[str, Any]] = {}
    for case in cases:
        cid = str(case.get("id", ""))
        if not cid or cid in by_id:
            raise AuditInputError(f"duplicate or missing case id: {cid!r}")
        if case.get("gold_label") not in LABEL_TO_INT:
            raise AuditInputError(f"{cid}: bad gold_label {case.get('gold_label')!r}")
        image = case.get("image")
        if not image or not (ROOT / str(image)).is_file():
            raise AuditInputError(f"{cid}: image not found: {image!r}")
        by_id[cid] = case

    # Validate authored variants before doing any inference.
    for cid, rows in authored.items():
        if cid not in by_id:
            raise AuditInputError(f"variant refers to unknown case {cid!r}")
        for row in rows:
            variant = str(row.get("variant", ""))
            if not variant.startswith("V") or variant not in VARIANTS:
                raise AuditInputError(f"{cid}: unknown variant {variant!r}")
            if row.get("gold_label") not in LABEL_TO_INT:
                raise AuditInputError(f"{cid}/{variant}: missing or bad gold_label")
            if variant in MECHANICAL:
                raise AuditInputError(
                    f"{cid}/{variant} is mechanical and must not be authored"
                )
            if not str(row.get("caption", "")).strip():
                raise AuditInputError(f"{cid}/{variant}: empty caption")

    return cases, authored


def iter_pairs(
    cases: list[dict[str, Any]], authored: dict[str, list[dict[str, Any]]]
) -> Any:
    """Yield (case, variant, caption, gold_label) in a stable order."""
    for case in cases:
        cid = str(case["id"])
        original = str(case["caption"])
        gold = str(case["gold_label"])
        yield case, "V0", original, gold
        for variant in VARIANTS[1:]:
            if variant in MECHANICAL:
                caption, variant_gold = mechanical_variant(variant, original, gold)
                yield case, variant, caption, variant_gold
                continue
            for row in authored.get(cid, []):
                if str(row["variant"]) == variant:
                    yield case, variant, str(row["caption"]), str(row["gold_label"])


def run(args: argparse.Namespace) -> dict[str, Any]:
    cases, authored = load_inputs(Path(args.manifest),
                                  Path(args.variants) if args.variants else None)
    runtime = load_research_runtime(
        args.checkpoint,
        temperature=args.temperature,
        temperature_provenance=args.temperature_provenance,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, str]] = set()
    if args.resume and output.is_file():
        for row in _read_jsonl(output):
            done.add((str(row.get("id")), str(row.get("variant"))))

    provenance = runtime.as_provenance()
    provenance.update(
        {
            "script": "scripts/run_failure_audit.py",
            "git_commit": _git_commit(),
            "manifest": str(args.manifest),
            "variants": str(args.variants) if args.variants else None,
            "limiting": args.limit,
        }
    )
    meta_path = output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    written = 0
    with output.open("a" if args.resume else "w", encoding="utf-8") as handle:
        for case, variant, caption, gold in iter_pairs(cases, authored):
            cid = str(case["id"])
            if (cid, variant) in done:
                continue
            if args.limit and written >= args.limit:
                break
            image_path = ROOT / str(case["image"])
            scores = score_pair(runtime, str(image_path), caption)
            predicted = INT_TO_LABEL.get(scores.pop("predicted_label"), "UNKNOWN")
            row = {
                "id": cid,
                "variant": variant,
                "caption": caption,
                "gold_label": gold,
                "predicted_label": predicted,
                "correct": predicted == gold,
                "mismatch_type": case.get("mismatch_type"),
                "strata": case.get("strata", {}),
                **scores,
                **provenance,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    return {"rows_written": written, "output": str(output), "meta": str(meta_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--variants", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument(
        "--temperature-provenance",
        default="supplied",
        help="where the temperature came from (recorded in every row)",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    try:
        summary = run(args)
    except (AuditInputError, FileNotFoundError, ValueError) as exc:
        print(f"Audit input error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
