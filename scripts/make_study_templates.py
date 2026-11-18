"""Create pre-filled research manifests from the validation split.

The generated files already contain ids, image paths, original captions and gold
labels.  Only the fields that require human judgement are left blank.  The held
out test split is never sampled.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset" / "ooc_dataset"
LABELS = {"0": "MATCHED", "1": "MISMATCHED"}
AUTHORED_VARIANTS = ("V1", "V3", "V4", "V6")


class TemplateError(ValueError):
    """Raised when a template cannot be created safely."""


def _rows(split: str) -> list[dict[str, str]]:
    if split == "test":
        raise TemplateError("the held-out test split must not be sampled")
    path = DATASET / f"{split}.csv"
    if not path.is_file():
        raise TemplateError(f"missing split file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image_path", "caption", "label", "caption_id", "image_id"}
    if rows and not required.issubset(rows[0]):
        raise TemplateError(f"{path} has unexpected columns: {sorted(rows[0])}")
    return [row for row in rows if (DATASET / row["image_path"]).is_file()]


def _sample(
    rows: list[dict[str, str]], count: int, seed: int
) -> list[dict[str, str]]:
    """Stratified sample, balanced across the two gold labels."""
    if count < 1:
        raise TemplateError("--cases must be positive")
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[str(row["label"])].append(row)
    keys = sorted(by_label)
    if not keys:
        raise TemplateError("no rows available for sampling")

    rng = random.Random(seed)
    per = max(1, count // len(keys))
    selected: list[dict[str, str]] = []
    for key in keys:
        pool = list(by_label[key])
        rng.shuffle(pool)
        selected.extend(pool[:per])
    # Fill any remainder deterministically from what is left.
    remaining = [r for r in rows if r not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, count - len(selected))])
    rng.shuffle(selected)
    return selected[:count]


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def create_templates(
    *, cases: int, outdir: Path, seed: int, split: str
) -> dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sampled = _sample(_rows(split), cases, seed)

    case_rows: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    annotation_rows: list[dict[str, Any]] = []
    rag_rows: list[dict[str, Any]] = []

    for index, row in enumerate(sampled, 1):
        cid = f"v{index:04d}"
        gold = LABELS.get(str(row["label"]))
        if gold is None:
            raise TemplateError(f"unknown label {row['label']!r}")
        case_rows.append(
            {
                "id": cid,
                "image": str((DATASET / row["image_path"]).relative_to(ROOT)).replace("\\", "/"),
                "caption": row["caption"],
                "gold_label": gold,
                "mismatch_type": None,
                "strata": {
                    "caption_id": row["caption_id"],
                    "image_id": row["image_id"],
                    "source": None,
                    "period": None,
                },
            }
        )
        for variant in AUTHORED_VARIANTS:
            variant_rows.append(
                {
                    "id": cid,
                    "variant": variant,
                    "caption": None,
                    "gold_label": None,
                    "author": None,
                }
            )
        if gold == "MISMATCHED":
            annotation_rows.append(
                {"id": cid, "mismatch_type": None, "annotator": None}
            )
        rag_rows.append(
            {
                "id": cid,
                "image": str((DATASET / row["image_path"]).relative_to(ROOT)).replace("\\", "/"),
                "claim": row["caption"],
                "gold_label": gold,
                "gold_outcome": None,
                "gold_evidence_ids": [],
                "non_rag_outcome": None,
                "split": split,
            }
        )

    outputs = {
        "study1_cases": outdir / "study1_cases.jsonl",
        "study1_variants": outdir / "study1_variants.jsonl",
        "study1_annotations": outdir / "study1_annotations.jsonl",
        "study4_manifest": outdir / "study4_manifest.jsonl",
    }
    _jsonl(outputs["study1_cases"], case_rows)
    _jsonl(outputs["study1_variants"], variant_rows)
    _jsonl(outputs["study1_annotations"], annotation_rows)
    _jsonl(outputs["study4_manifest"], rag_rows)

    readme = outdir / "README.md"
    readme.write_text(
        "# Research data templates\n\n"
        f"Generated from `{split}.csv` with seed {seed} ({len(sampled)} cases).\n"
        "The held-out test split is never sampled.\n\n"
        "| File | Fill in |\n|---|---|\n"
        "| `study1_cases.jsonl` | `mismatch_type`; `strata.source` and `strata.period` where known |\n"
        "| `study1_variants.jsonl` | `caption`, `gold_label`, `author` for V1, V3, V4, V6 |\n"
        "| `study1_annotations.jsonl` | `mismatch_type`, `annotator` |\n"
        "| `study4_manifest.jsonl` | `gold_outcome`, `gold_evidence_ids`, `non_rag_outcome` |\n\n"
        "V2 (shorten) and V5 (negate) are generated by the audit driver — do not author them.\n",
        encoding="utf-8",
    )

    return {
        "cases": len(sampled),
        "variants": len(variant_rows),
        "annotations": len(annotation_rows),
        "outputs": {k: str(v) for k, v in outputs.items()},
        "readme": str(readme),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=300)
    parser.add_argument("--outdir", type=Path, default=ROOT / "data" / "research")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--from-split", default="val", choices=("val", "train"))
    args = parser.parse_args(argv)
    try:
        summary = create_templates(
            cases=args.cases, outdir=args.outdir, seed=args.seed, split=args.from_split
        )
    except TemplateError as exc:
        print(f"Template error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
