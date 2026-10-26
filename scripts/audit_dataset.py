"""Audit the local corpus.

    python scripts/audit_dataset.py [--dataset-root PATH] [--report PATH]

Runs the checks of document 02 §5, verifies the split structure, computes the
shortcut ceilings of §13.1, writes ``reports/dataset_audit.json``, and exits
non-zero when a check fails, so a corrupt or substituted corpus cannot be
used unnoticed.

The corpus is the local copy at ``dataset/ooc_dataset/``.  Nothing is written
outside ``reports/``.
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("train", "val", "test")
EXPECTED_ROWS = {"train": 25000, "val": 7024, "test": 7264}


def read_rows(root, split):
    with open(root / ("%s.csv" % split), newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def check(name, ok, detail, failures):
    status = "pass" if ok else "fail"
    if not ok:
        failures.append("%s: %s" % (name, detail))
    return {"check": name, "status": status, "detail": detail}


def audit(root):
    failures = []
    results = []
    rows = {split: read_rows(root, split) for split in SPLITS}

    # 1. schema
    expected = ["image_path", "caption", "label", "caption_id", "image_id"]
    schema_ok = all(list(rows[s][0].keys()) == expected for s in SPLITS)
    results.append(
        check("schema", schema_ok, "columns %r" % (list(rows["train"][0].keys()),), failures)
    )

    # 2. the labels.json rule
    labels_path = root / "labels.json"
    rule = {}
    if labels_path.is_file():
        try:
            rule = json.loads(labels_path.read_text(encoding="utf-8"))
        except Exception as exc:
            rule = {"error": str(exc)}
    observed = sorted({row["label"] for split in SPLITS for row in rows[split]})
    rule_ok = (
        rule.get("0") == "consistent"
        and rule.get("1") == "out_of_context"
        and "falsified" in str(rule.get("rule", ""))
        and set(observed) <= {"0", "1"}
    )
    results.append(
        check("labels_json_rule", rule_ok, "rule %r, observed labels %r" % (rule, observed), failures)
    )

    # 3. row counts and balance
    counts_ok = all(len(rows[s]) == EXPECTED_ROWS[s] for s in SPLITS)
    results.append(
        check(
            "row_counts",
            counts_ok,
            {s: len(rows[s]) for s in SPLITS},
            failures,
        )
    )
    balance = {}
    balance_ok = True
    for split in SPLITS:
        c = Counter(r["label"] for r in rows[split])
        balance[split] = dict(c)
        if not (c["0"] * 2 == len(rows[split]) and c["1"] * 2 == len(rows[split])):
            balance_ok = False
    results.append(check("class_balance", balance_ok, balance, failures))

    # 4. pair structure
    pair_report = {}
    pair_ok = True
    for split in SPLITS:
        groups = defaultdict(list)
        for row in rows[split]:
            groups[row["caption_id"]].append(row["label"])
        bad = [k for k, v in groups.items() if len(v) != 2 or sorted(v) != ["0", "1"]]
        pair_report[split] = {"groups": len(groups), "malformed": len(bad)}
        if bad:
            pair_ok = False
    results.append(check("pair_structure", pair_ok, pair_report, failures))

    # 5. label 0 rows carry their own photograph
    mismatch = 0
    for split in SPLITS:
        for row in rows[split]:
            if row["label"] == "0" and row["caption_id"] != row["image_id"]:
                mismatch += 1
    results.append(check("label0_caption_eq_image", mismatch == 0, mismatch, failures))

    # 6 and 7. no overlap across splits.  A caption id repeats *within* a
    # split - it is the pair key - so only the pairwise intersections matter.
    ids = {
        field: {split: {row[field] for row in rows[split]} for split in SPLITS}
        for field in ("caption_id", "image_id")
    }
    overlap = {}
    for field, per_split in ids.items():
        shared = []
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            shared.extend(sorted(per_split[a] & per_split[b]))
        overlap[field] = shared
    overlap_ok = not any(overlap.values())
    results.append(
        check(
            "cross_split_overlap",
            overlap_ok,
            {k: len(v) for k, v in overlap.items()},
            failures,
        )
    )

    # 8. every referenced photograph exists
    missing = 0
    for split in SPLITS:
        for row in rows[split]:
            if not (root / row["image_path"]).is_file():
                missing += 1
    results.append(check("images_exist", missing == 0, missing, failures))

    # 9. one caption text and one path per id
    integrity = {"caption_texts": 0, "image_paths": 0}
    for split in SPLITS:
        texts, paths = defaultdict(set), defaultdict(set)
        for row in rows[split]:
            texts[row["caption_id"]].add(row["caption"])
            paths[row["image_id"]].add(row["image_path"])
        integrity["caption_texts"] += sum(1 for v in texts.values() if len(v) > 1)
        integrity["image_paths"] += sum(1 for v in paths.values() if len(v) > 1)
    integrity_ok = integrity["caption_texts"] == 0 and integrity["image_paths"] == 0
    results.append(check("referential_integrity", integrity_ok, integrity, failures))

    # 10. geometry and caption length
    geometry = {"shorter_side_min": None, "shorter_side_max": None, "sampled": 0}
    longest_caption = 0
    for split in SPLITS:
        for row in rows[split]:
            longest_caption = max(longest_caption, len(row["caption"].split()))
    geometry["longest_caption_words"] = longest_caption
    try:
        from PIL import Image

        for split in SPLITS:
            for row in rows[split][:60]:
                with Image.open(root / row["image_path"]) as image:
                    side = min(image.width, image.height)
                if geometry["shorter_side_min"] is None:
                    geometry["shorter_side_min"] = side
                    geometry["shorter_side_max"] = side
                else:
                    geometry["shorter_side_min"] = min(geometry["shorter_side_min"], side)
                    geometry["shorter_side_max"] = max(geometry["shorter_side_max"], side)
                geometry["sampled"] += 1
    except Exception as exc:
        geometry["error"] = "Pillow is unavailable: %s" % exc
    geometry_ok = (
        geometry.get("shorter_side_min") is not None
        and abs(geometry["shorter_side_min"] - 256) <= 1
        and abs(geometry["shorter_side_max"] - 256) <= 1
        and longest_caption <= 77
    )
    results.append(check("geometry_and_caption_length", geometry_ok, geometry, failures))

    # 11. the shortcut ceilings
    ceilings = {}
    for split in SPLITS:
        by_caption = defaultdict(list)
        by_image = defaultdict(list)
        for row in rows[split]:
            by_caption[row["caption_id"]].append(row["label"])
            by_image[row["image_id"]].append(row["label"])

        # A predictor that sees only the caption does best by guessing the
        # majority label within each caption group.
        caption_correct = sum(
            max(labels.count("0"), labels.count("1")) for labels in by_caption.values()
        )
        # A predictor that sees only the photograph does best by guessing the
        # label it saw with that photograph.
        image_correct = 0
        for labels in by_image.values():
            image_correct += (
                len(labels) if len(set(labels)) == 1 else len(labels) // 2
            )
        both = sum(1 for labels in by_caption.values() if len(set(labels)) == 2)
        ceilings[split] = {
            "caption_only_ceiling": round(caption_correct / len(rows[split]), 6),
            "caption_ids_with_both_labels": both,
            "image_only_in_split_ceiling": round(image_correct / len(rows[split]), 6),
        }
    results.append(check("shortcut_ceilings", True, ceilings, failures))

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_root": str(root),
        "verdict": "pass" if not failures else "fail",
        "failures": failures,
        "checks": results,
        "shortcut_ceilings": ceilings,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default=os.environ.get("OOC_DATASET_ROOT", str(ROOT / "dataset" / "ooc_dataset")),
    )
    parser.add_argument("--report", default=str(ROOT / "reports" / "dataset_audit.json"))
    args = parser.parse_args(argv)

    root = Path(args.dataset_root)
    if not root.is_dir():
        print("the corpus is not present at %s" % root, file=sys.stderr)
        return 2

    report = audit(root)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for entry in report["checks"]:
        mark = "ok  " if entry["status"] == "pass" else "FAIL"
        print("%s %-32s %s" % (mark, entry["check"], entry["detail"]))
    print("\nreport: %s" % report_path)
    print("verdict: %s" % report["verdict"])
    return 0 if report["verdict"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
