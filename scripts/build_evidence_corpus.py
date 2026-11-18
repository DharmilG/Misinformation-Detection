"""Build a closed-world evidence corpus from consistent dataset captions.

For every image in the validation split that has a consistent image-caption
pair, the real caption is a genuine statement about what the image depicts.  It
can therefore serve as evidence when a different caption is attached to the
same image.  This is a *closed-world* corpus: it contains only real dataset
captions, not web text, and its scope must be stated in any report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SPLIT_FILES = {"train", "val"}


def build(split: str, out: Path) -> dict[str, Any]:
    if split not in SPLIT_FILES:
        raise ValueError(f"split must be one of {sorted(SPLIT_FILES)}")
    source = ROOT / "dataset" / "ooc_dataset" / f"{split}.csv"
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    seen: set[str] = set()
    passages: list[dict[str, str]] = []
    for row in rows:
        if str(row.get("label")) != "0":
            continue  # only consistent pairs are evidence
        image_id = str(row["image_id"])
        caption = str(row["caption"]).strip()
        if not image_id or not caption or image_id in seen:
            continue
        seen.add(image_id)
        passages.append(
            {
                "passage_id": image_id,
                "title": caption[:120],
                "source_name": f"NewsCLIPpings {split} consistent pairs",
                "source_type": "dataset ground-truth caption",
                "published_at": None,
                "excerpt": caption,
                "url": None,
                "rights_wording": "Dataset caption; research evaluation only.",
                "storage_scope": "excerpt",
            }
        )
    passages.sort(key=lambda p: p["passage_id"])

    payload = json.dumps(passages, ensure_ascii=False, sort_keys=True).encode("utf-8")
    meta = {
        "scope": "closed-world dataset captions",
        "split": split,
        "source_file": str(source.relative_to(ROOT)).replace("\\", "/"),
        "passage_count": len(passages),
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(passages, indent=2, ensure_ascii=False), encoding="utf-8")
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {
        "passages": len(passages),
        "output": str(out),
        "meta": str(meta_path),
        "manifest_sha256": meta["manifest_sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val", choices=sorted(SPLIT_FILES))
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "research" / "evidence_corpus.json")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(build(args.split, args.out), indent=2))
    except ValueError as exc:
        print(f"Corpus error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
