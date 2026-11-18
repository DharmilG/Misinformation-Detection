"""Fill the non-RAG arm of the paired manifest with real classifier predictions.

The baseline outcome is derived from the classifier's own predicted label on
the claim, with no evidence.  Gold labels are never copied into this field.
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

from src.evaluation.research_runtime import (  # noqa: E402
    load_research_runtime,
    score_pair,
)


def _outcome(predicted_label: int) -> str:
    """1 = MISMATCHED -> CONTRADICTED; 0 = MATCHED -> SUPPORTED."""
    return "CONTRADICTED" if int(predicted_label) == 1 else "SUPPORTED"


def build(
    manifest: Path, checkpoint: str | Path, temperature: float, provenance: str, out: Path
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with Path(manifest).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    runtime = load_research_runtime(
        checkpoint, temperature=temperature, temperature_provenance=provenance
    )

    for row in rows:
        image = row.get("image")
        claim = str(row["claim"])
        if not image:
            raise ValueError("manifest rows need an image for a real baseline prediction")
        scores = score_pair(runtime, str(ROOT / image), claim)
        predicted = int(scores["predicted_label"])
        row["non_rag_outcome"] = _outcome(predicted)
        row["non_rag_source"] = "classifier_no_evidence"

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta = {
        "source": str(manifest),
        "output": str(out),
        "mapping": {"1": "CONTRADICTED", "0": "SUPPORTED"},
        "provenance": runtime.as_provenance(),
    }
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"rows": len(rows), "output": str(out), "meta": str(meta_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--temperature-provenance", default="supplied")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        summary = build(
            Path(args.manifest), args.checkpoint, args.temperature,
            args.temperature_provenance, Path(args.output),
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Manifest error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
