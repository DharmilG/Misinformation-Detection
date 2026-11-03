"""Generate a paired RAG/non-RAG evaluation manifest from the live pipeline.

Selects a stratified sample from the test split, runs the full
classifier + retrieval + assessment pipeline, and writes a JSONL manifest
compatible with evaluate_rag_non_rag.py.

Usage:
    python scripts/generate_rag_eval_manifest.py [--n 50]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clip_model.classifier import run_trained_classifier
from src.retrieval.corpus import retrieve_passages
from src.rag.assessment import run_assessment


# Outcome mapping from ground-truth label
LABEL_TO_GOLD = {0: "SUPPORTED", 1: "CONTRADICTED"}

# Non-RAG outcome derived from classifier prediction alone (no evidence)
CLS_TO_NON_RAG = {"MISMATCHED": "CONTRADICTED", "MATCHED": "SUPPORTED"}


def stratified_sample(test_df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Select n cases with unique captions, balanced across classes.

    In NewsCLIPpings every caption appears twice (label=0 with matching image,
    label=1 with mismatched image).  We pick n/2 captions and take their
    label=0 row, and n/2 different captions and take their label=1 row.
    """
    per_class = n // 2
    # Get unique captions for each class
    matched = test_df[test_df["label"] == 0].drop_duplicates(subset=["caption"], keep="first")
    mismatched = test_df[test_df["label"] == 1].drop_duplicates(subset=["caption"], keep="first")
    # Pick disjoint caption sets
    matched_sample = matched.sample(n=min(per_class, len(matched)), random_state=42)
    used_captions = set(matched_sample["caption"])
    mismatched_pool = mismatched[~mismatched["caption"].isin(used_captions)]
    mismatched_sample = mismatched_pool.sample(n=min(per_class, len(mismatched_pool)), random_state=42)
    sampled = pd.concat([matched_sample, mismatched_sample]).reset_index(drop=True)
    return sampled


def process_case(row: pd.Series, dataset_root: Path, idx: int) -> dict | None:
    """Run the full pipeline for one test case and return a manifest row."""
    image_path = dataset_root / row["image_path"]
    caption = row["caption"]
    label = int(row["label"])

    if not image_path.is_file():
        return None

    # Read image bytes
    image_bytes = image_path.read_bytes()

    # 1. Classifier
    clf_result = run_trained_classifier(image_bytes, caption)
    if clf_result.status != "OK":
        non_rag_outcome = "INCONCLUSIVE"
    else:
        non_rag_outcome = CLS_TO_NON_RAG.get(clf_result.label, "INCONCLUSIVE")

    # 2. Retrieval
    retrieval_result = retrieve_passages(caption)

    # 3. Assessment
    assessment = run_assessment(caption, retrieval_result.passages)
    rag_outcome = assessment.outcome if assessment.outcome else "INCONCLUSIVE"

    # 4. Gold
    gold_outcome = LABEL_TO_GOLD.get(label, "INCONCLUSIVE")

    return {
        "id": f"eval_{idx:04d}",
        "claim": caption,
        "gold_outcome": gold_outcome,
        "non_rag_outcome": non_rag_outcome,
        "rag_outcome": rag_outcome,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50, help="Number of test cases")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "eval" / "rag_non_rag_manifest.jsonl")
    args = parser.parse_args()

    test_csv = ROOT / "dataset" / "ooc_dataset" / "test.csv"
    dataset_root = ROOT / "dataset" / "ooc_dataset"

    if not test_csv.is_file():
        print(f"ERROR: {test_csv} not found", file=sys.stderr)
        return 1

    print(f"Loading {test_csv} ...")
    test_df = pd.read_csv(test_csv)
    print(f"  Test set: {len(test_df)} rows")

    sampled = stratified_sample(test_df, args.n)
    print(f"  Stratified sample: {len(sampled)} rows "
          f"({(sampled['label']==0).sum()} matched, {(sampled['label']==1).sum()} mismatched)")

    rows = []
    for idx, (_, row) in enumerate(sampled.iterrows()):
        result = process_case(row, dataset_root, idx)
        if result:
            rows.append(result)
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(sampled)}...")

    # Write manifest
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
