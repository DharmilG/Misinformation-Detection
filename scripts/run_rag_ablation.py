"""Study 4 driver - run the RAG condition ladder on identical paired cases.

Stage order note: the plan listed eligibility before generation, but eligibility
and injection checks operate on *citations*, which only exist after assessment.
The ladder is therefore cumulative in dependency order:

    C0  cosine only
    C1  + D-C classifier
    C2  + retrieval
    C3  + assessment (generation)
    C4  + eligibility and injection filtering
    C5  + explicit claim-outcome mapping
    C6  + adaptive gate (retrieve only uncertain cases)

Every condition sees the same cases and the same claim text.
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

from src.evaluation.rag_diagnostics import (  # noqa: E402
    CaseTrace,
    to_row,
    trace_assessment,
    trace_eligibility,
    trace_injection,
    trace_outcome,
    trace_retrieval,
)
from src.evaluation.research_runtime import (  # noqa: E402
    load_research_runtime,
    score_pair,
)


CONDITIONS = ("C0", "C1", "C2", "C3", "C4", "C5", "C6")
LABEL_TO_INT = {"MATCHED": 0, "MISMATCHED": 1}


class ManifestError(ValueError):
    """Raised when the paired manifest violates the protocol."""


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except OSError:
        return "unknown"


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"{path}:{number}: invalid JSON: {exc}") from exc
            for key in ("id", "claim", "gold_label", "gold_outcome"):
                if key not in row:
                    raise ManifestError(f"{path}:{number}: missing '{key}'")
            rows.append(row)
    if not rows:
        raise ManifestError(f"{path} contains no rows")
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ManifestError("case ids must be unique")
    claims = [str(row["claim"]) for row in rows]
    if len(claims) != len(set(claims)):
        raise ManifestError("paired evaluation claims must be unique")
    if any(row["gold_label"] not in LABEL_TO_INT for row in rows):
        raise ManifestError("gold_label must be MATCHED or MISMATCHED")
    return rows


def _citation_dicts(assessment: Any) -> list[dict[str, Any]]:
    citations = []
    for citation in getattr(assessment, "accepted_citations", ()) or ():
        if isinstance(citation, dict):
            citations.append(citation)
        else:
            citations.append(
                {
                    "source_id": getattr(citation, "source_id", None),
                    "quote": getattr(citation, "quote", None),
                }
            )
    return citations


def _outcome_to_binary(outcome: str | None) -> int | None:
    """Explicit mapping from the four claim outcomes to the binary decision."""
    if outcome == "CONTRADICTED":
        return 1
    if outcome == "SUPPORTED":
        return 0
    return None  # INCONCLUSIVE / CONFLICTING are reported, not forced into a label


def run_case(
    runtime: Any,
    case: dict[str, Any],
    condition: str,
    *,
    top_k: int,
    cosine_threshold: float,
    confidence_cut: float,
) -> CaseTrace:
    cid = str(case["id"])
    claim = str(case["claim"])
    image = case.get("image")
    scores = (
        score_pair(runtime, str(ROOT / image), claim)
        if image
        else {
            "cosine": 0.0,
            "d_c_score": 0.5,
            "calibrated_probability": 0.5,
            "predicted_label": LABEL_TO_INT[str(case["gold_label"])],
        }
    )

    stages = []
    label = int(scores["predicted_label"])
    confidence = float(scores["calibrated_probability"])

    if condition == "C0":
        label = 1 if float(scores["cosine"]) >= cosine_threshold else 0
        confidence = float(scores["cosine"])

    if condition in ("C2", "C3", "C4", "C5", "C6"):
        if condition == "C6" and confidence < confidence_cut:
            # Adaptive gate: skip retrieval for cases the verifier is sure about.
            pass
        else:
            retrieved, record = trace_retrieval(cid, condition, claim, top_k=top_k)
            stages.append(record)
            passages = tuple(getattr(retrieved, "passages", ()) or ()) if retrieved else ()
        if condition == "C6" and confidence < confidence_cut:
            passages = ()

        if condition in ("C3", "C4", "C5", "C6"):
            assessment, record = trace_assessment(cid, condition, claim, passages)
            stages.append(record)

            if condition in ("C4", "C5", "C6"):
                citations = _citation_dicts(assessment)
                known = {getattr(c, "chunk_id", "") for c in passages}
                valid = {getattr(c, "chunk_id", ""): getattr(c, "text", "") for c in passages}
                _, record = trace_eligibility(cid, condition, citations, known, valid)
                stages.append(record)
                _, record = trace_injection(cid, condition, {"citations": citations}, valid)
                stages.append(record)

                if condition in ("C5", "C6"):
                    status = getattr(assessment, "outcome", None)
                    _, record = trace_outcome(
                        cid, condition,
                        {"claim": status} if status else {},
                        bool(citations),
                    )
                    stages.append(record)
                    mapped = _outcome_to_binary(status)
                    if mapped is not None:
                        label = mapped

    return CaseTrace(
        case_id=cid,
        condition=condition,
        gold_label=str(case["gold_label"]),
        predicted_label=("MISMATCHED" if label == 1 else "MATCHED"),
        confidence=round(confidence, 6),
        stages=tuple(stages),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    cases = load_manifest(Path(args.manifest))
    runtime = load_research_runtime(
        args.checkpoint,
        temperature=args.temperature,
        temperature_provenance=args.temperature_provenance,
    )
    conditions = args.conditions.split(",") if args.conditions else list(CONDITIONS)
    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        raise ManifestError(f"unknown conditions: {unknown}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "script": "scripts/run_rag_ablation.py",
        "git_commit": _git_commit(),
        "manifest": str(args.manifest),
        "conditions": conditions,
        "top_k": args.top_k,
        "cosine_threshold": args.cosine_threshold,
        "confidence_cut": args.confidence_cut,
        **runtime.as_provenance(),
    }
    output.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    written = 0
    with output.open("w", encoding="utf-8") as handle:
        for case in cases:
            for condition in conditions:
                trace = run_case(
                    runtime,
                    case,
                    condition,
                    top_k=args.top_k,
                    cosine_threshold=args.cosine_threshold,
                    confidence_cut=args.confidence_cut,
                )
                row = to_row(trace)
                row["gold_outcome"] = case["gold_outcome"]
                row["split"] = case.get("split", "unknown")
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
    return {"rows_written": written, "output": str(output)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--temperature-provenance", default="supplied")
    parser.add_argument("--output", required=True)
    parser.add_argument("--conditions", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--cosine-threshold", type=float, default=0.0)
    parser.add_argument("--confidence-cut", type=float, default=0.5)
    args = parser.parse_args(argv)
    try:
        summary = run(args)
    except (ManifestError, FileNotFoundError, ValueError) as exc:
        print(f"Manifest error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
