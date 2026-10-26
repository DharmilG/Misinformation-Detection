"""F10 - the end-to-end case manifest specification of Document 07 §8.

Verifies the schema, purpose categories, degradation matrix coverage, and gold
label rules for data/eval/e2e_cases.jsonl.
"""

import json
from pathlib import Path
import pytest

from tests.ids import case, known_gap

ROOT = Path(__file__).resolve().parent.parent.parent
E2E_MANIFEST = ROOT / "data" / "eval" / "e2e_cases.jsonl"


@known_gap(
    "e2e_manifest.file_exists",
    "F10",
    "fast",
    "doc/07 §8.1",
    "The end-to-end case manifest exists at data/eval/e2e_cases.jsonl.",
    "data/eval/e2e_cases.jsonl has not yet been generated",
)
def test_e2e_manifest_file_exists():
    assert E2E_MANIFEST.is_file(), f"Expected manifest file at {E2E_MANIFEST}"


@known_gap(
    "e2e_manifest.record_schema",
    "F10",
    "fast",
    "doc/07 §8.2",
    "Each record contains case_id, purpose, input, expected_path, expected_status, expected_render, gold_outcome.",
    "e2e_cases manifest records are not yet present",
)
def test_e2e_manifest_record_schema():
    assert E2E_MANIFEST.is_file()
    required_keys = {
        "case_id",
        "purpose",
        "input",
        "expected_path",
        "expected_status",
        "expected_render",
        "gold_outcome",
    }
    with open(E2E_MANIFEST, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            record = json.loads(line)
            missing = required_keys - set(record.keys())
            assert not missing, f"Line {idx} missing fields: {missing}"


@known_gap(
    "e2e_manifest.purpose_categories",
    "F10",
    "fast",
    "doc/07 §8.2",
    "Case purpose must be functionality, retrieval_coverage, semantic_correctness, or failure_handling.",
    "e2e_cases manifest is not yet present",
)
def test_e2e_manifest_purpose_categories():
    assert E2E_MANIFEST.is_file()
    valid_purposes = {
        "functionality",
        "retrieval_coverage",
        "semantic_correctness",
        "failure_handling",
    }
    with open(E2E_MANIFEST, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            assert record["purpose"] in valid_purposes


@known_gap(
    "e2e_manifest.degradation_coverage",
    "F10",
    "fast",
    "doc/07 §8.3",
    "Every row of the degradation matrix in Document 07 §7.1 has at least one case in the manifest.",
    "degradation test cases are not yet populated in e2e manifest",
)
def test_e2e_manifest_degradation_coverage():
    assert E2E_MANIFEST.is_file()
    # At least these failure conditions from doc 07 §7.1:
    required_failures = {
        "empty_upload",
        "unsupported_format",
        "corrupt_image",
        "empty_caption",
        "overlong_caption",
        "encoder_absent",
        "checkpoint_absent",
        "index_missing",
        "empty_retrieval",
        "generator_failure",
    }
    covered = set()
    with open(E2E_MANIFEST, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("purpose") == "failure_handling":
                covered.add(rec.get("failure_condition"))
    missing = required_failures - covered
    assert not missing, f"Missing degradation coverage for: {missing}"


@known_gap(
    "e2e_manifest.gold_outcome_optionality",
    "F10",
    "fast",
    "doc/07 §8.3",
    "Not every case has a gold label; corrupt inputs test rejection and rendering, with gold_outcome as null.",
    "e2e manifest is not yet populated",
)
def test_e2e_manifest_gold_outcome_optionality():
    assert E2E_MANIFEST.is_file()
    has_null_gold = False
    with open(E2E_MANIFEST, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("gold_outcome") is None:
                has_null_gold = True
                break
    assert has_null_gold, "Expected failure handling cases with null gold_outcome"
