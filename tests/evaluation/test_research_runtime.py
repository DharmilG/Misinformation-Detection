"""Fast conformance cases for the Study 1 driver's input rules."""

import pytest

from scripts.run_failure_audit import (  # noqa: E402
    AuditInputError,
    LABEL_TO_INT,
    load_inputs,
    mechanical_variant,
)
from tests.ids import case  # noqa: E402


@case(
    "audit.mechanical_variants_labels",
    "F10",
    "fast",
    "doc/07 §12.2",
    "Mechanically generated variants preserve or flip the gold label by construction.",
)
def test_mechanical_variants_labels():
    short, short_label = mechanical_variant("V2", "A flooded street, with cars", "MISMATCHED")
    neg, neg_label = mechanical_variant("V5", "A red car", "MATCHED")
    assert short == "A flooded street"
    assert short_label == "MISMATCHED"
    assert neg.startswith("It is not true that")
    assert neg_label == "MISMATCHED"  # flipped from MATCHED
    assert LABEL_TO_INT["MATCHED"] != LABEL_TO_INT["MISMATCHED"]


@case(
    "audit.rejects_missing_gold",
    "F10",
    "fast",
    "doc/07 §12.2",
    "An audit case without a gold label is rejected before any inference.",
)
def test_rejects_missing_gold(tmp_path):
    manifest = tmp_path / "cases.jsonl"
    manifest.write_text(
        '{"id":"c1","image":"data/sample/test_3.webp","caption":"x"}\n', encoding="utf-8"
    )
    with pytest.raises(AuditInputError, match="gold_label"):
        load_inputs(manifest, None)


@case(
    "audit.rejects_authored_mechanical",
    "F10",
    "fast",
    "doc/07 §12.2",
    "A manually authored mechanical variant is rejected as a contract violation.",
)
def test_rejects_authored_mechanical(tmp_path):
    manifest = tmp_path / "cases.jsonl"
    manifest.write_text(
        '{"id":"c1","image":"data/sample/test_3.webp","caption":"x","gold_label":"MATCHED"}\n',
        encoding="utf-8",
    )
    variants = tmp_path / "variants.jsonl"
    variants.write_text(
        '{"id":"c1","variant":"V5","caption":"not x","gold_label":"MISMATCHED"}\n',
        encoding="utf-8",
    )
    with pytest.raises(AuditInputError, match="mechanical"):
        load_inputs(manifest, variants)


@case(
    "audit.accepts_bom_manifest",
    "F10",
    "fast",
    "doc/07 §12.2",
    "A UTF-8 manifest with a BOM is accepted, as produced by Windows editors.",
)
def test_accepts_bom_manifest(tmp_path):
    manifest = tmp_path / "cases.jsonl"
    manifest.write_bytes(
        b'\xef\xbb\xbf{"id":"c1","image":"data/sample/test_3.webp",'
        b'"caption":"x","gold_label":"MATCHED"}\n'
    )
    cases, authored = load_inputs(manifest, None)
    assert len(cases) == 1 and authored == {}
