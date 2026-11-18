"""Fast conformance cases for the research template generator."""

import json
import pytest

from scripts.make_study_templates import (  # noqa: E402
    AUTHORED_VARIANTS,
    TemplateError,
    create_templates,
)
from tests.ids import case  # noqa: E402


def _read(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@case(
    "template.no_test_split_rows",
    "F10",
    "fast",
    "doc/07 §12.2",
    "The template generator refuses to sample the held-out test split.",
)
def test_no_test_split_rows(tmp_path):
    with pytest.raises(TemplateError, match="test split"):
        create_templates(cases=2, outdir=tmp_path, seed=1, split="test")


@case(
    "template.variants_prefilled",
    "F10",
    "fast",
    "doc/07 §12.2",
    "Each sampled case has exactly the four human-authored variant slots, all blank.",
)
def test_variants_prefilled(tmp_path):
    summary = create_templates(cases=4, outdir=tmp_path, seed=7, split="val")
    cases = _read(tmp_path / "study1_cases.jsonl")
    variants = _read(tmp_path / "study1_variants.jsonl")
    assert summary["variants"] == 4 * len(AUTHORED_VARIANTS)
    assert len(variants) == 4 * len(AUTHORED_VARIANTS)
    for case_row in cases:
        slots = [v for v in variants if v["id"] == case_row["id"]]
        assert tuple(v["variant"] for v in slots) == AUTHORED_VARIANTS
        assert all(v["caption"] is None and v["author"] is None for v in slots)


@case(
    "template.reproducible_seed",
    "F10",
    "fast",
    "doc/07 §12.2",
    "The same seed produces byte-identical templates.",
)
def test_reproducible_seed(tmp_path):
    first = tmp_path / "a"
    second = tmp_path / "b"
    create_templates(cases=6, outdir=first, seed=42, split="val")
    create_templates(cases=6, outdir=second, seed=42, split="val")
    for name in ("study1_cases.jsonl", "study1_variants.jsonl", "study4_manifest.jsonl"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
