"""The label mapping contract of Document 02 §6.

The project maps raw dataset annotations onto a unified three-class scheme:
MATCHED, MISMATCHED, and UNKNOWN. Source labels are always retained so
annotations remain traceable and re-derivable.
"""

import csv
from pathlib import Path
import pytest

from tests.ids import case, known_gap

SPLITS = ("train", "val", "test")


def _labels_module():
    import importlib
    return importlib.import_module("labels")


@known_gap(
    "labels.module_exists",
    "F5",
    "fast",
    "doc/02 §6.1",
    "The project provides a dataset labels module defining the Label enum and mapping logic.",
    "there is no labels module",
)
def test_labels_module_exists():
    _labels_module()


@known_gap(
    "labels.enum_values",
    "F5",
    "fast",
    "doc/02 §6.1",
    "The Label enum defines exactly MATCHED, MISMATCHED, and UNKNOWN.",
    "there is no Label enum",
)
def test_enum_values():
    mod = _labels_module()
    assert hasattr(mod, "Label"), "Label enum is not defined"
    Label = mod.Label
    assert {e.value for e in Label} == {"MATCHED", "MISMATCHED", "UNKNOWN"}


@known_gap(
    "labels.map_source_0",
    "F5",
    "fast",
    "doc/02 §6.2",
    "Source label 0 maps to MATCHED (consistent pairing).",
    "the label mapper is not implemented",
)
def test_map_source_0():
    mod = _labels_module()
    assert mod.map_label("0") == mod.Label.MATCHED
    assert mod.map_label(0) == mod.Label.MATCHED


@known_gap(
    "labels.map_source_1",
    "F5",
    "fast",
    "doc/02 §6.2",
    "Source label 1 maps to MISMATCHED (out-of-context pairing).",
    "the label mapper is not implemented",
)
def test_map_source_1():
    mod = _labels_module()
    assert mod.map_label("1") == mod.Label.MISMATCHED
    assert mod.map_label(1) == mod.Label.MISMATCHED


@known_gap(
    "labels.map_source_unknown",
    "F5",
    "fast",
    "doc/02 §6.2",
    "Empty, null, or unrecognised source values map conservatively to UNKNOWN without coercion.",
    "the label mapper is not implemented",
)
def test_map_source_unknown():
    mod = _labels_module()
    for bad in (None, "", "unknown", 2, -1, "corrupted"):
        assert mod.map_label(bad) == mod.Label.UNKNOWN


@known_gap(
    "labels.source_annotations_retained",
    "F5",
    "fast",
    "doc/02 §6.3",
    "The mapped record preserves original source_label and source_label_raw fields.",
    "the mapping output does not retain source annotations",
)
def test_source_annotations_retained():
    mod = _labels_module()
    record = mod.annotate_record({"caption_id": "1", "image_id": "1", "label": "0"})
    assert record["label"] == mod.Label.MATCHED
    assert record["source_label"] == "consistent"
    assert record["source_label_raw"] == 0


@case(
    "labels.corpus_values_decidable",
    "F5",
    "data",
    "doc/02 §5, §6.2",
    "Every observed row in the corpus has a decidable source label (0 or 1) with zero UNKNOWN outcomes.",
)
def test_corpus_values_decidable(dataset_root):
    for split in SPLITS:
        with open(dataset_root / ("%s.csv" % split), newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        labels = {row["label"] for row in rows}
        assert labels == {"0", "1"}, "%s split contained non-decidable label values: %r" % (split, labels)
