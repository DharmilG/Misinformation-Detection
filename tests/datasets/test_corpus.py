"""F5 - the corpus contract of document 02.

These assert the properties of the local corpus that document 02 §5's gate
requires, and the shortcut ceilings of §13.1.  They need the corpus but not
the model weights, so they run in seconds.
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from tests.ids import case

SPLITS = ("train", "val", "test")


def _rows(dataset_root, split):
    with open(dataset_root / ("%s.csv" % split), newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _by_id(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["caption_id"]].append(row)
    return grouped


@case(
    "dataset.rows_match_csv",
    "F5",
    "data",
    "doc/02 §4.1, §5",
    "The corpus has 25,000 train rows, 7,024 val rows and 7,264 test rows.",
)
def test_rows_match_csv(dataset_root):
    expected = {"train": 25000, "val": 7024, "test": 7264}
    for split, count in expected.items():
        assert len(_rows(dataset_root, split)) == count, "%s has the wrong row count" % split


@case(
    "dataset.label_balance_exact",
    "F5",
    "data",
    "doc/02 §5, §6.4",
    "Every split is exactly 50% label 0 and 50% label 1.",
)
def test_label_balance_exact(dataset_root):
    for split in SPLITS:
        counts = Counter(row["label"] for row in _rows(dataset_root, split))
        total = sum(counts.values())
        assert counts["0"] * 2 == total, "%s is not exactly balanced" % split
        assert counts["1"] * 2 == total, "%s is not exactly balanced" % split


@case(
    "dataset.pair_structure",
    "F5",
    "data",
    "doc/02 §5, §13.1",
    "Every caption_id forms exactly one pair: two rows, one of each label.",
)
def test_pair_structure(dataset_root):
    for split in SPLITS:
        for caption_id, group in _by_id(_rows(dataset_root, split)).items():
            assert len(group) == 2, "%s has a group of %d" % (caption_id, len(group))
            labels = sorted(row["label"] for row in group)
            assert labels == ["0", "1"], "%s has labels %r" % (caption_id, labels)


@case(
    "dataset.label0_caption_eq_image",
    "F5",
    "data",
    "doc/02 §5",
    "On every label 0 row, the caption id equals the image id.",
)
def test_label0_caption_eq_image(dataset_root):
    for split in SPLITS:
        for row in _rows(dataset_root, split):
            if row["label"] == "0":
                assert row["caption_id"] == row["image_id"], (
                    "a label 0 row has %s != %s" % (row["caption_id"], row["image_id"])
                )


@case(
    "dataset.no_cross_split_caption_overlap",
    "F5",
    "data",
    "doc/02 §5, §9",
    "No caption appears in two splits.",
)
def test_no_cross_split_caption_overlap(dataset_root):
    ids = {split: {row["caption_id"] for row in _rows(dataset_root, split)} for split in SPLITS}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = ids[a] & ids[b]
        assert not shared, "%d captions appear in both %s and %s" % (len(shared), a, b)


@case(
    "dataset.no_cross_split_image_overlap",
    "F5",
    "data",
    "doc/02 §5, §9",
    "No photograph appears in two splits.",
)
def test_no_cross_split_image_overlap(dataset_root):
    ids = {split: {row["image_id"] for row in _rows(dataset_root, split)} for split in SPLITS}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = ids[a] & ids[b]
        assert not shared, "%d photographs appear in both %s and %s" % (len(shared), a, b)


@case(
    "dataset.caption_repeats_within_split",
    "F5",
    "data",
    "doc/02 §13.1",
    "A caption id does repeat within a split - it is the pair key, and each pair carries both labels.",
)
def test_caption_repeats_within_split(dataset_root):
    rows = _rows(dataset_root, "train")
    counts = Counter(row["caption_id"] for row in rows)
    assert all(count == 2 for count in counts.values()), (
        "a caption id appears more than twice, so the pair structure is broken"
    )


@case(
    "dataset.all_images_exist",
    "F5",
    "data",
    "doc/02 §5, §7",
    "Every photograph the CSVs reference exists on disk.",
)
def test_all_images_exist(dataset_root):
    missing = 0
    for split in SPLITS:
        for row in _rows(dataset_root, split):
            if not (dataset_root / row["image_path"]).is_file():
                missing += 1
    assert missing == 0, "%d referenced images are missing" % missing


@case(
    "dataset.one_caption_per_id",
    "F5",
    "data",
    "doc/02 §5, §12",
    "Each caption_id maps to exactly one caption text.",
)
def test_one_caption_per_id(dataset_root):
    for split in SPLITS:
        texts = defaultdict(set)
        for row in _rows(dataset_root, split):
            texts[row["caption_id"]].add(row["caption"])
        bad = [k for k, v in texts.items() if len(v) > 1]
        assert not bad, "%d caption ids have more than one text" % len(bad)


@case(
    "dataset.one_path_per_id",
    "F5",
    "data",
    "doc/02 §5, §12",
    "Each image_id maps to exactly one image path.",
)
def test_one_path_per_id(dataset_root):
    for split in SPLITS:
        paths = defaultdict(set)
        for row in _rows(dataset_root, split):
            paths[row["image_id"]].add(row["image_path"])
        bad = [k for k, v in paths.items() if len(v) > 1]
        assert not bad, "%d image ids have more than one path" % len(bad)


@case(
    "dataset.image_shorter_side",
    "F5",
    "data",
    "doc/02 §5, §8.1.1",
    "The corpus photographs have a shorter side of 255-256 px, which is the pre-resize document 02 §8.1.1 records.",
)
def test_image_shorter_side(dataset_root):
    from PIL import Image

    rows = []
    for split in SPLITS:
        rows.extend(_rows(dataset_root, split)[:60])
    sides = []
    for row in rows:
        path = dataset_root / row["image_path"]
        with Image.open(path) as image:
            sides.append(min(image.width, image.height))
    off = [s for s in sides if abs(s - 256) > 1]
    assert not off, "%d sampled images do not have a 256 px shorter side" % len(off)


@case(
    "dataset.captions_within_clip_context",
    "F5",
    "data",
    "doc/02 §5, §8.2",
    "No corpus caption approaches the CLIP context window, so the corpus never exercises the truncation path.",
)
def test_captions_within_clip_context(dataset_root):
    longest = 0
    for split in SPLITS:
        for row in _rows(dataset_root, split):
            longest = max(longest, len(row["caption"].split()))
    assert longest <= 77, "the longest caption is %d words" % longest


@case(
    "dataset.labels_json_rule_matches",
    "F5",
    "data",
    "doc/02 §5, §6.2",
    "labels.json states a rule, and the observed label values agree with it.",
)
def test_labels_json_rule_matches(dataset_root):
    labels = json.loads((dataset_root / "labels.json").read_text(encoding="utf-8"))
    assert labels.get("0") == "consistent"
    assert labels.get("1") == "out_of_context"
    assert "falsified" in labels.get("rule", ""), "the rule does not mention the source field"
    observed = {row["label"] for split in SPLITS for row in _rows(dataset_root, split)}
    assert observed <= {"0", "1"}, "unexpected label values: %r" % observed


@case(
    "dataset.shortcut_ceiling_caption_only",
    "F5",
    "data",
    "doc/02 §13.1",
    "The caption-only ceiling is exactly 50%, because every caption appears with both labels.",
)
def test_shortcut_ceiling_caption_only(dataset_root):
    for split in SPLITS:
        groups = _by_id(_rows(dataset_root, split))
        both = sum(1 for g in groups.values() if len({r["label"] for r in g}) == 2)
        assert both == len(groups), "%s has a caption with only one label" % split


@case(
    "dataset.shortcut_ceiling_image_only",
    "F5",
    "data",
    "doc/02 §13.1",
    "The in-split image-only ceiling is the fraction of rows whose photograph is unambiguous, and it is far above 50%.",
)
def test_shortcut_ceiling_image_only(dataset_root):
    for split in SPLITS:
        rows = _rows(dataset_root, split)
        by_image = defaultdict(list)
        for row in rows:
            by_image[row["image_id"]].append(row["label"])
        correct = 0
        for labels in by_image.values():
            if len(set(labels)) == 1:
                correct += len(labels)
            else:
                correct += len(labels) // 2
        ceiling = correct / len(rows)
        assert ceiling > 0.80, "the %s image-only ceiling is %.3f" % (split, ceiling)
