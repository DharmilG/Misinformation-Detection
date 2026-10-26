"""The split configurations and boundaries contract of Document 02 §9.

The project distinguishes three configurations:
- newsclippings-official-v1: the preserved release split files.
- newsclippings-grouped-v1: seeded grouping over leakage-safe components.
- dev configuration: 2,000/500/500 class-balanced fast iteration subsample.
"""

from pathlib import Path
import pytest

from tests.ids import case, known_gap


def _splits_module():
    import importlib
    return importlib.import_module("splits")


@known_gap(
    "splits.module_exists",
    "F5",
    "fast",
    "doc/02 §9",
    "The project provides a dataset splits configuration module.",
    "there is no splits module",
)
def test_splits_module_exists():
    _splits_module()


@known_gap(
    "splits.configuration_name",
    "F5",
    "fast",
    "doc/02 §9.3",
    "The official split configuration is identified as 'newsclippings-official-v1'.",
    "the split configuration constants are not defined",
)
def test_configuration_name():
    mod = _splits_module()
    assert mod.OFFICIAL_CONFIGURATION == "newsclippings-official-v1"


@known_gap(
    "splits.grouped_configuration_name",
    "F5",
    "fast",
    "doc/02 §9.3",
    "The grouped split configuration is identified as 'newsclippings-grouped-v1'.",
    "the split configuration constants are not defined",
)
def test_grouped_configuration_name():
    mod = _splits_module()
    assert mod.GROUPED_CONFIGURATION == "newsclippings-grouped-v1"


@known_gap(
    "splits.dev_configuration_targets",
    "F5",
    "fast",
    "doc/02 §9.4",
    "The dev configuration defines exact targets: 2,000 train, 500 val, and 500 test samples.",
    "the dev configuration is not defined",
)
def test_dev_configuration_targets():
    mod = _splits_module()
    targets = mod.DEV_TARGETS
    assert targets == {"train": 2000, "val": 500, "test": 500}


@known_gap(
    "splits.dev_sampled_from_official",
    "F5",
    "fast",
    "doc/02 §9.4",
    "The dev configuration is drawn strictly from the official configuration preserving class balance.",
    "the dev sampler is not implemented",
)
def test_dev_sampled_from_official():
    mod = _splits_module()
    mock_official = {
        "train": [{"sample_id": f"t_{i}", "label": "MATCHED" if i % 2 == 0 else "MISMATCHED"} for i in range(100)],
        "val": [{"sample_id": f"v_{i}", "label": "MATCHED" if i % 2 == 0 else "MISMATCHED"} for i in range(50)],
        "test": [{"sample_id": f"te_{i}", "label": "MATCHED" if i % 2 == 0 else "MISMATCHED"} for i in range(50)],
    }
    dev = mod.subsample_dev(mock_official, targets={"train": 20, "val": 10, "test": 10}, seed=42)
    for split, target in (("train", 20), ("val", 10), ("test", 10)):
        assert len(dev[split]) == target
        matched = sum(1 for r in dev[split] if r["label"] == "MATCHED")
        assert matched == target // 2, "%s split in dev subsample was not exactly balanced" % split
