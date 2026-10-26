"""F7 - the dataset-scale feature extraction of document 02 §14.

The pipeline extracts a whole split, in manifest row order, with the source
identifiers saved alongside the vectors so the rows line up with the corpus's
CSVs.  None of that exists yet, so every case here is a known gap.
"""

import json
from pathlib import Path

import pytest

from tests.ids import known_gap

CACHE_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "features"


@known_gap(
    "pipeline.module_exists",
    "F7",
    "fast",
    "doc/02 §14",
    "The project provides a dataset feature-extraction module.",
    "there is no extraction module",
)
def test_pipeline_module_exists():
    import importlib

    importlib.import_module("extraction")


@known_gap(
    "pipeline.extract_full_split",
    "F7",
    "data",
    "doc/02 §14.3",
    "A whole split is extracted, and the number of extracted rows matches the manifest.",
    "the extraction is hard-coded to a single sample image",
)
def test_extract_full_split(dataset_root, tmp_path):
    from extraction import extract_split

    rows = extract_split(dataset_root, "val", output=tmp_path)
    assert len(rows) == 7024, "the val split produced %d rows" % len(rows)


@known_gap(
    "pipeline.alignment_preserved",
    "F7",
    "data",
    "doc/02 §14.3",
    "The caption_id and image_id arrays are saved alongside the vectors, so the rows line up with the CSVs by construction.",
    "the identifiers are not saved, so the alignment can only be assumed",
)
def test_alignment_preserved(dataset_root, tmp_path):
    import csv

    from extraction import extract_split

    extract_split(dataset_root, "val", output=tmp_path)
    with open(dataset_root / "val.csv", newline="", encoding="utf-8") as fh:
        expected = [row["caption_id"] for row in csv.DictReader(fh)]

    saved = json.loads((tmp_path / "ids.json").read_text(encoding="utf-8"))
    assert saved["caption_ids"] == expected, "the saved caption ids do not match the CSV"


@known_gap(
    "pipeline.deterministic",
    "F7",
    "data",
    "doc/01 §14.2",
    "Extracting the same split twice produces identical features.",
    "there is no extraction pipeline to run twice",
)
def test_deterministic(dataset_root, tmp_path):
    import numpy as np

    from extraction import extract_split

    first = extract_split(dataset_root, "val", output=tmp_path / "a")
    second = extract_split(dataset_root, "val", output=tmp_path / "b")
    a = np.load(tmp_path / "a" / "image_embeddings.npy")
    b = np.load(tmp_path / "b" / "image_embeddings.npy")
    assert np.array_equal(a, b), "two extractions of the same split differ"


@known_gap(
    "pipeline.cache_key_carries_manifest_hash",
    "F7",
    "data",
    "doc/01 §11.3",
    "The cache key includes the source manifest hash, so a changed corpus invalidates the cache.",
    "there is no cache key",
)
def test_cache_key_carries_manifest_hash(dataset_root, tmp_path):
    from extraction import extract_split

    extract_split(dataset_root, "val", output=tmp_path)
    meta = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert meta.get("source_manifest_sha256"), "the cache metadata does not record the manifest hash"


@known_gap(
    "pipeline.no_pickle_in_cache",
    "F7",
    "data",
    "doc/01 §11.2",
    "The cache loads with allow_pickle=False.",
    "there is no cache",
)
def test_no_pickle_in_cache(dataset_root, tmp_path):
    import numpy as np

    from extraction import extract_split

    extract_split(dataset_root, "val", output=tmp_path)
    with np.load(tmp_path / "image_embeddings.npy", allow_pickle=False) as array:
        assert array.shape == (7024, 512)
