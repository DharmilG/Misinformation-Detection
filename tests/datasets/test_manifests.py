"""The manifest format and record contract of Document 02 §12.

Manifests are JSONL files that bind source records to hashes, labels,
pair groups, and quarantine flags. Every record is immutable and verifiable.
"""

import json
from pathlib import Path
import pytest

from tests.ids import case, known_gap

REQUIRED_FIELDS = {
    "sample_id",
    "dataset",
    "dataset_version",
    "source_record_id",
    "source_file",
    "configuration",
    "split",
    "image_path",
    "image_sha256",
    "caption_raw",
    "caption",
    "caption_sha256",
    "label",
    "source_label",
    "source_label_raw",
    "source_extra",
    "caption_id",
    "image_id",
    "pair_group_id",
    "event_id",
    "flags",
    "manifest_version",
}


def _manifest_module():
    import importlib
    return importlib.import_module("manifests")


@known_gap(
    "manifest.module_exists",
    "F5",
    "fast",
    "doc/02 §12.1",
    "The project provides a manifest builder and validation module.",
    "there is no manifests module",
)
def test_manifest_module_exists():
    _manifest_module()


@known_gap(
    "manifest.record_schema",
    "F5",
    "fast",
    "doc/02 §12.2",
    "Every manifest record carries the complete 22-field schema defined in Document 02 §12.2.",
    "the manifest builder is not implemented",
)
def test_record_schema():
    mod = _manifest_module()
    sample_record = mod.create_record(
        source_record_id=0,
        source_file="train.csv",
        split="train",
        configuration="newsclippings-official-v1",
        image_path="images/sample.jpg",
        caption_raw="Sample caption",
        caption="sample caption",
        label="MATCHED",
        source_label="consistent",
        source_label_raw=0,
        caption_id="101",
        image_id="101",
    )
    missing = REQUIRED_FIELDS - set(sample_record.keys())
    assert not missing, "record is missing fields: %s" % missing


@known_gap(
    "manifest.sample_id_format",
    "F5",
    "fast",
    "doc/02 §12.2",
    "The sample_id follows the format 'nclp_<split>_<index:06d>' and is unique.",
    "the manifest builder is not implemented",
)
def test_sample_id_format():
    mod = _manifest_module()
    sample_id = mod.format_sample_id("train", 42)
    assert sample_id == "nclp_train_000042"


@known_gap(
    "manifest.version_is_one",
    "F5",
    "fast",
    "doc/02 §12.2, §12.6",
    "The manifest_version integer is 1.",
    "the manifest builder is not implemented",
)
def test_manifest_version_is_one():
    mod = _manifest_module()
    assert mod.MANIFEST_VERSION == 1


@known_gap(
    "manifest.atomic_write",
    "F5",
    "fast",
    "doc/02 §12.4",
    "Writing a manifest writes to a temporary file first and renames atomically.",
    "the manifest writer is not implemented",
)
def test_manifest_atomic_write(tmp_path):
    mod = _manifest_module()
    target = tmp_path / "manifest.jsonl"
    records = [{"sample_id": "test_1", "manifest_version": 1}]
    mod.write_manifest(target, records)
    assert target.is_file()
    assert not (tmp_path / "manifest.jsonl.tmp").exists()


@known_gap(
    "manifest.hash_consistency",
    "F5",
    "fast",
    "doc/02 §12.2, §12.5",
    "The image_sha256 and caption_sha256 are 64-character lowercase hexadecimal digests.",
    "the manifest hasher is not implemented",
)
def test_hash_consistency():
    import re
    mod = _manifest_module()
    caption_hash = mod.hash_caption("A standard news caption.")
    assert re.fullmatch(r"[0-9a-f]{64}", caption_hash), "caption hash is not a 64-char hex digest: %r" % caption_hash
