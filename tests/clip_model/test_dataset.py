"""The PyTorch feature dataset reader and alignment checks of Document 03 §5.

Tests FeaturePairDataset, read-only numpy serving, UNKNOWN sample exclusion,
and manifest-to-cache alignment verification.
"""

from pathlib import Path
import pytest

from tests.ids import case, known_gap

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")


def _dataset_module():
    import importlib
    return importlib.import_module("classifier_dataset")


@case(
    "classifier_data.module_exists",
    "F2",
    "fast",
    "doc/03 §5.2",
    "The project provides a classifier dataset module defining FeaturePairDataset.",
    requires=("torch", "numpy"),
)
def test_dataset_module_exists():
    _dataset_module()


@case(
    "classifier_data.feature_pair_serves_triples",
    "F2",
    "fast",
    "doc/03 §5.2",
    "FeaturePairDataset serves (features, label, sample_id) triples.",
    requires=("torch", "numpy"),
)
def test_feature_pair_serves_triples():
    mod = _dataset_module()
    # Mock data: 2 samples, 512-d features
    features = np.random.randn(2, 512).astype(np.float32)
    labels = np.array([0, 1], dtype=np.int64)
    sample_ids = ["s1", "s2"]
    ds = mod.FeaturePairDataset(features=features, labels=labels, sample_ids=sample_ids)
    assert len(ds) == 2
    item = ds[0]
    assert len(item) == 3
    feat, label, sid = item
    assert isinstance(feat, torch.Tensor)
    assert label == 0
    assert sid == "s1"


@case(
    "classifier_data.unknown_excluded",
    "F2",
    "fast",
    "doc/03 §5.3",
    "UNKNOWN records are excluded from supervised training by default.",
    requires=("torch", "numpy"),
)
def test_unknown_excluded():
    mod = _dataset_module()
    features = np.random.randn(3, 512).astype(np.float32)
    labels = ["MATCHED", "UNKNOWN", "MISMATCHED"]
    sample_ids = ["s1", "s2", "s3"]
    ds = mod.FeaturePairDataset(features=features, labels=labels, sample_ids=sample_ids, exclude_unknown=True)
    assert len(ds) == 2
    assert ds.sample_ids == ["s1", "s3"]


@case(
    "classifier_data.read_only_numpy",
    "F2",
    "fast",
    "doc/03 §5.2",
    "FeaturePairDataset holds arrays as read-only NumPy and performs no in-place mutation.",
    requires=("torch", "numpy"),
)
def test_read_only_numpy():
    mod = _dataset_module()
    features = np.random.randn(2, 512).astype(np.float32)
    labels = np.array([0, 1], dtype=np.int64)
    sample_ids = ["s1", "s2"]
    ds = mod.FeaturePairDataset(features=features, labels=labels, sample_ids=sample_ids)
    assert not ds.features.flags.writeable, "features array in dataset is mutable"


@case(
    "classifier_data.manifest_cache_alignment",
    "F2",
    "fast",
    "doc/03 §5.1",
    "Every training run verifies manifest-to-cache alignment; row count or hash mismatch raises.",
    requires=("torch", "numpy"),
)
def test_manifest_cache_alignment():
    mod = _dataset_module()
    manifest_records = [{"sample_id": "s1", "image_sha256": "h1", "caption_sha256": "c1"}]
    cache_ids = ["s1", "s2"]  # Mismatch: 1 manifest record vs 2 cache entries
    with pytest.raises(Exception):
        mod.verify_alignment(manifest_records, cache_ids)
