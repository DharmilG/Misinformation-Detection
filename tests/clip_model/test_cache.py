"""F4 - the feature cache contract of document 01 §11.

Three of these check the properties of the artefact the current pipeline
produces, and pass today.  The rest assert the cache contract itself - the key,
the metadata, the atomic write, the load validation.
"""

import json
from pathlib import Path

import pytest

from tests.ids import case

np = pytest.importorskip("numpy")

CACHE_MODULE = "feature_cache"


def _cache_module():
    import importlib

    return importlib.import_module(CACHE_MODULE)


# ---------------------------------------------------------------------------
# what the current artefact satisfies
# ---------------------------------------------------------------------------
@case(
    "cache.saved_array_shapes",
    "F4",
    "slow",
    "doc/01 §11.2",
    "The saved feature arrays have shape (1, 512).",
    requires=("torch", "transformers", "numpy"),
)
def test_saved_array_shapes(extractor, tmp_path):
    from tests.fixtures import captions, images

    path = images.write(tmp_path / "solid.png", images.solid_rgb())
    out = extractor.extract_features(str(path), captions.SHORT)
    target = tmp_path / "features.npz"
    np.savez(
        target,
        image_features=out["image_features"].numpy(),
        text_features=out["text_features"].numpy(),
    )
    with np.load(target, allow_pickle=False) as data:
        assert data["image_features"].shape == (1, 512)
        assert data["text_features"].shape == (1, 512)


@case(
    "cache.npz_no_pickle",
    "F4",
    "slow",
    "doc/01 §11.2",
    "The saved features load with allow_pickle=False, so no arbitrary code can run on load.",
    requires=("torch", "transformers", "numpy"),
)
def test_npz_no_pickle(extractor, tmp_path):
    from tests.fixtures import captions, images

    path = images.write(tmp_path / "solid.png", images.solid_rgb())
    out = extractor.extract_features(str(path), captions.SHORT)
    target = tmp_path / "features.npz"
    np.savez(
        target,
        image_features=out["image_features"].numpy(),
        text_features=out["text_features"].numpy(),
    )
    with np.load(target, allow_pickle=False) as data:
        assert data["image_features"].shape == (1, 512)


@case(
    "cache.npz_roundtrip_values_identical",
    "F4",
    "slow",
    "doc/01 §11.6",
    "Saving and loading the features preserves the values exactly.",
    requires=("torch", "transformers", "numpy"),
)
def test_npz_roundtrip_values_identical(extractor, tmp_path):
    from tests.fixtures import captions, images

    path = images.write(tmp_path / "solid.png", images.solid_rgb())
    out = extractor.extract_features(str(path), captions.SHORT)
    original = out["image_features"].numpy()
    target = tmp_path / "features.npz"
    np.savez(
        target,
        image_features=original,
        text_features=out["text_features"].numpy(),
    )
    with np.load(target, allow_pickle=False) as data:
        assert np.array_equal(data["image_features"], original)


# ---------------------------------------------------------------------------
# the cache contract
# ---------------------------------------------------------------------------
@case(
    "cache.module_exists",
    "F4",
    "fast",
    "doc/01 §11.1",
    "The project provides a feature cache module with a key, a write and a read.",
)
def test_cache_module_exists():
    module = _cache_module()
    for name in ("cache_key", "write_cache", "read_cache"):
        assert hasattr(module, name), "the cache module does not define %s" % name


@case(
    "cache.key_components",
    "F4",
    "fast",
    "doc/01 §11.3",
    "The cache key covers every input that can change the output, including the model revision, the feature extractor version, the preprocessing version, the dtype, the normalisation flag, the tokenizer limit, the library versions, and the source manifest hash.",
)
def test_key_components():
    key = _cache_module().cache_key(
        model_id="openai/clip-vit-base-patch32",
        model_revision="0" * 40,
        feature_extractor_version=1,
        preprocessing_version=1,
        dtype="float32",
        normalized=True,
        tokenizer_max_length=77,
        library_versions={"transformers": "4.0.0"},
        source_manifest_sha256="a" * 64,
    )
    for part in ("openai/clip-vit-base-patch32", "0" * 40, "float32"):
        assert part in key, "the key does not include %r" % part


@case(
    "cache.metadata_present",
    "F4",
    "fast",
    "doc/01 §11.4",
    "The cache records the metadata block, including the model id, the revision, the versions, the dtype, the normalisation flag, the tokenizer limit, the device, the row count, and the manifest hash.",
)
def test_metadata_present(tmp_path):
    target = tmp_path / "cache"
    target.mkdir()
    _cache_module().write_cache(target, image_features=np.zeros((1, 512), "float32"),
                                text_features=np.zeros((1, 512), "float32"))
    meta = target / "metadata.json"
    assert meta.is_file(), "the cache has no metadata.json"
    payload = json.loads(meta.read_text(encoding="utf-8"))
    for field in ("model_id", "model_revision", "dtype", "normalized", "row_count"):
        assert field in payload, "the metadata is missing %s" % field


@case(
    "cache.atomic_write",
    "F4",
    "fast",
    "doc/01 §11.5",
    "An interrupted write leaves no loadable cache; the metadata is written last.",
)
def test_atomic_write(tmp_path):
    target = tmp_path / "cache"
    target.mkdir()
    module = _cache_module()
    module.write_cache(target, image_features=np.zeros((1, 512), "float32"),
                       text_features=np.zeros((1, 512), "float32"))
    # A cache whose metadata is missing is not loadable.
    (target / "metadata.json").unlink()
    with pytest.raises(Exception):
        module.read_cache(target)


@case(
    "cache.load_validates",
    "F4",
    "fast",
    "doc/01 §11.6",
    "A cache whose revision, version, or normalisation flag does not match the current configuration is rejected on load.",
)
def test_load_validates(tmp_path):
    target = tmp_path / "cache"
    target.mkdir()
    module = _cache_module()
    module.write_cache(target, image_features=np.zeros((1, 512), "float32"),
                       text_features=np.zeros((1, 512), "float32"))
    meta = target / "metadata.json"
    payload = json.loads(meta.read_text(encoding="utf-8"))
    payload["model_revision"] = "f" * 40
    meta.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Exception):
        module.read_cache(target)


@case(
    "cache.device_excluded_by_default",
    "F4",
    "fast",
    "doc/01 §11.7",
    "The device is not part of the cache key by default, so a cache is shareable across machines.",
)
def test_device_excluded_by_default():
    module = _cache_module()
    common = dict(
        model_id="openai/clip-vit-base-patch32",
        model_revision="0" * 40,
        feature_extractor_version=1,
        preprocessing_version=1,
        dtype="float32",
        normalized=True,
        tokenizer_max_length=77,
        library_versions={},
        source_manifest_sha256="b" * 64,
    )
    first = module.cache_key(**common)
    second = module.cache_key(device="cuda", **common)
    assert first == second, "the device changed the cache key by default"


@case(
    "cache.device_namespace_flag",
    "F4",
    "fast",
    "doc/01 §11.7",
    "The device_namespace flag segregates caches by device when strict reproducibility is required.",
)
def test_device_namespace_flag():
    module = _cache_module()
    common = dict(
        model_id="openai/clip-vit-base-patch32",
        model_revision="0" * 40,
        feature_extractor_version=1,
        preprocessing_version=1,
        dtype="float32",
        normalized=True,
        tokenizer_max_length=77,
        library_versions={},
        source_manifest_sha256="b" * 64,
    )
    first = module.cache_key(device_namespace=True, device="cpu", **common)
    second = module.cache_key(device_namespace=True, device="cuda", **common)
    assert first != second, "the device_namespace flag did not segregate the caches"
