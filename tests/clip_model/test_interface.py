"""F2 - the encoder's interface.

These assert the interface the specification requires.  Each one names the
clause it enforces and the change that is needed, and each is a declared gap:
the implementation does not yet meet it.

The behavioural conformance cases live in ``test_encoder.py``.
"""

import pytest

from tests.ids import case, known_gap

torch = pytest.importorskip("torch")

# ---------------------------------------------------------------------------
# F2 - interface conformance
# ---------------------------------------------------------------------------
def _extractor_class():
    from clip_features import CLIPFeatureExtractor

    return CLIPFeatureExtractor


@known_gap(
    "clip.api.encode_pair_exists",
    "F2",
    "fast",
    "doc/01 §7.1",
    "The encoder exposes encode_pair(image, text).",
    "the encoder exposes extract_features(image_path, caption) only",
    requires=("torch", "transformers"),
)
def test_encode_pair_exists():
    assert hasattr(_extractor_class(), "encode_pair")


@known_gap(
    "clip.api.encode_images_exists",
    "F2",
    "fast",
    "doc/01 §7.1",
    "The encoder exposes encode_images(images) for bulk extraction.",
    "there is no bulk image API",
    requires=("torch", "transformers"),
)
def test_encode_images_exists():
    assert hasattr(_extractor_class(), "encode_images")


@known_gap(
    "clip.api.encode_texts_exists",
    "F2",
    "fast",
    "doc/01 §7.1",
    "The encoder exposes encode_texts(texts) for bulk text extraction.",
    "there is no bulk text API",
    requires=("torch", "transformers"),
)
def test_encode_texts_exists():
    assert hasattr(_extractor_class(), "encode_texts")


@known_gap(
    "clip.api.encode_batch_exists",
    "F2",
    "fast",
    "doc/01 §7.1",
    "The encoder exposes encode_batch(pairs) for aligned image-caption pairs.",
    "there is no batch API",
    requires=("torch", "transformers"),
)
def test_encode_batch_exists():
    assert hasattr(_extractor_class(), "encode_batch")


@known_gap(
    "clip.api.revision_pinned",
    "F2",
    "fast",
    "doc/01 §4.2",
    "The checkpoint revision is pinned to a specific commit SHA in src/clip_model/constants.py.",
    "there is no constants module and the revision is not pinned",
)
def test_revision_pinned():
    import re

    constants = Path(__file__).resolve().parent.parent / "src" / "clip_model" / "constants.py"
    assert constants.is_file(), "src/clip_model/constants.py does not exist"
    text = constants.read_text(encoding="utf-8")
    match = re.search(r"MODEL_REVISION\s*=\s*[\"']([^\"']+)[\"']", text)
    assert match, "MODEL_REVISION is not defined"
    assert re.fullmatch(r"[0-9a-f]{40}", match.group(1)), (
        "MODEL_REVISION is not a commit SHA: %r" % match.group(1)
    )


@known_gap(
    "clip.api.encoder_info_provenance",
    "F2",
    "fast",
    "doc/01 §4.3",
    "The encoder reports its own provenance: the model id, the revision, and the weights hash.",
    "there is no provenance record",
    requires=("torch", "transformers"),
)
def test_encoder_info_provenance(extractor):
    info = getattr(extractor, "info", None)
    assert info is not None, "the encoder does not report its provenance"
    for field in ("model_id", "model_revision", "weights_sha256"):
        assert getattr(info, field, None), "the provenance record is missing %s" % field


@known_gap(
    "clip.api.device_selectable",
    "F2",
    "fast",
    "doc/01 §9.1",
    "The device can be selected, and an unavailable device raises rather than falling back silently.",
    "the encoder always loads on the default device",
    requires=("torch", "transformers"),
)
def test_device_selectable():
    import inspect

    cls = _extractor_class()
    params = set(inspect.signature(cls.__init__).parameters)
    params |= set(inspect.signature(cls.extract_features).parameters)
    assert "device" in params, "the device cannot be selected"


@known_gap(
    "clip.api.single_load_per_process",
    "F2",
    "fast",
    "doc/01 §6.1",
    "The weights are loaded once per process, not once per instance.",
    "the weights are loaded in __init__, so every instance loads again",
    requires=("torch", "transformers"),
)
def test_single_load_per_process(monkeypatch):
    import clip_features

    calls = {"model": 0, "processor": 0}

    class FakeModel:
        def __init__(self):
            self.config = type("C", (), {"projection_dim": 512})()
            self.training = False

        def eval(self):
            return self

    def fake_model(*args, **kwargs):
        calls["model"] += 1
        return FakeModel()

    def fake_processor(*args, **kwargs):
        calls["processor"] += 1
        return object()

    monkeypatch.setattr(clip_features.CLIPModel, "from_pretrained", staticmethod(fake_model))
    monkeypatch.setattr(
        clip_features.CLIPProcessor, "from_pretrained", staticmethod(fake_processor)
    )

    cls = clip_features.CLIPFeatureExtractor
    cls()
    cls()
    assert calls["model"] == 1, "the weights were loaded %d times" % calls["model"]
    assert calls["processor"] == 1, "the processor was loaded %d times" % calls["processor"]


@known_gap(
    "clip.api.errors_module_exists",
    "F2",
    "fast",
    "doc/01 §13.1",
    "The project defines ModelUnavailableError, DeviceUnavailableError, FeatureExtractionError and FeatureCacheError.",
    "there is no errors module",
)
def test_errors_module_exists():
    import errors  # noqa: F401  - the import is the assertion

    for name in (
        "ModelUnavailableError",
        "DeviceUnavailableError",
        "FeatureExtractionError",
        "FeatureCacheError",
    ):
        assert hasattr(errors, name), "the error module does not define %s" % name


@known_gap(
    "clip.api.typed_errors",
    "F2",
    "fast",
    "doc/01 §13.1, §13.2",
    "A missing image raises a project error with a user-safe message, not a raw library exception.",
    "a missing image raises the raw FileNotFoundError",
    requires=("torch", "transformers"),
)
def test_typed_errors(extractor, tmp_path):
    import errors

    with pytest.raises(errors.FeatureExtractionError):
        extractor.extract_features(str(tmp_path / "missing.png"), captions.SHORT)
