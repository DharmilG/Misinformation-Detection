"""F1 - the encoder's behaviour.

These exercise the API that exists today against the behavioural clauses of
document 01.  Most of them pass; they are the real baseline.

The interface conformance cases live in ``test_interface.py``.  The split is
the one document 01 section 15.2 makes: a uniformly red suite would hide the
fact that the projection, the pooling, the shapes, the dtype, and the
finiteness are already correct.
"""

import pytest

from tests.fixtures import captions, images
from tests.ids import case, known_gap

torch = pytest.importorskip("torch")

# ---------------------------------------------------------------------------
# F1 - encoder behaviour
# ---------------------------------------------------------------------------
@case(
    "clip.extract.shape",
    "F1",
    "slow",
    "doc/01 §7.2",
    "The image and text features are float32 arrays of shape (1, 512).",
    requires=("torch", "transformers"),
)
def test_extract_shape(extractor, tmp_path):
    path = images.write(tmp_path / "solid.png", images.solid_rgb())
    out = extractor.extract_features(str(path), captions.SHORT)
    assert tuple(out["image_features"].shape) == (1, 512)
    assert tuple(out["text_features"].shape) == (1, 512)


@case(
    "clip.extract.dtype",
    "F1",
    "slow",
    "doc/01 §7.2",
    "The features are float32.",
    requires=("torch", "transformers"),
)
def test_extract_dtype(extractor, tmp_path):
    path = images.write(tmp_path / "solid.png", images.solid_rgb())
    out = extractor.extract_features(str(path), captions.SHORT)
    assert out["image_features"].dtype == torch.float32
    assert out["text_features"].dtype == torch.float32


@case(
    "clip.extract.finite",
    "F1",
    "slow",
    "doc/01 §7.2",
    "Every feature value is finite.",
    requires=("torch", "transformers"),
)
def test_extract_finite(extractor, tmp_path):
    path = images.write(tmp_path / "solid.png", images.solid_rgb())
    out = extractor.extract_features(str(path), captions.SHORT)
    assert torch.isfinite(out["image_features"]).all()
    assert torch.isfinite(out["text_features"]).all()


@case(
    "clip.extract.projection_dim_from_config",
    "F1",
    "slow",
    "doc/01 §8.1",
    "The feature width equals the checkpoint's projection_dim, so the vectors are in the shared projection space and not the 768-dimensional vision hidden state.",
    requires=("torch", "transformers"),
)
def test_projection_dim_from_config(extractor, tmp_path):
    path = images.write(tmp_path / "solid.png", images.solid_rgb())
    out = extractor.extract_features(str(path), captions.SHORT)
    expected = extractor.model.config.projection_dim
    assert out["image_features"].shape[-1] == expected
    assert out["text_features"].shape[-1] == expected


@case(
    "clip.extract.text_pooling_is_eot",
    "F1",
    "slow",
    "doc/01 §8.3",
    "The text vector equals the projection of the EOT-position pooled state, not a mean over the sequence.",
    requires=("torch", "transformers"),
)
def test_text_pooling_is_eot(extractor, tmp_path):
    path = images.write(tmp_path / "solid.png", images.solid_rgb())
    out = extractor.extract_features(str(path), captions.SHORT)
    inputs = extractor.processor(text=[captions.SHORT], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_out = extractor.model.text_model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        manual = extractor.model.text_projection(text_out.pooler_output)
        # The library returns the projected vector already normalised, so the
        # manual projection is normalised the same way before the comparison.
        manual = manual / manual.norm(p=2, dim=-1, keepdim=True)
    assert torch.allclose(manual, out["text_features"], atol=1e-5)

    # A mean over the tokens is a different vector, so the test would notice
    # if the pooling ever changed.
    with torch.no_grad():
        hidden = text_out.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        mean = (hidden * mask).sum(1) / mask.sum(1)
        mean_projected = extractor.model.text_projection(mean)
        mean_projected = mean_projected / mean_projected.norm(p=2, dim=-1, keepdim=True)
    assert not torch.allclose(mean_projected, out["text_features"], atol=1e-3), (
        "the mean over the tokens matched, so the test cannot tell the two apart"
    )


@case(
    "clip.extract.no_autograd_graph",
    "F1",
    "slow",
    "doc/01 §6.3",
    "The returned features carry no autograd graph.",
    requires=("torch", "transformers"),
)
def test_no_autograd_graph(extractor, tmp_path):
    path = images.write(tmp_path / "solid.png", images.solid_rgb())
    out = extractor.extract_features(str(path), captions.SHORT)
    assert out["image_features"].requires_grad is False
    assert out["text_features"].requires_grad is False


@case(
    "clip.extract.eval_mode",
    "F1",
    "slow",
    "doc/01 §6.2",
    "The encoder is in eval() mode, so dropout cannot make repeated calls differ.",
    requires=("torch", "transformers"),
)
def test_eval_mode(extractor):
    assert extractor.model.training is False


@case(
    "clip.extract.repeatable_on_cpu",
    "F1",
    "slow",
    "doc/01 §14.2",
    "The same input twice gives bit-identical output on CPU.",
    requires=("torch", "transformers"),
)
def test_repeatable_on_cpu(extractor, tmp_path):
    path = images.write(tmp_path / "solid.png", images.solid_rgb())
    first = extractor.extract_features(str(path), captions.SHORT)
    second = extractor.extract_features(str(path), captions.SHORT)
    assert torch.equal(first["image_features"], second["image_features"])
    assert torch.equal(first["text_features"], second["text_features"])


@case(
    "clip.extract.unit_norm",
    "F1",
    "slow",
    "doc/01 §8.2",
    "The extractor L2-normalises the image and text vectors before returning them.",
    requires=("torch", "transformers"),
)
def test_unit_norm(extractor, tmp_path):
    path = images.write(tmp_path / "solid.png", images.solid_rgb())
    out = extractor.extract_features(str(path), captions.SHORT)
    for key in ("image_features", "text_features"):
        norm = float(out[key].float().norm(dim=-1)[0])
        assert abs(norm - 1.0) < 1e-6, "%s norm was %r" % (key, norm)


@case(
    "clip.extract.norm_is_idempotent",
    "F1",
    "slow",
    "doc/01 §8.2",
    "Normalising an already normalised vector is a no-op, so an explicit second normalisation cannot change a result.",
    requires=("torch", "transformers"),
)
def test_norm_is_idempotent(extractor, tmp_path):
    path = images.write(tmp_path / "solid.png", images.solid_rgb())
    out = extractor.extract_features(str(path), captions.SHORT)
    for key in ("image_features", "text_features"):
        vector = out[key].float()
        once = vector / vector.norm(p=2, dim=-1, keepdim=True)
        twice = once / once.norm(p=2, dim=-1, keepdim=True)
        assert torch.allclose(once, twice, atol=1e-6), "%s is not idempotent" % key


@known_gap(
    "clip.extract.token_metadata",
    "F1",
    "slow",
    "doc/01 §7.2",
    "The result reports the CLIP token count, the original token count, and whether the caption was truncated.",
    "the result carries only the two feature tensors",
)
def test_token_metadata(extractor, tmp_path):
    path = images.write(tmp_path / "solid.png", images.solid_rgb())
    out = extractor.extract_features(str(path), captions.SHORT)
    for key in ("text_token_count", "text_original_token_count", "text_truncated"):
        assert key in out, "the result does not report %s" % key


@known_gap(
    "clip.extract.long_caption_truncates",
    "F1",
    "slow",
    "doc/01 §5.2, §7.2",
    "A caption longer than the 77-token window is truncated rather than raising, and the truncation is reported.",
    "the processor is called without truncation, so a long caption overflows the position embedding",
)
def test_long_caption_truncates(extractor, tmp_path):
    path = images.write(tmp_path / "solid.png", images.solid_rgb())
    out = extractor.extract_features(str(path), captions.LONG)
    assert torch.isfinite(out["text_features"]).all()
