"""F3 - the preprocessing contract of document 04 §7.

The extractor opens the image and converts it to RGB.  The parts of document
04 §7 it does not implement are recorded as known gaps.
"""

import pytest

from tests.fixtures import captions, images
from tests.ids import case, known_gap

torch = pytest.importorskip("torch")


@case(
    "preprocess.rgb_mode",
    "F3",
    "slow",
    "doc/04 §7.1",
    "A grayscale and a CMYK image are both accepted and produce valid features.",
    requires=("torch", "transformers"),
)
def test_rgb_mode(extractor, tmp_path):
    for name, image in (
        ("gray.png", images.grayscale()),
        ("cmyk.jpg", images.cmyk()),
    ):
        path = images.write(tmp_path / name, image)
        out = extractor.extract_features(str(path), captions.SHORT)
        assert out["image_features"].shape == (1, 512)
        assert torch.isfinite(out["image_features"]).all()


@case(
    "preprocess.corrupt_image_rejected",
    "F3",
    "slow",
    "doc/04 §7.2",
    "A file that is not a decodable image is rejected rather than silently encoded.",
    requires=("torch", "transformers"),
)
def test_corrupt_image_rejected(extractor, tmp_path):
    path = images.corrupt_bytes(tmp_path / "broken.png")
    with pytest.raises(Exception):
        extractor.extract_features(str(path), captions.SHORT)


@case(
    "preprocess.exif_applied",
    "F3",
    "slow",
    "doc/04 §7.2",
    "An image whose EXIF orientation is set is transposed before it is encoded.",
    requires=("torch", "transformers"),
)
def test_exif_applied(extractor, tmp_path):
    from PIL import Image, ImageOps

    tagged = images.exif_rotated(tmp_path / "tagged.jpg", orientation=6)
    with Image.open(tagged) as opened:
        manual = ImageOps.exif_transpose(opened).convert("RGB")
    manual_path = tmp_path / "manual.jpg"
    manual.save(manual_path, "JPEG", quality=95)

    a = extractor.extract_features(str(tagged), captions.SHORT)
    b = extractor.extract_features(str(manual_path), captions.SHORT)
    assert torch.allclose(a["image_features"], b["image_features"], atol=1e-3), (
        "the EXIF orientation made no difference, so it was not applied"
    )


@known_gap(
    "preprocess.alpha_white",
    "F3",
    "slow",
    "doc/04 §7.3",
    "A transparent pixel is composited onto white, not discarded.",
    "the alpha channel is discarded, so a transparent pixel keeps its stored colour",
    requires=("torch", "transformers"),
)
def test_alpha_white(extractor, tmp_path):
    from PIL import Image

    path = images.write(tmp_path / "alpha.png", images.rgba_with_transparent_pixel())
    with Image.open(path) as opened:
        converted = opened.convert("RGB")
    assert converted.getpixel((0, 0)) == (255, 255, 255), (
        "a transparent pixel became %r, not white" % (converted.getpixel((0, 0)),)
    )
    out = extractor.extract_features(str(path), captions.SHORT)
    assert torch.isfinite(out["image_features"]).all()


@known_gap(
    "preprocess.downscale_to_2048",
    "F3",
    "slow",
    "doc/04 §7.4",
    "The image handed to the processor is at most 2048 px on its longest side.",
    "the original image is handed to the processor, with no downscale step",
    requires=("torch", "transformers"),
)
def test_downscale_to_2048(extractor, tmp_path, monkeypatch):
    path = images.write(tmp_path / "big.png", images.solid_rgb(size=(3000, 2000)))

    seen = {}
    inner = extractor.processor

    class Spy:
        """Records the image the processor is given."""

        def __call__(self, *args, **kwargs):
            given = kwargs.get("images")
            if given is None and args:
                given = args[0]
            if isinstance(given, (list, tuple)):
                given = given[0]
            if given is not None:
                seen["size"] = (given.width, given.height)
            return inner(*args, **kwargs)

    monkeypatch.setattr(extractor, "processor", Spy())

    extractor.extract_features(str(path), captions.SHORT)

    assert "size" in seen, "the processor was never called with an image"
    width, height = seen["size"]
    assert max(width, height) <= 2048, (
        "the processor received a %dx%d image, so the 2048 px downscale was not applied"
        % (width, height)
    )


@case(
    "preprocess.downscale_is_not_observable_in_the_embedding",
    "F3",
    "slow",
    "doc/02 §8.1.2",
    "The 2048 px downscale does not change the embedding: the processor's own resize and the relative crop dominate, so the accepted corpus skew is small.",
    requires=("torch", "transformers"),
)
def test_downscale_is_not_observable_in_the_embedding(extractor, tmp_path):
    from PIL import Image

    path = images.write(tmp_path / "big.png", images.solid_rgb(size=(3000, 2000)))
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        scale = 2048 / max(image.width, image.height)
        scaled = image.resize(
            (round(image.width * scale), round(image.height * scale)),
            Image.Resampling.LANCZOS,
        )
    scaled_path = tmp_path / "scaled.png"
    scaled.save(scaled_path, "PNG")

    direct = extractor.extract_features(str(path), captions.SHORT)
    through_chain = extractor.extract_features(str(scaled_path), captions.SHORT)
    assert torch.allclose(
        direct["image_features"], through_chain["image_features"], atol=1e-3
    ), "the 2048 px downscale changed the embedding"
