"""The input validation rules, image and text normalization, and request schema of Document 04 §6, §7, §8, §9.

Tests rejection of invalid/corrupt/oversized inputs, deterministic RGB conversion,
caption Unicode NFKC normalization, and InputRequest schema invariants.
"""

from io import BytesIO
import pytest

from tests.ids import case, known_gap

PIL_Image = pytest.importorskip("PIL.Image")


def _input_module():
    import importlib
    return importlib.import_module("input_pipeline")


@known_gap(
    "input_pipeline.module_exists",
    "F3",
    "fast",
    "doc/04 §9.3",
    "The project provides an input pipeline module exposing build_input_request.",
    "there is no input_pipeline module",
    requires=("PIL",),
)
def test_input_pipeline_module_exists():
    _input_module()


@known_gap(
    "input_pipeline.empty_file_rejected",
    "F3",
    "fast",
    "doc/04 §6.1, §10.2",
    "Empty 0-byte upload raises InputValidationError with EMPTY_FILE code.",
    "input validation is not implemented",
    requires=("PIL",),
)
def test_empty_file_rejected():
    mod = _input_module()
    with pytest.raises(Exception) as excinfo:
        mod.build_input_request(image_bytes=b"", filename="test.jpg", caption="A valid caption")
    assert getattr(excinfo.value, "code", None) == mod.InputErrorCode.EMPTY_FILE


@known_gap(
    "input_pipeline.image_too_large_rejected",
    "F3",
    "fast",
    "doc/04 §6.1",
    "Upload exceeding max_upload_bytes (10 MB) raises IMAGE_TOO_LARGE before attempting decode.",
    "upload size check is not implemented",
    requires=("PIL",),
)
def test_image_too_large_rejected():
    mod = _input_module()
    oversized = b"0" * (10 * 1024 * 1024 + 1)
    with pytest.raises(Exception) as excinfo:
        mod.build_input_request(image_bytes=oversized, filename="huge.jpg", caption="A valid caption")
    assert getattr(excinfo.value, "code", None) == mod.InputErrorCode.IMAGE_TOO_LARGE


@known_gap(
    "input_pipeline.pixel_count_limit",
    "F3",
    "fast",
    "doc/04 §6.1",
    "Image exceeding max_pixels (40 MP) raises IMAGE_TOO_MANY_PIXELS before loading full pixels.",
    "pixel count limit is not implemented",
    requires=("PIL",),
)
def test_pixel_count_limit():
    mod = _input_module()
    # Configure custom low limit to test without allocating 40 MP
    cfg = mod.InputConfig(max_pixels=100)
    buf = BytesIO()
    img = PIL_Image.new("RGB", (20, 20))  # 400 pixels > 100 max
    img.save(buf, format="PNG")
    with pytest.raises(Exception) as excinfo:
        mod.build_input_request(image_bytes=buf.getvalue(), filename="test.png", caption="Caption", config=cfg)
    assert getattr(excinfo.value, "code", None) == mod.InputErrorCode.IMAGE_TOO_MANY_PIXELS


@known_gap(
    "input_pipeline.image_too_small_rejected",
    "F3",
    "fast",
    "doc/04 §6.1",
    "Image with width or height under 32 px raises IMAGE_TOO_SMALL.",
    "minimum dimension validation is not implemented",
    requires=("PIL",),
)
def test_image_too_small_rejected():
    mod = _input_module()
    buf = BytesIO()
    img = PIL_Image.new("RGB", (16, 64))
    img.save(buf, format="PNG")
    with pytest.raises(Exception) as excinfo:
        mod.build_input_request(image_bytes=buf.getvalue(), filename="small.png", caption="Caption")
    assert getattr(excinfo.value, "code", None) == mod.InputErrorCode.IMAGE_TOO_SMALL


@known_gap(
    "input_pipeline.unsupported_format_rejected",
    "F3",
    "fast",
    "doc/04 §6.1, §7.1",
    "Non-JPEG/PNG formats (e.g. text file or GIF) raise UNSUPPORTED_FORMAT.",
    "format validation is not implemented",
    requires=("PIL",),
)
def test_unsupported_format_rejected():
    mod = _input_module()
    with pytest.raises(Exception) as excinfo:
        mod.build_input_request(image_bytes=b"plain text file", filename="file.jpg", caption="Caption")
    assert getattr(excinfo.value, "code", None) == mod.InputErrorCode.UNSUPPORTED_FORMAT


@known_gap(
    "input_pipeline.animated_image_rejected",
    "F3",
    "fast",
    "doc/04 §6.1, §7.1",
    "Multi-frame animated images raise ANIMATED_IMAGE.",
    "animation check is not implemented",
    requires=("PIL",),
)
def test_animated_image_rejected():
    mod = _input_module()
    buf = BytesIO()
    img1 = PIL_Image.new("RGB", (64, 64), color="red")
    img2 = PIL_Image.new("RGB", (64, 64), color="blue")
    img1.save(buf, format="GIF", save_all=True, append_images=[img2])
    with pytest.raises(Exception) as excinfo:
        mod.build_input_request(image_bytes=buf.getvalue(), filename="anim.gif", caption="Caption")
    assert getattr(excinfo.value, "code", None) in (
        mod.InputErrorCode.ANIMATED_IMAGE,
        mod.InputErrorCode.UNSUPPORTED_FORMAT,
    )


@known_gap(
    "input_pipeline.caption_empty_rejected",
    "F3",
    "fast",
    "doc/04 §6.1, §8.2",
    "Whitespace-only or empty caption raises CAPTION_EMPTY.",
    "caption empty check is not implemented",
    requires=("PIL",),
)
def test_caption_empty_rejected():
    mod = _input_module()
    buf = BytesIO()
    img = PIL_Image.new("RGB", (64, 64))
    img.save(buf, format="PNG")
    with pytest.raises(Exception) as excinfo:
        mod.build_input_request(image_bytes=buf.getvalue(), filename="test.png", caption="   \n\t  ")
    assert getattr(excinfo.value, "code", None) == mod.InputErrorCode.CAPTION_EMPTY


@known_gap(
    "input_pipeline.caption_too_short_rejected",
    "F3",
    "fast",
    "doc/04 §6.1, §8.2",
    "Caption with fewer than 3 characters after normalization raises CAPTION_TOO_SHORT.",
    "caption minimum length check is not implemented",
    requires=("PIL",),
)
def test_caption_too_short_rejected():
    mod = _input_module()
    buf = BytesIO()
    img = PIL_Image.new("RGB", (64, 64))
    img.save(buf, format="PNG")
    with pytest.raises(Exception) as excinfo:
        mod.build_input_request(image_bytes=buf.getvalue(), filename="test.png", caption="ab")
    assert getattr(excinfo.value, "code", None) == mod.InputErrorCode.CAPTION_TOO_SHORT


@known_gap(
    "input_pipeline.caption_too_long_rejected",
    "F3",
    "fast",
    "doc/04 §6.1, §8.2",
    "Caption with more than 1000 characters after normalization raises CAPTION_TOO_LONG.",
    "caption maximum length check is not implemented",
    requires=("PIL",),
)
def test_caption_too_long_rejected():
    mod = _input_module()
    buf = BytesIO()
    img = PIL_Image.new("RGB", (64, 64))
    img.save(buf, format="PNG")
    long_caption = "word " * 250  # >1000 chars
    with pytest.raises(Exception) as excinfo:
        mod.build_input_request(image_bytes=buf.getvalue(), filename="test.png", caption=long_caption)
    assert getattr(excinfo.value, "code", None) == mod.InputErrorCode.CAPTION_TOO_LONG


@known_gap(
    "input_pipeline.caption_nfkc_normalized",
    "F3",
    "fast",
    "doc/04 §8.1",
    "Caption normalizes Unicode NFKC, collapses whitespace runs, strips edges, and emits TEXT_NORMALIZED.",
    "caption normalization is not implemented",
    requires=("PIL",),
)
def test_caption_nfkc_normalized():
    mod = _input_module()
    buf = BytesIO()
    img = PIL_Image.new("RGB", (64, 64))
    img.save(buf, format="PNG")
    # "  fullwidth　spaces\t\nand   words  "
    raw = "  Headline \u3000 with   extra   spaces\t\n  "
    req = mod.build_input_request(image_bytes=buf.getvalue(), filename="test.png", caption=raw)
    assert req.caption == "Headline with extra spaces"
    assert req.caption_raw == raw
    assert mod.WarningCode.TEXT_NORMALIZED in req.warnings


@known_gap(
    "input_pipeline.rgb_conversion_alpha_flattened",
    "F3",
    "fast",
    "doc/04 §7.3",
    "RGBA image transparent background is composited over white (255, 255, 255) and emits ALPHA_FLATTENED.",
    "alpha channel flattening is not implemented",
    requires=("PIL",),
)
def test_rgb_conversion_alpha_flattened():
    mod = _input_module()
    buf = BytesIO()
    # Fully transparent RGBA pixel
    img = PIL_Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    img.save(buf, format="PNG")
    req = mod.build_input_request(image_bytes=buf.getvalue(), filename="test.png", caption="Caption text")
    assert req.image.image.mode == "RGB"
    pixel = req.image.image.getpixel((0, 0))
    assert pixel == (255, 255, 255)
    assert mod.WarningCode.ALPHA_FLATTENED in req.warnings


@known_gap(
    "input_pipeline.request_invariants",
    "F3",
    "fast",
    "doc/04 §9.2",
    "InputRequest satisfies UUID4 request_id, 64-char lowercase sha256, and timezone-aware created_at.",
    "InputRequest schema invariants are not implemented",
    requires=("PIL",),
)
def test_request_invariants():
    mod = _input_module()
    buf = BytesIO()
    img = PIL_Image.new("RGB", (64, 64), color="green")
    img.save(buf, format="JPEG")
    req = mod.build_input_request(image_bytes=buf.getvalue(), filename="photo.jpg", caption="Normal caption")
    import uuid
    # Valid UUID
    uuid.UUID(req.request_id)
    # Valid 64-hex SHA-256
    assert len(req.image.sha256) == 64 and all(c in "0123456789abcdef" for c in req.image.sha256)
    # Timezone-aware
    assert req.created_at.tzinfo is not None
    assert req.schema_version == 1
