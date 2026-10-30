"""
tests/test_input_validation.py
==============================
Tests for src/input_pipeline/validation.py.

Spec: 05_REPORTING_AND_UI §11, §13.
"""
from __future__ import annotations

import io
from PIL import Image
import pytest

from src.input_pipeline.validation import (
    ValidationError,
    build_input_request,
)
from src.utils.config import MAX_CAPTION_CHARS, MAX_IMAGE_BYTES, MIN_CAPTION_CHARS


def _make_dummy_image_bytes(
    width: int = 100,
    height: int = 100,
    fmt: str = "JPEG",
    mode: str = "RGB",
) -> bytes:
    img = Image.new(mode, (width, height), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


class TestInputValidation:
    def test_valid_input_succeeds(self) -> None:
        img_bytes = _make_dummy_image_bytes()
        req, warnings, errors = build_input_request(
            img_bytes,
            "test.jpg",
            "This is a valid test caption for testing.",
        )
        assert errors == []
        assert req is not None
        assert req.image.original_filename == "test.jpg"
        assert req.image.original_width == 100
        assert req.image.original_height == 100
        assert req.caption.original == "This is a valid test caption for testing."
        assert req.caption.normalized == "This is a valid test caption for testing."

    def test_corrupt_image_fails(self) -> None:
        req, warnings, errors = build_input_request(
            b"not a real image stream",
            "bad.jpg",
            "Valid caption",
        )
        assert req is None
        assert any(e.code == "IMAGE_CORRUPT" for e in errors)

    def test_image_too_large_fails(self) -> None:
        huge_bytes = b"0" * (MAX_IMAGE_BYTES + 1024)
        req, warnings, errors = build_input_request(
            huge_bytes,
            "huge.jpg",
            "Valid caption",
        )
        assert req is None
        assert any(e.code == "IMAGE_TOO_LARGE" for e in errors)

    def test_caption_too_short_fails(self) -> None:
        img_bytes = _make_dummy_image_bytes()
        req, warnings, errors = build_input_request(
            img_bytes,
            "test.jpg",
            "hi",
        )
        assert req is None
        assert any(e.code == "CAPTION_TOO_SHORT" for e in errors)

    def test_empty_caption_fails(self) -> None:
        img_bytes = _make_dummy_image_bytes()
        req, warnings, errors = build_input_request(
            img_bytes,
            "test.jpg",
            "   ",
        )
        assert req is None
        assert any(e.code == "CAPTION_TOO_SHORT" for e in errors)

    def test_caption_too_long_fails(self) -> None:
        img_bytes = _make_dummy_image_bytes()
        long_cap = "a" * (MAX_CAPTION_CHARS + 10)
        req, warnings, errors = build_input_request(
            img_bytes,
            "test.jpg",
            long_cap,
        )
        assert req is None
        assert any(e.code == "CAPTION_TOO_LONG" for e in errors)

    def test_alpha_flattened_warning(self) -> None:
        img_bytes = _make_dummy_image_bytes(mode="RGBA", fmt="PNG")
        req, warnings, errors = build_input_request(
            img_bytes,
            "alpha.png",
            "Valid caption",
        )
        assert errors == []
        assert req is not None
        assert "ALPHA_FLATTENED" in warnings

    def test_image_downscaled_warning(self) -> None:
        img_bytes = _make_dummy_image_bytes(width=1600, height=1200)
        req, warnings, errors = build_input_request(
            img_bytes,
            "large.jpg",
            "Valid caption",
        )
        assert errors == []
        assert req is not None
        assert "IMAGE_DOWNSCALED" in warnings

    def test_text_normalized_warning(self) -> None:
        img_bytes = _make_dummy_image_bytes()
        req, warnings, errors = build_input_request(
            img_bytes,
            "test.jpg",
            "  Lots   of    spaces   here  ",
        )
        assert errors == []
        assert req is not None
        assert "TEXT_NORMALIZED" in warnings
        assert req.caption.normalized == "Lots of spaces here"
        assert req.caption.original == "  Lots   of    spaces   here  "

    def test_caption_truncated_for_clip_warning(self) -> None:
        img_bytes = _make_dummy_image_bytes()
        # 85 words, exceeds 77
        long_words = "word " * 85
        req, warnings, errors = build_input_request(
            img_bytes,
            "test.jpg",
            long_words.strip(),
        )
        assert errors == []
        assert req is not None
        assert "CAPTION_TRUNCATED_FOR_CLIP" in warnings
