"""
src/input_pipeline/validation.py
=================================
Image and caption validation.

Spec: 05_REPORTING_AND_UI.md §11 (input-rejection view) and
      04_INPUT_AND_INFERENCE_PIPELINE (implied validation contract).

Returns:
  - On success: (InputRequest, list[warning_codes])
  - On failure: (None, list[ValidationError])

A validation error means no report model is constructed.
Warning codes mean the request proceeds with a note rendered in the UI.

This module has no Streamlit imports.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from typing import Optional

from src.utils.config import (
    CLIP_MAX_TOKENS,
    MAX_CAPTION_CHARS,
    MAX_IMAGE_BYTES,
    MAX_LONG_EDGE_PX,
    MIN_CAPTION_CHARS,
    SUPPORTED_IMAGE_FORMATS,
)

try:
    from PIL import Image, ImageOps
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


# ── Dataclasses ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ValidationError:
    code: str          # machine code, e.g. "IMAGE_TOO_LARGE"
    message: str       # human-readable


@dataclass(frozen=True)
class _ImageInfo:
    image_sha256: str
    processed_bytes: bytes      # possibly resized / alpha-removed / EXIF-rotated
    original_width: int
    original_height: int
    original_filename: str


@dataclass(frozen=True)
class _CaptionInfo:
    original: str
    normalized: str
    caption_sha256: str


@dataclass(frozen=True)
class InputRequest:
    image: _ImageInfo
    caption: _CaptionInfo


# ── Image validation ──────────────────────────────────────────────────────

def _validate_image(
    image_bytes: bytes,
    filename: str,
) -> tuple[Optional[_ImageInfo], list[str], list[ValidationError]]:
    """
    Validate and preprocess an image.

    Returns (ImageInfo | None, warning_codes, errors).
    """
    warnings: list[str] = []
    errors: list[ValidationError] = []

    # 1. Size check (raw bytes)
    if len(image_bytes) > MAX_IMAGE_BYTES:
        mb = MAX_IMAGE_BYTES // (1024 * 1024)
        errors.append(ValidationError(
            code="IMAGE_TOO_LARGE",
            message=(
                f"The uploaded file is {len(image_bytes) // (1024 * 1024)} MB. "
                f"The maximum accepted size is {mb} MB."
            ),
        ))
        return None, warnings, errors

    if not _PIL_AVAILABLE:
        errors.append(ValidationError(
            code="PIL_UNAVAILABLE",
            message="Image processing library (Pillow) is not installed.",
        ))
        return None, warnings, errors

    # 2. Open and format check
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
        img = Image.open(io.BytesIO(image_bytes))  # re-open after verify
    except Exception as exc:
        errors.append(ValidationError(
            code="IMAGE_CORRUPT",
            message=f"The file could not be opened as an image: {exc}",
        ))
        return None, warnings, errors

    if img.format not in SUPPORTED_IMAGE_FORMATS:
        errors.append(ValidationError(
            code="IMAGE_FORMAT_UNSUPPORTED",
            message=(
                f"Image format '{img.format}' is not supported. "
                f"Accepted formats: {', '.join(SUPPORTED_IMAGE_FORMATS)}."
            ),
        ))
        return None, warnings, errors

    original_width, original_height = img.size

    # 3. EXIF orientation
    try:
        img = ImageOps.exif_transpose(img)
        if img.size != (original_width, original_height):
            warnings.append("EXIF_ROTATED")
    except Exception:
        pass  # non-fatal

    # 4. Alpha channel
    if img.mode in ("RGBA", "LA", "PA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "PA":
            img = img.convert("RGBA")
        bg.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
        img = bg
        warnings.append("ALPHA_FLATTENED")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # 5. Downscale
    w, h = img.size
    if max(w, h) > MAX_LONG_EDGE_PX:
        ratio = MAX_LONG_EDGE_PX / max(w, h)
        img = img.resize(
            (int(w * ratio), int(h * ratio)),
            Image.LANCZOS,
        )
        warnings.append("IMAGE_DOWNSCALED")

    # 6. Serialize back to JPEG bytes
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    processed_bytes = buf.getvalue()

    image_sha256 = hashlib.sha256(processed_bytes).hexdigest()

    return (
        _ImageInfo(
            image_sha256=image_sha256,
            processed_bytes=processed_bytes,
            original_width=original_width,
            original_height=original_height,
            original_filename=filename,
        ),
        warnings,
        errors,
    )


# ── Caption validation ────────────────────────────────────────────────────

def _validate_caption(
    caption: str,
) -> tuple[Optional[_CaptionInfo], list[str], list[ValidationError]]:
    """
    Validate and normalize a caption.

    Returns (CaptionInfo | None, warning_codes, errors).
    """
    warnings: list[str] = []
    errors: list[ValidationError] = []

    original = caption

    if not caption or len(caption.strip()) < MIN_CAPTION_CHARS:
        errors.append(ValidationError(
            code="CAPTION_TOO_SHORT",
            message=(
                f"Caption must be at least {MIN_CAPTION_CHARS} characters. "
                "Please enter a descriptive caption for the image."
            ),
        ))
        return None, warnings, errors

    if len(caption) > MAX_CAPTION_CHARS:
        errors.append(ValidationError(
            code="CAPTION_TOO_LONG",
            message=(
                f"Caption is {len(caption)} characters. "
                f"The maximum is {MAX_CAPTION_CHARS} characters."
            ),
        ))
        return None, warnings, errors

    # Normalise: collapse internal whitespace, strip edges
    import unicodedata
    normalized = " ".join(caption.split())
    normalized = unicodedata.normalize("NFC", normalized)

    if normalized != original.strip():
        warnings.append("TEXT_NORMALIZED")

    # CLIP token limit — we can't count tokens without the model loaded,
    # so use a conservative word-count proxy (CLIP_MAX_TOKENS tokens ≈ 77).
    # The classifier handles actual truncation; we only warn here.
    rough_token_estimate = len(normalized.split())
    if rough_token_estimate > CLIP_MAX_TOKENS:
        warnings.append("CAPTION_TRUNCATED_FOR_CLIP")

    caption_sha256 = hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    return (
        _CaptionInfo(
            original=original,
            normalized=normalized,
            caption_sha256=caption_sha256,
        ),
        warnings,
        errors,
    )


# ── Public API ────────────────────────────────────────────────────────────

def build_input_request(
    image_bytes: bytes,
    filename: str,
    caption: str,
) -> tuple[Optional[InputRequest], list[str], list[ValidationError]]:
    """
    Validate an (image, caption) pair.

    Returns:
        (InputRequest, warning_codes, [])   on success
        (None,         [],            errors) on failure

    Never raises — validation errors are returned in the list.
    """
    all_warnings: list[str] = []
    all_errors: list[ValidationError] = []

    image_info, img_warnings, img_errors = _validate_image(image_bytes, filename)
    all_warnings.extend(img_warnings)
    all_errors.extend(img_errors)

    caption_info, cap_warnings, cap_errors = _validate_caption(caption)
    all_warnings.extend(cap_warnings)
    all_errors.extend(cap_errors)

    if all_errors:
        return None, all_warnings, all_errors

    assert image_info is not None
    assert caption_info is not None

    return InputRequest(image=image_info, caption=caption_info), all_warnings, []
