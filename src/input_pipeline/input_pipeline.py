"""Input validation, image and text normalization, and request schema of Document 04 §6-§10.

Provides build_input_request to decode and validate raw user uploads into a single,
immutable InputRequest object consumed by downstream inference stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
from io import BytesIO
from pathlib import Path
import re
import unicodedata
import uuid

from PIL import Image, ImageFile, ImageOps

ImageFile.LOAD_TRUNCATED_IMAGES = False


class InputErrorCode(str, Enum):
    EMPTY_FILE = "EMPTY_FILE"
    IMAGE_TOO_LARGE = "IMAGE_TOO_LARGE"
    IMAGE_TOO_MANY_PIXELS = "IMAGE_TOO_MANY_PIXELS"
    IMAGE_TOO_SMALL = "IMAGE_TOO_SMALL"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    ANIMATED_IMAGE = "ANIMATED_IMAGE"
    CORRUPT_IMAGE = "CORRUPT_IMAGE"
    CAPTION_EMPTY = "CAPTION_EMPTY"
    CAPTION_TOO_SHORT = "CAPTION_TOO_SHORT"
    CAPTION_TOO_LONG = "CAPTION_TOO_LONG"


class WarningCode(str, Enum):
    IMAGE_DOWNSCALED = "IMAGE_DOWNSCALED"
    EXIF_ROTATED = "EXIF_ROTATED"
    ALPHA_FLATTENED = "ALPHA_FLATTENED"
    TEXT_NORMALIZED = "TEXT_NORMALIZED"
    LOW_CROP_RETENTION = "LOW_CROP_RETENTION"


class InputValidationError(ValueError):
    """Raised for any user-correctable input problem."""

    def __init__(self, code: InputErrorCode, field: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.field = field
        self.message = message

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True)
class InputConfig:
    max_upload_bytes: int = 10 * 1024 * 1024  # 10 MB
    max_pixels: int = 40_000_000             # 40 MP
    min_dimension: int = 32                  # min width & height
    max_dimension: int = 2048                # longest side after downscale
    min_caption_chars: int = 3
    max_caption_chars: int = 1000


DEFAULT_INPUT_CONFIG = InputConfig()


@dataclass(frozen=True)
class ImageInput:
    image: Image.Image
    format: str
    mime_type: str
    filename: str
    width: int
    height: int
    original_width: int
    original_height: int
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class InputRequest:
    request_id: str
    image: ImageInput
    caption_raw: str
    caption: str
    warnings: tuple[WarningCode, ...]
    created_at: datetime
    schema_version: int = 1


def build_input_request(
    image_bytes: bytes,
    filename: str,
    caption: str,
    config: InputConfig = DEFAULT_INPUT_CONFIG,
) -> InputRequest:
    """Validate and normalize one image-caption pair.

    Raises:
        InputValidationError: for any user-correctable input problem.
        TypeError: if arguments have the wrong type.
    """
    if not isinstance(image_bytes, (bytes, bytearray)):
        raise TypeError("image_bytes must be bytes")
    if not isinstance(filename, str):
        raise TypeError("filename must be str")
    if not isinstance(caption, str):
        raise TypeError("caption must be str")

    # 1. Byte-level checks before decode
    if len(image_bytes) == 0:
        raise InputValidationError(
            InputErrorCode.EMPTY_FILE, "image", "Uploaded file is empty (0 bytes)."
        )
    if len(image_bytes) > config.max_upload_bytes:
        raise InputValidationError(
            InputErrorCode.IMAGE_TOO_LARGE,
            "image",
            f"Image exceeds maximum upload limit ({config.max_upload_bytes} bytes).",
        )

    # 2. Header inspection & decode
    try:
        buffer = BytesIO(image_bytes)
        img = Image.open(buffer)
    except (Image.DecompressionBombError,):
        raise InputValidationError(
            InputErrorCode.IMAGE_TOO_MANY_PIXELS,
            "image",
            "Image exceeds maximum pixel count limit.",
        )
    except Exception:
        raise InputValidationError(
            InputErrorCode.UNSUPPORTED_FORMAT,
            "image",
            "Unsupported or corrupt image format.",
        )

    # 3. Check for animated image
    if getattr(img, "n_frames", 1) > 1:
        raise InputValidationError(
            InputErrorCode.ANIMATED_IMAGE,
            "image",
            "Multi-frame animated images are not supported.",
        )

    # 4. Check supported formats (JPEG, PNG only)
    fmt = (img.format or "").upper()
    if fmt not in {"JPEG", "PNG"}:
        raise InputValidationError(
            InputErrorCode.UNSUPPORTED_FORMAT,
            "image",
            f"Format '{fmt}' is not supported. Only JPEG and PNG are accepted.",
        )

    # 5. Dimension & pixel count checks on header
    w, h = img.size
    if w * h > config.max_pixels:
        raise InputValidationError(
            InputErrorCode.IMAGE_TOO_MANY_PIXELS,
            "image",
            f"Image pixel count ({w * h}) exceeds maximum ({config.max_pixels}).",
        )
    if w < config.min_dimension or h < config.min_dimension:
        raise InputValidationError(
            InputErrorCode.IMAGE_TOO_SMALL,
            "image",
            f"Image dimension ({w}x{h}) is smaller than minimum ({config.min_dimension} px).",
        )

    # 6. Verify and load
    try:
        img.verify()
        buffer.seek(0)
        img = Image.open(buffer)
        img.load()
    except (Image.DecompressionBombError,):
        raise InputValidationError(
            InputErrorCode.IMAGE_TOO_MANY_PIXELS,
            "image",
            "Image exceeds maximum pixel count limit.",
        )
    except Exception:
        raise InputValidationError(
            InputErrorCode.CORRUPT_IMAGE,
            "image",
            "Image data is corrupt or cannot be decoded.",
        )

    warnings: list[WarningCode] = []

    # 7. EXIF orientation
    try:
        exif = img.getexif()
        orientation = exif.get(0x0112)
        if orientation and orientation != 1:
            warnings.append(WarningCode.EXIF_ROTATED)
        transposed = ImageOps.exif_transpose(img)
        if transposed is not None:
            img = transposed
    except Exception:
        pass

    orig_w, orig_h = img.size

    # 8. RGB conversion & alpha flattening
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        white_bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        composed = Image.alpha_composite(white_bg, rgba)
        img = composed.convert("RGB")
        warnings.append(WarningCode.ALPHA_FLATTENED)
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # 9. Downscale to max_dimension preserving aspect ratio
    curr_w, curr_h = img.size
    longest = max(curr_w, curr_h)
    if longest > config.max_dimension:
        scale = config.max_dimension / longest
        new_w = max(1, round(curr_w * scale))
        new_h = max(1, round(curr_h * scale))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        warnings.append(WarningCode.IMAGE_DOWNSCALED)

    # 10. Hash normalized pixels
    digest = hashlib.sha256()
    digest.update(f"{img.width}x{img.height}:RGB:".encode("ascii"))
    digest.update(img.tobytes())
    img_sha256 = digest.hexdigest()

    # 11. Filename sanitization and mime
    clean_fn = Path(filename).name
    clean_fn = "".join(c for c in clean_fn if c.isprintable())[:255]
    if not clean_fn:
        clean_fn = "upload"
    mime_type = "image/jpeg" if fmt == "JPEG" else "image/png"

    # 12. Caption normalization
    raw_caption = caption
    text = raw_caption.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFKC", text)
    for bad_ch in ("\u200b", "\ufeff", "\u00ad"):
        text = text.replace(bad_ch, "")
    cleaned_chars = []
    for c in text:
        if c in ("\n", "\r", "\t"):
            cleaned_chars.append(" ")
        elif unicodedata.category(c) == "Cc":
            continue
        else:
            cleaned_chars.append(c)
    text = "".join(cleaned_chars)
    text = re.sub(r"\s+", " ", text)
    norm_caption = text.strip()

    # 13. Caption length validation
    if len(norm_caption) == 0:
        raise InputValidationError(
            InputErrorCode.CAPTION_EMPTY,
            "caption",
            "Caption is empty or contains only whitespace.",
        )
    if len(norm_caption) < config.min_caption_chars:
        raise InputValidationError(
            InputErrorCode.CAPTION_TOO_SHORT,
            "caption",
            f"Caption has fewer than {config.min_caption_chars} characters after normalization.",
        )
    if len(norm_caption) > config.max_caption_chars:
        raise InputValidationError(
            InputErrorCode.CAPTION_TOO_LONG,
            "caption",
            f"Caption exceeds {config.max_caption_chars} characters after normalization.",
        )
    if norm_caption != raw_caption:
        warnings.append(WarningCode.TEXT_NORMALIZED)

    # 14. Deduplicate warnings preserving order
    deduped_warnings = tuple(dict.fromkeys(warnings).keys())

    image_input = ImageInput(
        image=img,
        format=fmt,
        mime_type=mime_type,
        filename=clean_fn,
        width=img.width,
        height=img.height,
        original_width=orig_w,
        original_height=orig_h,
        size_bytes=len(image_bytes),
        sha256=img_sha256,
    )

    return InputRequest(
        request_id=str(uuid.uuid4()),
        image=image_input,
        caption_raw=raw_caption,
        caption=norm_caption,
        warnings=deduped_warnings,
        created_at=datetime.now(timezone.utc),
        schema_version=1,
    )
