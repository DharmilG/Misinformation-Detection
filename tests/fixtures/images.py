"""Synthetic image builders.

Nothing here is dataset content.  Every image is generated, so the suite
carries no third-party photography and no binary fixtures.
"""

import io
import os
import struct
import zlib
from pathlib import Path


def _require_pil():
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - import guard
        import pytest

        pytest.skip("Pillow is not installed")
    return Image


def solid_rgb(size=(64, 48), color=(120, 40, 200)):
    """A plain RGB image."""
    Image = _require_pil()
    return Image.new("RGB", size, color)


def rgba_with_transparent_pixel(size=(64, 48), colour=(0, 0, 0, 0)):
    """An RGBA image with a fully transparent pixel of a known colour.

    The stored colour is black, so discarding the alpha channel yields black
    while compositing onto white would yield white.  The two are
    distinguishable, which is the point.
    """
    Image = _require_pil()
    image = Image.new("RGBA", size, (0, 0, 0, 255))
    image.putpixel((0, 0), colour)
    return image


def grayscale(size=(64, 48)):
    Image = _require_pil()
    return Image.new("L", size, 128)


def cmyk(size=(64, 48)):
    Image = _require_pil()
    return Image.new("CMYK", size, (10, 20, 30, 5))


def exif_rotated(path, size=(64, 48), orientation=6):
    """A JPEG whose EXIF orientation tag is set.

    Orientation 6 means "rotate 90 degrees clockwise for display".
    """
    Image = _require_pil()
    image = Image.new("RGB", size, (200, 30, 30))
    exif = Image.Exif()
    exif[0x0112] = orientation
    image.save(path, "JPEG", exif=exif)
    return Path(path)


def wide_aspect(size=(800, 100)):
    """A very wide image, so the centre crop discards most of it."""
    Image = _require_pil()
    return Image.new("RGB", size, (10, 120, 200))


def animated_gif(path, frames=3):
    Image = _require_pil()
    images = [Image.new("RGB", (32, 32), (i * 60, 40, 40)) for i in range(frames)]
    images[0].save(path, save_all=True, append_images=images[1:], duration=100, loop=0)
    return Path(path)


def corrupt_bytes(path):
    """A file that is not a decodable image."""
    Path(path).write_bytes(b"this is definitely not an image, not even a little bit")
    return Path(path)


def truncated_jpeg(path):
    """A JPEG whose header is valid but whose data is cut off."""
    Image = _require_pil()
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_name = tmp.name
    Image.new("RGB", (128, 128), (10, 20, 30)).save(tmp_name, "JPEG", quality=95)
    data = Path(tmp_name).read_bytes()
    Path(path).write_bytes(data[: len(data) // 3])
    os.unlink(tmp_name)
    return Path(path)


def write(path, image):
    """Write an image, choosing a format from the suffix."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "JPEG" if path.suffix.lower() in (".jpg", ".jpeg") else path.suffix.lstrip(".").upper()
    image.save(path, fmt)
    return path
