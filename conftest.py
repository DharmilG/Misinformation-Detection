"""Repository-root pytest configuration.

This module runs before test collection.  It does two things:

1.  Making ``src/clip_model`` importable, so the implementation modules in that
    directory can be imported by bare name from ``tests/clip_model``.
2.  Materialising the sample artefact the scripts in ``src/clip_model`` read, so
    those scripts run rather than exiting with "sample not found".

It deliberately excludes nothing.  ``pytest.ini`` sets ``testpaths = tests``, so
a bare run reaches only the suite, and ``src/`` holds no file named
``test_*.py`` for anything to trip over.  ``tests/spec/test_suite.py`` asserts that
second half: a stray script beside the implementation fails the run with a
message naming it, rather than being quietly ignored.
"""

import io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CLIP_DIR = ROOT / "src" / "clip_model"
SAMPLE_DIR = ROOT / "data" / "sample"

if str(CLIP_DIR) not in sys.path:
    sys.path.insert(0, str(CLIP_DIR))

def _ensure_sample_image() -> Path:
    """Create ``data/sample/test_3.webp`` if it is not already there.

    The sample scripts in ``src/clip_model`` read this exact path.  It is
    generated rather than committed so that no binary artefact and no
    dataset content enters the repository.
    """
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    path = SAMPLE_DIR / "test_3.webp"
    if path.exists():
        return path
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return path
    image = Image.new("RGB", (320, 240), (36, 54, 96))
    draw = ImageDraw.Draw(image)
    draw.rectangle([40, 40, 280, 200], fill=(214, 226, 240))
    draw.ellipse([120, 90, 200, 170], fill=(200, 60, 60))
    try:
        image.save(path, "WEBP", quality=90)
    except Exception:
        image.save(path, "PNG")
    return path


_ensure_sample_image()
