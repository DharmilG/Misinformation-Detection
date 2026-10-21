"""Repository-root pytest configuration.

This module is imported before test collection, so it is the only place that can
influence *whether* a file is collected at all.  It handles three things:

1.  Making ``src/clip_model`` importable, so the feature-extraction scripts in
    that directory can be imported by name.
2.  Keeping the script-like files that live beside the implementation out of
    collection.  They assert at import time and depend on artefacts that a
    developer may not have generated, so collecting them turns a missing file
    into a whole-suite collection error.  ``tests/test_harness.py`` asserts that
    the exclusion list stays exactly this small and deliberate.
3.  Materialising the sample artefacts the scripts expect, so those scripts run
    rather than exiting with "sample not found".
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

#: Files beside the implementation that look like tests but are scripts.  They
#: execute their assertions at import time and read artefacts that may not
#: exist, so collecting them produces a collection error for the whole suite.
#: Kept deliberately small; ``tests/test_harness.py`` asserts this list.
SCRIPT_LIKE_MODULES = [
    "src/clip_model/test_clip.py",
    "src/clip_model/test_saved_features.py",
]

collect_ignore_glob = [str(p).replace("\\", "/") for p in SCRIPT_LIKE_MODULES]


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
