"""The prediction context warnings and crop-retention calculation of Document 04 §24.

Tests crop-retention fraction formula min(W, H) / max(W, H), default 0.5 threshold trigger,
and warning list deduplication with order preservation.
"""

import pytest

from tests.ids import case


def _warnings_module():
    import importlib
    return importlib.import_module("inference_warnings")


@case(
    "input_warnings.module_exists",
    "F2",
    "fast",
    "doc/04 §24",
    "The project provides an inference warnings module calculating context alerts.",
)
def test_warnings_module_exists():
    _warnings_module()


@case(
    "input_warnings.crop_retention_formula",
    "F2",
    "fast",
    "doc/04 §24.1",
    "retained_area_fraction is computed as min(W, H) / max(W, H) from original dimensions.",
)
def test_crop_retention_formula():
    mod = _warnings_module()
    # 1200 x 600 image -> min/max = 600 / 1200 = 0.5
    fraction = mod.calculate_crop_retention(width=1200, height=600)
    assert abs(fraction - 0.50) < 1e-5

    # 1600 x 400 image -> min/max = 400 / 1600 = 0.25
    fraction_extreme = mod.calculate_crop_retention(width=1600, height=400)
    assert abs(fraction_extreme - 0.25) < 1e-5


@case(
    "input_warnings.crop_retention_threshold",
    "F2",
    "fast",
    "doc/04 §24.1",
    "Images with retained area fraction < 0.5 emit LOW_CROP_RETENTION warning.",
)
def test_crop_retention_threshold():
    mod = _warnings_module()
    # 800 x 600 -> fraction = 0.75 >= 0.5 (no warning)
    warns_normal = mod.check_crop_warnings(width=800, height=600, threshold=0.5)
    assert "LOW_CROP_RETENTION" not in warns_normal

    # 1500 x 500 -> fraction = 0.333 < 0.5 (emits warning)
    warns_extreme = mod.check_crop_warnings(width=1500, height=500, threshold=0.5)
    assert "LOW_CROP_RETENTION" in warns_extreme


@case(
    "input_warnings.deduplicated_order_preserved",
    "F2",
    "fast",
    "doc/04 §9.2, §24",
    "Warning collections preserve emission order and eliminate duplicate warning codes.",
)
def test_deduplicated_order_preserved():
    mod = _warnings_module()
    raw_warnings = ["TEXT_NORMALIZED", "IMAGE_DOWNSCALED", "TEXT_NORMALIZED", "LOW_CROP_RETENTION"]
    deduped = mod.clean_warnings(raw_warnings)
    assert list(deduped) == ["TEXT_NORMALIZED", "IMAGE_DOWNSCALED", "LOW_CROP_RETENTION"]
