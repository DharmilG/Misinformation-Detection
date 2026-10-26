"""The leakage grouping and audit contract of Document 02 §10 and §11.

Tests that leakage detection produces four distinct exposure fields,
uses the defined bipartite link sets, evaluates giant component feasibility,
and reports overlap without silently repairing the official splits.
"""

from pathlib import Path
import pytest

from tests.ids import case, known_gap


def _leakage_module():
    import importlib
    return importlib.import_module("leakage")


@known_gap(
    "leakage.module_exists",
    "F5",
    "fast",
    "doc/02 §10.1",
    "The project provides a leakage grouping and exposure audit module.",
    "there is no leakage module",
)
def test_leakage_module_exists():
    _leakage_module()


@known_gap(
    "leakage.link_sets_defined",
    "F5",
    "fast",
    "doc/02 §10.4",
    "The link set configurations L0 (image only), L1 (image + caption source), and L2 (image + text) are defined.",
    "the link sets are not defined",
)
def test_link_sets_defined():
    mod = _leakage_module()
    assert hasattr(mod, "LINK_SETS"), "LINK_SETS is not defined"
    assert set(mod.LINK_SETS.keys()) >= {"L0", "L1", "L2"}


@known_gap(
    "leakage.exposure_fields_individual",
    "F5",
    "fast",
    "doc/02 §11.2.1",
    "Exposure is reported as four distinct fields, never collapsed into a single metric.",
    "the exposure report format is not implemented",
)
def test_exposure_fields_individual():
    mod = _leakage_module()
    mock_splits = {
        "train": [{"image_id": "img_1", "caption_id": "cap_1"}],
        "test": [{"image_id": "img_1", "caption_id": "cap_2"}],
    }
    report = mod.audit_exposure(mock_splits)
    for field in (
        "exposure_exact_image_pct",
        "exposure_near_image_pct",
        "exposure_shared_caption_source_pct",
        "exposure_any_pct",
    ):
        assert field in report, "missing exposure field: %s" % field
        assert 0.0 <= report[field] <= 100.0


@known_gap(
    "leakage.exposure_any_is_union",
    "F5",
    "fast",
    "doc/02 §11.2.1",
    "The exposure_any_pct field is computed as a set union over affected records, not an arithmetic average.",
    "exposure_any calculation is not implemented",
)
def test_exposure_any_is_union():
    mod = _leakage_module()
    # If one sample matches both exact image and shared caption, it counts as 1 affected record
    records = [
        {"sample_id": "s_1", "exact_image_leak": True, "shared_caption_leak": True},
        {"sample_id": "s_2", "exact_image_leak": False, "shared_caption_leak": False},
    ]
    exposure = mod.compute_exposure_union(records)
    assert exposure["exposure_any_pct"] == 50.0  # 1 out of 2 records affected


@known_gap(
    "leakage.decision_rule_threshold",
    "F5",
    "fast",
    "doc/02 §10.6",
    "A link set is feasible for grouping only if its largest connected component is <= 10% of records.",
    "the grouping feasibility decision rule is not implemented",
)
def test_decision_rule_threshold():
    mod = _leakage_module()
    # Under 10% passes feasibility
    assert mod.is_feasible_component_size(largest_share=0.08, threshold=0.10) is True
    # Over 10% (giant component) fails feasibility
    assert mod.is_feasible_component_size(largest_share=0.25, threshold=0.10) is False


@known_gap(
    "leakage.audit_does_not_modify",
    "F5",
    "fast",
    "doc/02 §11.3",
    "The leakage audit strictly reports overlap without altering row assignments or dropping records.",
    "the audit safety invariant is not implemented",
)
def test_audit_does_not_modify():
    mod = _leakage_module()
    mock_splits = {
        "train": [{"sample_id": "t1", "image_id": "shared_img"}],
        "test": [{"sample_id": "te1", "image_id": "shared_img"}],
    }
    before_len = len(mock_splits["test"])
    mod.audit_exposure(mock_splits)
    assert len(mock_splits["test"]) == before_len, "audit modified the test split in place"
