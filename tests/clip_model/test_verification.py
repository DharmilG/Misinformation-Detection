"""The second-machine verification procedure of Document 03 §28.

Tests verification execution using the standard prediction path, JSON report schema,
tampered fixture rejection prior to inference, and quantitative acceptance criteria.
"""

from pathlib import Path
import pytest

from tests.ids import case, known_gap


def _verification_module():
    import importlib
    return importlib.import_module("classifier_verification")


@known_gap(
    "verification.module_exists",
    "F2",
    "fast",
    "doc/03 §28",
    "The project provides a verification runner module.",
    "there is no verification module",
)
def test_verification_module_exists():
    _verification_module()


@known_gap(
    "verification.normal_prediction_path",
    "F2",
    "fast",
    "doc/03 §28.2",
    "Verification uses the standard LoadedClassifier prediction path, not a special verification bypass.",
    "verification runner is not implemented",
)
def test_normal_prediction_path():
    mod = _verification_module()
    assert hasattr(mod, "run_verification")
    assert mod.USES_NORMAL_INFERENCE_PATH is True


@known_gap(
    "verification.report_schema",
    "F2",
    "fast",
    "doc/03 §28.5",
    "The verification report contains all required schema fields including tolerance, environment, and differences.",
    "verification report schema is not implemented",
)
def test_report_schema():
    mod = _verification_module()
    report = mod.generate_empty_report()
    required_keys = {
        "verified_at",
        "release_name",
        "model_version",
        "run_id",
        "artifact_sha256",
        "environment",
        "tolerance",
        "samples_checked",
        "class_agreements",
        "max_score_difference",
        "max_probability_difference",
        "result",
    }
    missing = required_keys - set(report.keys())
    assert not missing, "verification report missing keys: %s" % missing


@known_gap(
    "verification.fixture_hash_check_first",
    "F2",
    "fast",
    "doc/03 §28.2",
    "Tampered fixture images or captions fail verification before any model prediction is executed.",
    "fixture hash checking is not implemented",
)
def test_fixture_hash_check_first(tmp_path):
    mod = _verification_module()
    tampered_img = tmp_path / "img0.jpg"
    tampered_img.write_bytes(b"altered bits")
    manifest = [{"path": "img0.jpg", "sha256": "0" * 64}]
    with pytest.raises(Exception):
        mod.verify_fixture_hashes(tmp_path, manifest)


@known_gap(
    "verification.acceptance_criteria",
    "F2",
    "fast",
    "doc/03 §28.3",
    "Verification passes only with 100% class agreement, score diff <= 1e-4, and prob diff <= 1e-4.",
    "acceptance criteria enforcement is not implemented",
)
def test_acceptance_criteria():
    mod = _verification_module()
    # Any class disagreement fails
    res_diff_class = mod.check_acceptance(
        class_mismatches=1,
        max_score_diff=0.0,
        max_prob_diff=0.0,
        score_tol=1e-4,
        prob_tol=1e-4,
    )
    assert res_diff_class is False

    # Score diff exceeding 1e-4 fails
    res_large_score = mod.check_acceptance(
        class_mismatches=0,
        max_score_diff=2e-4,
        max_prob_diff=0.0,
        score_tol=1e-4,
        prob_tol=1e-4,
    )
    assert res_large_score is False

    # Perfect or sub-tolerance run passes
    res_pass = mod.check_acceptance(
        class_mismatches=0,
        max_score_diff=5e-5,
        max_prob_diff=5e-5,
        score_tol=1e-4,
        prob_tol=1e-4,
    )
    assert res_pass is True
