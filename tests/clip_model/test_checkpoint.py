"""The classifier checkpoint loading contract and verification procedure of Document 03 §27 and §28.

Tests safe loading with weights_only=True, hash verification before
deserialization, strict state_dict loading, and the second-machine
verification tolerances.
"""

from pathlib import Path
import pytest

from tests.ids import case, known_gap

torch = pytest.importorskip("torch")


def _checkpoint_module():
    import importlib
    return importlib.import_module("checkpoint")


@known_gap(
    "checkpoint.module_exists",
    "F2",
    "fast",
    "doc/03 §27",
    "The project provides a checkpoint loading and verification module.",
    "there is no checkpoint module",
    requires=("torch",),
)
def test_checkpoint_module_exists():
    _checkpoint_module()


@known_gap(
    "checkpoint.safe_load_weights_only",
    "F2",
    "fast",
    "doc/03 §27.3",
    "Checkpoint weights are deserialized with weights_only=True on CPU without arbitrary code execution.",
    "safe checkpoint loading is not implemented",
    requires=("torch",),
)
def test_safe_load_weights_only(tmp_path):
    mod = _checkpoint_module()
    weights_path = tmp_path / "weights.pt"
    torch.save({"linear.weight": torch.randn(2, 1024), "linear.bias": torch.randn(2)}, weights_path)
    state = mod.load_weights_safe(weights_path)
    assert "linear.weight" in state


@known_gap(
    "checkpoint.verify_before_deserialize",
    "F2",
    "fast",
    "doc/03 §27.3",
    "File hashes are verified against release.json before any file is deserialized; tampered files raise CheckpointIntegrityError.",
    "pre-deserialization hash verification is not implemented",
    requires=("torch",),
)
def test_verify_before_deserialize(tmp_path):
    mod = _checkpoint_module()
    tampered_file = tmp_path / "weights.pt"
    tampered_file.write_bytes(b"corrupted or altered payload")
    manifest = {"weights.pt": {"sha256": "0" * 64}}
    with pytest.raises(Exception) as excinfo:
        mod.verify_checkpoint_integrity(tmp_path, manifest)
    assert "Integrity" in type(excinfo.value).__name__ or "hash" in str(excinfo.value).lower()


@known_gap(
    "checkpoint.strict_state_dict",
    "F2",
    "fast",
    "doc/03 §27.2",
    "Model state dicts are loaded with strict=True; unexpected or missing keys raise an error.",
    "strict state loading is not implemented",
    requires=("torch",),
)
def test_strict_state_dict():
    mod = _checkpoint_module()
    model = torch.nn.Linear(10, 2)
    bad_state = {"linear.extra_unexpected_key": torch.randn(2, 2)}
    with pytest.raises(Exception):
        mod.load_state_dict_strict(model, bad_state)


@known_gap(
    "checkpoint.verification_tolerances",
    "F2",
    "fast",
    "doc/03 §28.3",
    "Second-machine verification requires 100% class agreement, max score difference <= 1e-4, and probability difference <= 1e-4.",
    "verification criteria checks are not implemented",
    requires=("torch",),
)
def test_verification_tolerances():
    mod = _checkpoint_module()
    # Identical predictions pass
    ref_preds = [{"reference_class": "MATCHED", "reference_score": 0.2500, "reference_calibrated_probability": 0.2000}]
    actual_preds = [{"predicted_class": "MATCHED", "score": 0.25005, "calibrated_probability": 0.20004}]
    result = mod.evaluate_verification_agreement(ref_preds, actual_preds, tolerance=1e-4)
    assert result["passed"] is True

    # Differing class fails
    actual_mismatch_class = [{"predicted_class": "MISMATCHED", "score": 0.2500, "calibrated_probability": 0.2000}]
    result_class_fail = mod.evaluate_verification_agreement(ref_preds, actual_mismatch_class, tolerance=1e-4)
    assert result_class_fail["passed"] is False


@known_gap(
    "checkpoint.loaded_classifier_api",
    "F2",
    "fast",
    "doc/03 §27.1",
    "LoadedClassifier exposes model, config, predict, score, and predict_proba.",
    "LoadedClassifier interface is not implemented",
    requires=("torch",),
)
def test_loaded_classifier_api():
    mod = _checkpoint_module()
    cls = getattr(mod, "LoadedClassifier", None)
    assert cls is not None, "LoadedClassifier class is not defined"
    for member in ("model", "config", "predict", "score", "predict_proba"):
        assert hasattr(cls, member) or member in getattr(cls, "__annotations__", {}), "LoadedClassifier missing member: %s" % member
