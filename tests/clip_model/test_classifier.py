"""The classifier candidate architectures, feature construction, thresholding, and calibration (Document 03 §4, §10, §12).

Tests candidate families A, B, C, D-B, and D-C, explicit interaction
features in Family C, threshold selection on uncalibrated scores, and
calibration invariants.
"""

import pytest

from tests.ids import case

torch = pytest.importorskip("torch")


def _classifier_module():
    import importlib
    return importlib.import_module("classifier")


@case(
    "classifier.module_exists",
    "F2",
    "fast",
    "doc/03 §4",
    "The project provides a classifier module defining head architectures and feature builders.",
    requires=("torch",),
)
def test_classifier_module_exists():
    _classifier_module()


@case(
    "classifier.candidate_families",
    "F2",
    "fast",
    "doc/03 §4.1",
    "The pre-registered candidates A, B, C, D-B, and D-C are defined with their expected input dimensions.",
    requires=("torch",),
)
def test_candidate_families():
    mod = _classifier_module()
    families = mod.CANDIDATE_FAMILIES
    assert set(families.keys()) == {"A", "B", "C", "D-B", "D-C"}
    assert families["A"]["input_dim"] == 1
    assert families["B"]["input_dim"] == 1024
    assert families["C"]["input_dim"] == 2048
    assert families["D-B"]["input_dim"] == 1024
    assert families["D-C"]["input_dim"] == 2048


@case(
    "classifier.family_c_features",
    "F2",
    "fast",
    "doc/03 §4.2",
    "Family C feature construction concatenates [v_i ; v_t ; v_i ⊙ v_t ; |v_i − v_t|] into a 2048-d vector.",
    requires=("torch",),
)
def test_family_c_features():
    mod = _classifier_module()
    v_i = torch.randn(1, 512)
    v_t = torch.randn(1, 512)
    z = mod.build_features("C", v_i, v_t)
    assert tuple(z.shape) == (1, 2048)
    expected_prod = v_i * v_t
    expected_diff = torch.abs(v_i - v_t)
    assert torch.allclose(z[:, 1024:1536], expected_prod)
    assert torch.allclose(z[:, 1536:2048], expected_diff)


@case(
    "classifier.family_d_architecture",
    "F2",
    "fast",
    "doc/03 §4.3",
    "Family D models use Linear -> ReLU -> Dropout -> Linear(2) producing 2-class logits.",
    requires=("torch",),
)
def test_family_d_architecture():
    mod = _classifier_module()
    model = mod.create_model("D-B", hidden_dim=256, dropout=0.1)
    x = torch.randn(4, 1024)
    logits = model(x)
    assert tuple(logits.shape) == (4, 2)


@case(
    "classifier.operating_threshold_selection",
    "F2",
    "fast",
    "doc/03 §10.1",
    "The operating threshold is selected on validation data maximizing validation macro-F1 on the uncalibrated score.",
    requires=("torch",),
)
def test_operating_threshold_selection():
    mod = _classifier_module()
    val_scores = [0.1, 0.4, 0.6, 0.9]
    val_labels = [0, 0, 1, 1]
    threshold = mod.select_threshold(val_scores, val_labels)
    assert 0.4 <= threshold <= 0.6


@case(
    "classifier.calibration_does_not_change_label",
    "F2",
    "fast",
    "doc/03 §12.1",
    "Temperature calibration changes the displayed probability but preserves predicted labels identically.",
    requires=("torch",),
)
def test_calibration_does_not_change_label():
    mod = _classifier_module()
    logits = torch.tensor([[-1.5, 1.2], [2.0, -0.8]])
    threshold = 0.5
    raw_labels = mod.predict_labels(logits, threshold=threshold)
    calibrated_labels = mod.predict_calibrated_labels(logits, temperature=1.8, threshold=threshold)
    assert raw_labels == calibrated_labels
