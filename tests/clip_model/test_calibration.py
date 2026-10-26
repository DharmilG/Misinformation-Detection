"""The temperature calibration fitting and evaluation of Document 03 §12.

Tests temperature scaling formula, validation-only fitting, Expected Calibration
Error (ECE) 10-bin calculation, and proper scoring rules (Brier score).
"""

import pytest

from tests.ids import case, known_gap

torch = pytest.importorskip("torch")


def _calibration_module():
    import importlib
    return importlib.import_module("classifier_calibration")


@known_gap(
    "calibration.module_exists",
    "F2",
    "fast",
    "doc/03 §12",
    "The project provides a calibration module implementing temperature scaling and metrics.",
    "there is no calibration module",
    requires=("torch",),
)
def test_calibration_module_exists():
    _calibration_module()


@known_gap(
    "calibration.temperature_scaling_formula",
    "F2",
    "fast",
    "doc/03 §12.1",
    "Calibrated probability is softmax(logits / T)[positive_class] with T >= 1.",
    "temperature scaling calculation is not implemented",
    requires=("torch",),
)
def test_temperature_scaling_formula():
    mod = _calibration_module()
    logits = torch.tensor([[0.0, 2.0]])
    temperature = 2.0
    # logits / T = [0.0, 1.0] -> softmax = [e^0 / (e^0 + e^1), e^1 / (e^0 + e^1)]
    prob = mod.compute_calibrated_probability(logits, temperature=temperature, positive_class_idx=1)
    import math
    expected_prob = math.exp(1.0) / (math.exp(0.0) + math.exp(1.0))
    assert abs(prob.item() - expected_prob) < 1e-5


@known_gap(
    "calibration.fit_on_validation_only",
    "F2",
    "fast",
    "doc/03 §12.2",
    "Temperature is fit by minimizing negative log-likelihood over T using validation predictions only.",
    "temperature optimizer is not implemented",
    requires=("torch",),
)
def test_fit_on_validation_only():
    mod = _calibration_module()
    val_logits = torch.tensor([[1.0, -1.0], [-2.0, 2.0], [0.5, -0.5]])
    val_labels = torch.tensor([0, 1, 0])
    t_opt = mod.fit_temperature(val_logits, val_labels)
    assert t_opt >= 1.0


@known_gap(
    "calibration.ece_ten_bins",
    "F2",
    "fast",
    "doc/03 §12.4",
    "Expected Calibration Error (ECE) uses 10 equal-width bins on predicted probabilities.",
    "ECE calculation is not implemented",
    requires=("torch",),
)
def test_ece_ten_bins():
    mod = _calibration_module()
    probs = [0.05, 0.15, 0.85, 0.95]
    labels = [0, 0, 1, 1]
    ece, bin_counts = mod.compute_ece(probs, labels, num_bins=10)
    assert len(bin_counts) == 10
    assert sum(bin_counts) == 4
    assert 0.0 <= ece <= 1.0


@known_gap(
    "calibration.brier_score",
    "F2",
    "fast",
    "doc/03 §12.4",
    "Brier score is computed as the mean squared difference between predicted probability and binary outcome.",
    "Brier score calculation is not implemented",
    requires=("torch",),
)
def test_brier_score():
    mod = _calibration_module()
    probs = [0.1, 0.9]
    labels = [0, 1]
    # ( (0.1 - 0)^2 + (0.9 - 1)^2 ) / 2 = (0.01 + 0.01) / 2 = 0.01
    brier = mod.compute_brier_score(probs, labels)
    assert abs(brier - 0.01) < 1e-5
