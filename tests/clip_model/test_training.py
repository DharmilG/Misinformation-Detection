"""The classifier training protocol, seeding, class weighting, and model selection (Document 03 §6, §7, §9).

Tests deterministic CPU training, pre-registered seeds 42–46,
inverse-frequency class weighting, and the two-stage selection protocol.
"""

import pytest

from tests.ids import case

torch = pytest.importorskip("torch")


def _training_module():
    import importlib
    return importlib.import_module("training")


@case(
    "training.module_exists",
    "F2",
    "fast",
    "doc/03 §7",
    "The project provides a training module implementing optimizer setup, training loop, and selection.",
    requires=("torch",),
)
def test_training_module_exists():
    _training_module()


@case(
    "training.seeds_fixed",
    "F2",
    "fast",
    "doc/03 §7.1",
    "The training seeds are fixed in advance to [42, 43, 44, 45, 46].",
    requires=("torch",),
)
def test_seeds_fixed():
    mod = _training_module()
    assert list(mod.SEEDS) == [42, 43, 44, 45, 46]


@case(
    "training.cpu_only_deterministic",
    "F2",
    "fast",
    "doc/03 §7.2, §7.5",
    "Head training defaults to CPU and enables deterministic algorithms for bitwise repeatability.",
    requires=("torch",),
)
def test_cpu_only_deterministic():
    mod = _training_module()
    config = mod.get_training_config()
    assert config["device"] == "cpu"
    assert config["deterministic"] is True


@case(
    "training.inverse_frequency_class_weights",
    "F2",
    "fast",
    "doc/03 §6.2",
    "Class weights follow w_c = N / (C * n_c) computed from the training split only and sum to C under true frequencies.",
    requires=("torch",),
)
def test_inverse_frequency_class_weights():
    mod = _training_module()
    # Mock training set: 70 class 0, 30 class 1 (N=100, C=2)
    weights = mod.compute_class_weights(counts={"0": 70, "1": 30})
    w0 = 100 / (2 * 70)
    w1 = 100 / (2 * 30)
    assert abs(weights[0] - w0) < 1e-5
    assert abs(weights[1] - w1) < 1e-5
    # Weighted sum: (70/100)*w0 + (30/100)*w1 == 1.0; unnormalized sum weighted by freq is 2.0 (C)
    expected_sum = (70 * w0 + 30 * w1) / 100 * 2
    assert abs(expected_sum - 2.0) < 1e-5


@case(
    "training.tie_break_fewer_parameters",
    "F2",
    "fast",
    "doc/03 §9.4",
    "Models tied within 0.005 validation macro-F1 break ties by choosing the candidate with fewer parameters.",
    requires=("torch",),
)
def test_tie_break_fewer_parameters():
    mod = _training_module()
    # Candidate B (2,050 params) vs Candidate C (4,098 params) within 0.005
    results = {
        "B": {"val_macro_f1": 0.812, "parameters": 2050},
        "C": {"val_macro_f1": 0.815, "parameters": 4098},
    }
    selected = mod.select_winning_candidate(results, tie_tolerance=0.005)
    assert selected == "B", "tie-break did not select the candidate with fewer parameters"
