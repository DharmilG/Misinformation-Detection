"""The evaluation metrics and orientation invariants of Document 03 §8.

Asserts the declared positive class is MISMATCHED, the confusion matrix
orientation (rows=true, cols=predicted), unweighted macro-F1, and continuous ROC-AUC.
"""

import pytest

from tests.ids import case, known_gap


def _metrics_module():
    import importlib
    return importlib.import_module("metrics")


@known_gap(
    "metrics.module_exists",
    "F2",
    "fast",
    "doc/03 §8",
    "The project provides a classifier evaluation metrics calculation module.",
    "there is no metrics module",
)
def test_metrics_module_exists():
    _metrics_module()


@known_gap(
    "metrics.positive_class_is_mismatched",
    "F2",
    "fast",
    "doc/03 §8.1",
    "The declared positive class for all metrics and ROC curves is MISMATCHED (index 1).",
    "the positive class is not defined",
)
def test_positive_class_is_mismatched():
    mod = _metrics_module()
    assert mod.POSITIVE_CLASS == "MISMATCHED"
    assert mod.POSITIVE_CLASS_INDEX == 1


@known_gap(
    "metrics.confusion_matrix_orientation",
    "F2",
    "fast",
    "doc/03 §8.2",
    "The confusion matrix has rows=true labels and columns=predicted labels in order [MATCHED, MISMATCHED].",
    "confusion matrix calculation is not implemented",
)
def test_confusion_matrix_orientation():
    mod = _metrics_module()
    # 2 true MATCHED (pred 1 MATCHED, 1 MISMATCHED) -> row 0: [1, 1]
    # 2 true MISMATCHED (pred 0 MATCHED, 2 MISMATCHED) -> row 1: [0, 2]
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    cm = mod.compute_confusion_matrix(y_true, y_pred)
    assert cm.tolist() == [[1, 1], [0, 2]], "confusion matrix is transposed or incorrectly indexed: %r" % cm


@known_gap(
    "metrics.macro_f1_unweighted",
    "F2",
    "fast",
    "doc/03 §8.2",
    "Macro-F1 is the unweighted arithmetic mean of the two class F1 scores.",
    "macro-F1 computation is not implemented",
)
def test_macro_f1_unweighted():
    mod = _metrics_module()
    # f1_0 = 0.8, f1_1 = 0.6 -> macro_f1 = 0.70
    macro_f1 = mod.compute_macro_f1(f1_class_0=0.8, f1_class_1=0.6)
    assert abs(macro_f1 - 0.70) < 1e-6


@known_gap(
    "metrics.roc_auc_continuous_scores",
    "F2",
    "fast",
    "doc/03 §8.2",
    "ROC-AUC is computed using continuous positive-class probabilities, never hard-thresholded labels.",
    "ROC-AUC computation is not implemented",
)
def test_roc_auc_continuous_scores():
    mod = _metrics_module()
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.4, 0.6, 0.9]
    auc = mod.compute_roc_auc(y_true, y_scores)
    assert abs(auc - 1.0) < 1e-5


@known_gap(
    "metrics.zero_division_safe",
    "F2",
    "fast",
    "doc/03 §8.2",
    "Zero-denominator cases in precision, recall, and F1 return 0.0 without raising exceptions.",
    "zero division handling is not implemented",
)
def test_zero_division_safe():
    mod = _metrics_module()
    # Zero positive predictions: TP=0, FP=0 -> Precision=0.0
    prec = mod.compute_precision(tp=0, fp=0)
    assert prec == 0.0
