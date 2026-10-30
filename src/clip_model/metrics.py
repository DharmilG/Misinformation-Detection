"""Evaluation metrics, orientation invariants, and safety rules (Document 03 §8).

Defines POSITIVE_CLASS = 'MISMATCHED' (index 1), confusion matrix orientation
(rows = true labels, columns = predicted labels), macro-F1, ROC-AUC, and
zero-division safe precision/recall/F1.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Positive class definition (doc/03 §8.1)
# ---------------------------------------------------------------------------

POSITIVE_CLASS: str = "MISMATCHED"
POSITIVE_CLASS_INDEX: int = 1
CLASS_NAMES: list[str] = ["MATCHED", "MISMATCHED"]


# ---------------------------------------------------------------------------
# Metric computation functions (doc/03 §8.2)
# ---------------------------------------------------------------------------

def compute_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> np.ndarray:
    """Compute 2x2 confusion matrix with rows=true, cols=predicted (doc/03 §8.2).

    Order of classes: [MATCHED (0), MISMATCHED (1)].
    Returns
    -------
    np.ndarray
        Shape (2, 2) where cm[i, j] is count of samples with true class i and pred class j.
    """
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)

    cm = np.zeros((2, 2), dtype=np.int64)
    for t in (0, 1):
        for p in (0, 1):
            cm[t, p] = np.sum((yt == t) & (yp == p))
    return cm


def compute_precision(
    tp: int | None = None,
    fp: int | None = None,
    *,
    y_true: Sequence[int] | None = None,
    y_pred: Sequence[int] | None = None,
    pos_label: int = 1,
) -> float:
    """Compute precision with zero-division safety (returns 0.0 when TP+FP == 0)."""
    if tp is None or fp is None:
        if y_true is None or y_pred is None:
            raise ValueError("Must provide either (tp, fp) or (y_true, y_pred)")
        yt = np.asarray(y_true, dtype=np.int64)
        yp = np.asarray(y_pred, dtype=np.int64)
        tp = int(np.sum((yt == pos_label) & (yp == pos_label)))
        fp = int(np.sum((yt != pos_label) & (yp == pos_label)))

    denom = tp + fp
    if denom == 0:
        return 0.0
    return float(tp / denom)


def compute_recall(
    tp: int | None = None,
    fn: int | None = None,
    *,
    y_true: Sequence[int] | None = None,
    y_pred: Sequence[int] | None = None,
    pos_label: int = 1,
) -> float:
    """Compute recall with zero-division safety (returns 0.0 when TP+FN == 0)."""
    if tp is None or fn is None:
        if y_true is None or y_pred is None:
            raise ValueError("Must provide either (tp, fn) or (y_true, y_pred)")
        yt = np.asarray(y_true, dtype=np.int64)
        yp = np.asarray(y_pred, dtype=np.int64)
        tp = int(np.sum((yt == pos_label) & (yp == pos_label)))
        fn = int(np.sum((yt == pos_label) & (yp != pos_label)))

    denom = tp + fn
    if denom == 0:
        return 0.0
    return float(tp / denom)


def compute_f1(
    precision: float | None = None,
    recall: float | None = None,
    *,
    tp: int | None = None,
    fp: int | None = None,
    fn: int | None = None,
) -> float:
    """Compute F1 score with zero-division safety (returns 0.0 when P+R == 0)."""
    if precision is None or recall is None:
        if tp is not None and fp is not None and fn is not None:
            precision = compute_precision(tp=tp, fp=fp)
            recall = compute_recall(tp=tp, fn=fn)
        else:
            raise ValueError("Must provide either (precision, recall) or (tp, fp, fn)")

    denom = precision + recall
    if denom == 0:
        return 0.0
    return float((2 * precision * recall) / denom)


def compute_macro_f1(
    f1_class_0: float | None = None,
    f1_class_1: float | None = None,
    *,
    y_true: Sequence[int] | None = None,
    y_pred: Sequence[int] | None = None,
) -> float:
    """Compute unweighted macro-F1 (arithmetic mean of class 0 and class 1 F1)."""
    if f1_class_0 is not None and f1_class_1 is not None:
        return float((f1_class_0 + f1_class_1) / 2.0)

    if y_true is not None and y_pred is not None:
        cm = compute_confusion_matrix(y_true, y_pred)
        # Class 0: MATCHED
        p0 = compute_precision(tp=cm[0, 0], fp=cm[1, 0])
        r0 = compute_recall(tp=cm[0, 0], fn=cm[0, 1])
        f1_0 = compute_f1(precision=p0, recall=r0)

        # Class 1: MISMATCHED
        p1 = compute_precision(tp=cm[1, 1], fp=cm[0, 1])
        r1 = compute_recall(tp=cm[1, 1], fn=cm[1, 0])
        f1_1 = compute_f1(precision=p1, recall=r1)

        return float((f1_0 + f1_1) / 2.0)

    raise ValueError("Must provide either (f1_class_0, f1_class_1) or (y_true, y_pred)")


def compute_roc_auc(
    y_true: Sequence[int],
    y_scores: Sequence[float],
) -> float:
    """Compute ROC-AUC using continuous positive-class probabilities (doc/03 §8.2)."""
    yt = np.asarray(y_true, dtype=np.int64)
    ys = np.asarray(y_scores, dtype=np.float64)

    pos_mask = (yt == 1)
    neg_mask = (yt == 0)
    n_pos = np.sum(pos_mask)
    n_neg = np.sum(neg_mask)

    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Compute ranks
    order = np.argsort(ys)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(ys) + 1)

    # Handle ties in scores
    _unique_scores, inverse, counts = np.unique(ys, return_inverse=True, return_counts=True)
    for idx, count in enumerate(counts):
        if count > 1:
            tie_mask = (inverse == idx)
            ranks[tie_mask] = np.mean(ranks[tie_mask])

    u_pos = np.sum(ranks[pos_mask]) - (n_pos * (n_pos + 1)) / 2.0
    return float(u_pos / (n_pos * n_neg))


def compute_accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Compute classification accuracy."""
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)
    if len(yt) == 0:
        return 0.0
    return float(np.mean(yt == yp))
