"""Temperature calibration fitting, evaluation, ECE, and Brier score (Document 03 §12).

Provides temperature scaling formula, validation-only temperature optimization,
Expected Calibration Error (ECE) with 10 bins, and Brier score calculations.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Calibrated probability calculation (doc/03 §12.1)
# ---------------------------------------------------------------------------

def compute_calibrated_probability(
    logits: torch.Tensor,
    temperature: float = 1.0,
    positive_class_idx: int = 1,
) -> torch.Tensor:
    """Compute calibrated probability: softmax(logits / T)[positive_class_idx] with T >= 1.0."""
    t = max(float(temperature), 1e-6)
    scaled_logits = logits / t
    probs = F.softmax(scaled_logits, dim=-1)
    return probs[..., positive_class_idx]


# ---------------------------------------------------------------------------
# Temperature fitting on validation data (doc/03 §12.2)
# ---------------------------------------------------------------------------

def fit_temperature(
    val_logits: torch.Tensor | np.ndarray,
    val_labels: torch.Tensor | np.ndarray | Sequence[int],
) -> float:
    """Fit optimal temperature T >= 1.0 by minimizing negative log-likelihood on validation data."""
    if not isinstance(val_logits, torch.Tensor):
        logits_t = torch.tensor(val_logits, dtype=torch.float32)
    else:
        logits_t = val_logits.detach().float()

    if not isinstance(val_labels, torch.Tensor):
        labels_t = torch.tensor(val_labels, dtype=torch.long)
    else:
        labels_t = val_labels.detach().long()

    # Numerical stability: max-subtract logits
    logits_t = logits_t - logits_t.max(dim=-1, keepdim=True).values

    def nll(temp: float) -> float:
        scaled = logits_t / temp
        loss = F.cross_entropy(scaled, labels_t, reduction="mean")
        return float(loss.item())

    # 1D search over T >= 1.0
    best_t = 1.0
    best_loss = nll(1.0)

    # Grid search followed by refinement
    candidates = np.linspace(1.0, 10.0, 181)  # step 0.05
    for t in candidates:
        loss = nll(float(t))
        if loss < best_loss:
            best_loss = loss
            best_t = float(t)

    # Fine refinement around best_t
    fine_candidates = np.linspace(max(1.0, best_t - 0.1), best_t + 0.1, 41)
    for t in fine_candidates:
        loss = nll(float(t))
        if loss < best_loss:
            best_loss = loss
            best_t = float(t)

    return max(1.0, float(best_t))


# ---------------------------------------------------------------------------
# Calibration metrics: ECE and Brier score (doc/03 §12.4)
# ---------------------------------------------------------------------------

def compute_ece(
    probs: Sequence[float],
    labels: Sequence[int],
    num_bins: int = 10,
) -> tuple[float, list[int]]:
    """Compute Expected Calibration Error (ECE) using equal-width bins on [0, 1].

    Parameters
    ----------
    probs : Sequence[float]
        Predicted positive-class probabilities.
    labels : Sequence[int]
        Binary true labels (0 or 1).
    num_bins : int
        Number of equal-width bins (default: 10).

    Returns
    -------
    ece : float
        The weighted expected calibration error in [0.0, 1.0].
    bin_counts : List[int]
        Sample count per bin.
    """
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    n = len(p)
    if n == 0:
        return 0.0, [0] * num_bins

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_counts: list[int] = []
    ece = 0.0

    for i in range(num_bins):
        low = bin_edges[i]
        high = bin_edges[i + 1]
        if i == num_bins - 1:
            mask = (p >= low) & (p <= high)
        else:
            mask = (p >= low) & (p < high)

        count = int(np.sum(mask))
        bin_counts.append(count)

        if count > 0:
            bin_acc = float(np.mean(y[mask]))
            bin_conf = float(np.mean(p[mask]))
            ece += (count / n) * abs(bin_acc - bin_conf)

    return float(ece), bin_counts


def compute_brier_score(
    probs: Sequence[float],
    labels: Sequence[int],
) -> float:
    """Compute Brier score as mean squared difference between predicted prob and label."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if len(p) == 0:
        return 0.0
    return float(np.mean((p - y) ** 2))
