"""Classifier architectures, feature builders, thresholding, and prediction (Document 03 §4, §10, §12).

Provides candidate family architectures (A, B, C, D-B, D-C), feature construction,
model creation, operating threshold selection, and prediction functions.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------------------
# Candidate families (doc/03 §4.1)
# ---------------------------------------------------------------------------

CANDIDATE_FAMILIES: dict[str, dict[str, Any]] = {
    "A": {
        "input_dim": 1,
        "parameters": 4,
        "head": "Linear",
        "description": "Logistic regression on cosine similarity",
    },
    "B": {
        "input_dim": 1024,
        "parameters": 2050,
        "head": "Linear",
        "description": "Linear head on concatenated [v_i ; v_t]",
    },
    "C": {
        "input_dim": 2048,
        "parameters": 4098,
        "head": "Linear",
        "description": "Linear head on explicit interaction features",
    },
    "D-B": {
        "input_dim": 1024,
        "parameters": 262914,
        "head": "MLP",
        "description": "MLP (hidden=256) on concatenated features",
    },
    "D-C": {
        "input_dim": 2048,
        "parameters": 525058,
        "head": "MLP",
        "description": "MLP (hidden=256) on explicit interaction features",
    },
}


# ---------------------------------------------------------------------------
# Feature construction (doc/03 §4.1, §4.2)
# ---------------------------------------------------------------------------

def build_features(
    family: str,
    image_embedding: torch.Tensor,
    text_embedding: torch.Tensor,
) -> torch.Tensor:
    """Construct classifier input features from normalized CLIP embeddings.

    Parameters
    ----------
    family : str
        Candidate family identifier ('A', 'B', 'C', 'D-B', or 'D-C').
    image_embedding : torch.Tensor
        Image embedding vector(s), shape (512,) or (batch, 512).
    text_embedding : torch.Tensor
        Text embedding vector(s), shape (512,) or (batch, 512).

    Returns
    -------
    torch.Tensor
        Constructed feature tensor of shape (batch, input_dim) or (1, input_dim).
    """
    v_i = image_embedding
    v_t = text_embedding

    if v_i.dim() == 1:
        v_i = v_i.unsqueeze(0)
    if v_t.dim() == 1:
        v_t = v_t.unsqueeze(0)

    if family == "A":
        # Cosine similarity between L2-normalized embeddings is the dot product.
        return (v_i * v_t).sum(dim=-1, keepdim=True)
    elif family in ("B", "D-B"):
        return torch.cat([v_i, v_t], dim=-1)
    elif family in ("C", "D-C"):
        # Explicit interaction: [v_i ; v_t ; v_i ⊙ v_t ; |v_i − v_t|]
        v_prod = v_i * v_t
        v_diff = torch.abs(v_i - v_t)
        return torch.cat([v_i, v_t, v_prod, v_diff], dim=-1)
    else:
        raise ValueError(f"Unknown candidate family '{family}'. Expected one of {list(CANDIDATE_FAMILIES.keys())}")


# ---------------------------------------------------------------------------
# Architecture specification (doc/03 §4.3, §4.4)
# ---------------------------------------------------------------------------

class MLPHead(nn.Module):
    """Family D architecture: Linear -> ReLU -> Dropout -> Linear(2)."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dim, 2)

    @property
    def first_layer_in_features(self) -> int:
        return self.input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.fc1(x))
        h = self.drop(h)
        return self.fc2(h)


class LinearHead(nn.Module):
    """Linear head for Families A, B, and C: Linear(input_dim, 2)."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.fc = nn.Linear(input_dim, 2)

    @property
    def first_layer_in_features(self) -> int:
        return self.input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def create_model(
    family: str,
    hidden_dim: int = 256,
    dropout: float = 0.1,
) -> nn.Module:
    """Create a classification head instance for the given candidate family.

    Parameters
    ----------
    family : str
        One of 'A', 'B', 'C', 'D-B', 'D-C'.
    hidden_dim : int
        Hidden dimension for MLP heads (default: 256).
    dropout : float
        Dropout probability for MLP heads (default: 0.1).

    Returns
    -------
    nn.Module
        Initialized PyTorch model producing 2-class logits.
    """
    if family not in CANDIDATE_FAMILIES:
        raise ValueError(f"Unknown family '{family}'. Expected one of {list(CANDIDATE_FAMILIES.keys())}")

    input_dim = CANDIDATE_FAMILIES[family]["input_dim"]
    if family in ("D-B", "D-C"):
        return MLPHead(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    else:
        return LinearHead(input_dim=input_dim)


# ---------------------------------------------------------------------------
# Threshold selection and prediction (doc/03 §10.1, §12.1)
# ---------------------------------------------------------------------------

def select_threshold(
    val_scores: Sequence[float],
    val_labels: Sequence[int],
    candidate_thresholds: Sequence[float] | None = None,
) -> float:
    """Select the operating threshold maximizing validation macro-F1 (doc/03 §10.1).

    Predeclared selection rule:
    1. Choose threshold maximizing validation macro-F1.
    2. If tied within 0.005, choose higher recall on MISMATCHED (positive class).
    3. If still tied, choose threshold closest to 0.5.
    """
    scores = np.asarray(val_scores, dtype=np.float64)
    labels = np.asarray(val_labels, dtype=np.int64)

    if candidate_thresholds is None:
        # Standard candidate thresholds between 0.01 and 0.99, including unique midpoints
        sorted_unique = np.unique(scores)
        midpoints = (sorted_unique[:-1] + sorted_unique[1:]) / 2.0
        standard_grid = np.linspace(0.01, 0.99, 99)
        candidates = np.unique(np.concatenate(([0.5], midpoints, standard_grid)))
    else:
        candidates = np.asarray(candidate_thresholds, dtype=np.float64)

    candidates = np.sort(candidates)

    best_threshold = 0.5
    best_macro_f1 = -1.0
    best_recall = -1.0
    best_dist_to_half = 1.0

    for t in candidates:
        preds = (scores >= t).astype(np.int64)

        # Class 0: MATCHED, Class 1: MISMATCHED
        tp_0 = np.sum((labels == 0) & (preds == 0))
        fp_0 = np.sum((labels == 1) & (preds == 0))
        fn_0 = np.sum((labels == 0) & (preds == 1))

        tp_1 = np.sum((labels == 1) & (preds == 1))
        fp_1 = np.sum((labels == 0) & (preds == 1))
        fn_1 = np.sum((labels == 1) & (preds == 0))

        p0 = tp_0 / (tp_0 + fp_0) if (tp_0 + fp_0) > 0 else 0.0
        r0 = tp_0 / (tp_0 + fn_0) if (tp_0 + fn_0) > 0 else 0.0
        f1_0 = (2 * p0 * r0) / (p0 + r0) if (p0 + r0) > 0 else 0.0

        p1 = tp_1 / (tp_1 + fp_1) if (tp_1 + fp_1) > 0 else 0.0
        r1 = tp_1 / (tp_1 + fn_1) if (tp_1 + fn_1) > 0 else 0.0
        f1_1 = (2 * p1 * r1) / (p1 + r1) if (p1 + r1) > 0 else 0.0

        macro_f1 = (f1_0 + f1_1) / 2.0
        recall_pos = r1
        dist_to_half = abs(t - 0.5)

        if macro_f1 > best_macro_f1 + 0.005:
            best_macro_f1 = macro_f1
            best_recall = recall_pos
            best_dist_to_half = dist_to_half
            best_threshold = float(t)
        elif abs(macro_f1 - best_macro_f1) <= 0.005:
            # Tied within 0.005: tie-break by recall on positive class (MISMATCHED)
            if recall_pos > best_recall + 1e-6:
                best_macro_f1 = max(best_macro_f1, macro_f1)
                best_recall = recall_pos
                best_dist_to_half = dist_to_half
                best_threshold = float(t)
            elif abs(recall_pos - best_recall) <= 1e-6:
                # Still tied: choose threshold closest to 0.5
                if dist_to_half < best_dist_to_half - 1e-6:
                    best_macro_f1 = max(best_macro_f1, macro_f1)
                    best_recall = recall_pos
                    best_dist_to_half = dist_to_half
                    best_threshold = float(t)

    return best_threshold


def predict_labels(
    logits_or_scores: torch.Tensor | Sequence[float],
    threshold: float = 0.5,
) -> list[int]:
    """Predict binary integer labels [0, 1] given raw logits or uncalibrated scores."""
    if isinstance(logits_or_scores, torch.Tensor):
        if logits_or_scores.dim() >= 2 and logits_or_scores.shape[-1] == 2:
            probs = F.softmax(logits_or_scores, dim=-1)
            scores = probs[:, 1].detach().cpu().numpy()
        else:
            scores = logits_or_scores.detach().cpu().numpy().reshape(-1)
    else:
        scores = np.asarray(logits_or_scores, dtype=np.float64)

    return [1 if s >= threshold else 0 for s in scores]


def predict_calibrated_labels(
    logits: torch.Tensor,
    temperature: float = 1.0,
    threshold: float = 0.5,
) -> list[int]:
    """Predict binary integer labels preserving raw uncalibrated decision rule (doc/03 §12.1).

    Temperature calibration changes displayed probability but preserves predicted labels identically.
    """
    return predict_labels(logits, threshold=threshold)
