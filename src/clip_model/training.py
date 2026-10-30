"""Classifier training protocol, seeding, class weighting, and model selection (Document 03 §6, §7, §9).

Implements deterministic CPU training, pre-registered seeds [42, 43, 44, 45, 46],
inverse-frequency class weighting, and the two-stage model selection protocol.
"""

from __future__ import annotations

import copy
import random
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from classifier import CANDIDATE_FAMILIES
from metrics import compute_macro_f1
from torch import nn
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Pre-registered seeds and default config (doc/03 §7.1, §7.4)
# ---------------------------------------------------------------------------

SEEDS: list[int] = [42, 43, 44, 45, 46]

PRE_REGISTERED_CANDIDATE_ORDER: list[str] = ["A", "B", "C", "D-B", "D-C"]

DEFAULT_TRAINING_CONFIG: dict[str, Any] = {
    "device": "cpu",
    "deterministic": True,
    "batch_size": 512,
    "epoch_cap": 50,
    "early_stopping_patience": 5,
    "early_stopping_min_delta": 0.0005,
    "learning_rate": 1e-3,
    "weight_decay": 1e-2,
    "dropout": 0.1,
    "hidden_dim": 256,
    "seeds": list(SEEDS),
}


def get_training_config(**overrides) -> dict[str, Any]:
    """Return a complete training configuration dictionary."""
    config = dict(DEFAULT_TRAINING_CONFIG)
    config.update(overrides)
    return config


# ---------------------------------------------------------------------------
# Seeding and determinism (doc/03 §7.4, §7.5)
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> torch.Generator:
    """Set global seeds across random, numpy, and torch; return seeded torch Generator."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True)
        except (RuntimeError, ValueError):
            pass
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


# ---------------------------------------------------------------------------
# Class weighting (doc/03 §6.2)
# ---------------------------------------------------------------------------

def compute_class_weights(counts: dict[str | int, int]) -> list[float]:
    """Compute inverse-frequency class weights w_c = N / (C * n_c).

    Parameters
    ----------
    counts : Dict[Union[str, int], int]
        Mapping from class name/index to count in the training split.
        Supports keys like '0', '1' or 0, 1 or 'MATCHED', 'MISMATCHED'.

    Returns
    -------
    List[float]
        [w_0, w_1] corresponding to classes 0 and 1.
    """
    # Normalize keys to integer indices 0 and 1
    normalized: dict[int, int] = {}
    for k, v in counts.items():
        if str(k) in ("0", "MATCHED"):
            normalized[0] = int(v)
        elif str(k) in ("1", "MISMATCHED"):
            normalized[1] = int(v)
        else:
            try:
                normalized[int(k)] = int(v)
            except ValueError:
                raise ValueError(f"Unknown class key: {k}")

    if 0 not in normalized or 1 not in normalized:
        raise ValueError("Counts must provide counts for both classes 0 and 1")

    n0 = normalized[0]
    n1 = normalized[1]
    N = n0 + n1
    C = 2

    w0 = N / (C * n0) if n0 > 0 else 0.0
    w1 = N / (C * n1) if n1 > 0 else 0.0

    return [float(w0), float(w1)]


# ---------------------------------------------------------------------------
# Model selection (doc/03 §9.4)
# ---------------------------------------------------------------------------

def select_winning_candidate(
    results: dict[str, dict[str, Any]],
    tie_tolerance: float = 0.005,
) -> str:
    """Select the winning candidate using the pre-registered tie-break rules (doc/03 §9.4).

    1. Candidate with highest validation macro-F1.
    2. Candidates within `tie_tolerance` of the highest are considered tied.
    3. Tie-break 1: Fewer parameters.
    4. Tie-break 2: Pre-registered candidate order A -> B -> C -> D-B -> D-C.
    """
    if not results:
        raise ValueError("Results dict cannot be empty")

    best_f1 = max(res["val_macro_f1"] for res in results.values())
    threshold_f1 = best_f1 - tie_tolerance

    # Find candidates within the tie band
    tied_candidates = [
        cand for cand, res in results.items()
        if res["val_macro_f1"] >= threshold_f1 - 1e-9
    ]

    def param_count(cand: str) -> int:
        res = results[cand]
        if "parameters" in res:
            return int(res["parameters"])
        return int(CANDIDATE_FAMILIES.get(cand, {}).get("parameters", 10**9))

    def order_index(cand: str) -> int:
        if cand in PRE_REGISTERED_CANDIDATE_ORDER:
            return PRE_REGISTERED_CANDIDATE_ORDER.index(cand)
        return len(PRE_REGISTERED_CANDIDATE_ORDER)

    # Sort primarily by fewest parameters, secondarily by pre-registered order
    sorted_tied = sorted(tied_candidates, key=lambda c: (param_count(c), order_index(c)))
    return sorted_tied[0]


# ---------------------------------------------------------------------------
# Training loop (doc/03 §7.1)
# ---------------------------------------------------------------------------

def train_classifier(
    model: nn.Module,
    train_dataset: Any,
    val_dataset: Any,
    class_weights: Sequence[float] | None = None,
    config: dict[str, Any] | None = None,
    seed: int = 42,
) -> tuple[nn.Module, dict[str, Any]]:
    """Train a classification head using the protocol defined in doc/03 §7."""
    cfg = get_training_config(**(config or {}))
    gen = set_seed(seed)

    device = torch.device(cfg["device"])
    model.to(device)

    weights_t = None
    if class_weights is not None:
        weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=weights_t)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        generator=gen,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    best_macro_f1 = -1.0
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    patience_counter = 0

    patience = cfg["early_stopping_patience"]
    min_delta = cfg["early_stopping_min_delta"]

    for epoch in range(1, cfg["epoch_cap"] + 1):
        model.train()
        for batch in train_loader:
            feats, labels, _ = batch
            feats = feats.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        # Validation evaluation
        model.eval()
        val_preds: list[int] = []
        val_targets: list[int] = []
        with torch.no_grad():
            for batch in val_loader:
                feats, labels, _ = batch
                feats = feats.to(device)
                logits = model(feats)
                preds = torch.argmax(logits, dim=-1).cpu().tolist()
                val_preds.extend(preds)
                val_targets.extend(labels.tolist())

        val_macro_f1 = compute_macro_f1(y_true=val_targets, y_pred=val_preds)

        if val_macro_f1 > best_macro_f1 + min_delta:
            best_macro_f1 = val_macro_f1
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Restore best checkpoint
    model.load_state_dict(best_weights)

    summary = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro_f1,
        "seed": seed,
    }
    return model, summary
