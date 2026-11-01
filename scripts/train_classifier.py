"""Train and evaluate the classifier from genuine frozen-CLIP features."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
CLIP_DIR = ROOT / "src" / "clip_model"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CLIP_DIR) not in sys.path:
    sys.path.insert(0, str(CLIP_DIR))

from classifier import build_features, create_model, select_threshold
from classifier_dataset import FeaturePairDataset
from metrics import compute_accuracy, compute_confusion_matrix, compute_macro_f1, compute_roc_auc
from training import compute_class_weights, train_classifier


def _load_features(root: Path, split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    directory = root / split
    required = (
        directory / "image_embeddings.npy",
        directory / "text_embeddings.npy",
        directory / "labels.npy",
        directory / "sample_ids.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing extracted CLIP features for '{split}': {', '.join(missing)}. "
            "Run scripts/extract_clip_features.py first."
        )
    image = np.load(required[0], allow_pickle=False)
    text = np.load(required[1], allow_pickle=False)
    labels = np.load(required[2], allow_pickle=False)
    ids = json.loads(required[3].read_text(encoding="utf-8"))
    if not (len(image) == len(text) == len(labels) == len(ids)):
        raise ValueError(f"Feature alignment failure in {directory}")
    return image, text, labels, ids


def _prepare_dataset(root: Path, split: str, family: str) -> FeaturePairDataset:
    image, text, labels, ids = _load_features(root, split)
    features = build_features(
        family,
        torch.from_numpy(image),
        torch.from_numpy(text),
    ).numpy()
    return FeaturePairDataset(features, labels, ids)


def _predict(model: torch.nn.Module, dataset: FeaturePairDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():
        logits = model(torch.from_numpy(np.array(dataset.features, copy=True))).cpu()
    probabilities = torch.softmax(logits, dim=-1)[:, 1].numpy()
    return dataset.labels, probabilities, logits.numpy()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-root", type=Path, default=ROOT / "data" / "features")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "checkpoint")
    parser.add_argument("--candidate", choices=("A", "B", "C", "D-B", "D-C"), default="D-C")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs must be positive")

    train = _prepare_dataset(args.features_root, "train", args.candidate)
    val = _prepare_dataset(args.features_root, "val", args.candidate)
    test = _prepare_dataset(args.features_root, "test", args.candidate)
    model = create_model(args.candidate)
    config = {"epoch_cap": args.epochs}
    model, training_summary = train_classifier(
        model, train, val,
        class_weights=compute_class_weights(train.class_counts()),
        config=config,
    )
    val_labels, val_scores, _ = _predict(model, val)
    threshold = select_threshold(val_scores, val_labels)
    test_labels, test_scores, test_logits = _predict(model, test)
    test_predictions = (test_scores >= threshold).astype(np.int64)
    metrics = {
        "candidate": args.candidate,
        "threshold": threshold,
        "validation_macro_f1": compute_macro_f1(y_true=val_labels, y_pred=(val_scores >= threshold).astype(np.int64)),
        "test_accuracy": compute_accuracy(test_labels, test_predictions),
        "test_macro_f1": compute_macro_f1(y_true=test_labels, y_pred=test_predictions),
        "test_roc_auc": compute_roc_auc(test_labels, test_scores),
        "test_confusion_matrix": compute_confusion_matrix(test_labels, test_predictions).tolist(),
        "training": training_summary,
        "rows": {"train": len(train), "val": len(val), "test": len(test)},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output / "weights.pt")
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output / "config.json").write_text(json.dumps({
        "candidate": args.candidate,
        "feature_source": str(args.features_root),
        "threshold_selected_on": "validation",
        "test_evaluated_once": True,
    }, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
