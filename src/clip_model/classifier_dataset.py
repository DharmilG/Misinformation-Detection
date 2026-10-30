"""Classifier dataset reader and alignment checks (Document 03 §5).

Provides :class:`FeaturePairDataset`, a PyTorch ``Dataset`` that serves
``(features, label, sample_id)`` triples from pre-extracted CLIP embeddings,
and :func:`verify_alignment` which checks manifest-to-cache consistency
before any training run begins.

Module name is ``classifier_dataset`` to match the test import.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Label constants
# ---------------------------------------------------------------------------
LABEL_MATCHED = "MATCHED"
LABEL_MISMATCHED = "MISMATCHED"
LABEL_UNKNOWN = "UNKNOWN"
VALID_TRAINING_LABELS = {LABEL_MATCHED, LABEL_MISMATCHED}
LABEL_TO_INDEX = {LABEL_MATCHED: 0, LABEL_MISMATCHED: 1}
INDEX_TO_LABEL = {0: LABEL_MATCHED, 1: LABEL_MISMATCHED}


class AlignmentError(Exception):
    """Raised when manifest-to-cache alignment checks fail (doc/03 §5.1)."""


class FeaturePairDataset(Dataset):
    """Serves ``(features, label, sample_id)`` triples from aligned arrays.

    Parameters
    ----------
    features : np.ndarray
        Shape ``(N, D)`` float32 feature vectors (already constructed for the
        selected candidate family, or raw 512-d CLIP embeddings).
    labels : Union[np.ndarray, Sequence]
        Integer labels (0/1) or string labels (``MATCHED``/``MISMATCHED``/
        ``UNKNOWN``).  When *exclude_unknown* is True, string labels are
        expected.
    sample_ids : Sequence[str]
        Stable sample identifiers, aligned to the feature rows.
    exclude_unknown : bool
        When True, rows whose label is ``UNKNOWN`` are excluded from the
        dataset.  Their count is recorded in :attr:`unknown_count`.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray | Sequence,
        sample_ids: Sequence[str],
        *,
        exclude_unknown: bool = False,
    ) -> None:
        labels_list: list = list(labels) if not isinstance(labels, np.ndarray) else labels.tolist()
        ids_list: list = list(sample_ids)

        if len(features) != len(labels_list) or len(features) != len(ids_list):
            raise AlignmentError(
                "features, labels and sample_ids must have the same length: "
                f"features={len(features)}, labels={len(labels_list)}, ids={len(ids_list)}"
            )

        self.unknown_count: int = 0

        if exclude_unknown:
            keep: list = []
            for i, lab in enumerate(labels_list):
                if str(lab) == LABEL_UNKNOWN:
                    self.unknown_count += 1
                else:
                    keep.append(i)
            features = features[keep]
            labels_list = [labels_list[i] for i in keep]
            ids_list = [ids_list[i] for i in keep]

        # Convert string labels to integer indices if needed.
        if labels_list and isinstance(labels_list[0], str):
            int_labels = []
            for lab in labels_list:
                if lab not in LABEL_TO_INDEX:
                    raise ValueError(
                        f"Invalid label '{lab}'. Expected one of {sorted(VALID_TRAINING_LABELS)}"
                    )
                int_labels.append(LABEL_TO_INDEX[lab])
            self._labels = np.array(int_labels, dtype=np.int64)
        else:
            self._labels = np.asarray(labels_list, dtype=np.int64)

        # Store features as read-only numpy (doc/03 §5.2).
        self._features = np.array(features, dtype=np.float32)
        self._features.flags.writeable = False

        self._sample_ids: list[str] = ids_list

    # -- Public properties ---------------------------------------------------

    @property
    def features(self) -> np.ndarray:
        """The read-only feature array."""
        return self._features

    @property
    def labels(self) -> np.ndarray:
        """Integer label array."""
        return self._labels

    @property
    def sample_ids(self) -> list[str]:
        """Stable sample identifiers, aligned to the feature rows."""
        return list(self._sample_ids)

    # -- Dataset interface ---------------------------------------------------

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        feat = torch.from_numpy(self._features[idx].copy())
        label = int(self._labels[idx])
        sid = self._sample_ids[idx]
        return feat, label, sid

    # -- Class counts --------------------------------------------------------

    def class_counts(self) -> dict[str, int]:
        """Per-class counts, keyed by label name."""
        counts: dict[str, int] = {LABEL_MATCHED: 0, LABEL_MISMATCHED: 0}
        for lab in self._labels:
            counts[INDEX_TO_LABEL[int(lab)]] += 1
        return counts


# ---------------------------------------------------------------------------
# Manifest / cache alignment verification  (doc/03 §5.1)
# ---------------------------------------------------------------------------

def verify_alignment(
    manifest_records: Sequence[dict[str, Any]],
    cache_ids: Sequence[str],
    *,
    cache_image_hashes: Sequence[str] | None = None,
    cache_caption_hashes: Sequence[str] | None = None,
) -> None:
    """Check that cached features and the manifest describe the same samples.

    Raises :class:`AlignmentError` on any mismatch.

    Parameters
    ----------
    manifest_records
        List of dicts, each with at least ``sample_id``.  Optionally
        ``image_sha256`` and ``caption_sha256``.
    cache_ids
        Ordered list of sample IDs from the feature cache.
    cache_image_hashes, cache_caption_hashes
        Per-row hashes from the cache, checked against the manifest when
        provided.
    """
    manifest_ids = [r["sample_id"] for r in manifest_records]

    # Row count.
    if len(manifest_ids) != len(cache_ids):
        raise AlignmentError(
            f"Row count mismatch: manifest has {len(manifest_ids)} records, "
            f"cache has {len(cache_ids)} entries"
        )

    # Sample identity: every manifest id appears exactly once in the cache.
    manifest_set = set(manifest_ids)
    cache_set = set(cache_ids)
    if manifest_set != cache_set:
        missing_in_cache = manifest_set - cache_set
        extra_in_cache = cache_set - manifest_set
        raise AlignmentError(
            f"Sample ID mismatch: {len(missing_in_cache)} manifest IDs missing "
            f"from cache, {len(extra_in_cache)} cache IDs missing from manifest"
        )

    # Duplicate detection.
    if len(manifest_ids) != len(manifest_set):
        raise AlignmentError("Duplicate sample_id in manifest")
    if len(cache_ids) != len(cache_set):
        raise AlignmentError("Duplicate sample_id in cache")

    # Per-row hash checks when available.
    if cache_image_hashes is not None:
        manifest_by_id = {r["sample_id"]: r for r in manifest_records}
        for sid, img_hash in zip(cache_ids, cache_image_hashes):
            expected = manifest_by_id[sid].get("image_sha256")
            if expected is not None and expected != img_hash:
                raise AlignmentError(
                    f"Image hash mismatch for sample {sid}: "
                    f"manifest={expected}, cache={img_hash}"
                )

    if cache_caption_hashes is not None:
        manifest_by_id = {r["sample_id"]: r for r in manifest_records}
        for sid, cap_hash in zip(cache_ids, cache_caption_hashes):
            expected = manifest_by_id[sid].get("caption_sha256")
            if expected is not None and expected != cap_hash:
                raise AlignmentError(
                    f"Caption hash mismatch for sample {sid}: "
                    f"manifest={expected}, cache={cap_hash}"
                )
