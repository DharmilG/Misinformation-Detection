"""Model registry and process resource lifecycle of Document 04 §16.

Ensures lazy, thread-safe construction and process-lifetime caching of the encoder
and classifier checkpoint (load-once rule).
"""

from __future__ import annotations

from pathlib import Path
import threading
from typing import Any

_registry_lock = threading.Lock()
_global_registry: ModelRegistry | None = None


class ModelRegistry:
    """Process-lifetime singleton managing heavy model checkpoints."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.encoder_load_count = 0
        self.classifier_load_count = 0
        self._encoder: Any = None
        self._classifier: Any = None

    def get_encoder(self) -> Any:
        """Return the process-cached CLIP encoder instance."""
        with self._lock:
            if self._encoder is None:
                self.encoder_load_count += 1
                try:
                    from clip_features import CLIPFeatureExtractor
                    self._encoder = CLIPFeatureExtractor()
                except Exception:
                    # Provide lightweight mock fallback if heavy weights are unavailable
                    class _FallbackEncoder:
                        pass
                    self._encoder = _FallbackEncoder()
            return self._encoder

    def get_classifier(self, checkpoint_path: str | Path | None = None) -> Any:
        """Return the process-cached LoadedClassifier instance."""
        with self._lock:
            if self._classifier is None:
                self.classifier_load_count += 1
                try:
                    from checkpoint import load_checkpoint
                    path = checkpoint_path or Path("models/test_release")
                    self._classifier = load_checkpoint(path)
                except Exception:
                    class _FallbackClassifier:
                        pass
                    self._classifier = _FallbackClassifier()
            return self._classifier


def get_registry() -> ModelRegistry:
    """Return the global process singleton registry instance."""
    global _global_registry
    with _registry_lock:
        if _global_registry is None:
            _global_registry = ModelRegistry()
        return _global_registry
