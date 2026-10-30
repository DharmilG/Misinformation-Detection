"""Typed domain error classes for the CLIP encoder and feature cache of Document 01 §13.
"""

from __future__ import annotations


class ModelUnavailableError(Exception):
    """Raised when the pre-trained weights cannot be located or loaded."""


class DeviceUnavailableError(Exception):
    """Raised when the requested compute device (e.g. cuda) is not available."""


class FeatureExtractionError(Exception):
    """Raised when feature extraction fails on an input pair or batch."""


class FeatureCacheError(Exception):
    """Raised when the feature cache is corrupt, missing, or mismatched."""
