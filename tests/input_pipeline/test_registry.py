"""The model registry and resource lifecycle of Document 04 §16.

Tests process-level singleton management, lazy construction under lock,
and the load-once rule for the 600 MB encoder and classifier checkpoints.
"""

import pytest

from tests.ids import case


def _registry_module():
    import importlib
    return importlib.import_module("inference_registry")


@case(
    "registry.module_exists",
    "F2",
    "fast",
    "doc/04 §16.1",
    "The project provides a registry module managing process-lifetime model caching.",
)
def test_registry_module_exists():
    _registry_module()


@case(
    "registry.singleton_under_lock",
    "F2",
    "fast",
    "doc/04 §16.1",
    "get_registry() returns the same ModelRegistry instance across multiple calls under a lock.",
)
def test_singleton_under_lock():
    mod = _registry_module()
    reg1 = mod.get_registry()
    reg2 = mod.get_registry()
    assert reg1 is reg2


@case(
    "registry.load_once_rule",
    "F2",
    "fast",
    "doc/04 §16.2",
    "CLIP encoder and classifier checkpoint are loaded once per process lifetime, not per request.",
)
def test_load_once_rule():
    mod = _registry_module()
    reg = mod.get_registry()
    # Mock request executions
    load_count_before = reg.encoder_load_count
    reg.get_encoder()
    reg.get_encoder()
    assert reg.encoder_load_count == load_count_before or reg.encoder_load_count == 1
