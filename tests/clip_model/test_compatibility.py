"""The environment and model compatibility validation of Document 03 §22.

Tests hard requirements enforcement (torch_min, weights_only_supported),
supported ranges warnings, actionable error reporting, and lockfile presence.
"""

import pytest

from tests.ids import case


def _compatibility_module():
    import importlib
    return importlib.import_module("classifier_compatibility")


@case(
    "compatibility.module_exists",
    "F2",
    "fast",
    "doc/03 §22",
    "The project provides a compatibility validation module.",
)
def test_compatibility_module_exists():
    _compatibility_module()


@case(
    "compatibility.hard_requirements_enforced",
    "F2",
    "fast",
    "doc/03 §22.3, §22.4",
    "Failing a hard requirement (e.g. weights_only unsupported) raises CheckpointIncompatibleError.",
)
def test_hard_requirements_enforced():
    mod = _compatibility_module()
    bad_env = {"torch": "1.12.0", "weights_only_supported": False}
    with pytest.raises(Exception) as excinfo:
        mod.check_compatibility(bad_env, hard_requirements={"torch_min": "2.1", "weights_only_supported": True})
    assert "Incompatible" in type(excinfo.value).__name__


@case(
    "compatibility.supported_ranges_warn",
    "F2",
    "fast",
    "doc/03 §22.4",
    "Out-of-range versions in supported_ranges emit warnings unless strict_compatibility is set.",
)
def test_supported_ranges_warn():
    mod = _compatibility_module()
    env = {"python": "3.13.0"}
    ranges = {"python": ">=3.10,<3.13"}
    with pytest.warns(UserWarning):
        mod.check_supported_ranges(env, ranges, strict=False)

    with pytest.raises(Exception):
        mod.check_supported_ranges(env, ranges, strict=True)


@case(
    "compatibility.error_message_states_versions",
    "F2",
    "fast",
    "doc/03 §22.4",
    "Every compatibility failure states both the installed version and the expected version.",
)
def test_error_message_states_versions():
    mod = _compatibility_module()
    env = {"torch": "1.13.1"}
    try:
        mod.check_compatibility(env, hard_requirements={"torch_min": "2.1"})
    except Exception as exc:
        msg = str(exc)
        assert "1.13.1" in msg, "error message missing installed version"
        assert "2.1" in msg, "error message missing expected version"


@case(
    "compatibility.lockfile_present_in_release",
    "F2",
    "fast",
    "doc/03 §22.5",
    "requirements-lock.txt exists in the release and pins exact package versions.",
)
def test_lockfile_present_in_release(tmp_path):
    mod = _compatibility_module()
    release_dir = tmp_path / "mock_release"
    release_dir.mkdir()
    (release_dir / "requirements-lock.txt").write_text("torch==2.3.1\nnumpy==1.26.4\n", encoding="utf-8")
    assert mod.verify_lockfile(release_dir) is True
