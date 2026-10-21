"""Shared fixtures and the tier-skip logic."""

import importlib
import io
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CLIP_DIR = ROOT / "src" / "clip_model"
DATASET_ROOT = Path(os.environ.get("OOC_DATASET_ROOT", ROOT / "dataset" / "ooc_dataset"))
SAMPLE_DIR = ROOT / "data" / "sample"

if str(CLIP_DIR) not in sys.path:
    sys.path.insert(0, str(CLIP_DIR))

#: Collected results, for the conformance report.
RESULTS = []

#: The collected items, by node id, so a test report can be traced back to
#: its registered case.
ITEM_BY_NODEID = {}


def _has_module(name):
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def _dataset_present():
    if not DATASET_ROOT.is_dir():
        return False
    return all((DATASET_ROOT / f).is_file() for f in ("train.csv", "val.csv", "test.csv"))


def _skip_reason(tier):
    """Why a tier cannot run here, or ``None`` when it can."""
    if tier == "data" and not _dataset_present():
        return "the corpus is not present at %s" % DATASET_ROOT
    if tier == "slow" and not (_has_module("torch") and _has_module("transformers")):
        return "torch and transformers are not installed"
    if tier == "eval":
        return "the labelled evaluation sets are not present"
    if tier == "benchmark":
        return "the candidate models are not present"
    if tier == "e2e":
        return "the full local stack is not present"
    return None


# --------------------------------------------------------------------------
# collection-time wiring
# --------------------------------------------------------------------------
def pytest_collection_modifyitems(session, config, items):
    """Apply the tier markers, the tier skips, and the declared xfails."""
    ITEM_BY_NODEID.clear()

    # The runner's filter is on the case id, so that "-k truncate" finds the
    # case whose id contains "truncate" and not a function whose name
    # happens to.  A filtered-out case is deselected, not skipped.
    needle = os.environ.get("TESTS_CASE_FILTER")
    if needle:
        needle = needle.lower()
        kept = []
        for item in items:
            mark = item.get_closest_marker("case")
            cid = str(mark.kwargs.get("cid", "")) if mark else ""
            if needle in cid.lower():
                kept.append(item)
        items[:] = kept

    for item in items:
        ITEM_BY_NODEID[item.nodeid] = item
        mark = item.get_closest_marker("case")
        if mark is None:
            continue

        tiers = mark.kwargs.get("tier") or ("fast",)
        if isinstance(tiers, str):
            tiers = (tiers,)
        for tier in tiers:
            # The marker is what makes ``-m "not slow"`` select correctly.
            item.add_marker(getattr(pytest.mark, tier))
            reason = _skip_reason(tier)
            if reason:
                item.add_marker(pytest.mark.skip(reason=reason))

        reason = mark.kwargs.get("xfail_reason")
        if reason:
            item.add_marker(pytest.mark.xfail(strict=True, reason=reason))
        for module in mark.kwargs.get("requires") or ():
            if not _has_module(module):
                item.add_marker(
                    pytest.mark.skip(reason="%s is not installed" % module)
                )


def pytest_runtest_logreport(report):
    """Record the outcome of every test, for the conformance report."""
    item = ITEM_BY_NODEID.get(report.nodeid)
    if item is None:
        return
    case_mark = item.get_closest_marker("case")
    if case_mark is None:
        return

    if report.when == "setup":
        if report.skipped:
            outcome = "skip"
        elif report.failed:
            outcome = "error"
        else:
            return
    elif report.when == "call":
        if report.passed:
            outcome = "pass"
        elif report.skipped:
            # A declared gap that still fails is an expected failure, not a
            # skip.  A strict xfail that passes is reported by pytest as a
            # failure, so it lands on the "fail" branch below.
            outcome = "xfail" if hasattr(report, "wasxfail") else "skip"
        else:
            outcome = "fail"
    else:
        return

    tiers = case_mark.kwargs.get("tier") or ()
    if isinstance(tiers, str):
        tiers = (tiers,)
    RESULTS.append(
        {
            "id": case_mark.kwargs.get("cid"),
            "family": case_mark.kwargs.get("family"),
            "tier": "+".join(tiers),
            "clause": case_mark.kwargs.get("clause"),
            "requirement": case_mark.kwargs.get("requirement"),
            "declared_gap": case_mark.kwargs.get("xfail_reason"),
            "outcome": outcome,
            "nodeid": report.nodeid,
        }
    )


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="session")
def project_root():
    return ROOT


@pytest.fixture(scope="session")
def dataset_root():
    if not _dataset_present():
        pytest.skip("the corpus is not present at %s" % DATASET_ROOT)
    return DATASET_ROOT


@pytest.fixture(scope="session")
def sample_dir():
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    return SAMPLE_DIR


@pytest.fixture(scope="session")
def extractor():
    """The feature extractor, loaded once per session.

    Skips with a reason when the runtime or the weights are unavailable, so a
    missing dependency never turns into a failure.
    """
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    try:
        from clip_features import CLIPFeatureExtractor
    except Exception as exc:  # pragma: no cover - import guard
        pytest.skip("clip_features is not importable: %s" % exc)
    try:
        return CLIPFeatureExtractor()
    except Exception as exc:
        pytest.skip("the CLIP model could not be loaded: %s" % exc)


@pytest.fixture
def tmp_cache_dir(tmp_path):
    return tmp_path
