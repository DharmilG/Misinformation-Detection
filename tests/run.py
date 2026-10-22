"""The test-suite command line.

    python -m tests.run                       everything
    python -m tests.run --tier fast           no weights, no dataset
    python -m tests.run --tier data           the corpus, no weights
    python -m tests.run --tier slow           the weights
    python -m tests.run --tier "data and slow"  both
    python -m tests.run -k truncate           filter by test id
    python -m tests.run -m "not slow"         a raw pytest expression
    python -m tests.run --list                every id, tier and clause
    python -m tests.run --report reports/conformance.json

The runner is a thin wrapper over pytest.  It adds the tier and filter
selection, the corpus and encoder-cache paths, and the conformance report.
"""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

try:
    from _pytest.outcomes import Skipped
except ImportError:  # pragma: no cover - pytest always provides it
    class Skipped(BaseException):
        """Stand-in for pytest's skip exception."""


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: The modules that carry the registered cases are found by walking the
#: package, not listed here.  See :func:`_discover`.

TIER_MARKERS = {
    "fast": "not slow and not data and not eval and not benchmark and not e2e",
    "data": "data",
    "slow": "slow",
    "eval": "eval",
    "benchmark": "benchmark",
    "e2e": "e2e",
    "all": "",
}

#: The tiers that can be combined.  "fast" is the absence of the others, so
#: it cannot take part in a conjunction.
COMBINABLE = ("data", "slow", "eval", "benchmark", "e2e")


def tier_expression(tier):
    """The pytest marker expression for a tier selection.

    A tier may be a single name, or several joined with ``and``, because the
    markers are additive: ``"data and slow"`` selects the cases that need both.
    """
    if tier in TIER_MARKERS:
        return TIER_MARKERS[tier]
    parts = [p.strip() for p in tier.split(" and ")]
    unknown = [p for p in parts if p not in COMBINABLE]
    if unknown:
        raise ValueError(
            "unknown tier: %s (known: %s)" % (", ".join(unknown), ", ".join(COMBINABLE))
        )
    return " and ".join("(%s)" % TIER_MARKERS[p] for p in parts)


def _discover():
    """Every case module in the suite.

    Found by walking the package rather than listed by hand, so that moving a
    test into a component folder is a ``git mv`` and nothing else.  pytest
    collects from the same tree, so the two always agree on which files carry
    the cases.
    """
    root = Path(__file__).resolve().parent
    names = []
    for path in sorted(root.rglob("test_*.py")):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts:
            continue
        names.append("tests." + ".".join(relative.with_suffix("").parts))
    return names


def _cases():
    """Every registered case, in module order."""
    found = []
    for name in _discover():
        try:
            module = importlib.import_module(name)
        except (Exception, Skipped) as exc:
            # A module whose dependency is absent is skipped, not failed.
            # pytest's own Skipped is a BaseException, so it is named here.
            print("could not import %s: %s" % (name, exc), file=sys.stderr)
            continue
        for attr in sorted(dir(module)):
            func = getattr(module, attr)
            for mark in getattr(func, "pytestmark", []):
                if mark.name == "case":
                    found.append((name, attr, mark.kwargs))
    return found


def _list_cases():
    rows = _cases()
    width = max([len(r[2].get("cid", "")) for r in rows] or [10])
    print("%-*s  %-10s  %-8s  %s" % (width, "ID", "TIER", "FAMILY", "CLAUSE"))
    print("-" * (width + 42))
    for _, _, kw in rows:
        tiers = kw.get("tier") or ()
        if isinstance(tiers, str):
            tiers = (tiers,)
        print(
            "%-*s  %-10s  %-8s  %s"
            % (
                width,
                kw.get("cid", "?"),
                "+".join(tiers),
                kw.get("family", "?"),
                kw.get("clause", "?"),
            )
        )
    print("\n%d cases" % len(rows))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tests.run", description=__doc__)
    parser.add_argument("--tier", default="all", help="one of: %s" % ", ".join(TIER_MARKERS))
    parser.add_argument("-k", dest="keyword", help="only run cases whose id contains this")
    parser.add_argument("-m", dest="expression", help="a raw pytest marker expression")
    parser.add_argument("--list", action="store_true", help="list the registered cases")
    parser.add_argument("--dataset-root", help="override the corpus location")
    parser.add_argument("--encoder-cache", help="override the Hugging Face cache location")
    parser.add_argument("--report", help="write the conformance report here")
    parser.add_argument("--rootdir", help="override the project root")
    args = parser.parse_args(argv)

    if args.list:
        return _list_cases()

    if args.dataset_root:
        os.environ["OOC_DATASET_ROOT"] = args.dataset_root
    if args.encoder_cache:
        os.environ["HF_HOME"] = args.encoder_cache
    if args.rootdir:
        os.environ["TESTS_PROJECT_ROOT"] = args.rootdir

    pytest_args = [str(ROOT)]
    if args.tier != "all":
        try:
            marker = tier_expression(args.tier)
        except ValueError as exc:
            parser.error(str(exc))
        if args.expression:
            pytest_args += ["-m", "(%s) and (%s)" % (marker, args.expression)]
        else:
            pytest_args += ["-m", marker]
    else:
        # An empty expression overrides the fast-only default in pytest.ini,
        # so "everything" really does mean everything.
        pytest_args += ["-m", args.expression or ""]
    if args.keyword:
        # The filter is on the case id, not on the test function name, so
        # ``-k truncate`` finds the case whose id contains "truncate".
        os.environ["TESTS_CASE_FILTER"] = args.keyword
    pytest_args += ["-q", "--color=no"]

    import pytest

    exit_code = pytest.main(pytest_args)

    if args.report:
        _write_report(args.report)
    return int(exit_code)


def _write_report(path):
    """Write the conformance report."""
    from tests import conftest as suite

    report_path = Path(path)
    if not report_path.is_absolute():
        report_path = ROOT / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)

    totals = {}
    for row in suite.RESULTS:
        totals[row["outcome"]] = totals.get(row["outcome"], 0) + 1

    payload = {
        "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "suite": "clip-feature-conformance",
        "totals": {
            "pass": totals.get("pass", 0),
            "fail": totals.get("fail", 0),
            "error": totals.get("error", 0),
            "xfail": totals.get("xfail", 0),
            "skip": totals.get("skip", 0),
        },
        "cases": sorted(suite.RESULTS, key=lambda r: (r.get("family") or "", r.get("id") or "")),
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nconformance report: %s" % report_path)
    print(
        "  pass=%d fail=%d xfail=%d skip=%d"
        % (
            payload["totals"]["pass"],
            payload["totals"]["fail"],
            payload["totals"]["xfail"],
            payload["totals"]["skip"],
        )
    )


if __name__ == "__main__":
    sys.exit(main())
