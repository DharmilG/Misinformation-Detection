"""F10 - the harness and the documentation.

These are the tests that keep the suite and the documents from drifting apart.
They are fast, and they are the reason the rest of the suite can be trusted.
"""

import io
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests import docmap
from tests.ids import case

ROOT = Path(__file__).resolve().parent.parent


@case(
    "harness.every_test_registered",
    "F10",
    "fast",
    "doc/07 §2.3",
    "Every collected test carries a registered case, with a tier and a clause.",
)
def test_every_test_registered(request):
    items = [i for i in request.session.items if i.get_closest_marker("case") is None]
    assert not items, "%d collected tests carry no case: %s" % (
        len(items),
        ", ".join(i.nodeid for i in items[:5]),
    )


@case(
    "harness.case_ids_unique",
    "F10",
    "fast",
    "doc/07 §12",
    "Every case id is unique, so a report never mixes two requirements under one name.",
)
def test_case_ids_unique(request):
    seen = {}
    for item in request.session.items:
        mark = item.get_closest_marker("case")
        if mark is None:
            continue
        cid = mark.kwargs.get("cid")
        assert cid not in seen, "the case id %s is used twice" % cid
        seen[cid] = item.nodeid
    assert seen, "no cases were collected"


@case(
    "harness.tier_taxonomy_matches_doc",
    "F10",
    "fast",
    "doc/07 §2.1",
    "The markers the suite uses are the markers Document 07 §2.1 declares.",
)
def test_tier_taxonomy_matches_doc():
    from tests.ids import TIERS

    path = docmap.path_for("07")
    text = io.open(path, "r", encoding="utf-8", newline="").read()
    section = text[text.find("### 2.1 The tiers") : text.find("### 2.2 The commands")]
    for marker in TIERS:
        if marker == "fast":
            # The fast tier is the absence of every other marker.
            assert "*(none)*" in section, "the tier table does not describe the fast tier"
            continue
        assert "`%s`" % marker in section, (
            "the marker %s is not named in the tier table" % marker
        )
    for marker in ("data", "slow", "eval", "benchmark", "e2e"):
        assert "| `%s` |" % marker in section, (
            "the marker %s has no row in the tier table" % marker
        )


@case(
    "harness.command_set_matches_doc",
    "F10",
    "fast",
    "doc/07 §2.2",
    "Every pytest marker expression the documentation shows is one the runner accepts.",
)
def test_command_set_matches_doc():
    from tests.run import TIER_MARKERS

    path = docmap.path_for("07")
    text = io.open(path, "r", encoding="utf-8", newline="").read()
    section = text[text.find("### 2.2 The commands") : text.find("### 2.3 Marker discipline")]
    expressions = re.findall(r'pytest -m "([^"]+)"', section)
    assert expressions, "the commands section shows no pytest invocations"
    for expression in expressions:
        for part in expression.split(" and "):
            part = part.strip()
            if part.startswith("not "):
                part = part[4:].strip()
            assert part in TIER_MARKERS, (
                "the documentation shows an unsupported expression: %r" % part
            )


@case(
    "harness.traceability_sections_resolve",
    "F10",
    "fast",
    "doc/07 §12",
    "Every section cited in the traceability table exists in the document it cites.",
)
def test_traceability_sections_resolve():
    rows = docmap.traceability_rows()
    assert rows, "the traceability table has no rows"
    broken = []
    for contract, cells in rows:
        for short, section in docmap.cited_sections(cells):
            if not docmap.resolves(short, section):
                broken.append("%s cites %s section %s, which does not exist" % (contract, short, section))
    assert not broken, "\n".join(broken)


@case(
    "harness.collection_clean",
    "F10",
    "fast",
    "doc/07 §2.3",
    "A full collection produces no errors, and the script-like files beside the implementation are excluded deliberately.",
)
def test_collection_clean():
    import conftest

    assert sorted(conftest.SCRIPT_LIKE_MODULES) == [
        "src/clip_model/test_clip.py",
        "src/clip_model/test_saved_features.py",
    ], "the exclusion list changed; a new stray test file would now be collected"

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "--color=no"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    output = result.stdout
    assert result.returncode == 0, "collection failed:\n%s" % output[-2000:]
    # The summary line reports collection errors; the body may legitimately
    # contain the word "error" in a test name.
    assert "errors" not in output.lower().split("\n")[-1], (
        "collection reported an error:\n%s" % output[-2000:]
    )


@case(
    "harness.tier_selection_is_composable",
    "F10",
    "fast",
    "doc/07 §2.1",
    "The markers are additive, so a tier selection can ask for several tiers at once.",
)
def test_tier_selection_is_composable():
    from tests.run import tier_expression

    expression = tier_expression("data and slow")
    assert "data" in expression and "slow" in expression, (
        "the composite tier selection lost a tier: %r" % expression
    )
    with pytest.raises(ValueError):
        tier_expression("data and nonsense")


@case(
    "harness.filter_matches_the_case_id",
    "F10",
    "fast",
    "doc/07 §2.2",
    "The -k filter selects by the case id, so a reader can find a case by the name in this document.",
)
def test_filter_matches_the_case_id():
    result = subprocess.run(
        [sys.executable, "-m", "tests.run", "--list"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout[-2000:]
    assert "clip.extract.long_caption_truncates" in result.stdout, (
        "the case list does not contain the truncation case"
    )
