"""F10 - the documentation and the repository.

The documents are the specification.  These tests check that the repository
still agrees with them.
"""

import hashlib
import io
import re
from pathlib import Path

import pytest

from tests import docmap
from tests.ids import case, known_gap

ROOT = Path(__file__).resolve().parent.parent

#: The SHA-256 of doc/00_PROJECT_OVERVIEW.md.  The overview is the single
#: document that must never change.
OVERVIEW_SHA256 = "952C443CB5037F26C6A29C6F57AC07C767E9F78C2A2304841DEC3A486C255641"


@case(
    "docs.overview_unchanged",
    "F10",
    "fast",
    "doc/00 §11.1",
    "The project overview is byte-identical to the version this documentation set was written against.",
)
def test_overview_unchanged():
    path = docmap.path_for("00")
    # The repository stores the document with LF endings; a Windows checkout
    # gives it CRLF.  Hashing the normalised content makes the check mean the
    # same thing on every platform.
    content = path.read_bytes().replace(b"\r\n", b"\n")
    digest = hashlib.sha256(content).hexdigest()
    assert digest.lower() == OVERVIEW_SHA256.lower(), (
        "the overview has changed: %s, expected %s" % (digest, OVERVIEW_SHA256)
    )


@known_gap(
    "docs.requirements_pinned",
    "F10",
    "fast",
    "doc/00 §11.1, doc/01 §4.2",
    "The runtime requirements pin their versions, so a run is reproducible.",
    "the requirements are unpinned",
)
def test_requirements_pinned():
    requirements = ROOT / "requirements.txt"
    assert requirements.is_file(), "requirements.txt does not exist"
    lines = [
        l.strip()
        for l in io.open(requirements, "r", encoding="utf-8").read().split("\n")
        if l.strip() and not l.strip().startswith("#")
    ]
    assert lines, "requirements.txt is empty"
    unpinned = [l for l in lines if "==" not in l and ">=" not in l]
    assert not unpinned, "these requirements are not pinned: %s" % ", ".join(unpinned)


@case(
    "docs.dataset_not_committed",
    "F10",
    "fast",
    "doc/02 §4.2",
    "The dataset, the derived manifests, and the generated samples are all excluded by .gitignore.",
)
def test_dataset_not_committed():
    ignore = io.open(ROOT / ".gitignore", "r", encoding="utf-8").read()
    for pattern in ("dataset/", "data/manifests/", "data/sample/"):
        assert re.search(r"(?m)^\s*%s" % re.escape(pattern), ignore), (
            "%s is not excluded by .gitignore" % pattern
        )


@case(
    "docs.canonical_owners_present",
    "F10",
    "fast",
    "doc/07 §11",
    "Every shared contract has a canonical owner section, and it exists.",
)
def test_canonical_owners_present():
    owners = {
        ("01", "7.2"): "the encoder's output contract",
        ("02", "6.2"): "the label mapping",
        ("02", "12.2"): "the manifest record schema",
        ("03", "16.2"): "the checkpoint file contents",
        ("04", "5"): "the input validation rules",
        ("04", "11.1"): "the status vocabulary",
        ("05", "8.2"): "the comparison mapping",
        ("06", "29"): "the citation and schema validation",
        ("06", "30"): "the outcome computation",
        ("07", "2"): "the test tier taxonomy",
    }
    missing = [
        "%s section %s (%s)" % (short, section, what)
        for (short, section), what in owners.items()
        if not docmap.resolves(short, section)
    ]
    assert not missing, "these canonical owners do not exist: %s" % "; ".join(missing)


@case(
    "docs.no_team_names_in_filenames",
    "F10",
    "fast",
    "doc/00 §11.1",
    "No documentation file is named after a person.",
)
def test_no_team_names_in_filenames():
    people = ["drashti", "dharmil", "dharmilg", "patel", "gajjar"]
    for path in docmap.DOC_DIR.glob("*.md"):
        lowered = path.name.lower()
        for person in people:
            assert person not in lowered, "%s is named after a person" % path.name


@case(
    "docs.every_document_present",
    "F10",
    "fast",
    "doc/00 §11",
    "Every document the set defines is present.",
)
def test_every_document_present():
    for short, name in docmap.FILES.items():
        path = docmap.DOC_DIR / name
        assert path.is_file(), "the document %s is missing" % name


@case(
    "docs.documentation_directory_is_exact",
    "F10",
    "fast",
    "doc/00 §11",
    "The documentation directory holds exactly the eight documents - the overview and the seven consolidated ones - and nothing else.",
)
def test_documentation_directory_is_exact():
    expected = set(docmap.FILES.values())
    found = {p.name for p in docmap.DOC_DIR.glob("*.md")}
    assert found == expected, (
        "the documentation directory does not match the expected set: unexpected %r, missing %r"
        % (sorted(found - expected), sorted(expected - found))
    )
    stray = [
        p.name
        for p in docmap.DOC_DIR.iterdir()
        if p.is_file() and p.suffix.lower() != ".md"
    ]
    assert not stray, "the documentation directory holds non-document files: %s" % ", ".join(stray)
