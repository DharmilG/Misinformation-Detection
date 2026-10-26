"""Resolution of document section references.

The documentation is addressed by its short number, for example ``04 §6`` for
``doc/04_INPUT_AND_INFERENCE_PIPELINE.md`` section 6.  This module maps those
numbers onto files and headings, so a test can check that a cited section
actually exists.
"""

import io
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC_DIR = ROOT / "doc"

#: Short number -> file name.
FILES = {
    "00": "00_PROJECT_OVERVIEW.md",
    "01": "01_CLIP_FEATURE_EXTRACTION.md",
    "02": "02_DATASET_PREPARATION.md",
    "03": "03_CLASSIFIER_TRAINING_AND_CHECKPOINTS.md",
    "04": "04_INPUT_AND_INFERENCE_PIPELINE.md",
    "05": "05_REPORTING_AND_UI.md",
    "06": "06_RAG_NON_RAG_EVALUATION.md",
    "07": "07_INTEGRATION_TESTING.md",
}

_HEADING = re.compile(r"^#{2,4}\s+(\d+(?:\.\d+)*)\.?\s")


def path_for(short):
    """The file for a short document number, or ``None``."""
    name = FILES.get(short)
    return DOC_DIR / name if name else None


def sections(short):
    """The set of section numbers a document defines."""
    path = path_for(short)
    if path is None or not path.is_file():
        return set()
    found = set()
    for line in io.open(path, "r", encoding="utf-8", newline="").read().split("\n"):
        match = _HEADING.match(line)
        if match:
            found.add(match.group(1))
    return found


def resolves(short, section):
    """Whether a cited section exists in a document."""
    return section in sections(short)


def sha256(path):
    """The SHA-256 of a file, as a hex string."""
    import hashlib

    digest = hashlib.sha256()
    with io.open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def traceability_rows():
    """The rows of Document 07's traceability table.

    Each row is ``(contract, cells)``, where ``cells`` is the list of table
    cells in that row.
    """
    path = path_for("07")
    if path is None or not path.is_file():
        return []
    text = io.open(path, "r", encoding="utf-8", newline="").read()
    start = text.find("## 12. Contract-to-test traceability")
    if start < 0:
        return []
    end = text.find("\n## ", start + 10)
    section = text[start : end if end > 0 else len(text)]
    rows = []
    for line in section.split("\n"):
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if not cells or cells[0] in ("Contract", "---") or set(cells[0]) <= {"-"}:
            continue
        rows.append((cells[0], cells))
    return rows


#: A cell that cites a section, for example "04 §6" or "01 §4, §4.2".
_CITE = re.compile(r"\b(\d{2})\s*§\s*(\d+(?:\.\d+)*)")


def cited_sections(cells):
    """Every ``doc §section`` citation in a traceability row."""
    found = []
    for cell in cells:
        for match in _CITE.finditer(cell):
            found.append((match.group(1), match.group(2)))
    return found
