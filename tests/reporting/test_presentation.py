"""Fast cases for the generated project presentation."""

from pathlib import Path

from pptx import Presentation
from pptx.oxml.ns import qn

from tests.ids import case

ROOT = Path(__file__).resolve().parents[2]
DECK = ROOT / "Project_Presentation.pptx"


def _deck():
    assert DECK.is_file(), "run python scripts/build_presentation.py first"
    return Presentation(str(DECK))


def _all_text(prs):
    return "\n".join(
        shape.text_frame.text
        for slide in prs.slides
        for shape in slide.shapes
        if shape.has_text_frame
    )


@case(
    "deck.slides_present",
    "F10",
    "fast",
    "doc/05 §3, doc/07 §12.2",
    "The deck has the planned slides, including the architecture and its explanation.",
)
def test_slides_present():
    prs = _deck()
    assert len(prs.slides) == 24
    text = _all_text(prs)
    for expected in (
        "SEMESTER 07",
        "BATCH ICT",
        "Real pairs. Plausible fakes.",
        "A 50:50 test by design",
        "Sanitize, canonicalize, cache",
        "Why candidate D-C won",
        "54.2%",
        "66.1%",
        "What is CLIP?",
        "How CLIP sees the world",
        "Why only the embeddings?",
        "What is the D-C head?",
        "Dhaval Lakum",
        "Dharmil Gajjar",
        "Dhruv Chauhan",
        "Dhruv Mandaliya",
        "Drashti Patel",
        "Stuti Patel",
        "How a request flows",
        "What each box is for",
        "Same score. Different answers.",
    ):
        assert expected in text, f"missing from deck: {expected}"


@case(
    "deck.no_stale_metrics",
    "F10",
    "fast",
    "doc/07 §12.2",
    "The deck carries only the current metrics, never the discarded fixture figures.",
)
def test_no_stale_metrics():
    text = _all_text(_deck())
    for stale in ("12.00%", "12%", "50 claims", "−38%"):
        assert stale not in text, f"stale figure in deck: {stale}"
    for current in ("69.20%", "71.43%", "INCONCLUSIVE", "414 conformance"):
        assert current in text, f"current figure missing from deck: {current}"


@case(
    "deck.transitions_applied",
    "F10",
    "fast",
    "doc/07 §12.2",
    "Every slide carries a fade transition for the page-to-page animation.",
)
def test_transitions_applied():
    prs = _deck()
    slides = list(prs.slides)
    assert all(s._element.findall(qn("p:transition")) for s in slides)
