"""Fast cases for the research-results UI surface."""

import json
import pathlib

from src.reporting import copy as C
from tests.ids import case

ROOT = pathlib.Path(__file__).resolve().parents[2]


@case(
    "ui.research_page_available",
    "F10",
    "fast",
    "doc/05 §3, doc/07 §12.2",
    "The research page is a registered navigation entry with a copy constant.",
)
def test_research_page_available():
    assert getattr(C, "H_RESEARCH", None) == "Research Results"
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "C.H_RESEARCH" in app
    assert "elif page == C.H_RESEARCH:" in app


@case(
    "ui.research_summary_valid",
    "F10",
    "fast",
    "doc/07 §12.2",
    "The committed research summary carries the protocol verdict and key metrics.",
)
def test_research_summary_valid():
    path = ROOT / "research" / "results_summary.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["rag"]["verdict"] == "INCONCLUSIVE"
    assert data["rag"]["ci95"][0] < 0 < data["rag"]["ci95"][1]
    assert data["rag"]["paired_delta"] == 0.0
    assert data["conformance"]["registered_cases"] > 0
