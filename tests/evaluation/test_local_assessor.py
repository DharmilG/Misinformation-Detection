"""Fast conformance cases for the swappable claim assessors."""

import pytest

from src.evaluation.local_assessor import (  # noqa: E402
    LocalModelAssessor,
    make_assessor,
)
from tests.ids import case  # noqa: E402


@case(
    "assessor.factory_selects_impl",
    "F10",
    "fast",
    "doc/07 §12.2",
    "The assessor factory selects exactly the requested implementation.",
)
def test_factory_selects_impl():
    assert make_assessor("heuristic").model_id == "heuristic"
    assert make_assessor("local-vlm").model_id == "qwen3:8b"
    with pytest.raises(ValueError, match="unknown assessor"):
        make_assessor("nonexistent")


@case(
    "assessor.local_failure_is_explicit",
    "F10",
    "fast",
    "doc/07 §12.2",
    "An unreachable local model returns UNAVAILABLE with a reason, never a fabricated outcome.",
)
def test_local_failure_is_explicit():
    assessor = LocalModelAssessor(model="unused", url="http://127.0.0.1:9/api/generate")
    result = assessor.assess("a claim", ())
    assert result.status == "UNAVAILABLE"
    assert result.outcome is None
    assert result.error_detail
    assert result.latency_ms >= 0


@case(
    "assessor.unknown_citations_dropped",
    "F10",
    "fast",
    "doc/07 §12.2",
    "A model that cites passage ids not supplied to it has those citations dropped.",
)
def test_unknown_citations_dropped(monkeypatch):
    import json as _json
    from src.evaluation import local_assessor as module

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return _json.dumps(
                {
                    "response": _json.dumps(
                        {
                            "outcome": "SUPPORTED",
                            "reason": "test",
                            "cited_passage_ids": ["p_real", "p_fabricated"],
                        }
                    )
                }
            ).encode("utf-8")

    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **k: _Response())
    assessor = LocalModelAssessor()
    result = assessor.assess("a claim", (type("P", (), {"chunk_id": "p_real", "text": "x"})(),))
    assert result.status == "OK"
    assert result.accepted_citations == ("p_real",)
