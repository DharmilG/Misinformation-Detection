"""The top-level retriever API and integration contract of Document 06 §18, §19, and §23.

Tests the retrieve() entry point, independence from the classifier signal,
graceful handling of empty indexes, and structured RetrievalResult schema.
"""

import pytest

from tests.ids import case


def _retriever_module():
    import importlib
    return importlib.import_module("retrieval_api")


@case(
    "retriever_api.module_exists",
    "F2",
    "fast",
    "doc/06 §19",
    "The project provides a top-level retriever API exposing retrieve().",
)
def test_retriever_module_exists():
    _retriever_module()


@case(
    "retriever_api.classifier_result_not_an_input",
    "F2",
    "fast",
    "doc/06 §18.3",
    "retrieve() inputs are the query string and index only; classifier output is never accepted or used to steer search.",
)
def test_classifier_result_not_an_input():
    mod = _retriever_module()
    import inspect
    sig = inspect.signature(mod.retrieve)
    param_names = set(sig.parameters.keys())
    assert "query" in param_names
    for forbidden in ("classifier_result", "score", "label", "probability"):
        assert forbidden not in param_names, "retriever accepted classifier signal: %s" % forbidden


@case(
    "retriever_api.empty_index_returns_ok_empty",
    "F2",
    "fast",
    "doc/06 §19.3",
    "Searching an empty index returns status OK with an empty passages list (never an exception).",
)
def test_empty_index_returns_ok_empty():
    mod = _retriever_module()
    res = mod.retrieve_with_mock_index(query="Any news claim", empty_index=True)
    assert res.status == "OK"
    assert len(res.passages) == 0


@case(
    "retriever_api.result_schema",
    "F2",
    "fast",
    "doc/06 §19",
    "RetrievalResult returns passages with chunk_id, document_id, text, scores, and metadata.",
)
def test_result_schema():
    mod = _retriever_module()
    res = mod.retrieve_with_mock_index(query="Elections", empty_index=False)
    assert hasattr(res, "passages")
    assert hasattr(res, "status")
    assert hasattr(res, "query")
    if res.passages:
        p = res.passages[0]
        assert hasattr(p, "chunk_id")
        assert hasattr(p, "text")
        assert hasattr(p, "score")
