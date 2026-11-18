"""Fast conformance cases for the RAG stage tracer and reporter."""

from src.evaluation import rag_diagnostics as diag
from scripts.report_rag_ablation import (  # noqa: E402
    PLACEHOLDER,
    outcome_distribution,
    primary_failure,
    retrieval_metrics,
)
from scripts.run_rag_ablation import load_manifest, ManifestError  # noqa: E402
from tests.ids import case  # noqa: E402


class _Passage:
    def __init__(self, chunk_id, text, score=0.8):
        self.chunk_id = chunk_id
        self.document_id = "doc"
        self.text = text
        self.score = score


class _Result:
    status = "OK"

    def __init__(self, passages):
        self.passages = passages


class _Assessment:
    status = "SKIPPED"
    outcome = None
    reason_text = None
    accepted_citations = ()
    rejected_citations = ()
    unresolved_items = ()
    conflicts = ()
    uncertainties = ()
    components = ()


@case(
    "rag_diag.stage_records_complete",
    "F10",
    "fast",
    "doc/07 §12.2",
    "A retrieval stage records its inputs, outputs and latency even when the call fails.",
)
def test_stage_records_complete(monkeypatch):
    monkeypatch.setattr(diag, "retrieve", lambda *a, **k: _Result([_Passage("p1", "text")]))
    _, record = diag.trace_retrieval("c1", "C2", "claim")
    assert record.stage == "retrieval"
    assert record.inputs["query"] == "claim"
    assert record.outputs["passage_count"] == 1
    assert record.error is None
    assert record.latency_ms >= 0


@case(
    "rag_diag.non_ok_status_kept",
    "F10",
    "fast",
    "doc/07 §12.2",
    "A non-OK assessment status is retained rather than silently dropped.",
)
def test_non_ok_status_kept(monkeypatch):
    monkeypatch.setattr(diag, "run_assessment", lambda *a, **k: _Assessment())
    _, record = diag.trace_assessment("c1", "C3", "claim", ())
    assert record.outputs["status"] == "SKIPPED"
    assert record.outputs["outcome"] is None


@case(
    "rag_diag.synthetic_passages_detected",
    "F10",
    "fast",
    "doc/07 §12.2",
    "Stub retrieval that fabricates passages from the query is flagged as synthetic.",
)
def test_synthetic_passages_detected(monkeypatch):
    monkeypatch.setattr(
        diag,
        "retrieve",
        lambda *a, **k: _Result([_Passage("chunk_0", "Relevant passage 0 concerning claim")]),
    )
    _, record = diag.trace_retrieval("c1", "C2", "claim")
    assert record.outputs["synthetic_passages"] is True


@case(
    "rag_diag.same_claims_enforced",
    "F10",
    "fast",
    "doc/07 §12.2",
    "A paired manifest with duplicate claims is rejected before any inference.",
)
def test_same_claims_enforced(tmp_path):
    path = tmp_path / "paired.jsonl"
    path.write_text(
        '{"id":"a","claim":"same","gold_label":"MATCHED","gold_outcome":"SUPPORTED"}\n'
        '{"id":"b","claim":"same","gold_label":"MISMATCHED","gold_outcome":"CONTRADICTED"}\n',
        encoding="utf-8",
    )
    try:
        load_manifest(path)
    except ManifestError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate claims were accepted")


@case(
    "rag_diag.recall_at_k",
    "F10",
    "fast",
    "doc/07 §12.2",
    "Retrieval recall@k is measured against declared gold evidence ids.",
)
def test_recall_at_k():
    rows = [
        {
            "retrieval_outputs": {"ids": ["p1", "p2"], "passage_count": 2},
            "retrieval_error": None,
            "gold_evidence_ids": ["p1", "p3"],
        }
    ]
    metrics = retrieval_metrics(rows)
    assert metrics["recall_at_k"] == 0.5
    assert metrics["precision_at_k"] == 0.5


@case(
    "rag_diag.outcome_distribution_zero_classes",
    "F10",
    "fast",
    "doc/07 §12.2",
    "The outcome distribution reports all four classes, including zeros.",
)
def test_outcome_distribution_zero_classes():
    counts = outcome_distribution([{"outcome_outputs": {"outcome": "INCONCLUSIVE"}}])
    assert counts["INCONCLUSIVE"] == 1
    assert counts["SUPPORTED"] == 0
    assert len(counts) == 4


@case(
    "rag_diag.primary_failure_precedence",
    "F10",
    "fast",
    "doc/07 §12.2",
    "Missing or synthetic retrieval is attributed to F1 before any downstream cause.",
)
def test_primary_failure_precedence():
    row = {"retrieval_outputs": {"synthetic_passages": True, "passage_count": 1}}
    assert primary_failure(row) == "F1"
    assert primary_failure({"retrieval_outputs": {}}) == "F1"
    assert primary_failure({"retrieval_error": "boom"}) == "F1"


@case(
    "rag_diag.generation_failure_attributed",
    "F10",
    "fast",
    "doc/07 §12.2",
    "A failed or unavailable assessment is attributed to generation, not to evidence relevance.",
)
def test_generation_failure_attributed():
    row = {
        "retrieval_outputs": {"synthetic_passages": False, "passage_count": 3},
        "assessment_outputs": {"status": "FAILED"},
    }
    assert primary_failure(row) == "F6"
    row["assessment_outputs"] = {"status": "UNAVAILABLE"}
    assert primary_failure(row) == "F6"


@case(
    "rag_diag.schema_mismatch_attributed",
    "F10",
    "fast",
    "doc/07 §12.2",
    "A passage-schema integration error is attributed to the interface, not to generation.",
)
def test_schema_mismatch_attributed():
    row = {
        "retrieval_outputs": {"synthetic_passages": False, "passage_count": 3},
        "assessment_outputs": {
            "status": "FAILED",
            "error_detail": "'RetrievedPassage' object has no attribute 'passage'",
        },
    }
    assert primary_failure(row) == "F10"


@case(
    "rag_diag.adapter_converts_flat_passages",
    "F10",
    "fast",
    "doc/07 §12.2",
    "Flat retrieval passages are adapted into the nested schema the assessor needs.",
)
def test_adapter_converts_flat_passages():
    from types import SimpleNamespace
    from src.evaluation.rag_diagnostics import _as_assessment_passages

    flat = [SimpleNamespace(chunk_id="p1", text="an excerpt", score=0.8)]
    result = _as_assessment_passages(tuple(flat))
    assert hasattr(result[0], "passage")
    assert result[0].passage.excerpt == "an excerpt"
    assert result[0].relevance_score == 0.8


@case(
    "rag_diag.placeholder_on_missing",
    "F10",
    "fast",
    "doc/07 §12.2",
    "An empty trace set leaves every diagnosis field as the exact placeholder.",
)
def test_placeholder_on_missing():
    metrics = retrieval_metrics([])
    assert metrics["recall_at_k"] == PLACEHOLDER
