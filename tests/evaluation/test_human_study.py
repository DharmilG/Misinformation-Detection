"""Fast conformance cases for the Study 3 experiment instrument."""

import json

import pytest

from scripts.report_human_study import report  # noqa: E402
from src.evaluation.study_server import (  # noqa: E402
    CONDITIONS,
    Experiment,
    StudyError,
)
from tests.ids import case  # noqa: E402


def _case(cid, gold="MISMATCHED"):
    return {"id": cid, "image": "data/sample/test_3.webp", "caption": "x", "gold_label": gold}


def _experiment(tmp_path):
    return Experiment(cases=[_case(f"c{i}") for i in range(4)], output=tmp_path / "a.jsonl")


@case(
    "study.counterbalanced_order",
    "F10",
    "fast",
    "doc/07 §12.2",
    "Each condition appears in each position across participants.",
)
def test_counterbalanced_order(tmp_path):
    exp = _experiment(tmp_path)
    orders = [exp.condition_order(i) for i in range(len(CONDITIONS))]
    for position in range(len(CONDITIONS)):
        assert len({order[position] for order in orders}) == len(CONDITIONS)


@case(
    "study.schedule_is_one_per_condition",
    "F10",
    "fast",
    "doc/07 §12.2",
    "A participant sees exactly one case per condition.",
)
def test_schedule_is_one_per_condition(tmp_path):
    exp = _experiment(tmp_path)
    schedule = exp.schedule(0)
    assert [row["condition"] for row in schedule] == list(CONDITIONS)
    assert len({row["id"] for row in schedule}) == 4


@case(
    "study.rejects_remote_bind",
    "F10",
    "fast",
    "doc/07 §12.2",
    "The experiment instrument refuses to bind to a non-loopback address.",
)
def test_rejects_remote_bind(tmp_path):
    with pytest.raises(StudyError, match="127.0.0.1"):
        Experiment(cases=[_case("c1")], output=tmp_path / "a.jsonl", host="0.0.0.0")


@case(
    "study.records_anonymous_answers",
    "F10",
    "fast",
    "doc/07 §12.2",
    "Answers are written with an anonymous participant id and no names.",
)
def test_records_anonymous_answers(tmp_path):
    exp = _experiment(tmp_path)
    exp.record(
        participant="p1", case_id="c0", condition="C0", decision="MATCHED",
        confidence=5, elapsed_ms=1200, evidence_viewed=False,
    )
    path = exp.flush()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["participant"] == "p1"
    assert "name" not in rows[0]


@case(
    "study.metrics_by_condition",
    "F10",
    "fast",
    "doc/07 §12.2",
    "The reporter computes accuracy and trust gap separately per condition.",
)
def test_metrics_by_condition():
    cases = [_case("c1")]
    answers = [
        {"participant": "p1", "case_id": "c1", "condition": "C1",
         "decision": "MATCHED", "confidence": 7, "elapsed_ms": 1000}
    ]
    result = report(answers, cases)
    assert result["by_condition"]["C1"]["accuracy"] == 0.0
    assert result["by_condition"]["C1"]["mean_confidence"] == 7.0
    assert result["by_condition"]["C0"]["n"] == 0


@case(
    "study.placeholder_on_missing",
    "F10",
    "fast",
    "doc/07 §12.2",
    "Unmeasured human-factors fields stay as placeholders.",
)
def test_placeholder_on_missing():
    result = report([], [_case("c1")])
    assert result["automation_bias_rate"] == "[PLANNED - NOT YET MEASURED]"
    assert result["qualitative_themes"] == "[PLANNED - NOT YET MEASURED]"
