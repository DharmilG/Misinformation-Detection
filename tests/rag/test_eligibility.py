"""The five-condition evidence eligibility rule of Document 06 §28.

Tests citation validity requirement, rejection of NEUTRAL relations,
context_sufficient requirement, component ID binding, and retention of ineligible citations.
"""

import pytest

from tests.ids import case


def _eligibility_module():
    import importlib
    return importlib.import_module("rag_eligibility")


@case(
    "rag_eligibility.module_exists",
    "F2",
    "fast",
    "doc/06 §28",
    "The project provides an evidence eligibility evaluation module.",
)
def test_eligibility_module_exists():
    _eligibility_module()


@case(
    "rag_eligibility.five_conditions_rule",
    "F2",
    "fast",
    "doc/06 §28.1",
    "A citation is eligible only when quote verified, relation SUPPORTS/CONTRADICTS, context sufficient, and component recognized.",
)
def test_five_conditions_rule():
    mod = _eligibility_module()
    valid_citation = {
        "passage_id": "c1",
        "quote": "Verified quote.",
        "relation": "SUPPORTS",
        "context_sufficient": True,
        "claim_component_id": "K1",
    }
    known_components = {"K1"}
    valid_passages = {"c1": "Verified quote."}

    # All pass
    res_pass = mod.evaluate_citation_eligibility(valid_citation, known_components, valid_passages)
    assert res_pass["eligible"] is True
    assert res_pass["reason"] is None


@case(
    "rag_eligibility.neutral_relation_ineligible",
    "F2",
    "fast",
    "doc/06 §28.1",
    "Citations with relation == NEUTRAL are marked eligible=False with reason RELATION_NEUTRAL.",
)
def test_neutral_relation_ineligible():
    mod = _eligibility_module()
    cit_neutral = {
        "passage_id": "c1",
        "quote": "Mentions topic without confirming.",
        "relation": "NEUTRAL",
        "context_sufficient": True,
        "claim_component_id": "K1",
    }
    res = mod.evaluate_citation_eligibility(cit_neutral, {"K1"}, {"c1": "Mentions topic without confirming."})
    assert res["eligible"] is False
    assert res["reason"] == "RELATION_NEUTRAL"


@case(
    "rag_eligibility.insufficient_context_ineligible",
    "F2",
    "fast",
    "doc/06 §28.1",
    "Citations with context_sufficient == False are marked eligible=False with reason INSUFFICIENT_CONTEXT.",
)
def test_insufficient_context_ineligible():
    mod = _eligibility_module()
    cit_insufficient = {
        "passage_id": "c1",
        "quote": "Quote without enough background.",
        "relation": "SUPPORTS",
        "context_sufficient": False,
        "claim_component_id": "K1",
    }
    res = mod.evaluate_citation_eligibility(cit_insufficient, {"K1"}, {"c1": "Quote without enough background."})
    assert res["eligible"] is False
    assert res["reason"] == "INSUFFICIENT_CONTEXT"


@case(
    "rag_eligibility.ineligible_citations_surfaced",
    "F2",
    "fast",
    "doc/06 §28.2",
    "Ineligible citations are not deleted; they are preserved in the result with eligible=False and reason.",
)
def test_ineligible_citations_surfaced():
    mod = _eligibility_module()
    citations = [
        {"passage_id": "c1", "quote": "A", "relation": "NEUTRAL", "context_sufficient": True, "claim_component_id": "K1"},
    ]
    processed = mod.process_citations(citations, {"K1"}, {"c1": "A"})
    assert len(processed) == 1
    assert processed[0]["eligible"] is False
    assert processed[0]["ineligibility_reason"] == "RELATION_NEUTRAL"
