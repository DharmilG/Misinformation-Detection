"""The per-component and claim-level outcome computation of Document 06 §30.

Tests per-component status derivation (SUPPORTED, CONTRADICTED, CONTESTED, UNRESOLVED),
claim outcome rules, reason codes, and the per-component definition of CONFLICTING.
"""

import pytest

from tests.ids import case, known_gap


def _outcome_module():
    import importlib
    return importlib.import_module("rag_outcome")


@known_gap(
    "rag_outcome.module_exists",
    "F2",
    "fast",
    "doc/06 §30",
    "The project provides a claim-level and component-level outcome computation module.",
    "there is no rag_outcome module",
)
def test_outcome_module_exists():
    _outcome_module()


@known_gap(
    "rag_outcome.per_component_status_logic",
    "F2",
    "fast",
    "doc/06 §30.1",
    "Component status is CONTESTED when both sup and con exist, SUPPORTED for sup-only, and CONTRADICTED for con-only.",
    "per-component status computation is not implemented",
)
def test_per_component_status_logic():
    mod = _outcome_module()
    # Component with both eligible supporting and contradicting citations
    stat_contested = mod.compute_component_status(has_eligible_support=True, has_eligible_contradict=True)
    assert stat_contested == "CONTESTED"

    # Support only
    stat_sup = mod.compute_component_status(has_eligible_support=True, has_eligible_contradict=False)
    assert stat_sup == "SUPPORTED"

    # Neither
    stat_unres = mod.compute_component_status(has_eligible_support=False, has_eligible_contradict=False)
    assert stat_unres == "UNRESOLVED"


@known_gap(
    "rag_outcome.all_components_supported_for_supported",
    "F2",
    "fast",
    "doc/06 §30.2",
    "A claim outcome is SUPPORTED only when EVERY component has status SUPPORTED.",
    "claim support threshold logic is not implemented",
)
def test_all_components_supported_for_supported():
    mod = _outcome_module()
    # 2 components both SUPPORTED -> claim SUPPORTED
    comp_states = {"K1": "SUPPORTED", "K2": "SUPPORTED"}
    outcome_all = mod.compute_claim_outcome(comp_states, has_citations=True)
    assert outcome_all["outcome"] == "SUPPORTED"

    # 1 SUPPORTED, 1 UNRESOLVED -> INCONCLUSIVE with INSUFFICIENT_EVIDENCE
    comp_partial = {"K1": "SUPPORTED", "K2": "UNRESOLVED"}
    outcome_partial = mod.compute_claim_outcome(comp_partial, has_citations=True)
    assert outcome_partial["outcome"] == "INCONCLUSIVE"
    assert outcome_partial["reason"] == "INSUFFICIENT_EVIDENCE"


@known_gap(
    "rag_outcome.conflicting_is_per_component",
    "F2",
    "fast",
    "doc/06 §30.4",
    "CONFLICTING requires eligible evidence on both sides of the SAME component, not different components.",
    "per-component conflict rule is not implemented",
)
def test_conflicting_is_per_component():
    mod = _outcome_module()
    # K1 CONTRADICTED, K2 SUPPORTED -> outcome is CONTRADICTED, NOT CONFLICTING
    comp_diff = {"K1": "CONTRADICTED", "K2": "SUPPORTED"}
    res_diff = mod.compute_claim_outcome(comp_diff, has_citations=True)
    assert res_diff["outcome"] == "CONTRADICTED"

    # K1 CONTESTED -> outcome is CONFLICTING
    comp_same = {"K1": "CONTESTED", "K2": "SUPPORTED"}
    res_same = mod.compute_claim_outcome(comp_same, has_citations=True)
    assert res_same["outcome"] == "CONFLICTING"


@known_gap(
    "rag_outcome.no_evidence_gives_inconclusive",
    "F2",
    "fast",
    "doc/06 §30.2, §30.3",
    "No citations returns outcome INCONCLUSIVE with reason code NO_EVIDENCE.",
    "empty citation outcome mapping is not implemented",
)
def test_no_evidence_gives_inconclusive():
    mod = _outcome_module()
    res = mod.compute_claim_outcome(component_statuses={}, has_citations=False)
    assert res["outcome"] == "INCONCLUSIVE"
    assert res["reason"] == "NO_EVIDENCE"
