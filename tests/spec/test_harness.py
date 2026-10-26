"""F10 - the evaluation harness invariants of Document 07 §3.

Each evaluation script asserts its own preconditions and refuses to run otherwise:
1. Split isolation: development script reads only dev split, test script reads only test split;
2. Configuration fingerprint matching;
3. Complete provenance (model version, corpus manifest hash, prompt version, retrieval config);
4. Output bound with context and provenance;
5. Fixture name enforcement.
"""

import pytest

from tests.ids import case, known_gap


def _harness_module():
    import importlib
    return importlib.import_module("src.evaluation.harness")


@known_gap(
    "harness.eval_split_isolation",
    "F10",
    "fast",
    "doc/07 §3",
    "The evaluation script refuses to run if it reads a split it is not named for.",
    "evaluation harness split isolation validator is not yet implemented",
)
def test_eval_split_isolation():
    mod = _harness_module()
    assert hasattr(mod, "assert_split_match")
    # Must raise if dev script is pointed at test split
    with pytest.raises(Exception):
        mod.assert_split_match(script_name="eval_retrieval_dev.py", split_name="test")


@known_gap(
    "harness.eval_fingerprint_precondition",
    "F10",
    "fast",
    "doc/07 §3",
    "The evaluation script aborts if the configuration fingerprint differs from the recorded one.",
    "evaluation harness fingerprint precondition checker is not yet implemented",
)
def test_eval_fingerprint_precondition():
    mod = _harness_module()
    assert hasattr(mod, "assert_fingerprint_match")
    with pytest.raises(Exception):
        mod.assert_fingerprint_match(
            active_fingerprint="abc" * 21 + "a",
            recorded_fingerprint="xyz" * 21 + "x",
        )


@known_gap(
    "harness.eval_provenance_present",
    "F10",
    "fast",
    "doc/07 §3",
    "Evaluation outputs require full provenance: model version, corpus manifest hash, prompt version, and retrieval config.",
    "evaluation provenance validator is not yet implemented",
)
def test_eval_provenance_present():
    mod = _harness_module()
    assert hasattr(mod, "validate_evaluation_provenance")
    provenance = {
        "model_version": "v1.0",
        "corpus_manifest_sha256": "f" * 64,
        "prompt_version": "1.0",
        "retrieval_config_hash": "e" * 64,
    }
    assert mod.validate_evaluation_provenance(provenance) is True
    incomplete = {"model_version": "v1.0"}
    with pytest.raises(Exception):
        mod.validate_evaluation_provenance(incomplete)


@known_gap(
    "harness.eval_output_context_bound",
    "F10",
    "fast",
    "doc/07 §3",
    "Output is written with configuration and provenance so numbers cannot be produced without context.",
    "evaluation output packaging is not yet implemented",
)
def test_eval_output_context_bound():
    mod = _harness_module()
    assert hasattr(mod, "build_evaluation_record")
    record = mod.build_evaluation_record(
        metrics={"recall_at_5": 0.85},
        configuration={"k": 5},
        provenance={"model_version": "v1.0"},
    )
    assert "metrics" in record
    assert "configuration" in record
    assert "provenance" in record


@known_gap(
    "harness.eval_fixture_named_match",
    "F10",
    "fast",
    "doc/07 §3",
    "Script aborts on attempt to evaluate on an un-named fixture or test set.",
    "evaluation fixture binding enforcement is not yet implemented",
)
def test_eval_fixture_named_match():
    mod = _harness_module()
    assert hasattr(mod, "assert_fixture_allowed")
    with pytest.raises(Exception):
        mod.assert_fixture_allowed(
            script_name="eval_classifier.py",
            fixture_path="data/eval/unnamed_fixture.jsonl",
        )
