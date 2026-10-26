# tests/input_pipeline

The tests for `src/input_pipeline/` — the input validation rules, the image
and text preprocessing, the request schema, the status model, and the decision
rule. Document 04 owns this component.

**Naming.** A file here mirrors a module there:
`src/input_pipeline/input_schema.py` → `tests/input_pipeline/test_input_schema.py`.

**Status.** The skeleton only. Nothing has moved yet — the files are still
flat in `tests/`. Document 07 §2 states the layout.

| File | Clause |
|---|---|
| `test_input_pipeline.py` | doc/04 §6, §9 |
| `test_pipeline.py` | doc/04 §11, §14, §17, §24 |
| `test_result.py` | doc/04 §9, §11 |
| `test_registry.py` | doc/04 §12 |
| `test_warnings.py` | doc/04 §23 |
| `test_inference_e2e.py` | doc/04 §25 |
