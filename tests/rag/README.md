# tests/rag

The tests for `src/rag/` — the prompt construction, the assembly, the
generation, the citation and schema validation, the eligibility rule, the
per-component outcome, and the prompt-injection defences. Document 06 §48 owns
this component.

**Naming.** A file here mirrors a module there:
`src/rag/eligibility.py` → `tests/rag/test_eligibility.py`.

**Status.** Populated. All 8 contract test suites for Document 06 §48 (prompt construction,
prompt assembly, schema and quote validation, evidence eligibility, outcome computation,
result schemas, prompt injection defenses, and RAG evaluation) are implemented.

| File | Clause |
|---|---|
| `test_prompts.py` | doc/06 §32 |
| `test_assembly.py` | doc/06 §32 |
| `test_validation.py` | doc/06 §29 |
| `test_eligibility.py` | doc/06 §28 |
| `test_outcome.py` | doc/06 §30 |
| `test_generation.py` | doc/06 §31 |
| `test_injection.py` | doc/06 §33 |
| `test_generation_eval.py` | doc/06 §34 |
