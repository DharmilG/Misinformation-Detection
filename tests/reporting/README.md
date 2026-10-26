# tests/reporting

The tests for `src/reporting/` — the report model, the per-section status
contract, the comparison matrix, the error and fallback matrix, the escaping
of untrusted text, the export, and the analysis fingerprint. Document 05 owns
this component.

**Naming.** A file here mirrors a module there:
`src/reporting/report_model.py` → `tests/reporting/test_reporting.py`.
`src/reporting/comparison.py` → `tests/reporting/test_comparison.py`.

**Status.** Populated. Both contract test suites for Document 05 (report model,
status contract, error/fallback matrix, fingerprint, export, escaping, and comparison
directions matrix) are implemented.

| File | Clause |
|---|---|
| `test_reporting.py` | doc/05 §2, §10, §11, §12, §15, §16, §18 |
| `test_comparison.py` | doc/05 §8 |
