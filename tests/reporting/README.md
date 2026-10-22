# tests/reporting

The tests for `src/reporting/` — the report model, the per-section status
contract, the comparison matrix, the error and fallback matrix, the escaping
of untrusted text, the export, and the analysis fingerprint. Document 05 owns
this component.

**Naming.** A file here mirrors a module there:
`src/reporting/format_report.py` → `tests/reporting/test_format_report.py`.

**Status.** The skeleton only. Nothing has moved yet — the files are still
flat in `tests/`. Document 07 §2 states the layout.

| File | Clause |
|---|---|
| `test_reporting.py` | doc/05 §2, §10, §12, §15, §16 |
| `test_comparison.py` | doc/05 §8 |
