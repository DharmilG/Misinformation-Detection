# `tests/evaluation/`

Mirrors `src/evaluation/` per doc/07 §2.6.

| File | Owns | Clause |
|---|---|---|
| `test_research_runtime.py` | Input validation and mechanical variants of the Study 1 audit | doc/07 §3, §12.3 |
| `test_failure_audit.py` | Paired metrics and placeholders of the Study 1 reporter | doc/07 §3, §12.3 |
| `test_rag_diagnostics.py` | Stage tracing and failure attribution of the Study 4 RAG ablation | doc/06 §42, doc/07 §12.3 |

All cases are `fast`: they use synthetic rows and stub modules, so no model weights, corpus, or network are needed.

The scripts these cases verify are:

- `scripts/run_failure_audit.py` / `scripts/report_failure_audit.py`
- `scripts/run_rag_ablation.py` / `scripts/report_rag_ablation.py` / `scripts/run_rag_fixes.py`
