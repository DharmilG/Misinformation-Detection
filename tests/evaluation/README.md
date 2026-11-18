# `tests/evaluation/`

Mirrors `src/evaluation/` per doc/07 §2.6.

| File | Owns | Clause |
|---|---|---|
| `test_research_runtime.py` | Input validation and mechanical variants of the Study 1 audit | doc/07 §12.2 |
| `test_failure_audit.py` | Paired metrics and placeholders of the Study 1 reporter | doc/07 §12.2 |
| `test_templates.py` | Template generation without touching the held-out test split | doc/07 §12.2 |
| `test_rag_diagnostics.py` | Stage tracing and failure attribution of the Study 4 RAG ablation | doc/06 §42, doc/07 §12.2 |
| `test_selective_prediction.py` | Risk–coverage, AURC and confidence gating for Study 2 | doc/07 §12.2 |
| `test_human_study.py` | Counterbalancing, anonymous logging and reporting for Study 3 | doc/07 §12.2 |

All cases are `fast`: they use synthetic rows and stub modules, so no model weights, corpus, or network are needed.

The scripts these cases verify are:

- `scripts/make_study_templates.py` — pre-filled research manifests
- `scripts/run_failure_audit.py` / `scripts/report_failure_audit.py`
- `scripts/run_selective_prediction.py`
- `scripts/run_rag_ablation.py` / `scripts/report_rag_ablation.py` / `scripts/run_rag_fixes.py`
- `scripts/run_human_study.py` / `scripts/report_human_study.py`
