# tests/retrieval

The tests for `src/retrieval/` — the evidence corpus, ingestion, chunking, the
embeddings, the index, the lexical and dense retrieval, fusion, the filters,
deduplication, and the retrieval API. Document 06 §47 owns this component.

**Naming.** A file here mirrors a module there:
`src/retrieval/filters.py` → `tests/retrieval/test_filters.py`.

**Status.** The skeleton only. Nothing has moved yet — the files are still
flat in `tests/`. Document 07 §2 states the layout.

| File | Clause |
|---|---|
| `test_corpus_terms.py` | doc/06 §6 |
| `test_ingest_safety.py` | doc/06 §9 |
| `test_chunking.py` | doc/06 §10, §11 |
| `test_embeddings.py` | doc/06 §10 |
| `test_index.py` | doc/06 §14 |
| `test_lexical.py` | doc/06 §14 |
| `test_fusion.py` | doc/06 §19 |
| `test_filters.py` | doc/06 §19 |
| `test_dedup.py` | doc/06 §19 |
| `test_retriever.py` | doc/06 §19 |
| `test_retrieval_eval.py` | doc/06 §9 |
