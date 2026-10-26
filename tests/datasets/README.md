# tests/datasets

The tests for `src/datasets/` — the corpus, its label gate, its manifests, its
splits, and the extraction of the dataset-scale features. Document 02 owns
this component.

**Naming.** A file here mirrors a module there:
`src/datasets/manifests.py` → `tests/datasets/test_manifests.py`.

**Status.** Populated for `test_corpus.py`, `test_skew.py`, and `test_pipeline.py`.
The remaining files listed below are planned. Document 07 §2 states the layout.

| File | Clause |
|---|---|
| `test_corpus.py` | doc/02 §4.1, §4.2, §5, §6.2, §6.4, §7, §8.1.1, §8.2, §9, §12, §13.1 |
| `test_skew.py` | doc/02 §8.1.1, §8.1.2, doc/04 §7.4.1 |
| `test_pipeline.py` | doc/02 §14, §14.3, doc/01 §11.2, §11.3, §14.2 |
| `test_labels.py` | doc/02 §5, §6.2, §6.3 |
| `test_manifests.py` | doc/02 §12.2, §12.4, §12.5, §12.6 |
| `test_splits.py` | doc/02 §9 |
| `test_leakage.py` | doc/02 §10, §11 |
