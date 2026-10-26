# tests/clip_model

The tests for `src/clip_model/` — the CLIP encoder, the feature cache, the
parity diagnostic, the golden-vector regression, and the classifier.
Document 01 owns the encoder and the cache; document 03 owns the classifier.

**Naming.** A file here mirrors a module there:
`src/clip_model/clip_features.py` → `tests/clip_model/test_clip_features.py`.

**Status.** Populated. All 15 contract test suites for Document 01 and
Document 03 (encoder, interface, preprocessing, cache, parity diagnostic,
golden vectors, classifier architecture, dataset reader, metrics, calibration,
training protocol, safe checkpoints, packaging gates, compatibility, and verification)
are implemented.

| File | Clause |
|---|---|
| `test_encoder.py` | doc/01 §5.2, §6.2, §6.3, §7.2, §8.1, §8.2, §8.3, §14.2 |
| `test_interface.py` | doc/01 §4.2, §4.3, §6.1, §7.1, §9.1, §13.1, §13.2 |
| `test_preprocessing.py` | doc/04 §7.1, §7.2, §7.3, §7.4, doc/02 §8.1.2 |
| `test_cache.py` | doc/01 §11.1, §11.2, §11.3, §11.4, §11.5, §11.6, §11.7 |
| `test_parity.py` | doc/01 §12.2, §12.4, §12.7, §12.8, §15.7 |
| `test_golden.py` | doc/01 §14.3, §15.5 |
| `test_classifier.py` | doc/03 §4, §10, §12 |
| `test_dataset.py` | doc/03 §5 |
| `test_metrics.py` | doc/03 §8 |
| `test_calibration.py` | doc/03 §12 |
| `test_training.py` | doc/03 §7, §9 |
| `test_checkpoint.py` | doc/03 §27, §28 |
| `test_packaging.py` | doc/03 §17, §25 |
| `test_compatibility.py` | doc/03 §22 |
| `test_verification.py` | doc/03 §28 |
