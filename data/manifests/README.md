# Manifests

The manifest for each split configuration is written here as JSON Lines, one
record per row, together with a `manifest_meta.json` that records everything
needed to reproduce it.

```text
data/manifests/
    README.md
    newsclippings-official-v1/
        train.jsonl
        val.jsonl
        test.jsonl
        manifest_meta.json
    newsclippings-grouped-v1/
        ...
    newsclippings-dev-v1/
        ...
```

**This directory is excluded from Git**, apart from this file. The manifests
carry the corpus's captions and the hashes derived from them, so they inherit
the corpus's redistribution restriction even though they contain no pixels.
See `doc/02_DATASET_PREPARATION.md` §4.2 and §4.3.

The manifests are produced by `scripts/prepare_dataset.py` from the corpus at
`dataset/ooc_dataset/`, and the corpus itself is checked first by
`scripts/audit_dataset.py`, which writes `reports/dataset_audit.json` and fails
loudly if a gate check fails.
