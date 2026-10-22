# tests/spec

The tests that belong to no component: they check the **documentation** and
the **suite itself**. Document 07 owns this folder.

These are not tests of the system. They are what stops the specification and
the implementation from drifting apart, which is what makes the rest of the
suite trustworthy:

- the project overview is byte-identical to the version the documentation set
  was written against;
- the documentation directory holds exactly the documents it should;
- every shared contract has a canonical owner section, and it exists;
- every collected test carries a registered case id, and the ids are unique;
- the tiers, the commands, and the traceability table in document 07 match
  what the suite actually does.

**Status.** The skeleton only. Nothing has moved yet — the files are still
flat in `tests/`. Document 07 §2 states the layout.

| File | Clause |
|---|---|
| `test_docs.py` | doc/00 §11, §11.1, doc/02 §4.2, doc/07 §11, §12 |
| `test_suite.py` | doc/07 §2.1, §2.2, §2.3, §12 |
| `test_harness.py` | doc/07 §3 |
| `test_e2e_cases.py` | doc/07 §8 |
| `test_rates.py` | doc/07 §9 |

`test_suite.py` and `test_harness.py` are different things and must not share
a name: the first checks this suite's own invariants, the second checks the
*evaluation* harness that produces the reported metrics.
