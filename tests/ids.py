"""The case registry.

Every test in the suite carries a ``case`` marker that names it, states the
document clause it enforces, and carries the requirement in words.  The registry
is what makes the suite a conformance report rather than a pile of asserts.

``tests/test_harness.py`` asserts that every collected test is registered, so a
test added without a case fails the suite.
"""

import pytest

#: Test families, and what each one covers.
FAMILIES = {
    "F1": "Encoder behaviour - the API that exists today, against the behavioural clauses",
    "F2": "Interface conformance - the API surface, provenance, device policy and error contract",
    "F3": "Preprocessing conformance - EXIF, alpha, downscale, colour mode",
    "F4": "Feature cache - the key, the metadata, the atomic write, load validation",
    "F5": "Dataset contract - the corpus integrity, its label gate, its split structure",
    "F6": "Preprocessing skew - the accepted 256 px pre-resize, measured",
    "F7": "Feature pipeline - the dataset-scale extraction and its alignment",
    "F8": "Parity - the narrowed NewsCLIPpings parity diagnostic",
    "F9": "Golden vectors - the offline regression on the encoder chain",
    "F10": "Harness and documentation - the suite and the documents agree",
}

#: The tiers, and what each one requires.
TIERS = {
    "fast": "the runtime dependencies installed; no model weights, no dataset, no network",
    "data": "plus the local corpus at dataset/ooc_dataset/",
    "slow": "plus the CLIP model weights",
    "eval": "plus the labelled evaluation sets",
    "benchmark": "plus the candidate models and corpora",
    "e2e": "the full local stack",
}


def case(cid, family, tier, clause, requirement, xfail_reason=None, requires=()):
    """Declare a test case.

    ``tier`` is a tier name, or a tuple of them when a case needs more than
    one - the preprocessing-skew measurement needs the corpus *and* the
    weights, and is marked ``("data", "slow")``.

    ``xfail_reason`` marks a requirement the implementation does not yet meet.
    The harness turns it into ``xfail(strict=True)``, so the suite is green
    today, a fixed gap reports "unexpectedly passing" and fails, and the
    declaration is the single place the gap is recorded.

    ``requires`` names importable modules the case needs.  A missing one is a
    skip with a reason, never a failure.
    """
    if isinstance(tier, str):
        tier = (tier,)
    return pytest.mark.case(
        cid=cid,
        family=family,
        tier=tuple(tier),
        clause=clause,
        requirement=requirement,
        xfail_reason=xfail_reason,
        requires=tuple(requires),
    )


def known_gap(cid, family, tier, clause, requirement, reason, requires=()):
    """Declare a case whose requirement is not yet implemented."""
    return case(cid, family, tier, clause, requirement, xfail_reason=reason, requires=requires)
