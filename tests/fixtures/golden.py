"""The golden-vector regression baseline.

A recorded set of image and caption pairs with their expected embeddings.  The
current build must reproduce them within the tolerance of document 01 §14.3.

This is the primary, offline, both-modality check on the encoder chain.  The
NewsCLIPpings parity diagnostic needs a download and covers the text path only,
so it cannot stand in for this.

The baseline is **not** committed with expected values yet - the vectors are not
known until the model has been run once on the reference machine.  Until then the
test reports that the baseline is absent, and the harness records that as a
known gap rather than a silent pass.
"""

import json
from pathlib import Path

#: Where the baseline lives.  It is derived data, so it is gitignored.
BASELINE_PATH = Path(__file__).resolve().parent.parent / "data" / "golden" / "clip_golden.json"

#: The pairs the baseline covers.  Fixed, so a later change to them invalidates
#: the baseline rather than passing silently.
PAIRS = [
    {"name": "solid", "image": {"kind": "solid_rgb", "size": [64, 48], "color": [120, 40, 200]},
     "caption": "A crowd gathers outside a courthouse on a rainy afternoon."},
    {"name": "wide", "image": {"kind": "wide_aspect", "size": [800, 100]},
     "caption": "A wide panoramic view of a city skyline at dusk."},
    {"name": "grayscale", "image": {"kind": "grayscale", "size": [64, 48]},
     "caption": "A monochrome portrait in low light."},
]

#: The tolerance the current build must reproduce the baseline within.  Matches
#: document 01 §14.3.
TOLERANCE = 1e-4


def load():
    """The recorded baseline, or ``None`` when it has not been produced yet."""
    if not BASELINE_PATH.is_file():
        return None
    try:
        return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def store(records):
    """Write a baseline."""
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return BASELINE_PATH


def compare(expected, actual, tolerance=TOLERANCE):
    """Compare two sets of recorded vectors.

    ``expected`` and ``actual`` map a name onto a sequence of floats.  The
    result says whether every pair is within the tolerance, and which pairs
    are not - so a regression names the pair that moved, not just the fact
    that something did.
    """
    mismatched = []
    for name in sorted(set(expected) | set(actual)):
        if name not in expected or name not in actual:
            mismatched.append(name)
            continue
        a = expected[name]
        b = actual[name]
        if len(a) != len(b):
            mismatched.append(name)
            continue
        worst = max((abs(x - y) for x, y in zip(a, b)), default=0.0)
        if worst > tolerance:
            mismatched.append(name)
    return {
        "matches": not mismatched,
        "mismatched": mismatched,
        "tolerance": tolerance,
    }
