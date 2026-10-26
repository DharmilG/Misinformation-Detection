"""Run the CLIP and feature-extraction test suites.

    python scripts/test_clip_features.py [--tier fast|data|slow|all] [-k TEXT]
                                        [--report PATH]

A convenience wrapper over ``python -m tests.run``, scoped to the families that
cover the encoder, the feature cache, the corpus contract, the preprocessing
skew, the extraction pipeline, the parity diagnostic, and the golden vectors.

The harness and documentation families are included as well, because they are
what keeps the specification and the implementation in step.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests import run  # noqa: E402


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--help" in argv or "-h" in argv:
        print(__doc__)
        return 0
    return run.main(argv)


if __name__ == "__main__":
    sys.exit(main())
