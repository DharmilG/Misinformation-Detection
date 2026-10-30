"""The label mapping contract of Document 02 §6.

Defines the unified 3-class Label enum (MATCHED, MISMATCHED, UNKNOWN) and maps
raw annotations while preserving source provenance.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class Label(str, Enum):
    MATCHED = "MATCHED"
    MISMATCHED = "MISMATCHED"
    UNKNOWN = "UNKNOWN"


def map_label(val: Any) -> Label:
    """Map source annotation value to unified Label enum."""
    if val is None:
        return Label.UNKNOWN
    s = str(val).strip()
    if s == "0":
        return Label.MATCHED
    if s == "1":
        return Label.MISMATCHED
    return Label.UNKNOWN


def annotate_record(record: dict[str, Any]) -> dict[str, Any]:
    """Annotate raw record with unified label, source_label, and source_label_raw."""
    rec = dict(record)
    raw = rec.get("label")
    mapped = map_label(raw)

    raw_val: Any = raw
    if raw is not None and str(raw).strip() in ("0", "1"):
        raw_val = int(str(raw).strip())

    source_label_str = "unknown"
    if mapped == Label.MATCHED:
        source_label_str = "consistent"
    elif mapped == Label.MISMATCHED:
        source_label_str = "out_of_context"

    rec["label"] = mapped
    rec["source_label"] = source_label_str
    rec["source_label_raw"] = raw_val
    return rec
