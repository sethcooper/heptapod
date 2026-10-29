"""
# digest.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Digests for research tools whose results can be large (literature, INSPIRE
and HEPData searches, HEPData records and tables).

The full structured result is written to the sandbox:

  <base_directory>/research/results/<call_id>.json

next to the raw response that :func:`tools.provenance.record_call` saves as
``research/calls/<call_id>.<ext>``, and the tool returns a digest instead: the
request echo and counts the tool chooses, ``results_path``, a compact list of
entries, and a ``note``. The list is cut from the end so the digest, without
its ``provenance`` block, serialises to at most ``budget`` characters; the
number of entries not listed is ``omitted``.

Writing never breaks the tool: if the sandbox cannot be written,
``results_path`` is None, the note says the full result was not saved, and the
digest is still returned.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from .provenance import attach

RESULTS_DIR = os.path.join("research", "results")
DIGEST_BUDGET = 6000


def clip(value: Any, limit: int) -> Any:
    """A string cut to ``limit`` characters with an ellipsis; other values as is."""
    if not isinstance(value, str) or len(value) <= limit:
        return value
    return value[: max(0, limit - 1)].rstrip() + "…"


def save_result(base_directory: str, call_id: str, result: Any) -> Optional[str]:
    """Write ``result`` as pretty JSON; the path relative to base_directory, or None."""
    try:
        base = os.path.realpath(base_directory)
        os.makedirs(os.path.join(base, RESULTS_DIR), exist_ok=True)
        rel = os.path.join(RESULTS_DIR, f"{call_id}.json")
        with open(os.path.join(base, rel), "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, default=str)
            fh.write("\n")
        return rel
    except (OSError, TypeError, ValueError):
        return None


def _size(obj: Dict[str, Any]) -> int:
    return len(json.dumps(obj, indent=2, default=str))


def digest(
    base_directory: str,
    result: Dict[str, Any],
    provenance: Dict[str, Any],
    head: Dict[str, Any],
    key: str,
    entries: List[Any],
    *,
    available: Optional[int] = None,
    what: str = "entries",
    budget: int = DIGEST_BUDGET,
) -> str:
    """Save ``result`` and return ``head`` + ``results_path`` + ``entries`` as JSON.

    ``key`` names the compact list in the digest (the same key as the full
    list in the saved file). ``available`` is how many entries the saved file
    holds (default ``len(entries)``); ``omitted`` counts those not listed.
    ``what`` names the entries in the note ("hits", "tables", "rows").
    """
    path = save_result(base_directory, provenance.get("call_id") or "result", result)
    total = len(entries) if available is None else available

    def build(n: int) -> Dict[str, Any]:
        obj = dict(head)
        obj["results_path"] = path
        obj[key] = entries[:n]
        if total > n:
            obj["omitted"] = total - n
        if path:
            listed = f"the first {n} of {total} {what}" if total > n else f"all {total} {what}"
            obj["note"] = (f"Compact digest listing {listed}; every field of every "
                           f"entry is in results_path (JSON, readable with Python or jq).")
        else:
            obj["note"] = ("Compact digest; the full result could not be written to "
                           "the sandbox, so only these fields are available.")
        return obj

    lo, hi = 0, len(entries)
    if _size(build(hi)) > budget:
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if _size(build(mid)) <= budget:
                lo = mid
            else:
                hi = mid - 1
    return attach(build(hi), provenance)
