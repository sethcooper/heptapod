"""
# yoda_io.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Read YODA files through the Rivet installation.

heptapod never imports YODA: `yoda_dump.py` runs under the install's Python
and prints JSON, and this module turns that into plain dictionaries the
comparison and plotting tools work on. Every histogram is a dict with the
keys `yoda_dump.py` documents -- `path`, `type`, `edges`, `values`,
`err_down`, `err_up`, `sumw`, `entries`, ...

Path conventions, as Rivet writes them:

    /ANA/d01-x01-y01              a final histogram of analysis ANA
    /ANA:OPT=V/d01-x01-y01        the same, from a run with an analysis option
    /ANA/d01-x01-y01[varname]     the same, for one weight variation
    /RAW/ANA/d01-x01-y01          the unscaled histogram as filled
    /RAW/_EVTCOUNT, /RAW/_XSEC    the run's event count and cross section
    /REF/ANA/d01-x01-y01          the measurement, in the analysis's data file
"""

import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tools.rivet.install import RivetError, RivetInstall, run_python

_DUMP_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "yoda_dump.py"
)

_VARIATION = re.compile(r"\[([^\]]*)\]$")


def dump_yoda(
    install: RivetInstall,
    paths: Sequence[str],
    patterns: Sequence[str] = (),
    timeout: float = 600,
) -> Dict[str, Any]:
    """
    The JSON `yoda_dump.py` prints for these files, as a dict.

    Raises:
        RivetError: when the helper fails, with its stderr
    """
    args = []
    if patterns:
        args += ["--patterns", ",".join(patterns)]
    returncode, out, err = run_python(
        install, _DUMP_SCRIPT, [*args, *paths], timeout=timeout
    )
    if returncode != 0:
        raise RivetError(f"reading YODA files failed:\n{err.strip()[-2000:]}")
    try:
        return json.loads(out)
    except json.JSONDecodeError as e:
        raise RivetError(
            f"unreadable output from yoda_dump.py: {e}\n{err.strip()[-1000:]}"
        ) from e


def objects_of(
    dump: Dict[str, Any], index: int = 0
) -> Dict[str, Dict[str, Any]]:
    """The objects of one dumped file, by path."""
    entry = dump["files"][index]
    if "error" in entry:
        raise RivetError(f"{entry['path']}: {entry['error']}")
    return {obj["path"]: obj for obj in entry["objects"]}


def read_yoda(
    install: RivetInstall, path: str, patterns: Sequence[str] = ()
) -> Dict[str, Dict[str, Any]]:
    """The objects of one YODA file, by path."""
    return objects_of(dump_yoda(install, [path], patterns))


def is_variation(path: str) -> bool:
    """Whether a path is one weight variation's copy of a histogram."""
    return bool(_VARIATION.search(path))


def variation_name(path: str) -> str:
    match = _VARIATION.search(path)
    return match.group(1) if match else ""


def nominal_path(path: str) -> str:
    """The path with any weight-variation suffix removed."""
    return _VARIATION.sub("", path)


def is_raw(path: str) -> bool:
    return path.startswith("/RAW/")


def is_reference(path: str) -> bool:
    return path.startswith("/REF/")


def is_internal(path: str) -> bool:
    """Rivet's own bookkeeping objects (_EVTCOUNT, _XSEC, _BEAMPZ, ...)."""
    parts = [p for p in nominal_path(path).split("/") if p]
    return any(part.startswith("_") for part in parts) or path.startswith(
        "/TMP/"
    )


def split_path(path: str) -> Tuple[str, str]:
    """
    (analysis, histogram) of a final or reference path, the analysis keeping
    its option string and the histogram its variation suffix.
    """
    parts = [p for p in path.split("/") if p]
    if parts and parts[0] in ("REF", "RAW"):
        parts = parts[1:]
    if len(parts) < 2:
        return (parts[0] if parts else ""), ""
    return parts[0], "/".join(parts[1:])


def analysis_name(path: str) -> str:
    """The analysis a path belongs to, without options."""
    return split_path(path)[0].split(":", 1)[0]


def reference_path(path: str) -> str:
    """The /REF path that holds the measurement a histogram is compared to."""
    analysis, histogram = split_path(nominal_path(path))
    return f"/REF/{analysis.split(':', 1)[0]}/{histogram}"


def bin_ranges(obj: Dict[str, Any]) -> List[Tuple[float, float]]:
    """
    Each bin's (low, high) of a one-dimensional object, aligned with its
    values.

    From the per-bin ranges the dump records when it has them; otherwise
    from consecutive axis edges, which is the same thing unless the binning
    has a gap.
    """
    if obj.get("bins"):
        return [(float(lo), float(hi)) for lo, hi in obj["bins"]]
    edges = obj.get("edges") or []
    return list(zip(edges[:-1], edges[1:]))


def final_histograms(
    objects: Dict[str, Dict[str, Any]], *, with_variations: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    The histograms an analysis produced for comparison: not raw, not
    reference, not Rivet's bookkeeping, and nominal unless asked otherwise.
    """
    return {
        path: obj
        for path, obj in objects.items()
        if not is_raw(path)
        and not is_reference(path)
        and not is_internal(path)
        and (with_variations or not is_variation(path))
    }


def variations_in(objects: Iterable[str]) -> List[str]:
    """The weight-variation names present, in first-seen order."""
    seen: Dict[str, None] = {}
    for path in objects:
        name = variation_name(path)
        if name:
            seen.setdefault(name, None)
    return list(seen)


def event_count(objects: Dict[str, Dict[str, Any]]) -> Optional[float]:
    """The number of events a Rivet run processed, from its _EVTCOUNT
    counter."""
    counter = objects.get("/RAW/_EVTCOUNT") or objects.get("/_EVTCOUNT")
    if counter and counter.get("total"):
        return counter["total"].get("entries")
    return None


def cross_section_pb(
    objects: Dict[str, Dict[str, Any]],
) -> Optional[Tuple[float, float]]:
    """The (value, error) in pb a Rivet run used, from its _XSEC estimate."""
    xsec = objects.get("/RAW/_XSEC") or objects.get("/_XSEC")
    if xsec and xsec.get("values"):
        value = xsec["values"][0]
        error = (xsec.get("err_up") or [0.0])[0]
        if value is not None:
            return float(value), float(error or 0.0)
    return None


def has_content(obj: Dict[str, Any]) -> bool:
    """Whether a dumped histogram holds anything but zeros."""
    if obj.get("entries") is not None:
        return any((e or 0) > 0 for e in obj["entries"])
    if obj.get("total") and obj["total"].get("entries") is not None:
        return (obj["total"]["entries"] or 0) > 0
    values = obj.get("values") or []
    return any(v is not None and v != 0 for v in values)


def binned_1d(obj: Dict[str, Any]) -> bool:
    """Whether an object is a one-dimensional binned estimate or histogram."""
    return (
        "edges" in obj
        and "edges_y" not in obj
        and obj.get("values") is not None
    )
