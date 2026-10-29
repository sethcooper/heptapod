"""
# yoda_dump.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Dump YODA files as JSON, for the Python that heptapod runs in.

This script runs under the Rivet installation's own interpreter -- the one
whose `yoda` module matches its library -- so heptapod's tools never import
YODA themselves. It prints one JSON document to stdout:

    {"yoda_version": "2.1.4",
     "files": [{"path": "...", "objects": [<object>, ...]}, ...]}

where each object is

    {"path": "/ANA/d01-x01-y01", "type": "Estimate1D", "title": "...",
     "annotations": {...},
     "edges": [x0, x1, ...],              binned 1D objects: axis edges
     "bins": [[lo, hi], ...],             binned 1D objects: one per bin
     "values": [...], "err_down": [...], "err_up": [...],   estimates
     "sumw": [...], "sumw2": [...], "entries": [...],       histograms
     "masked": [i, ...],
     "total": {"sumw": s, "entries": n}}           histograms and counters

Two-dimensional objects carry "edges_y" and row-major "values"; anything
else is listed with its type only. Errors are the total of every error
source, positive, so a symmetric error appears in both columns. It handles
the YODA 2 API (Rivet 4) and falls back to the YODA 1 names (Rivet 3) where
they differ.

Usage: yoda_dump.py [--patterns P1,P2] FILE [FILE ...]
"""

import argparse
import json
import math
import sys


def _call(obj, *names, default=None):
    """The first of these methods that the object has, called; else default."""
    for name in names:
        method = getattr(obj, name, None)
        if method is not None:
            try:
                return method()
            except (
                Exception
            ):  # a YODA accessor that needs an argument, or a RangeError
                continue
    return default


def _float(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _bin_errors(bin_):
    """(down, up) of a bin's total error, both positive."""
    pair = _call(bin_, "totalErr", "quadSum")
    if pair is not None:
        try:
            down, up = pair
            return abs(float(down)), abs(float(up))
        except (TypeError, ValueError):
            pass
    neg = _call(bin_, "totalErrNeg", "errMinus")
    pos = _call(bin_, "totalErrPos", "errPlus")
    if neg is not None and pos is not None:
        return abs(float(neg)), abs(float(pos))
    avg = _call(bin_, "totalErrAvg", "errAvg", "err", default=0.0)
    try:
        avg = abs(float(avg))
    except (TypeError, ValueError):
        avg = 0.0
    return avg, avg


def _annotations(obj):
    try:
        return {str(k): str(v) for k, v in obj.annotationsDict().items()}
    except Exception:
        out = {}
        for key in _call(obj, "annotations", default=[]) or []:
            try:
                out[str(key)] = str(obj.annotation(key))
            except Exception:
                pass
        return out


def _edges_1d(obj):
    edges = _call(obj, "xEdges")
    if edges is None:
        bins = obj.bins()
        edges = [b.xMin() for b in bins] + ([bins[-1].xMax()] if bins else [])
    return [float(e) for e in edges]


def _bin_ranges(bins):
    """Each bin's [low, high]: the axis edges alone cannot say where a
    binning has a gap, and a few reference histograms have one."""
    return [[_float(b.xMin()), _float(b.xMax())] for b in bins]


def _estimate_1d(obj, record):
    bins = obj.bins()
    record["edges"] = _edges_1d(obj)
    record["bins"] = _bin_ranges(bins)
    values, down, up, masked = [], [], [], []
    for index, b in enumerate(bins):
        if _call(b, "isMasked", default=False):
            masked.append(index)
        values.append(_float(_call(b, "val", "y", default=float("nan"))))
        d, u = _bin_errors(b)
        down.append(_float(d))
        up.append(_float(u))
    record.update(
        {"values": values, "err_down": down, "err_up": up, "masked": masked}
    )


def _histo_1d(obj, record):
    bins = obj.bins()
    record["edges"] = _edges_1d(obj)
    record["bins"] = _bin_ranges(bins)
    record["sumw"] = [_float(b.sumW()) for b in bins]
    record["sumw2"] = [_float(b.sumW2()) for b in bins]
    record["entries"] = [_float(b.numEntries()) for b in bins]
    record["masked"] = [
        i for i, b in enumerate(bins) if _call(b, "isMasked", default=False)
    ]
    record["total"] = {
        "sumw": _float(_call(obj, "sumW", default=0.0)),
        "sumw2": _float(_call(obj, "sumW2", default=0.0)),
        "entries": _float(_call(obj, "numEntries", default=0.0)),
    }
    # Overflow is informative: a histogram whose entries all missed its range
    # was probably filled in the wrong units.
    try:
        with_overflow = obj.bins(True)
        record["total"]["entries_in_range"] = _float(
            sum(b.numEntries() for b in bins)
        )
        record["total"]["entries_all"] = _float(
            sum(b.numEntries() for b in with_overflow)
        )
    except Exception:
        pass


def _estimate_0d(obj, record):
    record["values"] = [_float(_call(obj, "val", default=float("nan")))]
    down, up = _bin_errors(obj)
    record["err_down"], record["err_up"] = [_float(down)], [_float(up)]


def _counter(obj, record):
    record["total"] = {
        "sumw": _float(obj.sumW()),
        "sumw2": _float(_call(obj, "sumW2", default=0.0)),
        "entries": _float(obj.numEntries()),
    }


def _scatter_2d(obj, record):
    points = obj.points()
    record["x"] = [_float(p.x()) for p in points]
    record["edges"] = [_float(p.xMin()) for p in points] + (
        [_float(points[-1].xMax())] if points else []
    )
    record["values"] = [_float(p.y()) for p in points]
    record["err_down"] = [_float(abs(p.yErrMinus())) for p in points]
    record["err_up"] = [_float(abs(p.yErrPlus())) for p in points]
    record["masked"] = []


def _binned_2d(obj, record, estimate):
    record["edges"] = [float(e) for e in obj.xEdges()]
    record["edges_y"] = [float(e) for e in obj.yEdges()]
    bins = obj.bins()
    if estimate:
        record["values"] = [
            _float(_call(b, "val", default=float("nan"))) for b in bins
        ]
        errors = [_bin_errors(b) for b in bins]
        record["err_down"] = [_float(d) for d, _ in errors]
        record["err_up"] = [_float(u) for _, u in errors]
    else:
        record["sumw"] = [_float(b.sumW()) for b in bins]
        record["sumw2"] = [_float(b.sumW2()) for b in bins]
        record["entries"] = [_float(b.numEntries()) for b in bins]
    record["masked"] = [
        i for i, b in enumerate(bins) if _call(b, "isMasked", default=False)
    ]


def dump_object(path, obj):
    kind = str(_call(obj, "type", default=type(obj).__name__))
    record = {
        "path": path,
        "type": kind,
        "title": str(_call(obj, "title", default="")),
        "annotations": _annotations(obj),
    }
    try:
        if kind in ("Estimate1D", "BinnedEstimate1D"):
            _estimate_1d(obj, record)
        elif kind in (
            "Histo1D",
            "BinnedHisto1D",
            "Profile1D",
            "BinnedProfile1D",
        ):
            _histo_1d(obj, record)
        elif kind == "Estimate0D":
            _estimate_0d(obj, record)
        elif kind == "Counter":
            _counter(obj, record)
        elif kind == "Scatter2D":
            _scatter_2d(obj, record)
        elif kind in ("Estimate2D", "BinnedEstimate2D"):
            _binned_2d(obj, record, estimate=True)
        elif kind in (
            "Histo2D",
            "BinnedHisto2D",
            "Profile2D",
            "BinnedProfile2D",
        ):
            _binned_2d(obj, record, estimate=False)
        else:
            record["unsupported"] = True
    except Exception as e:  # one odd object must not sink the whole dump
        record["error"] = f"{type(e).__name__}: {e}"
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description="Dump YODA files as JSON")
    parser.add_argument(
        "--patterns", default="", help="comma-separated path regexes to keep"
    )
    parser.add_argument("files", nargs="+")
    args = parser.parse_args(argv)

    import yoda  # noqa: PLC0415 - the Rivet installation's module, found via PYTHONPATH

    patterns = [p for p in args.patterns.split(",") if p]
    out = {
        "yoda_version": str(_call(yoda, "version", default="")),
        "files": [],
    }
    for path in args.files:
        try:
            objects = yoda.read(path, asdict=True, patterns=patterns or "")
        except Exception as e:
            out["files"].append(
                {"path": path, "error": f"{type(e).__name__}: {e}"}
            )
            continue
        out["files"].append(
            {
                "path": path,
                "objects": [
                    dump_object(p, obj) for p, obj in sorted(objects.items())
                ],
            }
        )
    json.dump(out, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
