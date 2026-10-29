"""
# plot.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Draw Rivet histograms against the measured data.

The figures follow the layout of Rivet's own plots: the measurement as
points with error bars, each prediction as a stepped line, a ratio panel of
prediction over data beneath, and the titles and axis labels from the
analysis's `.plot` file. They are drawn with matplotlib rather than Rivet's
`rivet-mkhtml`, which needs a LaTeX installation; the labels' LaTeX is
rendered by matplotlib's own mathtext where it can be and shown as plain
text where it cannot.
"""

import html
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from orchestral.tools.base.field_utils import RuntimeField

from tools.rivet import yoda_io
from tools.rivet.base import RivetToolBase, runtime_list
from tools.rivet.catalogue import (
    base_name,
    load_catalogue,
    plot_style,
    plot_styles,
)
from tools.rivet.compare import compare_histograms
from tools.rivet.install import RivetError

IMAGE_FORMATS = ("png", "pdf", "svg")

# Rivet's plotting macros, in matplotlib mathtext terms.
_MACROS = [
    (r"\pT", r"p_\mathrm{T}"),
    (r"\pt", r"p_\mathrm{T}"),
    (r"\ET", r"E_\mathrm{T}"),
    (r"\TeV", r"\,\mathrm{TeV}"),
    (r"\GeV", r"\,\mathrm{GeV}"),
    (r"\MeV", r"\,\mathrm{MeV}"),
    (r"\text", r"\mathrm"),
    (r"\textrm", r"\mathrm"),
    (r"\mathsf", r"\mathrm"),
    (r"\rm ", r"\mathrm "),
    (r"\unit", r"\,"),
    (r"\antibar", r"\bar"),
    (r"\ensuremath", ""),
]
# Colours and line styles for successive predictions.
_COLOURS = [
    "#d62728",
    "#1f77b4",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#17becf",
]


def mathtext(label: str) -> str:
    """A .plot label as matplotlib mathtext, macros translated."""
    out = label
    for macro, replacement in _MACROS:
        out = out.replace(macro, replacement)
    # \frac{a}{b} is fine; \dfrac is not.
    out = out.replace(r"\dfrac", r"\frac")
    return out


def plain_text(label: str) -> str:
    """A label with its LaTeX stripped, for when mathtext refuses it."""
    out = re.sub(r"\\([a-zA-Z]+)", r"\1 ", label)
    return " ".join(re.sub(r"[${}^_*]", "", out).split())


def _as_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    return value.strip() in ("1", "true", "True", "yes")


def _as_float(value: Optional[str]) -> Optional[float]:
    try:
        return float(value) if value is not None and value.strip() else None
    except ValueError:
        return None


def _step_xy(
    ranges: Sequence[Tuple[float, float]], values: Sequence[Optional[float]]
) -> Tuple[List[float], List[float]]:
    """x and y for a stepped line over the bins' (low, high) ranges, NaN
    where a bin has no value and across a gap in the binning."""
    xs: List[float] = []
    ys: List[float] = []
    previous_hi = None
    for (lo, hi), v in zip(ranges, values):
        if previous_hi is not None and lo > previous_hi:
            xs += [previous_hi, lo]
            ys += [float("nan"), float("nan")]
        y = float("nan") if v is None else v
        xs += [lo, hi]
        ys += [y, y]
        previous_hi = hi
    return xs, ys


def _envelope(
    variations: Sequence[Dict[str, Any]],
) -> Optional[Tuple[List[float], List[float]]]:
    """Per-bin (min, max) over the weight variations of one histogram."""
    if not variations:
        return None
    n = len(variations[0].get("values") or [])
    lows, highs = [], []
    for i in range(n):
        column = [
            v["values"][i]
            for v in variations
            if len(v.get("values") or []) == n and v["values"][i] is not None
        ]
        if not column:
            lows.append(float("nan"))
            highs.append(float("nan"))
        else:
            lows.append(min(column))
            highs.append(max(column))
    return lows, highs


def draw_histogram(
    path: str,
    style: Dict[str, str],
    ref: Optional[Dict[str, Any]],
    predictions: Sequence[
        Tuple[str, Dict[str, Any], Optional[Tuple[List[float], List[float]]]]
    ],
    output: str,
    *,
    ratio: bool = True,
    dpi: int = 150,
    comparisons: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
) -> None:
    """
    Draw one histogram: data points, stepped predictions with optional
    variation bands, and the ratio panel.

    Parameters:
        path: The histogram path, used as a fallback title
        style: The merged .plot settings for it
        ref: The dumped reference estimate, or None for a prediction-only plot
        predictions: (label, dumped estimate, envelope) per file
        output: Image file to write; the suffix selects the format
        ratio: Draw the prediction/data panel when there is data
        comparisons: compare_histograms results per prediction, for the legend
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    with_ratio = ratio and ref is not None
    if with_ratio:
        fig, (ax, ax_ratio) = plt.subplots(
            2,
            1,
            figsize=(6.4, 6.4),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
        )
    else:
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        ax_ratio = None

    edges = (ref or predictions[0][1])["edges"]
    ranges = yoda_io.bin_ranges(ref or predictions[0][1])
    centres = [(lo + hi) / 2 for lo, hi in ranges]
    half_widths = [(hi - lo) / 2 for lo, hi in ranges]
    positive: List[float] = []

    if ref is not None:
        y = np.array(
            [np.nan if v is None else v for v in ref["values"]], dtype=float
        )
        err = np.array(
            [
                [d or 0.0 for d in ref["err_down"]],
                [u or 0.0 for u in ref["err_up"]],
            ],
            dtype=float,
        )
        ax.errorbar(
            centres,
            y,
            yerr=err,
            xerr=half_widths,
            fmt="o",
            color="black",
            markersize=3.5,
            capsize=0,
            linewidth=1.0,
            label="Data",
            zorder=5,
        )
        positive += [v for v in y if np.isfinite(v) and v > 0]

    for index, (label, obj, envelope) in enumerate(predictions):
        colour = _COLOURS[index % len(_COLOURS)]
        xs, ys = _step_xy(yoda_io.bin_ranges(obj), obj["values"])
        legend = label
        if (
            comparisons
            and index < len(comparisons)
            and comparisons[index]
            and comparisons[index].get("comparable")
        ):
            c = comparisons[index]
            legend = f"{label}  ($\\chi^2$/ndf = {c['chi2']:.1f}/{c['ndf']})"
        ax.plot(
            xs,
            ys,
            color=colour,
            linewidth=1.4,
            label=legend,
            zorder=4 - index * 0.01,
        )
        # The prediction's own statistical uncertainty, as a light band.
        mc_err = [
            (d or 0.0, u or 0.0)
            for d, u in zip(obj["err_down"], obj["err_up"])
        ]
        lo_stat = [
            v - d if v is not None else np.nan
            for v, (d, _) in zip(obj["values"], mc_err)
        ]
        hi_stat = [
            v + u if v is not None else np.nan
            for v, (_, u) in zip(obj["values"], mc_err)
        ]
        ax.fill_between(
            *_step_xy(yoda_io.bin_ranges(obj), lo_stat),
            _step_xy(yoda_io.bin_ranges(obj), hi_stat)[1],
            color=colour,
            alpha=0.18,
            linewidth=0,
            step=None,
            zorder=2,
        )
        if envelope is not None:
            ax.fill_between(
                *_step_xy(yoda_io.bin_ranges(obj), envelope[0]),
                _step_xy(yoda_io.bin_ranges(obj), envelope[1])[1],
                color=colour,
                alpha=0.12,
                linewidth=0,
                hatch="////",
                zorder=1,
                label=f"{label} weight variations" if index == 0 else None,
            )
        positive += [v for v in obj["values"] if v is not None and v > 0]

    title = style.get("Title", "") or path
    ax.set_title(mathtext(title), fontsize=10)
    y_label = style.get("YLabel", "")
    if y_label:
        ax.set_ylabel(mathtext(y_label), fontsize=10)
    x_label = style.get("XLabel", "")
    (ax_ratio or ax).set_xlabel(mathtext(x_label), fontsize=10)

    log_x = _as_bool(style.get("LogX"))
    log_y = _as_bool(style.get("LogY"))
    if log_y is None:
        # Rivet's default is a log scale; keep it unless the values do not
        # allow one.
        log_y = bool(positive) and (
            max(positive) / min(positive) > 20 if min(positive) > 0 else False
        )
    if log_x:
        ax.set_xscale("log")
    if log_y and positive:
        ax.set_yscale("log")
    x_min, x_max = _as_float(style.get("XMin")), _as_float(style.get("XMax"))
    if x_min is None:
        x_min = (
            edges[0]
            if not log_x or edges[0] > 0
            else next((e for e in edges if e > 0), None)
        )
    if x_max is None:
        x_max = edges[-1]
    if x_min is not None and x_max is not None and x_max > x_min:
        ax.set_xlim(x_min, x_max)
    y_min, y_max = _as_float(style.get("YMin")), _as_float(style.get("YMax"))
    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)
    ax.legend(fontsize=8, frameon=False)
    ax.tick_params(labelsize=8)
    ax.grid(False)

    if with_ratio:
        y_data = np.array(
            [np.nan if v is None else v for v in ref["values"]], dtype=float
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_down = np.array([d or 0.0 for d in ref["err_down"]]) / y_data
            rel_up = np.array([u or 0.0 for u in ref["err_up"]]) / y_data
        ax_ratio.fill_between(
            *_step_xy(ranges, list(1 - rel_down)),
            _step_xy(ranges, list(1 + rel_up))[1],
            color="#ffd54f",
            alpha=0.6,
            linewidth=0,
            zorder=1,
        )
        ax_ratio.axhline(1.0, color="black", linewidth=0.8, zorder=2)
        for index, (label, obj, envelope) in enumerate(predictions):
            colour = _COLOURS[index % len(_COLOURS)]
            if len(yoda_io.bin_ranges(obj)) != len(ranges):
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                values = np.array(
                    [np.nan if v is None else v for v in obj["values"]],
                    dtype=float,
                )
                r = values / y_data
                r_lo = (
                    values - np.array([d or 0.0 for d in obj["err_down"]])
                ) / y_data
                r_hi = (
                    values + np.array([u or 0.0 for u in obj["err_up"]])
                ) / y_data
            ax_ratio.plot(
                *_step_xy(ranges, list(r)),
                color=colour,
                linewidth=1.4,
                zorder=4,
            )
            ax_ratio.fill_between(
                *_step_xy(ranges, list(r_lo)),
                _step_xy(ranges, list(r_hi))[1],
                color=colour,
                alpha=0.18,
                linewidth=0,
                zorder=3,
            )
            if envelope is not None:
                with np.errstate(divide="ignore", invalid="ignore"):
                    e_lo = np.array(envelope[0]) / y_data
                    e_hi = np.array(envelope[1]) / y_data
                ax_ratio.fill_between(
                    *_step_xy(ranges, list(e_lo)),
                    _step_xy(ranges, list(e_hi))[1],
                    color=colour,
                    alpha=0.12,
                    linewidth=0,
                    hatch="////",
                    zorder=2,
                )
        r_min = _as_float(style.get("RatioPlotYMin"))
        r_max = _as_float(style.get("RatioPlotYMax"))
        ax_ratio.set_ylim(
            0.5 if r_min is None else r_min, 1.5 if r_max is None else r_max
        )
        ax_ratio.set_ylabel("MC / Data", fontsize=9)
        ax_ratio.tick_params(labelsize=8)
        if log_x:
            ax_ratio.set_xscale("log")

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    try:
        fig.savefig(output, dpi=dpi, bbox_inches="tight")
    except (ValueError, RuntimeError):
        # A label mathtext cannot render: fall back to plain text and try once
        # more.
        ax.set_title(plain_text(title), fontsize=10)
        ax.set_ylabel(plain_text(y_label), fontsize=10)
        (ax_ratio or ax).set_xlabel(plain_text(x_label), fontsize=10)
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_text(plain_text(text.get_text()))
        fig.savefig(output, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)


def write_index(
    output_dir: str,
    entries: List[Dict[str, Any]],
    labels: Sequence[str],
    rivet_version: str,
) -> str:
    """An index.html gallery of the plots, grouped by analysis."""
    by_analysis: Dict[str, List[Dict[str, Any]]] = {}
    for entry in entries:
        by_analysis.setdefault(entry["analysis"], []).append(entry)
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Rivet "
        "plots</title>",
        "<style>body{font-family:sans-serif;margin:1.5em;color:#222}h2{margin-top:2em;"
        "border-bottom:1px solid "
        "#ccc}.grid{display:flex;flex-wrap:wrap;gap:1em}"
        ".plot{width:320px;font-size:0.85em}.plot img{width:100%;border:1px "
        "solid #ddd}"
        ".ok{color:#2a7}.bad{color:#c33}.warn{color:#b80}</style></head><body>",
        f"<h1>Rivet plots</h1><p>Predictions: "
        f"{html.escape(', '.join(labels))}. "
        f"Rivet {html.escape(rivet_version)}.</p>",
    ]
    for analysis in sorted(by_analysis):
        parts.append(
            f"<h2 "
            f"id='{html.escape(analysis)}'>{html.escape(analysis)}</h2><div "
            f"class='grid'>"
        )
        for entry in by_analysis[analysis]:
            caption = html.escape(entry.get("title") or entry["histogram"])
            verdicts = []
            for label, comparison in zip(labels, entry.get("comparisons", [])):
                if comparison and comparison.get("comparable"):
                    css = (
                        "ok"
                        if comparison["p_value"] >= 0.05
                        else (
                            "warn" if comparison["p_value"] >= 1e-3 else "bad"
                        )
                    )
                    verdicts.append(
                        f"<span class='{css}'>{html.escape(label)}: "
                        f"&chi;&sup2;/ndf = "
                        f"{comparison['chi2']:.1f}/{comparison['ndf']}, "
                        f"p = {comparison['p_value']:.2g}</span>"
                    )
            relative = os.path.relpath(entry["image"], output_dir)
            parts.append(
                f"<div class='plot'><a href='{html.escape(relative)}'><img "
                f"src='{html.escape(relative)}' "
                f"alt='{caption}'></a><div>{caption}</div><div>{'<br>'.join(verdicts)}</div></div>"
            )
        parts.append("</div>")
    parts.append("</body></html>")
    index = os.path.join(output_dir, "index.html")
    with open(index, "w", encoding="utf-8") as handle:
        handle.write("\n".join(parts))
    return index


class RivetPlotTool(RivetToolBase):
    """
    Plot a Rivet run's histograms against the measured data, in the layout
    of Rivet's own plots: data points with error bars, each prediction as a
    stepped line with its statistical band, the weight-variation envelope
    when the run had variations, and a prediction/data ratio panel. Titles
    and axis labels come from the analysis's .plot file; the legend carries
    each prediction's chi2/ndf against the data.

    Several YODA files draw on the same axes, which is how to compare tunes,
    generators or settings. Histograms without reference data (the MC_*
    analyses) are drawn prediction-only.

    Inputs (runtime):
      - yoda_paths, labels: the runs to draw and their legend names
      - analyses, histograms: restrict to analyses or a path regex
      - output_dir: where the images go (default rivet/plots), one directory
        per analysis, with an index.html gallery and index.json
      - image_format: png (default), pdf or svg; dpi for png
      - ratio: draw the ratio panel (default True)
      - variation_band: draw the envelope of the weight variations
        (default True)
      - max_plots: stop after this many (default 200)

    Output (JSON): the directory, the gallery, the number of plots and a
    per-analysis list of images with each prediction's chi2/ndf and p-value.
    """

    yoda_paths: List[str] = runtime_list(
        "Rivet YODA outputs to draw (one or more, on the same axes)"
    )
    labels: List[str] = runtime_list(
        "A legend label per YODA file (default: file names)"
    )
    analyses: List[str] = runtime_list(
        "Only these analyses (default: all present)"
    )
    histograms: str = RuntimeField(
        default="", description="Regular expression selecting histogram paths"
    )
    reference_paths: List[str] = runtime_list(
        "Extra YODA files with /REF reference data"
    )
    output_dir: str = RuntimeField(
        default="rivet/plots",
        description="Directory for the images and the index",
    )
    image_format: str = RuntimeField(
        default="png", description="png, pdf or svg"
    )
    dpi: int = RuntimeField(
        default=150, description="Resolution of png output"
    )
    ratio: bool = RuntimeField(
        default=True, description="Draw the prediction/data ratio panel"
    )
    variation_band: bool = RuntimeField(
        default=True,
        description="Draw the envelope of the weight variations as a band",
    )
    max_plots: int = RuntimeField(
        default=200, description="At most this many histograms"
    )

    def _run(self) -> str:
        if not self.yoda_paths:
            return self.format_error(
                error="Missing Parameter", reason="yoda_paths is required"
            )
        files = []
        for rel in self.yoda_paths:
            full = self._safe_path(rel)
            if full is None or not os.path.isfile(full):
                return self.format_error(
                    error="File Not Found",
                    reason=f"YODA file {rel} not found in the sandbox",
                )
            files.append(full)
        labels = list(self.labels) or [
            os.path.splitext(os.path.basename(f))[0] for f in files
        ]
        if len(labels) != len(files):
            return self.format_error(
                error="Invalid Parameter",
                reason=f"{len(labels)} labels for {len(files)} files",
            )
        if len(set(labels)) != len(labels):
            labels = [self._rel(f) for f in files]
        extra_refs = []
        for rel in self.reference_paths:
            full = self._safe_path(rel)
            if full is None or not os.path.isfile(full):
                return self.format_error(
                    error="File Not Found",
                    reason=f"reference file {rel} not found",
                )
            extra_refs.append(full)
        output_dir = self._safe_path(self.output_dir)
        if output_dir is None:
            return self.format_error(
                error="Access Denied",
                reason="output_dir escapes base_directory",
            )
        if self.image_format not in IMAGE_FORMATS:
            return self.format_error(
                error="Invalid Parameter",
                reason=f"image_format must be one of "
                f"{', '.join(IMAGE_FORMATS)}",
            )
        try:
            selector = re.compile(self.histograms) if self.histograms else None
        except re.error as e:
            return self.format_error(
                error="Invalid Parameter",
                reason=f"histograms is not a regular expression: {e}",
            )
        try:
            import matplotlib  # noqa: F401, PLC0415 - a bundle dependency
        except ImportError:
            return self.format_error(
                error="Dependency Missing",
                reason="matplotlib is not installed",
                suggestion="Install the rivet bundle's dependencies",
            )

        try:
            install = self._install()
            load_catalogue(install.data_dir)
            dumps = yoda_io.dump_yoda(install, files)
            per_file = [
                yoda_io.objects_of(dumps, i) for i in range(len(files))
            ]
        except (RivetError, FileNotFoundError) as e:
            return self.format_error(
                error="Rivet Not Available", reason=str(e)
            )

        # Every nominal histogram present in any file, in order.
        wanted = {base_name(a) for a in self.analyses}
        paths: Dict[str, None] = {}
        for objects in per_file:
            for path, obj in yoda_io.final_histograms(objects).items():
                if not yoda_io.binned_1d(obj):
                    continue
                if selector is not None and not selector.search(path):
                    continue
                if wanted and yoda_io.analysis_name(path) not in wanted:
                    continue
                paths.setdefault(path, None)
        if not paths:
            return self.format_error(
                error="Nothing To Plot",
                reason="no one-dimensional histograms match in the YODA files",
                suggestion="Check `analyses` and `histograms`, or run "
                "RunRivetTool first",
            )

        # Reference data for every analysis involved.
        references: Dict[str, Dict[str, Any]] = {}
        ref_files = []
        for name in sorted({yoda_io.analysis_name(p) for p in paths}):
            for suffix in (".yoda.gz", ".yoda"):
                candidate = os.path.join(install.data_dir, name + suffix)
                if os.path.exists(candidate):
                    ref_files.append(candidate)
                    break
        ref_files += extra_refs
        if ref_files:
            try:
                dump = yoda_io.dump_yoda(install, ref_files)
                for index in range(len(ref_files)):
                    references.update(
                        {
                            p: o
                            for p, o in yoda_io.objects_of(dump, index).items()
                            if yoda_io.is_reference(p)
                        }
                    )
            except RivetError as e:
                return self.format_error(
                    error="Unreadable Reference Data", reason=str(e)
                )

        styles_cache: Dict[str, Any] = {}
        entries: List[Dict[str, Any]] = []
        skipped: List[str] = []
        limit = max(1, int(self.max_plots))
        for path in list(paths)[:limit]:
            analysis, histogram = yoda_io.split_path(path)
            name = base_name(analysis)
            if name not in styles_cache:
                styles_cache[name] = plot_styles(install.data_dir, name)
            style = plot_style(styles_cache[name], f"/{name}/{histogram}")
            ref = references.get(yoda_io.reference_path(path))
            predictions = []
            comparisons = []
            for label, objects in zip(labels, per_file):
                obj = objects.get(path)
                if obj is None or not yoda_io.binned_1d(obj):
                    continue
                envelope = None
                if self.variation_band:
                    variations = [
                        o
                        for p, o in objects.items()
                        if yoda_io.nominal_path(p) == path
                        and yoda_io.is_variation(p)
                    ]
                    envelope = _envelope(variations) if variations else None
                predictions.append((label, obj, envelope))
                comparisons.append(
                    compare_histograms(obj, ref) if ref is not None else None
                )
            if not predictions:
                continue
            safe = re.sub(r"[^\w.\-]+", "_", histogram)
            image = os.path.join(
                output_dir,
                re.sub(r"[^\w.\-]+", "_", analysis),
                f"{safe}.{self.image_format}",
            )
            try:
                draw_histogram(
                    path,
                    style,
                    ref,
                    predictions,
                    image,
                    ratio=self.ratio,
                    dpi=int(self.dpi),
                    comparisons=comparisons,
                )
            except (
                Exception
            ) as e:  # one bad histogram must not sink the gallery
                skipped.append(f"{path}: {type(e).__name__}: {e}")
                continue
            entries.append(
                {
                    "analysis": analysis,
                    "histogram": histogram,
                    "path": path,
                    "title": style.get("Title", ""),
                    "image": image,
                    "has_data": ref is not None,
                    "comparisons": [
                        (
                            {
                                "chi2": c["chi2"],
                                "ndf": c["ndf"],
                                "p_value": c["p_value"],
                                "comparable": True,
                            }
                            if c and c.get("comparable")
                            else None
                        )
                        for c in comparisons
                    ],
                }
            )

        os.makedirs(output_dir, exist_ok=True)
        index_html = write_index(output_dir, entries, labels, install.version)
        for entry in entries:
            entry["image"] = self._rel(entry["image"])
        with open(
            os.path.join(output_dir, "index.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump({"labels": labels, "plots": entries}, handle, indent=1)

        per_analysis: Dict[str, List[Dict[str, Any]]] = {}
        for entry in entries:
            per_analysis.setdefault(entry["analysis"], []).append(
                {
                    "histogram": entry["histogram"],
                    "image": entry["image"],
                    "title": entry["title"],
                    **(
                        {
                            "chi2_ndf": [
                                f"{c['chi2']:.1f}/{c['ndf']}" if c else None
                                for c in entry["comparisons"]
                            ],
                            "p_value": [
                                float(f"{c['p_value']:.3g}") if c else None
                                for c in entry["comparisons"]
                            ],
                        }
                        if entry["has_data"]
                        else {"data": "none"}
                    ),
                }
            )
        result: Dict[str, Any] = {
            "status": "ok",
            "output_dir": self._rel(output_dir),
            "index_html": self._rel(index_html),
            "index_json": self._rel(os.path.join(output_dir, "index.json")),
            "n_plots": len(entries),
            "labels": labels,
            "analyses": per_analysis,
        }
        if len(paths) > limit:
            result["note"] = (
                f"{len(paths) - limit} histograms not drawn; raise max_plots "
                f"or narrow `histograms`"
            )
        if skipped:
            result["skipped"] = skipped
        return json.dumps(result, ensure_ascii=False)
