"""
# compare.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Compare Rivet output with the measurements it reproduces.

Rivet writes the generator's prediction binned exactly as the publication,
and ships the published numbers as `/REF/...` objects. This module puts the
two side by side, bin by bin, and summarises the agreement as a chi-square
with its p-value, the pulls, and the ratio. The numbers are what a physicist
would compute by hand from the two histograms; nothing is fitted.
"""

import json
import math
import os
import re
from typing import Any, Dict, List, Sequence

from orchestral.tools.base.field_utils import RuntimeField

from tools.rivet import yoda_io
from tools.rivet.base import RivetToolBase, runtime_list
from tools.rivet.catalogue import (
    base_name,
    load_catalogue,
    plot_style,
    plot_styles,
)
from tools.rivet.install import RivetError, RivetInstall


def chi2_survival(chi2: float, ndf: int) -> float:
    """
    P(X >= chi2) for a chi-square variable with ndf degrees of freedom: the
    regularised upper incomplete gamma function Q(ndf/2, chi2/2), by series
    for small arguments and continued fraction otherwise (Numerical Recipes
    6.2), accurate to about 1e-12.
    """
    if ndf <= 0 or chi2 < 0 or not math.isfinite(chi2):
        return float("nan")
    if chi2 == 0:
        return 1.0
    a, x = 0.5 * ndf, 0.5 * chi2
    log_gamma_a = math.lgamma(a)
    if x < a + 1.0:
        # Series for P(a, x); Q = 1 - P.
        term = total = 1.0 / a
        denominator = a
        for _ in range(10000):
            denominator += 1.0
            term *= x / denominator
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        p = total * math.exp(-x + a * math.log(x) - log_gamma_a)
        return max(0.0, min(1.0, 1.0 - p))
    # Lentz's continued fraction for Q(a, x).
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    q = math.exp(-x + a * math.log(x) - log_gamma_a) * h
    return max(0.0, min(1.0, q))


def _same_binning(a: Sequence[float], b: Sequence[float]) -> bool:
    if len(a) != len(b):
        return False
    return all(
        abs(x - y) <= 1e-6 * max(1.0, abs(x), abs(y)) for x, y in zip(a, b)
    )


def compare_histograms(
    mc: Dict[str, Any], ref: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Bin-by-bin comparison of a prediction with a measurement.

    Both are dumped 1D estimates on the same binning. Bins where either side
    is undefined (masked, NaN) or neither has an uncertainty are left out.
    The uncertainty used per bin is the one on the side facing the other
    value: the data's upper error when the prediction lies above the data,
    its lower error below, and the reverse for the prediction's own error.

    A bin where the prediction is empty -- no generated event reached it, so
    its value and uncertainty are both zero -- carries no information about
    the model, only about the sample size; such bins are left out of the
    chi-square and counted as `mc_empty_bins`, because treating a zero with
    no uncertainty as a prediction would give it an infinite pull.

    Returns:
        A dict with the chi-square, its degrees of freedom and p-value, the
        same after scaling the prediction to the data's integral (the shape
        comparison), the pulls, the ratio, and the per-bin table.
    """
    if not yoda_io.binned_1d(mc) or not yoda_io.binned_1d(ref):
        return {
            "comparable": False,
            "reason": f"not one-dimensional estimates ({mc.get('type')}, "
            f"{ref.get('type')})",
        }
    mc_bins = yoda_io.bin_ranges(mc)
    ref_bins = yoda_io.bin_ranges(ref)
    if not _same_binning(
        [b for pair in mc_bins for b in pair],
        [b for pair in ref_bins for b in pair],
    ):
        return {
            "comparable": False,
            "reason": f"binning differs ({len(mc_bins)} vs "
            f"{len(ref_bins)} bins)",
        }

    masked = set(mc.get("masked", [])) | set(ref.get("masked", []))
    rows = []
    for i, (lo, hi) in enumerate(ref_bins):
        if i in masked:
            continue
        y_d, y_m = ref["values"][i], mc["values"][i]
        if y_d is None or y_m is None:
            continue
        above = y_m > y_d
        sigma_d = (ref["err_up"][i] if above else ref["err_down"][i]) or 0.0
        sigma_m = (mc["err_down"][i] if above else mc["err_up"][i]) or 0.0
        sigma = math.hypot(sigma_d, sigma_m)
        rows.append(
            {
                "low": lo,
                "high": hi,
                "data": y_d,
                "data_err": sigma_d,
                "mc": y_m,
                "mc_err": sigma_m,
                "sigma": sigma,
            }
        )
    empty = [
        r for r in rows if r["mc"] == 0 and r["mc_err"] == 0 and r["data"] != 0
    ]
    for r in empty:
        r["mc_empty"] = True
    used = [r for r in rows if r["sigma"] > 0 and not r.get("mc_empty")]
    if not used:
        return {
            "comparable": False,
            "reason": (
                "the prediction is empty"
                if empty
                else "no bins with both a value and an uncertainty"
            ),
            "mc_empty_bins": len(empty),
        }

    for r in used:
        r["pull"] = (r["mc"] - r["data"]) / r["sigma"]
        r["ratio"] = r["mc"] / r["data"] if r["data"] else None
    chi2 = sum(r["pull"] ** 2 for r in used)
    ndf = len(used)
    pulls = [r["pull"] for r in used]
    result: Dict[str, Any] = {
        "comparable": True,
        "n_bins": ndf,
        "chi2": chi2,
        "ndf": ndf,
        "chi2_ndf": chi2 / ndf,
        "p_value": chi2_survival(chi2, ndf),
        "max_abs_pull": max(abs(p) for p in pulls),
        "n_bins_beyond_3sigma": sum(1 for p in pulls if abs(p) > 3),
        "mean_pull": sum(pulls) / ndf,
        "mc_empty_bins": len(empty),
        "mc_empty": False,
    }
    ratios = [r["ratio"] for r in used if r["ratio"] is not None]
    if ratios:
        result["mean_ratio_mc_over_data"] = sum(ratios) / len(ratios)

    # Shape: scale the prediction so its integral over the compared bins equals
    # the data's.
    data_integral = sum(r["data"] * (r["high"] - r["low"]) for r in used)
    mc_integral = sum(r["mc"] * (r["high"] - r["low"]) for r in used)
    if mc_integral and data_integral and ndf > 1:
        scale = data_integral / mc_integral
        shape_pulls = []
        for r in used:
            sigma = math.hypot(r["data_err"], r["mc_err"] * scale)
            if sigma > 0:
                shape_pulls.append((r["mc"] * scale - r["data"]) / sigma)
        if shape_pulls:
            chi2_shape = sum(p**2 for p in shape_pulls)
            result.update(
                {
                    "normalisation_mc_over_data": 1.0 / scale,
                    "chi2_shape": chi2_shape,
                    "ndf_shape": len(shape_pulls) - 1,
                    "p_value_shape": chi2_survival(
                        chi2_shape, len(shape_pulls) - 1
                    ),
                }
            )
    result["bins"] = rows
    return result


def verdict(comparison: Dict[str, Any]) -> str:
    """One word on the agreement, for a reader skimming many histograms."""
    if not comparison.get("comparable"):
        return "not compared"
    if comparison.get("mc_empty"):
        return "empty prediction"
    p = comparison["p_value"]
    if comparison.get("mc_empty_bins"):
        return (
            ("consistent where predicted; " if p >= 0.05 else "")
            + f"{comparison['mc_empty_bins']} empty prediction bins "
            "(more events needed)"
        )
    if p >= 0.05:
        return "consistent"
    if comparison.get("p_value_shape", 0.0) >= 0.05:
        return "shape consistent, normalisation off"
    if p >= 1e-3:
        return "tension"
    return "disagreement"


class RivetCompareTool(RivetToolBase):
    """
    Compare a Rivet run's histograms with the published measurements, bin by
    bin, and summarise the agreement.

    For every histogram that has reference data (`/REF/...` in the analysis's
    data file, or in `reference_paths`), the tool reports the chi-square
    against the data with its p-value, the same for the shape alone (the
    prediction scaled to the data's integral, which separates a
    normalisation offset from a wrong shape), the largest pull, and the
    mean prediction/data ratio. The full per-bin tables go to `output_path`
    as JSON; the tool result lists each analysis's summary and its worst
    histograms.

    The statistics are plain: data and prediction uncertainties added in
    quadrature per bin, no correlations (the publications' correlation
    matrices are not in the YODA files), so a p-value is a guide to where
    to look, not a measurement of the model. With few generated events the
    prediction's own uncertainty dominates and everything looks consistent;
    the run tool warns below 1000 events.

    Inputs (runtime):
      - yoda_paths: one or more Rivet outputs (e.g. from different tunes or
        generators), with optional `labels`
      - analyses: restrict to these analyses (default: every one in the files)
      - histograms: a regular expression on histogram paths
      - reference_paths: extra YODA files holding /REF objects, for analyses
        whose data is not in the installation
      - output_path: the JSON report (default rivet/comparison.json)
      - max_listed: worst histograms listed per analysis in the result

    Output (JSON): per file and analysis, the number of histograms compared,
    how many are consistent at p > 0.05, the mean chi2/ndf, and the worst
    histograms with their verdicts; plus histograms without reference data.
    """

    yoda_paths: List[str] = runtime_list(
        "Rivet YODA outputs to compare with data (one or more)"
    )
    labels: List[str] = runtime_list(
        "A label per YODA file, for the report (default: file names)"
    )
    analyses: List[str] = runtime_list(
        "Only these analyses (default: all present)"
    )
    histograms: str = RuntimeField(
        default="",
        description="Regular expression selecting histogram paths, e.g. "
        "'d0[1-3]'",
    )
    reference_paths: List[str] = runtime_list(
        "Extra YODA files with /REF reference data"
    )
    output_path: str = RuntimeField(
        default="rivet/comparison.json",
        description="Where the full per-bin report is written",
    )
    max_listed: int = RuntimeField(
        default=5,
        description="Worst histograms listed per analysis in the result",
    )

    def _run(self) -> str:
        if not self.yoda_paths:
            return self.format_error(
                error="Missing Parameter",
                reason="yoda_paths is required",
                suggestion="Name the YODA file RunRivetTool wrote",
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
        labels = list(self.labels) if self.labels else []
        if labels and len(labels) != len(files):
            return self.format_error(
                error="Invalid Parameter",
                reason=f"{len(labels)} labels for {len(files)} files",
            )
        if not labels:
            labels = [os.path.splitext(os.path.basename(f))[0] for f in files]
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
        output = self._safe_path(self.output_path)
        if output is None:
            return self.format_error(
                error="Access Denied",
                reason="output_path escapes base_directory",
            )
        try:
            selector = re.compile(self.histograms) if self.histograms else None
        except re.error as e:
            return self.format_error(
                error="Invalid Parameter",
                reason=f"histograms is not a regular expression: {e}",
            )

        try:
            install = self._install()
            catalogue = load_catalogue(install.data_dir)
            dumps = yoda_io.dump_yoda(install, files)
        except (RivetError, FileNotFoundError) as e:
            return self.format_error(
                error="Rivet Not Available", reason=str(e)
            )

        wanted = {base_name(a) for a in self.analyses}
        report: Dict[str, Any] = {
            "files": [],
            "rivet_version": install.version,
        }
        result_files = []
        references: Dict[str, Dict[str, Any]] = {}
        try:
            for index, (path, label) in enumerate(zip(files, labels)):
                objects = yoda_io.objects_of(dumps, index)
                histograms = yoda_io.final_histograms(objects)
                if selector is not None:
                    histograms = {
                        p: o
                        for p, o in histograms.items()
                        if selector.search(p)
                    }
                if wanted:
                    histograms = {
                        p: o
                        for p, o in histograms.items()
                        if yoda_io.analysis_name(p) in wanted
                    }
                needed = {yoda_io.analysis_name(p) for p in histograms}
                self._load_references(
                    install, catalogue, needed, extra_refs, references
                )
                file_report = self._compare_file(
                    label,
                    self._rel(path),
                    histograms,
                    references,
                    install.data_dir,
                )
                report["files"].append(file_report)
                result_files.append(self._summarise(file_report))
        except RivetError as e:
            return self.format_error(error="Unreadable YODA", reason=str(e))

        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        with open(output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1)
        return json.dumps(
            {
                "status": "ok",
                "report": self._rel(output),
                "files": result_files,
                "method": "chi2 over bins with data and MC uncertainties in "
                "quadrature, no "
                "correlations; 'shape' scales the prediction to the data's "
                "integral first",
                "next": "RivetPlotTool draws prediction and data together, "
                "with the ratio",
            },
            ensure_ascii=False,
        )

    def _load_references(
        self,
        install: RivetInstall,
        catalogue: Any,
        analyses: set,
        extra_refs: List[str],
        references: Dict[str, Dict[str, Any]],
    ) -> None:
        """Add the /REF objects of these analyses to `references`."""
        paths = []
        for name in sorted(analyses):
            if name in references.get("__loaded__", {}):
                continue
            for suffix in (".yoda.gz", ".yoda"):
                candidate = os.path.join(install.data_dir, name + suffix)
                if os.path.exists(candidate):
                    paths.append(candidate)
                    break
        paths += [
            p
            for p in extra_refs
            if p not in references.get("__files__", set())
        ]
        loaded = references.setdefault("__loaded__", {})
        for name in analyses:
            loaded[name] = True
        references.setdefault("__files__", set()).update(extra_refs)
        if not paths:
            return
        dump = yoda_io.dump_yoda(install, paths)
        for index in range(len(paths)):
            for path, obj in yoda_io.objects_of(dump, index).items():
                if yoda_io.is_reference(path):
                    references[path] = obj

    def _compare_file(
        self,
        label: str,
        path: str,
        histograms: Dict[str, Dict[str, Any]],
        references: Dict[str, Dict[str, Any]],
        data_dir: str,
    ) -> Dict[str, Any]:
        per_analysis: Dict[str, Dict[str, Any]] = {}
        styles: Dict[str, Any] = {}
        for hist_path, obj in sorted(histograms.items()):
            analysis, histogram = yoda_io.split_path(hist_path)
            entry = per_analysis.setdefault(
                analysis, {"histograms": [], "without_reference": []}
            )
            ref = references.get(yoda_io.reference_path(hist_path))
            if ref is None:
                entry["without_reference"].append(hist_path)
                continue
            comparison = compare_histograms(obj, ref)
            comparison["path"] = hist_path
            name = base_name(analysis)
            if name not in styles:
                styles[name] = plot_styles(data_dir, name)
            # Rivet writes "~" where a histogram has no title of its own.
            title = plot_style(styles[name], f"/{name}/{histogram}").get(
                "Title", ""
            )
            comparison["title"] = title or (obj.get("title") or "").replace(
                "~", ""
            )
            comparison["verdict"] = verdict(comparison)
            entry["histograms"].append(comparison)
        return {"label": label, "path": path, "analyses": per_analysis}

    def _summarise(self, file_report: Dict[str, Any]) -> Dict[str, Any]:
        """The compact view of one file's comparison, for the tool result."""
        summary: Dict[str, Any] = {
            "label": file_report["label"],
            "analyses": {},
        }
        for analysis, entry in file_report["analyses"].items():
            compared = [h for h in entry["histograms"] if h.get("comparable")]
            consistent = [h for h in compared if h["p_value"] >= 0.05]
            ordered = sorted(compared, key=lambda h: h["p_value"])
            record: Dict[str, Any] = {
                "histograms_compared": len(compared),
                "consistent_at_5pct": len(consistent),
                "mean_chi2_ndf": (
                    sum(h["chi2_ndf"] for h in compared) / len(compared)
                )
                if compared
                else None,
                "worst": [
                    {
                        "path": h["path"],
                        "title": h["title"],
                        "chi2_ndf": round(h["chi2_ndf"], 3),
                        "p_value": float(f"{h['p_value']:.3g}"),
                        "max_abs_pull": round(h["max_abs_pull"], 2),
                        "mean_ratio_mc_over_data": round(
                            h.get("mean_ratio_mc_over_data", float("nan")), 3
                        ),
                        **(
                            {"mc_empty_bins": h["mc_empty_bins"]}
                            if h.get("mc_empty_bins")
                            else {}
                        ),
                        "verdict": h["verdict"],
                    }
                    for h in ordered[: max(1, int(self.max_listed))]
                ],
            }
            not_compared = [
                h for h in entry["histograms"] if not h.get("comparable")
            ]
            if not_compared:
                record["not_compared"] = [
                    {"path": h["path"], "reason": h["reason"]}
                    for h in not_compared[:10]
                ]
            if entry["without_reference"]:
                record["without_reference_data"] = len(
                    entry["without_reference"]
                )
            summary["analyses"][analysis] = record
        return summary
