"""
# run.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Run Rivet analyses on HepMC3 events.

The `rivet` program does the work; this tool does what it leaves to the
user. Before the run: that every analysis exists, that the events' beams and
sqrt(s) are what each analysis was measured at (Rivet removes an analysis
whose beam *particles* mismatch, but runs one at the wrong *energy* without a
word), that a cross section is at hand when an analysis needs one, and that
the analyses are ones whose results can be trusted. After the run: what
Rivet warned about, how many events reached the histograms, and which
histograms stayed empty.
"""

import collections
import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from orchestral.tools.base.field_utils import RuntimeField

from tools.rivet import yoda_io
from tools.rivet.base import RivetToolBase, runtime_list
from tools.rivet.catalogue import (
    Analysis,
    Catalogue,
    base_name,
    beam_name,
    caution,
    load_catalogue,
)
from tools.rivet.install import RivetError, run_program

# Rivet's own messages worth recognising in its log.
_REMOVED = re.compile(
    r"Analysis '([^']+)' is incompatible with the provided beams: removing"
)
_UNKNOWN = re.compile(r"'([^']+)' is not a known Rivet analysis")
_ALL_REMOVED = re.compile(
    r"All analyses were incompatible with the first event's beams"
)
_PROGRESS = re.compile(r"^Event (\d+)", re.MULTILINE)
_XSEC_LINE = re.compile(r"Cross-section = ([0-9.eE+-]+) pb")
_LOG_LINE = re.compile(r"^(Rivet\.[\w.]+): (WARN|ERROR)\s+(.*)$")

# Lines of the log that are banners and reminders, not findings.
_BOILERPLATE = (
    "MCnet usage guidelines",
    "Please acknowledge Rivet",
    "running on machine",
    "Reading events from",
    "Finished event loop",
    "Rivet run completed",
    "Histograms written to",
)


def inspect_hepmc(path: str) -> Optional[Dict[str, Any]]:
    """
    What the first event of a HepMC3 file says about the run: beams, sqrt(s),
    weights and whether a cross section is recorded. None without pyhepmc.
    """
    try:
        import pyhepmc  # noqa: PLC0415 - optional, a bundle dependency
    except ImportError:
        return None
    with pyhepmc.open(path) as handle:
        try:
            event = next(iter(handle))
        except StopIteration:
            raise ValueError(f"{path} holds no events")
    beams = list(event.beams)
    if len(beams) != 2:
        beams = [p for p in event.particles if p.status == 4][:2]
    out: Dict[str, Any] = {
        "n_weights": len(event.weights),
        "weight_names": list(event.weight_names),
    }
    if len(beams) == 2:
        pa, pb = (b.momentum for b in beams)
        s = (
            (pa.e + pb.e) ** 2
            - (pa.px + pb.px) ** 2
            - (pa.py + pb.py) ** 2
            - (pa.pz + pb.pz) ** 2
        )
        out["beam_ids"] = [beams[0].pid, beams[1].pid]
        out["beams"] = [beam_name(beams[0].pid), beam_name(beams[1].pid)]
        out["beam_energies_gev"] = [pa.e, pb.e]
        out["sqrt_s_gev"] = math.sqrt(max(s, 0.0))
    xsec = event.cross_section
    if xsec is not None:
        valid = getattr(xsec, "is_valid", True)
        if valid() if callable(valid) else valid:
            try:
                value = xsec.xsec(0)
            except (RuntimeError, IndexError):
                value = None
            if value and math.isfinite(value) and value > 0:
                out["cross_section_pb"] = value
    return out


def parse_rivet_log(text: str) -> Dict[str, Any]:
    """Rivet's findings in its log: removed or unknown analyses, warnings,
    progress."""
    warnings: "collections.Counter[str]" = collections.Counter()
    errors: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if (
            not stripped
            or stripped.startswith("Event ")
            or any(b in stripped for b in _BOILERPLATE)
        ):
            continue
        match = _LOG_LINE.match(stripped)
        if match:
            logger, level, message = match.groups()
            message = " ".join(message.split())
            if level == "ERROR":
                errors.append(f"{logger}: {message}")
            else:
                warnings[f"{logger}: {message}"] += 1
        elif "Error" in stripped or "error" in stripped.split(":")[0].lower():
            errors.append(stripped)
    progress = _PROGRESS.findall(text)
    xsec = _XSEC_LINE.findall(text)
    return {
        "removed": _REMOVED.findall(text),
        "unknown": _UNKNOWN.findall(text),
        "all_removed": bool(_ALL_REMOVED.search(text)),
        "warnings": [
            f"{msg} (x{n})" if n > 1 else msg
            for msg, n in warnings.most_common()
        ],
        "errors": list(dict.fromkeys(errors)),
        "last_event": int(progress[-1]) if progress else None,
        "cross_section_pb": float(xsec[-1]) if xsec else None,
    }


class AnalysisCheckError(Exception):
    """A requested analysis cannot sensibly be run on these events."""

    def __init__(
        self, error: str, reason: str, suggestion: str = "", context: str = ""
    ) -> None:
        super().__init__(reason)
        self.error = error
        self.reason = reason
        self.suggestion = suggestion
        self.context = context


def check_analyses(
    catalogue: Catalogue,
    specs: Sequence[str],
    custom_dirs: Sequence[str],
    run_info: Dict[str, Any],
    *,
    ignore_beams: bool = False,
    cross_section_pb: float = -1.0,
) -> Tuple[Dict[str, Analysis], List[str]]:
    """
    Check every requested analysis before spending a run on it.

    Parameters:
        catalogue: The installation's analyses
        specs: Analysis names, with options as Rivet writes them
        custom_dirs: Directories of user analyses, which the catalogue does
            not know and Rivet will judge
        run_info: What :func:`inspect_hepmc` found, or {} when unknown
        ignore_beams: Report beam and energy mismatches as warnings, not errors
        cross_section_pb: A cross section given by hand, if any

    Returns:
        (known analyses by name, warnings)

    Raises:
        AnalysisCheckError: for an unknown analysis, a beam or energy
            mismatch, or a missing cross section
    """
    warnings: List[str] = []
    known: Dict[str, Analysis] = {}
    unknown: List[Tuple[str, List[str]]] = []
    incompatible: List[str] = []
    need_xsec: List[str] = []
    for spec in specs:
        name = base_name(spec)
        analysis = catalogue.get(name)
        if analysis is None:
            if any(
                os.path.exists(os.path.join(d, name + ".info"))
                or any(
                    f.startswith("Rivet") and f.endswith((".so", ".dylib"))
                    for f in os.listdir(d)
                )
                for d in custom_dirs
            ):
                continue  # a user analysis: Rivet decides
            unknown.append((name, catalogue.suggest(name)))
            continue
        known[name] = analysis
        if run_info.get("beams"):
            ok, why = analysis.compatible(
                run_info["beams"], run_info.get("sqrt_s_gev")
            )
            if not ok:
                (warnings if ignore_beams else incompatible).append(
                    f"{name}: {why}"
                )
        if (
            analysis.needs_cross_section
            and cross_section_pb <= 0
            and run_info
            and "cross_section_pb" not in run_info
        ):
            need_xsec.append(name)
        option_names = [o.split("=", 1)[0] for o in analysis.options]
        for option in spec.split(":")[1:]:
            key = option.split("=", 1)[0]
            if option_names and key not in option_names:
                warnings.append(
                    f"{name}: option '{key}' is not one the analysis declares "
                    f"({', '.join(option_names)})"
                )
        for note in caution(analysis):
            warnings.append(f"{name}: {note}")
    if unknown:
        lines = [
            f"'{name}'"
            + (f" (did you mean {', '.join(close)}?)" if close else "")
            for name, close in unknown
        ]
        raise AnalysisCheckError(
            "Unknown Analysis",
            f"not in Rivet's catalogue: {'; '.join(lines)}",
            "RivetFindAnalysesTool searches the analyses by measurement, "
            "beams and energy",
        )
    if incompatible:
        raise AnalysisCheckError(
            "Incompatible Analyses",
            "; ".join(incompatible),
            "Generate at the analysis's beams and energy, choose an analysis "
            "for yours "
            "(RivetFindAnalysesTool with beams and energy_gev), or pass "
            "ignore_beams=True to "
            "run anyway, knowing the comparison to data is then meaningless",
            f"events: beams {' '.join(run_info.get('beams', []))} at "
            f"sqrt(s) = {run_info.get('sqrt_s_gev', float('nan')):.4g} GeV",
        )
    if need_xsec:
        raise AnalysisCheckError(
            "Cross Section Needed",
            f"{', '.join(need_xsec)} need the cross section and the events "
            f"carry none",
            "Pass cross_section_pb, or generate with PythiaFromRunCardTool, "
            "whose HepMC3 "
            "output records it",
        )
    return known, warnings


def output_warnings(summary: Dict[str, Any], log: Dict[str, Any]) -> List[str]:
    """What a reader of a Rivet run's output should know: removed analyses,
    Rivet's own warnings, empty histograms, thin statistics."""
    warnings = [
        f"Rivet removed {name}: its beams do not match the events"
        for name in log["removed"]
    ]
    warnings += log["warnings"]
    for name, entry in summary["analyses"].items():
        if entry["raw_entries"] == 0:
            warnings.append(
                f"{name}: no event entered any histogram. The events do not "
                f"contain what "
                "the analysis selects; compare the process with the "
                "analysis's run_info "
                "(RivetFindAnalysesTool with name=...)"
            )
        elif entry["empty"]:
            warnings.append(
                f"{name}: {len(entry['empty'])} of {entry['histograms']} "
                "histograms are empty"
            )
    if summary["n_events"] is not None and summary["n_events"] < 1000:
        warnings.append(
            f"only {int(summary['n_events'])} events were processed; "
            f"published "
            "distributions typically need 1e5 to 1e6 events for the "
            "statistical "
            "error to be below the data's"
        )
    return warnings


def summarise_output(objects: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Per-analysis contents of a Rivet YODA output."""
    per_analysis: Dict[str, Dict[str, Any]] = {}
    for path, obj in yoda_io.final_histograms(objects).items():
        analysis, histogram = yoda_io.split_path(path)
        entry = per_analysis.setdefault(
            analysis,
            {"histograms": 0, "filled": 0, "empty": [], "raw_entries": 0.0},
        )
        entry["histograms"] += 1
        raw = objects.get("/RAW" + path)
        filled = (
            yoda_io.has_content(raw)
            if raw is not None
            else yoda_io.has_content(obj)
        )
        if filled:
            entry["filled"] += 1
        else:
            entry["empty"].append(histogram)
        if raw is not None and raw.get("total"):
            entry["raw_entries"] += float(raw["total"].get("entries") or 0.0)
    for entry in per_analysis.values():
        entry["raw_entries"] = int(entry["raw_entries"])
        if len(entry["empty"]) > 10:
            entry["empty"] = entry["empty"][:10] + [
                f"... {len(entry['empty']) - 10} more"
            ]
    return {
        "n_events": yoda_io.event_count(objects),
        "cross_section_pb": yoda_io.cross_section_pb(objects),
        "weight_variations": yoda_io.variations_in(objects),
        "analyses": per_analysis,
    }


class RunRivetTool(RivetToolBase):
    """
    Run Rivet analyses on a HepMC3 event file and write their histograms to
    a YODA file.

    The events come from PythiaFromRunCardTool with output_format="hepmc3"
    and store="full" -- the complete record, so that analyses using the
    decay history (prompt leptons, b-hadrons, unstable particles) see it --
    or from ConvertEventsTool, or from any generator writing HepMC3. A
    gzipped file (.hepmc.gz) is read directly.

    Before running, the tool checks what Rivet does not: that each analysis
    exists (misspellings get suggestions), that the file's beams and sqrt(s)
    are what the analysis was measured at (an analysis at the wrong energy
    would otherwise run silently and fill nonsense), and that a cross section
    is available when an analysis needs one. Pass ignore_beams=True to run
    an analysis on beams it was not made for, knowingly.

    Inputs (runtime):
      - hepmc_path (or hepmc_paths for several files of the same run)
      - analyses: names, with options as Rivet writes them
        ("MC_ZINC:LMODE=MU"); RivetFindAnalysesTool lists them
      - output_path: the YODA file to write (default rivet/Rivet.yoda); the
        run's logs go next to it
      - cross_section_pb: overrides the cross section in the file, in pb
      - max_events, skip_events: process a slice of the file
      - skip_weights / nominal_weight: run the nominal weight only, or name
        which weight is nominal
      - analysis_path: sandbox directories holding your own compiled
        analyses (RivetANA.so with ANA.info/.plot/.yoda)
      - preload_paths: YODA files to preload, as `rivet -p` does
      - ignore_beams, timeout_s

    Output (JSON): the YODA path, the number of events processed and the
    cross section used, each analysis's histogram count and how many were
    filled, Rivet's warnings, cautions about the analyses, and the logs.
    Compare to data with RivetCompareTool and plot with RivetPlotTool.
    """

    hepmc_path: str = RuntimeField(
        default="",
        description="HepMC3 event file (.hepmc or .hepmc.gz), relative to the "
        "sandbox",
    )
    hepmc_paths: List[str] = runtime_list(
        "Several HepMC3 files of the same run, processed as one"
    )
    analyses: List[str] = runtime_list(
        "Rivet analyses to run, e.g. ['ATLAS_2012_I1204784', "
        "'MC_ZINC:LMODE=MU']"
    )
    output_path: str = RuntimeField(
        default="rivet/Rivet.yoda",
        description="YODA output file, relative to the sandbox",
    )
    cross_section_pb: float = RuntimeField(
        default=-1.0,
        description="Cross section in pb to use instead of the one in the "
        "file; -1 (default) uses the file's",
    )
    ignore_beams: bool = RuntimeField(
        default=False,
        description="Run analyses whose beams or energy do not match the "
        "events, knowingly",
    )
    max_events: int = RuntimeField(
        default=-1, description="Process at most this many events; -1 for all"
    )
    skip_events: int = RuntimeField(
        default=0, description="Skip this many events first"
    )
    skip_weights: bool = RuntimeField(
        default=False,
        description="Run on the nominal weight only, ignoring variations",
    )
    nominal_weight: str = RuntimeField(
        default="", description="Name of the weight to treat as nominal"
    )
    analysis_path: List[str] = runtime_list(
        "Sandbox directories holding your own compiled analyses and their "
        "data files"
    )
    preload_paths: List[str] = runtime_list("YODA files to preload (rivet -p)")
    timeout_s: int = RuntimeField(
        default=86400, description="Wall-clock limit for the run, in seconds"
    )

    def _run(self) -> str:
        inputs = [p for p in [self.hepmc_path, *self.hepmc_paths] if p]
        if not inputs:
            return self.format_error(
                error="Missing Parameter",
                reason="hepmc_path is required",
                suggestion="Name a HepMC3 file; PythiaFromRunCardTool writes "
                "one "
                "with output_format='hepmc3'",
            )
        if not self.analyses:
            return self.format_error(
                error="Missing Parameter",
                reason="analyses is required",
                suggestion="Name at least one analysis; RivetFindAnalysesTool "
                "finds the ones for your beams and energy",
            )
        event_files = []
        for rel in inputs:
            full = self._safe_path(rel)
            if full is None:
                return self.format_error(
                    error="Access Denied",
                    reason="path escapes base_directory",
                    context=f"hepmc_path={rel}",
                )
            if not os.path.isfile(full):
                return self.format_error(
                    error="File Not Found",
                    reason="HepMC3 file does not exist",
                    context=f"hepmc_path={rel}",
                )
            event_files.append(full)
        output = self._safe_path(self.output_path)
        if output is None:
            return self.format_error(
                error="Access Denied",
                reason="output_path escapes base_directory",
                context=f"output_path={self.output_path}",
            )
        custom_dirs = []
        for rel in self.analysis_path:
            full = self._safe_path(rel)
            if full is None or not os.path.isdir(full):
                return self.format_error(
                    error="Invalid Parameter",
                    reason=f"analysis_path {rel} is not a directory in the "
                    f"sandbox",
                )
            custom_dirs.append(full)
        preloads = []
        for rel in self.preload_paths:
            full = self._safe_path(rel)
            if full is None or not os.path.isfile(full):
                return self.format_error(
                    error="File Not Found",
                    reason=f"preload file {rel} not found",
                )
            preloads.append(full)

        try:
            install = self._install()
            catalogue = load_catalogue(install.data_dir)
        except (RivetError, FileNotFoundError) as e:
            return self.format_error(
                error="Rivet Not Available", reason=str(e)
            )

        # What the events are: beams, energy, weights, cross section.
        warnings: List[str] = []
        try:
            run_info = inspect_hepmc(event_files[0])
        except Exception as e:
            return self.format_error(
                error="Unreadable Input",
                reason=f"{type(e).__name__}: {e}",
                context=f"hepmc_path={inputs[0]}",
                suggestion="The file must be HepMC3 ASCII, plain or gzipped",
            )
        if run_info is None:
            warnings.append(
                "pyhepmc is not installed, so the beams and energy of the "
                "events "
                "were not checked against the analyses before running"
            )
            run_info = {}

        # Every analysis: known, compatible, and with what it needs.
        try:
            checked, notes = check_analyses(
                catalogue,
                self.analyses,
                custom_dirs,
                run_info,
                ignore_beams=self.ignore_beams,
                cross_section_pb=self.cross_section_pb,
            )
        except AnalysisCheckError as e:
            return self.format_error(
                error=e.error,
                reason=e.reason,
                context=e.context,
                suggestion=e.suggestion,
            )
        warnings += notes

        # The run itself.
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        if os.path.exists(output):
            os.remove(output)
        argv = [install.executable("rivet"), "-o", output]
        for spec in self.analyses:
            argv += ["-a", spec]
        if self.cross_section_pb > 0:
            argv += ["-x", repr(float(self.cross_section_pb))]
        if self.max_events > 0:
            argv += ["-n", str(int(self.max_events))]
        if self.skip_events > 0:
            argv += ["--nskip", str(int(self.skip_events))]
        if self.ignore_beams:
            argv.append("--ignore-beams")
        if self.skip_weights:
            argv.append("--skip-weights")
        if self.nominal_weight:
            argv += ["--nominal-weight", self.nominal_weight]
        for directory in custom_dirs:
            argv += ["--analysis-path-append", directory]
        for preload in preloads:
            argv += ["-p", preload]
        argv += event_files
        extra_env = {}
        if custom_dirs:
            extra_env["RIVET_DATA_PATH"] = os.pathsep.join(
                custom_dirs + [install.data_dir]
            )
        log_prefix = os.path.join(os.path.dirname(output), "rivet")
        try:
            run = run_program(
                install,
                argv,
                cwd=self.base_directory,
                log_prefix=log_prefix,
                timeout=self.timeout_s or None,
                extra_env=extra_env,
            )
        except RivetError as e:
            return self.format_error(error="Rivet Error", reason=str(e))

        log = parse_rivet_log(
            run.tail("stdout", 200000) + "\n" + run.tail("stderr", 200000)
        )
        logs = {
            "stdout": self._rel(run.stdout_log),
            "stderr": self._rel(run.stderr_log),
        }
        if run.timed_out:
            return self.format_error(
                error="Rivet Timed Out",
                reason=f"no result after {self.timeout_s} s",
                context=json.dumps(logs),
                suggestion="Raise timeout_s or process fewer events "
                "(max_events)",
            )
        if run.returncode != 0 or not os.path.exists(output):
            if log["unknown"]:
                reason = f"unknown analysis: {', '.join(log['unknown'])}"
            elif log["all_removed"]:
                reason = (
                    "every analysis was incompatible with the events' beams: "
                    + ", ".join(log["removed"])
                )
            elif log["errors"]:
                reason = log["errors"][0]
            else:
                reason = f"rivet exited with status {run.returncode}"
            return self.format_error(
                error="Rivet Run Failed",
                reason=reason,
                context=f"logs: {logs}\n{run.tail('stderr', 2000)}",
                suggestion="RivetFindAnalysesTool lists the analyses that fit "
                "the events' beams and energy",
            )

        try:
            objects = yoda_io.read_yoda(install, output)
        except RivetError as e:
            return self.format_error(
                error="Unreadable Output",
                reason=str(e),
                context=f"output={self._rel(output)}",
            )
        summary = summarise_output(objects)

        warnings += output_warnings(summary, log)

        result: Dict[str, Any] = {
            "status": "ok",
            "output_yoda": self._rel(output),
            "n_events": summary["n_events"],
            "cross_section_pb": (
                summary["cross_section_pb"][0]
                if summary["cross_section_pb"]
                else log["cross_section_pb"]
            ),
            "analyses": {
                name: {
                    "histograms": entry["histograms"],
                    "filled": entry["filled"],
                    "raw_entries": entry["raw_entries"],
                    **({"empty": entry["empty"]} if entry["empty"] else {}),
                    **(
                        {
                            "reference_data": checked[
                                base_name(name)
                            ].has_reference_data
                        }
                        if base_name(name) in checked
                        else {}
                    ),
                }
                for name, entry in summary["analyses"].items()
            },
            "weight_variations": summary["weight_variations"],
            "events": {
                k: run_info[k]
                for k in ("beams", "sqrt_s_gev", "n_weights")
                if k in run_info
            },
            "rivet": {
                **install.describe(),
                "elapsed_s": run.elapsed_s,
                "logs": logs,
            },
            "next": "RivetCompareTool compares the histograms with the "
            "measured data; "
            "RivetPlotTool draws them together",
        }
        if warnings:
            result["warnings"] = warnings
        return json.dumps(result, ensure_ascii=False)
