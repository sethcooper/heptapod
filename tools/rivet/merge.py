"""
# merge.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Combine the YODA outputs of several Rivet runs into one."""

import json
import os
from typing import Any, Dict, List

from orchestral.tools.base.field_utils import RuntimeField

from tools.rivet import yoda_io
from tools.rivet.base import RivetToolBase, runtime_list
from tools.rivet.install import RivetError, run_program
from tools.rivet.run import parse_rivet_log, summarise_output


class RivetMergeTool(RivetToolBase):
    """
    Merge the YODA files of several Rivet runs into one, with `rivet-merge`.

    Large samples are generated as independent runs -- different seeds, or
    different machines -- and analysed separately; this combines their
    histograms as if one run had seen every event, re-running each
    analysis's finalize step so normalised distributions come out right.
    Plain histogram addition (`yodamerge`) does not do that.

    Two situations:
      - equivalent=True (default): the runs are the same process with
        different seeds. Histograms are combined weighted by statistics.
      - equivalent=False: the runs are different processes (e.g. the W and
        the Z contribution), each with its own cross section, to be added.
        Give cross_sections_pb to override what the files carry.

    Inputs (runtime):
      - yoda_paths: two or more Rivet outputs
      - output_path: the merged YODA file (default rivet/merged.yoda)
      - equivalent: see above
      - cross_sections_pb: one cross section per file, when they must be set
      - histograms: only merge histograms whose path matches this regex

    Output (JSON): the merged file, its event count, cross section and
    per-analysis contents.
    """

    yoda_paths: List[str] = runtime_list(
        "Rivet YODA outputs to merge (two or more)"
    )
    output_path: str = RuntimeField(
        default="rivet/merged.yoda", description="The merged YODA file"
    )
    equivalent: bool = RuntimeField(
        default=True,
        description="The runs are the same process with different seeds "
        "(default); False adds different processes",
    )
    cross_sections_pb: List[float] = runtime_list(
        "A cross section in pb per file, overriding the files' own"
    )
    histograms: str = RuntimeField(
        default="",
        description="Only merge histograms whose path matches this regular "
        "expression",
    )
    timeout_s: int = RuntimeField(
        default=3600, description="Wall-clock limit, in seconds"
    )

    def _run(self) -> str:
        if len(self.yoda_paths) < 2:
            return self.format_error(
                error="Missing Parameter",
                reason="yoda_paths needs at least two files",
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
        if self.cross_sections_pb and len(self.cross_sections_pb) != len(
            files
        ):
            return self.format_error(
                error="Invalid Parameter",
                reason=f"{len(self.cross_sections_pb)} cross sections for "
                f"{len(files)} files",
            )
        output = self._safe_path(self.output_path)
        if output is None:
            return self.format_error(
                error="Access Denied",
                reason="output_path escapes base_directory",
            )
        try:
            install = self._install()
            merger = install.executable("rivet-merge")
        except RivetError as e:
            return self.format_error(
                error="Rivet Not Available", reason=str(e)
            )

        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        argv = [merger, "-o", output]
        if self.equivalent:
            argv.append("-e")
        if self.histograms:
            argv += ["-m", self.histograms]
        for index, path in enumerate(files):
            spec = path
            if self.cross_sections_pb:
                spec += f":={float(self.cross_sections_pb[index])!r}"
            argv.append(spec)
        try:
            run = run_program(
                install,
                argv,
                cwd=self.base_directory,
                log_prefix=os.path.splitext(output)[0] + ".merge",
                timeout=self.timeout_s or None,
            )
        except RivetError as e:
            return self.format_error(error="Rivet Error", reason=str(e))
        logs = {
            "stdout": self._rel(run.stdout_log),
            "stderr": self._rel(run.stderr_log),
        }
        if run.timed_out or run.returncode != 0 or not os.path.exists(output):
            log = parse_rivet_log(
                run.tail("stdout", 100000) + "\n" + run.tail("stderr", 100000)
            )
            return self.format_error(
                error="Merge Failed",
                reason=(
                    log["errors"][0]
                    if log["errors"]
                    else f"rivet-merge exited with status {run.returncode}"
                ),
                context=f"logs: {logs}\n{run.tail('stderr', 2000)}",
            )
        try:
            objects = yoda_io.read_yoda(install, output)
        except RivetError as e:
            return self.format_error(error="Unreadable Output", reason=str(e))
        summary = summarise_output(objects)
        result: Dict[str, Any] = {
            "status": "ok",
            "output_yoda": self._rel(output),
            "n_inputs": len(files),
            "equivalent": bool(self.equivalent),
            "n_events": summary["n_events"],
            "cross_section_pb": summary["cross_section_pb"][0]
            if summary["cross_section_pb"]
            else None,
            "analyses": {
                name: {"histograms": e["histograms"], "filled": e["filled"]}
                for name, e in summary["analyses"].items()
            },
            "logs": logs,
        }
        log = parse_rivet_log(
            run.tail("stdout", 100000) + "\n" + run.tail("stderr", 100000)
        )
        if log["warnings"]:
            result["warnings"] = log["warnings"]
        return json.dumps(result, ensure_ascii=False)
