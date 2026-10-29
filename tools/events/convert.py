"""
# convert.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Converting an event dataset between formats: ConvertEventsTool.

One tool covers every pair of the formats tools.event_io reads and writes --
ROOT, HepMC3 and JSONL -- since reading and writing go through that layer and
never depend on the container. Its main use is HepMC3 for Delphes or Rivet
from a sample already generated in another format; generation can also write
HepMC3 directly.
"""

import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField
from pydantic import Field

from tools import event_io

# HepMC3 gives cross sections in pb; PythiaFromRunCardTool's manifest in mb.
_PB_PER_MB = 1.0e9

# Masses (GeV, PDG 2024) of the usual beam particles, for beams given by id
# and energy. Any other id is looked up in Pythia's particle data.
_BEAM_MASSES = {
    2212: 0.93827208816,
    2112: 0.93956542052,
    11: 0.00051099895,
    13: 0.1056583755,
    22: 0.0,
}


class ConvertEventsTool(BaseTool):
    """
    Convert an event dataset between ROOT, HepMC3 and JSONL.

    The formats are taken from the file suffixes: .root, .hepmc (or .hepmc3),
    and .jsonl. Every event is converted, streamed in chunks, so a dataset of
    any size can be.

    Writing HepMC3 needs what an event record does not hold: the run's two
    beams, and ideally the weight names and cross section. They are read from
    the manifest.json PythiaFromRunCardTool writes next to its dataset, or
    from `manifest_path`; without a manifest, pass `beam_ids` and
    `beam_energies_gev`. A complete record (store="full") is written as its
    vertex genealogy, as Pythia's own HepMC3 output would be; a final-state
    record as the final state between the two beams, where the final state is
    status 1 or a positive Pythia code (11 and above), so an LHE-derived
    record's intermediate resonances are left out. `hepmc_record` forces
    either.

    Reading HepMC3 keeps its final-state (status 1) particles, with the event
    weights.

    Inputs (runtime):
      - input_path: dataset to convert
      - output_path: dataset to write; its suffix selects the format
      - manifest_path: the run's manifest.json, when it is not next to the
        input
      - beam_ids / beam_energies_gev: the two beams (A along +z, B along -z),
        when there is no manifest

    Output (JSON):
      {"status": "ok", "output": "<relative path>", "format": "hepmc3",
       "n_events": <int>, "beams_from": "manifest" | "arguments" | null}
    """

    input_path: str = RuntimeField(
        description="Event dataset to convert (.root, .hepmc or .jsonl)"
    )
    output_path: str = RuntimeField(
        description="Dataset to write; its suffix (.root, .hepmc or .jsonl) "
        "selects the format"
    )
    hepmc_record: str = RuntimeField(
        default="auto",
        description="For HepMC3 output: 'auto' (default) writes the full "
        "event genealogy when the input is a complete record (store='full') "
        "and the final state otherwise; 'final-state' forces the final state; "
        "'full' refuses anything but a complete record",
    )
    manifest_path: str = RuntimeField(
        default="",
        description="The run's manifest.json, when it is not next to the "
        "input; supplies beams, weight names and cross section for HepMC3",
    )
    beam_ids: List[int] = Field(
        default_factory=list,
        json_schema_extra={"runtime": True},
        description="PDG ids of the two beams, for HepMC3 output without a "
        "manifest",
    )
    beam_energies_gev: List[float] = Field(
        default_factory=list,
        json_schema_extra={"runtime": True},
        description="Energies (GeV) of the two beams, for HepMC3 output "
        "without a manifest",
    )

    base_directory: str = StateField(
        default=".", description="Base directory for safe path resolution"
    )

    def _setup(self) -> None:
        self.base_directory = os.path.abspath(self.base_directory)
        if not os.path.isdir(self.base_directory):
            raise ValueError(
                f"Base directory does not exist: {self.base_directory}"
            )

    def _safe_path(self, rel: str) -> Optional[str]:
        full = os.path.abspath(os.path.join(self.base_directory, rel))
        if full != self.base_directory and not full.startswith(
            self.base_directory + os.sep
        ):
            return None
        return full

    def _run(self) -> str:
        try:
            self._setup()
        except Exception as e:
            return self.format_error(error="Path Error", reason=str(e))

        src = self._safe_path(self.input_path)
        dst = self._safe_path(self.output_path)
        if src is None or dst is None:
            return self.format_error(
                error="Access Denied",
                reason="input_path or output_path escapes base_directory",
                context=f"input_path={self.input_path}, "
                f"output_path={self.output_path}",
            )
        if not os.path.exists(src):
            return self.format_error(
                error="File Not Found", reason=f"{self.input_path} not found"
            )
        try:
            fmt_in = event_io.detect_format(src)
            fmt_out = event_io.detect_format(dst)
        except ValueError as e:
            return self.format_error(
                error="Invalid Parameter",
                reason=str(e),
                suggestion="Use a .root, .hepmc or .jsonl suffix",
            )
        if os.path.abspath(src) == os.path.abspath(dst):
            return self.format_error(
                error="Invalid Parameter",
                reason="input_path and output_path are the same file",
            )

        beams: Optional[List[Dict[str, float]]] = None
        beams_from: Optional[str] = None
        manifest: Dict[str, Any] = {}
        if fmt_out == "hepmc3":
            try:
                manifest = self._manifest(src)
                beams, beams_from = self._beams(manifest)
            except ValueError as e:
                return self.format_error(
                    error="Missing Beams",
                    reason=str(e),
                    suggestion="Keep the run's manifest.json next to the "
                    "input, pass "
                    "manifest_path, or pass beam_ids and beam_energies_gev",
                )

        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            n_events = 0
            # The writer's store spec only guards what it may hold; conversion
            # writes the records as read. A complete record becomes the HepMC3
            # genealogy, a final-state record the final state between the
            # beams.
            with event_io.EventWriter(
                dst, store="full", beams=beams, hepmc_record=self.hepmc_record
            ) as writer:
                if fmt_out == "hepmc3":
                    outputs = manifest.get("outputs", {})
                    names = outputs.get("weights", {}).get("names")
                    if names:
                        writer.set_weight_names(names)
                    xsec = outputs.get("xsec", {})
                    if "sigmaGen_mb" in xsec:
                        writer.set_cross_section(
                            xsec["sigmaGen_mb"] * _PB_PER_MB,
                            xsec.get("sigmaErr_mb", 0.0) * _PB_PER_MB,
                        )
                for chunk in event_io.iter_events(src, step_size=1000):
                    n_events += writer.write_reduced(chunk)
        except Exception as e:
            return self.format_error(
                error="Conversion Error",
                reason=f"{type(e).__name__}: {e}",
                context=f"{self.input_path} ({fmt_in}) -> {self.output_path}",
            )

        return json.dumps(
            {
                "status": "ok",
                "output": os.path.relpath(dst, self.base_directory),
                "format": fmt_out,
                "n_events": n_events,
                "beams_from": beams_from,
            },
            separators=(",", ":"),
        )

    def _manifest(self, src: str) -> Dict[str, Any]:
        """The run's manifest: the one given, else the one beside the input,
        else {}."""
        if self.manifest_path:
            path = self._safe_path(self.manifest_path)
            if path is None or not os.path.exists(path):
                raise ValueError(
                    f"manifest_path {self.manifest_path} not found in the "
                    f"sandbox"
                )
        else:
            path = os.path.join(os.path.dirname(src), "manifest.json")
            if not os.path.exists(path):
                return {}
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _beams(
        self, manifest: Dict[str, Any]
    ) -> Tuple[List[Dict[str, float]], str]:
        """The two beams, from the arguments if given, else from the
        manifest."""
        if self.beam_ids or self.beam_energies_gev:
            if len(self.beam_ids) != 2 or len(self.beam_energies_gev) != 2:
                raise ValueError(
                    "beam_ids and beam_energies_gev must each give two values"
                )
            beams = []
            for pid, energy, sign in zip(
                self.beam_ids, self.beam_energies_gev, (1.0, -1.0)
            ):
                mass = _beam_mass(int(pid))
                pz = sign * math.sqrt(max(float(energy) ** 2 - mass**2, 0.0))
                beams.append(
                    {
                        "id": int(pid),
                        "px": 0.0,
                        "py": 0.0,
                        "pz": pz,
                        "e": float(energy),
                        "m": mass,
                    }
                )
            return beams, "arguments"
        if manifest.get("beams"):
            return manifest["beams"], "manifest"
        raise ValueError(
            "HepMC3 output needs the run's beams, and no manifest records them"
        )


def _beam_mass(pid: int) -> float:
    """A beam particle's mass in GeV, from the table or Pythia's particle
    data."""
    if abs(pid) in _BEAM_MASSES:
        return _BEAM_MASSES[abs(pid)]
    try:
        from tools.pythia.runtime import (
            STDERR,
            STDOUT,
            FdSilence,
            require_pythia,
        )

        with FdSilence(STDOUT, STDERR):
            pythia = require_pythia().Pythia("", printBanner=False)
        if pythia.particleData.isParticle(pid):
            return float(pythia.particleData.m0(pid))
    except ImportError:
        pass
    raise ValueError(
        f"no mass known for beam particle {pid}; pass a manifest instead"
    )
