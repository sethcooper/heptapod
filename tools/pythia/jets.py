"""
# jets.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Jet clustering with Pythia 8's SlowJet: JetClusterSlowJetTool.
"""

import json
import math
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField
from pydantic import Field
from tqdm import tqdm

from tools import event_io

from .runtime import STDERR, STDOUT, TQDM_CONFIG, FdSilence, require_pythia

_ALGO_TO_POWER = {
    "kt": 1,  # kT
    "ca": 0,  # Cambridge/Aachen
    "antikt": -1,  # anti-kT
}


class JetClusterSlowJetTool(BaseTool):
    """
    Cluster jets with Pythia8.SlowJet for a single event, or a whole dataset.

    Input (choose ONE):
      - events_path  (any dataset PythiaFromRunCardTool writes, .root or
                      .jsonl, read through tools.event_io)
      - npy_path     (zero-padded array of shape [N_ev, N_max, 5] holding
                      [px, py, pz, E, pid])
    with either event_index for one event, or cluster_all=True and an
    output_path for every event.

    Neither mode holds the dataset in memory: one event is read on its own,
    and cluster_all streams the dataset in chunks, writing each event's jets
    as it goes, so a dataset of any size can be clustered.

    Params:
      - algorithm: 'antikt' | 'kt' | 'ca'
      - R: radius parameter (e.g. 0.4)
      - ptmin: minimum jet pT (GeV)
      - etamax: |eta| max for constituents
      - mass_option: 1 = E-scheme (typical), see Pythia8 doc
      - include_invisible: False (default) clusters visible particles only,
        as SlowJet's select=2 does: neutrinos, and any particle Pythia's
        particle data marks invisible (e.g. neutralinos), are left out
      - invisible_pids: extra |PDG id|s to treat as invisible, for BSM states
        Pythia's particle table does not know (unknown ids count as visible)
    Output:
      JSON with jets [{px, py, pz, E, m, pT, eta, y, phi, n_const,
      constituents}].
      eta is pseudorapidity and y rapidity. Each constituent's "event_index" is
      its position in the event's particle list as stored in the dataset.
    """

    # Tool arguments. Optional fields take non-None defaults:
    # RuntimeField(default=None) is read by the schema generator as "no
    # default", which marks the field required.
    events_path: str = RuntimeField(
        default="",
        description="Path to an event dataset (.root or .jsonl) written by "
        "PythiaFromRunCardTool",
    )
    npy_path: str = RuntimeField(
        default="", description="Path to padded events .npy"
    )
    output_path: str = RuntimeField(
        default="",
        description="Path to save output .jsonl (required if "
        "cluster_all=True)",
    )
    event_index: int = RuntimeField(
        default=-1,
        description="0-based event index to cluster (not needed if "
        "cluster_all=True)",
    )
    cluster_all: bool = RuntimeField(
        default=False, description="Cluster all events in dataset"
    )

    # SlowJet arguments
    algorithm: str = RuntimeField(
        default="antikt", description="antikt | kt | ca"
    )
    R: float = RuntimeField(default=0.4, description="Jet radius")
    ptmin: float = RuntimeField(default=20.0, description="Min jet pT [GeV]")
    etamax: float = RuntimeField(
        default=5.0, description="Max |eta| for constituents"
    )
    mass_option: int = RuntimeField(
        default=1, description="Recombination scheme (SlowJet massOption)"
    )
    include_invisible: bool = RuntimeField(
        default=False,
        description="Cluster invisible particles (neutrinos, neutralinos, "
        "...) too. Default False: visible only",
    )
    invisible_pids: List[int] = Field(
        default_factory=list,
        json_schema_extra={"runtime": True},
        description="Extra |PDG id|s to treat as invisible, for BSM states "
        "Pythia's particle table does not know",
    )

    # Sandbox root
    base_directory: str = StateField(
        default=".", description="Base directory for safe path resolution"
    )

    def _get_pythia8(self):
        """Get or create a shared Pythia8 module reference."""
        if not hasattr(self, "_pythia8_module"):
            self._pythia8_module = require_pythia()
        return self._pythia8_module

    def _setup(self):
        """Setup base directory and validate it exists."""
        self.base_directory = os.path.abspath(self.base_directory)
        if not os.path.isdir(self.base_directory):
            raise ValueError(
                f"Base directory does not exist or is not a directory: "
                f"{self.base_directory}"
            )

    def _safe_path(self, rel: Optional[str]) -> Optional[str]:
        """
        Resolve rel against base_directory.
        Return absolute path if and only if it is inside base_directory.
        Otherwise return None.
        """
        if not rel:
            return None
        # If user passed an absolute path, interpret it relative to
        # base_directory anyway so that "/tmp/x" cannot escape.
        rel_norm = rel.lstrip(os.sep)
        full = os.path.abspath(os.path.join(self.base_directory, rel_norm))
        if (
            not full.startswith(self.base_directory + os.sep)
            and full != self.base_directory
        ):
            return None
        return full

    @staticmethod
    def _cluster_inputs(
        particles: List[Dict[str, Any]],
    ) -> List[Dict[str, float]]:
        """
        Put one event's particles in the form the SlowJet setup expects.

        The columnar layout spells energy "e"; the clustering code predates it
        and expects "E", as the .npy path supplies. Each particle keeps its
        position in the stored event as "i", so constituents can be reported
        against the dataset.
        """
        return [
            {
                "id": int(p.get("id", 0)),
                "px": float(p["px"]),
                "py": float(p["py"]),
                "pz": float(p["pz"]),
                "E": float(p["e"]),
                "m": float(p.get("m", 0.0)),
                "i": position,
            }
            for position, p in enumerate(particles)
        ]

    @staticmethod
    def _npy_inputs(rows: Any) -> List[Dict[str, float]]:
        """One event of a padded .npy array; all-zero rows are padding."""
        particles = []
        for j, row in enumerate(rows):
            px, py, pz, E, pid = (float(v) for v in row)
            if px == 0 and py == 0 and pz == 0 and E == 0 and pid == 0:
                continue
            m2 = max(E * E - (px * px + py * py + pz * pz), 0.0)
            particles.append(
                {
                    "id": int(pid),
                    "px": px,
                    "py": py,
                    "pz": pz,
                    "E": E,
                    "m": math.sqrt(m2),
                    "i": j,
                }
            )
        return particles

    def _events(
        self, src: str, is_npy: bool
    ) -> Iterator[Tuple[int, List[Dict[str, float]]]]:
        """Stream (index, particles) over every event of a dataset."""
        if is_npy:
            array = np.load(src, allow_pickle=False, mmap_mode="r")
            for index in range(array.shape[0]):
                yield index, self._npy_inputs(array[index])
            return
        index = 0
        for chunk in event_io.iter_events(src, step_size=1000):
            for particles in chunk["prt"].to_list():
                yield index, self._cluster_inputs(particles)
                index += 1

    def _event(
        self, src: str, is_npy: bool, index: int
    ) -> List[Dict[str, float]]:
        """Read one event, and only that event."""
        if is_npy:
            array = np.load(src, allow_pickle=False, mmap_mode="r")
            if index >= array.shape[0]:
                raise IndexError(
                    f"event_index {index} out of range for {array.shape[0]} "
                    f"events"
                )
            return self._npy_inputs(array[index])
        events = event_io.read_events(src, index, index + 1)
        if not len(events):
            raise IndexError(
                f"event_index {index} out of range for "
                f"{event_io.count_events(src)} events"
            )
        return self._cluster_inputs(events["prt"][0].to_list())

    def _particle_data(self, pythia8: Any) -> Any:
        """
        Pythia's particle data, for the visibility of each species.

        A hand-built Event knows nothing about the particles in it until it is
        attached to a ParticleData table; without one every particle counts as
        visible, and neutrinos end up inside jets. Building the table costs a
        Pythia constructor, so it is made once per tool instance.
        """
        if not hasattr(self, "_pythia_for_pdata"):
            with FdSilence(STDOUT, STDERR):
                self._pythia_for_pdata = pythia8.Pythia("", printBanner=False)
        return self._pythia_for_pdata.particleData

    def _build_pythia_event(
        self, pythia8: Any, particles: List[Dict[str, float]]
    ) -> Tuple[Any, Dict[int, Dict[str, float]]]:
        """
        Build a Pythia8.Event holding the particles to cluster.

        Event.reset() puts Pythia's system entry (id 90) at index 0, so the
        particles occupy indices 1..n. The returned map takes each event index
        back to the particle it came from, which is how constituents are
        reported against the dataset rather than against this event.

        Returns:
            The event, and a map of event index -> particle dict
        """
        evt = pythia8.Event()
        evt.init("", self._particle_data(pythia8))
        evt.reset()

        extra_invisible = {abs(int(pid)) for pid in self.invisible_pids}
        by_index: Dict[int, Dict[str, float]] = {}
        for p in particles:
            pid = int(p.get("id", 0))
            if not self.include_invisible and abs(pid) in extra_invisible:
                continue
            # Event.append(id, status, col, acol, px, py, pz, e, m); status 1
            # marks the particle final so SlowJet will consider it.
            k = evt.append(
                pid,
                1,
                0,
                0,
                p["px"],
                p["py"],
                p["pz"],
                p["E"],
                float(p.get("m", 0.0)),
            )
            by_index[int(k)] = p
        return evt, by_index

    def _cluster(
        self, event_particles: List[Dict[str, float]], pythia8
    ) -> Dict[str, Any]:
        """Perform jet clustering on the provided event particles."""
        power = _ALGO_TO_POWER.get(self.algorithm.lower())
        if power is None:
            raise ValueError(
                f"Unsupported algorithm '{self.algorithm}'. Use one of "
                f"{list(_ALGO_TO_POWER)}"
            )

        evt, by_index = self._build_pythia_event(pythia8, event_particles)

        # select: 1 clusters every final particle, 2 only the visible ones,
        # judged by the particle data the event is attached to.
        select = 1 if self.include_invisible else 2
        sj = pythia8.SlowJet(
            int(power),
            float(self.R),
            float(self.ptmin),
            float(self.etamax),
            select,
            int(self.mass_option),
        )

        # Quietly (supress FastJet banner) cluster with SlowJet.
        with FdSilence(STDOUT, STDERR):
            sj.analyze(evt)

        n_jets = int(sj.sizeJet())
        jets = []
        for j in range(n_jets):
            # constituents(j) returns indices into evt, not into the dataset.
            constituents = []
            for k in sj.constituents(j):
                p = by_index[int(k)]
                constituents.append(
                    {
                        "event_index": int(p["i"]),
                        "id": int(p.get("id", 0)),
                        "px": float(p["px"]),
                        "py": float(p["py"]),
                        "pz": float(p["pz"]),
                        "E": float(p["E"]),
                        "m": float(p.get("m", 0.0)),
                    }
                )

            p4 = sj.p(j)
            jets.append(
                {
                    "index": j,
                    "px": float(p4.px()),
                    "py": float(p4.py()),
                    "pz": float(p4.pz()),
                    "E": float(p4.e()),
                    "m": float(sj.m(j)),
                    "pT": float(sj.pT(j)),
                    "eta": float(p4.eta()),
                    "y": float(sj.y(j)),
                    "phi": float(sj.phi(j)),
                    "n_const": len(constituents),
                    "constituents": constituents,
                }
            )

        result = {
            "algorithm": self.algorithm.lower(),
            "R": float(self.R),
            "ptmin": float(self.ptmin),
            "etamax": float(self.etamax),
            "mass_option": int(self.mass_option),
            "include_invisible": bool(self.include_invisible),
            "data": {
                "n_jets": n_jets,
                "jets": jets,
            },
        }
        return result

    def _run(self) -> str:
        """Run jet clustering and return JSON summary."""
        try:
            self._setup()
        except Exception as e:
            return self.format_error(error="Path Error", reason=str(e))

        if bool(self.events_path) == bool(self.npy_path):
            return self.format_error(
                error="Invalid Parameters",
                reason="give exactly one input: events_path or npy_path",
                suggestion="events_path for a dataset from "
                "PythiaFromRunCardTool",
            )
        field = "events_path" if self.events_path else "npy_path"
        is_npy = bool(self.npy_path)
        src = self._safe_path(self.events_path or self.npy_path)
        if src is None:
            return self.format_error(
                error="Path Error",
                reason=f"{field} is outside allowed base_directory",
            )
        if not os.path.exists(src):
            return self.format_error(
                error="File Not Found", reason=f"{src} not found"
            )

        if not self.cluster_all and int(self.event_index) < 0:
            return self.format_error(
                error="Missing Parameter",
                reason="event_index is required when cluster_all=False",
                suggestion="Provide event_index to cluster a single event or "
                "set cluster_all=True",
            )
        outpath = None
        if self.cluster_all:
            if not self.output_path:
                return self.format_error(
                    error="Missing Parameter",
                    reason="output_path is required when cluster_all=True",
                    suggestion="Provide output_path to save clustered jets",
                )
            outpath = self._safe_path(self.output_path)
            if outpath is None:
                return self.format_error(
                    error="Path Error",
                    reason="output_path is outside allowed base_directory",
                )

        n_events = 0
        results = []
        try:
            pythia8 = self._get_pythia8()
            if self.cluster_all:
                os.makedirs(os.path.dirname(outpath), exist_ok=True)
                with open(outpath, "w", encoding="utf-8") as outfile:
                    for index, particles in tqdm(
                        self._events(src, is_npy),
                        desc="Clustering events",
                        **TQDM_CONFIG,
                    ):
                        if not particles:
                            continue
                        result = self._cluster(particles, pythia8)
                        result["event_index"] = index
                        outfile.write(
                            json.dumps(
                                result,
                                separators=(",", ":"),
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        n_events += 1
            else:
                index = int(self.event_index)
                particles = self._event(src, is_npy, index)
                if not particles:
                    return self.format_error(
                        error="Empty Event",
                        reason="No particles at event_index",
                    )
                result = self._cluster(particles, pythia8)
                result["event_index"] = index
                results = [result]
                n_events = 1
        except Exception as e:
            return self.format_error(
                error="Clustering Error", reason=f"{type(e).__name__}: {e}"
            )

        # Return summary
        if self.cluster_all:
            # Streaming mode - return summary only
            return json.dumps(
                {
                    "status": "ok",
                    "n_events": n_events,
                    "output_file": os.path.relpath(
                        outpath, self.base_directory
                    ),
                },
                separators=(",", ":"),
                ensure_ascii=False,
            )
        else:
            # Single event mode - return full result
            return json.dumps(
                {"status": "ok", "n_events": n_events, "results": results},
                separators=(",", ":"),
                ensure_ascii=False,
            )
