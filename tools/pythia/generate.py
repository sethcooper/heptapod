"""
# generate.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Event generation with Pythia 8 from a run card: PythiaFromRunCardTool,
through Pythia's Python interface, and PythiaCPPFromRunCardTool, through a
C++ driver compiled against a Pythia 8 install. Both write the same dataset
from the same card; which Pythia ran is recorded in the manifest.
"""

import datetime
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField
from pydantic import Field

from tools import event_io
from tools.events.generation import (
    GenerationError,
    GenerationPlan,
    RivetPlan,
    cross_section_fields,
    cross_section_pb_from_mb,
    run_generation,
)
from tools.cpp.build import BuildError, configured_path, resolve_install

from .card import IncludeError, include_search_path, inline_includes
from .driver import DriverInitError, PythiaDriverSource
from .runtime import (
    STDERR,
    STDOUT,
    FdSilence,
    get_info,
    import_pythia,
    log_tail,
    pythia_version,
    weight_names,
)

# Schema of manifest.json. The dataset's own schema is recorded inside it, as
# outputs.events_schema, since it depends on the output format.
MANIFEST_SCHEMA_VERSION = "pythia-run-manifest-1.0"


def _beams(pythia: object) -> List[Dict[str, float]]:
    """
    The two beam particles of an initialised run, as four-momenta in GeV.

    Beam A travels along +z and B along -z, except for Beams:frameType = 3,
    where the card gives each beam's momentum. Masses come from Pythia's own
    particle data, so the beams are exactly the particles Pythia collided.
    """
    import math

    info = get_info(pythia)
    settings = pythia.settings
    beams = []
    for side, pid, energy, sign in (
        ("A", info.idA(), info.eA(), 1.0),
        ("B", info.idB(), info.eB(), -1.0),
    ):
        mass = float(pythia.particleData.m0(pid))
        if settings.mode("Beams:frameType") == 3:
            px, py, pz = (
                float(settings.parm(f"Beams:p{axis}{side}")) for axis in "xyz"
            )
            energy = math.sqrt(px * px + py * py + pz * pz + mass * mass)
        else:
            px, py, pz = (
                0.0,
                0.0,
                sign * math.sqrt(max(energy * energy - mass * mass, 0.0)),
            )
        beams.append(
            {
                "id": int(pid),
                "px": px,
                "py": py,
                "pz": pz,
                "e": float(energy),
                "m": mass,
            }
        )
    return beams


def _particle_names() -> Dict[str, str]:
    """PDG id -> name, from the settings snapshot. Empty if unavailable."""
    try:
        from .settings_db import load_snapshot

        return load_snapshot().get("particles", {})
    except Exception:
        return {}


# Pythia setting names are case-insensitive, and "=" may or may not be spaced.
_LHEF_LINE = re.compile(r"^\s*beams:lhef\s*=", re.IGNORECASE)

# Pythia accepts seeds in [-1, 900000000]; 0 derives one from the clock, which
# is irreproducible, so the tool accepts only an explicit positive seed.
_SEED_MAX = 900_000_000


def _edit_pythia_card(
    card_text: str,
    *,
    lhe_path: Optional[str] = None,
    seed: Optional[int] = None,
) -> str:
    """
    Apply the tool's runtime overrides to a run card.

    The overrides are appended as a trailing block rather than substituted in
    place. Pythia applies a card top to bottom and the last assignment wins, so
    a trailing block is guaranteed to take effect however the card spells the
    setting -- lower case, no spaces around "=", or not at all. The copy
    written to the run directory is then an exact record of what ran.

    Existing Beams:LHEF lines are commented out rather than left to be
    overridden, so the recorded card names only the file that was showered.

    Parameters:
        card_text: Original .cmnd file content
        lhe_path: LHE file to shower; also sets Beams:frameType = 4, without
            which Pythia ignores Beams:LHEF
        seed: Random seed, which then takes precedence over any in the card

    Returns:
        The card with overrides applied, unchanged when there are none
    """
    overrides: List[str] = []
    if lhe_path is not None:
        overrides += ["Beams:frameType = 4", f"Beams:LHEF = {lhe_path}"]
    if seed is not None:
        overrides += ["Random:setSeed = on", f"Random:seed = {int(seed)}"]
    if not overrides:
        return card_text

    lines = card_text.splitlines()
    if lhe_path is not None:
        lines = [
            f"! {line.strip()}   (replaced below)"
            if _LHEF_LINE.match(line)
            else line
            for line in lines
        ]
    lines += [
        "",
        "! ---- Set by PythiaFromRunCardTool; overrides anything above ----",
        *overrides,
    ]
    return "\n".join(lines) + "\n"


#: Applied before the card is read, so anything the card sets wins. They keep
#: Pythia's own reporting out of the way; the log still receives its errors.
_QUIET_SETTINGS = (
    "Print:quiet = on",
    "Init:showProcesses = off",
    "Init:showMultipartonInteractions = off",
    "Init:showChangedSettings = off",
    "Init:showChangedParticleData = off",
    "Next:numberShowInfo = 0",
    "Next:numberShowProcess = 0",
    "Next:numberShowEvent = 0",
)


class VertexLayoutError(RuntimeError):
    """nextBatch stored the production vertices against the wrong
    particles."""


class _PythonSource:
    """
    Events from Pythia's Python interface, in-process, through
    Pythia.nextBatch.

    The generation loop reads events from a source with this interface;
    tools.pythia.driver.PythiaDriverSource is the other implementation, a
    Pythia 8 install behind the C++ driver.

    The first batch is checked against Pythia's own event record: the
    extension a Pythia 8.317 build makes stores the production vertices
    shifted onto the wrong particles (the pythia8mc wheel does not), and a
    store that keeps vertices must not be written from it.
    """

    def __init__(
        self,
        pythia8: Any,
        card: str,
        log_path: str,
        *,
        lhe: bool,
        store: str = event_io.DEFAULT_STORE,
        describe: Optional[Dict[str, Any]] = None,
        notes: Optional[List[str]] = None,
    ) -> None:
        self.pythia8 = pythia8
        self.card = card
        self.log_path = log_path
        self.lhe = lhe
        self.keeps_vertices = store != "finals+kinematics"
        self._describe = dict(describe or {"backend": "pythia8mc"})
        #: Warnings about how Pythia was found, for the run's result.
        self.notes: List[str] = list(notes or [])
        #: How many weights nextBatch wrote per event, once known.
        self._weights_width: Optional[int] = None
        self._vertices_checked = False
        self.pythia: Any = None
        self.beams: List[Dict[str, float]] = []
        self.version: Optional[str] = None
        self.failed = 0
        self.lhe_exhausted = False

    def start(self) -> bool:
        """Construct and initialise Pythia from the card; False if init
        failed."""
        # Pythia8 is C++ and writes settings/init errors straight to
        # stdout/stderr. Under toolbase's MCP stdio transport, stdout IS the
        # JSON-RPC channel, so any leak corrupts it ("Failed to parse
        # JSONRPC message from server"). `Print:quiet` only suppresses
        # normal logging, not hard errors/aborts (e.g. an xmldoc version
        # mismatch), so redirect the fds around every Pythia C++ call. The
        # output goes to pythia.log rather than /dev/null: it is the only
        # place Pythia explains why init failed.
        with FdSilence(STDOUT, STDERR, target=self.log_path):
            pythia = self.pythia8.Pythia("", printBanner=False)
            for setting in _QUIET_SETTINGS:
                pythia.readString(setting)
            pythia.readFile(str(self.card))
            init_ok = pythia.init()
            if not init_ok:
                # Print:quiet (which the card may also set) hides Pythia's
                # error messages as they happen, but stat() always lists
                # them, so the log explains the failure either way.
                pythia.stat()
        self.pythia = pythia
        if init_ok:
            self.beams = _beams(pythia)
            self.version = pythia_version(pythia)
        return bool(init_ok)

    def next_batch(self, want: int) -> Any:
        """Up to `want` events; the shortfall is counted as failures, or as the
        LHE file running out."""
        # Redirect the fds across generation: per-event Pythia warnings would
        # otherwise reach stdout and corrupt the MCP transport.
        with FdSilence(STDOUT, STDERR, target=self.log_path):
            # "skip" drops failed events from the batch, so the shortfall is
            # the failure count -- the same accounting the per-event loop
            # used to do by hand.
            batch = self.pythia.nextBatch(want, "skip")
        produced = len(batch)
        if produced and self._weights_width is None:
            try:
                self._weights_width = len(batch.info.weights[0])
            except Exception:
                pass
        if produced and not self._vertices_checked:
            self._vertices_checked = True
            if not self._vertices_aligned(batch):
                message = (
                    "this Pythia's Python interface stores the production "
                    "vertices against the wrong particles (the extension a "
                    "Pythia 8.317 build makes does; the pythia8mc wheel does "
                    "not)"
                )
                if self.keeps_vertices:
                    raise VertexLayoutError(
                        message + ", so a dataset with vertices cannot be "
                        "written from it. Use PythiaCPPFromRunCardTool for "
                        "this install, the pythia8mc wheel, or "
                        "store='finals+kinematics', which carries none."
                    )
                self.notes.append(
                    message.capitalize() + "; the store written carries no "
                    "vertices, so the dataset is unaffected."
                )
        # infoPython() returns a snapshot, not a live view, so it is fetched
        # afresh wherever it is read.
        info = get_info(self.pythia)
        if self.lhe and info is not None and info.atEndOfFile():
            # The LHE file ran out. That shortfall is the input's size, not
            # events Pythia failed to generate.
            self.lhe_exhausted = True
        else:
            self.failed += want - produced
        return batch

    def _vertices_aligned(self, batch: Any) -> bool:
        """Whether the batch's last event carries its production vertices on
        the particles Pythia's event record (still holding that event) says
        have one. True when there is nothing to compare."""
        if "vProd" not in batch.prt.fields:
            return True
        try:
            import awkward as ak

            missing = ak.to_list(ak.is_none(batch.prt[-1].vProd))
            event = self.pythia.event
            if len(missing) != event.size():
                return True
            return all(
                missing[i] == (not event[i].hasVertex())
                for i in range(len(missing))
            )
        except Exception:
            return True

    def weight_names(self) -> List[str]:
        """The names of the weights nextBatch stores, nominal first.

        Pythia's batch interface has written two layouts: every weight of
        weightNameVector (the variations and their groups), or the group
        weights only, which a build's own extension does. The names match
        whichever the batch holds.
        """
        info = get_info(self.pythia)
        names = weight_names(info)
        width = self._weights_width
        if width is None or width == len(names):
            return names
        try:
            groups = int(info.nWeightGroups())
            if width == groups:
                return ["Weight"] + [
                    str(info.getGroupName(i)) for i in range(1, groups)
                ]
        except Exception:
            pass
        return names[:width]

    def cross_section(self) -> Dict[str, float]:
        """sigmaGen_mb, sigmaErr_mb, weightSum, nAccepted, nTried so far."""
        out: Dict[str, float] = {}
        try:
            info = get_info(self.pythia)
            for key, attr in (
                ("sigmaGen_mb", "sigmaGen"),
                ("sigmaErr_mb", "sigmaErr"),
                ("weightSum", "weightSum"),
                ("nAccepted", "nAccepted"),
                ("nTried", "nTried"),
            ):
                if hasattr(info, attr):
                    out[key] = getattr(info, attr)()
        except Exception:
            pass
        return out

    def cross_section_pb(self) -> Dict[str, float]:
        """The running estimate in pb, as HepMC3 and Rivet take it."""
        return cross_section_pb_from_mb(self.cross_section())

    def finish(self) -> None:
        """Close the run. stat() writes Pythia's cross-section table to
        pythia.log, where it belongs in the run's provenance."""
        with FdSilence(STDOUT, STDERR, target=self.log_path):
            try:
                self.pythia.stat()
            except Exception:
                pass

    def abort(self) -> None:
        """Nothing to stop: Pythia runs in this process."""

    def describe(self) -> Dict[str, Any]:
        return {**self._describe, "version": self.version}


# ====================================================================== #
# =================== Pythia event generation tools ==================== #
# ====================================================================== #

# The two run-card tools share one body, so their descriptions are composed
# from the same pieces: what the run does, then how each reaches Pythia.
_DOC_INPUTS = """
    Inputs (runtime):
      - data_dir: relative output directory under base_directory for the
        run's artifacts
      - cmnd_path: relative path to a valid Pythia8 .cmnd configuration file
      - n_events: number of events to generate
      - seed: random seed in [1, 900000000]. When given it overrides any seed
        in the card; -1 (default) leaves the seed to the card, or to Pythia's
        fixed default if the card sets none
      - output_format: "root" (default), "hepmc3" or "jsonl". ROOT holds an
        event in about 7 KiB against JSONL's 60, and is read back without
        parsing anything. HepMC3 is what Delphes and Rivet read: the final
        state between the two beams, with the weights, their names and the
        cross section; it cannot hold the "full" store spec. JSONL is
        human-readable, which is what a small sample wants
      - store: what to keep -- "finals+kinematics" (default), "finals", or
        "full". This is the largest lever on dataset size: the same events
        cost 58.6 KiB each under "full" and 7 KiB under the default
      - precision: "f32" (default) or "f64" for the stored kinematics
      - chunk_size: events generated per batch, bounding peak memory
        (default 10000)
      - select_pids: optional |PDG id| whitelist; only these species are
        stored (matched on absolute value, so 321 keeps K+ and K-). Applied
        after the store spec, so the summary describes what was written
      - shower_lhe: if True, use Pythia to shower and hadronise LHE events
        (requires lhe_path)
      - lhe_path: path to an LHE file to shower (required when
        shower_lhe=True). It sets Beams:frameType = 4 and Beams:LHEF,
        overriding the card
      - max_output_gb: stop early if the dataset would exceed this size
        (default 2.0; 0 or negative disables the guard)
      - rivet_analyses: stream the events into these Rivet analyses instead
        of writing a file (needs store="full" and rivet_path); rivet_output
        names the YODA file, rivet_ignore_beams runs mismatched analyses
      - base_directory: sandbox root for all file operations
      - pythia_path: a Pythia 8 install prefix; see below
"""

_DOC_RUN = """
    Behavior:
      1. Copy the provided .cmnd file into the output directory as run.cmnd,
         with any `include` inlined and the seed and LHE overrides appended,
         so it records exactly what ran. A relative include is found next to
         the card; an included file must lie inside base_directory or
         Pythia's own directories.
      2. Initialize Pythia8 from run.cmnd.
      3. Record Pythia's own log in pythia.log alongside the dataset.
      4. Generate events in batches. Each batch is reduced to the store
         spec, folded into the run summary, written, and released, so a
         sample far larger than memory can be produced.
      5. Write the dataset as events.root, events.hepmc or events.jsonl,
         alongside a manifest recording the run, including its beams.
      6. Report the run summary, cross section, and weight statistics.

    Note that generation is dominated by the physics, not by the writing: the
    event itself costs about 2.7 ms, so a faster serializer moves the total
    from roughly 4.8 to 3.4 ms per event rather than transforming it. The
    format's real wins are on the other side -- an eightfold smaller dataset
    that is read back several times faster and needs no parsing.

    LHE Showering Mode:
      When shower_lhe=True, Pythia showers and hadronises pre-generated LHE
      events rather than generating them itself. In this mode:
        - lhe_path is REQUIRED and specifies the path to the input LHE file
        - The tool sets Beams:frameType = 4 and Beams:LHEF, whatever the
          card says
        - Pythia adds parton showering and hadronization to matrix-element
          level events
        - An LHE file with fewer events than n_events ends the run early; the
          shortfall is reported as "lhe_exhausted", not as failed events

    Output (JSON):
      {
        "status": "ok",
        "data_dir": "<relative output directory>",
        "events": "<relative path to the events file>",
        "format": "root" | "hepmc3" | "jsonl",
        "store": "<store spec applied>",
        "accepted": <int>,
        "failed": <int>,
        "backend": "<what ran Pythia: pythia8mc, pythia8 or cpp>",
        "xsec": {"sigmaGen_mb": <float>, "sigmaErr_mb": <float>,
                 "sigmaGen_pb": <float>, "sigmaErr_pb": <float>,
                 "weightSum": <float>},
        "weighted": <bool>,
        "summary": {...},
        "warnings": ["..."]
      }

    Run summary:
      A generated sample is not something to open -- hundreds of particles per
      event, and a million events is gigabytes whatever the format. The result
      and the manifest therefore carry a "summary" accumulated while
      generating: particles per event, mean leading pT, and the most common
      species by name. Read that to check the card produced the intended
      process; minimum bias and ttbar do not look alike in it.

      "max_output_gb" bounds the damage of an over-large request. When the
      dataset would exceed it, generation stops, "truncated" is set, and the
      events already written remain complete and usable.

    Event weights:
      Every event carries a "weight" holding its generation weight, whatever
      the store spec. Most runs are unweighted and every weight is 1.0.
      Cross-section biasing (PhaseSpace:bias2Selection), ME/PS merging, and
      shower uncertainty variations produce weights spanning many orders of
      magnitude. When any weight differs from 1.0 the result reports
      "weighted": true along with the observed range, and every downstream
      histogram, efficiency and yield must use the per-event weight.

      A run with weight variations (UncertaintyBands:doVariations, or LHE
      reweighting) also stores every event's variation weights in "weights",
      in the order the manifest's outputs.weights.names gives them; the first
      is the nominal weight. Weights are stored in double precision whatever
      `precision` says, since a weight sum over many events needs it.

    Errors:
      Returns BaseTool.format_error JSON on any failure including:
        - invalid or missing cmnd_path
        - no Pythia to run (see above)
        - initialization or event generation failures
        - file read/write or permission issues

    Notes:
      - All paths must remain inside base_directory for safety.
      - Output files are always written under data_dir: run.cmnd, pythia.log,
        manifest.json and the dataset.
      - sigmaGen is in mb (1 mb = 1e9 pb), with the pb figure beside it for
        comparison with Sherpa. For a weighted sample, normalise a histogram
        by sigmaGen / weightSum, not sigmaGen / accepted.
"""

_DOC_PYTHON = (
    """
    Generate hadron-level events with Pythia 8 from a .cmnd run card,
    through Pythia's Python interface.
"""
    + _DOC_INPUTS
    + """
    Which Pythia:
      Pythia runs in this process through its Python interface and
      Pythia.nextBatch, which builds each batch as an Awkward Array in C++.
      The interface is the pythia8mc wheel or conda's pythia8 module, or the
      pythia8 extension of a Pythia 8 build configured --with-python when
      pythia_path names that build; the extension must have been built for
      the Python serving heptapod, and when it cannot be loaded the module
      on sys.path runs instead and the result says so. The extension a
      Pythia 8.317 build makes stores production vertices against the wrong
      particles, so a store with vertices is refused through it. For a
      build without a Python interface, or for any Pythia version the wheel
      does not have, use PythiaCPPFromRunCardTool: it takes the same inputs
      and writes the same dataset. The manifest records which Pythia ran
      under inputs.generator.
"""
    + _DOC_RUN
)

_DOC_CPP = (
    """
    Generate hadron-level events with Pythia 8 from a .cmnd run card,
    through a C++ driver compiled against the Pythia 8 install named by
    pythia_path.
"""
    + _DOC_INPUTS
    + """
    Which Pythia:
      The install named by pythia_path (or PYTHIA_PATH), the prefix holding
      bin/pythia8-config, at whatever version and with whatever plugins it
      was built. heptapod-pythia, a small C++ driver
      (tools/pythia/pythia_driver.cc), is compiled against it once on first
      use (cached under ~/.cache/heptapod/cpp or $HEPTAPOD_CPP_CACHE) and
      streams the event record back as it generates. The driver needs only
      Pythia itself and a C++ compiler on PATH (CXX overrides), not HepMC3
      or Python bindings. The store spec, the summary and the ROOT, HepMC3
      and JSONL writers are the same code PythiaFromRunCardTool uses, so
      the dataset does not depend on which tool ran; the manifest records
      it under inputs.generator.
"""
    + _DOC_RUN
)


class _PythiaRunCardTool(BaseTool):
    """The run shared by PythiaFromRunCardTool and PythiaCPPFromRunCardTool;
    a subclass supplies the source of events in _source."""

    # --------------------------- Runtime fields --------------------------- #
    data_dir: str = RuntimeField(
        description="Relative output directory for dataset, e.g. 'data/run001'"
    )
    cmnd_path: str = RuntimeField(
        description="Relative path to Pythia .cmnd run card template"
    )
    n_events: int = RuntimeField(description="Number of events to generate")
    # Optional fields take non-None defaults: RuntimeField(default=None) is
    # read by the schema generator as "no default", which marks the field
    # required.
    seed: int = RuntimeField(
        default=-1,
        description="Random seed in [1, 900000000]; overrides any seed in the "
        "card. -1 (default) leaves the seed to the card",
    )
    output_format: str = RuntimeField(
        default="root",
        description="Dataset format: 'root' (default, compact and fast), "
        "'hepmc3' (for Delphes or Rivet; with store='full' it carries the "
        "complete event genealogy, as Pythia's own HepMC3 output does) or "
        "'jsonl' (human-readable)",
    )
    store: str = RuntimeField(
        default=event_io.DEFAULT_STORE,
        description="What to store: 'finals+kinematics' (default: final-state "
        "momenta, ids and masses), 'finals' (final state with all fields, "
        "incl. vertices), or 'full' (every particle, with lineage)",
    )
    precision: str = RuntimeField(
        default="f32",
        description="Floating-point precision for stored kinematics: 'f32' "
        "(default) or 'f64'",
    )
    chunk_size: int = RuntimeField(
        default=10000,
        description="Events generated per batch; bounds peak memory",
    )
    select_pids: List[int] = Field(
        default_factory=list,
        json_schema_extra={"runtime": True},
        description="Optional |PDG id| whitelist; when set, store only "
        "particles "
        "whose |id| is listed (matched on absolute value, so 321 keeps "
        "both signs). Empty/unset stores all particles",
    )
    shower_lhe: bool = RuntimeField(
        default=False,
        description="If True, use Pythia for showering/hadronization of LHE "
        "events (requires lhe_path)",
    )
    lhe_path: str = RuntimeField(
        default="",
        description="Optional; leave unset for normal generation from the run "
        "card. Set it (with shower_lhe=True) to shower/hadronize a "
        "pre-generated LHE file",
    )
    max_output_gb: float = RuntimeField(
        default=2.0,
        description="Stop early if the dataset would exceed this many GB (0 "
        "or negative disables the guard)",
    )
    gzip_output: bool = RuntimeField(
        default=False,
        description="Write HepMC3 or JSONL output gzipped (events.hepmc.gz, "
        "events.jsonl.gz); ROOT is compressed already. Rivet reads gzipped "
        "HepMC3 directly",
    )
    rivet_analyses: List[str] = Field(
        default_factory=list,
        json_schema_extra={"runtime": True},
        description="Run these Rivet analyses on the events as they are "
        "generated, streamed through a pipe, instead of writing an event "
        "file: no multi-gigabyte HepMC3 on disk. Needs store='full' and a "
        "configured rivet_path; output_format is then HepMC3 regardless. "
        "Names as RivetFindAnalysesTool lists them, with options "
        "('MC_ZINC:LMODE=MU')",
    )
    rivet_output: str = RuntimeField(
        default="",
        description="YODA file for the Rivet histograms when rivet_analyses "
        "is given (default: data_dir/rivet.yoda)",
    )
    rivet_ignore_beams: bool = RuntimeField(
        default=False,
        description="With rivet_analyses: run analyses whose beams or energy "
        "do not match the run, knowingly",
    )
    # ---------------------------------------------------------------------- #

    # ---------------------------- State fields ---------------------------- #
    base_directory: str = StateField(
        description="Base directory for safe path resolution"
    )
    rivet_path: str = StateField(
        default="",
        description="Rivet installation prefix (holds bin/rivet-config), for "
        "rivet_analyses; RIVET_PATH when unset",
    )
    pythia_path: str = StateField(
        default="",
        description="Pythia 8 install prefix; PYTHIA_PATH when unset",
    )
    # ---------------------------------------------------------------------- #

    def _setup(self):
        """Setup base directory and validate it exists."""
        self.base_directory = os.path.abspath(self.base_directory)
        if not os.path.exists(self.base_directory):
            raise ValueError(
                f"Base directory does not exist: {self.base_directory}"
            )

    def _safe_path(self, rel: str) -> Optional[str]:
        """Ensures that the path is within the allowed base directory."""
        full = os.path.abspath(os.path.join(self.base_directory, rel))
        if full != self.base_directory and not full.startswith(
            self.base_directory + os.sep
        ):
            return None
        return full

    def _locate(self) -> Optional[str]:
        """Find the Pythia this tool runs, before anything else in the run
        can import one; a format_error string when there is none."""
        raise NotImplementedError

    def _source(
        self,
        *,
        card: str,
        log_path: str,
        work_dir: str,
        select_pids: Optional[List[int]],
        lhe: bool,
    ) -> Any:
        """The source of events for this run, from what _locate found."""
        raise NotImplementedError

    def _run(self) -> str:
        """Run Pythia event generation and return JSON summary."""
        # Check required parameters.
        for key in ("data_dir", "cmnd_path", "n_events"):
            if getattr(self, key, None) in (None, ""):
                return self.format_error(
                    error="Missing Parameter",
                    reason=f"{key} is required",
                    suggestion="Provide required runtime fields",
                )

        # Validate shower_lhe mode requirements.
        if self.shower_lhe and not self.lhe_path:
            return self.format_error(
                error="Missing Parameter",
                reason="lhe_path is required when shower_lhe=True",
                suggestion="Provide lhe_path for LHE showering mode or set "
                "shower_lhe=False",
            )

        # Validate the dataset options before spending a generation run on
        # them.
        fmt = str(self.output_format).lower()
        if fmt not in event_io.FORMATS:
            return self.format_error(
                error="Invalid Parameter",
                reason=f"unknown output_format '{self.output_format}'",
                context=f"valid formats: {', '.join(event_io.FORMATS)}",
                suggestion="Use 'root' for generated samples, 'hepmc3' for "
                "Delphes or "
                "Rivet, 'jsonl' for small inspectable ones",
            )
        if self.store not in event_io.STORE_SPECS:
            return self.format_error(
                error="Invalid Parameter",
                reason=f"unknown store spec '{self.store}'",
                context=f"valid specs: {', '.join(event_io.STORE_SPECS)}",
                suggestion="Name one of the listed specs, or omit `store` for "
                "the default",
            )
        streaming = bool(self.rivet_analyses)
        if streaming:
            # Rivet reads HepMC3 and its analyses walk the decay history, so
            # the stream is the complete record whatever else was asked for.
            fmt = "hepmc3"
            if self.store != "full":
                return self.format_error(
                    error="Invalid Parameter",
                    reason="Rivet analyses need the complete event record",
                    context=f"store='{self.store}'",
                    suggestion="Pass store='full' with rivet_analyses: "
                    "Rivet's projections for prompt leptons, heavy-flavour "
                    "hadrons and unstable particles read the decay history, "
                    "which the final-state stores drop",
                )
        seed = int(self.seed)
        if seed != -1 and not 1 <= seed <= _SEED_MAX:
            return self.format_error(
                error="Invalid Parameter",
                reason=f"seed {seed} is outside [1, {_SEED_MAX}]",
                context="Pythia rejects larger seeds, and 0 seeds from the "
                "clock, which cannot be reproduced",
                suggestion="Pass a seed in range, or -1 to keep the card's "
                "seed",
            )
        if str(self.precision) not in ("f32", "f64"):
            return self.format_error(
                error="Invalid Parameter",
                reason=f"unknown precision '{self.precision}'",
                context="valid precisions: f32, f64",
                suggestion="Use 'f32' unless full double precision is "
                "genuinely needed",
            )

        outdir = self._safe_path(self.data_dir)
        cmnd_src = self._safe_path(self.cmnd_path)

        # Check for safe paths.
        if not outdir or not cmnd_src:
            return self.format_error(
                error="Access Denied",
                reason="Path escapes base_directory",
                context=f"data_dir={self.data_dir}, "
                f"cmnd_path={self.cmnd_path}",
                suggestion="Use paths inside the allowed base directory",
            )

        # The Pythia this tool runs. Found first: resolving the card's
        # includes below may import the wheel, and Pythia's Python interface
        # loads only once per process.
        problem = self._locate()
        if problem:
            return problem

        # Check if run card exists.
        if not os.path.exists(cmnd_src):
            return self.format_error(
                error="File Not Found",
                reason="Run card does not exist",
                context=f"path={self.cmnd_path}",
                suggestion="Provide a valid .cmnd file path",
            )

        # Validate and resolve LHE path if provided.
        lhe_path_abs = None
        if self.lhe_path:
            lhe_path_abs = self._safe_path(self.lhe_path)
            if not lhe_path_abs:
                return self.format_error(
                    error="Access Denied",
                    reason="lhe_path escapes base_directory",
                    context=f"lhe_path={self.lhe_path}",
                    suggestion="Use paths inside the allowed base directory",
                )
            if not os.path.exists(lhe_path_abs):
                return self.format_error(
                    error="File Not Found",
                    reason="LHE file does not exist",
                    context=f"path={self.lhe_path}",
                    suggestion="Provide a valid LHE file path",
                )

        # Create output directory.
        os.makedirs(outdir, exist_ok=True)
        cmnd_dst = os.path.join(outdir, "run.cmnd")
        log_path = os.path.join(outdir, "pythia.log")
        if os.path.exists(log_path):
            os.remove(log_path)

        # Read template card, apply runtime edits if provided, then write to
        # output.
        try:
            with open(cmnd_src, "r", encoding="utf-8") as f:
                card_text = f.read()
        except Exception as e:
            return self.format_error(
                error="Read Error",
                reason=str(e),
                context=f"path={self.cmnd_path}",
                suggestion="Verify file exists and is readable",
            )

        # Inline any include, so run.cmnd is complete on its own. Pythia would
        # resolve a relative include against the server's working directory,
        # not the card's, and the included file could change after the run.
        try:
            card_text = inline_includes(
                card_text,
                card_dir=os.path.dirname(cmnd_src),
                allowed_roots=[
                    Path(self.base_directory),
                    *include_search_path(None),
                ],
            )
        except IncludeError as e:
            return self.format_error(
                error="Include Error",
                reason=str(e),
                context=f"path={self.cmnd_path}",
                suggestion="Give the include a path relative to the card, "
                "inside the sandbox",
            )

        # Append the runtime overrides, so run.cmnd records exactly what ran.
        # The LHE path is absolute so Pythia finds it whatever its working
        # directory; n_events is not a card setting, since the loop controls
        # it.
        card_text = _edit_pythia_card(
            card_text, lhe_path=lhe_path_abs, seed=seed if seed != -1 else None
        )

        # Write modified card to output directory.
        try:
            with open(cmnd_dst, "w", encoding="utf-8") as f:
                f.write(card_text)
        except Exception as e:
            return self.format_error(
                error="Write Error",
                reason=str(e),
                context=f"dst={self.data_dir}/run.cmnd",
                suggestion="Verify permissions and disk space",
            )

        # Check every writer dependency before generating: a missing one
        # would otherwise surface only after init, as an error about the
        # disk.
        try:
            event_io._require_awkward()
            if fmt == "root":
                event_io._require_uproot()
            if fmt == "hepmc3":
                event_io._require_pyhepmc()
        except Exception as e:
            return self.format_error(
                error="Dependency Missing",
                reason=str(e),
                suggestion="Install the missing package in the current "
                "runtime",
            )
        rivet_install = rivet_catalogue = None
        rivet_yoda = ""
        if streaming:
            from tools.rivet.catalogue import load_catalogue
            from tools.rivet.install import RivetError, resolve_rivet

            try:
                rivet_install = resolve_rivet(self.rivet_path)
                rivet_catalogue = load_catalogue(rivet_install.data_dir)
            except (RivetError, FileNotFoundError) as e:
                return self.format_error(
                    error="Rivet Not Available", reason=str(e)
                )
            rivet_yoda = (
                self._safe_path(self.rivet_output)
                if self.rivet_output
                else os.path.join(outdir, "rivet.yoda")
            )
            if not rivet_yoda:
                return self.format_error(
                    error="Access Denied",
                    reason="rivet_output escapes base_directory",
                    context=f"rivet_output={self.rivet_output}",
                )

        # Pythia's initialisation from run.cmnd.
        select_pids = (
            sorted({abs(int(pid)) for pid in self.select_pids}) or None
        )
        source = self._source(
            card=cmnd_dst,
            log_path=log_path,
            work_dir=outdir,
            select_pids=select_pids,
            lhe=bool(lhe_path_abs),
        )
        try:
            init_ok = source.start()
        except DriverInitError:
            init_ok = False
        except BuildError as e:
            return self.format_error(
                error="C++ Build Failed",
                reason=str(e),
                suggestion="Check that pythia_path is a complete Pythia 8 "
                "install and a C++ compiler is on PATH",
            )
        except Exception as e:
            return self.format_error(
                error="Pythia Error",
                reason=str(e),
                suggestion="Check Pythia installation and run card syntax",
            )
        if not init_ok:
            return self.format_error(
                error="Pythia Init Failed",
                reason="Initialization returned false",
                context=log_tail(log_path)
                or f"see {os.path.relpath(log_path, self.base_directory)}",
                suggestion="Fix the errors Pythia reports above; "
                "ValidatePythiaRunCardTool checks a card before a run",
            )

        store = self.store
        suffix = {"root": ".root", "jsonl": ".jsonl", "hepmc3": ".hepmc"}[fmt]
        if self.gzip_output and fmt != "root":
            suffix += ".gz"
        events_path = os.path.join(outdir, "events" + suffix)
        beams = source.beams

        # The loop itself is shared with the other generator tools: batches
        # from the source, reduced to the store spec, summarised and written
        # under the size guard, or streamed into Rivet.
        plan = GenerationPlan(
            events_path=events_path,
            fmt=fmt,
            store=store,
            precision=str(self.precision),
            requested=int(self.n_events),
            chunk=max(1, int(self.chunk_size)),
            select_pids=select_pids,
            max_bytes=(
                int(float(self.max_output_gb) * 1e9)
                if float(self.max_output_gb) > 0
                else 0
            ),
            rivet=(
                RivetPlan(
                    analyses=list(self.rivet_analyses),
                    output_yoda=rivet_yoda,
                    install=rivet_install,
                    catalogue=rivet_catalogue,
                    ignore_beams=bool(self.rivet_ignore_beams),
                )
                if streaming
                else None
            ),
            base_directory=self.base_directory,
            generator="Pythia",
            log_name=os.path.relpath(log_path, self.base_directory),
            diagnostics=lambda: log_tail(log_path),
        )
        try:
            outcome = run_generation(source, plan)
        except GenerationError as e:
            return self.format_error(**e.as_kwargs())
        events_path = outcome.events_path
        accepted = outcome.accepted
        failed = outcome.failed
        truncated = outcome.truncated
        lhe_exhausted = outcome.lhe_exhausted
        bytes_written = outcome.bytes_written
        summary = outcome.summary
        weights = outcome.weights
        rivet_result = outcome.rivet

        # Cross-section information, in mb as Pythia reports it and in pb.
        xsec = cross_section_fields(source)
        events_name = os.path.basename(events_path)

        # Create manifest file.
        manifest = {
            "schema": MANIFEST_SCHEMA_VERSION,
            "created_utc": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
            "pythia_version": source.version,
            "beams": beams,
            "inputs": {
                "run_card": "run.cmnd",
                "generator": source.describe(),
                "n_events_requested": int(self.n_events),
                "seed": seed if seed != -1 else None,
                **({"select_pids": select_pids} if select_pids else {}),
                "output_format": fmt,
                "store": store,
                **(
                    {
                        "hepmc3_record": "full"
                        if store == "full"
                        else "final-state"
                    }
                    if fmt == "hepmc3"
                    else {}
                ),
                "precision": str(self.precision),
                "shower_lhe": bool(self.shower_lhe),
                **({"lhe_path": self.lhe_path} if self.lhe_path else {}),
            },
            "outputs": {
                **(
                    {
                        "events": events_name,
                        "events_schema": event_io.dataset_schema(fmt),
                        "events_bytes": int(bytes_written),
                    }
                    if not streaming
                    else {"rivet": rivet_result}
                ),
                "n_events_written": int(accepted),
                "n_events_failed": int(failed),
                "truncated": bool(truncated),
                **({"lhe_exhausted": True} if lhe_exhausted else {}),
                "pythia_log": "pythia.log",
                **({"xsec": xsec} if xsec else {}),
                **({"weights": weights} if weights else {}),
                **(
                    {"summary": summary.finalize(_particle_names())}
                    if accepted
                    else {}
                ),
            },
        }
        manifest_path = os.path.join(outdir, "manifest.json")

        # Write manifest file.
        try:
            with open(manifest_path, "w", encoding="utf-8") as mf:
                json.dump(manifest, mf, indent=2)
        except Exception as e:
            return self.format_error(
                error="Write Error",
                reason=str(e),
                context=f"path={manifest_path}",
                suggestion="Verify disk space and permissions",
            )

        # Create result object.
        result = {
            "status": "ok",
            "data_dir": os.path.relpath(outdir, self.base_directory),
            **(
                {"events": os.path.relpath(events_path, self.base_directory)}
                if not streaming
                else {"rivet": rivet_result}
            ),
            "format": fmt,
            "store": store,
            **(
                {"hepmc3_record": "full" if store == "full" else "final-state"}
                if fmt == "hepmc3"
                else {}
            ),
            "manifest_json": os.path.relpath(
                manifest_path, self.base_directory
            ),
            "accepted": int(accepted),
            "failed": int(failed),
            "backend": source.describe()["backend"],
            "pythia_log": os.path.relpath(log_path, self.base_directory),
        }
        if xsec:
            result["xsec"] = xsec
        # The summary is what an agent should read instead of the dataset,
        # which is far too large to open at any realistic event count.
        if accepted:
            result["summary"] = summary.finalize(
                _particle_names(), top_species=5
            )
        if accepted and not streaming:
            result["events_bytes"] = int(bytes_written)
            result["bytes_per_event"] = bytes_written / accepted

        # Truncation and weighting are independent and can both apply, so
        # warnings accumulate rather than overwrite.
        run_warnings: List[str] = list(getattr(source, "notes", []))

        if truncated:
            projected = (
                bytes_written / accepted * int(self.n_events)
                if accepted
                else 0
            )
            result["truncated"] = True
            run_warnings.append(
                f"Stopped after {accepted} of {self.n_events} events: "
                f"{events_name} "
                f"reached the {self.max_output_gb} GB limit, and the full "
                f"request "
                f"projected to ~{projected / 1e9:.1f} GB at "
                f"{bytes_written / accepted / 1024:.0f} KiB per event. The "
                f"events "
                "written so far are complete and usable. To generate the full "
                "sample, store less (store='finals+kinematics'), lower "
                "n_events, "
                "or raise max_output_gb deliberately."
            )

        if lhe_exhausted:
            result["lhe_exhausted"] = True
            run_warnings.append(
                f"The LHE file ran out after {accepted} of {self.n_events} "
                f"events. "
                "Every event it held was showered; generate a larger LHE "
                "sample "
                "for more."
            )

        if weights:
            result["weighted"] = weights["weighted"]
            if weights["weighted"]:
                # Surfaced in the tool result, not just the manifest: an
                # analysis that ignores these weights is wrong by whatever the
                # spread happens to be, and nothing downstream will complain.
                result["weight_range"] = [
                    weights["weight_min"],
                    weights["weight_max"],
                ]
                run_warnings.append(
                    "Events are WEIGHTED. Every histogram, efficiency and "
                    "yield built from "
                    + ("these events" if streaming else events_name)
                    + " must use the per-event 'weight' field; treating the "
                    "events as equally probable gives a wrong result."
                )
            if "names" in weights:
                result["weight_variations"] = weights["names"][1:]

        if streaming:
            result["next"] = (
                "RivetCompareTool compares the histograms with the measured "
                "data; "
                "RivetPlotTool draws them together"
            )
            # Surfaced once, with the run's other warnings; the manifest keeps
            # its own copy.
            run_warnings += result["rivet"].pop("warnings", [])
        if run_warnings:
            result["warnings"] = run_warnings
        return json.dumps(result, separators=(",", ":"), ensure_ascii=False)


class PythiaFromRunCardTool(_PythiaRunCardTool):
    __doc__ = _DOC_PYTHON

    pythia_path: str = StateField(
        default="",
        description="Pythia 8 install prefix whose lib/ holds the pythia8 "
        "extension of a build configured --with-python; PYTHIA_PATH when "
        "unset. Without one, the pythia8mc or pythia8 module on sys.path",
    )

    def _locate(self) -> Optional[str]:
        try:
            self._pythia = import_pythia(self.pythia_path)
        except ImportError as e:
            return self.format_error(
                error="Dependency Missing",
                reason=str(e),
                suggestion="Install pythia8mc in the toolkit environment, or "
                "use PythiaCPPFromRunCardTool with a Pythia 8 build named "
                "by pythia_path",
            )
        return None

    def _source(
        self,
        *,
        card: str,
        log_path: str,
        work_dir: str,
        select_pids: Optional[List[int]],
        lhe: bool,
    ) -> Any:
        pythia = self._pythia
        return _PythonSource(
            pythia.module,
            card,
            log_path,
            lhe=lhe,
            store=self.store,
            describe=pythia.describe(),
            notes=[pythia.note] if pythia.note else None,
        )


class PythiaCPPFromRunCardTool(_PythiaRunCardTool):
    __doc__ = _DOC_CPP

    pythia_path: str = StateField(
        default="",
        description="Pythia 8 install prefix (holds bin/pythia8-config); "
        "PYTHIA_PATH when unset",
    )

    def _locate(self) -> Optional[str]:
        try:
            self._install = resolve_install(
                "pythia", configured_path("pythia", self.pythia_path)
            )
        except BuildError as e:
            return self.format_error(
                error="Setup Error",
                reason=str(e),
                suggestion="Point pythia_path at a Pythia 8 build, or use "
                "PythiaFromRunCardTool with the pythia8mc wheel",
            )
        return None

    def _source(
        self,
        *,
        card: str,
        log_path: str,
        work_dir: str,
        select_pids: Optional[List[int]],
        lhe: bool,
    ) -> Any:
        return PythiaDriverSource(
            self._install,
            card=card,
            log_path=log_path,
            n_events=int(self.n_events),
            chunk=max(1, int(self.chunk_size)),
            store=self.store,
            select_pids=select_pids,
            work_dir=work_dir,
        )
