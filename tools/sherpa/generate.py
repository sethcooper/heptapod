"""
# generate.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Generating events with Sherpa 3 from a run card: SherpaFromRunCardTool,
through Sherpa's Python interface, and SherpaCPPFromRunCardTool, through a
C++ driver compiled against a Sherpa install.

Both write the same datasets as the Pythia tools, through the same loop
(tools.events.generation) and the same writers (tools.event_io): ROOT,
HepMC3 or JSONL under a store spec, with every event's weight and the run's
summary and manifest. What is Sherpa's here is the card, the beams read from
it, and the two sources, each a process of its own since Sherpa cannot be
initialised twice in one: the Python interface (the sherpa-mc wheel, or a
local build's module) in a worker, and the heptapod-sherpa driver compiled
against a Sherpa install -- a local build, or the wheel's own libraries --
which is the one that carries Sherpa's weight variations.
"""

import datetime
import json
import os
import subprocess
from typing import Any, ClassVar, Dict, List, Optional

from orchestral.tools.base.field_utils import RuntimeField, StateField
from orchestral.tools.base.tool import BaseTool
from pydantic import Field

from tools import event_io
from tools.events.generation import (
    GenerationError,
    GenerationPlan,
    RivetPlan,
    cross_section_fields,
    particle_names,
    run_generation,
)
from tools.cpp.build import BuildError, configured_path, resolve_install
from tools.sherpa.card import (
    CardError,
    has_variations,
    parse_card,
    prepare_card,
)
from tools.sherpa.backends import (
    DriverInitError,
    SherpaDriverSource,
    WorkerSource,
)
from tools.sherpa.record import beams_from_card
from tools.sherpa.runtime import (
    WHEEL_INSTALL,
    import_sherpa,
    log_tail,
    wheel_prefix,
)
from tools.sherpa.validate import REQUIRED

#: Schema of manifest.json. The dataset's own schema is recorded inside it,
#: as outputs.events_schema, since it depends on the output format.
MANIFEST_SCHEMA_VERSION = "sherpa-run-manifest-1.0"

_FORMATS = ("root", "hepmc3", "jsonl")


def _convert_ufo(prefix: str, ufo_path: str) -> None:
    """
    Convert a UFO model with the install's Sherpa-generate-model, so the
    model library lands in the install it will be loaded from. Done once:
    a converted model leaves a ``<ufo_path>.sherpa`` directory behind.
    """
    if os.path.isdir(ufo_path + ".sherpa"):
        return
    script = os.path.join(prefix, "bin", "Sherpa-generate-model")
    if not os.path.isfile(script):
        raise BuildError(
            f"{script} not found: this Sherpa was built without UFO support"
        )
    libdir = next(
        (
            d
            for d in (
                os.path.join(prefix, "lib", "SHERPA-MC"),
                os.path.join(prefix, "lib64", "SHERPA-MC"),
            )
            if os.path.isdir(d)
        ),
        os.path.join(prefix, "lib", "SHERPA-MC"),
    )
    proc = subprocess.run(
        [
            script,
            "--root_dir",
            os.path.join(prefix, "share", "SHERPA-MC"),
            "--install_dir",
            libdir,
            "--auto_convert",
            ufo_path,
        ],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    if proc.returncode != 0:
        raise BuildError(
            "Sherpa-generate-model failed:\n"
            + (proc.stdout + proc.stderr)[-3000:]
        )


# The two run-card tools share one body, so their descriptions are composed
# from the same pieces: what the run does, then how each reaches Sherpa.
_DOC_INPUTS = """
    Inputs (runtime):
      - data_dir: relative output directory under base_directory for the
        run's artifacts
      - card_path: relative path to the Sherpa .yaml run card
      - n_events: number of events to generate
      - seed: random seed; -1 (default) leaves it to the card or to Sherpa
      - output_format: "root" (default), "hepmc3" or "jsonl". ROOT is
        compact and read without parsing; HepMC3 is what Delphes and Rivet
        read, and with store="full" carries the vertex genealogy; JSONL is
        human-readable
      - store: "finals+kinematics" (default), "finals" or "full" -- the
        final-state momenta, the final state with vertices, or every
        particle with the blob it came from and went into
      - precision: "f32" (default) or "f64" for the stored kinematics
      - chunk_size: events generated per batch, bounding peak memory
        (default 10000)
      - select_pids: optional |PDG id| whitelist; only these species are
        stored
      - ufo_path: a UFO model directory to generate with; converted for the
        Sherpa in use on first use
      - gzip_output: write HepMC3 or JSONL gzipped
      - max_output_gb: stop early if the dataset would exceed this size
        (default 2.0; 0 or negative disables the guard)
      - rivet_analyses: stream the events into these Rivet analyses instead
        of writing a file (needs store="full" and rivet_path); rivet_output
        names the YODA file, rivet_ignore_beams runs mismatched analyses
      - base_directory: sandbox root for all file operations
      - sherpa_path: a Sherpa install prefix; see below
      - rivet_path: a Rivet install prefix, for rivet_analyses
"""

_DOC_RUN = """
    Behavior:
      1. Copy the card into the output directory as run.yaml, unchanged but
         for the lines a UFO model needs. The seed and the event count go to
         Sherpa on its command line.
      2. Initialise Sherpa from run.yaml, in the output directory, where it
         keeps its Process and Results directories; a second run there
         reuses the integration.
      3. Generate events in batches, reduce each to the store spec, fold it
         into the run summary and write it.
      4. Write the dataset as events.root, events.hepmc or events.jsonl,
         with a manifest recording the run, its beams, cross section and
         weight statistics.

    Output (JSON):
      {
        "status": "ok",
        "data_dir": "<relative output directory>",
        "events": "<relative path to the events file>",
        "format": "root" | "hepmc3" | "jsonl",
        "store": "<store spec applied>",
        "accepted": <int>,
        "failed": <int>,
        "backend": "<what ran Sherpa: sherpa-mc, sherpa or cpp>",
        "xsec": {"sigmaGen_pb": <float>, "sigmaErr_pb": <float>,
                 "sigmaGen_mb": <float>, "sigmaErr_mb": <float>,
                 "weightSum": <float>},
        "weighted": <bool>,
        "summary": {...},
        "warnings": ["..."]
      }

    Event weights:
      Sherpa's event weight is a cross section in pb: in Sherpa's default
      unweighted mode every event carries the same weight, the total cross
      section, and sum of weights over the trials is the cross section.
      "weighted" reports whether the weights vary, as they do in weighted
      and partially unweighted modes; then every histogram, efficiency and
      yield must use the per-event weight. Variations (SCALE_VARIATIONS,
      PDF_VARIATIONS, ...) are stored in "weights", named as Sherpa names
      them, by SherpaCPPFromRunCardTool; Sherpa's Python interface does not
      expose them.

    Errors:
      Returns BaseTool.format_error JSON on any failure including:
        - a card that is not valid YAML or lacks BEAMS, BEAM_ENERGIES or
          PROCESSES (ValidateSherpaRunCardTool explains what is wrong)
        - no Sherpa to run (see above)
        - initialisation or event generation failures, with Sherpa's own
          message from sherpa.log
        - file read/write or permission issues

    Notes:
      - All paths must remain inside base_directory for safety.
      - Output files are always written under data_dir: run.yaml,
        sherpa.log, manifest.json, the dataset, and Sherpa's Process and
        Results directories.
      - The cross section is reported in pb, as Sherpa gives it, and in mb
        beside it for comparison with Pythia.
"""

_DOC_PYTHON = (
    """
    Generate hadron-level events with Sherpa 3 from a YAML run card, through
    Sherpa's Python interface.
"""
    + _DOC_INPUTS
    + """
    Which Sherpa:
      Sherpa cannot be initialised twice in one process, so it runs in a
      worker started with this interpreter, through its Python interface:
      the sherpa-mc wheel, or the module of a Sherpa build configured with
      Python support when sherpa_path names that build (the module must
      have been built for the Python serving heptapod). The interface
      exposes the nominal weight only: a card with SCALE_VARIATIONS or
      PDF_VARIATIONS wants SherpaCPPFromRunCardTool, which takes the same
      inputs and writes the same dataset with the variations in it. The
      manifest records which Sherpa ran under inputs.generator.
"""
    + _DOC_RUN
)

_DOC_CPP = (
    """
    Generate hadron-level events with Sherpa 3 from a YAML run card, through
    a C++ driver compiled against a Sherpa install.
"""
    + _DOC_INPUTS
    + """
    Which Sherpa:
      The install named by sherpa_path (or SHERPA_PATH), the prefix holding
      bin/Sherpa-config, at whatever version it was built; without a path,
      the sherpa-mc wheel's own libraries and headers, when the wheel is
      installed. heptapod-sherpa, a small C++ driver
      (tools/sherpa/sherpa_driver.cc), is compiled against it once on first
      use (cached under ~/.cache/heptapod/cpp or $HEPTAPOD_CPP_CACHE) and
      runs Sherpa in a process of its own, streaming the event record back.
      It needs a C++ compiler on PATH (CXX overrides). It reads Sherpa's
      weights map, so SCALE_VARIATIONS and PDF_VARIATIONS arrive as named
      weights. The dataset is otherwise the one SherpaFromRunCardTool
      writes; the manifest records which Sherpa ran under inputs.generator.
"""
    + _DOC_RUN
)


class _SherpaRunCardTool(BaseTool):
    """The run shared by SherpaFromRunCardTool and SherpaCPPFromRunCardTool;
    a subclass supplies the source of events in _source."""

    #: Whether the source is Sherpa's Python interface, which drops the
    #: weight variations.
    python_interface: ClassVar[bool] = True

    # --------------------------- Runtime fields --------------------------- #
    data_dir: str = RuntimeField(
        description="Relative output directory for dataset, e.g. 'data/run001'"
    )
    card_path: str = RuntimeField(
        description="Relative path to the Sherpa .yaml run card"
    )
    n_events: int = RuntimeField(description="Number of events to generate")
    # Optional fields take non-None defaults: RuntimeField(default=None) is
    # read by the schema generator as "no default", which marks the field
    # required.
    seed: int = RuntimeField(
        default=-1,
        description="Random seed, passed to Sherpa as -R; -1 (default) "
        "leaves the seed to the card or to Sherpa",
    )
    output_format: str = RuntimeField(
        default="root",
        description="Dataset format: 'root' (default, compact and fast), "
        "'hepmc3' (for Delphes or Rivet; with store='full' it carries the "
        "vertex genealogy) or 'jsonl' (human-readable)",
    )
    store: str = RuntimeField(
        default=event_io.DEFAULT_STORE,
        description="What to store: 'finals+kinematics' (default: final-state "
        "momenta, ids and masses), 'finals' (final state with all fields, "
        "incl. vertices), or 'full' (every particle, with the blobs it came "
        "from and went into)",
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
        "particles whose |id| is listed (matched on absolute value, so 321 "
        "keeps both signs). Empty/unset stores all particles",
    )
    ufo_path: str = RuntimeField(
        default="",
        description="Optional UFO model directory (relative); leave unset for "
        "Standard Model runs",
    )
    gzip_output: bool = RuntimeField(
        default=False,
        description="Write HepMC3 or JSONL output gzipped (events.hepmc.gz, "
        "events.jsonl.gz); ROOT is already compressed",
    )
    max_output_gb: float = RuntimeField(
        default=2.0,
        description="Stop generating when the dataset would exceed this size "
        "in GB; 0 or negative disables the guard",
    )
    rivet_analyses: List[str] = Field(
        default_factory=list,
        json_schema_extra={"runtime": True},
        description="Run these Rivet analyses on the events as they are "
        "generated, streamed through a pipe, instead of writing an event "
        "file. Needs store='full' and a configured rivet_path; "
        "output_format is then HepMC3 regardless. Options go on the name "
        "(MC_ZINC:LMODE=MU)",
    )
    rivet_output: str = RuntimeField(
        default="",
        description="With rivet_analyses: the YODA file to write (default "
        "<data_dir>/rivet.yoda)",
    )
    rivet_ignore_beams: bool = RuntimeField(
        default=False,
        description="With rivet_analyses: run analyses whose beams or energy "
        "do not match the run, knowingly",
    )

    # ---------------------------- State fields ---------------------------- #
    base_directory: str = StateField(
        default=".", description="Base directory for safe path resolution"
    )
    sherpa_path: str = StateField(
        default="", description="Sherpa install prefix; SHERPA_PATH when unset"
    )
    rivet_path: str = StateField(
        default="",
        description="Rivet install prefix (holds bin/rivet-config), for "
        "rivet_analyses. RIVET_PATH when unset",
    )

    def _setup(self) -> None:
        self.base_directory = os.path.abspath(self.base_directory)

    def _safe_path(self, rel: str) -> Optional[str]:
        full = os.path.abspath(os.path.join(self.base_directory, rel))
        if not (full + os.sep).startswith(self.base_directory + os.sep):
            return None
        return full

    def _locate(self) -> Optional[str]:
        """Find the Sherpa this tool runs and set self._prefix to its install
        prefix; a format_error string when there is none."""
        raise NotImplementedError

    def _source(
        self,
        *,
        card: str,
        log_path: str,
        work_dir: str,
        store: str,
        select_pids: Optional[List[int]],
        seed: Optional[int],
        beams: List[Dict[str, float]],
    ) -> Any:
        """The source of events for this run, from what _locate found."""
        raise NotImplementedError

    def _run(self) -> str:
        for key in ("data_dir", "card_path", "n_events"):
            if getattr(self, key, None) in (None, ""):
                return self.format_error(
                    error="Missing Parameter",
                    reason=f"{key} is required",
                    suggestion="Provide required runtime fields",
                )
        outdir = self._safe_path(self.data_dir)
        card_src = self._safe_path(self.card_path)
        if not outdir or not card_src:
            return self.format_error(
                error="Access Denied",
                reason="Path escapes base_directory",
                context=f"data_dir={self.data_dir}, "
                f"card_path={self.card_path}",
                suggestion="Use paths inside the allowed base directory",
            )
        if not os.path.isfile(card_src):
            return self.format_error(
                error="File Not Found",
                reason="Run card does not exist",
                context=f"path={self.card_path}",
                suggestion="Provide a valid .yaml card path",
            )
        fmt = str(self.output_format).lower()
        if fmt not in _FORMATS:
            return self.format_error(
                error="Invalid Parameter",
                reason=f"output_format must be one of {', '.join(_FORMATS)}",
                context=f"output_format={self.output_format}",
            )
        store = str(self.store)
        if store not in event_io.STORE_SPECS:
            return self.format_error(
                error="Invalid Parameter",
                reason="unknown store spec",
                context=f"store={store}",
                suggestion=f"Use one of {', '.join(event_io.STORE_SPECS)}",
            )
        streaming = bool(self.rivet_analyses)
        if streaming:
            # Rivet's projections walk the decay history, so what goes down
            # the stream is the complete record whatever else was asked for.
            fmt = "hepmc3"
            if store != "full":
                return self.format_error(
                    error="Invalid Parameter",
                    reason="Rivet analyses need the complete event record",
                    context=f"store='{store}'",
                    suggestion="Pass store='full' with rivet_analyses: "
                    "Rivet's projections for prompt leptons, heavy-flavour "
                    "hadrons and unstable particles read the decay history, "
                    "which the final-state stores drop",
                )
        seed: Optional[int] = None if int(self.seed) == -1 else int(self.seed)

        # The Sherpa this tool runs.
        problem = self._locate()
        if problem:
            return problem
        prefix = self._prefix

        # The card: valid YAML with what every run needs.
        try:
            with open(card_src, "r", encoding="utf-8") as handle:
                card_text = handle.read()
        except OSError as e:
            return self.format_error(
                error="Read Error",
                reason=str(e),
                context=f"path={self.card_path}",
            )
        try:
            card = parse_card(card_text)
        except CardError as e:
            return self.format_error(
                error="Invalid Card",
                reason=str(e),
                context=f"line {e.line}"
                if e.line
                else f"path={self.card_path}",
                suggestion="ValidateSherpaRunCardTool reports every problem "
                "in the card with its line",
            )
        missing = [key for key in REQUIRED if key not in card]
        if missing:
            return self.format_error(
                error="Invalid Card",
                reason=f"the card does not set {', '.join(missing)}",
                suggestion="A Sherpa card needs BEAMS, BEAM_ENERGIES and a "
                "PROCESSES list; ValidateSherpaRunCardTool checks the rest",
            )

        # Check every writer dependency before generating.
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

        # The beams, from the card. Sherpa's own particle data cannot be
        # asked before Sherpa is initialised, so the masses come from the
        # table in tools.sherpa.record.
        try:
            beams = beams_from_card(card)
        except (ValueError, TypeError) as e:
            return self.format_error(
                error="Invalid Card",
                reason=str(e),
                suggestion="BEAMS is one PDG id or a pair, BEAM_ENERGIES one "
                "energy in GeV or a pair",
            )

        # The run directory: the card as run.yaml, Sherpa's log beside it.
        os.makedirs(outdir, exist_ok=True)
        card_dst = os.path.join(outdir, "run.yaml")
        log_path = os.path.join(outdir, "sherpa.log")
        select_pids = (
            sorted({abs(int(pid)) for pid in self.select_pids}) or None
        )

        source = self._source(
            card=card_dst,
            log_path=log_path,
            work_dir=outdir,
            store=store,
            select_pids=select_pids,
            seed=seed,
            beams=beams,
        )

        ufo_model = None
        if self.ufo_path:
            ufo_dir = self._safe_path(self.ufo_path)
            if not ufo_dir or not os.path.isdir(ufo_dir):
                return self.format_error(
                    error="File Not Found",
                    reason="ufo_path is not a directory inside base_directory",
                    context=f"ufo_path={self.ufo_path}",
                )
            try:
                _convert_ufo(prefix or "", ufo_dir)
            except BuildError as e:
                return self.format_error(
                    error="UFO Conversion Failed", reason=str(e)
                )
            ufo_model = ufo_dir
        try:
            with open(card_dst, "w", encoding="utf-8") as handle:
                handle.write(prepare_card(card_text, ufo_model=ufo_model))
        except OSError as e:
            return self.format_error(
                error="Write Error",
                reason=str(e),
                context=f"dst={self.data_dir}/run.yaml",
                suggestion="Verify permissions and disk space",
            )

        try:
            init_ok = source.start()
        except DriverInitError:
            init_ok = False
        except BuildError as e:
            return self.format_error(
                error="C++ Build Failed",
                reason=str(e),
                suggestion="Check that sherpa_path is a complete Sherpa "
                "install and a C++ compiler is on PATH",
            )
        if not init_ok:
            return self.format_error(
                error="Sherpa Init Failed",
                reason="Initialization failed",
                context=log_tail(log_path)
                or f"see {os.path.relpath(log_path, self.base_directory)}",
                suggestion="Fix what Sherpa reports above; "
                "ValidateSherpaRunCardTool(dry_run_init=True) checks a card "
                "before a run",
            )

        suffix = {"root": ".root", "jsonl": ".jsonl", "hepmc3": ".hepmc"}[fmt]
        if self.gzip_output and fmt != "root":
            suffix += ".gz"
        events_path = os.path.join(outdir, "events" + suffix)
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
            generator="Sherpa",
            log_name=os.path.relpath(log_path, self.base_directory),
            diagnostics=lambda: log_tail(log_path),
        )
        try:
            outcome = run_generation(source, plan)
        except GenerationError as e:
            return self.format_error(**e.as_kwargs())
        events_path = outcome.events_path
        accepted = outcome.accepted
        xsec = cross_section_fields(source)
        weights = outcome.weights
        events_name = os.path.basename(events_path)

        manifest = {
            "schema": MANIFEST_SCHEMA_VERSION,
            "created_utc": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
            "sherpa_version": source.version,
            "beams": beams,
            "inputs": {
                "run_card": "run.yaml",
                "generator": source.describe(),
                "n_events_requested": int(self.n_events),
                "seed": seed,
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
                **({"ufo_path": self.ufo_path} if self.ufo_path else {}),
            },
            "outputs": {
                **(
                    {
                        "events": events_name,
                        "events_schema": event_io.dataset_schema(fmt),
                        "events_bytes": int(outcome.bytes_written),
                    }
                    if not streaming
                    else {"rivet": outcome.rivet}
                ),
                "n_events_written": int(accepted),
                "n_events_failed": int(outcome.failed),
                "truncated": bool(outcome.truncated),
                "sherpa_log": "sherpa.log",
                **({"xsec": xsec} if xsec else {}),
                **({"weights": weights} if weights else {}),
                **(
                    {"summary": outcome.summary.finalize(particle_names())}
                    if accepted
                    else {}
                ),
            },
        }
        manifest_path = os.path.join(outdir, "manifest.json")
        try:
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2)
        except OSError as e:
            return self.format_error(
                error="Write Error",
                reason=str(e),
                context=f"path={manifest_path}",
                suggestion="Verify disk space and permissions",
            )

        result: Dict[str, Any] = {
            "status": "ok",
            "data_dir": os.path.relpath(outdir, self.base_directory),
            **(
                {"events": os.path.relpath(events_path, self.base_directory)}
                if not streaming
                else {"rivet": dict(outcome.rivet)}
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
            "failed": int(outcome.failed),
            "backend": source.describe()["backend"],
            "sherpa_log": os.path.relpath(log_path, self.base_directory),
        }
        if xsec:
            result["xsec"] = xsec
        if accepted:
            result["summary"] = outcome.summary.finalize(
                particle_names(), top_species=5
            )
        if accepted and not streaming:
            result["events_bytes"] = int(outcome.bytes_written)
            result["bytes_per_event"] = outcome.bytes_written / accepted

        run_warnings: List[str] = []
        if outcome.truncated:
            run_warnings.append(
                f"Output truncated at {accepted} events by max_output_gb="
                f"{self.max_output_gb}; the events written are complete"
            )
        if weights:
            result["weighted"] = weights["weighted"]
            if weights["weighted"]:
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
        if (
            self.python_interface
            and has_variations(card)
            and len(source.weight_names()) <= 1
        ):
            run_warnings.append(
                "The card asks for weight variations, which Sherpa's Python "
                "interface does not expose; only the nominal weight is "
                "stored. Run the card with SherpaCPPFromRunCardTool, whose "
                "driver reads the weights map."
            )
        if streaming:
            result["next"] = (
                "RivetCompareTool compares the histograms with the measured "
                "data; RivetPlotTool draws them together"
            )
            run_warnings += result["rivet"].pop("warnings", [])
        if run_warnings:
            result["warnings"] = run_warnings
        return json.dumps(result, separators=(",", ":"), ensure_ascii=False)


class SherpaFromRunCardTool(_SherpaRunCardTool):
    __doc__ = _DOC_PYTHON

    python_interface: ClassVar[bool] = True
    sherpa_path: str = StateField(
        default="",
        description="Sherpa install prefix whose lib/python3.*/site-packages "
        "holds the Sherpa module of a build with Python support; "
        "SHERPA_PATH when unset. Without one, the sherpa-mc wheel",
    )

    def _locate(self) -> Optional[str]:
        try:
            self._sherpa = import_sherpa(self.sherpa_path)
        except ImportError as e:
            return self.format_error(
                error="Dependency Missing",
                reason=str(e),
                suggestion="Install sherpa-mc in the toolkit environment, or "
                "use SherpaCPPFromRunCardTool with a Sherpa install named "
                "by sherpa_path",
            )
        self._prefix = self._sherpa.prefix
        return None

    def _source(
        self,
        *,
        card: str,
        log_path: str,
        work_dir: str,
        store: str,
        select_pids: Optional[List[int]],
        seed: Optional[int],
        beams: List[Dict[str, float]],
    ) -> Any:
        return WorkerSource(
            self._sherpa,
            card=card,
            log_path=log_path,
            n_events=int(self.n_events),
            chunk=max(1, int(self.chunk_size)),
            store=store,
            select_pids=select_pids,
            seed=seed,
            work_dir=work_dir,
            beams=beams,
        )


class SherpaCPPFromRunCardTool(_SherpaRunCardTool):
    __doc__ = _DOC_CPP

    python_interface: ClassVar[bool] = False
    sherpa_path: str = StateField(
        default="",
        description="Sherpa install prefix (holds bin/Sherpa-config); "
        "SHERPA_PATH when unset, else the sherpa-mc wheel's own libraries",
    )

    def _locate(self) -> Optional[str]:
        path = configured_path("sherpa", self.sherpa_path) or wheel_prefix()
        if not path:
            return self.format_error(
                error="Setup Error",
                reason="sherpa_path is not configured and the sherpa-mc wheel "
                "is not installed, so there is no Sherpa to build against",
                suggestion="Point sherpa_path at a Sherpa install prefix "
                "(`tb config set heptapod sherpa_path <prefix>`), or install "
                f"the wheel with `{WHEEL_INSTALL}`",
            )
        try:
            self._install = resolve_install("sherpa", path)
        except BuildError as e:
            return self.format_error(error="Setup Error", reason=str(e))
        self._prefix = self._install.prefix
        return None

    def _source(
        self,
        *,
        card: str,
        log_path: str,
        work_dir: str,
        store: str,
        select_pids: Optional[List[int]],
        seed: Optional[int],
        beams: List[Dict[str, float]],
    ) -> Any:
        return SherpaDriverSource(
            self._install,
            card=card,
            log_path=log_path,
            n_events=int(self.n_events),
            chunk=max(1, int(self.chunk_size)),
            store=store,
            select_pids=select_pids,
            seed=seed,
            work_dir=work_dir,
            beams=beams,
        )
