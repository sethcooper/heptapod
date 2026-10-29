"""
# generation.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

The generation loop shared by the event generator tools.

A run-card tool (PythiaFromRunCardTool, SherpaFromRunCardTool) owns what is
particular to its generator: the card, the backends, the manifest. What
happens between the first event and the last is the same for both and lives
here. Events come from a source in batches; each batch is reduced to the
store spec, folded into the running summary and weight statistics, and
written to a ROOT, HepMC3 or JSONL dataset, or into Rivet through a pipe,
under a size guard, until the request is met or the generator runs dry.

The source interface
--------------------
A source yields batches as the Awkward Arrays ``Pythia.nextBatch`` produces
(a ``prt`` record of particles and an ``info`` record of per-event scalars
with a ragged ``weights``), whatever generator stands behind it::

    start() -> bool             initialise; False when the generator refused
    next_batch(want) -> array   up to `want` events; empty once the run is over
    weight_names() -> [str]     nominal first
    cross_section() -> dict     the generator's own figures, for the manifest
    cross_section_pb() -> dict  sigma_pb, sigma_err_pb, n_accepted, n_tried
    finish()                    end the run; the generator's own summary
    abort()                     stop after a failure in the tool
    describe() -> dict          what ran, for the manifest ("backend", ...)
    beams                       the two beams as {id, px, py, pz, e, m}
    version                     the generator's version
    failed                      events the generator failed to produce so far
    lhe_exhausted               an input event file ran out
"""

import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from tqdm import tqdm

from tools import event_io

#: HepMC3 and Rivet work in pb; Pythia reports mb.
PB_PER_MB = 1.0e9

#: Progress bars go to stderr, on one line, so they never touch the MCP
#: transport on stdout.
TQDM_CONFIG = {
    "ncols": 80,
    "leave": True,
    "dynamic_ncols": False,
    "mininterval": 0.1,
    "ascii": False,
}


class GenerationError(Exception):
    """A failure the tool reports through ``format_error``."""

    def __init__(
        self, error: str, reason: str, context: str = "", suggestion: str = ""
    ) -> None:
        super().__init__(f"{error}: {reason}")
        self.error = error
        self.reason = reason
        self.context = context
        self.suggestion = suggestion

    def as_kwargs(self) -> Dict[str, str]:
        """The arguments for ``BaseTool.format_error``."""
        out = {"error": self.error, "reason": self.reason}
        if self.context:
            out["context"] = self.context
        if self.suggestion:
            out["suggestion"] = self.suggestion
        return out


@dataclass
class RivetPlan:
    """Run Rivet analyses on the events as they are generated."""

    analyses: List[str]
    output_yoda: str
    install: Any
    catalogue: Any
    ignore_beams: bool = False
    timeout_s: int = 3600


@dataclass
class GenerationPlan:
    """What a run should produce, independent of the generator."""

    events_path: str
    fmt: str
    store: str
    precision: str
    requested: int
    chunk: int
    select_pids: Optional[List[int]] = None
    max_bytes: int = 0
    rivet: Optional[RivetPlan] = None
    base_directory: str = "."
    generator: str = "the generator"
    log_name: str = "the generator's log"
    #: Called for the context of a generation error: the tail of the
    #: generator's log, typically.
    diagnostics: Optional[Callable[[], str]] = None


@dataclass
class GenerationOutcome:
    """What a run produced."""

    events_path: str
    accepted: int = 0
    failed: int = 0
    truncated: bool = False
    lhe_exhausted: bool = False
    bytes_written: int = 0
    summary: event_io.SummaryAccumulator = field(
        default_factory=event_io.SummaryAccumulator
    )
    weights: Dict[str, Any] = field(default_factory=dict)
    rivet: Dict[str, Any] = field(default_factory=dict)


def rivet_run_info(beams: List[Dict[str, float]]) -> Dict[str, object]:
    """The beams, sqrt(s) and bookkeeping of a run, as the Rivet checks read
    them."""
    from tools.rivet.catalogue import beam_name

    a, b = beams
    s = (
        (a["e"] + b["e"]) ** 2
        - (a["px"] + b["px"]) ** 2
        - (a["py"] + b["py"]) ** 2
        - (a["pz"] + b["pz"]) ** 2
    )
    return {
        "beams": [beam_name(int(a["id"])), beam_name(int(b["id"]))],
        "beam_ids": [int(a["id"]), int(b["id"])],
        "sqrt_s_gev": math.sqrt(max(s, 0.0)),
        # The stream records the running cross section with every event.
        "cross_section_pb": True,
    }


def cross_section_fields(source: Any) -> Dict[str, float]:
    """
    The run's cross section for the manifest and result, in mb and pb.

    Pythia reports mb and Sherpa pb; both are given so a comparison between
    generators needs no conversion, and ``weightSum`` is what a weighted
    sample is normalised by.
    """
    out: Dict[str, float] = {}
    pb = source.cross_section_pb()
    if "sigma_pb" in pb:
        out["sigmaGen_mb"] = float(pb["sigma_pb"]) / PB_PER_MB
        out["sigmaErr_mb"] = float(pb.get("sigma_err_pb", 0.0)) / PB_PER_MB
        out["sigmaGen_pb"] = float(pb["sigma_pb"])
        out["sigmaErr_pb"] = float(pb.get("sigma_err_pb", 0.0))
    native = source.cross_section()
    if "weightSum" in native:
        out["weightSum"] = float(native["weightSum"])
    return out


def cross_section_pb_from_mb(native: Dict[str, float]) -> Dict[str, float]:
    """A Pythia-style cross section (sigmaGen_mb, sigmaErr_mb, nAccepted,
    nTried) as the pb figures the loop hands to HepMC3 and Rivet."""
    if "sigmaGen_mb" not in native:
        return {}
    return {
        "sigma_pb": float(native["sigmaGen_mb"]) * PB_PER_MB,
        "sigma_err_pb": float(native.get("sigmaErr_mb", 0.0)) * PB_PER_MB,
        "n_accepted": int(native.get("nAccepted", -1)),
        "n_tried": int(native.get("nTried", -1)),
    }


def particle_names() -> Dict[str, str]:
    """PDG id (as a string) to name, from the Pythia settings snapshot the
    repository ships; empty if unavailable."""
    try:
        from tools.pythia.settings_db import load_snapshot

        return load_snapshot().get("particles", {})
    except Exception:
        return {}


def _rel(path: str, base: str) -> str:
    return os.path.relpath(path, base)


def run_generation(source: Any, plan: GenerationPlan) -> GenerationOutcome:
    """
    Generate events from a started source according to the plan.

    The source has been started (``start()`` returned True) so its beams are
    known. On return the source is finished and the dataset, or the Rivet
    output, is complete.

    Raises:
        GenerationError: for every failure, with the message the tool shows
    """
    base = plan.base_directory
    events_path = plan.events_path
    outcome = GenerationOutcome(events_path=events_path)
    rivet_notes: List[str] = []

    stream = None
    if plan.rivet is not None:
        from tools.rivet.run import AnalysisCheckError, check_analyses
        from tools.rivet.stream import RivetStream, RivetStreamError

        try:
            _, rivet_notes = check_analyses(
                plan.rivet.catalogue,
                plan.rivet.analyses,
                [],
                rivet_run_info(source.beams),
                ignore_beams=plan.rivet.ignore_beams,
            )
        except AnalysisCheckError as e:
            source.abort()
            raise GenerationError(
                e.error, e.reason, e.context, e.suggestion
            ) from e
        stream = RivetStream(
            plan.rivet.install,
            plan.rivet.analyses,
            output_yoda=plan.rivet.output_yoda,
            work_dir=os.path.dirname(events_path),
            ignore_beams=plan.rivet.ignore_beams,
            timeout_s=plan.rivet.timeout_s,
        )
        try:
            stream.start()
        except RivetStreamError as e:
            source.abort()
            logs = ", ".join(
                _rel(p, base) for p in (stream.stdout_log, stream.stderr_log)
            )
            raise GenerationError(
                "Rivet Failed To Start",
                str(e),
                f"logs: {logs}",
                "RivetFindAnalysesTool lists the analyses and their names",
            ) from e
        events_path = stream.fifo_path
        outcome.events_path = events_path

    accepted = 0
    bytes_written = 0
    truncated = False
    lhe_exhausted = False
    summary = outcome.summary
    weights_seen = event_io.WeightAccumulator()

    # Generate events in batches and stream them to disk. Batching bounds
    # peak memory: a chunk is materialised, written and released, so a
    # sample far larger than memory is possible.
    try:
        requested = int(plan.requested)
        chunk = max(1, int(plan.chunk))
        with (
            event_io.EventWriter(
                events_path,
                store=plan.store,
                precision=plan.precision,
                fmt=plan.fmt,
                beams=source.beams,
            ) as writer,
            tqdm(
                total=requested,
                desc="Generating events",
                unit="evt",
                file=__import__("sys").stderr,
                **TQDM_CONFIG,
            ) as progress,
        ):
            while accepted < requested and not truncated:
                want = min(chunk, requested - accepted)
                batch = source.next_batch(want)
                produced = len(batch)
                if source.lhe_exhausted:
                    lhe_exhausted = True
                if not produced:
                    # Nothing came back, so nothing will; stop rather than
                    # spin on a generator that cannot make events.
                    break

                # Reduce once and use the result for both the summary and
                # the file, so the store spec is applied a single time and
                # the summary describes what was actually written.
                reduced = event_io.apply_store(
                    batch,
                    store=plan.store,
                    precision=plan.precision,
                    select_pids=plan.select_pids,
                )
                summary.add(reduced)
                weights_seen.add(reduced)
                if plan.fmt == "hepmc3":
                    # HepMC3 records the weight names per run and the cross
                    # section per event, as a running estimate.
                    if not accepted:
                        writer.set_weight_names(
                            source.weight_names() or ["Weight"]
                        )
                    running = source.cross_section_pb()
                    if "sigma_pb" in running:
                        writer.set_cross_section(
                            float(running["sigma_pb"]),
                            float(running.get("sigma_err_pb", 0.0)),
                            int(running.get("n_accepted", -1)),
                            int(running.get("n_tried", -1)),
                        )
                writer.write_reduced(reduced)
                accepted += produced
                progress.update(produced)
                if stream is not None:
                    # A writer to a pipe nobody reads blocks or fails
                    # silently; stop as soon as Rivet is gone.
                    stream.check()
                if lhe_exhausted:
                    break
                if stream is not None:
                    continue

                # Size guard. A hadron-level event costs several KiB, so a
                # request for a million silently asks for gigabytes. The cost
                # per event is only knowable from real events, so it is
                # checked as they are written; what has been generated
                # already is kept rather than discarded.
                bytes_written = (
                    os.path.getsize(events_path)
                    if os.path.exists(events_path)
                    else 0
                )
                if plan.max_bytes and bytes_written > plan.max_bytes:
                    truncated = True
        # The writer buffers, so the size on disk is only final once closed.
        bytes_written = writer.stats["bytes"]
    except OSError as e:
        source.abort()
        if stream is not None and not stream.alive():
            # The pipe broke because its reader went away: Rivet's failure
            # is the story, not the write.
            message = stream.failure_message()
            stream.close()
            raise GenerationError(
                "Rivet Stopped",
                message,
                f"after {accepted} events; logs: "
                + _rel(stream.stderr_log, base),
                "Read Rivet's log; an analysis may have failed on an event",
            ) from e
        if stream is not None:
            stream.close()
        raise GenerationError(
            "Write Error",
            str(e),
            f"path={events_path}",
            "Verify disk space and permissions",
        ) from e
    except Exception as e:
        source.abort()
        if stream is not None and type(e).__name__ == "RivetStreamError":
            stream.close()
            raise GenerationError(
                "Rivet Stopped",
                str(e),
                f"after {accepted} events; logs: "
                + _rel(stream.stderr_log, base),
                "Read Rivet's log; an analysis may have failed on an event",
            ) from e
        if stream is not None:
            stream.close()
        detail = plan.diagnostics() if plan.diagnostics else ""
        raise GenerationError(
            "Generation Error",
            f"{type(e).__name__}: {e}",
            detail or f"see {plan.log_name}",
            f"Check {plan.log_name} for the {plan.generator} diagnostics",
        ) from e

    # The pipe is closed: Rivet finalises its histograms and exits.
    if stream is not None:
        outcome.rivet = _finish_rivet(stream, plan, rivet_notes)

    # Close the run; the generator's own summary lands in its log.
    source.finish()

    outcome.accepted = int(accepted)
    outcome.failed = int(source.failed)
    outcome.truncated = bool(truncated)
    outcome.lhe_exhausted = bool(lhe_exhausted)
    outcome.bytes_written = int(bytes_written)
    weights = weights_seen.finalize()
    names = source.weight_names()
    if weights and len(names) > 1:
        weights["names"] = names
    outcome.weights = weights
    return outcome


def _finish_rivet(
    stream: Any, plan: GenerationPlan, notes: List[str]
) -> Dict[str, Any]:
    """Wait for Rivet, read its YODA file and describe what it filled."""
    from tools.rivet import yoda_io
    from tools.rivet.install import RivetError
    from tools.rivet.run import output_warnings, summarise_output
    from tools.rivet.stream import RivetStreamError

    assert plan.rivet is not None
    base = plan.base_directory
    try:
        outcome = stream.finish()
        rivet_summary = summarise_output(
            yoda_io.read_yoda(plan.rivet.install, plan.rivet.output_yoda)
        )
    except (RivetStreamError, RivetError) as e:
        raise GenerationError(
            "Rivet Failed",
            str(e),
            "logs: " + _rel(stream.stderr_log, base),
            "Read Rivet's log for the analysis that failed",
        ) from e
    return {
        "output_yoda": _rel(plan.rivet.output_yoda, base),
        "analyses": {
            name: {
                "histograms": entry["histograms"],
                "filled": entry["filled"],
                "raw_entries": entry["raw_entries"],
                **({"empty": entry["empty"]} if entry["empty"] else {}),
            }
            for name, entry in rivet_summary["analyses"].items()
        },
        "n_events": rivet_summary["n_events"],
        "weight_variations": rivet_summary["weight_variations"],
        "rivet_version": plan.rivet.install.version,
        "elapsed_s": outcome["elapsed_s"],
        "logs": {k: _rel(v, base) for k, v in outcome["logs"].items()},
        "warnings": notes + output_warnings(rivet_summary, outcome["log"]),
    }
