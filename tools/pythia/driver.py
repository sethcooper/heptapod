"""
# driver.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

A Pythia 8 install as a source of events for PythiaCPPFromRunCardTool.

PythiaFromRunCardTool calls Pythia.nextBatch through the Python interface
and gets an Awkward Array back. The C++ tool runs heptapod-pythia, a small
driver compiled against the configured install (tools/pythia/pythia_driver.cc),
which streams the event record to its stdout one JSON line per event. This
module reads that stream and rebuilds the array nextBatch would have
produced, field for field, so everything downstream of generation -- the
store spec, the run summary, the weight statistics, and the ROOT, HepMC3 and
JSONL writers -- is the same code for both tools.

The driver needs nothing of the install but Pythia itself: no HepMC3, no
ROOT, no Python bindings. It is built once per install into the C++ cache.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from tools.events.generation import cross_section_pb_from_mb
from tools.cpp.build import Install
from tools.cpp.source import DriverInitError, DriverSource  # noqa: F401

#: The driver source beside this module, and the name it is cached under.
DRIVER_SOURCE = Path(__file__).with_name("pythia_driver.cc")
DRIVER_NAME = "heptapod-pythia"

#: Order of the per-event scalars on an event line ("i"), matching the
#: `info` record of nextBatch.
INFO_FIELDS = (
    "id1",
    "id2",
    "x1",
    "x2",
    "pdf1",
    "pdf2",
    "alphaS",
    "alphaEM",
    "Q2Fac",
    "Q2Ren",
    "mHat",
    "sHat",
    "tHat",
    "uHat",
    "pT2Hat",
)
_INFO_INTS = ("id1", "id2")

#: Order of the values in a particle row ("p"), for the two row layouts the
#: driver writes: every field, or the kinematic subset.
FULL_ROW = (
    "id",
    "status",
    "mother1",
    "mother2",
    "daughter1",
    "daughter2",
    "col",
    "acol",
    "px",
    "py",
    "pz",
    "e",
    "m",
    "scale",
    "pol",
    "tau",
    "vx",
    "vy",
    "vz",
    "vt",
)
KINEMATIC_ROW = ("id", "status", "px", "py", "pz", "e", "m")
_INT_FIELDS = (
    "id",
    "status",
    "mother1",
    "mother2",
    "daughter1",
    "daughter2",
    "col",
    "acol",
)


def batch_from_events(events: Sequence[Dict[str, Any]]) -> Any:
    """
    The Awkward Array Pythia.nextBatch would have returned for these events.

    Parameters:
        events: Parsed event lines from the driver, each with "p" (particle
            rows), "i" (per-event scalars) and "w" (weights, nominal first)

    Returns:
        An array with a `prt` field of particle records -- `p` and `vProd`
        nested as Pythia nests them -- and an `info` record with the scalars
        and a ragged `weights`, so `event_io.apply_store` treats it exactly as
        it treats a batch from the pip wheel
    """
    import awkward as ak
    import numpy as np

    counts = np.fromiter(
        (len(ev["p"]) for ev in events), dtype=np.int64, count=len(events)
    )
    rows = [row for ev in events for row in ev["p"]]
    width = len(rows[0]) if rows else len(KINEMATIC_ROW)
    layout = FULL_ROW if width == len(FULL_ROW) else KINEMATIC_ROW
    flat = (
        np.asarray(rows, dtype=np.float64)
        if rows
        else np.zeros((0, len(layout)), dtype=np.float64)
    )
    column = {name: flat[:, k] for k, name in enumerate(layout)}
    for name in _INT_FIELDS:
        if name in column:
            column[name] = column[name].astype(np.int32)

    fields: Dict[str, Any] = {
        name: column[name]
        for name in layout
        if name not in ("px", "py", "pz", "e", "vx", "vy", "vz", "vt")
    }
    fields["p"] = ak.zip(
        {axis: column[axis] for axis in ("px", "py", "pz", "e")}
    )
    if "vx" in column:
        fields["vProd"] = ak.zip(
            {
                "px": column["vx"],
                "py": column["vy"],
                "pz": column["vz"],
                "e": column["vt"],
            }
        )
    # Pythia's field order, so the record prints and compares the same way.
    order = [
        name
        for name in (
            "id",
            "status",
            "mother1",
            "mother2",
            "daughter1",
            "daughter2",
            "col",
            "acol",
            "p",
            "m",
            "scale",
            "pol",
            "vProd",
            "tau",
        )
        if name in fields
    ]
    particles = ak.unflatten(
        ak.zip({name: fields[name] for name in order}, depth_limit=1), counts
    )

    scalars = (
        np.asarray([ev["i"] for ev in events], dtype=np.float64)
        if events
        else np.zeros((0, len(INFO_FIELDS)), dtype=np.float64)
    )
    info: Dict[str, Any] = {}
    for k, name in enumerate(INFO_FIELDS):
        values = scalars[:, k]
        info[name] = values.astype(np.int32) if name in _INFO_INTS else values
    weights = [ev["w"] for ev in events]
    info["weights"] = ak.unflatten(
        np.fromiter((w for ws in weights for w in ws), dtype=np.float64),
        np.fromiter(
            (len(ws) for ws in weights), dtype=np.int64, count=len(weights)
        ),
    )
    return ak.Array({"prt": particles, "info": ak.zip(info, depth_limit=1)})


class PythiaDriverSource(DriverSource):
    """
    Events from heptapod-pythia, consumed the way the generation loop in
    generate.py consumes Pythia.nextBatch. The protocol and its reader are
    shared with the Sherpa driver (tools.cpp.source.DriverSource); what is
    Pythia's here is the command line and the row layout.
    """

    def __init__(
        self,
        install: Install,
        *,
        card: str,
        log_path: str,
        n_events: int,
        chunk: int,
        store: str,
        select_pids: Optional[Sequence[int]],
        work_dir: str,
    ) -> None:
        arguments = [
            "--card",
            card,
            "--log",
            log_path,
            "--nevents",
            str(int(n_events)),
            "--chunk",
            str(max(1, int(chunk))),
            "--store",
            store,
        ]
        if select_pids:
            arguments += [
                "--select-pids",
                ",".join(str(abs(int(p))) for p in select_pids),
            ]
        if install.env.get("PYTHIA8DATA"):
            arguments += ["--xmldoc", install.env["PYTHIA8DATA"]]
        super().__init__(
            install,
            driver_source=DRIVER_SOURCE,
            driver_name=DRIVER_NAME,
            arguments=arguments,
            log_path=log_path,
            work_dir=work_dir,
            batch_builder=batch_from_events,
        )

    def cross_section_pb(self) -> Dict[str, float]:
        """The running estimate in pb, as HepMC3 and Rivet take it."""
        return cross_section_pb_from_mb(self.cross_section())
