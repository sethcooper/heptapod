"""
# event_io.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Event dataset input/output.

Every heptapod tool that reads or writes an event dataset goes through this
module, so the on-disk format is a parameter rather than something baked into
each tool.

Three formats are supported, for three different jobs:

  ROOT   the default for generated samples. Written with uproot, which needs no
         ROOT installation, and read by anything in the HEP ecosystem. Roughly
         7 KiB per hadron-level event.

  JSONL  the legacy schema (``evtjsonl-1.0``), one JSON object per event.
         Around nine times larger and five times slower to read, but
         human-readable and dependency-free, which makes it the right choice
         for the small samples someone actually opens and inspects.

  HepMC3 the interchange format of generator, detector-simulation and
         analysis programs (Delphes, Rivet), written as HepMC3 ASCII with
         the event weights, their names and the cross section. What it
         holds depends on the record written. A final-state record (the
         "finals" store specs) becomes two beam particles (status 4) and
         the stored particles as their direct products (status 1): all
         Delphes needs. A complete record (the "full" store spec) becomes
         the vertex genealogy Pythia's own converter writes -- the same
         vertices, HepMC status codes and production positions as
         ``Pythia8ToHepMC3`` -- which is what Rivet's decay-history
         projections (prompt leptons, heavy-flavour hadrons, unstable
         particles) need. Reading keeps the final-state (status 1)
         particles of any HepMC3 file.

The ROOT output is a TTree rather than an RNTuple, though uproot writes either.
RNTuple is about 6% smaller, but uproot cannot extend one, and a format that
cannot be appended to forces the whole sample to be held in memory before
writing -- which is the scalability problem this module exists to avoid. A
TTree is also readable by any ROOT version, whereas RNTuple is still settling.

The in-memory representation throughout is an Awkward Array, which is what
``Pythia.nextBatch`` produces natively. The stored layout is:

    prt      var * {id, status, px, py, pz, e, m, ...}   per-particle, ragged
    weight   float                                        nominal event weight
    weights  var * float                                  variation weights
    info_*   float                                        hard-process scalars

``weight`` is always written, whatever the store spec, because a weighted
sample analysed as unweighted is wrong in shape rather than in normalisation.

Note the module is named ``event_io`` rather than ``io`` so that it cannot
shadow the standard library module of that name.
"""

import itertools
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

# Schema written into JSONL rows. Kept at 1.0 because the additions since
# (the per-event weight) are optional fields that older readers ignore, and
# every reader in the toolkit checks structure rather than this string.
JSONL_SCHEMA_VERSION = "evtjsonl-1.0"

# Name of the TTree written into ROOT files.
ROOT_TREE_NAME = "Events"

# Schema of the ROOT layout: an "Events" TTree with the `prt_*` particle
# branches, `weight` and optional `weights`, described in the module docstring.
ROOT_SCHEMA_VERSION = "heptapod-events-root-1.0"

# HepMC3's own ASCII format; the identifier is the one its files declare.
HEPMC3_SCHEMA_VERSION = "HepMC3-Asciiv3"

FORMATS = ("root", "jsonl", "hepmc3")

# HepMC3 particle status codes: a final-state particle, a hadron or lepton
# that decayed, and an incoming beam. Any other Pythia status is written as
# its positive counterpart, which is Pythia's own convention.
_HEPMC_FINAL = 1
_HEPMC_DECAYED = 2
_HEPMC_BEAM = 4

# The particle fields a record needs for its genealogy to be written to HepMC3.
_GENEALOGY_FIELDS = ("status", "mother1", "mother2", "daughter1")
#: The other way a record can carry its genealogy: every particle names the
#: vertex it came from and the vertex it went into (-1 for none), with the
#: HepMC status itself stored negative for particles that did not reach the
#: final state. This is what a generator with explicit vertices (Sherpa's
#: blobs) gives.
_VERTEX_FIELDS = ("status", "prod_vertex", "end_vertex")
_HEPMC_INTERMEDIATE = 11

# What an HepMC3 writer may hold: the vertex genealogy of a complete record,
# the final state only, or whichever the record supports.
HEPMC_RECORDS = ("auto", "full", "final-state")


def dataset_schema(fmt: str) -> str:
    """The schema identifier of a dataset written in `fmt`."""
    return {
        "root": ROOT_SCHEMA_VERSION,
        "jsonl": JSONL_SCHEMA_VERSION,
        "hepmc3": HEPMC3_SCHEMA_VERSION,
    }[fmt]


# Fields kept by the "finals+kinematics" store spec. `m` is stored explicitly
# rather than recovered from E and p: at float32, m^2 = E^2 - |p|^2 is a
# catastrophic cancellation. For a 100 GeV pion, E^2 is order 1e4 where the
# float32 spacing is order 1e-3, while m^2 is 0.0195 -- a several-percent error
# on a mass that is known exactly at generation time.
_KINEMATIC_FIELDS = ("id", "status", "px", "py", "pz", "e", "m")

# What each store spec keeps. The default exists because a hadron-level event
# is ~1400 particles of which ~400 are final-state, and most analyses read only
# the momenta of those.
STORE_SPECS: Dict[str, str] = {
    "full": "Every particle and every field the generator produces, "
    "including the shower history, production vertices and lifetimes.",
    "finals": "Final-state particles (status > 0) with all fields, including "
    "production vertices and lifetimes. Needed for displaced-vertex "
    "and long-lived-particle work.",
    "finals+kinematics": "Final-state particles with id, status, "
    "four-momentum "
    "and mass. Covers most analyses at a fraction of the size.",
}

DEFAULT_STORE = "finals+kinematics"
DEFAULT_PRECISION = "f32"

_PRECISIONS = ("f32", "f64")


# ====================================================================== #
# =========================== Helper functions ========================= #
# ====================================================================== #


def _require_awkward() -> Any:
    """Ensure awkward is available, or raise ImportError."""
    try:
        import awkward as ak

        return ak
    except Exception as e:
        raise ImportError(
            "awkward is not available, install to use event datasets "
            "(e.g. `pip install awkward`)."
        ) from e


def _require_uproot() -> Any:
    """Ensure uproot is available, or raise ImportError."""
    try:
        import uproot

        return uproot
    except Exception as e:
        raise ImportError(
            "uproot is not available, install to read or write ROOT event "
            "files (e.g. `pip install uproot`)."
        ) from e


def _require_pyhepmc() -> Any:
    """Ensure pyhepmc is available, or raise ImportError."""
    try:
        import pyhepmc

        return pyhepmc
    except Exception as e:
        raise ImportError(
            "pyhepmc is not available, install to read or write HepMC3 event "
            "files (e.g. `pip install pyhepmc`)."
        ) from e


def detect_format(path: str) -> str:
    """
    Infer the dataset format from a path.

    Parameters:
        path: Path to an event dataset

    Returns:
        One of "root", "jsonl" or "hepmc3". A trailing ".gz" is allowed on
        JSONL and HepMC3 datasets, which are then read and written gzipped

    Raises:
        ValueError: if the suffix is not a format this module handles
    """
    gzipped = path.lower().endswith(".gz")
    suffix = Path(path[:-3] if gzipped else path).suffix.lower()
    if suffix == ".root":
        if gzipped:
            raise ValueError(
                "ROOT files are compressed internally; drop the .gz suffix"
            )
        return "root"
    if suffix in (".jsonl", ".json"):
        return "jsonl"
    if suffix in (".hepmc", ".hepmc3"):
        return "hepmc3"
    raise ValueError(
        f"cannot infer event format from '{path}'; expected a .root, .jsonl "
        f"or "
        ".hepmc suffix"
    )


def _open_text(path: str, mode: str) -> Any:
    """Open a text dataset for reading or writing, gzipped when its name ends
    in .gz."""
    if path.lower().endswith(".gz"):
        import gzip

        return gzip.open(path, mode + "t", encoding="utf-8")
    return open(
        path, mode, encoding="utf-8", buffering=262144 if "w" in mode else -1
    )


def _float_type(precision: str) -> Any:
    """Map a precision name onto a numpy dtype."""
    import numpy as np

    if precision not in _PRECISIONS:
        raise ValueError(
            f"unknown precision '{precision}'; expected one of "
            f"{', '.join(_PRECISIONS)}"
        )
    return np.float32 if precision == "f32" else np.float64


# ====================================================================== #
# ============================ Store specs ============================= #
# ====================================================================== #


def apply_store(
    array: Any,
    store: str = DEFAULT_STORE,
    precision: str = DEFAULT_PRECISION,
    select_pids: Optional[Sequence[int]] = None,
) -> Any:
    """
    Reduce a batch of events to what a store spec keeps.

    This is the single largest lever on dataset size: the same 200 events
    occupy 58.6 KiB each under "full" and 6.8 KiB each under
    "finals+kinematics", because most of a hadron-level event is shower history
    that no analysis reads.

    Parameters:
        array: Awkward Array as produced by Pythia.nextBatch, with a `prt`
            field of per-particle records and an `info` field of per-event
            scalars
        store: One of the keys of STORE_SPECS
        precision: "f32" or "f64" for the floating-point fields
        select_pids: Optional |PDG id| whitelist. Only particles whose absolute
            id is listed are kept, so 321 keeps K+ and K-. Under "full" the
            mother/daughter indices still refer to the unfiltered record

    Returns:
        A new Awkward Array holding only the selected particles and fields

    Raises:
        ValueError: on an unknown store spec or precision
    """
    ak = _require_awkward()

    if store not in STORE_SPECS:
        raise ValueError(
            f"unknown store spec '{store}'; expected one of "
            f"{', '.join(STORE_SPECS)}"
        )
    float_type = _float_type(precision)

    particles = array["prt"] if "prt" in array.fields else array
    # Every spec but "full" keeps only what survives to the end of the event.
    if store != "full":
        particles = particles[particles.status > 0]
    if select_pids:
        particles = particles[_abs_id_in(ak, particles["id"], select_pids)]

    fields = _flatten_particles(ak, particles, float_type)
    if store == "finals+kinematics":
        fields = {
            name: fields[name] for name in _KINEMATIC_FIELDS if name in fields
        }

    out: Dict[str, Any] = {"prt": ak.zip(fields)}
    out.update(_event_columns(ak, array, float_type, store))
    return ak.Array(out)


def final_state(particles: Any) -> Any:
    """
    Jagged mask of the final-state particles, under either status convention.

    Pythia marks a final-state particle with a positive status, and its codes
    are all 11 or above; the HEPEVT and Les Houches convention of LHE files and
    HepMC uses 1, with 2 for a decayed particle or intermediate resonance and
    3 for documentation lines. A particle with status 1 or above 10 is final
    under both. A record with no status field is taken to be all final state.
    """
    ak = _require_awkward()
    if "status" not in particles.fields:
        return ak.ones_like(particles["id"], dtype=bool)
    status = particles["status"]
    return (status == 1) | (status > 10)


def _abs_id_in(ak: Any, ids: Any, pids: Sequence[int]) -> Any:
    """Jagged boolean mask: True where |id| is one of `pids`."""
    import numpy as np

    flat = np.abs(ak.to_numpy(ak.flatten(ids)))
    keep = np.isin(flat, np.asarray(sorted({abs(int(p)) for p in pids})))
    return ak.unflatten(keep, ak.num(ids))


def _event_columns(
    ak: Any, array: Any, float_type: Any, store: str
) -> Dict[str, Any]:
    """
    Build the per-event columns that accompany the particle record.

    The nominal weight is always included. The hard-process scalars (x1, x2,
    the PDF values, the Mandelstams) are only worth their space alongside the
    fuller particle records, and are dropped by the kinematics-only spec.
    """
    import numpy as np

    # Weights are kept in double precision whatever the kinematics use: they
    # are summed over whole samples, and variation weights can differ from the
    # nominal by less than float32 resolves.
    columns: Dict[str, Any] = {}
    if "info" not in array.fields:
        # A dataset read back from disk already carries flat weight columns.
        if "weight" in array.fields:
            columns["weight"] = array["weight"]
            if "weights" in array.fields:
                columns["weights"] = array["weights"]
        return columns

    info = array["info"]
    if "weights" in info.fields:
        weights = info["weights"]
        # Entry zero is the nominal weight; the rest are variations, and are
        # kept only when a run actually produced them.
        columns["weight"] = ak.values_astype(
            ak.fill_none(ak.firsts(weights), 1.0), np.float64
        )
        if int(ak.max(ak.num(weights)) or 0) > 1:
            columns["weights"] = ak.values_astype(weights, np.float64)
    else:
        columns["weight"] = ak.values_astype(
            ak.ones_like(ak.num(array["prt"])), np.float64
        )

    if store != "finals+kinematics":
        for name in info.fields:
            if name == "weights":
                continue
            columns[f"info_{name}"] = ak.values_astype(info[name], float_type)
    return columns


def _flatten_particles(
    ak: Any, particles: Any, float_type: Any
) -> Dict[str, Any]:
    """
    Flatten Pythia's nested particle record into scalar branches.

    Pythia nests the four-momentum under `p` and the production vertex under
    `vProd`. Flattening them to px/py/pz/e and vx/vy/vz/vt keeps the field
    names identical to the long-standing JSONL schema, so a tool migrating
    between the two formats does not also have to learn a new layout.
    """

    def cast(values: Any) -> Any:
        return ak.values_astype(values, float_type)

    fields: Dict[str, Any] = {}
    for name in (
        "id",
        "status",
        "mother1",
        "mother2",
        "daughter1",
        "daughter2",
        "col",
        "acol",
        "prod_vertex",
        "end_vertex",
        "info",
    ):
        if name in particles.fields:
            fields[name] = particles[name]

    if "p" in particles.fields:
        momentum = particles["p"]
        fields["px"] = cast(momentum["px"])
        fields["py"] = cast(momentum["py"])
        fields["pz"] = cast(momentum["pz"])
        fields["e"] = cast(momentum["e"])
    for name in ("m", "scale", "pol", "tau"):
        if name in particles.fields:
            fields[name] = cast(particles[name])

    # vProd is an option type in Pythia's schema (a particle need not carry a
    # production vertex), so fill the gaps before casting rather than
    # propagating None into a float branch that ROOT would have to model as a
    # variable-length column.
    if "vProd" in particles.fields:
        vertex = ak.fill_none(
            particles["vProd"], {"px": 0.0, "py": 0.0, "pz": 0.0, "e": 0.0}
        )
        fields["vx"] = cast(vertex["px"])
        fields["vy"] = cast(vertex["py"])
        fields["vz"] = cast(vertex["pz"])
        fields["vt"] = cast(vertex["e"])
    return fields


# ====================================================================== #
# ========================= HepMC3 genealogy =========================== #
# ====================================================================== #


def has_genealogy(array: Any) -> bool:
    """
    Whether a batch is a complete record whose genealogy HepMC3 can carry.

    A complete record keeps every particle with its status and either its
    mother and daughter indices (Pythia's record) or the vertices it came
    from and went into (a vertex graph such as Sherpa's); the final-state
    store specs drop both the intermediate particles and, with them, any
    meaning the indices had.
    """
    ak = _require_awkward()

    particles = array["prt"] if "prt" in array.fields else array
    if not has_genealogy_fields(particles.fields):
        return False
    if len(particles) == 0:
        return False
    return bool(ak.any(particles["status"] < 0))


def has_genealogy_fields(fields: Sequence[str]) -> bool:
    """Whether a particle record names its genealogy, in either
    representation (see :func:`has_genealogy`)."""
    return all(name in fields for name in _GENEALOGY_FIELDS) or all(
        name in fields for name in _VERTEX_FIELDS
    )


def _is_hadron(ids: Any) -> Any:
    """
    Pythia's ``ParticleDataEntry::isHadron`` over an array of PDG ids.

    Vectorised transcription of the C++: exotic hadrons (9xxxxxx with no zero
    digit), K_L and K_S, and any id above 100 with non-zero quark digits that
    is neither a SUSY/technicolour code (1e6 to 9e6) nor a special code
    (9.9e6 and up).
    """
    import numpy as np

    a = np.abs(np.asarray(ids, dtype=np.int64))
    d1, d2, d3 = a % 10, a // 10 % 10, a // 100 % 10
    d4, d5 = a // 1000 % 10, a // 10000 % 10
    exotic = (
        (a // 1000000 == 9)
        & (a < 10000000)
        & (d1 != 0)
        & (d2 != 0)
        & (d3 != 0)
        & (d4 != 0)
        & (d5 != 0)
    )
    ordinary = (a > 100) & ~((a >= 1000000) & (a <= 9000000)) & (a < 9900000)
    ordinary &= (a == 130) | (a == 310) | ((d1 != 0) & (d2 != 0) & (d3 != 0))
    return exotic | ordinary


def hepmc_status(ids: Any, status: Any, daughter1: Any) -> Any:
    """
    Pythia's ``Particle::statusHepMC`` over the arrays of one complete record.

    Final-state particles are 1 and the beams (status -12) are 4. A hadron,
    muon or tau is 2 when it decayed normally: its first daughter has a decay
    status (91 to 94) and is not a copy of itself, which excludes
    Bose-Einstein shifts. Every other negative status in Pythia's range is
    written as its positive counterpart, and anything outside it as 0.

    Parameters:
        ids: PDG ids, indexed as the record is
        status: Pythia status codes
        daughter1: Index of each particle's first daughter, into the same
            record

    Returns:
        The HepMC3 status codes, as an int32 array
    """
    import numpy as np

    ids = np.asarray(ids, dtype=np.int64)
    status = np.asarray(status, dtype=np.int64)
    daughter1 = np.clip(np.asarray(daughter1, dtype=np.int64), 0, len(ids) - 1)

    out = np.zeros(len(ids), dtype=np.int32)
    final = status > 0
    beam = status == -12
    out[final] = _HEPMC_FINAL
    out[beam] = _HEPMC_BEAM

    unstable = _is_hadron(ids) | (np.abs(ids) == 13) | (np.abs(ids) == 15)
    daughter_status = np.abs(status[daughter1])
    decayed = (
        ~final
        & ~beam
        & unstable
        & (ids[daughter1] != ids)
        & (daughter_status > 90)
        & (daughter_status < 95)
    )
    out[decayed] = _HEPMC_DECAYED

    other = ~final & ~beam & ~decayed & (status <= -11) & (status >= -200)
    out[other] = -status[other]
    return out


def hepmc_event_from_record(record: Dict[str, Any], event_number: int) -> Any:
    """
    Build one HepMC3 event from a complete Pythia record, as Pythia would.

    This reproduces ``Pythia8ToHepMC3::fill_next_event`` for the particles
    and vertices: the system entry (index 0) is dropped, every other particle
    gets Pythia's HepMC status, and the vertices follow Pythia's rule. A
    particle's production vertex is the end vertex of its lowest-index
    mother when that mother already has one; otherwise a new vertex is made
    with all of its mothers incoming, in the sense of ``Particle::motherList``
    (a single mother, a carbon copy, the range of a string system, or two
    separate mothers). The vertex position is the production vertex of the
    first outgoing particle to carry one. Checked against Pythia's own
    converter on soft and hard QCD, Drell-Yan, heavy-flavour and e+e- events
    (tools/tests/fixtures/hepmc_reference).

    The particles are numbered in topological order -- every mother before
    its daughters -- rather than in Pythia's record order, which lists some
    copies before their originals. HepMC3's ASCII writer emits particles by
    number and its reader resolves a mother on sight, so Pythia's order would
    produce a file the reader rejects.

    Parameters:
        record: One event's particle columns as numpy arrays: id, status,
            mother1, mother2, daughter1, px, py, pz, e, m, vx, vy, vz, vt;
            the production vertex in mm, the rest in GeV
        event_number: The event number to write

    Returns:
        A ``pyhepmc.GenEvent`` with its particles and vertices, without run
        info, weights or cross section

    Raises:
        ValueError: when the record is not complete -- the first entry is not
            the system entry, or an index points outside the record -- which
            is what a record filtered with `select_pids` looks like
    """
    import numpy as np

    pyhepmc = _require_pyhepmc()

    ids = np.asarray(record["id"], dtype=np.int64)
    status = np.asarray(record["status"], dtype=np.int64)
    mother1 = np.asarray(record["mother1"], dtype=np.int64)
    mother2 = np.asarray(record["mother2"], dtype=np.int64)
    daughter1 = np.asarray(record["daughter1"], dtype=np.int64)
    n = len(ids)
    if (
        n == 0
        or status[0] != -11
        or int(max(mother1.max(), mother2.max(), daughter1.max())) >= n
    ):
        raise ValueError(
            "the HepMC3 genealogy needs the complete Pythia record, with the "
            "system entry first and every mother and daughter index inside "
            "it; a "
            "record filtered with select_pids cannot be written this way"
        )

    # Particle::motherList, vectorised into a kind per particle: 0 none,
    # 1 a single mother (or carbon copy), 2 a string-system range, 3 two
    # separate mothers. A zero among two "separate" mothers is dropped, as
    # Pythia's converter drops it.
    magnitude = np.abs(status)
    none = (
        (magnitude == 11)
        | (magnitude == 12)
        | ((mother1 == 0) & (mother2 == 0))
    )
    single = ~none & ((mother2 == 0) | (mother2 == mother1))
    span = (
        ~none
        & ~single
        & (
            ((magnitude > 80) & (magnitude < 90))
            | ((magnitude > 100) & (magnitude < 107))
        )
    )
    low = np.where(single, mother1, np.minimum(mother1, mother2))
    high = np.where(single, mother1, np.maximum(mother1, mother2))
    kind = np.where(none, 0, np.where(single, 1, np.where(span, 2, 3)))
    kind = np.where((kind == 3) & (low == 0), 1, kind)
    low = np.where(low == 0, high, low)
    located = (
        (record["vx"] != 0)
        | (record["vy"] != 0)
        | (record["vz"] != 0)
        | (record["vt"] != 0)
    )

    kinds, lows, highs = kind.tolist(), low.tolist(), high.tolist()
    beam = ((magnitude == 11) | (magnitude == 12)).tolist()
    has_position = located.tolist()
    vx, vy, vz, vt = (
        record[name].tolist() for name in ("vx", "vy", "vz", "vt")
    )

    # Pass one, without HepMC3 objects: which particles share a vertex. A
    # mother added to a new vertex leaves the one it was in, as HepMC3's own
    # add_particle_in moves it.
    end_vertex: Dict[int, int] = {}  # particle index -> vertex number
    production: List[Optional[int]] = [None] * n
    incoming: List[List[int]] = []
    outgoing: List[List[int]] = []
    position: List[Optional[Tuple[float, float, float, float]]] = []
    for i in range(1, n):
        k = kinds[i]
        if k == 0:
            continue
        first = lows[i]
        number = end_vertex.get(first)
        if number is None:
            number = len(incoming)
            if k == 1:
                mothers: List[int] = [first]
            elif k == 2:
                mothers = list(range(first, highs[i] + 1))
            else:
                mothers = [first, highs[i]]
            for mother in mothers:
                previous = end_vertex.get(mother)
                if previous is not None:
                    incoming[previous].remove(mother)
                end_vertex[mother] = number
            incoming.append(mothers)
            outgoing.append([])
            position.append(None)
        if has_position[i] and position[number] is None:
            position[number] = (vx[i], vy[i], vz[i], vt[i])
        outgoing[number].append(i)
        production[i] = number

    # A particle from nowhere -- no mothers, no daughters, not a beam -- would
    # otherwise never reach the file. Pythia's converter gives it a vertex of
    # its own, with a warning; the vertex is kept, the warning is not.
    for i in range(1, n):
        if kinds[i] == 0 and i not in end_vertex and not beam[i]:
            production[i] = len(incoming)
            incoming.append([])
            outgoing.append([i])
            position.append(None)

    codes = hepmc_status(ids, status, daughter1)
    return _assemble_hepmc_event(
        pyhepmc,
        record,
        event_number,
        first=1,
        incoming=incoming,
        outgoing=outgoing,
        production=production,
        end_vertex=end_vertex,
        position=position,
        status_codes=codes,
    )


def hepmc_event_from_vertices(
    record: Dict[str, Any], event_number: int
) -> Any:
    """
    Build one HepMC3 event from a record that carries its genealogy as a
    vertex graph.

    This is the representation a generator with explicit vertices gives, and
    what the Sherpa tools write under ``store="full"``: every particle names
    the vertex it came from (``prod_vertex``) and the vertex it went into
    (``end_vertex``), -1 for none, and ``status`` holds the HepMC status
    itself -- 1 for a final-state particle, otherwise negative: -4 for a
    beam, -2 for a decayed hadron or lepton, -3 for a documentation entry,
    -11 for everything intermediate. Each vertex's position is the production
    vertex of the first outgoing particle to carry one.

    A particle with neither vertex and no final status gets a vertex of its
    own, as Pythia's converter gives one to a particle from nowhere. A vertex
    with incoming particles but nothing outgoing cannot be expressed through
    ``from_hepevt`` and is dropped; its incoming particles then appear to end
    nowhere, with their status still saying otherwise.

    Parameters:
        record: One event's particle columns as numpy arrays: id, status,
            prod_vertex, end_vertex, px, py, pz, e, m, vx, vy, vz, vt
        event_number: The event number to write

    Returns:
        A ``pyhepmc.GenEvent`` with its particles and vertices, without run
        info, weights or cross section

    Raises:
        ValueError: on an empty record
    """
    import numpy as np

    pyhepmc = _require_pyhepmc()

    ids = np.asarray(record["id"], dtype=np.int64)
    status = np.asarray(record["status"], dtype=np.int64)
    prod = np.asarray(record["prod_vertex"], dtype=np.int64)
    end = np.asarray(record["end_vertex"], dtype=np.int64)
    n = len(ids)
    if n == 0:
        raise ValueError("an event with no particles cannot be written")

    located = (
        (record["vx"] != 0)
        | (record["vy"] != 0)
        | (record["vz"] != 0)
        | (record["vt"] != 0)
    ).tolist()
    vx, vy, vz, vt = (
        record[name].tolist() for name in ("vx", "vy", "vz", "vt")
    )

    vertex_ids = sorted(
        set(prod[prod >= 0].tolist()) | set(end[end >= 0].tolist())
    )
    number_of = {vertex: k for k, vertex in enumerate(vertex_ids)}
    incoming: List[List[int]] = [[] for _ in vertex_ids]
    outgoing: List[List[int]] = [[] for _ in vertex_ids]
    position: List[Optional[Tuple[float, float, float, float]]] = [None] * len(
        vertex_ids
    )
    production: List[Optional[int]] = [None] * n
    end_vertex: Dict[int, int] = {}
    prods, ends = prod.tolist(), end.tolist()
    for i in range(n):
        if prods[i] >= 0:
            number = number_of[prods[i]]
            outgoing[number].append(i)
            production[i] = number
            if position[number] is None and located[i]:
                position[number] = (vx[i], vy[i], vz[i], vt[i])
        if ends[i] >= 0:
            number = number_of[ends[i]]
            incoming[number].append(i)
            end_vertex[i] = number
    for i in range(n):
        if production[i] is None and i not in end_vertex:
            production[i] = len(incoming)
            incoming.append([])
            outgoing.append([i])
            position.append(None)

    codes = np.where(status > 0, _HEPMC_FINAL, -status).astype(np.int32)
    return _assemble_hepmc_event(
        pyhepmc,
        record,
        event_number,
        first=0,
        incoming=incoming,
        outgoing=outgoing,
        production=production,
        end_vertex=end_vertex,
        position=position,
        status_codes=codes,
    )


def _assemble_hepmc_event(
    pyhepmc: Any,
    record: Dict[str, Any],
    event_number: int,
    *,
    first: int,
    incoming: List[List[int]],
    outgoing: List[List[int]],
    production: Sequence[Optional[int]],
    end_vertex: Dict[int, int],
    position: List[Optional[Tuple[float, float, float, float]]],
    status_codes: Any,
) -> Any:
    """
    Number the particles and build the event in one C++ call.

    ``from_hepevt`` makes one vertex per distinct (first, last) parent range,
    so a vertex structure is reproduced when every vertex's incoming set is a
    contiguous block of the new numbering with no other vertex's members
    inside it. Ordering particles by the vertex they end in (finals last)
    does that -- the sets are disjoint, since a particle ends in at most one
    vertex -- and is topological too: a particle's production vertex comes
    before its end vertex, so every mother is numbered before its daughters,
    which is what HepMC3's ASCII writer and reader need.

    Parameters:
        first: Index of the first particle to write; 1 skips Pythia's system
            entry, 0 writes every particle
        incoming, outgoing: Particle indices per vertex number
        production: Each particle's production vertex number, or None
        end_vertex: Each particle's end vertex number, where it has one
        position: Each vertex's position, or None
        status_codes: The HepMC status per particle, indexed as the record
    """
    import numpy as np

    ids = np.asarray(record["id"], dtype=np.int64)
    n = len(ids)
    order = _topological_order(
        len(incoming), incoming, outgoing, production, end_vertex
    )
    rank = {number: index for index, number in enumerate(order)}
    last = len(order)
    permutation = sorted(
        range(first, n),
        key=lambda i: rank[end_vertex[i]] if i in end_vertex else last,
    )
    body = np.asarray(permutation, dtype=np.int64)
    slot = np.empty(n, dtype=np.int64)
    slot[body] = np.arange(len(body))
    slots = slot.tolist()

    count = n - first
    parents = np.full((count, 2), -1, dtype=np.int32)
    vx = np.zeros(count)
    vy = np.zeros(count)
    vz = np.zeros(count)
    vt = np.zeros(count)
    # Vertices with nothing incoming are rare, and built by hand.
    leftovers: List[int] = []
    for number in order:
        members = incoming[number]
        if not members:
            leftovers.append(number)
            continue
        first_slot = min(slots[m] for m in members)
        last_slot = max(slots[m] for m in members)
        if last_slot - first_slot + 1 != len(members):
            raise AssertionError(
                "vertex incoming particles are not contiguous"
            )
        where = [slots[d] for d in outgoing[number]]
        parents[where] = (first_slot, last_slot)
        if position[number] is not None:
            vx[where], vy[where], vz[where], vt[where] = position[number]

    event = pyhepmc.GenEvent(pyhepmc.Units.GEV, pyhepmc.Units.MM)
    event.from_hepevt(
        event_number,
        np.ascontiguousarray(record["px"][body], dtype=np.float64),
        np.ascontiguousarray(record["py"][body], dtype=np.float64),
        np.ascontiguousarray(record["pz"][body], dtype=np.float64),
        np.ascontiguousarray(record["e"][body], dtype=np.float64),
        np.ascontiguousarray(record["m"][body], dtype=np.float64),
        ids[body].astype(np.int32),
        np.asarray(status_codes, dtype=np.int32)[body],
        parents=parents,
        vx=vx,
        vy=vy,
        vz=vz,
        vt=vt,
        fortran=False,
    )
    if leftovers:
        particles = event.particles
        for number in leftovers:
            vertex = pyhepmc.GenVertex()
            for daughter in outgoing[number]:
                vertex.add_particle_out(particles[slots[daughter]])
            if position[number] is not None:
                vertex.position = pyhepmc.FourVector(*position[number])
            event.add_vertex(vertex)
    return event


def _topological_order(
    count: int,
    incoming: Sequence[Sequence[int]],
    outgoing: Sequence[Sequence[int]],
    production: Sequence[Optional[int]],
    end_vertex: Dict[int, int],
) -> List[int]:
    """
    The order in which to add vertices so that every one follows the
    production vertices of its incoming particles: ``GenEvent::add_tree``'s
    walk from the root particles, which is the order Pythia's converter adds
    them in. Vertices not reached from a root follow, in creation order.
    """
    from collections import deque

    added = [False] * count
    order: List[int] = []
    roots = [
        end_vertex[i] for i in sorted(end_vertex) if production[i] is None
    ]
    pending = deque(dict.fromkeys(roots))
    # A vertex deferred to its parents more often than there are vertices
    # is waiting on itself: the graph has a cycle, which no order resolves.
    deferrals = 0
    while pending:
        number = pending[0]
        deferred = False
        for mother in incoming[number]:
            parent = production[mother]
            if parent is not None and not added[parent]:
                pending.appendleft(parent)
                deferred = True
        if deferred:
            deferrals += 1
            if deferrals > count * count + count:
                raise ValueError(
                    "the vertex graph has a cycle; no particle order puts "
                    "every mother before its daughters"
                )
            continue
        pending.popleft()
        if added[number]:
            continue
        added[number] = True
        order.append(number)
        for daughter in outgoing[number]:
            child = end_vertex.get(daughter)
            if child is not None and not added[child]:
                pending.append(child)
    order.extend(number for number in range(count) if not added[number])
    return order


# ====================================================================== #
# ============================== Writing =============================== #
# ====================================================================== #


class EventWriter:
    """
    Chunked writer for an event dataset.

    Generation produces events in batches, and a sample is normally far larger
    than memory, so the dataset has to be written incrementally. uproot can
    only extend a tree through the handle that created it -- reopening a closed
    file to append raises "cannot view preexisting TTrees" -- so the handle is
    held open for the lifetime of this object rather than reopened per batch.

    Use as a context manager:

        with EventWriter("events.root", store="finals+kinematics") as writer:
            for _ in range(n_chunks):
                writer.write(pythia.nextBatch(chunk_size))
        stats = writer.stats

    HepMC3 output also needs the run's two beams, which a batch of events does
    not carry, and takes the names of the event weights and the cross section
    once they are known:

        with EventWriter("events.hepmc", beams=beams) as writer:
            writer.set_weight_names(names)
            writer.set_cross_section(sigma_pb, error_pb)
            writer.write(batch)

    Attributes:
        stats: Populated on close with "path", "format", "n_events" and "bytes"
    """

    def __init__(
        self,
        path: str,
        *,
        store: str = DEFAULT_STORE,
        precision: str = DEFAULT_PRECISION,
        fmt: Optional[str] = None,
        compression: str = "zstd",
        beams: Optional[Sequence[Dict[str, Any]]] = None,
        hepmc_record: str = "auto",
    ) -> None:
        """
        Parameters:
            path: Destination; the suffix selects the format unless `fmt`
                is given
            store: Store spec, one of the keys of STORE_SPECS
            precision: "f32" or "f64" for the floating-point particle fields
            fmt: Explicit format ("root", "jsonl" or "hepmc3")
            compression: Compression for ROOT output ("zstd", "lz4", or "none")
            beams: For HepMC3, the two beam particles, each a dict with "id",
                "px", "py", "pz", "e" and "m" in GeV. A complete record
                carries its own beams, so they are only used when the final
                state alone is written
            hepmc_record: For HepMC3, what to write: "full" (the vertex
                genealogy, which needs a complete record), "final-state" (the
                stored final-state particles between the two beams), or
                "auto" (the genealogy whenever the record carries one)

        Raises:
            ValueError: on an unknown format, store spec, precision,
                compression or HepMC3 record; for HepMC3, on beams that are
                not two particles
        """
        self.path = path
        self.store = store
        self.precision = precision
        self.format = fmt or detect_format(path)
        self.compression = compression.lower()
        self.n_events = 0
        self.stats: Dict[str, Any] = {}
        self.beams = [dict(beam) for beam in beams] if beams else None
        self.weight_names: Optional[List[str]] = None
        self.hepmc_record = hepmc_record
        self._cross_section: Optional[Tuple[float, float, int, int]] = None

        if self.format not in FORMATS:
            raise ValueError(f"unknown event format '{self.format}'")
        if self.format == "hepmc3":
            if hepmc_record not in HEPMC_RECORDS:
                raise ValueError(
                    f"unknown HepMC3 record '{hepmc_record}'; expected one of "
                    f"{', '.join(HEPMC_RECORDS)}"
                )
            if hepmc_record != "full" and (
                self.beams is None or len(self.beams) != 2
            ):
                raise ValueError(
                    "HepMC3 output needs the run's two beam particles"
                )
        if self.compression not in ("zstd", "lz4", "none"):
            raise ValueError(
                f"unknown compression '{compression}'; expected zstd, lz4 or "
                f"none"
            )
        if store not in STORE_SPECS:
            raise ValueError(
                f"unknown store spec '{store}'; expected one of "
                f"{', '.join(STORE_SPECS)}"
            )
        _float_type(precision)

        self._handle: Any = None
        self._tree_made = False
        self._run_info: Any = None

    def set_weight_names(self, names: Sequence[str]) -> None:
        """Name the event weights, nominal first (HepMC3 records them per
        run)."""
        self.weight_names = [str(name) for name in names]

    def set_cross_section(
        self,
        sigma_pb: float,
        error_pb: float,
        accepted: int = -1,
        attempted: int = -1,
    ) -> None:
        """Cross section attached to the events written from here on, in pb.

        HepMC3 carries it per event, so a generator's running estimate can be
        updated batch by batch; the last event holds the final value.
        """
        self._cross_section = (
            float(sigma_pb),
            float(error_pb),
            int(accepted),
            int(attempted),
        )

    # ------------------------------------------------------------------ #

    def __enter__(self) -> "EventWriter":
        self._open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _open(self) -> None:
        """Open the destination, replacing anything already there."""
        if self.format == "root":
            uproot = _require_uproot()
            codec = {
                "zstd": uproot.ZSTD(3),
                "lz4": uproot.LZ4(4),
                "none": None,
            }[self.compression]
            self._handle = uproot.recreate(self.path, compression=codec)
        elif self.format == "hepmc3":
            self._handle = _require_pyhepmc().open(self.path, "w")
        else:
            self._handle = _open_text(self.path, "w")

    def write(self, array: Any) -> int:
        """
        Reduce a batch to the store spec and append it to the dataset.

        Parameters:
            array: Awkward Array of events, as produced by Pythia.nextBatch

        Returns:
            Number of events written from this batch
        """
        return self.write_reduced(
            apply_store(array, store=self.store, precision=self.precision)
        )

    def write_reduced(self, array: Any) -> int:
        """
        Append a batch that has already had the store spec applied.

        A caller that also summarises what it writes should reduce once and
        pass the result to both, rather than reducing again here.

        Parameters:
            array: Awkward Array already in the stored layout

        Returns:
            Number of events written from this batch
        """
        if self._handle is None:
            self._open()
        count = len(array)
        if not count:
            return 0

        if self.format == "root":
            self._write_root_chunk(array)
        elif self.format == "hepmc3":
            self._write_hepmc_chunk(array)
        else:
            self._write_jsonl_chunk(array)
        self.n_events += count
        return count

    def _write_root_chunk(self, array: Any) -> None:
        """Extend the TTree, creating it from the first batch's types."""
        payload = {field: array[field] for field in array.fields}
        if not self._tree_made:
            # The tree's branch types are fixed at creation, so they are taken
            # from the first batch; every later batch has the same store spec
            # applied and therefore the same layout.
            self._handle.mktree(
                ROOT_TREE_NAME,
                {name: values.type for name, values in payload.items()},
            )
            self._tree_made = True
        self._handle[ROOT_TREE_NAME].extend(payload)

    def _write_jsonl_chunk(self, array: Any) -> None:
        """
        Append a batch in the legacy evtjsonl-1.0 schema.

        Retained so a small sample can still be read by a person, and so tools
        not yet migrated to the columnar path keep working.
        """
        ak = _require_awkward()

        # event_id continues across batches, so it is offset by the number of
        # events already written. The caller updates n_events after this
        # returns, so self.n_events is still the count before this batch.
        for offset, record in enumerate(ak.to_list(array)):
            particles = record.get("prt", [])
            # The JSONL schema spells energy "E" and carries a per-particle
            # index; both predate the columnar layout and are preserved so that
            # existing readers are unaffected.
            rows = []
            for position, particle in enumerate(particles):
                row: Dict[str, Any] = {"i": position}
                for name, value in particle.items():
                    row["E" if name == "e" else name] = value
                rows.append(row)
            payload = {
                "schema": JSONL_SCHEMA_VERSION,
                "event_id": self.n_events + offset,
                "weight": record.get("weight", 1.0),
                **(
                    {"weights": record["weights"]}
                    if record.get("weights")
                    else {}
                ),
                "data": {"n": len(rows), "particles": rows},
            }
            self._handle.write(
                json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
                + "\n"
            )

    def _write_hepmc_chunk(self, array: Any) -> None:
        """
        Append a batch as HepMC3 events.

        A complete record (every particle, with mother and daughter indices)
        is written as its vertex genealogy; anything else as the final state
        between the two beams. "auto" picks by what the batch carries.
        """
        if self.hepmc_record == "full":
            if not has_genealogy(array):
                raise ValueError(
                    "the HepMC3 genealogy needs a complete record: every "
                    "particle "
                    "with its status, mother and daughter indices "
                    "(store='full')"
                )
            self._write_hepmc_genealogy(array)
        elif self.hepmc_record == "final-state" or not has_genealogy(array):
            self._write_hepmc_final_state(array)
        else:
            self._write_hepmc_genealogy(array)

    def _hepmc_run(self, pyhepmc: Any, width: int) -> Any:
        """
        The per-run cross section object, creating the run info on first use.

        HepMC3 keeps the weight names once per run and the cross section once
        per event, so the names are fixed by the first batch written.
        """
        if self._run_info is None:
            self._run_info = pyhepmc.GenRunInfo()
            names = self.weight_names or ["Weight"]
            if len(names) != width:
                raise ValueError(
                    f"{width} weight(s) per event but {len(names)} weight "
                    f"name(s)"
                )
            self._run_info.weight_names = names
        cross_section = None
        if self._cross_section is not None:
            cross_section = pyhepmc.GenCrossSection()
            cross_section.set_cross_section(*self._cross_section)
        return cross_section

    def _write_hepmc_final_state(self, array: Any) -> None:
        """
        Append a batch as final-state HepMC3 events.

        Each event is one interaction vertex: the two beams in (status 4), the
        stored particles out (status 1). It is built from flat arrays in one
        call per event (GenEvent.from_hepevt) rather than particle by particle.
        """
        import numpy as np

        ak = _require_awkward()
        pyhepmc = _require_pyhepmc()

        # Everything written is labelled final state, so only the final state
        # is written, whatever the input holds.
        particles = array["prt"][final_state(array["prt"])]
        counts = ak.to_numpy(ak.num(particles))
        flat = {
            name: ak.to_numpy(ak.flatten(particles[name])).astype(np.float64)
            for name in ("px", "py", "pz", "e")
        }
        ids = ak.to_numpy(ak.flatten(particles["id"])).astype(np.int32)
        if "m" in particles.fields:
            masses = ak.to_numpy(ak.flatten(particles["m"])).astype(np.float64)
        else:
            p2 = flat["px"] ** 2 + flat["py"] ** 2 + flat["pz"] ** 2
            masses = np.sqrt(np.clip(flat["e"] ** 2 - p2, 0.0, None))
        nominal = ak.to_numpy(array["weight"]).astype(np.float64)
        variations = (
            ak.to_list(array["weights"]) if "weights" in array.fields else None
        )
        cross_section = self._hepmc_run(
            pyhepmc, len(variations[0]) if variations else 1
        )

        beam = {
            key: np.array([b[key] for b in self.beams], dtype=np.float64)
            for key in ("px", "py", "pz", "e", "m")
        }
        beam_ids = np.array([b["id"] for b in self.beams], dtype=np.int32)

        start = 0
        for offset, n in enumerate(counts):
            window = slice(start, start + n)
            start += n
            parents = np.full((n + 2, 2), -1, dtype=np.int32)
            parents[2:] = (0, 1)  # every product comes from both beams
            event = pyhepmc.GenEvent(pyhepmc.Units.GEV, pyhepmc.Units.MM)
            event.from_hepevt(
                self.n_events + offset,
                *(
                    np.concatenate([beam[key], flat[key][window]])
                    for key in ("px", "py", "pz", "e")
                ),
                np.concatenate([beam["m"], masses[window]]),
                np.concatenate([beam_ids, ids[window]]),
                np.concatenate(
                    [
                        np.full(2, _HEPMC_BEAM, np.int32),
                        np.full(n, _HEPMC_FINAL, np.int32),
                    ]
                ),
                parents=parents,
                fortran=False,
            )
            event.run_info = self._run_info
            event.weights = (
                variations[offset] if variations else [float(nominal[offset])]
            )
            if cross_section is not None:
                event.cross_section = cross_section
            self._handle.write(event)

    def _write_hepmc_genealogy(self, array: Any) -> None:
        """
        Append a batch of complete records as HepMC3 events with their vertex
        genealogy.

        A Pythia record (mother and daughter indices) is converted as
        Pythia's own ``Pythia8ToHepMC3`` converts it, so the file is the one
        a C++ main program would have written; a vertex-graph record
        (``prod_vertex``, ``end_vertex``) is written vertex by vertex. See
        :func:`hepmc_event_from_record` and :func:`hepmc_event_from_vertices`.
        """
        import numpy as np

        ak = _require_awkward()
        pyhepmc = _require_pyhepmc()

        particles = array["prt"]
        counts = ak.to_numpy(ak.num(particles))
        columns: Dict[str, Any] = {}
        pythia_record = all(
            name in particles.fields for name in _GENEALOGY_FIELDS
        )
        int_names = (
            ("id", "status", "mother1", "mother2", "daughter1")
            if pythia_record
            else ("id", "status", "prod_vertex", "end_vertex")
        )
        build = (
            hepmc_event_from_record
            if pythia_record
            else hepmc_event_from_vertices
        )
        for name in int_names:
            columns[name] = ak.to_numpy(ak.flatten(particles[name])).astype(
                np.int64
            )
        for name in ("px", "py", "pz", "e", "m"):
            columns[name] = ak.to_numpy(ak.flatten(particles[name])).astype(
                np.float64
            )
        total = len(columns["id"])
        for name in ("vx", "vy", "vz", "vt"):
            if name in particles.fields:
                columns[name] = ak.to_numpy(
                    ak.flatten(particles[name])
                ).astype(np.float64)
            else:
                columns[name] = np.zeros(total, dtype=np.float64)
        pdf_columns = (
            "info_id1",
            "info_id2",
            "info_x1",
            "info_x2",
            "info_Q2Fac",
            "info_pdf1",
            "info_pdf2",
        )
        pdf = (
            {name: ak.to_numpy(array[name]) for name in pdf_columns}
            if all(name in array.fields for name in pdf_columns)
            else None
        )
        nominal = ak.to_numpy(array["weight"]).astype(np.float64)
        variations = (
            ak.to_list(array["weights"]) if "weights" in array.fields else None
        )
        cross_section = self._hepmc_run(
            pyhepmc, len(variations[0]) if variations else 1
        )

        start = 0
        for offset, n in enumerate(counts):
            window = slice(start, start + n)
            start += n
            event = build(
                {name: values[window] for name, values in columns.items()},
                self.n_events + offset,
            )
            event.run_info = self._run_info
            event.weights = (
                variations[offset] if variations else [float(nominal[offset])]
            )
            if cross_section is not None:
                event.cross_section = cross_section
            if pdf is not None:
                info = pyhepmc.GenPdfInfo()
                info.parton_id1 = int(pdf["info_id1"][offset])
                info.parton_id2 = int(pdf["info_id2"][offset])
                info.x1 = float(pdf["info_x1"][offset])
                info.x2 = float(pdf["info_x2"][offset])
                info.scale = math.sqrt(
                    max(float(pdf["info_Q2Fac"][offset]), 0.0)
                )
                info.xf1 = float(pdf["info_pdf1"][offset])
                info.xf2 = float(pdf["info_pdf2"][offset])
                event.pdf_info = info
            self._handle.write(event)

    def close(self) -> Dict[str, Any]:
        """Close the dataset and return its statistics."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        self.stats = {
            "path": self.path,
            "format": self.format,
            "store": self.store,
            "precision": self.precision,
            "n_events": self.n_events,
            "bytes": os.path.getsize(self.path)
            if os.path.exists(self.path)
            else 0,
        }
        return self.stats


def write_events(
    array: Any,
    path: str,
    *,
    store: str = DEFAULT_STORE,
    precision: str = DEFAULT_PRECISION,
    fmt: Optional[str] = None,
    compression: str = "zstd",
    beams: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Write a batch of events to disk in one shot.

    A convenience wrapper around EventWriter for callers that already hold the
    whole sample. Chunked generation should use EventWriter directly.

    Parameters:
        array: Awkward Array of events, as produced by Pythia.nextBatch
        path: Destination path; the suffix selects the format unless `fmt`
            is given
        store: Store spec, one of the keys of STORE_SPECS
        precision: "f32" or "f64" for floating-point fields
        fmt: Explicit format override ("root", "jsonl" or "hepmc3")
        compression: Compression for ROOT output ("zstd", "lz4", or "none")
        beams: The two beam particles, required for HepMC3 (see EventWriter)

    Returns:
        Dict with "path", "format", "store", "precision", "n_events" and
        "bytes"

    Raises:
        ImportError: if the backend for the chosen format is unavailable
        ValueError: on an unknown format, store spec, or precision
    """
    with EventWriter(
        path,
        store=store,
        precision=precision,
        fmt=fmt,
        compression=compression,
        beams=beams,
    ) as writer:
        writer.write(array)
    return writer.stats


# ====================================================================== #
# ============================== Reading =============================== #
# ====================================================================== #


def load_events(
    path: str,
    *,
    columns: Optional[Sequence[str]] = None,
    fmt: Optional[str] = None,
) -> Any:
    """
    Read an event dataset into an Awkward Array.

    Parameters:
        path: Path to a .root, .jsonl or .hepmc dataset
        columns: Optional field names to read. Honoured natively by the ROOT
            backend, which reads only those columns off disk; applied after
            parsing for the row formats, which have no column structure
        fmt: Explicit format override ("root", "jsonl" or "hepmc3")

    Returns:
        Awkward Array with a `prt` field of per-particle records, and an `info`
        field where the source provides one

    Raises:
        FileNotFoundError: if the dataset does not exist
        ImportError: if the backend for the format is unavailable
        ValueError: on an unknown format
    """
    fmt = _checked_format(path, fmt)
    if fmt == "root":
        return _read_root(path, columns=columns)
    return _require_awkward().Array(list(_row_events(path, fmt, columns)))


def read_events(
    path: str,
    start: int,
    stop: int,
    *,
    fmt: Optional[str] = None,
) -> Any:
    """
    Read events [start, stop) of a dataset, without reading the rest.

    ROOT reads only the baskets that hold those entries. JSONL skips the lines
    before `start` without parsing them; HepMC3, which has no index, reads
    past them.

    Parameters:
        path: Path to a .root, .jsonl or .hepmc dataset
        start: Index of the first event
        stop: One past the index of the last event; clipped to the dataset

    Returns:
        Awkward Array of at most stop - start events, in the load_events layout
    """
    if start < 0 or stop < start:
        raise ValueError(f"invalid event range [{start}, {stop})")
    fmt = _checked_format(path, fmt)
    if fmt == "root":
        uproot = _require_uproot()
        with uproot.open(path) as handle:
            tree = handle[ROOT_TREE_NAME]
            stop = min(stop, tree.num_entries)
            return _regroup_particles(
                tree.arrays(entry_start=min(start, stop), entry_stop=stop)
            )
    events = itertools.islice(
        _row_events(path, fmt, None, skip=start), stop - start
    )
    return _require_awkward().Array(list(events))


def count_events(path: str, *, fmt: Optional[str] = None) -> int:
    """Number of events in a dataset, counted without parsing any event."""
    fmt = _checked_format(path, fmt)
    if fmt == "root":
        uproot = _require_uproot()
        with uproot.open(path) as handle:
            return int(handle[ROOT_TREE_NAME].num_entries)
    with _open_text(path, "r") as handle:
        if fmt == "jsonl":
            return sum(1 for line in handle if line.strip())
        # A HepMC3 ASCII event begins with an "E" line.
        return sum(1 for line in handle if line.startswith("E "))


def iter_events(
    path: str,
    *,
    step_size: int = 10000,
    columns: Optional[Sequence[str]] = None,
    fmt: Optional[str] = None,
) -> Iterator[Any]:
    """
    Read an event dataset in chunks, in the same layout as `load_events`.

    A dataset is written in chunks because it may not fit in memory; this is
    the matching reader, for consumers that make one pass over the events.

    Parameters:
        path: Path to a .root, .jsonl or .hepmc dataset
        step_size: Events per chunk
        columns: Optional field names to read, as for `load_events`
        fmt: Explicit format override ("root", "jsonl" or "hepmc3")

    Yields:
        Awkward Arrays of at most `step_size` events, in file order
    """
    step_size = max(1, int(step_size))
    fmt = _checked_format(path, fmt)
    if fmt == "root":
        uproot = _require_uproot()
        with uproot.open(path) as handle:
            tree = handle[ROOT_TREE_NAME]
            wanted = None
            if columns is not None:
                available = set(tree.keys())
                wanted = [
                    c if c in available else c.replace(".", "_")
                    for c in columns
                ]
            for chunk in tree.iterate(wanted, step_size=step_size):
                yield _regroup_particles(chunk)
        return
    ak = _require_awkward()
    rows = _row_events(path, fmt, columns)
    while True:
        chunk = list(itertools.islice(rows, step_size))
        if not chunk:
            return
        yield ak.Array(chunk)


def _checked_format(path: str, fmt: Optional[str]) -> str:
    """The dataset's format, after checking the file exists and the format is
    known."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"event dataset does not exist: {path}")
    fmt = fmt or detect_format(path)
    if fmt not in FORMATS:
        raise ValueError(f"unknown event format '{fmt}'")
    return fmt


def _row_events(
    path: str, fmt: str, columns: Optional[Sequence[str]], skip: int = 0
) -> Iterator[Dict[str, Any]]:
    """
    Stream the events of a row-oriented dataset (JSONL or HepMC3) as dicts in
    the columnar layout, after skipping the first `skip`.
    """
    if fmt == "jsonl":
        with _open_text(path, "r") as handle:
            for line in handle:
                if not line.strip():
                    continue
                if skip:
                    skip -= 1
                    continue
                yield _jsonl_row_to_event(json.loads(line), columns)
        return
    with _require_pyhepmc().open(path) as handle:
        for event in handle:
            if skip:
                skip -= 1
                continue
            yield _hepmc_to_event(event, columns)


def _read_root(path: str, *, columns: Optional[Sequence[str]]) -> Any:
    """Read a ROOT TTree, optionally projecting onto a subset of columns."""
    uproot = _require_uproot()

    with uproot.open(path) as handle:
        tree = handle[ROOT_TREE_NAME]
        if columns is None:
            arrays = tree.arrays()
        else:
            # A caller may name a column either as it is stored ("prt_id") or
            # as it appears in the regrouped layout ("prt.id").
            available = set(tree.keys())
            wanted = [
                c if c in available else c.replace(".", "_") for c in columns
            ]
            arrays = tree.arrays(wanted)
    return _regroup_particles(arrays)


def _regroup_particles(array: Any) -> Any:
    """
    Rebuild the nested `prt` record from uproot's split branches.

    A jagged record branch is written as one branch per field (`prt_id`,
    `prt_px`, ...) plus an `nprt` counter, and uproot returns it that way.
    Regrouping here keeps the layout a caller sees identical whichever format
    the dataset came from, so downstream tools never branch on the container.
    """
    ak = _require_awkward()

    particle_fields = {
        name[len("prt_") :]: array[name]
        for name in array.fields
        if name.startswith("prt_")
    }
    if not particle_fields:
        return array

    present = set(array.fields)
    regrouped: Dict[str, Any] = {"prt": ak.zip(particle_fields)}
    for name in array.fields:
        if name.startswith("prt_"):
            continue
        # uproot writes an "n<branch>" counter alongside every jagged branch:
        # "nweights" for the flat "weights" branch, "nprt" for the split
        # "prt_*" group. The length is implicit in the awkward layout, so the
        # counters are dropped rather than surfaced as fields of their own.
        if name.startswith("n"):
            counted = name[1:]
            if counted in present or any(
                f.startswith(counted + "_") for f in present
            ):
                continue
        regrouped[name] = array[name]
    return ak.Array(regrouped)


def _jsonl_row_to_event(
    row: Dict[str, Any], columns: Optional[Sequence[str]]
) -> Dict[str, Any]:
    """Convert one evtjsonl-1.0 row into the columnar event layout."""
    rows = []
    for particle in row.get("data", row).get("particles", []):
        # "E" and the per-particle index "i" belong to the JSONL schema only;
        # the columnar layout spells energy "e" and needs no index, since
        # position in the list is the index.
        record = {
            ("e" if name == "E" else name): value
            for name, value in particle.items()
            if name != "i"
        }
        if columns is not None:
            record = {k: v for k, v in record.items() if k in columns}
        rows.append(record)
    event = {"prt": rows, "weight": float(row.get("weight", 1.0))}
    if row.get("weights"):
        event["weights"] = [float(w) for w in row["weights"]]
    return event


def _hepmc_to_event(
    event: Any, columns: Optional[Sequence[str]]
) -> Dict[str, Any]:
    """Convert one HepMC3 event's final state into the columnar event
    layout."""
    arrays = event.numpy.particles
    final = arrays.status == _HEPMC_FINAL
    fields = {
        "id": arrays.pid[final].tolist(),
        "status": arrays.status[final].tolist(),
        "px": arrays.px[final].tolist(),
        "py": arrays.py[final].tolist(),
        "pz": arrays.pz[final].tolist(),
        "e": arrays.e[final].tolist(),
        "m": arrays.generated_mass[final].tolist(),
    }
    if columns is not None:
        fields = {
            name: values for name, values in fields.items() if name in columns
        }
    rows = [dict(zip(fields, values)) for values in zip(*fields.values())]
    weights = [float(w) for w in event.weights]
    out: Dict[str, Any] = {
        "prt": rows,
        "weight": weights[0] if weights else 1.0,
    }
    if len(weights) > 1:
        out["weights"] = weights
    return out


# ====================================================================== #
# ============================= Summaries ============================== #
# ====================================================================== #


class SummaryAccumulator:
    """
    Running description of a sample, accumulated batch by batch.

    A generated sample is far too large to open -- hundreds of particles per
    event -- so a summary built while writing is the only practical description
    of what a run produced. It is enough to tell whether a run card produced
    the process it was meant to: minimum bias and ttbar do not share a
    multiplicity, a leading pT, or a species mix.

    Incremental because generation writes in chunks and never holds the whole
    sample; `summarize` wraps this for callers that do.
    """

    def __init__(self) -> None:
        self.n_events = 0
        self.n_particles = 0
        self.n_particles_min: Optional[int] = None
        self.n_particles_max = 0
        self.leading_pt_sum = 0.0
        self.species: Dict[int, int] = {}

    def add(self, array: Any) -> None:
        """Fold one batch of events into the running totals."""
        ak = _require_awkward()
        import numpy as np

        if not len(array):
            return
        particles = array["prt"] if "prt" in array.fields else array
        counts = ak.to_numpy(ak.num(particles))

        self.n_events += len(array)
        self.n_particles += int(counts.sum())
        self.n_particles_max = max(self.n_particles_max, int(counts.max()))
        self.n_particles_min = (
            int(counts.min())
            if self.n_particles_min is None
            else min(self.n_particles_min, int(counts.min()))
        )

        pt = np.sqrt(particles["px"] ** 2 + particles["py"] ** 2)
        # An event that recorded no particles at all would otherwise put a None
        # into the mean.
        leading = ak.fill_none(ak.max(pt, axis=1), 0.0)
        self.leading_pt_sum += float(ak.sum(leading))

        ids, id_counts = np.unique(
            ak.to_numpy(ak.flatten(particles["id"])), return_counts=True
        )
        for pid, count in zip(ids.tolist(), id_counts.tolist()):
            self.species[pid] = self.species.get(pid, 0) + count

    def finalize(
        self,
        particle_names: Optional[Dict[str, str]] = None,
        top_species: int = 10,
    ) -> Dict[str, Any]:
        """
        Return the accumulated summary.

        Parameters:
            particle_names: Optional PDG id (as a string) to name mapping, so
                species are reported by name rather than by code
            top_species: How many of the most common species to report

        Returns:
            Dict with event count, particle multiplicity, mean leading pT and
            the most common species, or an empty dict if nothing was added
        """
        if not self.n_events:
            return {}
        names = particle_names or {}
        ranked = sorted(self.species.items(), key=lambda kv: -kv[1])[
            :top_species
        ]
        return {
            "n_events": self.n_events,
            "particles_per_event": {
                "mean": self.n_particles / self.n_events,
                "min": self.n_particles_min,
                "max": self.n_particles_max,
            },
            "leading_pt_mean_gev": self.leading_pt_sum / self.n_events,
            "top_species": [
                {
                    "id": int(pid),
                    "name": names.get(str(int(pid)), str(int(pid))),
                    "count": int(count),
                    "per_event": count / self.n_events,
                }
                for pid, count in ranked
            ],
        }


def summarize(
    array: Any,
    particle_names: Optional[Dict[str, str]] = None,
    top_species: int = 10,
) -> Dict[str, Any]:
    """
    Describe a sample without anyone having to read the events.

    Parameters:
        array: Awkward Array of events
        particle_names: Optional PDG id (as a string) to name mapping
        top_species: How many of the most common species to report

    Returns:
        Dict with event count, particle multiplicity, mean leading pT, and the
        most common species
    """
    accumulator = SummaryAccumulator()
    accumulator.add(array)
    return accumulator.finalize(
        particle_names=particle_names, top_species=top_species
    )


class WeightAccumulator:
    """
    Running statistics over the per-event generation weights.

    Kept incremental for the same reason as SummaryAccumulator: generation
    writes in chunks, and the weights have to be characterised without holding
    the sample.
    """

    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.minimum = float("inf")
        self.maximum = float("-inf")

    def add(self, array: Any) -> None:
        """Fold one batch of weights into the running totals."""
        nominal = _nominal_weights(array)
        if nominal is None or not len(nominal):
            return
        self.count += len(nominal)
        self.total += float(nominal.sum())
        self.minimum = min(self.minimum, float(nominal.min()))
        self.maximum = max(self.maximum, float(nominal.max()))

    def finalize(self) -> Dict[str, Any]:
        """
        Return the accumulated weight statistics.

        Returns:
            Dict with "weighted", "weight_sum", "weight_min", "weight_max" and
            "weight_mean", or an empty dict if nothing was added
        """
        if not self.count:
            return {}
        return {
            "weighted": not math.isclose(
                self.minimum, self.maximum, rel_tol=1e-9, abs_tol=0.0
            ),
            "weight_sum": self.total,
            "weight_min": self.minimum,
            "weight_max": self.maximum,
            "weight_mean": self.total / self.count,
        }


def _nominal_weights(array: Any) -> Any:
    """
    Extract the nominal per-event weight as a numpy array.

    Handles both layouts: straight from Pythia.nextBatch the weights are nested
    under `info.weights`, while a dataset read back from disk carries a flat
    `weight` column.
    """
    ak = _require_awkward()

    if "weight" in array.fields:
        return ak.to_numpy(array["weight"])
    if "info" in array.fields and "weights" in array["info"].fields:
        # Entry zero is the nominal weight; any others are variations.
        return ak.to_numpy(
            ak.fill_none(ak.firsts(array["info"]["weights"]), 1.0)
        )
    return None


def weight_stats(array: Any) -> Dict[str, Any]:
    """
    Summarise the per-event generation weights.

    Most runs are unweighted and every weight is 1.0. Cross-section biasing,
    ME/PS merging and shower uncertainty variations are not such runs, and a
    weighted sample analysed as though it were unweighted is wrong in shape,
    not merely in normalisation. `weighted` is the flag downstream tools should
    branch on.

    Parameters:
        array: Awkward Array of events, either straight from Pythia.nextBatch
            (weights nested under `info.weights`) or read back from a dataset
            (a flat `weight` column)

    Returns:
        Dict with "weighted", "weight_sum", "weight_min", "weight_max" and
        "weight_mean", or an empty dict when the array carries no weights
    """
    accumulator = WeightAccumulator()
    accumulator.add(array)
    return accumulator.finalize()
