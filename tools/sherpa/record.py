"""
# record.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Sherpa's event record as a heptapod dataset batch.

Sherpa keeps an event as a list of blobs, each a vertex with incoming and
outgoing particles: the beams, the signal process, the showers, colour
reconnection, fragmentation and every hadron decay. This module turns that
into the columnar batch the shared generation loop and ``event_io`` write,
with the same layout for both backends: the sherpa-mc wheel read in-process
here, and the C++ driver (``sherpa_driver.cc``), which emits the rows this
module parses.

The particle columns are those of the Pythia datasets where the meaning is
the same (id, status, four-momentum, mass, production vertex) plus the
genealogy in the form Sherpa has it: ``prod_vertex`` and ``end_vertex`` name
the blob a particle came from and went into, -1 for none, and ``info`` is
Sherpa's one-letter production-step flag. ``status`` is the HepMC status, 1
for a final-state particle and otherwise negative: -4 for a beam, -2 for a
hadron or lepton that decayed, -3 for a documentation entry, -11 for
everything intermediate. The final-state store specs therefore select
``status > 0`` exactly as they do for Pythia, and ``store="full"`` reaches
HepMC3 as the vertex graph Sherpa's own HepMC3 output describes.
"""

from typing import Any, Dict, List, Optional, Sequence

#: The columns of a particle row, as the driver writes them and the wheel
#: backend builds them.
ROW = (
    "id",
    "status",
    "info",
    "prod_vertex",
    "end_vertex",
    "px",
    "py",
    "pz",
    "e",
    "m",
    "vx",
    "vy",
    "vz",
    "vt",
)
_INT_FIELDS = ("id", "status", "info", "prod_vertex", "end_vertex")

#: Per-event scalars, in the order of the "i" entry of an event line: the
#: number of trials behind the event, the matrix-element weight, and the
#: squared renormalisation and factorisation scales in GeV^2.
INFO_FIELDS = ("trials", "me_weight", "mu_r2", "mu_f2")

#: HepMC statuses, stored negative for particles that are not final.
FINAL = 1
DECAYED = 2
DOCUMENTATION = 3
BEAM = 4
INTERMEDIATE = 11

#: Sherpa's part_status codes.
_ACTIVE = 1
_DOCUMENTATION = 3

#: Masses of the particles that appear as beams, in GeV, for the beam
#: four-momenta when Sherpa's own particle data cannot be asked.
BEAM_MASSES = {
    11: 0.000510999,
    13: 0.1056584,
    22: 0.0,
    2212: 0.93827209,
    2112: 0.93956542,
}


def hepmc_status(
    *,
    has_production: bool,
    has_decay: bool,
    sherpa_status: int,
    decays_alone: bool,
    unstable: bool,
    breaks_up: bool = False,
) -> int:
    """
    The HepMC status of one particle, signed as the dataset stores it.

    A particle from no blob is a beam. One that is active and goes into no
    blob is final. One whose decay blob has it as the only incoming particle
    and gives out no partons, and which is a hadron, muon or tau, decayed.
    Sherpa's documentation entries keep that meaning. Everything else is
    intermediate -- including the beam hadron's copy that breaks up into
    its partons, which Rivet would otherwise take for a decayed hadron and
    count every particle of the event as non-prompt.
    """
    if not has_production:
        return -BEAM
    if not has_decay and sherpa_status == _ACTIVE:
        return FINAL
    if sherpa_status == _DOCUMENTATION:
        return -DOCUMENTATION
    if has_decay and decays_alone and unstable and not breaks_up:
        return -DECAYED
    return -INTERMEDIATE


def _is_parton(flavour: Any, kf: int) -> bool:
    """A quark, gluon or diquark: what a beam hadron breaks up into."""
    try:
        return bool(
            flavour.IsQuark() or flavour.IsGluon() or flavour.IsDiQuark()
        )
    except Exception:
        return kf <= 6 or kf == 21 or (1100 <= kf <= 5505 and kf % 100 < 10)


def _is_unstable(kf: int, flavour: Any) -> bool:
    """A hadron, muon or tau: what the decayed status is for."""
    if kf in (13, 15):
        return True
    try:
        return bool(flavour.IsHadron())
    except Exception:
        return kf > 100


def event_rows(blobs: Any) -> Dict[str, Any]:
    """
    One event from the sherpa-mc wheel as the row layout the driver writes.

    Particles are identified by object, not by Sherpa's particle number:
    colour reconnection copies a particle under the same number, and the
    copies are distinct vertices' business. Rows come out in the order the
    blobs list them, which the HepMC3 writer reorders topologically anyway.

    Parameters:
        blobs: The ``Blob_List`` of one generated event

    Returns:
        ``{"p": rows, "i": scalars, "w": [nominal weight]}``
    """
    vertex_of: Dict[int, int] = {}
    for k in range(blobs.size()):
        vertex_of[blobs[k].Id()] = k

    rows: List[List[float]] = []
    seen: Dict[int, int] = {}

    def add(particle: Any) -> None:
        key = int(particle.this)
        if key in seen:
            return
        seen[key] = len(rows)
        flavour = particle.Flav()
        kf = int(flavour.Kfcode())
        pid = -kf if flavour.IsAnti() else kf
        has_production = bool(particle.HasProdBlob())
        has_decay = bool(particle.HasDecBlob())
        prod = (
            vertex_of.get(particle.ProdBlob().Id(), -1)
            if has_production
            else -1
        )
        end = vertex_of.get(particle.DecBlob().Id(), -1) if has_decay else -1
        breaks_up = False
        if has_decay:
            decay = particle.DecBlob()
            breaks_up = any(
                _is_parton(
                    decay.OutPart(j).Flav(),
                    int(decay.OutPart(j).Flav().Kfcode()),
                )
                for j in range(decay.NOutP())
            )
        status = hepmc_status(
            has_production=has_production,
            has_decay=has_decay,
            sherpa_status=int(particle.Stat()),
            decays_alone=has_decay and particle.DecBlob().NInP() == 1,
            unstable=_is_unstable(kf, flavour),
            breaks_up=breaks_up,
        )
        momentum = particle.Momentum()
        origin = particle.XProd()
        rows.append(
            [
                pid,
                status,
                ord(particle.Info()),
                prod,
                end,
                float(momentum[1]),
                float(momentum[2]),
                float(momentum[3]),
                float(momentum[0]),
                float(momentum.Mass()),
                float(origin[1]),
                float(origin[2]),
                float(origin[3]),
                float(origin[0]),
            ]
        )

    for k in range(blobs.size()):
        blob = blobs[k]
        for j in range(blob.NInP()):
            add(blob.InPart(j))
        for j in range(blob.NOutP()):
            add(blob.OutPart(j))

    signal = blobs.GetFirst(1)
    scalars = [
        _blob_value(signal, "Trials", 1.0),
        _blob_value(signal, "MEWeight", 0.0),
        _blob_value(signal, "Renormalization_Scale", 0.0),
        _blob_value(signal, "Factorization_Scale", 0.0),
    ]
    return {"p": rows, "i": scalars, "w": [float(blobs.Weight())]}


def _blob_value(blob: Any, name: str, default: float) -> float:
    try:
        value = blob[name]
    except Exception:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def beams_from_card(
    card: Dict[str, Any], mass_of: Optional[Any] = None
) -> List[Dict[str, float]]:
    """
    The two beams of a run from its card's ``BEAMS`` and ``BEAM_ENERGIES``.

    Sherpa accepts one value for both beams or a pair. Beam 1 travels along
    +z and beam 2 along -z. Masses come from ``mass_of(kf)`` when given
    (Sherpa's own particle data), else from a small table.

    Raises:
        ValueError: when the card does not name both beams and their
            energies
    """
    beams = card.get("BEAMS")
    energies = card.get("BEAM_ENERGIES")
    if beams is None or energies is None:
        raise ValueError("the card needs BEAMS and BEAM_ENERGIES")
    ids = [int(x) for x in (beams if isinstance(beams, list) else [beams] * 2)]
    e = [
        float(x)
        for x in (energies if isinstance(energies, list) else [energies] * 2)
    ]
    if len(ids) == 1:
        ids = ids * 2
    if len(e) == 1:
        e = e * 2
    if len(ids) != 2 or len(e) != 2:
        raise ValueError("BEAMS and BEAM_ENERGIES name one or two values")
    out = []
    for pid, energy, sign in ((ids[0], e[0], 1.0), (ids[1], e[1], -1.0)):
        mass = None
        if mass_of is not None:
            try:
                mass = float(mass_of(pid))
            except Exception:
                mass = None
        if mass is None:
            mass = BEAM_MASSES.get(abs(pid), 0.0)
        pz = sign * (max(energy * energy - mass * mass, 0.0)) ** 0.5
        out.append(
            {
                "id": pid,
                "px": 0.0,
                "py": 0.0,
                "pz": pz,
                "e": energy,
                "m": mass,
            }
        )
    return out


def untangle(rows: List[List[float]]) -> List[List[float]]:
    """
    Make the vertex graph of one event acyclic, in place.

    Sherpa's shower blob is one vertex for both showers: it takes the beam
    partons in and gives out the partons that enter the hard process, and
    takes the hard process's products in and gives out their showered
    descendants. The hard-process vertex and the shower vertex therefore
    feed each other, and no topological order exists. Every such mutual pair
    is split: the shower vertex keeps what comes from the hard process and
    what descends from it, and a new vertex takes the rest -- the incoming
    beam partons and the outgoing partons that enter the hard process. A
    cycle that is not of this form is broken by detaching the particle that
    closes it from its end vertex.

    Parameters:
        rows: An event's particle rows (see ROW); the vertex columns are
            rewritten

    Returns:
        The same rows
    """
    PROD, END = 3, 4
    for _ in range(64):
        produced_by: Dict[int, List[int]] = {}
        consumed_by: Dict[int, List[int]] = {}
        for i, row in enumerate(rows):
            if row[PROD] >= 0:
                produced_by.setdefault(int(row[PROD]), []).append(i)
            if row[END] >= 0:
                consumed_by.setdefault(int(row[END]), []).append(i)
        cycle = _find_cycle(rows, produced_by)
        if cycle is None:
            return rows
        fresh = (
            max(max(produced_by, default=-1), max(consumed_by, default=-1)) + 1
        )
        if len(cycle) == 2:
            u, v = cycle
            # v feeds u and u feeds v: split v (the vertex with more
            # outgoing particles, the shower) as described above.
            if len(produced_by.get(u, [])) > len(produced_by.get(v, [])):
                u, v = v, u
            into_u = {i for i in produced_by[v] if int(rows[i][END]) == u}
            from_u = {i for i in consumed_by[v] if int(rows[i][PROD]) == u}
            for i in into_u:
                rows[i][PROD] = fresh
            for i in consumed_by[v]:
                if i not in from_u:
                    rows[i][END] = fresh
        else:
            closing = cycle[-1]
            for i in consumed_by.get(cycle[0], []):
                if int(rows[i][PROD]) == closing:
                    rows[i][END] = -1
                    if rows[i][1] < 0:
                        rows[i][1] = -INTERMEDIATE
                    break
    return rows


def _find_cycle(
    rows: List[List[float]], produced_by: Dict[int, List[int]]
) -> Optional[List[int]]:
    """One cycle of the vertex graph as a list of vertices (the last one
    feeds the first), or None."""
    edges: Dict[int, List[int]] = {}
    for vertex, particles in produced_by.items():
        for i in particles:
            end = int(rows[i][4])
            if end >= 0:
                edges.setdefault(vertex, []).append(end)
    state: Dict[int, int] = {}
    stack: List[int] = []

    def visit(vertex: int) -> Optional[List[int]]:
        state[vertex] = 1
        stack.append(vertex)
        for nxt in edges.get(vertex, ()):
            if state.get(nxt) == 1:
                return stack[stack.index(nxt) :]
            if nxt not in state:
                found = visit(nxt)
                if found:
                    return found
        stack.pop()
        state[vertex] = 2
        return None

    for vertex in sorted(edges):
        if vertex not in state:
            found = visit(vertex)
            if found:
                return found
    return None


def batch_from_events(events: Sequence[Dict[str, Any]]) -> Any:
    """
    The Awkward Array for a sequence of event rows, in the layout the shared
    generation loop and ``event_io`` expect.

    Parameters:
        events: Parsed events, each ``{"p": rows, "i": scalars, "w":
            weights}`` with the nominal weight first

    Returns:
        An array with a ``prt`` field of particle records -- ``p`` and
        ``vProd`` nested as Pythia nests them, plus ``prod_vertex``,
        ``end_vertex`` and ``info`` -- and an ``info`` record with the
        scalars and a ragged ``weights``
    """
    import awkward as ak
    import numpy as np

    counts = np.fromiter(
        (len(ev["p"]) for ev in events), dtype=np.int64, count=len(events)
    )
    for ev in events:
        untangle(ev["p"])
    rows = [row for ev in events for row in ev["p"]]
    flat = (
        np.asarray(rows, dtype=np.float64)
        if rows
        else np.zeros((0, len(ROW)), dtype=np.float64)
    )
    column = {name: flat[:, k] for k, name in enumerate(ROW)}
    for name in _INT_FIELDS:
        column[name] = column[name].astype(np.int32)

    fields: Dict[str, Any] = {
        "id": column["id"],
        "status": column["status"],
        "info": column["info"],
        "prod_vertex": column["prod_vertex"],
        "end_vertex": column["end_vertex"],
        "p": ak.zip({axis: column[axis] for axis in ("px", "py", "pz", "e")}),
        "m": column["m"],
        "vProd": ak.zip(
            {
                "px": column["vx"],
                "py": column["vy"],
                "pz": column["vz"],
                "e": column["vt"],
            }
        ),
    }
    particles = ak.unflatten(ak.zip(fields, depth_limit=1), counts)

    scalars = (
        np.asarray([ev["i"] for ev in events], dtype=np.float64)
        if events
        else np.zeros((0, len(INFO_FIELDS)), dtype=np.float64)
    )
    info: Dict[str, Any] = {
        name: scalars[:, k] for k, name in enumerate(INFO_FIELDS)
    }
    weights = [ev["w"] for ev in events]
    info["weights"] = ak.unflatten(
        np.fromiter((w for ws in weights for w in ws), dtype=np.float64),
        np.fromiter(
            (len(ws) for ws in weights), dtype=np.int64, count=len(weights)
        ),
    )
    return ak.Array({"prt": particles, "info": ak.zip(info, depth_limit=1)})
