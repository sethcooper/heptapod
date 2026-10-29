"""
# census.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Deterministic translation of a process census (EnumerateProcesses output,
FeynTable graph encoding) into EstimateDecayWidthNDA requests, and the
per-channel NDA estimates built from them.

Each census diagram becomes one native diagram dict that a person could have
written by hand and can rerun alone:

- particles: UFO spin, mass and colour at the supplied card;
- vertices: Lorentz type from the UFO structure, connectivity (``legs``),
  coupling magnitude evaluated at the card (CKM included), and the coupling's
  mass dimension;
- fermion vertices: chiral couplings gL, gR from the projector content of the
  vertex terms (an unprojected term counts in both);
- propagators: mass and width at the card, spin, the colour factor of the
  line's decay (for the narrow-width split);
- colour factor: the exact SU(3) colour sum of the diagram squared, averaged
  over the parent's colours, contracted from the UFO colour structures.

A channel's estimate combines its diagrams' estimates: amplitudes
sqrt(Gamma_i) carrying the phase of their coupling product add coherently
among diagrams that differ only in the flavour of internal lines (so CKM
unitarity cancels as in GIM), and incoherently otherwise; also per coupling
class. Each diagram carries its evaluated expression (evaluated.py) with the
model's coupling names.

Channels with an external massless vector boson are infrared sensitive at
tree level: with lower-multiplicity estimates they get the soft-collinear
estimate at the pair-mass regulator, otherwise none. Channels where a
zero-width particle can go on shell inside the decay (a lower-multiplicity
decay followed by that particle's) get the sequential narrow-width estimate
from lower-multiplicity estimates, otherwise none. Channels exchanging a
non-negligible photon or gluon internally get none.
"""
import hashlib
import itertools
import json
import math
import re
import string
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from tools.ufo.parser import parse_particles
from tools.ufo.evaluate import evaluate_parameters, evaluate_couplings, vertex_couplings
from tools.ufo.cards import card_overrides

SCHEMA = "heptapod-census-nda-0.1"
PRESCRIPTION = (
    "One EstimateDecayWidthNDA request per census diagram (UFO masses, widths, spins and "
    "colours at the card; vertex couplings evaluated at the card, chiral parts for fermion "
    "vertices; exact SU(3) colour factor; connectivity for propagator virtualities and the "
    "narrow-width split). Channel: coherent among diagrams differing only in internal-line "
    "flavour (coupling phases), incoherent otherwise; "
    "channels with an external massless vector boson get no estimate."
)
REGULATOR_FRACTION = 1e-6
PAIR_MASS_Y = 0.05 ** 2
# Lightest hadron (pi0): a particle lighter than this cannot decay into quarks.
HADRON_THRESHOLD_GEV = 0.1349768
COMBINATION = ("amplitudes sqrt(Gamma_i) with their coupling phases added coherently among diagrams "
               "that differ only in internal-line flavour; other diagrams added incoherently")
FIELD_DIMENSION = {1: 1.0, 2: 1.5, 3: 1.0}  # UFO 2s+1 -> mass dimension
COLOR_NAME = {3: "triplet", -3: "antitriplet", 8: "octet"}


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ── SU(3) colour algebra ────────────────────────────────────────────────────

def _su3():
    l = np.zeros((8, 3, 3), dtype=complex)
    l[0][0, 1] = l[0][1, 0] = 1
    l[1][0, 1], l[1][1, 0] = -1j, 1j
    l[2][0, 0], l[2][1, 1] = 1, -1
    l[3][0, 2] = l[3][2, 0] = 1
    l[4][0, 2], l[4][2, 0] = -1j, 1j
    l[5][1, 2] = l[5][2, 1] = 1
    l[6][1, 2], l[6][2, 1] = -1j, 1j
    l[7] = np.diag([1, 1, -2]) / math.sqrt(3)
    T = l / 2
    comm = np.einsum("aij,bjk->abik", T, T) - np.einsum("bij,ajk->abik", T, T)
    anti = np.einsum("aij,bjk->abik", T, T) + np.einsum("bij,ajk->abik", T, T)
    f = (-2j * np.einsum("abik,cki->abc", comm, T)).real
    d = (2 * np.einsum("abik,cki->abc", anti, T)).real
    eps = np.zeros((3, 3, 3))
    for p in itertools.permutations(range(3)):
        eps[p] = np.linalg.det(np.eye(3)[list(p)])
    return T, f, d, eps


T_SU3, F_SU3, D_SU3, EPS = _su3()
COLOR_TENSORS = {
    "T": (T_SU3, (8, 3, 3)),
    "f": (F_SU3, (8, 8, 8)),
    "d": (D_SU3, (8, 8, 8)),
    "Epsilon": (EPS, (3, 3, 3)),
    "EpsilonBar": (EPS, (3, 3, 3)),
}


def _split_top(expr: str, sep: str) -> List[str]:
    out, depth, cur = [], 0, ""
    for ch in expr:
        depth += ch == "("
        depth -= ch == ")"
        if ch == sep and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    return out + [cur]


def color_factors(color: str):
    """[(coefficient, name, indices)] for a UFO colour string that is a single
    product of factors (e.g. 'T(3,2,1)', 'f(1,2,3)', 'Identity(1,2)', '1')."""
    coefficient, factors = 1.0 + 0j, []
    for factor in _split_top(color.replace(" ", ""), "*"):
        m = re.fullmatch(r"(\w+)\(([-\d,]+)\)", factor)
        if m:
            factors.append((m.group(1), [int(x) for x in m.group(2).split(",")]))
        else:
            coefficient *= complex(eval(factor, {"__builtins__": {}}, {"complex": complex}))
    return coefficient, factors


def diagram_color_factor(vertex_colors, slot_edges, slot_reps, edges, initial_edge) -> float:
    """Sum over colours of |C|^2 divided by the parent's colour dimension.
    vertex_colors[k]: colour string of vertex k; slot_edges[k][s]: graph edge
    index at UFO slot s (1-based in the strings); slot_reps[k][s]: colour rep;
    edges: number of edges; initial_edge: index of the incoming edge."""
    letters = iter(string.ascii_letters)
    edge_label = {}
    edge_dim = {}
    operands, labels = [], []
    coefficient = 1.0 + 0j
    for k, color in enumerate(vertex_colors):
        c, factors = color_factors(color)
        coefficient *= c
        internal = {}
        for name, indices in factors:
            if name == "Identity":
                dims = []
                for i in indices:
                    dims.append(abs(slot_reps[k][i - 1]) if i > 0 else None)
                dim = next((x for x in dims if x), 3)
                tensor, dims = np.eye(dim), (dim, dim)
            elif name in COLOR_TENSORS:
                tensor, dims = COLOR_TENSORS[name]
            else:
                raise ValueError("Unsupported colour structure: " + name)
            letter = ""
            for i, dim in zip(indices, dims):
                if i > 0:
                    e = slot_edges[k][i - 1]
                    if e not in edge_label:
                        edge_label[e] = next(letters)
                    edge_dim[e] = dim
                    letter += edge_label[e]
                else:
                    if i not in internal:
                        internal[i] = next(letters)
                    letter += internal[i]
            operands.append(tensor)
            labels.append(letter)
    # A colour line whose two ends both appear stays contracted; the rest are free.
    counts = {}
    for letter in "".join(labels):
        counts[letter] = counts.get(letter, 0) + 1
    free = "".join(sorted({x for x in "".join(labels) if counts[x] == 1}))
    if operands:
        amplitude = np.einsum(",".join(labels) + "->" + free, *operands) * coefficient
        total = float(np.sum(np.abs(amplitude) ** 2))
    else:
        total = abs(coefficient) ** 2
    parent = edge_dim.get(initial_edge, 1)
    return total / parent


# ── Lorentz structure ───────────────────────────────────────────────────────

def _monomials(structure: str) -> List[str]:
    """Top-level additive terms, signs kept."""
    out, depth, cur = [], 0, ""
    for ch in structure.replace(" ", ""):
        if ch in "+-" and depth == 0 and cur and cur[-1] not in "*/(eE":
            out.append(cur)
            cur = ch
            continue
        depth += ch == "("
        depth -= ch == ")"
        cur += ch
    return [m for m in out + [cur] if m]


def _numeric(term: str) -> complex:
    value = 1.0 + 0j
    sign = -1 if term.startswith("-") else 1
    for factor in _split_top(term.lstrip("+-"), "*"):
        if not re.match(r"[A-Za-z]+\(", factor) or factor.startswith("complex("):
            value *= complex(eval(factor, {"__builtins__": {}}, {"complex": complex, "cmath": None}))
    return sign * value


def chiral_parts(structure: str):
    """(coefficient of P_L, coefficient of P_R) of a fermion Lorentz structure;
    a term with no projector counts in both."""
    left = right = 0j
    for term in _monomials(structure):
        c = _numeric(term)
        has_m, has_p = "ProjM(" in term, "ProjP(" in term
        left += c if has_m or not has_p else 0
        right += c if has_p or not has_m else 0
    return left, right


def derivatives(structure: str) -> int:
    first = _monomials(structure)[0] if structure.strip() else ""
    return first.count("P(")


# ── translation ─────────────────────────────────────────────────────────────

class CensusModel:
    """The UFO at the card, as the translator needs it."""

    def __init__(self, ufo_dir: str, param_card: Optional[str]):
        self.dir = str(ufo_dir)
        overrides = {}
        if param_card:
            overrides, missing = card_overrides(self.dir, str(param_card))
            if missing:
                raise ValueError("Parameter card missing external parameters: " + ", ".join(missing))
        values, unresolved = evaluate_parameters(self.dir, overrides)
        couplings, errors = evaluate_couplings(self.dir, overrides)
        if unresolved or errors:
            raise ValueError("Unresolved UFO parameters/couplings: " + str((unresolved, errors)))
        self.couplings = couplings
        self.values = values
        particles = parse_particles(str(Path(self.dir) / "particles.py"))
        self.by_var = {p["var"]: p for p in particles}
        self.by_pdg = {}
        for p in particles:
            def value(ref):
                z = complex(values[str(ref).split(".")[-1]])
                return abs(z.real)
            self.by_pdg[int(p["pdg_code"])] = {
                "name": p["name"],
                "spin": int(p["spin"]),
                "color": int(p["color"]),
                "charge": float(p.get("charge") or 0),
                "mass": value(p["mass"]),
                "width": value(p["width"]),
            }
        self.vertices = {v["name"]: v for v in vertex_couplings(self.dir)}

    def particle(self, pdg: int) -> Dict[str, Any]:
        p = self.by_pdg[pdg]
        out = {"label": p["name"], "spin": (abs(p["spin"]) - 1) / 2, "mass": p["mass"], "charge": p["charge"]}
        if p["color"] in COLOR_NAME:
            out["color"] = COLOR_NAME[p["color"]]
        return out

    def anti(self, pdg: int) -> int:
        return -pdg if -pdg in self.by_pdg else pdg


def _vertex_entry(model: CensusModel, vertex: Dict[str, Any], legs: List[str]) -> Dict[str, Any]:
    ufo = model.vertices[vertex["ufo_vertex"]]
    spins = [abs(int(model.by_var[v]["spin"])) for v in ufo["particles"]]
    terms = vertex["terms"]
    n = len(spins)
    dimension = round(4 - sum(FIELD_DIMENSION.get(s, 1.0) for s in spins) - derivatives(terms[0]["structure"]))
    fermions = spins.count(2)
    entry: Dict[str, Any] = {"legs": legs, "ufo_vertex": vertex["ufo_vertex"],
                             "ufo_couplings": sorted({t["coupling"] for t in terms})}
    if fermions == 2 and n == 3:
        gl = gr = 0j
        for t in terms:
            c = complex(model.couplings[t["coupling"]])
            left, right = chiral_parts(t["structure"])
            gl += c * left
            gr += c * right
        base = "vector-chiral" if 3 in spins else "yukawa-chiral"
        entry.update(type=f"{base}-3pt", coupling={"gL": abs(gl), "gR": abs(gr)})
    else:
        magnitude = math.sqrt(sum(abs(complex(model.couplings[t["coupling"]])) ** 2 for t in terms))
        if fermions == 4:
            base = "dim6-4fermion"
        elif dimension == 1:
            base = "scalar-3pt" if n == 3 else "custom"
        elif n == 4:
            base = "scalar-4pt"
        else:
            base = "gauge-vector"
        entry.update(type=f"{base}-{n}pt", coupling=magnitude)
    if dimension:
        entry["coupling_dimension"] = dimension
    return entry


def _slot_assignment(model: CensusModel, vertex: Dict[str, Any], incident: List[tuple]):
    """Map a vertex's incident edges (edge index, PDG leaving the vertex)
    onto the UFO vertex's particle slots."""
    ufo = model.vertices[vertex["ufo_vertex"]]
    slots = [int(model.by_var[v]["pdg_code"]) for v in ufo["particles"]]
    free = list(range(len(slots)))
    edges, reps = [None] * len(slots), [int(model.by_var[v]["color"]) for v in ufo["particles"]]
    for e, pdg in incident:
        s = next((s for s in free if slots[s] == pdg), None)
        if s is None:
            raise ValueError(f"Edge PDG {pdg} has no slot in {vertex['ufo_vertex']}")
        free.remove(s)
        edges[s] = e
    return edges, reps


def translate_diagram(model: CensusModel, graph: Dict[str, Any]) -> Dict[str, Any]:
    """Native EstimateDecayWidthNDA diagram dict for one census graph."""
    vertices = graph["vertices"]
    vid = {v["id"]: k for k, v in enumerate(vertices)}
    internal = [e for e in graph["edges"] if e["role"] == "internal"]
    legs = [[] for _ in vertices]
    incident = [[] for _ in vertices]
    for index, e in enumerate(graph["edges"]):
        a, b, pdg = e["from"], e["to"], e["pdg"]
        # UFO vertex particles are listed outgoing: a particle entering the
        # vertex occupies its antiparticle's slot.
        if a == "in:0":
            legs[vid[b]].append("in:0")
            incident[vid[b]].append((index, model.anti(pdg)))
        elif b.startswith("out:"):
            legs[vid[a]].append(b)
            incident[vid[a]].append((index, pdg))
        else:
            j = internal.index(e)
            legs[vid[a]].append("prop:" + str(j))
            legs[vid[b]].append("prop:" + str(j))
            incident[vid[a]].append((index, pdg))
            incident[vid[b]].append((index, model.anti(pdg)))
    final = [model.particle(p) for p in graph["final_pdgs"]]
    initial = [model.particle(graph["initial_pdgs"][0])]
    props = []
    for e in internal:
        p = model.by_pdg[e["pdg"]]
        props.append({"label": p["name"], "mass": p["mass"], "width": p["width"], "spin": (abs(p["spin"]) - 1) / 2})
    slots = [_slot_assignment(model, v, incident[k]) for k, v in enumerate(vertices)]
    colors = []
    for v in vertices:
        strings = sorted({t["color"] for t in v["terms"]})
        colors.append(strings[0])  # a vertex with several colour structures uses the first
    initial_edge = next(i for i, e in enumerate(graph["edges"]) if e["from"] == "in:0")
    color = diagram_color_factor(colors, [s[0] for s in slots], [s[1] for s in slots],
                                 len(graph["edges"]), initial_edge)
    diagram = {
        "initial": initial,
        "final": final,
        "vertices": [_vertex_entry(model, v, legs[k]) for k, v in enumerate(vertices)],
        "propagators": props,
        "color_factor": color,
        "check_quantum_numbers": False,
    }
    # Colour factor of each line's decay, for the narrow-width split.
    from .kinematics import far_side
    for j, (prop, side) in enumerate(zip(props, far_side(diagram))):
        region = _far_vertices(diagram, j)
        cut = graph["edges"].index(internal[j])
        sub_colors = [colors[k] for k in region]
        sub_edges = [slots[k][0] for k in region]
        sub_reps = [slots[k][1] for k in region]
        prop["decay_color_factor"] = diagram_color_factor(sub_colors, sub_edges, sub_reps, len(graph["edges"]), cut)
        prop["final_legs"] = side
    return diagram


from .evaluated import TEX_NAMES, BARE_NAMES, tex_name, bare_name, evaluated_formula, compact_formula, number  # noqa: E402,F401


def join_tex(parts) -> str:
    """Concatenate TeX names, with a space after a control word that a letter
    follows (\\gamma d d -> "\\gamma dd", not the undefined "\\gammadd")."""
    out = ""
    for part in parts:
        if out and part[:1].isalpha() and re.search(r"\\[A-Za-z]+$", out):
            out += " "
        out += part
    return out


def symbolic_diagram(model: CensusModel, graph: Dict[str, Any], request: Dict[str, Any]) -> Dict[str, Any]:
    """The same diagram for EstimateDecayWidthFormulaNDA: particle labels,
    vertex types and coupling names built from each vertex's particles
    (scalars, then vectors, then fermions), no numbers."""
    def particle(entry):
        out = {"label": tex_name(entry["label"]), "spin": entry["spin"], "massive": entry["mass"] > 0}
        return out

    def coupling_name(vertex):
        spins = {0: 0, 1: 0, 2: 2, 3: 1}
        names = [model.by_var[v] for v in model.vertices[vertex["ufo_vertex"]]["particles"]]

        def key(p):
            spin, pdg = abs(p["spin"]), abs(int(p["pdg_code"]))
            # Fermions: up-type quark before down-type, charged lepton before neutrino.
            second = (pdg % 2 if pdg <= 6 else (pdg + 1) % 2) if spin == 2 and pdg <= 16 else 0
            return ({1: 0, 3: 1, 2: 2}.get(spin, 3), second, pdg)

        order = sorted(names, key=key)
        return "g_{" + join_tex(bare_name(p["name"]) for p in order) + "}"

    vertices = []
    for v, numeric in zip(graph["vertices"], request["vertices"]):
        # One coupling name per vertex: a scaling formula needs the coupling,
        # not its chiral decomposition (the numeric request keeps that).
        vtype = re.sub(r"-\d+pt$", "", numeric["type"])
        vtype = {"vector-chiral": "gauge-vector", "yukawa-chiral": "yukawa"}.get(vtype, vtype)
        vertices.append({"type": vtype, "coupling": coupling_name(v)})
    return {
        "initial": [particle(x) for x in request["initial"]],
        "final": [particle(x) for x in request["final"]],
        "vertices": vertices,
        "propagators": [{"label": bare_name(x["label"]), "spin": x["spin"], "massive": x["mass"] > 0}
                        for x in request["propagators"]],
    }


def _far_vertices(diagram: Dict[str, Any], j: int) -> List[int]:
    vertices = diagram["vertices"]
    ends = {}
    start = None
    for k, v in enumerate(vertices):
        for leg in v["legs"]:
            kind, i = leg.split(":")
            if kind == "prop":
                ends.setdefault(int(i), []).append(k)
            elif kind == "in":
                start = k
    seen, todo = {start}, [start]
    while todo:
        k = todo.pop()
        for leg in vertices[k]["legs"]:
            kind, i = leg.split(":")
            if kind == "prop" and int(i) != j:
                for other in ends[int(i)]:
                    if other not in seen:
                        seen.add(other)
                        todo.append(other)
    return sorted(set(range(len(vertices))) - seen)


def coupling_phase(model: CensusModel, graph: Dict[str, Any]) -> complex:
    """Unit phase of the product of the diagram's vertex couplings (for a
    fermion vertex, of its larger chiral part)."""
    product = 1.0 + 0j
    for v in graph["vertices"]:
        gl = gr = 0j
        for t in v["terms"]:
            c = complex(model.couplings[t["coupling"]])
            if "ProjM(" in t["structure"] or "ProjP(" in t["structure"] or "Gamma(" in t["structure"]:
                left, right = chiral_parts(t["structure"])
                gl += c * left
                gr += c * right
            else:
                gl += c
        z = gl if abs(gl) >= abs(gr) else gr
        product *= z / abs(z) if abs(z) else 1.0
    return product


def skeleton(model: CensusModel, graph: Dict[str, Any]) -> str:
    """Diagrams with the same skeleton differ only in the flavour of internal
    lines (same spin, colour and charge); their amplitudes are added
    coherently, as in GIM cancellations."""
    edges = []
    for e in graph["edges"]:
        if e["role"] == "internal":
            p = model.by_pdg[e["pdg"]]
            tag = ["line", abs(p["spin"]), p["color"], p["charge"]]
        else:
            tag = ["external", e["pdg"]]
        edges.append([e["from"], e["to"], tag])
    vertices = [[v["id"], sorted({t["lorentz"] for t in v["terms"]})] for v in graph["vertices"]]
    return json.dumps([sorted(vertices), sorted(edges)], sort_keys=True)


def coupling_magnitude(vertex: Dict[str, Any]) -> float:
    c = vertex["coupling"]
    return math.sqrt(c["gL"] ** 2 + c["gR"] ** 2) if isinstance(c, dict) else float(c)


def channel_tex(labels) -> str:
    return " ".join(tex_name(x) for x in labels)


def diagram_formula(request: Dict[str, Any], symbolic: Optional[Dict[str, Any]], response: Dict[str, Any]):
    """One diagram's estimate in TeX, with the model's coupling names:
    {formula, definitions} in its parameters (compact_formula) and
    {formula_evaluated, definitions_evaluated} factor by factor, which
    reproduces the width when the definitions are substituted. The formula
    falls back to the evaluated one when there is no parametric form; all
    None when the estimate has no evaluated form."""
    parent = request["initial"][0]["label"]
    final = [x["label"] for x in request["final"]]
    if response.get("narrow_width"):
        terms, definitions = [], []
        for t in response["narrow_width"]["resonances"]:
            R = t["resonance"]
            far = t["resonance"]["final_legs"]
            rest = [x for i, x in enumerate(final) if i not in far]
            name = bare_name(R["label"])
            prod = r"\Gamma(%s\to %s)" % (tex_name(parent), channel_tex(rest + [R["label"]]))
            dec = r"\Gamma(%s\to %s)" % (tex_name(R["label"]), channel_tex([final[i] for i in far]))
            terms.append(r"\frac{%s\,%s}{\Gamma_{%s}}" % (prod, dec, name))
            definitions.append(r"%s=%s,\ %s=%s\,\mathrm{GeV}" % (
                prod, number(t["production"]["width_gev"]), dec, number(t["decay"]["width_gev"])))
        counts = {}
        for x in final:
            counts[x] = counts.get(x, 0) + 1
        sym = math.prod(math.factorial(k) for k in counts.values())
        formula = r"\Gamma=" + ("+".join(terms) if sym == 1 else r"\frac{1}{%d}\left(%s\right)" % (sym, "+".join(terms)))
        return {"formula": formula, "definitions": definitions,
                "formula_evaluated": formula, "definitions_evaluated": definitions}
    none = {"formula": None, "definitions": [], "formula_evaluated": None, "definitions_evaluated": []}
    if response.get("formula_type") != "evaluated" or symbolic is None:
        return none
    lines = [x["label"] for x in request["propagators"]]
    couplings = [v["coupling"] for v in symbolic["vertices"]]
    out = evaluated_formula(response["factors"], parent=parent, final=final, lines=lines, couplings=couplings,
                            coupling_values=[coupling_magnitude(v) for v in request["vertices"]])
    if not out:
        return none
    short = compact_formula(response["factors"], parent=parent, final=final, lines=lines, couplings=couplings)
    return {"formula": short["formula"] if short else out["formula"],
            "definitions": short["definitions"] if short else out["definitions"],
            "formula_evaluated": out["formula"], "definitions_evaluated": out["definitions"]}


def combine(diagrams: List[Dict[str, Any]]) -> float:
    """Coherent sum within each skeleton group, incoherent across groups."""
    groups: Dict[str, complex] = {}
    for d in diagrams:
        amplitude = math.sqrt(d["response"]["width_gev"]) * d["phase"]
        groups[d["skeleton"]] = groups.get(d["skeleton"], 0j) + amplitude
    return sum(abs(a) ** 2 for a in groups.values())


def infrared_sensitive(model: CensusModel, final_pdgs: List[int]) -> bool:
    return any(abs(model.by_pdg[p]["spin"]) == 3 and model.by_pdg[p]["mass"] == 0 for p in final_pdgs)


def far_side_mass(model: CensusModel, graph: Dict[str, Any], edge: Dict[str, Any]) -> float:
    """Summed mass of the final-state particles on the side of an internal line
    away from the parent: the line's virtuality is at least its square."""
    edges = [e for e in graph["edges"] if e is not edge]
    near, queue = {"in:0"}, ["in:0"]
    while queue:
        node = queue.pop()
        for e in edges:
            for a, b in ((e["from"], e["to"]), (e["to"], e["from"])):
                if a == node and b not in near:
                    near.add(b)
                    queue.append(b)
    return sum(model.by_pdg[e["pdg"]]["mass"] for e in graph["edges"]
               if e["to"].startswith("out:") and e["to"] not in near)


def internal_massless_vector(model: CensusModel, process: Dict[str, Any]) -> bool:
    """Some diagram exchanges a massless vector (photon, gluon) whose
    virtuality can reach zero: every final-state particle on its far side is
    massless, so the uncut width diverges in the collinear limit. A photon or
    gluon whose far side carries mass (gamma* -> tau+ tau-, g* -> b bbar) stays
    above that mass squared and gives a finite width."""
    return any(e["role"] == "internal" and abs(model.by_pdg[e["pdg"]]["spin"]) == 3
               and model.by_pdg[e["pdg"]]["mass"] == 0 and far_side_mass(model, d["graph"], e) == 0
               for d in process["diagrams"] for e in d["graph"]["edges"])


def split_exchanges(model: CensusModel, graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Photon and gluon lines of one diagram whose far side carries mass: for
    each, the vector's pdg and the final-state legs on its far side."""
    out = []
    for e in graph["edges"]:
        p = model.by_pdg[e["pdg"]]
        if e["role"] != "internal" or abs(p["spin"]) != 3 or p["mass"] != 0:
            continue
        if far_side_mass(model, graph, e) == 0:
            continue
        rest = [x for x in graph["edges"] if x is not e]
        near, queue = {"in:0"}, ["in:0"]
        while queue:
            node = queue.pop()
            for x in rest:
                for a, b in ((x["from"], x["to"]), (x["to"], x["from"])):
                    if a == node and b not in near:
                        near.add(b)
                        queue.append(b)
        legs = sorted(int(x["to"][4:]) for x in graph["edges"] if x["to"].startswith("out:") and x["to"] not in near)
        out.append({"vector": abs(e["pdg"]), "legs": legs})
    return out


def channel_key(model: CensusModel, initial: int, final: List[int]):
    """Order-free key of a channel, the same for its charge conjugate."""
    a = (initial, tuple(sorted(final)))
    b = (model.anti(initial), tuple(sorted(model.anti(f) for f in final)))
    return min(a, b)


def coloured(model: CensusModel, pdg: int) -> bool:
    return abs(model.by_pdg[pdg]["color"]) == 3


def closed(model: CensusModel, initial: int, final: List[int]) -> bool:
    """A decay into quarks of a particle lighter than the lightest hadron."""
    return model.by_pdg[initial]["mass"] < HADRON_THRESHOLD_GEV and any(coloured(model, f) for f in final)


def load_lower(model: CensusModel, paths) -> Dict[Any, Dict[str, Any]]:
    """Channels with a width from lower-multiplicity EstimateCatalogueNDA
    outputs (same model and card), by channel_key."""
    out = {}
    for path in paths or []:
        path = Path(path)
        summary = json.loads((path / "nda.json" if path.is_dir() else path).read_text())
        for row in summary["processes"]:
            if row.get("width_gev") is None or "initial_pdgs" not in row:
                continue
            out[channel_key(model, row["initial_pdgs"][0], row["final_pdgs"])] = row
    return out


def alpha_values(model: CensusModel):
    """alpha and alpha_s at the card."""
    v = model.values
    alpha = 1 / abs(complex(v["aEWM1"])) if "aEWM1" in v else abs(complex(v["aEW"]))
    return alpha, abs(complex(v["aS"]))


def flavour_tex(model: CensusModel, pdg: int) -> str:
    return bare_name(model.by_pdg[abs(pdg)]["name"])


def soft_collinear(model: CensusModel, process: Dict[str, Any], lower, y: float) -> Optional[Dict[str, Any]]:
    """Gamma(P -> X + vectors) ~ Gamma(P -> X) prod_V (c_V ln^2(1/y))^(n_V) / n_V!,
    c_gamma = sum_i alpha Q_i^2 / (2 pi), c_g = sum_i alpha_s C_F / (2 pi) over
    the final-state fermions of X that couple (charged; quarks). None without
    Gamma(P -> X); a dict with width_gev None when no fermion emits."""
    initial = process["initial_pdgs"][0]
    final = process["final_pdgs"]
    vectors = [f for f in final if abs(model.by_pdg[f]["spin"]) == 3 and model.by_pdg[f]["mass"] == 0]
    rest = [f for f in final if f not in vectors]
    base = lower.get(channel_key(model, initial, rest))
    if base is None:
        return None
    fermions = [f for f in rest if abs(model.by_pdg[f]["spin"]) == 2]
    alpha, alpha_s = alpha_values(model)
    L2 = math.log(1 / y) ** 2
    factor, parts, definitions = 1.0, [], [r"y=%s" % number(y)]
    for pdg in sorted(set(vectors), key=lambda p: model.by_pdg[p]["color"]):
        n = vectors.count(pdg)
        if model.by_pdg[pdg]["color"] == 8:
            emitters = [f for f in fermions if coloured(model, f)]
            k = len(emitters)
            c = k * alpha_s * 4 / 3 / (2 * math.pi)
            tex = r"\frac{\alpha_sC_F}{\pi}" if k == 2 else r"\frac{%s\alpha_sC_F}{2\pi}" % (k if k != 1 else "")
            definitions.append(r"\alpha_s=%s,\ C_F=\tfrac43" % number(alpha_s))
        else:
            emitters = [f for f in fermions if model.by_pdg[f]["charge"]]
            c = sum(model.by_pdg[f]["charge"] ** 2 for f in emitters) * alpha / (2 * math.pi)
            terms: Dict[str, int] = {}
            for f in emitters:
                t = "Q_{%s}^2" % flavour_tex(model, f) if coloured(model, f) else "1"
                terms[t] = terms.get(t, 0) + 1
            if len(terms) == 1:
                (t, k), = terms.items()
                t = "" if t == "1" else " " + t
                tex = r"\frac{\alpha%s}{\pi}" % t if k == 2 else r"\frac{%s\alpha%s}{2\pi}" % (k if k != 1 else "", t)
            else:
                tex = r"\frac{\alpha(%s)}{2\pi}" % "+".join(
                    (str(k) if k > 1 else "") + t for t, k in sorted(terms.items(), key=lambda x: x[0] == "1"))
            quarks = sorted({abs(f) for f in emitters if coloured(model, f)})
            definitions.append(r",\ ".join([r"\alpha=1/%s" % number(1 / alpha)] + [
                r"Q_{%s}=%s" % (flavour_tex(model, q), number(model.by_pdg[q]["charge"])) for q in quarks]))
        if not emitters:
            return {"width_gev": None, "reason": "no final-state fermion emits the massless vector: no collinear logarithm"}
        factor *= (c * L2) ** n / math.factorial(n)
        parts.append(tex + r"\ln^2\frac1y" if n == 1 else r"\frac12\left(%s\ln^2\frac1y\right)^2" % tex)
    base_tex = r"\Gamma(%s\to %s)" % (tex_name(process["initial"][0]),
                                         channel_tex([model.by_pdg[f]["name"] for f in rest]))
    definitions.insert(0, r"%s=%s\,\mathrm{GeV}" % (base_tex, number(base["width_gev"])))
    return {"width_gev": base["width_gev"] * factor, "factor": factor, "base": base["id"],
            "base_width_gev": base["width_gev"],
            "formula": r"\Gamma\sim%s\,%s" % (base_tex, r"\,".join(parts)), "definitions": definitions}


def splitting(model: CensusModel, process: Dict[str, Any], diagrams: List[Dict[str, Any]], lower) -> Optional[Dict[str, Any]]:
    """Photon or gluon emitted off the fermions of X and split into a massive
    pair F Fbar: Gamma(P -> X F Fbar) ~ Gamma(P -> X) c_V k_F/(3 pi) ln^3(M^2/4m_F^2)/3,
    with c_V the soft-collinear coefficient of X (as soft_collinear), k_F =
    alpha Q_F^2 N_F for a photon and alpha_s T_F for a gluon. The cube is the
    soft-collinear ln^2(M^2/q^2) of the emission integrated against the
    splitting's dq^2/q^2 from 4 m_F^2 to M^2. Summed over the distinct
    (V, F, X) of the diagrams' split lines; None when a Gamma(P -> X) is missing
    or no fermion of X emits."""
    initial = process["initial_pdgs"][0]
    final = process["final_pdgs"]
    M = model.by_pdg[initial]["mass"]
    alpha, alpha_s = alpha_values(model)
    seen, terms = set(), []
    for d in diagrams:
        for x in d["split"]:
            pair = [final[i] for i in x["legs"]]
            if len(pair) != 2 or pair[0] != -pair[1]:
                return None
            F = abs(pair[0])
            X = [f for i, f in enumerate(final) if i not in x["legs"]]
            key = (x["vector"], F, channel_key(model, initial, X))
            if key in seen:
                continue
            seen.add(key)
            base = lower.get(key[2])
            if base is None:
                return None
            fermions = [f for f in X if abs(model.by_pdg[f]["spin"]) == 2]
            mF = model.by_pdg[F]["mass"]
            L = math.log(M ** 2 / (4 * mF ** 2))
            if model.by_pdg[x["vector"]]["color"] == 8:
                emitters = [f for f in fermions if coloured(model, f)]
                c = len(emitters) * alpha_s * 4 / 3 / (2 * math.pi)
                k = alpha_s / 2
                c_tex, k_tex = r"c_g", r"\frac{\alpha_s T_F}{3\pi}"
                c_def = r"c_g=%d\,\frac{\alpha_s C_F}{2\pi}=%s" % (len(emitters), number(c))
            else:
                emitters = [f for f in fermions if model.by_pdg[f]["charge"]]
                c = sum(model.by_pdg[f]["charge"] ** 2 for f in emitters) * alpha / (2 * math.pi)
                nF = 3 if coloured(model, F) else 1
                k = alpha * model.by_pdg[F]["charge"] ** 2 * nF
                c_tex = r"c_\gamma"
                k_tex = (r"\frac{%s\alpha Q_{%s}^2}{3\pi}" % ("3" if nF == 3 else "", flavour_tex(model, F))
                         if coloured(model, F) else r"\frac{\alpha}{3\pi}")
                c_def = r"c_\gamma=\sum_{i\in X}\frac{\alpha Q_i^2}{2\pi}=%s" % number(c)
            if not emitters:
                continue
            width = base["width_gev"] * c * k / (3 * math.pi) * L ** 3 / 3
            base_tex = r"\Gamma(%s\to %s)" % (tex_name(process["initial"][0]),
                                               channel_tex([model.by_pdg[f]["name"] for f in X]))
            terms.append({"vector": x["vector"], "pair": F, "base": base["id"], "base_width_gev": base["width_gev"],
                          "c": c, "k": k, "log": L, "width_gev": width,
                          "tex": r"%s\,%s\,%s\,\tfrac13\ln^3\frac{M^2}{4m_{%s}^2}" % (base_tex, c_tex, k_tex,
                                                                                          flavour_tex(model, F)),
                          "definitions": [r"%s=%s\,\mathrm{GeV}" % (base_tex, number(base["width_gev"])), c_def,
                                          r"\ln\frac{M^2}{4m_{%s}^2}=%s" % (flavour_tex(model, F), number(L))]})
    if not terms:
        return None
    definitions = []
    for t in terms:
        definitions += [d for d in t["definitions"] if d not in definitions]
    return {"width_gev": sum(t["width_gev"] for t in terms), "terms": terms,
            "formula": r"\Gamma\sim" + "+".join(t["tex"] for t in terms), "definitions": definitions}


def sequential(model: CensusModel, process: Dict[str, Any], stable_ids: List[str], lower) -> Optional[Dict[str, Any]]:
    """On-shell part of a channel where a zero-width particle Y goes on shell:
    sum over distinct (Y, S) of Gamma(P -> X Y) Gamma(Y -> S) / Gamma_Y, with
    Gamma_Y the sum of Y's lower-multiplicity widths that are open. None
    when a factor is missing."""
    from .kinematics import far_side, virtuality_range
    initial = process["initial_pdgs"][0]
    final = process["final_pdgs"]
    M = model.by_pdg[initial]["mass"]
    masses = [model.by_pdg[f]["mass"] for f in final]
    seen, terms = set(), []
    for d in process["diagrams"]:
        if d["id"] not in stable_ids:
            continue
        graph = d["graph"]
        request = translate_diagram(model, graph)
        internal = [e for e in graph["edges"] if e["role"] == "internal"]
        sides = far_side(request)
        for j, (e, legs) in enumerate(zip(internal, sides)):
            p = model.by_pdg[e["pdg"]]
            lo, hi = virtuality_range(M, masses, legs)
            if p["width"] > 0 or not (lo < p["mass"] ** 2 < hi):
                continue
            # The particle flowing towards the far side.
            far = set(_far_vertices(request, j))
            vid = {v["id"]: k for k, v in enumerate(graph["vertices"])}
            Y = e["pdg"] if vid.get(e["to"]) in far else model.anti(e["pdg"])
            key = (Y, tuple(legs))
            if key in seen:
                continue
            seen.add(key)
            S = [final[i] for i in legs]
            X = [final[i] for i in range(len(final)) if i not in legs]
            production = lower.get(channel_key(model, initial, X + [Y]))
            decay = lower.get(channel_key(model, Y, S))
            total = [r for k, r in lower.items() if k[0] in (Y, model.anti(Y)) and not closed(model, Y, list(k[1]))]
            if production is None or decay is None:
                return None
            width_Y = sum(r["width_gev"] for r in total)
            open_decay = not closed(model, Y, S)
            terms.append({"Y": model.by_pdg[Y]["name"], "final_legs": legs, "production": production["id"],
                          "production_width_gev": production["width_gev"], "decay": decay["id"],
                          "decay_width_gev": decay["width_gev"] if open_decay else 0.0,
                          "width_Y_gev": width_Y, "width_Y_channels": [r["id"] for r in total],
                          "width_gev": production["width_gev"] * decay["width_gev"] / width_Y
                          if open_decay and width_Y > 0 else 0.0,
                          "X": [model.by_pdg[f]["name"] for f in X], "S": [model.by_pdg[f]["name"] for f in S]})
    if not terms:
        return None
    width = sum(t["width_gev"] for t in terms)
    parent = tex_name(process["initial"][0])
    pieces, definitions = [], []
    for t in terms:
        name = bare_name(t["Y"])
        prod = r"\Gamma(%s\to %s)" % (parent, channel_tex(t["X"] + [t["Y"]]))
        dec = r"\Gamma(%s\to %s)" % (tex_name(t["Y"]), channel_tex(t["S"]))
        pieces.append(r"\frac{%s\,%s}{\Gamma_{%s}}" % (prod, dec, name))
        definitions.append(r"%s=%s,\ %s=%s\,\mathrm{GeV}" % (prod, number(t["production_width_gev"]), dec,
                                                             number(t["decay_width_gev"])))
        definitions.append(r"\Gamma_{%s}=%s\,\mathrm{GeV}" % (name, number(t["width_Y_gev"])) + (
            r"\ (%d\ \mathrm{channels})" % len(t["width_Y_channels"])))
    return {"width_gev": width, "terms": terms, "formula": r"\Gamma\approx" + "+".join(pieces),
            "definitions": definitions}


def estimate_catalogue(catalogue_path, ufo_dir, param_card, output_dir, base_directory,
                       processes: Optional[List[str]] = None, lower_estimates: Optional[List[str]] = None,
                       pair_mass_y: float = PAIR_MASS_Y) -> Dict[str, Any]:
    """Translate and estimate every (or the selected) census process. Writes
    one file per process (each diagram's request and response) and nda.json."""
    from .nda_tool import EstimateDecayWidthNDATool
    from .nda_formula_tool import EstimateDecayWidthFormulaNDATool

    catalogue_path = Path(catalogue_path)
    if catalogue_path.is_dir():
        catalogue_path = catalogue_path / "catalogue.json"
    catalogue = json.loads(catalogue_path.read_text())
    point = catalogue.get("model", {}).get("parameter_point", {})
    if param_card and point.get("param_card_sha256") not in (None, sha(param_card)):
        raise ValueError("The parameter card is not the one the census was built with")
    files = {f["path"]: f["sha256"] for f in catalogue.get("model", {}).get("files", [])}
    for path, digest in files.items():
        if (Path(ufo_dir) / path).is_file() and sha(Path(ufo_dir) / path) != digest:
            raise ValueError("The UFO differs from the census model: " + path)
    model = CensusModel(ufo_dir, param_card)
    lower = load_lower(model, lower_estimates)
    selected = set(processes or [])
    output = Path(output_dir)
    (output / "estimates").mkdir(parents=True, exist_ok=False)
    rows = []
    for process in catalogue["processes"]:
        if selected and process["id"] not in selected:
            continue
        row = {"id": process["id"], "initial": process["initial"], "final": process["final"],
               "initial_pdgs": process["initial_pdgs"], "final_pdgs": process["final_pdgs"]}
        if infrared_sensitive(model, process["final_pdgs"]):
            row.update(status="requires_regulator",
                       reason="external massless vector boson: the uncut tree-level width is infrared divergent")
            radiative = soft_collinear(model, process, lower, pair_mass_y) if lower else None
            if radiative and radiative["width_gev"] is not None:
                row.update(status="estimated", width_gev=radiative["width_gev"], formula=radiative["formula"],
                           definitions=radiative["definitions"], methods=["soft-collinear"],
                           regulator={"pair_mass_y": pair_mass_y}, base=radiative["base"],
                           base_width_gev=radiative["base_width_gev"], emission_factor=radiative["factor"])
                row.pop("reason")
            elif radiative:
                row["reason"] = radiative["reason"]
            rows.append(row)
            continue
        diagrams, failures, stable = [], [], []
        for d in process["diagrams"]:
            try:
                request = translate_diagram(model, d["graph"])
                text = EstimateDecayWidthNDATool(
                    diagram=request, include_summary=False, record_finding=False,
                    process_label=process["id"], base_directory=str(base_directory))._run()
                if text.startswith("Error:"):
                    (stable if "On-Shell Intermediate Without Width" in text else failures).append(
                        {"diagram": d["id"], "error": text})
                    continue
                response = json.loads(text)
            except Exception as exc:  # recorded per diagram; the channel is then incomplete
                failures.append({"diagram": d["id"], "error": f"{type(exc).__name__}: {exc}"})
                continue
            if response.get("status") != "ok":
                failures.append({"diagram": d["id"], "error": response})
                continue
            entry = {"id": d["id"], "class_id": d["class_id"], "request": request, "response": response,
                     "skeleton": skeleton(model, d["graph"]), "phase": coupling_phase(model, d["graph"]),
                     "massless_exchange": internal_massless_vector(model, {"diagrams": [d]}),
                     "split": split_exchanges(model, d["graph"])}
            # The labelled formula for the same diagram, from the formula tool.
            try:
                symbolic = symbolic_diagram(model, d["graph"], request)
                ftext = EstimateDecayWidthFormulaNDATool(diagram=symbolic, base_directory=str(base_directory))._run()
                fresponse = json.loads(ftext) if not ftext.startswith("Error:") else {"status": "error", "error": ftext}
            except Exception as exc:
                symbolic, fresponse = None, {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            entry["formula_request"] = symbolic
            entry["formula_response"] = fresponse
            entry["formula_scaling"] = fresponse.get("formula") if fresponse.get("status") == "ok" else None
            entry.update(diagram_formula(request, symbolic, response))
            diagrams.append(entry)
        if stable:
            row.update(status="on_shell_intermediate",
                       reason="a zero-width particle can go on shell inside this decay: the final state is a "
                              "lower-multiplicity decay followed by that particle's decay",
                       diagrams=[x["diagram"] for x in stable])
            chain = sequential(model, process, row["diagrams"], lower) if lower else None
            if chain and chain["width_gev"] > 0:
                row.update(status="estimated", width_gev=chain["width_gev"], formula=chain["formula"],
                           definitions=chain["definitions"], methods=["sequential narrow width"],
                           sequential=chain["terms"])
                row.pop("reason")
            elif chain:
                row.update(status="unphysical", sequential=chain["terms"],
                           reason="the on-shell particle decays only into quarks while lighter than the lightest "
                                  "hadron: an artifact of massless quarks on the card")
            rows.append(row)
            continue
        # A photon or gluon exchange that can reach zero virtuality and is not
        # negligible (at least REGULATOR_FRACTION of the largest diagram) makes
        # the uncut width divergent; one suppressed by small couplings
        # (h -> u ubar gamma* via y_u) does not.
        top = max((x["response"]["width_gev"] for x in diagrams), default=0.0)
        if any(x["massless_exchange"] and x["response"]["width_gev"] >= REGULATOR_FRACTION * top for x in diagrams):
            row.update(status="requires_regulator",
                       reason="internal massless vector boson with massless daughters: the uncut tree-level width "
                              "diverges in the collinear limit; it is defined with an invariant-mass cut")
            rows.append(row)
            continue
        # A photon or gluon into a massive pair: the midpoint of its virtuality
        # misses the pole at q^2 -> 4 m_F^2, so those diagrams are replaced by
        # the splitting estimate; the other diagrams keep theirs (incoherent).
        split = None
        if lower and any(x["split"] and x["response"]["width_gev"] >= REGULATOR_FRACTION * top for x in diagrams):
            split = splitting(model, process, [x for x in diagrams if x["split"]], lower)
        kept = [x for x in diagrams if not (split and x["split"])]
        classes = []
        for c in process["classes"]:
            members = [d for d in diagrams if d["class_id"] == c["id"]]
            top_d = max(members, key=lambda x: x["response"]["width_gev"]) if members else None
            replaced = bool(split) and bool(members) and all(d["split"] for d in members)
            classes.append({"id": c["id"], "orders": c["orders"], "diagram_ids": c["diagram_ids"],
                            "width_gev": combine(members),
                            **{k: top_d[k] if top_d else (None if k.startswith("formula") else [])
                               for k in ("formula", "definitions", "formula_evaluated", "definitions_evaluated")},
                            "method": "splitting" if replaced else "diagram"})
        rest = combine(kept)
        total = rest + (split["width_gev"] if split else 0.0)
        dominant = max(kept, key=lambda x: x["response"]["width_gev"]) if kept else None
        if split:
            formula = split["formula"] + (r"+\Gamma_{\mathrm{other}}" if kept else "")
            definitions = split["definitions"] + ([r"\Gamma_{\mathrm{other}}=%s\,\mathrm{GeV}\ (%d\ \mathrm{diagrams})"
                                                   % (number(rest), len(kept))] if kept else [])
        else:
            formula = dominant["formula"] if dominant else None
            definitions = dominant["definitions"] if dominant else []
        evaluated = {"formula_evaluated": dominant["formula_evaluated"], "definitions_evaluated":
                     dominant["definitions_evaluated"]} if dominant and not split else {}
        for d in diagrams:
            d["phase"] = [d["phase"].real, d["phase"].imag]
        if split:
            row.update(splitting=split["terms"], other_width_gev=rest)
        row.update(status="estimated" if not failures else "incomplete", width_gev=total, formula=formula,
                   definitions=definitions, **evaluated, formula_diagram=None if split else dominant["id"] if dominant else None,
                   formula_fraction=None if split else dominant["response"]["width_gev"] / total if dominant and total else None,
                   classes=[{k: c[k] for k in ("id", "orders", "width_gev", "formula", "definitions",
                                               "formula_evaluated", "definitions_evaluated", "method")}
                            for c in classes],
                   methods=(["collinear splitting"] if split else []) + sorted({d["response"]["method"] for d in kept}),
                   combination=COMBINATION)
        if failures:
            row["failures"] = failures
        (output / "estimates" / (process["id"] + ".json")).write_text(json.dumps(
            {"process": process["id"], "width_gev": total, "formula": formula, "definitions": definitions, **evaluated,
             "formula_diagram": None if split else dominant["id"] if dominant else None, "classes": classes,
             "diagrams": diagrams, "splitting": split["terms"] if split else None,
             "failures": failures}, indent=1, default=str))
        row["estimate"] = "estimates/" + process["id"] + ".json"
        rows.append(row)
    summary = {
        "schema": SCHEMA,
        "catalogue": str(catalogue_path),
        "catalogue_sha256": sha(catalogue_path),
        "param_card_sha256": sha(param_card) if param_card else None,
        "prescription": PRESCRIPTION,
        "processes": rows,
    }
    (output / "nda.json").write_text(json.dumps(summary, indent=1))
    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return {"status": "ok" if not counts.get("incomplete") else "partial", "counts": counts,
            "summary": str(output / "nda.json")}
