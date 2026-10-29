"""Bounded tree-level 1->N decay census (any N >= 2) using explicit FeynGraph models.

No loops or scattering. Numerical coupling/kinematic filters require a parsed
UFO (or explicit built-in masses). With a UFO, diagrams are in unitary gauge:
Goldstone and ghost propagators are dropped, as MadGraph does by default.

Graphs use the FeynTable encoding: vertices {id, ufo_vertex, terms}; directed
edges {from, to, pdg, role}, with the parent at in:0 and the daughters at
out:0..N-1 in numerically sorted PDG order; an internal edge's PDG is the
particle flowing from `from` to `to`. A coupling class is the set of diagrams
with the same amplitude orders and the same set of coupling monomials (one
term chosen at every vertex).
"""

from __future__ import annotations
import hashlib
import itertools
import json
import math
from collections import Counter
from pathlib import Path

from .feyngraph_interface import FeynGraphInterface
from tools.ufo.parser import parse_particles
from tools.ufo.evaluate import (
    evaluate_parameters,
    evaluate_couplings,
    parse_coupling_orders,
    vertex_couplings,
    is_ghost,
    is_goldstone,
)
from tools.ufo.cards import card_overrides
from tools.ufo.lorentz import parse_lorentz

SCHEMA = "heptapod-process-catalogue-0.2"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    Path(path).write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def real_mass(value):
    z = complex(value)
    if not math.isfinite(z.real) or not math.isfinite(z.imag) or abs(z.imag) > 1e-10:
        raise ValueError("Mass must be finite and real")
    return abs(z.real)


def expansion_orders(directory):
    """The UFO's truncations: coupling orders whose expansion_order is below
    the UFO default of 99 (e.g. an EFT operator inserted at most once)."""
    import ast
    path = Path(directory) / "coupling_orders.py"
    out = {}
    if not path.exists():
        return out
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "CouplingOrder":
            kw = {k.arg: k.value for k in node.keywords}
            name, limit = kw.get("name"), kw.get("expansion_order")
            if isinstance(name, ast.Constant) and isinstance(limit, ast.Constant) and type(limit.value) is int and limit.value < 99:
                out[name.value] = limit.value
    return out


def selected_orders(actual, requested):
    return all(actual.get(k, 0) == v for k, v in (requested or {}).items())


class ProcessModel:
    def __init__(self, model="SM", base_directory=None, param_card=None, masses=None):
        self.interface = FeynGraphInterface(model, base_directory)
        self.info = self.interface.model_info()
        self.parts = {p["name"]: p for p in self.info["particles"]}
        if len({p["pdg"] for p in self.parts.values()}) != len(self.parts):
            raise ValueError("Particle PDGs must be unique")
        self.vertex_data = {}
        self.expansion = {}
        self.charges = None
        self.masses = None
        self.point = {"source": "not_evaluated", "param_card_sha256": None}
        self.raw_vertices = []
        if self.info["source"] == "ufo":
            directory = self.interface.model
            parsed = parse_particles(str(Path(directory) / "particles.py"))
            declarations = {p["name"]: p for p in parsed}
            if set(declarations) != set(self.parts):
                raise ValueError("UFO/FeynGraph particle inventory mismatch")
            self.physical = {
                n: p
                for n, p in self.parts.items()
                if not is_ghost(declarations[n]) and not is_goldstone(declarations[n])
            }
            self.charges = {n: float(declarations[n].get("charge") or 0) for n in declarations}
            overrides = {}
            if param_card:
                overrides, missing = card_overrides(directory, str(param_card))
                if missing:
                    raise ValueError(
                        "Parameter card missing external parameters: "
                        + ", ".join(missing)
                    )
            values, unresolved = evaluate_parameters(directory, overrides)
            coupling_values, errors = evaluate_couplings(directory, overrides)
            if unresolved or errors:
                raise ValueError(
                    "Unresolved UFO parameters/couplings: " + str((unresolved, errors))
                )
            if any(
                not math.isfinite(complex(v).real) or not math.isfinite(complex(v).imag)
                for v in coupling_values.values()
            ):
                raise ValueError("Nonfinite coupling")
            self.masses = {
                n: real_mass(values[str(p["mass"]).split(".")[-1]])
                for n, p in declarations.items()
            }
            self.point = {
                "source": "param_card" if param_card else "ufo_defaults",
                "param_card_sha256": sha(param_card) if param_card else None,
            }
            lorentz = parse_lorentz(directory)
            orders = parse_coupling_orders(directory)
            self.expansion = expansion_orders(directory)
            variables = {p["var"]: p for p in parsed}
            self.raw_vertices = vertex_couplings(directory)
            if len(self.raw_vertices) != len(self.interface._model_obj.vertices()):
                raise ValueError("UFO/FeynGraph vertex inventory mismatch")
            for v in self.raw_vertices:
                terms = []
                for term in v["couplings"]:
                    ci, li = term["color_index"], term["lorentz_index"]
                    name = term["coupling"]
                    if abs(coupling_values[name]) == 0:
                        continue
                    lname = v["lorentz"][li]
                    terms.append(
                        dict(
                            color_index=ci,
                            lorentz_index=li,
                            color=v["color"][ci],
                            lorentz=lname,
                            structure=lorentz[lname]["structure"],
                            coupling=name,
                            orders=orders[name],
                        )
                    )
                self.vertex_data[v["name"]] = {
                    "name": v["name"],
                    "particles": [variables[n]["name"] for n in v["particles"]],
                    "particle_pdgs": [
                        int(variables[n]["pdg_code"]) for n in v["particles"]
                    ],
                    "terms": terms,
                }
        else:
            if param_card:
                raise ValueError(
                    "A parameter card requires an explicit UFO, not the built-in SM"
                )
            self.physical = {
                n: p
                for n, p in self.parts.items()
                if not n.startswith("gh") and n not in {"G0", "G+", "G-"}
            }
            self.point["limitations"] = [
                "Built-in model has no exposed numeric parameter/term interface; no zero-coupling filtering"
            ]
        if masses is not None:
            if self.info["source"] == "ufo":
                raise ValueError(
                    "Use a parameter card for UFO masses, not a second inconsistent mass map"
                )
            resolved = {
                self.interface.resolve_label(k): real_mass(v) for k, v in masses.items()
            }
            if set(self.physical) - resolved.keys():
                raise ValueError(
                    "Explicit built-in masses must cover every physical particle"
                )
            self.masses = resolved
            self.point["source"] = "explicit_masses"
            self.point["masses"] = resolved
        self.info["physical_particles"] = sorted(self.physical)
        self.info["excluded_particles"] = sorted(set(self.parts) - set(self.physical))
        self.info["parameter_point"] = self.point

    def vertex(self, name, orders):
        if self.info["source"] == "builtin":
            return {
                "name": name,
                "terms": None,
                "terms_status": "not_exposed_by_builtin_model",
            }
        v = dict(self.vertex_data[name])
        v["terms"] = [t for t in v["terms"] if selected_orders(t["orders"], orders)]
        return v

    def process_id(self, parent, final):
        return (
            "model-"
            + self.info["model_sha256"][:12]
            + "-tree-"
            + str(parent["pdg"])
            + "-to-"
            + "-".join(str(p["pdg"]) for p in final)
        )

    def open(self, parent, final):
        return self.masses is None or self.masses[parent["name"]] > sum(
            self.masses[p["name"]] for p in final
        )



def diagram_record(model, diagram, parent, final, orders):
    """FeynTable graph, amplitude orders and coupling monomials of one diagram,
    or None when it is out of scope (unphysical propagator, zero coupling)."""
    if any(p.particle().name() not in model.physical for p in diagram.propagators()):
        return None  # unitary gauge: no Goldstone or ghost lines
    vid = {v.id(): "v" + str(k) for k, v in enumerate(diagram.vertices())}
    vertices = []
    for v in diagram.vertices():
        data = model.vertex(v.interaction().name(), None)
        if model.info["source"] == "ufo" and not data["terms"]:
            return None
        vertices.append({"id": vid[v.id()], "ufo_vertex": data["name"], "terms": data["terms"]})
    leg = diagram.incoming()[0]
    edges = [{"from": "in:0", "to": vid[leg.vertex().id()], "pdg": parent["pdg"], "role": "external"}]
    free = {}
    for i, p in enumerate(final):
        free.setdefault(p["name"], []).append(i)
    for leg in diagram.outgoing():
        slot = free[leg.particle().name()].pop(0)
        edges.append({"from": vid[leg.vertex().id()], "to": "out:" + str(slot), "pdg": final[slot]["pdg"], "role": "external"})
    for p in diagram.propagators():
        a, b = p.vertices()
        edges.append({"from": vid[a.id()], "to": vid[b.id()], "pdg": p.particle().pdg(), "role": "internal"})
    graph = {
        "initial_pdgs": [parent["pdg"]],
        "final_pdgs": [p["pdg"] for p in final],
        "vertices": vertices,
        "edges": edges,
    }
    if model.info["source"] != "ufo":
        return graph, diagram.orders(), None
    amplitude, monomials = amplitude_class(vertices)
    if amplitude != {k: n for k, n in diagram.orders().items() if n}:
        raise ValueError("FeynGraph and UFO coupling orders disagree")
    if not selected_orders(amplitude, orders):
        return None
    if not orders and any(amplitude.get(k, 0) > v for k, v in model.expansion.items()):
        return None  # beyond the model's own truncation (as MG5 by default)
    return graph, amplitude, monomials


def amplitude_class(vertices):
    """Amplitude coupling orders and the distinct coupling monomials (one term
    chosen at every vertex) of a graph's vertices."""
    total = Counter()
    for v in vertices:
        per = {json.dumps(t["orders"], sort_keys=True) for t in v["terms"]}
        if len(per) != 1:
            raise ValueError("Mixed coupling orders within a vertex are not supported: " + v["ufo_vertex"])
        total.update(v["terms"][0]["orders"])
    monomials = {
        json.dumps(dict(Counter(choice)), sort_keys=True)
        for choice in itertools.product(*[sorted({t["coupling"] for t in v["terms"]}) for v in vertices])
    }
    return {k: n for k, n in total.items() if n}, [json.loads(m) for m in sorted(monomials)]


def enumerate_processes(
    model, output, *, parents=None, orders=None, final_multiplicity=2,
    max_candidates=20000, render=True
):
    import feyngraph as fg

    if type(final_multiplicity) is not int or final_multiplicity < 2:
        raise ValueError("final_multiplicity must be an integer >= 2")
    n = final_multiplicity
    if orders is not None and (
        not isinstance(orders, dict)
        or any(type(v) != int or v < 0 for v in orders.values())
    ):
        raise ValueError("Orders must map coupling names to nonnegative integers")
    if type(max_candidates) != int or max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    selected = (
        [model.interface.resolve_label(x) for x in parents]
        if parents is not None
        else list(model.physical)
    )
    if (
        not selected
        or len(selected) != len(set(selected))
        or any(x not in model.physical for x in selected)
    ):
        raise ValueError("Parents must be distinct physical particles")
    particles = sorted(model.physical.values(), key=lambda p: p["pdg"])
    total = len(selected) * math.comb(len(particles) + n - 1, n)
    if total > max_candidates:
        raise ValueError(
            f"Candidate budget exceeded: {total} > {max_candidates}; restrict parents or raise the explicit budget"
        )
    selector = fg.DiagramSelector()
    for k, v in (orders or {}).items():
        selector.select_coupling_power(k, v)
    rows = []
    exclusions = []
    counts = Counter()
    output = Path(output)
    for name in sorted(selected, key=lambda x: model.parts[x]["pdg"]):
        parent = model.parts[name]
        for final in itertools.combinations_with_replacement(particles, n):
            id = model.process_id(parent, final)
            if model.charges is not None and abs(
                model.charges[name] - sum(model.charges[p["name"]] for p in final)
            ) > 1e-9:
                exclusions.append({"id": id, "reason": "charge_violating"})
                counts["charge_violating"] += 1
                continue
            if not model.open(parent, final):
                exclusions.append({"id": id, "reason": "kinematically_closed"})
                counts["kinematically_closed"] += 1
                continue
            diagrams = model.interface.enumerate_diagrams(
                [name], [p["name"] for p in final], selector=selector
            )
            accepted = []
            classes = {}
            for diagram in diagrams:
                record = diagram_record(model, diagram, parent, final, orders)
                if record is None:
                    continue
                graph, amplitude, monomials = record
                diagram_id = id + "-d" + str(len(accepted) + 1)
                folder = output / "diagrams"
                folder.mkdir(exist_ok=True)
                graph_path = folder / (diagram_id + ".json")
                dump(graph_path, graph)
                key = json.dumps([amplitude, monomials], sort_keys=True)
                group = classes.setdefault(
                    key,
                    {"id": "c" + str(len(classes) + 1), "orders": amplitude,
                     "coupling_monomials": monomials, "diagram_ids": []},
                )
                group["diagram_ids"].append(diagram_id)
                entry = {
                    "id": diagram_id,
                    "class_id": group["id"],
                    "graph": graph,
                    "graph_path": graph_path.relative_to(output).as_posix(),
                }
                if render:
                    svg = folder / (diagram_id + ".svg")
                    tex = folder / (diagram_id + ".tex")
                    diagram.draw_svg(str(svg))
                    diagram.draw_tikz(str(tex))
                    entry.update(
                        image=svg.relative_to(output).as_posix(),
                        tikz=tex.relative_to(output).as_posix(),
                    )
                accepted.append(entry)
            if not accepted:
                reason = (
                    "no_diagrams" if not diagrams else "zero_or_out_of_scope_couplings"
                )
                exclusions.append({"id": id, "reason": reason})
                counts[reason] += 1
                continue
            rows.append(
                {
                    "id": id,
                    "initial": [name],
                    "final": [p["name"] for p in final],
                    "initial_pdgs": [parent["pdg"]],
                    "final_pdgs": [p["pdg"] for p in final],
                    "loops": 0,
                    "diagrams": accepted,
                    "classes": list(classes.values()),
                }
            )
            counts["accepted"] += 1
    catalogue = {
        "schema_version": SCHEMA,
        "model": model.info,
        "scope": {
            "kind": "decay",
            "loops": 0,
            "final_multiplicity": n,
            "parents": selected,
            "orders": orders,
            "gauge": "unitary" if model.info["source"] == "ufo" else "builtin",
            "expansion_orders": None if orders else (model.expansion or None),
            "kinematics": (
                "strict_on_shell" if model.masses is not None else "not_applied"
            ),
            "flavors": "resolved",
            "charge_conjugates": "separate",
        },
        "coverage": {
            "candidates": total,
            "outcomes": dict(counts),
            "complete_within_scope": sum(counts.values()) == total,
        },
        "processes": rows,
    }
    dump(output / "catalogue.json", catalogue)
    dump(output / "exclusions.json", exclusions)
    return catalogue
