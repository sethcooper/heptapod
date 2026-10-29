"""Independent MG5 check of a 1->N catalogue (any N >= 2); diagram generation only.

MG5 regenerates every channel in scope from the same explicit UFO, in unitary
gauge, with its own census over the physical states. Each MG5 diagram is
rebuilt as a FeynTable graph: MG5's currents give the topology, and internal
lines are oriented from the UFO vertices' field content, so no MG5 sign
convention is assumed. Catalogue and MG5 graphs are compared by canonical key.
No event generation or rate claim.
"""

from collections import Counter, defaultdict
import json
import logging
import re
import shutil
import sys
import tempfile
from pathlib import Path
from .catalogue import SCHEMA, amplitude_class, selected_orders, sha
from .graph_codec import signature

NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def antiparticles(model):
    by_name = {p["name"]: p["pdg"] for p in model.parts.values()}
    return {p["pdg"]: by_name[p["antiname"]] for p in model.parts.values()}


def orient_edges(graph, model, anti):
    """Set internal PDGs from the UFO all-outgoing field content of each vertex."""
    remaining = {
        v["id"]: Counter(model.vertex_data[v["ufo_vertex"]]["particle_pdgs"])
        for v in graph["vertices"]
    }
    internal = []
    for e in graph["edges"]:
        if e["role"] == "internal":
            internal.append(e)
        elif e["from"] == "in:0":
            remaining[e["to"]].subtract([anti[e["pdg"]]])
        else:
            remaining[e["from"]].subtract([e["pdg"]])
    while internal:
        for e in list(internal):
            done = False
            for endpoint, other in [(e["from"], e["to"]), (e["to"], e["from"])]:
                if sum(x["from"] == endpoint or x["to"] == endpoint for x in internal) != 1:
                    continue
                counts = remaining[endpoint]
                if any(v < 0 for v in counts.values()) or sum(counts.values()) != 1:
                    raise ValueError("UFO/graph incidence mismatch")
                field = next(k for k, v in counts.items() if v == 1)
                e["pdg"] = field if endpoint == e["from"] else anti[field]
                remaining[endpoint].subtract([field])
                remaining[other].subtract([anti[field]])
                internal.remove(e)
                done = True
                break
            if done:
                break
        else:
            raise ValueError("Disconnected or cyclic tree")
    if any(any(counts.values()) for counts in remaining.values()):
        raise ValueError("Vertex field incidence mismatch")
    return graph


def check_scope(model, scope):
    if model.info["source"] != "ufo":
        raise ValueError("Independent MG5 validation requires the same explicit UFO, not an assumed built-in SM match")
    if scope["kind"] != "decay" or scope["loops"] != 0 or type(scope["final_multiplicity"]) is not int or scope["final_multiplicity"] < 2:
        raise ValueError("Supports tree-level 1->N decays, N >= 2")
    if (scope.get("gauge") != "unitary" or scope.get("kinematics") != "strict_on_shell"
            or scope.get("flavors") != "resolved" or scope.get("charge_conjugates") != "separate"):
        raise ValueError("Unsupported scope conventions")
    parents, orders = scope["parents"], scope["orders"] or {}
    if not parents or len(parents) != len(set(parents)) or any(x not in model.physical for x in parents):
        raise ValueError("Invalid parent scope")
    if any(not NAME.fullmatch(k) or type(v) != int or v < 0 for k, v in orders.items()):
        raise ValueError("Invalid coupling-order scope")


def mg_inventory(model, scope, mg5_path):
    """{process id: Counter of canonical graph keys} from MG5 for the scope."""
    check_scope(model, scope)
    n, orders = scope["final_multiplicity"], scope["orders"] or {}
    anti = antiparticles(model)
    sys.path.insert(0, str(Path(mg5_path).resolve()))
    from madgraph.interface.master_interface import MasterCmd
    from madgraph.core.diagram_generation import NoDiagramException

    logging.disable(logging.CRITICAL)
    lookup = defaultdict(list)
    for v in model.raw_vertices:
        key = (
            tuple(model.vertex_data[v["name"]]["particle_pdgs"]),
            tuple(v["lorentz"]),
            tuple(sorted(((t["color_index"], t["lorentz_index"]), t["coupling"]) for t in v["couplings"])),
        )
        lookup[key].append(v["name"])
    # Upper bounds only: a tree has at most n-1 vertices. Exact scope orders
    # are applied to the rebuilt graphs, not left to MG5's syntax.
    bounds = {}
    for vertex in model.vertex_data.values():
        for term in vertex["terms"]:
            for name, power in term["orders"].items():
                if not NAME.fullmatch(name):
                    raise ValueError("Unsupported coupling-order name")
                bounds[name] = max(bounds.get(name, 0), power)
    constraints = "".join(" " + k + "<=" + str(v * (n - 1)) for k, v in sorted(bounds.items()))
    byid = {p["pdg"]: p for p in model.physical.values()}
    records, coverage = {}, []
    with tempfile.TemporaryDirectory() as temp:
        copy = Path(temp).resolve() / ("catalogue_ufo_" + Path(temp).name)
        shutil.copytree(model.interface.model, copy,
                        ignore=shutil.ignore_patterns("restrict_*.dat", "*.pkl", "__pycache__"))
        cmd = MasterCmd()
        cmd.exec_cmd("set gauge unitary", printcmd=False)
        cmd.exec_cmd("import model " + str(copy), printcmd=False)
        # MG5 may rename particles to its own conventions (e.g. MSSM sfermions):
        # address every state by PDG code through the model MG5 loaded.
        def mg_name(pdg):
            part = cmd._curr_model.get_particle(abs(pdg))
            if part is None:
                raise ValueError("MG5 model lacks PDG " + str(pdg))
            return part.get("name") if pdg > 0 or part.get("self_antipart") else part.get("antiname")
        cmd.exec_cmd("define census_states = " + " ".join(mg_name(p["pdg"]) for p in model.physical.values()), printcmd=False)
        for parent in scope["parents"]:
            command = "generate " + mg_name(model.parts[parent]["pdg"]) + " > " + " ".join(["census_states"] * n) + constraints
            try:
                cmd.exec_cmd(command, printcmd=False)
            except NoDiagramException:
                coverage.append({"parent": parent, "command": command, "status": "no_diagrams"})
                continue
            coverage.append({"parent": parent, "command": command, "status": "generated"})
            for amp in cmd._curr_amps:
                finals = sorted(amp.get("process").get_final_ids())
                final = [byid[p] for p in finals]
                if not model.open(model.parts[parent], final):
                    continue
                id = model.process_id(model.parts[parent], final)
                legs = list(amp.get("process").get("legs"))
                outgoing = sorted([e for e in legs if e.get("state")], key=lambda e: (e.get("id"), e.get("number")))
                slots = {e.get("number"): "out:" + str(i) for i, e in enumerate(outgoing)}
                slots.update({e.get("number"): "in:0" for e in legs if not e.get("state")})
                keys = Counter()
                for d in amp.get("diagrams"):
                    currents, vertices, edges = dict(slots), [], []
                    chain = d.get("vertices")
                    for index, v in enumerate(chain):
                        i = cmd._curr_model.get_interaction(v.get("id"))
                        names = lookup.get((
                            tuple(p.get_pdg_code() for p in i.get("particles")),
                            tuple(i.get("lorentz")),
                            tuple(sorted(i.get("couplings").items())),
                        ), [])
                        if len(names) != 1:
                            raise ValueError("Ambiguous MG/UFO interaction mapping: " + id)
                        vid = "v" + str(index)
                        terms = model.vertex(names[0], None)["terms"]
                        if not terms:
                            break
                        vertices.append({"id": vid, "ufo_vertex": names[0], "terms": terms})
                        inputs = v.get("legs") if index == len(chain) - 1 else v.get("legs")[:-1]
                        for leg in inputs:
                            node = currents.pop(leg.get("number"))
                            if node == "in:0":
                                edges.append({"from": node, "to": vid, "pdg": model.parts[parent]["pdg"], "role": "external"})
                            elif node.startswith("out:"):
                                edges.append({"from": vid, "to": node, "pdg": finals[int(node[4:])], "role": "external"})
                            else:
                                edges.append({"from": node, "to": vid, "pdg": 0, "role": "internal"})
                        if index < len(chain) - 1:
                            currents[v.get("legs")[-1].get("number")] = vid
                    else:
                        if currents:
                            raise ValueError("Unconsumed MG currents: " + id)
                        graph = orient_edges({"initial_pdgs": [model.parts[parent]["pdg"]], "final_pdgs": finals,
                                              "vertices": vertices, "edges": edges}, model, anti)
                        if selected_orders(amplitude_class(vertices)[0], orders):
                            keys[signature(graph, n, anti.get)] += 1
                if keys:
                    records[id] = keys
    return records, {
        "coverage": coverage,
        "gauge": "unitary",
        "model_restrictions": "none; restrict_*.dat omitted from scratch copy",
        "mg5_version": (Path(mg5_path) / "VERSION").read_text().strip(),
        "mg5_version_sha256": sha(Path(mg5_path) / "VERSION"),
        "comparison": "canonical FeynTable graphs (topology, signed lines, UFO vertex terms); classes recomputed from vertex terms",
        "kind": "diagram generation, not numerical integration",
    }


def validate_catalogue(model, catalogue_path, mg5_path):
    path = Path(catalogue_path)
    data = json.loads(path.read_text())
    findings = []
    if data["schema_version"] != SCHEMA:
        raise ValueError("Unknown catalogue schema")
    if data["model"]["model_sha256"] != model.info["model_sha256"] or data["model"]["parameter_point"] != model.point:
        raise ValueError("Model or parameter point mismatch")
    expected, execution = mg_inventory(model, data["scope"], mg5_path)
    n = data["scope"]["final_multiplicity"]
    anti = antiparticles(model)
    rows = data["processes"]
    actual = {p["id"]: p for p in rows}
    if len(actual) != len(rows):
        findings.append({"code": "duplicate_ids"})
    for id in sorted(expected.keys() - actual.keys()):
        findings.append({"id": id, "code": "missing_process"})
    for id in sorted(actual.keys() - expected.keys()):
        findings.append({"id": id, "code": "extra_process"})
    for id in sorted(expected.keys() & actual.keys()):
        a = actual[id]
        parent = model.parts.get(a["initial"][0]) if a.get("initial") else None
        final = [model.parts.get(x) for x in a.get("final", [])]
        if (parent is None or None in final or model.process_id(parent, final) != id
                or a["initial_pdgs"] != [parent["pdg"]] or a["final_pdgs"] != [p["pdg"] for p in final]
                or a["final_pdgs"] != sorted(a["final_pdgs"]) or a["loops"] != 0):
            findings.append({"id": id, "code": "process_identity"})
            continue
        keys, by_diagram = Counter(), {}
        for d in a["diagrams"]:
            g = d["graph"]
            try:
                if g["initial_pdgs"] != a["initial_pdgs"] or g["final_pdgs"] != a["final_pdgs"]:
                    raise ValueError("graph externals differ from the process")
                for v in g["vertices"]:
                    if v["ufo_vertex"] not in model.vertex_data or v["terms"] != model.vertex(v["ufo_vertex"], None)["terms"]:
                        raise ValueError("vertex terms differ from the UFO: " + str(v.get("ufo_vertex")))
                keys[signature(g, n, anti.get)] += 1
                by_diagram[d["id"]] = amplitude_class(g["vertices"])
            except (KeyError, TypeError, ValueError) as exc:
                findings.append({"id": id, "code": "graph_invalid", "diagram": d.get("id"), "detail": str(exc)})
                continue
            graph_path = (path.parent / d["graph_path"]).resolve()
            if not graph_path.is_relative_to(path.parent.resolve()):
                raise ValueError("Escaped graph artifact")
            if json.loads(graph_path.read_text()) != g:
                findings.append({"id": id, "code": "graph_file_mismatch", "diagram": d["id"]})
        # Keys quotient identical-particle permutations, so diagrams that differ
        # by exchanging identical legs share a key: compare with multiplicity.
        if keys != expected[id]:
            findings.append({"id": id, "code": "diagram_mismatch",
                             "missing": sum((expected[id] - keys).values()), "extra": sum((keys - expected[id]).values())})
        ids = [d["id"] for d in a["diagrams"]]
        if len(set(ids)) != len(ids):
            findings.append({"id": id, "code": "duplicate_diagram_ids"})
        members = [i for c in a["classes"] for i in c["diagram_ids"]]
        if sorted(members) != sorted(ids) or len({c["id"] for c in a["classes"]}) != len(a["classes"]):
            findings.append({"id": id, "code": "class_partition"})
        seen = set()
        for c in a["classes"]:
            key = json.dumps([c["orders"], c["coupling_monomials"]], sort_keys=True)
            if key in seen:
                findings.append({"id": id, "code": "class_not_maximal", "class": c["id"]})
            seen.add(key)
            for did in c["diagram_ids"]:
                if did in by_diagram and list(by_diagram[did]) != [c["orders"], c["coupling_monomials"]]:
                    findings.append({"id": id, "code": "class_orders_or_monomials", "diagram": did})
        for d in a["diagrams"]:
            if not any(d["id"] in c["diagram_ids"] and d.get("class_id") == c["id"] for c in a["classes"]):
                findings.append({"id": id, "code": "class_membership", "diagram": d["id"]})
    return {
        "schema_version": "heptapod-catalogue-validation-0.2",
        "status": "passed" if not findings else "failed",
        "catalogue_sha256": sha(path),
        "model_sha256": model.info["model_sha256"],
        "parameter_point": model.point,
        "expected_processes": len(expected),
        "submitted_processes": len(rows),
        "findings": findings,
        "execution": execution,
    }
