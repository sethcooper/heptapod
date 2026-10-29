"""Canonical keys for small tree graphs in the FeynTable encoding.

A key is invariant under internal vertex renaming and under permutations of
identical external particles, so graphs from different generators compare
equal exactly when they are the same diagram with the same vertex terms.
"""

import itertools
import json


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"))


def require(value, message):
    if not value:
        raise ValueError(message)


def signature(graph, n, anti):
    """Canonical key of a 1->n tree graph. anti(pdg) gives the antiparticle PDG
    (itself for a self-conjugate field): reversing an internal edge conjugates it."""
    vertices = graph["vertices"]
    edges = graph["edges"]
    ids = [v["id"] for v in vertices]
    require(1 <= len(ids) <= n - 1 and len(ids) == len(set(ids)), "Invalid tree vertices")
    externals = ["in:0"] + ["out:" + str(i) for i in range(n)]
    require(not set(ids) & set(externals), "Vertex/external ID collision")
    counts = dict.fromkeys(ids + externals, 0)
    adj = {v: set() for v in ids}
    for edge in edges:
        a, b = edge["from"], edge["to"]
        require(a != b and a in counts and b in counts, "Invalid edge endpoint")
        require(type(edge["pdg"]) is int and edge["pdg"] != 0, "Invalid edge PDG")
        counts[a] += 1
        counts[b] += 1
        external = (a in externals) + (b in externals)
        if external:
            if a == "in:0":
                require(edge["pdg"] == graph["initial_pdgs"][0] and b in ids, "Incoming edge mismatch")
            elif b.startswith("out:"):
                require(a in ids and edge["pdg"] == graph["final_pdgs"][int(b.split(":")[1])],
                        "Outgoing edge mismatch")
            else:
                raise ValueError("External momentum orientation mismatch")
        require(external <= 1 and edge["role"] == ("external" if external else "internal"),
                "Incorrect edge role")
        if not external:
            adj[a].add(b)
            adj[b].add(a)
    require(all(counts[x] == 1 for x in externals), "Missing/duplicate external leg")
    require(all(counts[x] >= 3 for x in ids), "Vertex with fewer than three legs")
    require(len(edges) == len(ids) + n, "Wrong tree edge count")
    reached = {ids[0]}
    while True:
        new = reached | set().union(*(adj[x] for x in reached))
        if new == reached:
            break
        reached = new
    require(len(reached) == len(ids), "Disconnected graph")
    final = graph["final_pdgs"]
    permutations = [
        order for order in itertools.permutations(range(n))
        if all(final[i] == final[order[i]] for i in range(n))
    ]
    keys = []
    for ordering in itertools.permutations(ids):
        mapping = {v: "v:" + str(i) for i, v in enumerate(ordering)}
        vs = sorted(
            [{"id": mapping[v["id"]], "ufo_vertex": v["ufo_vertex"],
              "terms": sorted(v["terms"], key=canonical)} for v in vertices],
            key=canonical,
        )
        for order in permutations:
            m = mapping | {"in:0": "in:0"} | {"out:" + str(i): "out:" + str(order[i]) for i in range(n)}
            es = []
            for e in edges:
                a, b, pdg = m[e["from"]], m[e["to"]], e["pdg"]
                if e["role"] == "internal" and a > b:
                    a, b, pdg = b, a, anti(pdg)
                es.append({"from": a, "to": b, "pdg": pdg, "role": e["role"]})
            es.sort(key=canonical)
            keys.append(canonical({"vertices": vs, "edges": es}))
    return min(keys)
