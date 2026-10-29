"""
# kinematics.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Propagator kinematics for NDA decay estimates, from optional diagram
connectivity.

A diagram may say which lines meet at each vertex (``vertices[i].legs``:
"in:0", "out:<i>", "prop:<j>"). From that, each internal line of a decay
carries the summed momentum of the final-state particles on its far side
(``final_legs``), so its virtuality is confined to

    (sum_{i in S} m_i)^2  <=  q^2  <=  (M - sum_{i not in S} m_i)^2.

Two naive prescriptions follow:

- a line with a width whose pole lies above the interval: |P(q^2)|^2 averaged
  over the interval (a Breit-Wigner average), which keeps the enhancement near
  a threshold where the interval ends just below the pole;
- a line without a width, or whose pole lies at or below the interval:
  |P(q^2)|^2 at the typical virtuality (lo + hi) / 2;
- a line that can go on shell (and has a width): the narrow-width
  factorization Gamma = Gamma(parent -> rest + R) Gamma(R -> S) / Gamma_R.
"""
import math
from typing import Any, Dict, List, Optional, Tuple


def _ref(leg: str) -> Tuple[str, int]:
    kind, index = leg.split(":")
    return kind, int(index)


def has_connectivity(diagram: Dict[str, Any]) -> bool:
    return bool(diagram.get("vertices")) and all(v.get("legs") for v in diagram["vertices"])


def far_side(diagram: Dict[str, Any]) -> List[List[int]]:
    """For each propagator, the final-state indices on its side away from the
    incoming particle. Requires ``legs`` on every vertex of a tree."""
    vertices = diagram["vertices"]
    props = diagram.get("propagators", [])
    ends: Dict[int, List[int]] = {j: [] for j in range(len(props))}
    start = None
    for k, v in enumerate(vertices):
        for leg in v["legs"]:
            kind, i = _ref(leg)
            if kind == "prop":
                ends[i].append(k)
            elif kind == "in":
                start = k
    if start is None or any(len(e) != 2 for e in ends.values()):
        raise ValueError("Connectivity must place in:0 once and each propagator at two vertices")
    sides = []
    for j in range(len(props)):
        # Walk from the incoming vertex without crossing line j.
        seen, todo = {start}, [start]
        while todo:
            k = todo.pop()
            for leg in vertices[k]["legs"]:
                kind, i = _ref(leg)
                if kind == "prop" and i != j:
                    for other in ends[i]:
                        if other not in seen:
                            seen.add(other)
                            todo.append(other)
        far = [k for k in ends[j] if k not in seen]
        if len(far) != 1:
            raise ValueError("Connectivity is not a tree")
        # Everything reachable from the far vertex without crossing j.
        region, todo = set(far), list(far)
        while todo:
            k = todo.pop()
            for leg in vertices[k]["legs"]:
                kind, i = _ref(leg)
                if kind == "prop" and i != j:
                    for other in ends[i]:
                        if other not in region:
                            region.add(other)
                            todo.append(other)
        sides.append(sorted(
            i for k in region for kind, i in map(_ref, vertices[k]["legs"]) if kind == "out"
        ))
    return sides


def virtuality_range(M: float, final_masses: List[float], legs: List[int]) -> Tuple[float, float]:
    inside = sum(final_masses[i] for i in legs)
    outside = sum(m for i, m in enumerate(final_masses) if i not in legs)
    return inside**2, max(M - outside, 0.0) ** 2


class OnShellStableLine(ValueError):
    """A zero-width line can go on shell inside the decay: the rate is the
    lower-multiplicity decay times the decay of a particle the card treats as
    stable, not a finite number for this final state."""


def propagator_factor(mass: float, width: float, lo: float, hi: float) -> float:
    """|P(q^2)|^2 for a line confined to lo <= q^2 <= hi: averaged over the
    interval when the line has a width (Breit-Wigner), otherwise taken at the
    typical virtuality (lo + hi) / 2.

    The average matters below a threshold: when hi sits just under the pole,
    the midpoint misses the enhancement at the upper end (h -> W l nu at
    m_h = 160 GeV is about 10x low at the midpoint). Lines without a width keep
    the midpoint, so a massless line reaching q^2 = 0 stays finite."""
    m2 = mass**2
    mg = mass * width
    if lo < m2 < hi and mg <= 0:
        raise OnShellStableLine(f"line of mass {mass:g} GeV with zero width can go on shell")
    if averages(mass, width, lo, hi):
        return (math.atan((hi - m2) / mg) - math.atan((lo - m2) / mg)) / (mg * (hi - lo))
    q2 = (lo + hi) / 2
    return 1.0 / max((q2 - m2) ** 2 + mg**2, 1e-300)


def averages(mass: float, width: float, lo: float, hi: float) -> bool:
    """Whether |P(q^2)|^2 is averaged over the interval: the line has a width
    and its pole lies above the interval's lower end. A pole at or below the
    lower end (tau* -> tau nu nubar starts at q^2 = m_tau^2) is no resonance:
    the line's own phase space closes there."""
    return mass * width > 0 and hi > lo and mass**2 > lo * (1 + 1e-9)


def resonant_lines(diagram: Dict[str, Any], M: float, final_masses: List[float]) -> List[int]:
    """Propagators that can go on shell inside the decay and have a width."""
    out = []
    for j, (prop, legs) in enumerate(zip(diagram.get("propagators", []), far_side(diagram))):
        lo, hi = virtuality_range(M, final_masses, legs)
        mass = prop.get("mass") or 0.0
        if mass > 0 and (prop.get("width") or 0) > 0 and lo < mass**2 < hi:
            out.append(j)
    return out


def split_at(diagram: Dict[str, Any], j: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Production (parent -> rest + R) and decay (R -> far side) diagrams
    for propagator j, with connectivity renumbered."""
    vertices, props, final = diagram["vertices"], diagram.get("propagators", []), diagram["final"]
    far = far_side(diagram)[j]
    ends = {}
    for k, v in enumerate(vertices):
        for leg in v["legs"]:
            kind, i = _ref(leg)
            if kind == "prop":
                ends.setdefault(i, []).append(k)

    def region(seed: int) -> set:
        seen, todo = {seed}, [seed]
        while todo:
            k = todo.pop()
            for leg in vertices[k]["legs"]:
                kind, i = _ref(leg)
                if kind == "prop" and i != j:
                    for other in ends[i]:
                        if other not in seen:
                            seen.add(other)
                            todo.append(other)
        return seen

    a, b = ends[j]
    side_b = region(b)
    decay_vertices = side_b if any(_ref(l) == ("out", far[0]) for k in side_b for l in vertices[k]["legs"]) else region(a)
    prod_vertices = set(range(len(vertices))) - decay_vertices
    R = props[j]
    resonance = {"label": R.get("label"), "mass": R["mass"]}
    if R.get("spin") is not None:
        resonance["spin"] = R["spin"]

    def build(keep, outs, line_ref, initial):
        pmap = {i: n for n, i in enumerate(i for i in range(len(props)) if i != j and set(ends[i]) <= keep)}
        omap = {i: n for n, i in enumerate(outs)}
        new_vertices = []
        for k in sorted(keep):
            legs = []
            for leg in vertices[k]["legs"]:
                kind, i = _ref(leg)
                if kind == "prop":
                    legs.append(line_ref if i == j else "prop:" + str(pmap[i]))
                elif kind == "out":
                    legs.append("out:" + str(omap[i]))
                else:
                    legs.append(leg)
            new_vertices.append({**vertices[k], "legs": legs})
        sub = {key: value for key, value in diagram.items()
               if key not in {"topology", "vertices", "propagators", "initial", "final",
                              "symmetry_factor", "color_factor"}}
        sub.update(initial=initial, vertices=new_vertices,
                   propagators=[{k: v for k, v in props[i].items() if k != "final_legs"} for i in pmap])
        return sub

    rest = [i for i in range(len(final)) if i not in far]
    # Identical-particle factors belong to the full final state, applied once
    # by the caller, not to each factor.
    production = build(prod_vertices, rest, "out:" + str(len(rest)), diagram["initial"])
    production["symmetry_factor"] = 1
    production["final"] = [final[i] for i in rest] + [resonance]
    decay = build(decay_vertices, far, "in:0", [resonance])
    decay["symmetry_factor"] = 1
    decay["final"] = [final[i] for i in far]
    return production, decay
