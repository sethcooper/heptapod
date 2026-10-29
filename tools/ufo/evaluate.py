"""
# evaluate.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Numerical values out of a UFO: parameters resolved in dependency order, and
coupling constants evaluated against them.

WHY THIS EXISTS. A compiled UFO answers "which fields interact" by inspection
-- parse_vertices does that -- but "with what strength" only by evaluating
parameters.py and couplings.py. Three independent agents building the same
model each wrote their own evaluator for exactly this, and each one hit a
failure that looked like physics:

  * a regex assuming ``Parameter(`` blocks close with ``)`` on its own line
    returned None for every parameter, silently, because the generated file
    closes on its last argument;
  * a walker that guessed the coupling dict was keyed ``(lorentz, colour)``
    made every right-handed coupling evaluate to exactly zero, which reads as
    a broken chirality structure rather than a transposed index;
  * a third rebuilt the whole namespace to re-evaluate a model at parameter
    points other than the card defaults.

The UFO is Python 2 and executes arbitrary code on import, so "just import it"
is not available and everyone reimplements the same ast walk. That is what
this module is for.

Nothing here imports the UFO. Expressions are whitelist-verified and evaluated
in a locked namespace containing the model's own parameters, ``cmath``, and
nothing else.
"""

from __future__ import annotations

import ast
import cmath
import os
from typing import Any, Dict, List, Optional, Tuple

# Expression nodes a UFO parameter or coupling may legitimately contain.
# Deliberately narrow: anything outside this raises rather than evaluating.
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
    ast.Call, ast.Attribute, ast.Tuple, ast.Mod, ast.FloorDiv,
)

# Callables a UFO expression may reference. ``complexconjugate`` is the UFO's
# own helper; the rest are cmath's.
_ALLOWED_CALLS = {"complexconjugate", "abs", "complex", "float", "int"}
_ALLOWED_CMATH = {
    "sqrt", "exp", "log", "sin", "cos", "tan", "asin", "acos", "atan", "pi",
}


def _verify(node: ast.AST) -> None:
    """Reject anything that is not arithmetic over parameters and cmath."""
    for n in ast.walk(node):
        if not isinstance(n, _ALLOWED_NODES):
            raise ValueError(f"disallowed syntax in UFO expression: {type(n).__name__}")
        if isinstance(n, ast.Attribute):
            if not (isinstance(n.value, ast.Name) and n.value.id == "cmath"):
                raise ValueError("only cmath.* attributes are allowed")
            if n.attr not in _ALLOWED_CMATH:
                raise ValueError(f"cmath.{n.attr} is not allowed")
        if isinstance(n, ast.Call):
            f = n.func
            ok = (isinstance(f, ast.Name) and f.id in _ALLOWED_CALLS) or (
                isinstance(f, ast.Attribute)
                and isinstance(f.value, ast.Name)
                and f.value.id == "cmath"
                and f.attr in _ALLOWED_CMATH
            )
            if not ok:
                raise ValueError("disallowed function call in UFO expression")


def _namespace(params: Dict[str, Any]) -> Dict[str, Any]:
    """The locked evaluation namespace: the model's parameters plus cmath."""
    ns: Dict[str, Any] = dict(params)
    ns["cmath"] = cmath
    ns["abs"] = abs
    ns["complex"] = complex
    ns["float"] = float
    ns["int"] = int
    ns["complexconjugate"] = lambda z: z.conjugate() if isinstance(z, complex) else z
    return ns


def safe_eval(expr: str, params: Dict[str, Any]) -> Any:
    """Evaluate one UFO expression against a parameter namespace.

    Args:
        expr: The expression source, as it appears in the UFO.
        params: Resolved parameter values.

    Returns:
        A float or complex.

    Raises:
        ValueError: The expression uses syntax outside the whitelist.
        NameError: It references a parameter that is not resolved.
    """
    tree = ast.parse(expr, mode="eval")
    _verify(tree)
    return eval(  # noqa: S307 - whitelist-verified AST, locked namespace
        compile(tree, "<ufo>", "eval"), {"__builtins__": {}}, _namespace(params))


def _free_names(expr: str) -> set:
    """Parameter names an expression depends on (cmath and helpers excluded)."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return set()
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and n.id not in _ALLOWED_CALLS and n.id != "cmath":
            out.add(n.id)
    return out


def parse_parameters(ufo_dir: str) -> List[Dict[str, Any]]:
    """Every parameter declared in parameters.py, external and internal.

    Parsed with ``ast``, so multi-line ``Parameter(...)`` blocks are read
    correctly however they are wrapped -- the generated file closes the call on
    its last argument line rather than on a line of its own, which is what
    defeats the obvious regex.

    Args:
        ufo_dir: The UFO directory.

    Returns:
        One dict per parameter with ``var`` (its Python name), ``name``,
        ``nature`` ("external" or "internal"), ``type`` ("real"/"complex"),
        ``value`` (a number for external parameters, the expression source for
        internal ones), ``lhablock`` and ``lhacode`` where declared.
    """
    path = os.path.join(ufo_dir, "parameters.py")
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            tree = ast.parse(fh.read())
    except (OSError, SyntaxError):
        return []

    out: List[Dict[str, Any]] = []
    for stmt in ast.walk(tree):
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        target, call = stmt.targets[0], stmt.value
        if not isinstance(target, ast.Name) or not isinstance(call, ast.Call):
            continue
        fname = getattr(call.func, "id", None) or getattr(call.func, "attr", None)
        if fname != "Parameter":
            continue
        kw = {k.arg: k.value for k in call.keywords if k.arg}
        rec: Dict[str, Any] = {"var": target.id}
        for key in ("name", "nature", "type", "lhablock", "texname"):
            node = kw.get(key)
            if isinstance(node, ast.Constant):
                rec[key] = node.value
        code = kw.get("lhacode")
        if isinstance(code, ast.List):
            rec["lhacode"] = [e.value for e in code.elts if isinstance(e, ast.Constant)]
        val = kw.get("value")
        if val is not None:
            if isinstance(val, ast.Constant) and isinstance(val.value, (int, float)):
                rec["value"] = val.value
            else:
                # Internal parameters carry an expression, either as a string
                # literal or as live Python.
                rec["value"] = (val.value if isinstance(val, ast.Constant)
                                else ast.unparse(val))
        rec.setdefault("name", target.id)
        rec.setdefault("nature", "external" if isinstance(rec.get("value"), (int, float))
                       else "internal")
        out.append(rec)
    return out


def evaluate_parameters(
    ufo_dir: str,
    overrides: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Resolve every parameter to a number, internals included.

    Internal parameters are expressions over other parameters. FeynRules
    topologically sorts them before writing parameters.py, but this does not
    rely on that: it sweeps repeatedly, resolving whatever has all its
    dependencies available, until a sweep makes no progress. A UFO from another
    producer, or one hand-edited, therefore still resolves.

    Args:
        ufo_dir: The UFO directory.
        overrides: Values to substitute for external parameters, so a model can
            be evaluated away from its card defaults. A parametrisation can be
            exact at one point and degenerate elsewhere, and this is how you
            find out.

    Returns:
        ``(values, unresolved)`` -- the resolved namespace, and the names that
        could not be resolved (a cycle, or a dependency the file never
        declares).
    """
    params = parse_parameters(ufo_dir)
    values: Dict[str, Any] = {}
    pending: List[Dict[str, Any]] = []

    for p in params:
        v = p.get("value")
        if isinstance(v, (int, float)) and p.get("nature") != "internal":
            values[p["var"]] = float(v)
        else:
            pending.append(p)

    for name, v in (overrides or {}).items():
        values[name] = v

    # Sweep until fixed point. Bounded by len(pending) sweeps, since each
    # productive sweep resolves at least one parameter.
    for _ in range(len(pending) + 1):
        progress = False
        still: List[Dict[str, Any]] = []
        for p in pending:
            if p["var"] in (overrides or {}):
                progress = True
                continue
            expr = p.get("value")
            if not isinstance(expr, str):
                still.append(p)
                continue
            if not _free_names(expr) <= set(values):
                still.append(p)
                continue
            try:
                values[p["var"]] = safe_eval(expr, values)
                progress = True
            except (ValueError, NameError, ZeroDivisionError, TypeError):
                still.append(p)
        pending = still
        if not progress:
            break

    return values, [p["var"] for p in pending]


def parse_couplings(ufo_dir: str) -> Dict[str, str]:
    """``{GC_name: expression}`` for every coupling in couplings.py."""
    path = os.path.join(ufo_dir, "couplings.py")
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            tree = ast.parse(fh.read())
    except (OSError, SyntaxError):
        return {}

    out: Dict[str, str] = {}
    for stmt in ast.walk(tree):
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        target, call = stmt.targets[0], stmt.value
        if not isinstance(target, ast.Name) or not isinstance(call, ast.Call):
            continue
        fname = getattr(call.func, "id", None) or getattr(call.func, "attr", None)
        if fname != "Coupling":
            continue
        kw = {k.arg: k.value for k in call.keywords if k.arg}
        val = kw.get("value")
        if isinstance(val, ast.Constant) and isinstance(val.value, str):
            out[target.id] = val.value
        elif val is not None:
            out[target.id] = ast.unparse(val)
    return out


def parse_coupling_orders(ufo_dir: str) -> Dict[str, Dict[str, int]]:
    """``{GC_name: {order: power}}``, e.g. ``{"GC_11": {"QCD": 1}}``.

    A coupling whose ``order`` is missing or not a literal dict maps to an
    empty dict, which counts as order zero in every coupling.
    """
    path = os.path.join(ufo_dir, "couplings.py")
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            tree = ast.parse(fh.read())
    except (OSError, SyntaxError):
        return {}

    out: Dict[str, Dict[str, int]] = {}
    for stmt in ast.walk(tree):
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        target, call = stmt.targets[0], stmt.value
        if not isinstance(target, ast.Name) or not isinstance(call, ast.Call):
            continue
        fname = getattr(call.func, "id", None) or getattr(call.func, "attr", None)
        if fname != "Coupling":
            continue
        kw = {k.arg: k.value for k in call.keywords if k.arg}
        try:
            orders = ast.literal_eval(kw["order"]) if "order" in kw else {}
        except ValueError:
            orders = {}
        out[target.id] = {str(k): int(v) for k, v in orders.items()} \
            if isinstance(orders, dict) else {}
    return out


def evaluate_couplings(
    ufo_dir: str,
    overrides: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Every coupling constant as a complex number.

    Args:
        ufo_dir: The UFO directory.
        overrides: External-parameter overrides, as for evaluate_parameters.

    Returns:
        ``(values, errors)`` -- ``{GC_name: complex}`` for those that evaluated,
        and ``{GC_name: reason}`` for those that did not.
    """
    values, _ = evaluate_parameters(ufo_dir, overrides)
    out: Dict[str, Any] = {}
    errs: Dict[str, str] = {}
    for name, expr in parse_couplings(ufo_dir).items():
        try:
            out[name] = safe_eval(expr, values)
        except (ValueError, NameError, ZeroDivisionError, TypeError) as exc:
            errs[name] = str(exc)
    return out, errs


def vertex_couplings(ufo_dir: str) -> List[Dict[str, Any]]:
    """Vertices with their coupling dict keys kept intact.

    THE KEY ORDER IS ``(colour, lorentz)``, not ``(lorentz, colour)``. Nothing
    in the UFO format states this and guessing wrong is not loud: for a vertex
    with one colour structure and two Lorentz structures -- a fermion current
    with left- and right-handed pieces -- a transposed lookup silently returns
    the wrong chiral coupling, or nothing at all, and the model reads as though
    its chirality structure were broken. Reported here so nobody has to guess.

    Args:
        ufo_dir: The UFO directory.

    Returns:
        One dict per vertex: ``particles``, ``lorentz``, ``color`` and
        ``couplings`` as ``[{"color_index", "lorentz_index", "coupling"}]``.
    """
    vpath = os.path.join(ufo_dir, "vertices.py")
    try:
        with open(vpath, "r", encoding="utf-8", errors="replace") as fh:
            tree = ast.parse(fh.read())
    except (OSError, SyntaxError):
        return []

    out: List[Dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fname = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if fname != "Vertex":
            continue
        kws = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        plist = kws.get("particles")
        if not isinstance(plist, ast.List):
            continue
        names = [e.attr for e in plist.elts if isinstance(e, ast.Attribute)]
        if not names:
            continue

        pairs: List[Dict[str, Any]] = []
        cmap = kws.get("couplings")
        if isinstance(cmap, ast.Dict):
            for k, v in zip(cmap.keys, cmap.values):
                if not (isinstance(k, ast.Tuple) and len(k.elts) == 2):
                    continue
                idx = [e.value if isinstance(e, ast.Constant) else None for e in k.elts]
                pairs.append({
                    "color_index": idx[0],
                    "lorentz_index": idx[1],
                    "coupling": v.attr if isinstance(v, ast.Attribute) else None,
                })

        def _names(key: str) -> List[str]:
            v = kws.get(key)
            if not isinstance(v, ast.List):
                return []
            return [e.attr if isinstance(e, ast.Attribute) else
                    (e.value if isinstance(e, ast.Constant) else None)
                    for e in v.elts]

        out.append({
            "name": ast.literal_eval(kws["name"]) if "name" in kws else None,
            "particles": names,
            "lorentz": _names("lorentz"),
            "color": _names("color"),
            "couplings": pairs,
        })
    return out


def ufo_gauge(ufo_dir: str) -> Dict[str, Any]:
    """Which gauge the UFO was written in, from its VERTICES.

    FeynmanGauge is a global set inside SM.fr. An add-on inherits it silently,
    it changes the vertex count by tens of entries, and no tool input or output
    used to mention it -- so a model can acquire a Goldstone and ghost sector
    its author never decided on.

    READ FROM VERTICES, NOT DECLARATIONS. An earlier version of this decided
    from particles.py, which is wrong and said so confidently: SM.fr declares
    G0, G+, G- and the ghosts unconditionally, in every gauge. Unitary gauge
    removes them from the INTERACTIONS, not from the particle list. That
    version reported "feynman" for a model built with feynman_gauge=false while
    FeynRulesToUFO reported the caller's `false` for the same directory -- two
    tools contradicting each other about one UFO, with the inferring one wrong.

    Goldstone participation is the discriminator, and ghosts are not. QCD
    ghosts survive unitary gauge (SM.fr's LGhost keeps its LGhs term either
    way), so `ghG ghG~ g` appears in both and proves nothing; electroweak
    Goldstones appear in vertices only under Feynman gauge.

    Args:
        ufo_dir: The UFO directory.

    Returns:
        ``{gauge, goldstones, ghosts, goldstone_vertices, ghost_vertices}`` --
        "feynman" when Goldstones take part in interactions, "unitary" when
        they do not, "unknown" when the UFO cannot be read. The declared lists
        are reported too, but they do not decide the verdict.
    """
    from .parser import parse_particles

    try:
        parts = parse_particles(os.path.join(ufo_dir, "particles.py"))
    except (OSError, SyntaxError):
        parts = []
    if not parts:
        return {"gauge": "unknown", "goldstones": [], "ghosts": [],
                "goldstone_vertices": 0, "ghost_vertices": 0}

    golds, ghosts = set(), set()
    for p in parts:
        var = str(p.get("var") or p.get("name") or "")
        if is_goldstone(p):
            golds.add(var)
        elif is_ghost(p):
            ghosts.add(var)

    g_verts = gh_verts = 0
    for v in vertex_couplings(ufo_dir):
        names = set(v["particles"])
        if names & golds:
            g_verts += 1
        elif names & ghosts:
            gh_verts += 1

    return {
        "gauge": "feynman" if g_verts else "unitary",
        "goldstones": sorted(golds),
        "ghosts": sorted(ghosts),
        "goldstone_vertices": g_verts,
        "ghost_vertices": gh_verts,
    }


def is_goldstone(particle: Dict[str, Any]) -> bool:
    """Whether a parsed particle record is a Goldstone boson.

    Reads the UFO's own flag first. The name test is only a fallback, and on
    its own it is wrong: a hidden-sector Goldstone is not called G0 or G+. One
    model declared ``Gd = Particle(..., mass = Param.MZp, goldstone = True)``
    for the state eaten by its Z', which a name test classifies as a physical
    new scalar -- and every V-Hd-Gd vertex then reads as new physics.

    Args:
        particle: A record from parse_particles.

    Returns:
        True for a Goldstone.
    """
    if particle.get("goldstone") is True or particle.get("GoldstoneBoson") is True:
        return True
    name = str(particle.get("var") or particle.get("name") or "")
    return name.startswith(("G0", "G__plus__", "G__minus__", "G+", "G-"))


def is_ghost(particle: Dict[str, Any]) -> bool:
    """Whether a parsed particle record is a Faddeev-Popov ghost.

    Ghosts carry a negative spin in the UFO. Note the type: parse_particles
    folds numeric fields to ``Fraction`` so that ``charge = -1/3`` stays exact,
    and ``isinstance(Fraction(-1, 1), (int, float))`` is False -- a spin test
    written that way silently classifies every ghost as a physical field, and
    is then rescued only by the ``gh`` name prefix that a differently-named
    ghost would not have.

    Args:
        particle: A record from parse_particles.

    Returns:
        True for a ghost.
    """
    spin = particle.get("spin")
    try:
        if spin is not None and float(spin) < 0:
            return True
    except (TypeError, ValueError):
        pass
    name = str(particle.get("var") or particle.get("name") or "")
    return name.startswith("gh")
