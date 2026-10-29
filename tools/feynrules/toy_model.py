"""
# toy_model.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
"""
WriteToyModel: a FeynRules .fr for a toy theory from a compact spec -- the
fields (spin, charge, colour, mass) and the interaction terms -- with
everything FeynRules is particular about generated: the gauge group and
photon, class keys (S[n], F[n]), kinetic and mass terms with covariant
derivatives, colour indices of scalars, parameters, hermitian conjugates.

Written by hand from a stated Lagrangian, a model most often goes wrong
here: class keys `S == {...}`, classes named S/F/E, a missing
gauge group, Sbar S for a coloured scalar (untied colour), Get[SM.fr].

Gauge modes:
  qed   a U(1) with its own photon A (charge e, ee = 0.30282212), standalone
        (WriteFeynArtsModel / FeynRulesToUFO with load_sm=false).
  none  no gauge group (scalars and fermions only), standalone.
  sm    an add-on to FeynRules' SM.fr (load_sm=true, lagrangian "LSM + LToy"):
        new fields get SM.fr's covariant derivative from their charge and
        hypercharge; an SU(2) doublet is given as {"doublet": [upper, lower]}
        in terms of declared physical fields, e.g. ["Hc", "(H0 + I A0)/Sqrt[2]"].
"""

import json
import os
import re
from fractions import Fraction
from typing import Any, Dict, Optional, Tuple, Union

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9]*$")
_RESERVED = {"S", "F", "V", "U", "T", "E", "I", "N", "C", "D", "O", "K", "Pi", "Gamma", "A", "G",
             "Z", "W", "H", "ee", "mu", "nu", "cc", "ii", "jj", "x", "del", "DC", "FS", "Ga", "Re", "Im"}
_SM_CLASSES = {"Z", "W", "G", "ghA", "ghZ", "ghWp", "ghWm", "ghG", "B", "Wi", "ghB", "ghWi", "vl", "l",
               "uq", "dq", "LL", "lR", "QL", "uR", "dR", "H", "G0", "GP", "Phi"}
#: SM.fr's own class numbers, which an add-on must not reuse.
_SM_USED = {"S": {1, 2, 3, 11}, "F": {1, 2, 3, 4, 11, 12, 13, 14, 15}, "V": {1, 2, 3, 4, 11, 12}}


class ToyModelError(ValueError):
    pass


def _num(x) -> str:
    """A charge or hypercharge as Mathematica: 2/3, -1, 0."""
    f = Fraction(str(x)) if not isinstance(x, Fraction) else x
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


def _real(x) -> str:
    v = float(x)
    return repr(v).replace("e", "*^")


class _Field:
    def __init__(self, spec: dict, gauge: str):
        self.name = str(spec.get("name", ""))
        if not _NAME.match(self.name):
            raise ToyModelError(f"field name {self.name!r}: letters and digits, starting with a letter")
        if self.name in _RESERVED or (gauge == "sm" and self.name in _SM_CLASSES):
            raise ToyModelError(f"field name {self.name!r} is taken (FeynArts heads, Mathematica built-ins, "
                                f"{'SM.fr classes, ' if gauge == 'sm' else ''}the photon A); choose another")
        self.doublet = spec.get("doublet")
        spin = str(spec.get("spin", "0" if self.doublet else "")).strip()
        if spin not in ("0", "1/2", "0.5"):
            raise ToyModelError(f"{self.name}: spin is 0 or 1/2")
        self.fermion = spin != "0"
        self.real = bool(spec.get("self_conjugate", False))
        if self.fermion and self.real:
            raise ToyModelError(f"{self.name}: Majorana fermions are not supported here")
        self.charge = Fraction(str(spec.get("charge", 0)))
        if self.real and self.charge:
            raise ToyModelError(f"{self.name}: a self-conjugate field has no charge")
        self.colour = int(spec.get("colour", 1))
        if self.colour not in (1, 3):
            raise ToyModelError(f"{self.name}: colour is 1 or 3")
        if self.real and self.colour != 1:
            raise ToyModelError(f"{self.name}: a coloured field is complex")
        self.hypercharge = spec.get("hypercharge")
        if self.doublet is not None:
            if gauge != "sm":
                raise ToyModelError(f"{self.name}: a doublet needs gauge 'sm'")
            if not (isinstance(self.doublet, list) and len(self.doublet) == 2) or self.hypercharge is None:
                raise ToyModelError(f"{self.name}: doublet is [upper, lower] in terms of physical fields, "
                                    "with a hypercharge")
            self.mass_sym, self.mass_val = None, None
            return
        self.mass_sym = str(spec.get("mass", f"M{self.name}"))
        if not _NAME.match(self.mass_sym):
            raise ToyModelError(f"{self.name}: mass symbol {self.mass_sym!r}")
        self.mass_val = spec.get("mass_value", 100.0)


def _expand_conjugates(term: str, fields: Dict[str, _Field]) -> str:
    """X^dag, X^\\dagger, X^* -> Xbar."""
    return re.sub(r"\b([A-Za-z][A-Za-z0-9]*)\s*\^\s*(?:dag|\\dagger|\*|†)", r"\1bar", term)


def _colour_scalars(term: str, fields: Dict[str, _Field]) -> str:
    """A coloured scalar's index written out and summed: Sqbar Sq -> Sqbar[cc] Sq[cc].
    Done when the term has one coloured scalar and its conjugate; otherwise
    the term must carry the indices itself."""
    for f in fields.values():
        if f.fermion or f.colour == 1 or f.doublet is not None:
            continue
        plain = len(re.findall(rf"(?<![\w$]){f.name}(?![\w$\[])", term))
        bar = len(re.findall(rf"(?<![\w$]){f.name}bar(?![\w$\[])", term))
        if plain == 0 and bar == 0:
            continue
        if plain == 1 and bar == 1:
            term = re.sub(rf"(?<![\w$]){f.name}bar(?![\w$\[])", f"{f.name}bar[cc]", term)
            term = re.sub(rf"(?<![\w$]){f.name}(?![\w$\[])", f"{f.name}[cc]", term)
        else:
            raise ToyModelError(f"term {term!r}: the coloured scalar {f.name} appears {plain} time(s) and "
                                f"{f.name}bar {bar} time(s); write its colour indices out, e.g. "
                                f"{f.name}bar[c1] {f.name}[c1]")
    return term


def _normalise(term: str, fields: Dict[str, _Field]) -> Tuple[str, bool]:
    t = term.strip()
    hc = False
    m = re.search(r"\+\s*(?:h\.?\s*c\.?|H\.?\s*C\.?|hermitian conjugate)\s*$", t)
    if m:
        t, hc = t[:m.start()].strip(), True
    t = re.sub(r"\bPL\b", "ProjM", t)
    t = re.sub(r"\bPR\b", "ProjP", t)
    t = re.sub(r"\bg5\b|\bgamma5\b", "Ga[5]", t)
    t = t.replace("**", "^")
    t = _expand_conjugates(t, fields)
    t = _colour_scalars(t, fields)
    return t, hc


def toy_model(spec: dict) -> Dict[str, object]:
    """The .fr text and what to pass to WriteFeynArtsModel / FeynRulesToUFO."""
    gauge = str(spec.get("gauge", "qed")).lower()
    if gauge not in ("qed", "none", "sm"):
        raise ToyModelError("gauge is 'qed' (standalone, own photon), 'none', or 'sm' (add-on to SM.fr)")
    name = str(spec.get("name", "ToyModel"))
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
        raise ToyModelError("name: letters, digits and _")
    fields = {}
    for fs in spec.get("fields") or []:
        f = _Field(fs, gauge)
        if f.name in fields:
            raise ToyModelError(f"field {f.name} declared twice")
        fields[f.name] = f
    if not fields:
        raise ToyModelError("no fields")
    params = dict(spec.get("parameters") or {})
    for p in params:
        if not _NAME.match(p) or p in _RESERVED or p in fields:
            raise ToyModelError(f"parameter name {p!r} is not usable")
    if gauge == "none" and any(f.charge for f in fields.values()):
        raise ToyModelError("charged fields need gauge 'qed' or 'sm'")
    # physical fields that are components of a doublet: their kinetic terms
    # come with the doublet's DC[Phibar[ii], mu] DC[Phi[ii], mu]
    in_doublet = {f.name for f in fields.values()
                  if any(re.search(rf"(?<![\w$]){f.name}(?![\w$])", c)
                         for g in fields.values() for c in (g.doublet or []))}
    notes = _sm_hypercharges(fields, in_doublet) if gauge == "sm" else []

    lines = [f'(* {name}.fr -- written by heptapod WriteToyModel. *)', "",
             f'M$ModelName = "{name}";', "",
             'M$Information = { Authors -> {"heptapod WriteToyModel"}, Version -> "1.0" };', ""]
    if gauge == "qed":
        lines += ["M$GaugeGroups = {",
                  "  U1QED == { Abelian -> True, CouplingConstant -> ee, GaugeBoson -> A, Charge -> Q }",
                  "};", ""]
    if any(f.colour == 3 for f in fields.values()) and gauge != "sm":
        lines += ["IndexRange[Index[Colour]] = NoUnfold[Range[3]];", "IndexStyle[Colour, c];", ""]
    plines = []
    if gauge == "qed":
        plines.append("  ee == { ParameterType -> External, Value -> 0.30282212, "
                      "InteractionOrder -> {QED, 1}, TeX -> e }")
    for p, v in params.items():
        plines.append(f"  {p} == {{ ParameterType -> External, Value -> {_real(v)}, "
                      f"InteractionOrder -> {{NP, 1}} }}")
    if plines:
        lines += ["M$Parameters = {", ",\n".join(plines), "};", ""]

    # classes
    used = {k: set(v) for k, v in _SM_USED.items()} if gauge == "sm" else {"S": set(), "F": set(), "V": set()}
    def next_index(kind: str) -> int:
        n = 1
        while n in used[kind]:
            n += 1
        used[kind].add(n)
        return n

    classes = []
    if gauge == "qed":
        used["V"].add(1)
        classes.append('  V[1] == { ClassName -> A, SelfConjugate -> True, Mass -> 0, Width -> 0, '
                       'ParticleName -> "a", PDG -> 22, FullName -> "Photon" }')
    particle_names = {}
    pdg = 9000100
    for f in fields.values():
        pdg += 1
        if f.doublet is not None:
            n = next_index("S")
            classes.append(f"  S[{n}] == {{ ClassName -> {f.name}, Unphysical -> True, Indices -> {{Index[SU2D]}}, "
                           f"FlavorIndex -> SU2D, SelfConjugate -> False, "
                           f"QuantumNumbers -> {{Y -> {_num(f.hypercharge)}}}, "
                           f"Definitions -> {{ {f.name}[1] -> {f.doublet[0]}, {f.name}[2] -> {f.doublet[1]} }} }}")
            continue
        kind = "F" if f.fermion else "S"
        n = next_index(kind)
        qn = [f"Q -> {_num(f.charge)}"] if f.charge else []
        if gauge == "sm" and f.hypercharge is not None and not f.real:
            qn.append(f"Y -> {_num(f.hypercharge)}")
        items = [f"ClassName -> {f.name}", f"SelfConjugate -> {'True' if f.real else 'False'}"]
        if f.colour == 3:
            items.append("Indices -> {Index[Colour]}")
        items.append(f"Mass -> {{{f.mass_sym}, {_real(f.mass_val)}}}")
        items.append("Width -> 0")
        if qn:
            items.append(f"QuantumNumbers -> {{{', '.join(qn)}}}")
        if f.real:
            items.append(f'ParticleName -> "{f.name}"')
            particle_names[f.name] = (f.name,)
        else:
            plus = "+" if f.charge > 0 else "-" if f.charge < 0 else ""
            minus = "-" if f.charge > 0 else "+" if f.charge < 0 else "~"
            pn, an = (f.name + plus, f.name + minus) if plus else (f.name, f.name + "~")
            items.append(f'ParticleName -> "{pn}", AntiParticleName -> "{an}"')
            particle_names[f.name] = (pn, an)
        items.append(f"PDG -> {pdg}")
        classes.append(f"  {kind}[{n}] == {{ " + ", ".join(items) + " }")
    lines += ["M$ClassesDescription = {", ",\n".join(classes), "};", ""]

    # kinetic and mass terms
    d = "del" if gauge == "none" else "DC"
    kin = []
    if gauge == "qed":
        kin.append("-1/4 FS[A, mu, nu] FS[A, mu, nu]")
    for f in fields.values():
        if f.doublet is not None:
            kin.append(f"DC[{f.name}bar[ii], mu] DC[{f.name}[ii], mu]")
        elif f.name in in_doublet:
            kin.append(f"- 1/2 {f.mass_sym}^2 {f.name}^2" if f.real
                       else f"- {f.mass_sym}^2 {f.name}bar {f.name}")
        elif f.fermion:
            kin.append(f"I {f.name}bar.Ga[mu].{d}[{f.name}, mu] - {f.mass_sym} {f.name}bar.{f.name}")
        elif f.real:
            kin.append(f"1/2 del[{f.name}, mu] del[{f.name}, mu] - 1/2 {f.mass_sym}^2 {f.name}^2")
        else:
            idx = "[cc]" if f.colour == 3 else ""
            kin.append(f"{d}[{f.name}bar{idx}, mu] {d}[{f.name}{idx}, mu] "
                       f"- {f.mass_sym}^2 {f.name}bar{idx} {f.name}{idx}")
    # "+" ends a line, never starts one: Get reads a line that completes an
    # expression as the whole statement, and the next "+ ..." is lost.
    lines.append("LKin := " + " +\n        ".join(f"({k})" for k in kin) + ";")

    terms = []
    for raw in spec.get("interactions") or []:
        t, hc = _normalise(str(raw), fields)
        terms.append(f"({t}) + HC[{t}]" if hc else f"({t})")
    lines.append("LInt := " + (" +\n        ".join(terms) if terms else "0") + ";")
    lines.append("LToy := LKin + LInt;")
    text = "\n".join(lines) + "\n"
    return {"fr": text, "load_sm": gauge == "sm",
            "lagrangian": "LSM + LToy" if gauge == "sm" else "LToy",
            "particle_names": {k: list(v) for k, v in particle_names.items()},
            "photon": "a" if gauge in ("qed", "sm") else None, "notes": notes}


def _sm_hypercharges(fields: Dict[str, "_Field"], in_doublet) -> list:
    """Under gauge 'sm' a field couples to the photon and the Z only through
    SM.fr's covariant derivative, which reads the hypercharge Y (Q = T3 + Y):
    a charged singlet written without Y has no photon vertex at all, and
    FeynRules says nothing. Every field here other than a doublet and its
    components is an SU(2) singlet (T3 = 0), so Y = Q: defaulted, with a
    note; a Y that contradicts Q is refused."""
    notes = []
    for f in fields.values():
        if f.doublet is not None or f.name in in_doublet or f.real:
            continue
        if f.hypercharge is None:
            if f.charge:
                f.hypercharge = f.charge
                notes.append(f"{f.name}: no hypercharge given; as an SU(2) singlet (T3 = 0) it gets "
                             f"Y = Q = {_num(f.charge)}, which gives its photon and Z couplings. An "
                             "SU(2) doublet is declared with 'doublet' and its own hypercharge")
            continue
        try:
            y = Fraction(str(f.hypercharge))
        except (ValueError, ZeroDivisionError):
            raise ToyModelError(f"{f.name}: hypercharge {f.hypercharge!r} is not a number") from None
        if y != f.charge:
            raise ToyModelError(
                f"{f.name}: hypercharge {_num(y)} with charge {_num(f.charge)} is ambiguous for an SU(2) "
                "singlet, which has Q = T3 + Y with T3 = 0, so Y = Q. Give Y = Q (or leave it out), or "
                "declare an SU(2) doublet with 'doublet' if the field belongs to one")
        f.hypercharge = y
    return notes


def apply_toy_spec(tool, spec, base_directory: str, stem: str) -> Optional[str]:
    """For a tool given toy_spec instead of a .fr: write the .fr and set the
    tool's model_path, lagrangian and load_sm. Returns an error text or None."""
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except json.JSONDecodeError as exc:
            return f"toy_spec is not JSON: {exc}"
    try:
        out = toy_model(spec)
    except ToyModelError as exc:
        return str(exc)
    path = os.path.join(base_directory, f"{spec.get('name') or stem}.fr")
    with open(path, "w") as fh:
        fh.write(out["fr"])
    tool.model_path, tool.lagrangian, tool.load_sm = path, out["lagrangian"], out["load_sm"]
    return None


# ── the tool ─────────────────────────────────────────────────────────────


class WriteToyModel(BaseTool):
    """
    Write a FeynRules .fr for a toy theory from a compact spec, so the model
    for a stated Lagrangian need not be written in FeynRules syntax. The
    generated file goes to WriteFeynArtsModel or FeynRulesToUFO with the
    load_sm and lagrangian this tool returns.

    gauge: "qed" (standalone, with its own photon 'a' and coupling e),
    "none" (no gauge group), or "sm" (an add-on to FeynRules' SM.fr: W, Z,
    photon, SM fermions as SM.fr has them).

    fields: [{"name", "spin": 0 | "1/2", "self_conjugate" (real scalar),
      "charge" (e.g. "2/3"), "colour": 1 | 3, "mass" (symbol, default M<name>),
      "mass_value", "hypercharge" (gauge 'sm' only)}]. Under gauge 'sm' the
      photon and Z couplings come from the hypercharge Y (Q = T3 + Y): a
      charged field that is not in a doublet is an SU(2) singlet, Y = Q, and
      Y defaults to Q (said in "notes"); a Y that contradicts Q is refused.
      A gauge 'sm' SU(2) doublet:
      {"name": "Phi2", "doublet": ["Hc", "(H0 + I A0)/Sqrt[2]"], "hypercharge": "1/2"}
      in terms of declared physical fields (their masses are put in directly).
    parameters: {"y": 0.8, "mu": 300, ...} -- every coupling the terms use.
    interactions: Lagrangian terms, FeynRules notation with conveniences:
      fermion bilinears Fbar.F, Fbar.Ga[5].F (g5 accepted), Fbar.(cL PL + cR PR).G
      (PL, PR = P_L, P_R); a complex scalar's conjugate Xbar (X^dag accepted);
      "+ h.c." at the end adds the hermitian conjugate; a coloured scalar's
      colour index is written out for you (Sqbar Sq -> Sqbar[cc] Sq[cc]).
      Kinetic and mass terms, and every gauge coupling, are generated: do not
      write them.

    Example (S -> gamma gamma through a coloured fermion and a charged scalar):
      {"name": "SAA", "gauge": "qed",
       "fields": [{"name": "Sz", "spin": 0, "self_conjugate": true, "mass_value": 250},
                  {"name": "Fq", "spin": "1/2", "charge": "2/3", "colour": 3, "mass_value": 150},
                  {"name": "ph", "spin": 0, "charge": 1, "mass_value": 200}],
       "parameters": {"y": 0.8, "muS": 300},
       "interactions": ["-y Sz Fqbar.Fq", "-muS Sz phbar ph"]}

    Output: {"status", "fr_path", "load_sm", "lagrangian", "particle_names"
    (per field: particle, antiparticle -- use these in processes), "notes"
    (defaults applied, e.g. a singlet's hypercharge),
    "next": the WriteFeynArtsModel / FeynRulesToUFO arguments, "fr" (the text)}.
    """

    spec: Union[Dict[str, Any], str] = RuntimeField(description="Model spec: name, gauge, fields, parameters, interactions")
    output_path: Optional[str] = RuntimeField(default=None, description=".fr path; default <name>.fr")

    base_directory: str = StateField(description="Working directory")

    def _run(self) -> str:
        from .feynarts_model import fr_syntax_problems
        spec = self.spec
        if isinstance(spec, str):
            try:
                spec = json.loads(spec)
            except json.JSONDecodeError as exc:
                return self.format_error(error="Invalid Input", reason=f"spec is not JSON: {exc}")
        try:
            out = toy_model(spec)
        except ToyModelError as exc:
            return self.format_error(error="Invalid Model Spec", reason=str(exc))
        problems = fr_syntax_problems(out["fr"], out["load_sm"])
        if problems:
            return self.format_error(error="Invalid Model Spec", reason="; ".join(problems))
        name = str(spec.get("name", "ToyModel"))
        path = self.output_path or f"{name}.fr"
        path = path if os.path.isabs(path) else os.path.join(self.base_directory, path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as fh:
            fh.write(out["fr"])
        nxt = {"model_path": path, "lagrangian": out["lagrangian"], "load_sm": out["load_sm"]}
        return json.dumps({"status": "ok", "fr_path": path, "load_sm": out["load_sm"],
                           "lagrangian": out["lagrangian"], "particle_names": out["particle_names"],
                           "photon": out["photon"],
                           **({"notes": out["notes"]} if out["notes"] else {}),
                           "next": {"WriteFeynArtsModel": dict(nxt, name=name),
                                    "FeynRulesToUFO": dict(nxt, output_dir=f"{name}_UFO")},
                           "fr": out["fr"]}, indent=2)
