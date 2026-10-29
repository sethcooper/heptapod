"""
# feynarts_model.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
"""
WriteFeynArtsModel: a FeynRules model as a FeynArts model, with a manifest
and a linter.

FeynRules' WriteFeynArtsOutput writes <name>.mod (classes and couplings),
<name>.gen (generic couplings) and <name>.pars. Nothing checks that what it
wrote is the model: a class it could not express, a vertex it dropped, or a
gauge sector it left half-built only shows up as a wrong amplitude. So the
same Lagrangian is also written as a UFO, and the linter compares them:

  - loads: FeynArts (FeynCalc's copy) initialises the model, after FeynCalc's
    FAPatch, and sees as many couplings as the file has;
  - vertices: every UFO vertex, particle by particle, has a FeynArts coupling
    (class couplings expanded over their flavour indices, honouring
    IndexDelta), and every FeynArts coupling class has a UFO vertex;
  - gauge: in Feynman gauge every massive vector has a Goldstone and every
    vector a ghost, in couplings; in unitary gauge no Goldstone couples.

The FeynCalc-patched copy (FAPatch renames FourVector, MetricTensor, ... and
the SM mass symbols MH, MT, ... to FCGV["MH"], ...) is written beside the
original as <name>_FeynCalc; plain FeynArts reads the original.
"""

import itertools
import json
import os
import re
import shutil
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

_FACLASS = re.compile(r"^HEPTAPOD-FACLASS\|(.*)$", re.MULTILINE)
_FAOUT = re.compile(r"^HEPTAPOD-FAOUT\|dir=(?P<dir>[^|]*)\|mod=(?P<mod>[^|]*)\|gen=(?P<gen>.*)$", re.MULTILINE)
_UFOOUT = re.compile(r"^HEPTAPOD-UFOOUT\|dir=(?P<dir>.*)$", re.MULTILINE)
_GAUGE = re.compile(r"^HEPTAPOD-GAUGE\|feynman=(?P<f>\w+)\|source=(?P<s>.*)$", re.MULTILINE)
_LAGR = re.compile(r"^HEPTAPOD-LAGRANGIAN\|expression=(?P<e>[^|]*)\|symbols=(?P<s>[^|]*)\|terms=(?P<t>\d+)",
                   re.MULTILINE)
_COUPLING = re.compile(r"^C\[\s*(?P<fields>.+?)\s*\]\s*==\s*(?P<value>.*)$", re.MULTILINE)
_FIELD = re.compile(r"(?P<sign>-?)(?P<kind>[FSVUT])\[(?P<n>\d+)(?:,\s*\{(?P<idx>[^}]*)\})?\]")
_WARNING = re.compile(r"(Warning[^\n]*|^[\w`$]+::\w+:[^\n]*)", re.MULTILINE)
_STAGE = re.compile(r"^HEPTAPOD-STAGE\|(\w+)$", re.MULTILINE)


def staged_warnings(stdout: str) -> Dict[str, List[str]]:
    """FeynRules' warnings and messages, by stage (load, ufo, feynarts)."""
    out: Dict[str, List[str]] = {}
    stage, pos = "load", 0
    marks = [(m.start(), m.group(1)) for m in _STAGE.finditer(stdout)] + [(len(stdout), None)]
    for start, nxt in marks:
        for w in _WARNING.findall(stdout[pos:start]):
            w = w.strip()
            # FeynRules runs Simplify with TimeConstraint -> 0.01 on purpose.
            if "Wolfram >= 14" in w or "Simplify::time" in w:
                continue
            if w not in out.get(stage, []):
                out.setdefault(stage, []).append(w)
        stage, pos = nxt, start
    return out


@dataclass
class FAClass:
    field: str                  # "F[2]"
    name: str                   # FeynRules class name, "l"
    names: List[str]            # UFO particle names of the members, "e-", ...
    antinames: List[str]
    pdg: List[str]
    ghost: str                  # the vector a ghost belongs to, "" otherwise
    goldstone: str              # the vector a Goldstone belongs to
    indices: List[str]          # "Generation", "Colour", "Gluon"
    selfconjugate: bool
    massive: bool = False       # from the .mod's Mass entry

    @property
    def kind(self) -> str:
        return {"F": "fermion", "S": "scalar", "V": "vector", "U": "ghost", "T": "tensor"}[self.field[0]]

    def as_dict(self) -> Dict[str, object]:
        d = dict(self.__dict__)
        d["kind"] = self.kind
        return d


_FRMEMBER = re.compile(r"^HEPTAPOD-FRMEMBER\|(.*)$", re.MULTILINE)


def _kv(line: str) -> Dict[str, str]:
    return dict(item.split("=", 1) for item in line.split("|") if "=" in item)


def _class_props(mod_path: str) -> Dict[str, Dict[str, object]]:
    """Mass, SelfConjugate and Indices of each class in M$ClassesDescription."""
    text = Path(mod_path).read_text(errors="replace")
    start = text.find("M$ClassesDescription")
    end = text.find("(*", start + 1) if start >= 0 else -1
    block = text[start:end if end > 0 else len(text)]
    heads = list(re.finditer(r"(?P<f>[FSVUT]\[\d+\])\s*==\s*\{", block))
    out = {}
    for i, h in enumerate(heads):
        body = block[h.end(): heads[i + 1].start() if i + 1 < len(heads) else len(block)]
        mass = re.search(r"Mass\s*->\s*([^,\n}]+)", body)
        sc = re.search(r"SelfConjugate\s*->\s*(\w+)", body)
        idx = re.search(r"Indices\s*->\s*\{([^}]*)\}", body)
        out[h.group("f")] = {
            "mass": mass.group(1).strip() if mass else "0",
            "selfconjugate": bool(sc and sc.group(1) == "True"),
            "indices": re.findall(r"Index\[(\w+)\]", idx.group(1)) if idx else []}
    return out


def class_translation(mod_path: str) -> Dict[str, str]:
    """FR$ClassesTranslation of a .mod: FeynRules name -> FeynArts field,
    empty unless FeynRules redefined the classes."""
    text = Path(mod_path).read_text(errors="replace")
    m = re.search(r"FR\$ClassesTranslation\s*=\s*\{(.*?)\};", text, re.DOTALL)
    if not m:
        return {}
    return dict(re.findall(r"(\w+)\s*->\s*([FSVUT]\[\d+\])", m.group(1)))


def build_classes(stdout: str, mod_path: str) -> Tuple[List[FAClass], bool]:
    """The FeynArts classes the .mod actually has, in UFO names, and whether
    FeynRules redefined them (one particle per class)."""
    props = _class_props(mod_path)
    translation = class_translation(mod_path)
    if not translation:
        classes = parse_classes(stdout)
        for c in classes:
            c.massive = props.get(c.field, {}).get("mass", "0") not in ("0", "")
        return classes, False
    members = {kv["name"]: kv for kv in (_kv(m.group(1)) for m in _FRMEMBER.finditer(stdout))}
    classes = []
    for name, fld in translation.items():
        info = members.get(name, {})
        p = props.get(fld, {})
        classes.append(FAClass(fld, name, [info.get("particle", name)], [info.get("anti", name + "~")],
                               [info["pdg"]] if info.get("pdg") else [], info.get("ghost", ""),
                               info.get("goldstone", ""), list(p.get("indices", [])),
                               bool(p.get("selfconjugate", False)),
                               massive=p.get("mass", "0") not in ("0", "")))
    return classes, True


def parse_classes(stdout: str) -> List[FAClass]:
    out = []
    for m in _FACLASS.finditer(stdout or ""):
        kv = dict(item.split("=", 1) for item in m.group(1).split("|") if "=" in item)
        split = (lambda s: [x for x in s.split(",") if x])
        out.append(FAClass(kv.get("field", ""), kv.get("class", ""), split(kv.get("names", "")),
                           split(kv.get("antinames", "")), split(kv.get("pdg", "")),
                           kv.get("ghost", ""), kv.get("goldstone", ""), split(kv.get("indices", "")),
                           kv.get("selfconjugate", "") == "True"))
    return out


@dataclass
class Coupling:
    fields: List[Tuple[bool, str, List[str]]]    # (anti, "F[2]", class-index names)
    value: str

    def key(self) -> Tuple[str, ...]:
        return tuple(sorted(("-" if a else "") + f for a, f, _ in self.fields))


def parse_couplings(mod_path: str) -> List[Coupling]:
    """The C[...] == {...} entries of a FeynArts .mod file."""
    text = Path(mod_path).read_text(errors="replace")
    out = []
    for m in _COUPLING.finditer(text):
        fields = []
        for f in _FIELD.finditer(m.group("fields")):
            idx = [i.strip() for i in (f.group("idx") or "").split(",") if i.strip()]
            fields.append((f.group("sign") == "-", f"{f.group('kind')}[{f.group('n')}]", idx))
        out.append(Coupling(fields, m.group("value")))
    return out


def untied_indices(mod_path: str) -> Dict[str, object]:
    """Couplings with an index that occurs nowhere in the coupling value.
    FeynRules writes them when a field's own indices are left implicit in
    the Lagrangian -- Sqbar Sq for a coloured scalar instead of
    Sqbar[c] Sq[c] (a fermion's are contracted inside Fbar.F, a scalar's are
    not) -- and the UFO it writes has colour '1' there too. FeynArts then sums
    every vertex's index on its own: a scalar loop gets N_c^(vertices)
    instead of N_c. Reported, not patched: the fix is in the .fr, and it
    fixes the UFO as well."""
    text = Path(mod_path).read_text(errors="replace")
    found = []
    for m in _COUPLING.finditer(text):
        value = m.group("value")
        for f in _FIELD.finditer(m.group("fields")):
            idx = [i.strip() for i in (f.group("idx") or "").split(",") if i.strip()]
            loose = [i for i in idx if not re.search(rf"\b{re.escape(i)}\b", value)]
            if loose:
                found.append("C[" + m.group("fields").strip() + "]")
                break
    return {"verdict": "failed" if found else "ok", "couplings": found,
            **({"fix": "a field's indices are not tied at these vertices: write the scalar's indices "
                       "out and sum them in the .fr, e.g. Sqbar[cc] Sq[cc] and DC[Sqbar[cc], mu] "
                       "DC[Sq[cc], mu] (fermions in Fbar.F are contracted by FeynRules, scalars are "
                       "not). Otherwise each vertex sums its colour on its own"} if found else {})}

def _deltas(value: str) -> List[Tuple[str, str]]:
    return re.findall(r"IndexDelta\[\s*(\w+)\s*,\s*(\w+)\s*\]", value)


def expand_coupling(c: Coupling, classes: Dict[str, FAClass]) -> List[Tuple[str, ...]]:
    """Every particle-level vertex a class coupling stands for, as sorted
    UFO names. Flavour indices run over the class members; colour indices
    are not expanded (UFO does not either). Index pairs tied by
    IndexDelta in the coupling take equal values.

    FeynArts and the UFO name a vertex's fields in conjugate conventions:
    FeynRules writes the ghost-Goldstone vertex ubar_W c_A G as
    C[-U[32], U[1], -S[3]] for FeynArts and as (ghWm, ghA~, G+) in the UFO.
    For every other SM vertex the conjugate is itself a vertex, so this
    shows only for ghosts; the FeynArts side is conjugated here."""
    slots = []
    for anti, fld, idx in c.fields:
        cl = classes.get(fld)
        if cl is None:
            return []
        names = cl.names if anti else cl.antinames
        slots.append((names, idx[0] if (idx and len(names) > 1) else None))
    deltas = _deltas(c.value)
    choices = []
    for names, ix in slots:
        choices.append(list(enumerate(names)))
    out = set()
    for combo in itertools.product(*choices):
        val = {slots[i][1]: k for i, (k, _) in enumerate(combo) if slots[i][1]}
        if any(a in val and b in val and val[a] != val[b] for a, b in deltas):
            continue
        out.add(tuple(sorted(name for _, name in combo)))
    return sorted(out)


def ufo_vertex_names(ufo_dir: str) -> List[Tuple[str, ...]]:
    """Each UFO vertex as the sorted names of its particles."""
    from tools.ufo import parse_particles, parse_vertices
    by_var = {}
    for rec in parse_particles(os.path.join(ufo_dir, "particles.py")):
        by_var[str(rec.get("var") or rec.get("name"))] = str(rec.get("name"))
    return [tuple(sorted(by_var.get(p, p) for p in v["particles"])) for v in parse_vertices(ufo_dir)]


def compare_with_ufo(couplings: Sequence[Coupling], classes: Sequence[FAClass],
                     ufo_dir: str) -> Dict[str, object]:
    """UFO vertices without a FeynArts counterpart (particle level), and
    FeynArts coupling classes without a UFO vertex."""
    by_field = {c.field: c for c in classes}
    fa_vertices = set()
    fa_class_of = {}
    for c in couplings:
        for v in expand_coupling(c, by_field):
            fa_vertices.add(v)
            fa_class_of.setdefault(v, c.key())
    ufo = ufo_vertex_names(ufo_dir)
    missing = sorted({v for v in ufo if v not in fa_vertices})
    ufo_set = set(ufo)
    fa_only = sorted({c.key() for c in couplings
                      if not any(v in ufo_set for v in expand_coupling(c, by_field))})
    return {"verdict": "ok" if not missing and not fa_only else "failed",
            "ufo_vertices": len(ufo), "ufo_distinct": len(ufo_set),
            "feynarts_couplings": len(couplings), "feynarts_particle_vertices": len(fa_vertices),
            "ufo_without_feynarts": [list(v) for v in missing],
            "feynarts_without_ufo": [list(k) for k in fa_only]}


def gauge_check(classes: Sequence[FAClass], couplings: Sequence[Coupling],
                feynman: bool) -> Dict[str, object]:
    """Ghosts and Goldstones against the gauge.

    Feynman gauge: every massive or self-interacting vector (one in a
    coupling of vectors only) has a ghost class -- an abelian massless
    vector's ghost decouples -- and every massive vector a Goldstone that
    appears in some coupling. Unitary gauge: no
    Goldstone appears in a coupling. (Masses come from the .mod: a vector
    is massive when its class declares a Mass other than 0.)"""
    in_couplings = {f for c in couplings for _, f, _ in c.fields}
    vectors = [c for c in classes if c.kind == "vector"]
    # FeynRules marks the antighost of a charged vector as belonging to Wbar.
    ghosts_of = Counter(g.ghost[:-3] if g.ghost.endswith("bar") else g.ghost
                        for g in classes if g.kind == "ghost")
    goldstones = [g for g in classes if g.goldstone]
    problems = []
    if feynman:
        # A ghost is needed for a massive vector (its Goldstone sector) or a
        # self-interacting one; an abelian massless vector's ghost decouples.
        coupled_to_vectors = {f for c in couplings
                              if sum(1 for _, g, _ in c.fields if g.startswith("V")) >= 2
                              and any(not g.startswith("V") for _, g, _ in c.fields) is False
                              for _, f, _ in c.fields}
        for v in vectors:
            if not ghosts_of.get(v.name) and (v.massive or v.field in coupled_to_vectors):
                problems.append(f"vector {v.name} ({v.field}) has no ghost")
        for v in vectors:
            if v.massive and not any(g.goldstone == v.name and g.field in in_couplings for g in goldstones):
                problems.append(f"massive vector {v.name} ({v.field}) has no Goldstone in any coupling")
    else:
        for g in goldstones:
            if g.field in in_couplings:
                problems.append(f"Goldstone {g.name} ({g.field}) couples in unitary gauge")
    return {"verdict": "ok" if not problems else "failed", "gauge": "feynman" if feynman else "unitary",
            "vectors": [v.name for v in vectors],
            "ghosts": [g.name for g in classes if g.kind == "ghost"],
            "goldstones": [g.name for g in goldstones],
            "goldstones_in_couplings": [g.name for g in goldstones if g.field in in_couplings],
            "problems": problems}


#: The script's own symbols are Global`, like every symbol the .mod defines:
#: a model mass called m writes m[___] := m, which a global m = "<path>"
#: turned into a SetDelayed::write and a false "loads" failure. Hence the
#: heptapod prefix.
FA_LOAD_SCRIPT = r'''$FeynCalcStartupMessages = False; $FAPatch = False; $LoadAddOns = {{"FeynArts"}};
Needs["FeynCalc`"];
FAPatch[PatchModelsOnly -> True, FAModelsDirectory -> "{dir}", Quiet -> True];
heptapodLoadModel = "{model}";
heptapodLoadResult = Check[InitializeModel[heptapodLoadModel, GenericModel -> heptapodLoadModel], $Failed];
Print["HEPTAPOD-FALOAD|ok=", ToString[heptapodLoadResult =!= $Failed], "|couplings=",
  ToString[Length[FeynArts`M$CouplingMatrices]]];
'''

def feynarts_load(model_dir: str, name: str, wolframscript: str, timeout: int = 300) -> Dict[str, object]:
    """Copy the model to <dir>_FeynCalc, FAPatch it, and initialise it in
    FeynCalc's FeynArts. Returns the check and the patched directory."""
    from tools.wolfram.wolfram_runner import WolframRunner
    patched = str(Path(model_dir).with_name(Path(model_dir).name + "_FeynCalc"))
    shutil.rmtree(patched, ignore_errors=True)
    shutil.copytree(model_dir, patched)
    script = Path(patched).parent / f"{name}_feynarts_load.wl"
    script.write_text(FA_LOAD_SCRIPT.format(dir=patched, model=os.path.join(patched, name)))
    res = WolframRunner(wolframscript, timeout_sec=timeout).run_script(str(script))
    m = re.search(r"HEPTAPOD-FALOAD\|ok=(\w+)\|couplings=(\d+)", res.stdout)
    messages = sorted(set(re.findall(r"^[\w`$]+::\w+:.*$", res.stdout + "\n" + res.stderr, re.MULTILINE)))
    if not m:
        return {"verdict": "failed", "patched_dir": patched, "script": str(script),
                "messages": messages or [res.stderr.strip()[-800:]]}
    return {"verdict": "ok" if m.group(1) == "True" and not messages else "failed",
            "patched_dir": patched, "couplings_loaded": int(m.group(2)), "messages": messages,
            "script": str(script)}


# ── the .fr as FeynRules reads it ────────────────────────────────────

TOY_TEMPLATE = "the oneloop skill's references/toy_models.md and toy_model.fr"


def template_text(load_sm: bool) -> str:
    """The tested template a refusal hands back verbatim (agents did not always
    find the skill's reference files): the SM add-on for load_sm, else the
    standalone QED toy."""
    name = "sm_addon_doublet.fr" if load_sm else "toy_model.fr"
    path = Path(__file__).resolve().parents[2] / "skills" / "oneloop" / "references" / name
    try:
        return f"\n\nA tested model to copy ({name}):\n" + path.read_text()
    except OSError:
        return ""
_COMMENT = re.compile(r"\(\*.*?\*\)", re.DOTALL)
_FR_ENTRY = re.compile(r"(?<![\w$])([SFVUT])\s*\[\s*(\d+)\s*\]\s*==\s*\{")
_RESERVED = {"S", "F", "V", "U", "T", "E", "I", "N", "C", "D", "O", "K", "Pi", "Gamma", "Beta",
             "Zeta", "Re", "Im", "Abs", "Exp", "Log", "Sum", "Tr", "Det"}
#: FeynRules' SM.fr class names (tools/feynrules/test_files/models/SM.fr).
_SM_FR_CLASSES = {"A", "Z", "W", "G", "ghA", "ghZ", "ghWp", "ghWm", "ghG", "B", "Wi", "ghB", "ghWi",
                  "vl", "l", "uq", "dq", "LL", "lR", "QL", "uR", "dR", "H", "G0", "GP", "Phi"}
_FR_BARE = re.compile(r"(?<![\w$\[])([A-Za-z][A-Za-z0-9]*)\s*==\s*\{[^{}]*ClassName")


@dataclass
class FRClass:
    head: str            # "S[1]"
    name: str            # ClassName
    unphysical: bool
    goldstone: bool
    charge: Optional[str] = None   # QuantumNumbers' Q, as written


def fr_classes(text: str) -> List[FRClass]:
    """The particle classes a .fr declares: `S[1] == {ClassName -> X, ...}`."""
    text = _COMMENT.sub(" ", text)
    out = []
    starts = [m for m in _FR_ENTRY.finditer(text)]
    for i, m in enumerate(starts):
        body = text[m.end(): starts[i + 1].start() if i + 1 < len(starts) else len(text)]
        cn = re.search(r"ClassName\s*->\s*([A-Za-z][A-Za-z0-9]*)", body)
        if not cn:
            continue
        q = re.search(r"QuantumNumbers\s*->\s*\{[^}]*?(?<![\w$])Q\s*->\s*([^,}]+)", body)
        out.append(FRClass(f"{m.group(1)}[{m.group(2)}]", cn.group(1),
                           bool(re.search(r"Unphysical\s*->\s*True", body)),
                           bool(re.search(r"Goldstone\s*->", body)),
                           q.group(1).strip() if q else None))
    return out


def dangling_lines(text: str) -> List[int]:
    """The .fr's lines that start a statement Get would read on its own."""
    return _dangling_lines(_COMMENT.sub(lambda m: " " + "\n" * m.group(0).count("\n"), text))


def _dangling_lines(code: str) -> List[int]:
    """Line numbers (1-based) that start with + or - at bracket depth 0 after
    a line that did not end its statement with ';' -- Get reads the previous
    line as a complete statement."""
    depth, out, prev_open = 0, [], False
    for n, line in enumerate(code.splitlines(), 1):
        s = line.strip()
        if depth == 0 and prev_open and s[:1] in ("+", "-"):
            out.append(n)
        for ch in line:
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth = max(0, depth - 1)
        if s:
            prev_open = not s.endswith(";")
    return out


def fr_syntax_problems(text: str, load_sm: bool, dangling: bool = True) -> List[str]:
    """What FeynRules would silently misread. An empty list is no guarantee."""
    code = _COMMENT.sub(lambda m: " " + "\n" * m.group(0).count("\n"), text)   # line numbers kept
    problems = []
    if re.search(r"(?<![\w$])Declare\s*\[", code) or re.search(r"(?m)^\s*\$ModelName", code):
        problems.append("this is not FeynRules syntax (Declare[...] and $ModelName are not FeynRules; "
                        "the model name is M$ModelName, parameters M$Parameters = {p == {...}}, "
                        "particles M$ClassesDescription = {S[1] == {ClassName -> X, ...}})")
    if re.search(r'(?<![\w$])Get\s*\[|<<', code):
        problems.append("the .fr loads another file (Get[...] or <<): FeynRules reads the model from "
                        "its own directory, where that path does not exist. For the SM pass "
                        "load_sm=true (it loads SM.fr) and lagrangian='LSM + <yours>'")
    lost = _dangling_lines(code) if dangling else []
    if lost:
        problems.append(f"lines {lost[:5]} start with + or - outside any bracket: Mathematica ends the "
                        "statement at the line before, and the rest of the sum is silently lost. Put the "
                        "operator at the end of the previous line, or wrap the sum in parentheses")
    bare = _FR_BARE.findall(code)
    if bare:
        problems.append(f"particle classes {sorted(set(bare))} are written `X == {{ClassName -> X, ...}}`; "
                        "FeynRules needs the field type and a number as the key: S[1] == {...} for a "
                        "scalar, F[1] for a fermion, V[1] for a vector (S, F, V themselves are "
                        "FeynArts' heads and must not be used as class names)")
    names = set(re.findall(r"ClassName\s*->\s*([A-Za-z][A-Za-z0-9]*)", code))
    reserved = sorted(names & _RESERVED)
    if reserved:
        problems.append(f"class names {reserved} are FeynArts field heads or Mathematica built-ins "
                        "(E is e = 2.718..., I the imaginary unit, N, C, D, O, K functions); the class "
                        "vanishes from the model -- rename it (e.g. E -> Lv)")
    if load_sm:
        clash = sorted(names & _SM_FR_CLASSES)
        if clash:
            problems.append(f"class names {clash} are SM.fr's own, which load_sm=true loads too "
                            "(FeynRules: 'Doubly defined particle classes'); rename them, or drop "
                            "load_sm for a standalone model with its own photon")
    if not fr_classes(text) and not load_sm:
        problems.append("the .fr declares no particle classes (M$ClassesDescription = "
                        "{S[1] == {ClassName -> ..., ...}, ...})")
    return problems


def model_check(declared: Sequence[FRClass], classes: Sequence["FAClass"],
                in_couplings: Dict[str, int], load_sm: bool, lagrangian: Optional[str]) -> Dict[str, object]:
    """Each physical class the .fr declares is in the FeynArts model and in a vertex."""
    by_name = {c.name: c for c in classes}
    problems = []
    for d in declared:
        if d.unphysical or d.goldstone or d.head.startswith("U"):
            continue
        c = by_name.get(d.name)
        if c is None:
            problems.append(f"{d.name} ({d.head} in the .fr) is not in the FeynArts model")
        elif in_couplings.get(c.field, 0) == 0:
            problems.append(f"{d.name} ({c.field}) is in no vertex: the Lagrangian "
                            f"{lagrangian!r} has no term with it (a missing term, a misspelt "
                            "field, or a term not added to the total; a charged field gets its "
                            "photon vertex from a kinetic term with DC[...])")
    return {"verdict": "failed" if problems else "ok",
            "declared": [d.name for d in declared], "problems": problems}


def _nonzero_charge(q: Optional[str]) -> bool:
    if q is None:
        return False
    try:
        return Fraction(q.replace(" ", "")) != 0
    except (ValueError, ZeroDivisionError):
        return True          # a symbolic charge: assume charged


def photon_check(declared: Sequence[FRClass], classes: Sequence["FAClass"],
                 couplings: Sequence[Coupling]) -> Dict[str, object]:
    """Every electrically charged physical class the .fr declares is in a
    vertex with the photon (class A). A charged field couples to the photon
    only through its covariant derivative: a kinetic term written with del,
    or, on SM.fr, a singlet declared without its hypercharge Y, leaves a
    'charged' particle that never sees the photon, and nothing else says so."""
    by_name = {c.name: c for c in classes}
    photon = by_name.get("A")
    charged = [d for d in declared if _nonzero_charge(d.charge) and not d.unphysical
               and not d.goldstone and not d.head.startswith("U")]
    if photon is None:
        return {"verdict": "not_applicable", "charged": [d.name for d in charged],
                "reason": "the model has no photon class named A (the toy models' and SM.fr's), so the "
                          "charged fields' photon vertices were not checked"}
    problems = []
    for d in charged:
        c = by_name.get(d.name)
        if c is None:
            continue           # the model lint reports a class missing from FeynArts
        if not any(any(f == c.field for _, f, _ in cp.fields) and any(f == photon.field for _, f, _ in cp.fields)
                   for cp in couplings):
            problems.append(f"{d.name} ({c.field}) has charge {d.charge} but no photon vertex: its kinetic term "
                            "needs DC[...] (not del), and on SM.fr (load_sm) a singlet needs its hypercharge, "
                            "QuantumNumbers -> {Q -> q, Y -> q}")
    return {"verdict": "failed" if problems else "ok", "charged": [d.name for d in charged],
            "problems": problems}


class WriteFeynArtsModel(BaseTool):
    """
    Write a FeynArts model from a FeynRules .fr model (FeynRules'
    WriteFeynArtsOutput), with a manifest, and lint it.

    Writes output_dir/<name>/<name>.mod, .gen, .pars (for FeynArts), and
    output_dir/<name>_FeynCalc/ -- the same files after FeynCalc's FAPatch,
    which FeynCalc's FeynArts needs (SM masses MH, MT, ... become
    FCGV["MH"], ...). With compare_ufo (default) the same Lagrangian is
    written as a UFO, output_dir/<name>_UFO, for the vertex check.

    Gauge: feynman_gauge true (default) builds the Goldstone and ghost
    sector, which a one-loop calculation in Feynman gauge needs; false is
    unitary gauge. R_xi with a general xi is not produced by FeynRules'
    models (GaugeXi stays 1 in their Lagrangians).

    The Lagrangian is required, as for FeynRulesToUFO ('LSM + LBSM',
    'LSM + LkinS1 + L1YukRR', ...). It is used in classes: expand_indices
    stays off, because FeynArts refuses a Lagrangian written in class
    members. Couplings are written out in the model's parameters unless
    rename_couplings (FeynRules' gc1, gc2, ...).

    The .fr is checked first for what FeynRules would silently misread
    (class keys written `X == {...}` instead of `S[1] == {...}`, non-FeynRules
    syntax); a worked standalone model is the oneloop skill's
    references/toy_models.md. A toy model with its own photon (and gluon)
    takes load_sm=false.

    Lint verdicts ("ok"/"failed", with details):
      - model: every physical class the .fr declares is in the FeynArts
        model and in at least one vertex;
      - index_deltas: no coupling leaves a field's index untied (a coloured
        scalar written Sqbar Sq instead of Sqbar[c] Sq[c] in the .fr, which
        would count N_c once per vertex);
      - loads: FeynCalc's FeynArts initialises the patched model without
        messages, seeing every coupling;
      - vertices: every UFO vertex has a FeynArts coupling, particle by
        particle (flavour indices expanded, IndexDelta honoured), and every
        FeynArts coupling a UFO vertex;
      - gauge: Feynman gauge has a ghost per vector and a coupled Goldstone
        per massive vector; unitary gauge has no coupled Goldstone;
      - photon: every electrically charged field (QuantumNumbers Q != 0) is
        in a vertex with the photon (a charged field without one has a
        kinetic term without DC, or, on SM.fr, no hypercharge Y).

    Output: {"status", "model": {"dir", "mod", "gen", "feyncalc_dir",
    "name"}, "manifest": {"fields", "couplings", "gauge", "lagrangian",
    "feynrules_warnings"}, "lint": {"index_deltas", "model", "photon", "loads", "vertices", "gauge"},
    "manifest_file", "logs"}
    """

    model_path: Optional[str] = RuntimeField(default=None, description="FeynRules .fr model file (or toy_spec)")
    toy_spec: Optional[Union[Dict[str, Any], str]] = RuntimeField(
        default=None, description="Instead of model_path: a WriteToyModel spec (fields, parameters, interactions)")
    lagrangian: Optional[str] = RuntimeField(default=None, description="Total Lagrangian, e.g. 'LSM + LBSM'")
    name: str = RuntimeField(default="Model_FA", description="FeynArts model name: letters, digits, _")
    output_dir: str = RuntimeField(default="feynarts_models", description="Parent directory for the model")
    load_sm: bool = RuntimeField(default=True, description="Load FeynRules' SM.fr first (the model is an add-on to the SM); false for a standalone toy model with its own photon")
    feynman_gauge: bool = RuntimeField(default=True, description="Feynman gauge (true) or unitary (false)")
    rename_couplings: bool = RuntimeField(default=False, description="FeynRules' gc1, gc2, ... names")
    compare_ufo: bool = RuntimeField(default=True, description="Also write the UFO and compare vertices")
    timeout_sec: int = RuntimeField(default=1200, description="Walltime limit for FeynRules")

    base_directory: str = StateField(description="Working directory")
    feynrules_path: Optional[str] = StateField(default=None, description="FeynRules installation root")
    wolframscript_path: Optional[str] = StateField(default=None, description="wolframscript")

    def _abs(self, p: str) -> str:
        return str(Path(p if os.path.isabs(p) else os.path.join(self.base_directory, p)).resolve())

    def _run(self) -> str:
        from .feynrules import _invoke_generator, _candidate_symbols
        from config_resolve import resolve_path
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", self.name):
            return self.format_error(error="Invalid Input", reason="name: letters, digits and _")
        if self.toy_spec is not None:
            from .toy_model import apply_toy_spec
            err = apply_toy_spec(self, self.toy_spec, self._abs(self.output_dir if os.path.isdir(
                self._abs(self.output_dir)) else "."), self.name)
            if err:
                return self.format_error(error="Invalid Model Spec", reason=err)
        if not self.model_path:
            return self.format_error(error="Missing Parameter", reason="model_path (a .fr) or toy_spec")
        model = self._abs(self.model_path)
        if not os.path.isfile(model):
            return self.format_error(error="File Not Found", reason=model)
        fr_text = Path(model).read_text(errors="replace")
        problems = fr_syntax_problems(fr_text, self.load_sm)
        if problems:
            return self.format_error(
                error="Invalid FeynRules Model",
                reason="; ".join(problems) + f". A worked standalone model: {TOY_TEMPLATE}."
                       + template_text(self.load_sm))
        fr = self.feynrules_path or resolve_path("feynrules_path")
        ws = self.wolframscript_path or resolve_path("wolframscript_path") or "wolframscript"
        out_dir = self._abs(self.output_dir)
        os.makedirs(out_dir, exist_ok=True)
        for old in (self.name, f"{self.name}_UFO", f"{self.name}_FeynCalc"):
            shutil.rmtree(os.path.join(out_dir, old), ignore_errors=True)
        args = [f"ModelName={self.name}",
                f"LoadSM={'true' if self.load_sm else 'false'}",
                f"FeynmanGauge={'true' if self.feynman_gauge else 'false'}",
                f"RenameCouplings={'true' if self.rename_couplings else 'false'}",
                f"UFOCheck={'true' if self.compare_ufo else 'false'}"]
        if self.lagrangian:
            args.append(f"Lagrangian={self.lagrangian}")
        proc, out_log, err_log, err = _invoke_generator(
            ws, fr, model, out_dir, os.path.join(out_dir, f"{self.name}_logs"), self.timeout_sec,
            self.base_directory, args, driver_name="FeynArts_generator.wl")
        logs = {"stdout": out_log, "stderr": err_log}
        if err:
            return self.format_error(error="FeynRules Error", reason=err)
        stdout = proc.stdout or ""
        fa = _FAOUT.search(stdout)
        if proc.returncode != 0 or not fa:
            msgs = [m for m in dict.fromkeys(ln.strip() for ln in stdout.splitlines()
                                             if re.match(r"^[\w`$]+::\w+:", ln.strip()))][:6]
            reason = next((ln.strip() for ln in stdout.splitlines() if ln.startswith("Error:")),
                          f"FeynRules stopped before writing the model (exit {proc.returncode})"
                          + (f"; its messages: {msgs}" if msgs else "") + f"; log {out_log}")
            cands = _candidate_symbols(stdout)
            if cands and "Lagrangian" in reason:
                reason += f" Symbols this model defines: {cands}"
            return self.format_error(error="FeynRules Error", reason=reason)

        classes, redefined = build_classes(stdout, fa.group("mod"))
        deltas = untied_indices(fa.group("mod"))
        couplings = parse_couplings(fa.group("mod"))
        masses = {f: p["mass"] for f, p in _class_props(fa.group("mod")).items()}
        g = _GAUGE.search(stdout)
        feynman = bool(g and g.group("f") == "True")
        lag = _LAGR.search(stdout)
        warnings = staged_warnings(stdout)
        in_couplings = Counter(f for c in couplings for _, f, _ in c.fields)
        manifest = {
            "fields": [{**c.as_dict(), "mass": masses.get(c.field, "0"),
                        "in_couplings": in_couplings.get(c.field, 0)} for c in classes],
            "couplings": len(couplings),
            "classes_redefined": redefined,
            "gauge": {"feynman": feynman, "source": g.group("s") if g else None},
            "lagrangian": {"expression": lag.group("e"), "terms": int(lag.group("t"))} if lag else None,
            "feynrules_warnings": warnings,
        }
        lint = {"index_deltas": deltas,
                "model": model_check(fr_classes(fr_text), classes, in_couplings, self.load_sm,
                                     self.lagrangian),
                "photon": photon_check(fr_classes(fr_text), classes, couplings),
                "gauge": gauge_check(classes, couplings, feynman)}
        u = _UFOOUT.search(stdout)
        if u and os.path.isfile(os.path.join(u.group("dir"), "vertices.py")):
            lint["vertices"] = compare_with_ufo(couplings, classes, u.group("dir"))
            lint["vertices"]["ufo_dir"] = u.group("dir")
        elif self.compare_ufo:
            lint["vertices"] = {"verdict": "not_applicable", "reason": "the UFO was not written"}
        lint["loads"] = feynarts_load(fa.group("dir"), self.name, ws)
        model_info = {"name": self.name, "dir": fa.group("dir"), "mod": fa.group("mod"),
                      "gen": fa.group("gen"), "feyncalc_dir": lint["loads"].get("patched_dir")}
        manifest_file = os.path.join(out_dir, f"{self.name}.manifest.json")
        Path(manifest_file).write_text(json.dumps({"model": model_info, "manifest": manifest,
                                                   "lint": lint}, indent=2))
        ok = all(v.get("verdict") in ("ok", "not_applicable") for v in lint.values())
        return json.dumps({"status": "ok" if ok else "lint_failed", "model": model_info,
                           "manifest": manifest, "lint": lint, "manifest_file": manifest_file,
                           "logs": logs}, indent=2)
