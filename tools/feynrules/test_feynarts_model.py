"""
# test_feynarts_model.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

WriteFeynArtsModel and its linter.

The parsing and comparison are tested on their own; the rest needs
FeynRules and Mathematica (feynrules_path, wolframscript_path) and skips
without them. Runs are serial: each starts a Mathematica kernel.
"""

import json
import re
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.feynrules.feynarts_model import (Coupling, FAClass, compare_with_ufo, expand_coupling,
                                            gauge_check, parse_couplings, staged_warnings)

MODELS = Path(__file__).parent / "test_files" / "models"


# ── parsing and comparison ───────────────────────────────────────────────

def _classes():
    return {c.field: c for c in [
        FAClass("V[1]", "A", ["a"], ["a"], ["22"], "", "", [], True),
        FAClass("V[3]", "W", ["W+"], ["W-"], ["24"], "", "", [], False, massive=True),
        FAClass("F[2]", "l", ["e-", "mu-", "ta-"], ["e+", "mu+", "ta+"], ["11", "13", "15"], "", "",
                ["Generation"], False),
        FAClass("S[3]", "GP", ["G+"], ["G-"], ["251"], "", "W", [], False),
        FAClass("U[1]", "ghA", ["ghA"], ["ghA~"], [], "A", "", [], False),
        FAClass("U[32]", "ghWm", ["ghWm"], ["ghWm~"], [], "Wbar", "", [], False),
    ]}


def test_couplings_parse_with_class_indices(tmp_path):
    mod = tmp_path / "x.mod"
    mod.write_text("M$CouplingMatrices = {\n"
                   "C[ -F[2, {e1x2}] , F[2, {e2x2}] , V[1] ] == {{(-I)*EL*IndexDelta[e1x2, e2x2], 0}},\n\n"
                   "C[ S[3] , -S[3] , V[1] ] == {{I*EL, 0}, {(-I)*EL, 0}}\n}\n")
    cs = parse_couplings(str(mod))
    assert [c.key() for c in cs] == [("-F[2]", "F[2]", "V[1]"), ("-S[3]", "S[3]", "V[1]")]
    assert cs[0].fields[0] == (True, "F[2]", ["e1x2"])


def test_expansion_honours_index_delta():
    """The photon coupling to leptons is generation diagonal: 3 vertices, not 9."""
    c = Coupling([(True, "F[2]", ["e1x2"]), (False, "F[2]", ["e2x2"]), (False, "V[1]", [])],
                 "{{(-I)*EL*IndexDelta[e1x2, e2x2], 0}}")
    assert expand_coupling(c, _classes()) == [("a", "e+", "e-"), ("a", "mu+", "mu-"), ("a", "ta+", "ta-")]
    off = Coupling(c.fields, "{{yl[e1x2, e2x2], 0}}")
    assert len(expand_coupling(off, _classes())) == 9


def test_ghost_vertex_is_compared_conjugated(tmp_path):
    """FeynRules writes ubar_W c_A G as C[-U[32], U[1], -S[3]] for FeynArts and as
    (ghWm, ghA~, G+) in the UFO: one side conjugated (observed on SM.fr)."""
    c = Coupling([(True, "U[32]", []), (False, "U[1]", []), (True, "S[3]", [])], "{{1, 0}}")
    assert expand_coupling(c, _classes()) == [("G+", "ghA~", "ghWm")]


def test_gauge_check_finds_missing_sectors():
    cl = list(_classes().values())
    w_coupled = [Coupling([(False, "S[3]", []), (True, "V[3]", []), (False, "V[1]", [])], "")]
    assert gauge_check(cl, w_coupled, feynman=True)["verdict"] == "ok"
    no_w_ghost = [c for c in cl if c.name != "ghWm"]
    bad = gauge_check(no_w_ghost, w_coupled, feynman=True)
    assert bad["verdict"] == "failed" and bad["problems"] == ["vector W (V[3]) has no ghost"]
    # a massless photon coupling to no other vector needs no ghost
    no_photon_ghost = [c for c in cl if c.name != "ghA"]
    assert gauge_check(no_photon_ghost, w_coupled, feynman=True)["verdict"] == "ok"
    aww = [Coupling([(False, "V[1]", []), (False, "V[3]", []), (True, "V[3]", [])], "")]
    assert "vector A (V[1]) has no ghost" in gauge_check(no_photon_ghost, w_coupled + aww, feynman=True)["problems"]
    no_goldstone = gauge_check(cl, [], feynman=True)
    assert "massive vector W (V[3]) has no Goldstone in any coupling" in no_goldstone["problems"]
    unitary = gauge_check(cl, w_coupled, feynman=False)
    assert unitary["verdict"] == "failed" and "GP" in unitary["problems"][0]
    assert gauge_check(cl, [], feynman=False)["verdict"] == "ok"


def test_class_translation_is_read(tmp_path):
    """A Lagrangian naming single flavours makes FeynRules give each particle its
    own class and record the map in the .mod (seen on the S1 leptoquark model)."""
    from tools.feynrules.feynarts_model import class_translation
    mod = tmp_path / "x.mod"
    mod.write_text("FR$ClassesTranslation={A -> V[1], e -> F[4], S1 -> S[4]};\n")
    assert class_translation(str(mod)) == {"A": "V[1]", "e": "F[4]", "S1": "S[4]"}
    mod.write_text("FR$ClassesTranslation={};\n")
    assert class_translation(str(mod)) == {}


def test_warnings_are_attributed_to_stages():
    out = ("Inner::incom: x\nHEPTAPOD-STAGE|ufo\nCreateDirectory::eexist: y\n"
           "HEPTAPOD-STAGE|feynarts\nWarning: Class members in the Lagrangian!\n")
    assert staged_warnings(out) == {"load": ["Inner::incom: x"], "ufo": ["CreateDirectory::eexist: y"],
                                    "feynarts": ["Warning: Class members in the Lagrangian!"]}


# ── with FeynRules ───────────────────────────────────────────────────────

def _configured():
    from config_resolve import resolve_path
    fr, ws = resolve_path("feynrules_path"), resolve_path("wolframscript_path")
    if not fr or not Path(fr, "FeynRules.m").is_file() or not ws:
        pytest.skip("feynrules_path / wolframscript_path not configured")
    return fr, ws


def _write(tmp, **kw):
    from tools.feynrules.feynarts_model import WriteFeynArtsModel
    fr, ws = _configured()
    out = WriteFeynArtsModel(base_directory=str(tmp), feynrules_path=fr,
                             wolframscript_path=ws, **kw)._run()
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return {"status": "error", "text": out}


@pytest.fixture(scope="module")
def sm_feynman(tmp_path_factory):
    return _write(tmp_path_factory.mktemp("smfa"), model_path=str(MODELS / "SM.fr"),
                  lagrangian="LSM", load_sm=False, name="SMFA", feynman_gauge=True)


def test_sm_in_feynman_gauge(sm_feynman):
    """SM.fr -> FeynArts: loads in FeynCalc's FeynArts with every coupling, every
    one of the UFO's vertices has a FeynArts counterpart and vice versa, and the
    Goldstone and ghost sectors are complete."""
    r = sm_feynman
    assert r["status"] == "ok", r.get("lint")
    lint = r["lint"]
    assert lint["loads"]["couplings_loaded"] == r["manifest"]["couplings"] == 93
    v = lint["vertices"]
    assert v["verdict"] == "ok" and v["ufo_vertices"] == 163
    g = lint["gauge"]
    assert g["gauge"] == "feynman" and set(g["goldstones_in_couplings"]) == {"G0", "GP"}
    assert set(g["ghosts"]) == {"ghA", "ghZ", "ghWp", "ghWm", "ghG"}
    assert Path(r["model"]["mod"]).is_file() and Path(r["model"]["feyncalc_dir"]).is_dir()


def test_sm_in_unitary_gauge(tmp_path):
    """Unitary gauge: no Goldstone couples, and the vertices still match the UFO."""
    r = _write(tmp_path, model_path=str(MODELS / "SM.fr"), lagrangian="LSM", load_sm=False,
               name="SMU", feynman_gauge=False)
    assert r["status"] == "ok", r.get("lint")
    assert r["lint"]["gauge"]["goldstones_in_couplings"] == []
    assert r["lint"]["vertices"]["verdict"] == "ok"


def test_leptoquark_model_loads(tmp_path):
    """The S1 leptoquark (3, 1, -1/3) on the SM."""
    r = _write(tmp_path, model_path=str(MODELS / "S1_LQ_RR.fr"), lagrangian="LSM + LBSM",
               load_sm=True, name="S1LQ")
    assert r["status"] == "ok", r.get("lint")
    assert r["manifest"]["classes_redefined"]
    s1 = [f for f in r["manifest"]["fields"] if f["names"] == ["S1"]]
    assert s1 and s1[0]["field"].startswith("S[") and s1[0]["indices"] == ["Colour"]


def test_missing_lagrangian_lists_candidates(tmp_path):
    r = _write(tmp_path, model_path=str(MODELS / "S1_LQ_RR.fr"), lagrangian="LNotThere",
               load_sm=True, name="Bad")
    assert "LNotThere" in json.dumps(r) and "LBSM" in json.dumps(r)


def test_untied_scalar_colour_is_a_lint_failure(tmp_path):
    """A coloured scalar written Sqbar Sq (no index) in the .fr: FeynRules writes
    its colour-singlet couplings with the field's and antifield's indices untied
    (and colour '1' in the UFO), so FeynArts sums each vertex's index on its own --
    N_c^3 for a triangle, N_c^2 for the seagull. Found by the oneloop toy-template
    test, whose X -> gamma gamma scalar loop was off Djouadi's A_0 and UV
    divergent. The fermion's IndexDelta is fine."""
    from tools.feynrules.feynarts_model import untied_indices
    mod = tmp_path / "m.mod"
    mod.write_text(COLOURED_SCALAR_MOD)
    out = untied_indices(str(mod))
    assert out["verdict"] == "failed" and len(out["couplings"]) == 3
    assert "Sqbar[cc] Sq[cc]" in out["fix"]
    assert not any("F[1" in c for c in out["couplings"])

def test_misread_fr_syntax_is_named():
    """Patterns from pilot agents' .fr files that FeynRules misreads silently."""
    from tools.feynrules.feynarts_model import fr_syntax_problems, fr_classes
    sm = (Path(__file__).parent / "test_files" / "models" / "SM.fr").read_text()
    assert fr_syntax_problems(sm, load_sm=False) == []
    assert len(fr_classes(sm)) > 20
    bare = 'M$ClassesDescription = { S == {ClassName -> S, Mass -> mS} };'
    assert "S[1] == {...}" in fr_syntax_problems(bare, load_sm=False)[0]
    lost = 'M$ClassesDescription = {S[1] == {ClassName -> X}};\nL := a X^2\n   + b X^3;\n'
    assert "silently lost" in fr_syntax_problems(lost, False)[0]
    assert "built-ins" in fr_syntax_problems('M$ClassesDescription = {F[1] == {ClassName -> E}};', False)[0]
    assert "not FeynRules syntax" in fr_syntax_problems('$ModelName = "x"; Declare[Particle] = {};', True)[0]
    assert "load_sm=true" in fr_syntax_problems('Get["models/sm.fr"];\nM$ClassesDescription = {S[4] == {ClassName -> X}};', True)[0]


def test_feynrules_to_ufo_refuses_a_misread_fr(tmp_path):
    from tools.feynrules.feynrules import FeynRulesToUFOTool
    fr = tmp_path / "bad.fr"
    fr.write_text('M$ModelName = "B";\nM$ClassesDescription = { F == {ClassName -> F, Mass -> mF} };\n')
    out = FeynRulesToUFOTool(base_directory=str(tmp_path), model_path=str(fr), lagrangian="L",
                             load_sm=False, feynrules_path=str(tmp_path), wolframscript_path="wolframscript")._run()
    assert "Invalid FeynRules Model" in out and "toy_model.fr" in out, out


# ── fail closed: false alarms and silent decouplings ────────────────────

#: A model whose mass is called m (from a dogfood run): the .mod defines
#: m[___] := m, which collided with the load script's own global m.
MASS_CALLED_M_FR = """M$ModelName = "PhiChiScat";

M$Parameters = {
  kap == { ParameterType -> External, Value -> 200.0, InteractionOrder -> {NP, 1} }
};

M$ClassesDescription = {
  S[1] == { ClassName -> phi, SelfConjugate -> True, Mass -> {m, 100.0}, Width -> 0, ParticleName -> "phi", PDG -> 9000101 },
  S[2] == { ClassName -> chi, SelfConjugate -> True, Mass -> {M, 250.0}, Width -> 0, ParticleName -> "chi", PDG -> 9000102 }
};

LKin := (1/2 del[phi, mu] del[phi, mu] - 1/2 m^2 phi^2) +
        (1/2 del[chi, mu] del[chi, mu] - 1/2 M^2 chi^2);
LInt := (-kap/2 chi phi phi);
LToy := LKin + LInt;
"""


def test_load_script_symbols_cannot_collide_with_the_model():
    """The FeynCalc load script's globals are heptapod-prefixed, as are the
    generators': a model symbol m, r, params ... is not overwritten."""
    from tools.feynrules.feynarts_model import FA_LOAD_SCRIPT
    code = FA_LOAD_SCRIPT.format(dir="/x", model="/x/M")
    assigned = re.findall(r"^(\w+)\s*=", code, re.MULTILINE)
    assert assigned and all(a.startswith("heptapod") for a in assigned), assigned
    for wl in ("FeynArts_generator.wl", "UFO_generator.wl"):
        text = (Path(__file__).parent / wl).read_text()
        top = set(re.findall(r"^([A-Za-z]\w*)(?:\[[^\]\n=]*\])?\s*:?=(?!=)", text, re.MULTILINE))
        assert all(t.startswith("heptapod") or t in {"FeynmanGauge"} for t in top), (wl, top)


def test_a_mass_called_m_loads(tmp_path):
    """Regression: a mass named m gave lint 'loads' failed (SetDelayed::write on
    the load script's m = "<path>") for a model that loads fine."""
    fr = tmp_path / "PhiChiScat.fr"
    fr.write_text(MASS_CALLED_M_FR)
    r = _write(tmp_path, model_path=str(fr), lagrangian="LToy", load_sm=False, name="PCScat")
    assert r["lint"]["loads"]["verdict"] == "ok", r["lint"]["loads"]
    assert r["status"] == "ok", r.get("lint")


def _photon_fixture(with_photon_vertex: bool):
    from tools.feynrules.feynarts_model import FRClass, photon_check
    declared = [FRClass("V[1]", "A", False, False), FRClass("F[5]", "Lx", False, False, "-1"),
                FRClass("S[4]", "Sz", False, False)]
    classes = [FAClass("V[1]", "A", ["a"], ["a"], ["22"], "", "", [], True),
               FAClass("F[5]", "Lx", ["Lx-"], ["Lx+"], [], "", "", [], False),
               FAClass("S[4]", "Sz", ["Sz"], ["Sz"], [], "", "", [], True)]
    cps = [Coupling([(True, "F[5]", []), (False, "F[5]", []), (False, "S[4]", [])], "{{-I yS, 0}}")]
    if with_photon_vertex:
        cps.append(Coupling([(True, "F[5]", []), (False, "F[5]", []), (False, "V[1]", [])], "{{I EL, 0}}"))
    return photon_check(declared, classes, cps)


def test_a_charged_field_without_a_photon_vertex_is_a_lint_failure():
    """s_to_zgamma dogfood: a charge -1 lepton declared on SM.fr without its
    hypercharge had no photon or Z vertex, and every lint said ok."""
    bad = _photon_fixture(False)
    assert bad["verdict"] == "failed" and "Lx" in bad["problems"][0] and "hypercharge" in bad["problems"][0]
    assert _photon_fixture(True)["verdict"] == "ok"
    assert fr_charge("QuantumNumbers -> {Y -> 1/6, Q -> 2/3}") == "2/3"
    assert fr_charge("QuantumNumbers -> {LeptonNumber -> 1}") is None


def fr_charge(qn: str):
    from tools.feynrules.feynarts_model import fr_classes
    return fr_classes(f"M$ClassesDescription = {{ F[1] == {{ ClassName -> X, {qn} }} }};")[0].charge


#: What WriteToyModel wrote, before the fix, for a charged singlet on SM.fr
#: with no hypercharge (the s_to_zgamma dogfood's SZAnoY.fr).
NO_HYPERCHARGE_FR = """M$ModelName = "SZAnoY";

M$Parameters = {
  yS == { ParameterType -> External, Value -> 1.0, InteractionOrder -> {NP, 1} }
};

M$ClassesDescription = {
  S[4] == { ClassName -> Sz, SelfConjugate -> True, Mass -> {MSz, 750.0}, Width -> 0, ParticleName -> "Sz", PDG -> 9000101 },
  F[5] == { ClassName -> Lx, SelfConjugate -> False, Mass -> {MLx, 400.0}, Width -> 0, QuantumNumbers -> {Q -> -1}, ParticleName -> "Lx-", AntiParticleName -> "Lx+", PDG -> 9000102 }
};

LKin := (1/2 del[Sz, mu] del[Sz, mu] - 1/2 MSz^2 Sz^2) +
        (I Lxbar.Ga[mu].DC[Lx, mu] - MLx Lxbar.Lx);
LInt := (-yS Sz Lxbar.Lx);
LToy := LKin + LInt;
"""


def test_charged_singlet_without_hypercharge_fails_the_photon_lint(tmp_path):
    fr = tmp_path / "SZAnoY.fr"
    fr.write_text(NO_HYPERCHARGE_FR)
    r = _write(tmp_path, model_path=str(fr), lagrangian="LSM + LToy", load_sm=True, name="SZAnoY")
    assert r["status"] == "lint_failed", r.get("lint")
    assert r["lint"]["photon"]["verdict"] == "failed" and "Lx" in r["lint"]["photon"]["problems"][0]
