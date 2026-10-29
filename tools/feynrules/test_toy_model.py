"""
# test_toy_model.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

WriteToyModel's generator, without Mathematica. The physics of what it writes
is tested in tools/oneloop/tests/test_oneloop.py (test_write_toy_model_*).
"""

import json

import pytest

from tools.feynrules.feynarts_model import fr_syntax_problems
from tools.feynrules.toy_model import ToyModelError, WriteToyModel, toy_model


def _spec(**kw):
    s = {"name": "T", "gauge": "qed",
         "fields": [{"name": "Sz", "spin": 0, "self_conjugate": True},
                    {"name": "Fq", "spin": "1/2", "charge": "2/3", "colour": 3},
                    {"name": "ph", "spin": 0, "charge": 1}],
         "parameters": {"y": 0.8, "muS": 300},
         "interactions": ["-y Sz Fqbar.Fq", "-muS Sz ph^dag ph"]}
    s.update(kw)
    return s


def test_generated_text():
    out = toy_model(_spec())
    fr = out["fr"]
    assert fr_syntax_problems(fr, False) == []
    assert "V[1] == { ClassName -> A" in fr and "U1QED" in fr
    assert "S[1] == { ClassName -> Sz" in fr and "F[1] == { ClassName -> Fq" in fr
    assert "Indices -> {Index[Colour]}" in fr and "IndexRange[Index[Colour]]" in fr
    assert "I Fqbar.Ga[mu].DC[Fq, mu] - MFq Fqbar.Fq" in fr
    assert "-muS Sz phbar ph" in fr
    assert out["lagrangian"] == "LToy" and not out["load_sm"]
    assert out["particle_names"]["ph"] == ["ph+", "ph-"]
    # no line starts with "+": Mathematica's Get would end the statement before it
    assert not any(ln.lstrip().startswith("+") for ln in fr.splitlines())


def test_conveniences():
    s = _spec(fields=[{"name": "X", "spin": 0, "self_conjugate": True},
                      {"name": "Lv", "spin": "1/2", "charge": -1},
                      {"name": "lf", "spin": "1/2", "charge": -1},
                      {"name": "Sq", "spin": 0, "charge": "2/3", "colour": 3}],
              parameters={"cL": 1, "cR": 2, "k": 3, "yP": 1},
              interactions=["X Lvbar.(cL PL + cR PR).lf + h.c.", "-k X Sq^dag Sq", "-I yP X Lvbar.g5.Lv"])
    fr = toy_model(s)["fr"]
    assert "(X Lvbar.(cL ProjM + cR ProjP).lf) + HC[X Lvbar.(cL ProjM + cR ProjP).lf]" in fr
    assert "-k X Sqbar[cc] Sq[cc]" in fr
    assert "Lvbar.Ga[5].Lv" in fr
    assert "DC[Sqbar[cc], mu] DC[Sq[cc], mu]" in fr


@pytest.mark.parametrize("change, match", [
    ({"fields": [{"name": "S", "spin": 0, "self_conjugate": True}]}, "taken"),
    ({"fields": [{"name": "E", "spin": "1/2"}]}, "taken"),
    ({"gauge": "none"}, "charged fields"),
    ({"interactions": ["-y Sz Fqbar.Fq Fqbar.Fq Sq Sq"],
      "fields": [{"name": "Sz", "spin": 0, "self_conjugate": True},
                 {"name": "Fq", "spin": "1/2", "colour": 3, "charge": 1},
                 {"name": "Sq", "spin": 0, "colour": 3, "charge": 1}]}, "colour indices out"),
    ({"fields": [{"name": "Ph", "doublet": ["a", "b"], "hypercharge": "1/2"}]}, "needs gauge 'sm'"),
])
def test_refused_with_reasons(change, match):
    with pytest.raises(ToyModelError, match=match):
        toy_model(_spec(**change))


def test_sm_addon_numbers_classes_past_sm_fr():
    s = {"name": "I", "gauge": "sm",
         "fields": [{"name": "H0", "spin": 0, "self_conjugate": True},
                    {"name": "Hc", "spin": 0, "charge": 1},
                    {"name": "Phi2", "doublet": ["Hc", "H0/Sqrt[2]"], "hypercharge": "1/2"}]}
    out = toy_model(s)
    assert "S[4] == { ClassName -> H0" in out["fr"] and "S[5] == { ClassName -> Hc" in out["fr"]
    assert "S[6] == { ClassName -> Phi2, Unphysical -> True" in out["fr"]
    assert out["load_sm"] and out["lagrangian"] == "LSM + LToy"
    assert fr_syntax_problems(out["fr"], True) == []
    with pytest.raises(ToyModelError, match="taken"):
        toy_model({"name": "I", "gauge": "sm", "fields": [{"name": "Phi", "spin": 0}]})


def test_the_tool_writes_the_file(tmp_path):
    out = json.loads(WriteToyModel(base_directory=str(tmp_path), spec=json.dumps(_spec()))._run())
    assert out["status"] == "ok" and (tmp_path / "T.fr").is_file()
    assert out["next"]["WriteFeynArtsModel"]["lagrangian"] == "LToy"
    bad = WriteToyModel(base_directory=str(tmp_path), spec={"name": "T", "fields": []})._run()
    assert "Invalid Model Spec" in bad and "no fields" in bad


def test_toy_spec_goes_straight_to_the_compilers(tmp_path):
    """WriteFeynArtsModel and FeynRulesToUFO take toy_spec in place of a .fr:
    the file is written and model_path, lagrangian and load_sm are set."""
    from tools.feynrules.toy_model import apply_toy_spec

    class Holder:
        model_path = lagrangian = load_sm = None
    h = Holder()
    assert apply_toy_spec(h, _spec(), str(tmp_path), "T") is None
    assert h.model_path.endswith("T.fr") and h.lagrangian == "LToy" and h.load_sm is False
    assert "no fields" in apply_toy_spec(h, {"name": "T", "fields": []}, str(tmp_path), "T")


def _sm_singlets(**lx):
    return {"name": "SZA", "gauge": "sm",
            "fields": [{"name": "Sz", "spin": 0, "self_conjugate": True},
                       {"name": "Lx", "spin": "1/2", "charge": -1, **lx},
                       {"name": "ph", "spin": 0, "charge": 1}],
            "parameters": {"yS": 1.0}, "interactions": ["-yS Sz Lxbar.Lx"]}


def test_sm_singlet_hypercharge_defaults_to_the_charge():
    """s_to_zgamma dogfood: under gauge 'sm' a charged singlet without
    'hypercharge' was written with Q only, and FeynRules gave it no photon or
    Z vertex. A singlet has T3 = 0, so Y = Q: defaulted, and said so."""
    out = toy_model(_sm_singlets())
    assert "QuantumNumbers -> {Q -> -1, Y -> -1}" in out["fr"]
    assert "QuantumNumbers -> {Q -> 1, Y -> 1}" in out["fr"]
    assert len(out["notes"]) == 2 and "Y = Q" in out["notes"][0]
    assert toy_model(_sm_singlets(hypercharge=-1))["notes"] == [out["notes"][1]]
    res = json.loads(WriteToyModel(base_directory="/tmp", output_path="/dev/null", spec=_sm_singlets())._run())
    assert res["status"] == "ok" and len(res["notes"]) == 2


def test_sm_singlet_hypercharge_contradicting_the_charge_is_refused():
    with pytest.raises(ToyModelError, match="ambiguous"):
        toy_model(_sm_singlets(hypercharge="-1/2"))
    # a doublet's components take the doublet's hypercharge: not touched
    s = {"name": "I", "gauge": "sm",
         "fields": [{"name": "H0", "spin": 0, "self_conjugate": True},
                    {"name": "Hc", "spin": 0, "charge": 1},
                    {"name": "Phi2", "doublet": ["Hc", "H0/Sqrt[2]"], "hypercharge": "1/2"}]}
    out = toy_model(s)
    assert out["notes"] == [] and "QuantumNumbers -> {Q -> 1}," in out["fr"]
