"""Census NDA: massive phase space, propagator kinematics from connectivity,
the narrow-width split, colour factors, and the census translation."""

import json
import math

import pytest

from tools.nda.census import combine, diagram_color_factor, estimate_catalogue
from tools.nda.nda_tool import EstimateDecayWidthNDATool
from tools.nda.phase_space import massive_phase_space

G = 0.6532 / math.sqrt(2)  # g / sqrt(2), the W coupling to a left-handed doublet


def run(diagram, tmp_path):
    out = EstimateDecayWidthNDATool(diagram=diagram, include_summary=False, record_finding=False,
                                    base_directory=str(tmp_path))._run()
    return json.loads(out)


def top_to_b_e_nu(legs=True):
    d = {
        "initial": [{"label": "t", "spin": "1/2", "mass": 173.0}],
        "final": [{"label": "b", "spin": "1/2", "mass": 4.7}, {"label": "e+", "spin": "1/2", "mass": 0.0},
                  {"label": "nu_e", "spin": "1/2", "mass": 0.0}],
        "vertices": [{"type": "left-handed", "coupling": G, "legs": ["in:0", "out:0", "prop:0"]},
                     {"type": "left-handed", "coupling": G, "legs": ["prop:0", "out:1", "out:2"]}],
        "propagators": [{"label": "W+", "mass": 80.4, "width": 2.05, "spin": 1}],
    }
    if not legs:
        for v in d["vertices"]:
            v.pop("legs")
    return d


def test_massive_phase_space_reduces_to_the_massless_formula():
    M = 100.0
    assert massive_phase_space(M, [0, 0, 0]) == pytest.approx(M**2 / (256 * math.pi**3), rel=1e-9)
    assert massive_phase_space(M, [0, 0, 0, 0]) == pytest.approx(M**4 / (2**11 * math.pi**5 * 12), rel=1e-9)
    beta = math.sqrt(1 - 4 * 0.3**2)
    assert massive_phase_space(1.0, [0.3, 0.3]) == pytest.approx(beta / (8 * math.pi))


def test_narrow_width_factorizes_an_on_shell_line(tmp_path):
    r = run(top_to_b_e_nu(), tmp_path)
    nw = r["narrow_width"]["resonances"][0]
    assert r["method"] == "narrow width" and nw["resonance"]["final_legs"] == [1, 2]
    assert r["width_gev"] == pytest.approx(nw["production"]["width_gev"] * nw["decay"]["width_gev"] / 2.05)
    assert 0.08 < r["width_gev"] < 0.3  # tree: Gamma(t -> b W) BR(W -> e nu) = 0.16 GeV
    # Without connectivity the diagram is estimated as before.
    assert run(top_to_b_e_nu(legs=False), tmp_path)["method"] != "narrow width"


def test_a_valence_suffix_does_not_hide_a_chiral_vertex(tmp_path):
    def w_to_e_nu(vtype):
        return run({"initial": [{"label": "W+", "spin": 1, "mass": 80.4}],
                    "final": [{"label": "e+", "spin": "1/2", "mass": 0.0}, {"label": "nu_e", "spin": "1/2", "mass": 0.0}],
                    "vertices": [{"type": vtype, "coupling": G}]}, tmp_path)["width_gev"]
    assert w_to_e_nu("left-handed-3pt") == pytest.approx(w_to_e_nu("left-handed"))
    assert w_to_e_nu("left-handed") == pytest.approx(G**2 * 80.4 / (24 * math.pi), rel=0.05)


def test_zero_width_line_that_can_go_on_shell_is_reported(tmp_path):
    d = top_to_b_e_nu()
    d["propagators"][0]["width"] = 0.0
    out = EstimateDecayWidthNDATool(diagram=d, include_summary=False, record_finding=False,
                                    base_directory=str(tmp_path))._run()
    assert out.startswith("Error: On-Shell Intermediate Without Width")


def test_colour_factors():
    # t -> b W: Identity(1,2) on the quark line; t -> b u d~: two lines; g -> q q~: T.
    assert diagram_color_factor(["Identity(1,2)"], [[1, 0, 2]], [[-3, 3, 1]], 3, 0) == pytest.approx(1)
    assert diagram_color_factor(["Identity(1,2)", "Identity(1,2)"], [[1, 0, 3], [4, 2, 3]],
                                [[-3, 3, 1], [-3, 3, 1]], 5, 0) == pytest.approx(3)
    assert diagram_color_factor(["T(3,2,1)"], [[1, 2, 0]], [[-3, 3, 8]], 3, 0) == pytest.approx(0.5)


def test_flavour_partners_add_coherently():
    a = {"response": {"width_gev": 1.0}, "skeleton": "s", "phase": 1 + 0j}
    b = {"response": {"width_gev": 1.0}, "skeleton": "s", "phase": -1 + 0j}
    c = {"response": {"width_gev": 1.0}, "skeleton": "other", "phase": 1 + 0j}
    assert combine([a, b]) == pytest.approx(0)  # GIM-like cancellation
    assert combine([a, c]) == pytest.approx(2)  # different topologies: incoherent


def test_census_translation_of_a_scalar_trilinear_is_exact(tmp_path):
    from tests.process_catalogue.test_catalogue import toy_model
    from tools.feyngraph.catalogue import ProcessModel, enumerate_processes
    pytest.importorskip("feyngraph")
    model_dir = toy_model(tmp_path / "toy")
    census = tmp_path / "census"
    census.mkdir()
    enumerate_processes(ProcessModel(str(model_dir)), census, parents=["s"], final_multiplicity=2, render=False)
    result = estimate_catalogue(census, model_dir, None, tmp_path / "nda", tmp_path)
    assert result["counts"] == {"estimated": 1}
    row = json.loads((tmp_path / "nda" / "nda.json").read_text())["processes"][0]
    # s -> x x through g s x x with g = 1 GeV: Gamma = beta g^2 / (16 pi M) / 2!.
    beta = math.sqrt(1 - 4 * (1.0 / 10.0) ** 2)
    assert row["width_gev"] == pytest.approx(beta / (16 * math.pi * 10.0) / 2, rel=1e-6)
    estimate = json.loads((tmp_path / "nda" / row["estimate"]).read_text())
    request = estimate["diagrams"][0]["request"]
    assert request["vertices"][0]["coupling_dimension"] == 1 and request["check_quantum_numbers"] is False
    # The request reruns on its own.
    assert run(request, tmp_path)["width_gev"] == pytest.approx(estimate["diagrams"][0]["response"]["width_gev"])


def test_below_threshold_line_is_averaged_over_its_virtuality(tmp_path):
    # t -> b W* with m_t just under m_W + m_b: the W can almost go on shell, and
    # the Breit-Wigner average over q^2 keeps the enhancement at the upper end.
    from tools.nda.kinematics import propagator_factor
    m, w = 80.4, 2.05
    lo, hi = 0.0, (84.0 - 4.7) ** 2
    mid = 1 / (((lo + hi) / 2 - m**2) ** 2 + (m * w) ** 2)
    avg = propagator_factor(m, w, lo, hi)
    assert avg == pytest.approx((math.atan((hi - m**2) / (m * w)) - math.atan(-m**2 / (m * w))) / (m * w * hi))
    assert avg > 5 * mid
    # A line without a width keeps the midpoint, so a massless line stays finite at q^2 = 0.
    assert propagator_factor(0.0, 0.0, 0.0, 100.0) == pytest.approx(1 / 50.0**2)


def test_census_channels_carry_the_formula_tool_expression(tmp_path):
    from tests.process_catalogue.test_catalogue import toy_model
    from tools.feyngraph.catalogue import ProcessModel, enumerate_processes
    pytest.importorskip("feyngraph")
    model_dir = toy_model(tmp_path / "toy")
    census = tmp_path / "census"
    census.mkdir()
    enumerate_processes(ProcessModel(str(model_dir)), census, parents=["s"], final_multiplicity=2, render=False)
    estimate_catalogue(census, model_dir, None, tmp_path / "nda", tmp_path)
    row = json.loads((tmp_path / "nda" / "nda.json").read_text())["processes"][0]
    estimate = json.loads((tmp_path / "nda" / row["estimate"]).read_text())
    d = estimate["diagrams"][0]
    # EstimateDecayWidthFormulaNDA on a labelled copy of the diagram: named couplings, no numbers.
    assert d["formula_response"]["status"] == "ok" and d["formula_scaling"] == d["formula_response"]["formula"]
    # The channel's formula is the evaluated expression of its largest diagram, with those names.
    assert d["formula"] == row["formula"] == estimate["formula"] and "g_{" in row["formula"]
    assert row["formula_fraction"] == pytest.approx(1.0)
    assert all(isinstance(v["coupling"], str) and v["coupling"].startswith("g_{") for v in d["formula_request"]["vertices"])
    assert "\\Gamma" in row["formula"]
    # Each coupling class carries the formula of its largest diagram.
    assert all(c["formula"] and "\\Gamma" in c["formula"] for c in row["classes"])


def test_pole_at_the_lower_end_is_not_a_resonance():
    # tau* -> tau nu nubar: the interval starts at q^2 = m_tau^2, where the
    # line's own phase space closes, so the midpoint is used, not 1/(m Gamma).
    from tools.nda.kinematics import propagator_factor
    m, w = 1.777, 2.27e-12
    lo, hi = m**2, 78.0**2
    assert propagator_factor(m, w, lo, hi) == pytest.approx(1 / (((lo + hi) / 2 - m**2) ** 2 + (m * w) ** 2))


def test_internal_massless_vector_requires_a_regulator_only_with_massless_daughters():
    from tools.nda.census import internal_massless_vector

    class Model:
        by_pdg = {22: {"spin": 3, "mass": 0.0}, 23: {"spin": 3, "mass": 91.19}, 24: {"spin": 3, "mass": 80.4},
                  -11: {"spin": 2, "mass": 0.000511}, 12: {"spin": 2, "mass": 0.0},
                  15: {"spin": 2, "mass": 1.777}, -15: {"spin": 2, "mass": 1.777},
                  2: {"spin": 2, "mass": 0.0}, -2: {"spin": 2, "mass": 0.0}}

    def process(boson, pair):
        # W+ -> e+ nu_e*, nu_e* -> nu_e boson*, boson* -> pair
        edges = [{"from": "in:0", "to": "v0", "role": "external", "pdg": 24},
                 {"from": "v0", "to": "out:0", "role": "external", "pdg": -11},
                 {"from": "v0", "to": "v1", "role": "internal", "pdg": 12},
                 {"from": "v1", "to": "out:1", "role": "external", "pdg": 12},
                 {"from": "v1", "to": "v2", "role": "internal", "pdg": boson},
                 {"from": "v2", "to": "out:2", "role": "external", "pdg": pair[0]},
                 {"from": "v2", "to": "out:3", "role": "external", "pdg": pair[1]}]
        return {"diagrams": [{"graph": {"edges": edges}}]}

    assert internal_massless_vector(Model, process(22, (2, -2)))
    assert not internal_massless_vector(Model, process(22, (15, -15)))
    assert not internal_massless_vector(Model, process(23, (2, -2)))


def test_coupling_names_stay_valid_tex():
    from tools.nda.census import join_tex
    assert join_tex([r"\gamma", "d", "d"]) == r"\gamma dd"
    assert join_tex(["Z", r"\nu_e", r"\nu_e"]) == r"Z\nu_e\nu_e"
    assert join_tex(["h", r"\tau", r"\tau"]) == r"h\tau\tau"


def test_evaluated_formula_multiplies_out_to_the_width(tmp_path):
    # t -> b e nu below the W threshold: every factor of the improved estimate.
    d = top_to_b_e_nu()
    d["initial"][0]["mass"] = 80.0
    r = run(d, tmp_path)
    assert r["method"].startswith("improved") and r["formula_type"] == "evaluated"
    assert r["factors"]["evaluated_width_gev"] == pytest.approx(r["width_gev"], rel=1e-12)
    assert r["formula"].startswith("\\Gamma=") and "\\langle|D_{W}|^2\\rangle" in r["formula"]
    assert any(x.startswith("\\bar E=") for x in r["definitions"])
    # Two-body, massless: closed-form phase space and energies.
    w = run({"initial": [{"label": "W+", "spin": 1, "mass": 80.4}],
             "final": [{"label": "e+", "spin": "1/2", "mass": 0.0}, {"label": "nu_e", "spin": "1/2", "mass": 0.0}],
             "vertices": [{"type": "left-handed", "coupling": G}]}, tmp_path)
    assert w["formula"] == "\\Gamma=\\frac{1}{3m_{W}}\\,\\frac{1}{8\\pi}\\,g_{1}^{2}\\,\\left(\\frac{2m_{W}}{2}\\right)^{2}"


def w_to_four_leptons_through_zww():
    # W+ -> W+* Z*, W+* -> e+ nu_e, Z* -> nu_mu nu_mu~: the gauge self-coupling.
    return {
        "initial": [{"label": "W+", "spin": 1, "mass": 80.4}],
        "final": [{"label": "e+", "spin": "1/2", "mass": 0.0}, {"label": "ve", "spin": "1/2", "mass": 0.0},
                  {"label": "vm", "spin": "1/2", "mass": 0.0}, {"label": "vm~", "spin": "1/2", "mass": 0.0}],
        "vertices": [{"type": "gauge-vector-3pt", "coupling": 0.575, "legs": ["in:0", "prop:0", "prop:1"]},
                     {"type": "left-handed", "coupling": G, "legs": ["prop:0", "out:0", "out:1"]},
                     {"type": "left-handed", "coupling": 0.37, "legs": ["prop:1", "out:2", "out:3"]}],
        "propagators": [{"label": "W+", "mass": 80.4, "width": 2.05, "spin": 1, "final_legs": [0, 1]},
                        {"label": "Z", "mass": 91.19, "width": 2.50, "spin": 1, "final_legs": [2, 3]}],
        "check_quantum_numbers": False,
    }


def test_a_momentum_coupling_carries_the_energy_of_its_legs(tmp_path):
    # The three-vector vertex multiplies a momentum: |M|^2 gets E_v^2 with E_v
    # the largest leg energy, here the parent's mass; the width then has the
    # dimension of a mass.
    r = run(w_to_four_leptons_through_zww(), tmp_path)
    f = r["factors"]
    assert f["vertex_energy_factor"] == pytest.approx(80.4 ** 2)
    assert [v["legs"] for v in f["derivative_vertices"]] == [["in:0", "prop:0", "prop:1"]]
    assert f["evaluated_width_gev"] == pytest.approx(r["width_gev"], rel=1e-12)
    assert "E_{v}^2" in r["formula"] and any(x.startswith("E_{v}=") for x in r["definitions"])
    # A fermion vertex carries none.
    top = top_to_b_e_nu()
    top["initial"][0]["mass"] = 80.0
    assert run(top, tmp_path)["factors"]["vertex_energy_factor"] == 1.0


def heavy_lepton_to_nu_w_w(vertex):
    # N -> nu Z*, Z* -> W+ W-, or the same final state with the two W's from the
    # vertex given ("gauge": the three-vector vertex; "lepton": N and a light lepton).
    if vertex == "gauge":
        vertices = [{"type": "left-handed", "coupling": 0.37, "legs": ["in:0", "out:0", "prop:0"]},
                    {"type": "gauge-vector-3pt", "coupling": 0.575, "legs": ["prop:0", "out:1", "out:2"]}]
        propagators = [{"label": "Z", "mass": 91.19, "width": 2.50, "spin": 1, "final_legs": [1, 2]}]
    else:
        vertices = [{"type": "left-handed", "coupling": G, "legs": ["in:0", "out:1", "prop:0"]},
                    {"type": "left-handed", "coupling": G, "legs": ["prop:0", "out:0", "out:2"]}]
        propagators = [{"label": "e-", "mass": 0.0, "width": 0.0, "spin": 0.5, "final_legs": [0, 2]}]
    return {
        "initial": [{"label": "N1", "spin": "1/2", "mass": 1000.0}],
        "final": [{"label": "ve", "spin": "1/2", "mass": 0.0}, {"label": "W+", "spin": 1, "mass": 80.4},
                  {"label": "W-", "spin": 1, "mass": 80.4}],
        "vertices": vertices,
        "propagators": propagators,
        "check_quantum_numbers": False,
    }


def test_longitudinal_growth_needs_a_heavy_leg_at_the_emitting_vertex(tmp_path):
    # Both W's from the three-vector vertex: neither grows, P = 2 each.
    f = run(heavy_lepton_to_nu_w_w("gauge"), tmp_path)["factors"]
    assert f["unsourced_vectors"] == [1, 2]
    # Through the lepton line the W at the N vertex grows; the one at the
    # light-lepton vertex does not.
    r = run(heavy_lepton_to_nu_w_w("lepton"), tmp_path)
    f = r["factors"]
    assert f["unsourced_vectors"] == [2]
    E = f["final_energies_gev"][1]
    assert f["external_polarization"] == pytest.approx(2 * 1000.0 * 2 * f["final_energies_gev"][0]
                                                       * (1 + E ** 2 / 80.4 ** 2) * 2)
    assert f["evaluated_width_gev"] == pytest.approx(r["width_gev"], rel=1e-12)
    # t -> b W: the top is the source, P = 1 + E^2/m_W^2.
    top = {"initial": [{"label": "t", "spin": "1/2", "mass": 172.7}],
           "final": [{"label": "b", "spin": "1/2", "mass": 4.18}, {"label": "W+", "spin": 1, "mass": 80.4}],
           "vertices": [{"type": "left-handed", "coupling": G, "legs": ["in:0", "out:0", "out:1"]}],
           "check_quantum_numbers": False}
    assert run(top, tmp_path)["factors"]["unsourced_vectors"] == []


def test_compact_formula_is_the_width_in_its_parameters(tmp_path):
    from tools.nda.evaluated import compact_formula
    r = run(w_to_four_leptons_through_zww(), tmp_path)
    f = r["factors"]
    c = compact_formula(f, parent="W+", final=["e+", "ve", "vm", "vm~"], lines=["W+", "Z"],
                        couplings=["g_{ZWW}", "g_{We\\nu}", "g_{Z\\nu\\nu}"])
    # W on its pole inside the interval, Z heavier than its virtuality, one
    # vertex energy E_v^2 ~ m_W^2: m_W^(1+8) / (m_W Gamma_W m_W^2 m_Z^4).
    assert c["formula"] == "\\Gamma\\approx\\kappa\\,g_{ZWW}^{2}g_{We\\nu}^{2}g_{Z\\nu\\nu}^{2}\\,\\frac{m_{W}^{6}}{m_{Z}^{4}\\Gamma_{W}}"
    symbolic = 80.4 ** 6 / (91.19 ** 4 * 2.05)
    assert c["kappa"] * f["coupling_sq"] * symbolic == pytest.approx(r["width_gev"], rel=1e-12)
    # Spin average 1/3, 0.8^2, the massless volume over m_W^4, (2E/m_W)^4 = (1/2)^4.
    assert c["kappa_0"] == pytest.approx(1 / 3 * 0.8 ** 2 / (2 ** 11 * 6 * 2 * math.pi ** 5) / 16)
    assert c["definitions"][0].startswith(
        "\\kappa=\\frac{1}{3}\\,\\frac{0.8^{2}}{2^{11}\\,3!\\,2!\\,\\pi^{5}}\\,\\left(\\frac{1}{2}\\right)^{4}\\,r=")
    # Massless two-body: kappa is exact and needs no r.
    w = run({"initial": [{"label": "W+", "spin": 1, "mass": 80.4}],
             "final": [{"label": "e+", "spin": "1/2", "mass": 0.0}, {"label": "ve", "spin": "1/2", "mass": 0.0}],
             "vertices": [{"type": "left-handed", "coupling": G}], "check_quantum_numbers": False}, tmp_path)
    two = compact_formula(w["factors"], parent="W+", final=["e+", "ve"], lines=[], couplings=["g_{We\\nu}"])
    assert two["definitions"] == ["\\kappa=\\frac{1}{3}\\,\\frac{1}{8\\,\\pi}=0.0133"]
    assert two["formula"] == "\\Gamma\\approx\\kappa\\,g_{We\\nu}^{2}\\,m_{W}"


def test_numbers_in_definitions():
    from tools.nda.evaluated import number
    assert [number(x) for x in (0, 3, 1.5, 9.8e-14, 0.4616, 29230.0)] == [
        "0", "3", "\\tfrac{3}{2}", "9.800\\times10^{-14}", "0.4616", "29230"]


class StubModel:
    """Photon (22), gluon (21), b (5, Q = -1/3), mu (13), nu (14), h (25)."""
    def __init__(self):
        p = lambda name, spin, color, charge, mass: {"name": name, "spin": spin, "color": color,
                                                     "charge": charge, "mass": mass, "width": 0.0}
        self.by_pdg = {22: p("a", 3, 1, 0, 0), 21: p("g", 3, 8, 0, 0), 5: p("b", 2, 3, -1 / 3, 4.18),
                       -5: p("b~", 2, -3, 1 / 3, 4.18), 13: p("mu-", 2, 1, -1, 0.1), -13: p("mu+", 2, 1, 1, 0.1),
                       14: p("vm", 2, 1, 0, 0), -14: p("vm~", 2, 1, 0, 0), 25: p("H", 1, 1, 0, 125.0)}
        self.values = {"aEWM1": 132.0, "aS": 0.118}

    def anti(self, pdg):
        return -pdg if -pdg in self.by_pdg else pdg


def test_soft_collinear_emission_off_the_final_fermions():
    from tools.nda.census import channel_key, soft_collinear
    m, y = StubModel(), 0.05 ** 2
    L2 = math.log(1 / y) ** 2
    lower = {channel_key(m, 25, [-5, 5]): {"id": "h-bb", "width_gev": 2e-3}}
    one = soft_collinear(m, {"initial": ["H"], "initial_pdgs": [25], "final_pdgs": [-5, 5, 22]}, lower, y)
    assert one["width_gev"] == pytest.approx(2e-3 * 2 * (1 / 9) / 132.0 / (2 * math.pi) * L2)
    assert "Q_{b}^2" in one["formula"] and one["base"] == "h-bb"
    two = soft_collinear(m, {"initial": ["H"], "initial_pdgs": [25], "final_pdgs": [-5, 5, 21, 21]}, lower, y)
    c = 2 * 0.118 * 4 / 3 / (2 * math.pi)
    assert two["width_gev"] == pytest.approx(2e-3 * (c * L2) ** 2 / 2)
    # No charged fermion: no collinear logarithm, no estimate.
    lower[channel_key(m, 25, [14, -14])] = {"id": "h-vv", "width_gev": 1e-9}
    none = soft_collinear(m, {"initial": ["H"], "initial_pdgs": [25], "final_pdgs": [14, -14, 22]}, lower, y)
    assert none["width_gev"] is None
    # Without the base channel there is nothing to multiply.
    assert soft_collinear(m, {"initial": ["H"], "initial_pdgs": [25], "final_pdgs": [13, -13, 22]}, lower, y) is None


def test_photon_or_gluon_split_into_a_massive_pair():
    # H -> b bbar mu+ mu-: gamma* -> mu+ mu- off the b quarks, and g* -> b bbar
    # cannot arise (no quark in X = mu+ mu-), gamma* -> b bbar off the muons can.
    from tools.nda.census import channel_key, splitting
    m = StubModel()
    lower = {channel_key(m, 25, [-5, 5]): {"id": "h-bb", "width_gev": 2e-3},
             channel_key(m, 25, [13, -13]): {"id": "h-mumu", "width_gev": 8e-7}}
    process = {"initial": ["H"], "initial_pdgs": [25], "final_pdgs": [-5, 5, 13, -13]}
    to_mumu = [{"split": [{"vector": 22, "legs": [2, 3]}]}]
    r = splitting(m, process, to_mumu, lower)
    alpha, L = 1 / 132.0, math.log(125.0 ** 2 / (4 * 0.1 ** 2))
    c = 2 * (1 / 9) * alpha / (2 * math.pi)
    assert r["width_gev"] == pytest.approx(2e-3 * c * alpha / (3 * math.pi) * L ** 3 / 3)
    assert r["terms"][0]["base"] == "h-bb" and r"\ln^3" in r["formula"]
    # The same split line in two diagrams counts once; a gluon into b bbar off
    # the muons has no emitter and adds nothing.
    both = splitting(m, process, to_mumu * 2 + [{"split": [{"vector": 21, "legs": [0, 1]}]}], lower)
    assert both["width_gev"] == pytest.approx(r["width_gev"])
    to_bb = splitting(m, process, [{"split": [{"vector": 22, "legs": [0, 1]}]}], lower)
    Lb = math.log(125.0 ** 2 / (4 * 4.18 ** 2))
    assert to_bb["width_gev"] == pytest.approx(
        8e-7 * 2 * alpha / (2 * math.pi) * 3 * alpha / 9 / (3 * math.pi) * Lb ** 3 / 3)
    # Without the lower channel there is nothing to multiply.
    assert splitting(m, process, to_mumu, {}) is None


def test_quark_decays_of_a_particle_below_the_lightest_hadron_are_closed():
    from tools.nda.census import closed
    m = StubModel()
    m.by_pdg[11] = {"name": "e-", "spin": 2, "color": 1, "charge": -1, "mass": 0.000511, "width": 0.0}
    assert closed(m, 13, [14, -5, 5]) and closed(m, 11, [5]) and not closed(m, 13, [14, -14, 11])
    assert not closed(m, 25, [5, -5])


def test_a_model_particle_number_is_a_subscript():
    from tools.nda.evaluated import tex_name, bare_name
    assert tex_name("N1") == "N_{1}" and tex_name("N1~") == "\\bar{N_{1}}" and bare_name("N1") == "N_{1}"
    assert tex_name("mu-") == "\\mu^-" and bare_name("W+") == "W"
