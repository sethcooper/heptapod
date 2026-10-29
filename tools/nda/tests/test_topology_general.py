"""
Tree topologies are general in the number of final states.

  - mu- -> e- ve~ vm through an internal W (2 vertices, 1 propagator)
    must not fail with "tree_3body requires at most 1 vertices, got 2".
  - H -> Z(*) Z(*) -> 4l must not fail with "Unknown topology: tree_4body".
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from tools.nda.topology import resolve_topology, validate_topology  # noqa: E402
from tools.nda.nda_tool import EstimateDecayWidthNDATool  # noqa: E402


def _leg(label, spin="1/2", mass=0.0):
    return {"label": label, "spin": spin, "mass": mass}


MUON_VIA_W = {
    "topology": "tree_3body",
    "initial": [_leg("mu-", mass=0.10566)],
    "final": [_leg("e-", mass=0.000511), _leg("nu_e_bar"), _leg("nu_mu")],
    "vertices": [{"type": "va", "coupling": "g_W"}, {"type": "va", "coupling": "g_W"}],
    "propagators": [{"label": "W-", "mass": 80.419, "spin": 1}],
    "couplings": {"g_W": 0.4619},
    "color_factor": 1.0,
}

H4L_VIA_ZZ = {
    "topology": "tree_4body",
    "initial": [{"label": "H", "spin": 0, "mass": 125.0}],
    "final": [_leg("e-", mass=0.000511), _leg("e+", mass=0.000511),
              _leg("mu-", mass=0.10566), _leg("mu+", mass=0.10566)],
    "vertices": [{"type": "gauge-vector", "coupling": "g_HZZ"},
                 {"type": "va", "coupling": "g_Zee"},
                 {"type": "va", "coupling": "g_Zmm"}],
    "propagators": [{"label": "Z", "spin": 1, "mass": 91.188, "width": 2.4414},
                    {"label": "Z", "spin": 1, "mass": 91.188, "width": 2.4414}],
    "couplings": {"g_HZZ": 0.7, "g_Zee": 0.38, "g_Zmm": 0.38},
}


def test_tree_3body_accepts_internal_w():
    ok, msg = validate_topology(MUON_VIA_W)
    assert ok, msg


def test_tree_3body_still_accepts_contact_vertex():
    d = dict(MUON_VIA_W, vertices=[{"type": "4fermion", "coupling": "G_F"}], propagators=[])
    ok, msg = validate_topology(d)
    assert ok, msg


def test_tree_4body_is_known_and_accepts_zz_cascade():
    assert resolve_topology("tree_4body") is not None
    ok, msg = validate_topology(H4L_VIA_ZZ)
    assert ok, msg


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_tree_nbody_vertex_ceiling_is_n_minus_1(n):
    """1 -> n has n+1 external legs; a tree has at most n-1 internal vertices."""
    v = n - 1
    d = {
        "topology": f"tree_{n}body",
        "initial": [_leg("X", spin=0, mass=1000.0)],
        "final": [_leg(f"f{i}") for i in range(n)],
        "vertices": [{"type": "3pt"} for _ in range(v)],
        "propagators": [{"label": f"P{i}", "mass": 100.0, "spin": 0} for i in range(v - 1)],
    }
    ok, msg = validate_topology(d)
    assert ok, msg
    # one vertex too many (with a matching extra propagator) is not a tree
    d["vertices"].append({"type": "3pt"})
    d["propagators"].append({"label": "Px", "mass": 100.0, "spin": 0})
    ok, msg = validate_topology(d)
    assert not ok


def test_tree_nbody_mprop_name_fixes_propagator_count():
    d = dict(H4L_VIA_ZZ, topology="tree_4body_2prop")
    assert validate_topology(d)[0]
    d = dict(H4L_VIA_ZZ, topology="tree_4body_1prop")
    assert not validate_topology(d)[0]


def test_loop_order_still_enforced_for_generic_tree():
    d = dict(MUON_VIA_W, propagators=[])  # 2 vertices, 0 propagators: disconnected
    assert not validate_topology(d)[0]


def test_unknown_topology_message_lists_the_forms():
    ok, msg = validate_topology(dict(MUON_VIA_W, topology="banana"))
    assert not ok and "tree_<N>body" in msg


@pytest.mark.parametrize("diagram", [MUON_VIA_W, H4L_VIA_ZZ], ids=["muon", "h4l"])
def test_nda_tool_estimates_campaign_diagrams(tmp_path, diagram):
    tool = EstimateDecayWidthNDATool(base_directory=str(tmp_path))
    out = tool.execute(diagram=diagram, process_label="regression")
    assert "Diagram Validation Failed" not in str(out)
    assert '"status":"ok"' in str(out).replace(" ", "")
