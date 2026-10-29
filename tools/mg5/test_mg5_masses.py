"""Unit tests for the declarative `masses` channel in the MadGraph tool (loop 16).

`_edit_mg5_card` injects `set mass <pdg> <gev>` into the run-card region (after
`launch`), replacing any existing set-mass for the same id, so the benchmark
spectrum survives card regeneration instead of being dropped as free text.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.mg5.mg5 import _edit_mg5_card, _normalize_masses


def test_normalize_masses_accepts_dict_and_pairs():
    assert _normalize_masses({"6": 173}) == {6: 173.0}
    assert _normalize_masses([[6, 173.0]]) == {6: 173.0}
    assert _normalize_masses({"1000021": "1600"}) == {1000021: 1600.0}
    assert _normalize_masses("garbage") == {}
    assert _normalize_masses({"x": "y"}) == {}  # junk entries dropped


def test_injects_masses_after_launch():
    card = ("import model MSSM_SLHA2\n"
            "generate p p > x1+ n2, (x1+ > w+ n1), (n2 > z n1)\n"
            "output proc\nlaunch\nset nevents 100\n")
    out = _edit_mg5_card(card, masses={"1000024": 550, "1000023": 550, "1000022": 200},
                         nevents=50000)
    assert "set mass 1000024 550" in out
    assert "set mass 1000022 200" in out
    assert "set nevents 50000" in out
    # masses must land AFTER launch (run-card region), not before
    assert out.index("launch") < out.index("set mass 1000024")


def test_replaces_existing_set_mass_idempotent():
    card = ("import model MSSM_SLHA2\ngenerate p p > go go\noutput proc\n"
            "launch\nset mass 1000021 800\n")
    out = _edit_mg5_card(card, masses={"1000021": 1600})
    assert out.count("set mass 1000021") == 1     # no duplicate
    assert "set mass 1000021 1600" in out
    assert "800" not in out                       # old value replaced


def test_no_masses_is_noop_for_spectrum():
    card = "import model sm\ngenerate p p > t t~\noutput proc\nlaunch\n"
    out = _edit_mg5_card(card, nevents=1000)
    assert "set mass" not in out


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"[ok] {name}")
    print("all mg5 mass-injection tests passed")


def test_relaunch_card_keeps_run_settings_only():
    from tools.mg5.mg5 import _relaunch_card
    card = "import model sm\ngenerate h > e+ e- mu+ mu-\noutput proc\nlaunch proc\nset nevents 100\nset ptl 0.0\ndone\n"
    out = _relaunch_card(card, "/abs/proc", nevents=5000, seed=7)
    assert out.splitlines() == ["launch /abs/proc", "set nevents 5000", "set ptl 0.0", "done", "set iseed 7"]
